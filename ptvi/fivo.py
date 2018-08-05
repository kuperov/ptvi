import math
import torch
from torch.distributions import LogNormal, Normal, Beta, Categorical

from ptvi import Model, global_param


# log_phatt = torch.logsumexp(, dim=0)

if torch.version.__version__ >= '0.4.1':
    def logsumexp(xs):
        return torch.logsumexp(xs, dim=0)
else:
    def logsumexp(xs):
        m = torch.max(xs)
        return m + torch.log(torch.sum(torch.exp(xs - m)))


class PFProposal(object):
    def conditional_sample(self, t, Z):
        raise NotImplementedError

    def conditional_log_prob(self, t, Z):
        raise NotImplementedError


class FilteredStateSpaceModel(Model):
    def __init__(
        self,
        input_length: int,
        proposal: PFProposal,
        num_particles: int = 50,
        resample=True,
    ):
        self.proposal: PFProposal = proposal
        self.num_particles = num_particles
        self.resample = resample
        super().__init__(input_length)

    def simulate(self, *args, **kwargs):
        """Simulate from p(x_{1:T} | θ)."""
        raise NotImplementedError

    def conditional_log_prob(self, t, y, Z, ζ):
        """Compute log p(x_t, z_t | y_{0:t-1}, z_{0:t-1}, ζ).

        Args:
            t: time index (zero-based)
            y: y_{0:t} vector of points observed up to this point (which may
               actually be longer, but should only be indexed up to t)
            z: z_{0:t} matrix of unobserved variables to condition on, one row
               per trajectory
            ζ: parameter to condition on; should be unpacked with self.unpack

        Return:
            1x1 tensor containing estimated conditional log marginal density
        """
        raise NotImplementedError

    def ln_prior(self, ζ):
        raise NotImplementedError

    def ln_joint(self, y, ζ):
        llik_hat, _, _ = self.simulate_log_phatN(y, ζ)
        lprior = self.ln_prior(ζ)
        return llik_hat + lprior

    def simulate_log_phatN(
        self, y: torch.Tensor, ζ: torch.Tensor, rewrite_history=True
    ):
        """Apply particle filter to estimate marginal likelihood log ^p(y | ζ)"""
        log_phatN = 0.
        log_w = torch.tensor([math.log(1 / self.num_particles)] * self.num_particles)
        Z = torch.zeros((self.input_length, self.num_particles))
        resampled = [False] * self.input_length
        for t in range(self.input_length):
            Z[t, :] = self.proposal.conditional_sample(t, Z)
            log_αt = self.conditional_log_prob(
                t, y, Z, ζ
            ) - self.proposal.conditional_log_prob(t, Z)
            log_phatt = logsumexp(log_w + log_αt)
            log_phatN += log_phatt
            log_w += log_αt - log_phatt
            ESS = 1. / torch.exp(2 * log_w).sum()
            if ESS < self.num_particles and self.resample:
                resampled[t] = True
                a = Categorical(torch.exp(log_w)).sample((self.num_particles,))
                if rewrite_history:
                    Z = Z[:, a]
                else:
                    Z[t, :] = Z[t, a]
                log_w = torch.tensor(
                    [math.log(1 / self.num_particles)] * self.num_particles
                )
        return log_phatN, Z, resampled


class AR1Proposal(PFProposal):
    """A simple linear/gaussian AR(1) to use as a particle filter proposal.

    .. math::
        z_t = μ + ρ * z_{t-1} + η_t
    """

    def __init__(self, μ, ρ, σ):
        assert -1 < ρ < 1 and σ > 0
        self.μ, self.ρ, self.σ = μ, ρ, σ

    def conditional_sample(self, t, Z):
        """Simulate z_t from q(z_t | z_{0:t-1}, y_{0:t}, φ)

        Z has an extra dimension, of N particles.
        """
        N = Z.shape[1]
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2) ** (-.5)).sample((N,))
        else:
            return Normal(self.μ + self.ρ * Z[t - 1], 1).sample()

    def conditional_log_prob(self, t, Z):
        """Compute log q(z_t | z_{t-1}, φ)"""
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2) ** (-.5)).log_prob(Z[t])
        else:
            return Normal(self.μ + self.ρ * Z[t - 1], 1).log_prob(Z[t])

    def __repr__(self):
        return (
            "AR(1) proposals:\n"
            f"\tz_t = {self.μ:.2f} + {self.ρ:.2f} * z_{{t-1}} + η_t\n"
            f"\tη_t ~ Ν(0,{self.σ:.2f})"
        )
