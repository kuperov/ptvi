import math
import torch
from torch.distributions import LogNormal, Normal, Beta, Categorical

import numpy as np

from ptvi import Model, global_param
import ptvi


if torch.version.__version__ >= "0.4.1":

    def logsumexp(xs):
        return torch.logsumexp(xs, dim=0)


else:

    def logsumexp(xs):
        m = torch.max(xs)
        return m + torch.log(torch.sum(torch.exp(xs - m)))


class FIVOResult(ptvi.MVNPosterior):
    def plot_latent(
        self, N: int = 100, α: float = 0.05, true_z=None, fc_steps: int = 0
    ):
        import matplotlib.pyplot as plt

        paths = self.sample_latent_paths(N, fc_steps=fc_steps)
        ci_bands = np.empty([self.input_length + fc_steps, 2])
        fxs, xs = range(self.input_length + fc_steps), range(self.input_length)
        perc = 100 * np.array([α * 0.5, 1. - α * 0.5])
        for t in fxs:
            ci_bands[t, :] = np.percentile(paths[:, t], q=perc)
        plt.fill_between(
            fxs, ci_bands[:, 0], ci_bands[:, 1], alpha=0.5, label=f"{(1-α)*100:.0f}% CI"
        )
        if true_z is not None:
            plt.plot(xs, true_z.numpy(), color="black", linewidth=2, label="z")
            plt.legend()
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color="black")
            plt.title(
                f"Posterior credible interval and " f"{fc_steps}-step-ahead forecast"
            )
        else:
            plt.title(f"Posterior credible interval")


class PFProposal(object):
    def conditional_sample(self, t, Z, N):
        raise NotImplementedError

    def conditional_log_prob(self, t, Z):
        raise NotImplementedError


class FilteredStateSpaceModel(Model):

    result_type = FIVOResult

    def __init__(self, input_length: int, num_particles: int = 50, resample=True):
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

    def ln_joint(self, y, ζ):
        llik_hat = self.simulate_log_phatN(y, ζ)
        lprior = self.ln_prior(ζ)
        return llik_hat + lprior

    def simulate_log_phatN(
        self, y: torch.Tensor, ζ: torch.Tensor, sample: torch.Tensor = None
    ):
        """Apply particle filter to estimate marginal likelihood log p^(y | ζ)"""
        log_phatN = 0.
        log_N = math.log(self.num_particles)
        log_w = torch.full((self.num_particles,), -log_N)
        Z = None
        proposal = self.proposal_for(y, ζ)
        for t in range(self.input_length):
            zt = proposal.conditional_sample(t, Z, self.num_particles).unsqueeze(0)
            Z = torch.cat([Z, zt]) if Z is not None else zt
            log_αt = self.conditional_log_prob(
                t, y, Z, ζ
            ) - proposal.conditional_log_prob(t, Z)
            log_phatt = logsumexp(log_w + log_αt)
            log_phatN += log_phatt
            log_w += log_αt - log_phatt
            with torch.no_grad():
                ESS = 1. / torch.exp(2 * log_w).sum()
                if self.resample and ESS < self.num_particles:
                    a = Categorical(torch.exp(log_w)).sample((self.num_particles,))
                    Z = (Z[:, a]).clone()
                    log_w = torch.full((self.num_particles,), -log_N)
        if sample is not None:
            with torch.no_grad():
                # samples should be M * T, where M is the number of samples
                assert sample.shape[0] >= self.input_length
                sample[: self.input_length] = Z[
                    :, Categorical(torch.exp(log_w)).sample()
                ]
        return log_phatN

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        """Return the proposal distribution for the given parameters.

        Args:
            y: data vector
            ζ: parameter vector
        """
        raise NotImplementedError


class AR1Proposal(PFProposal):
    """A simple linear/gaussian AR(1) to use as a particle filter proposal.

    .. math::
        z_t = μ + ρ * z_{t-1} + η_t
    """

    def __init__(self, μ, ρ, σ):
        assert -1 < ρ < 1 and σ > 0, f"ρ={ρ} and σ={σ}"
        self.μ, self.ρ, self.σ = μ, ρ, σ

    def conditional_sample(self, t, Z, N):
        """Simulate z_t from q(z_t | z_{0:t-1}, y_{0:t}, φ)

        Z has an extra dimension, of N particles.
        """
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
