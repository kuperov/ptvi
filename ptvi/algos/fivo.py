import math
from typing import Dict
import torch
from torch.distributions import LogNormal, Normal, Beta, Categorical

from ptvi import Model, global_param


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
        """Apply particle filter to estimate log ^p(y | ζ)"""
        log_phatN = 0.
        log_w = torch.tensor([math.log(1 / self.num_particles)] * self.num_particles)
        Z = torch.zeros((self.input_length, self.num_particles))
        resampled = [False] * self.input_length
        for t in range(self.input_length):
            Z[t, :] = self.proposal.conditional_sample(t, Z)
            log_αt = self.conditional_log_prob(
                t, y, Z, ζ
            ) - self.proposal.conditional_log_prob(t, Z)
            log_phatt = torch.logsumexp(log_w + log_αt, dim=0)
            log_phatN += log_phatt
            log_w += log_αt - log_phatt
            ESS = 1. / torch.exp(2 * log_w).sum()
            if ESS < self.num_particles and self.resample:
                resampled[t] = True
                a = Categorical(torch.exp(log_w)).sample((self.num_particles,))
                Znew = Z[:, a]
                Z = Znew
                log_w = torch.tensor(
                    [math.log(1 / self.num_particles)] * self.num_particles
                )
        return log_phatN, Z, resampled


class FilteredStochasticVolatilityModel(FilteredStateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)
    """

    name = "Particle filtered stochastic volatility model"
    a = global_param(prior=LogNormal(0, 1), transform="log", rename="α")
    b = global_param(prior=Normal(0, 1))
    c = global_param(prior=Beta(1, 1), transform="logit", rename="ψ")

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)
        z_true = torch.empty((self.input_length,))
        z_true[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z_true[t] = b + c * z_true[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(a) * torch.exp(z_true / 2)).sample()
        return x, z_true

    def conditional_log_prob(self, t, y, z, ζ):
        """Compute log p(x_t, z_t | y_{0:t-1}, z_{0:t-1}, ζ).

        Args:
            t: time index (zero-based)
            y: y_{0:t} vector of points observed up to this point (which may
               actually be longer, but should only be indexed up to t)
            z: z_{0:t} vector of unobserved variables to condition on (ditto,
               array may be longer)
            ζ: parameter to condition on; should be unpacked with self.unpack
        """
        (a, _), b, (c, _) = self.unpack(ζ)
        if t == 0:
            log_pzt = Normal(b, (1 - c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(b + c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def ln_prior(self, ζ):
        (_, α), b, (_, ψ) = self.unpack(ζ)
        return (
            self.α_prior.log_prob(α)
            + self.b_prior.log_prob(b)
            + self.ψ_prior.log_prob(ψ)
        )

    def __repr__(self):
        return (
            f"Stochastic volatility model:\n"
            f"\tx_t = exp(a * z_t/2) ε_t      t=1, …, {self.input_length}\n"
            f"\tz_t = b + c * z_{{t-1}} + ν_t,  t=2, …, {self.input_length}\n"
            f"\tz_1 ~ N(b, sqrt(1/(1 - c^2)))\n"
            f"\twhere ε_t, ν_t ~ Ν(0,1)"
        )


class AR1Proposal(PFProposal):
    """A simple linear/gaussian AR(1) to use as a particle filter proposal.

    .. math::
        z_t = μ + ρ * z_{t-1} + ε_t

    The error term ε_t can be of any location-scale family. For example:

        > from torch.distributions import *
        > prop1 = AR1Proposal(ρ=0.9)  # by default, Normal(0, 1) and μ=0.
        > prop2 = AR1Proposal(ρ=0.9, μ=0., ε_scale=1.5, ε_type=StudentT, df=3)
        > prop3 = AR1Proposal(ρ=0.9, μ=0., ε_scale=1.5, ε_type=Gumbel)
        > prop4 = AR1Proposal(ρ=0.9, μ=0., ε_scale=1.5, ε_type=Cauchy)
    """

    def __init__(
        self,
        ρ: float,
        μ: float = 0,
        ε_scale: float = 1.,
        ε_type: type = None,
        **ε_kwargs: Dict,
    ):
        assert -1 < ρ < 1
        self.ε_type = ε_type or Normal
        self.ε_kwargs = ε_kwargs or {}
        self.μ, self.ρ, self.ε_scale = μ, ρ, ε_scale

    def conditional_sample(self, t, Z):
        """Simulate z_t from q(z_t | z_{t-1}, φ)

        Z has an extra dimension, of N particles.
        """
        N = Z.shape[1]
        if t == 0:
            uncond_dist = self.ε_type(
                loc=0, scale=(1 - self.ρ ** 2) ** (-.5) * self.ε_scale, **self.ε_kwargs
            )
            return uncond_dist.sample((N,))
        else:
            cond_dist = self.ε_type(
                loc=self.μ + self.ρ * Z[t - 1], scale=self.ε_scale, **self.ε_kwargs
            )
            return cond_dist.sample()

    def conditional_log_prob(self, t, Z):
        """Compute log q(z_t | z_{t-1}, φ)"""
        if t == 0:
            uncond_dist = self.ε_type(
                loc=0, scale=(1 - self.ρ ** 2) ** (-.5) * self.ε_scale, **self.ε_kwargs
            )
            return uncond_dist.log_prob(Z[t])
        else:
            cond_dist = self.ε_type(
                loc=self.μ + self.ρ * Z[t - 1], scale=self.ε_scale, **self.ε_kwargs
            )
            return cond_dist.log_prob(Z[t])

    def __repr__(self):
        typename = self.ε_type.__name__
        params = {"loc": 0, "scale": self.ε_scale}
        params.update(self.ε_kwargs)
        param_str = ", ".join(f"{n}={v:.2f}" for n, v in params.items())
        return (
            "AR(1) proposals:\n"
            f"\tz_t = {self.μ:.2f} + {self.ρ:.2f} * z_{{t-1}} + ε_t\n"
            f"\tε_t ~ {typename}({param_str})"
        )
