import torch
from torch.distributions import *
import matplotlib.pyplot as plt
from ptvi.stopping import *
from ptvi.dist import InvGamma
from ptvi import (VIModel, VITimeSeriesResult, local_param, global_param)


class LocalLevelModel(VIModel):
    # approximating density: q(u, LL')

    result_class = VITimeSeriesResult
    name = 'Local level model'
    z = local_param()
    γ = global_param(prior=Normal(0, 3))
    η = global_param(prior=Normal(0, 3), transform='log', rename='ψ')
    σ = global_param(prior=InvGamma(1, 5), transform='log', rename='ς')
    ρ = global_param(prior=Beta(1, 1), transform='logit', rename='φ')

    def ln_joint(self, y, ζ):
        """Computes the log likelihood plus the log prior at ζ."""
        z, γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)
        ar1_uncond_var = torch.pow((1 - torch.pow(ρ, 2)), -0.5)
        llikelihood = (
            Normal(γ + η * z, σ).log_prob(y).sum()
            + Normal(ρ * z[:-1], 1).log_prob(z[1:]).sum()
            + Normal(0., ar1_uncond_var).log_prob(z[0])
        )
        lprior = (
            self.γ_prior.log_prob(γ)
            + self.η_prior.log_prob(η)
            + self.ς_prior.log_prob(ς)
            + self.φ_prior.log_prob(φ)
        )
        return llikelihood + lprior

    def simulate(self, γ: float, η: float, σ: float, ρ: float):
        z = torch.empty([self.input_length])
        z[0] = Normal(0, 1/(1 - ρ**2)**0.5).sample()
        for i in range(1, self.input_length):
            z[i] = ρ*z[i-1] + Normal(0, 1).sample()
        y = Normal(γ + η*z, σ).sample()
        return y, z

    def sample_paths(self, N=100, fc_steps=0):
        """Sample N paths from the model, forecasting fc_steps additional steps.

        Args:
            N:        number of paths to sample
            fc_steps: number of steps to project forward

        Returns:
            Nx(τ+fc_steps) tensor of sample paths
        """
        paths = torch.empty((N, self.input_length + fc_steps))
        ζs = MultivariateNormal(self.u, scale_tril=self.L).sample((N,))
        _τ = self.input_length
        for i in range(N):
            ζ = ζs[i]
            z, γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)
            if fc_steps > 0:
                z = torch.cat([z, torch.zeros(fc_steps)])
            # project states forward
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = z[t-1]*ρ + Normal(0, 1).sample()
            paths[i, :] = Normal(γ + η*z, σ).sample()
        return paths
