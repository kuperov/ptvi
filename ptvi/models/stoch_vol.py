import torch
from torch.distributions import *
from ptvi import Model, local_param, global_param, LogNormalPrior, BetaPrior


class StochVolModel(Model):
    name = "Stochastic volatility model"
    b = local_param()
    # λ = global_param(prior=Normal(0, 1e-4))
    σ = global_param(prior=LogNormalPrior(0, 1), rename="α", transform="log")
    φ = global_param(prior=BetaPrior(2, 2), rename="ψ", transform="logit")

    def ln_joint(self, y, ζ):
        # b, λ, (σ, α), (φ, ψ) = self.unpack(ζ)
        b, (σ, α), (φ, ψ) = self.unpack(ζ)
        ar1_sd = torch.pow(1 - torch.pow(φ, 2), -0.5)
        llikelihood = (
            Normal(0, torch.exp(.5 * (σ * b))).log_prob(y).sum()
            # Normal(0, torch.exp(.5 * (λ + σ * b))).log_prob(y).sum()
            + Normal(φ * b[:-1], 1).log_prob(b[1:]).sum()
            + Normal(0., ar1_sd).log_prob(b[0])
        )
        lprior = (
            self.ψ_prior.log_prob(ψ)
            + self.α_prior.log_prob(α)
            # + self.λ_prior.log_prob(λ)
        )
        return llikelihood + lprior

    def simulate(self, λ=0., σ=0.5, φ=0.95):
        assert σ > 0 and 0 < φ < 1
        b = torch.zeros(self.input_length)
        φ = torch.tensor(φ)
        ar1_sd = torch.pow(1 - torch.pow(φ, 2), -0.5)
        b[0] = Normal(0, ar1_sd).sample()
        for t in range(1, self.input_length):
            b[t] = Normal(φ * b[t - 1], 1).sample()
        y = Normal(loc=0., scale=torch.exp(0.5 * (λ + σ * b))).sample()
        return y, b

    def sample_observed(self, ζ, y, fc_steps=0):
        b, (σ, α), (φ, ψ) = self.unpack(ζ)
        λ = 0.
        if fc_steps > 0:
            b = torch.cat([b, torch.zeros(fc_steps)])
        for t in range(self.input_length, self.input_length + fc_steps):
            b[t] = b[t - 1] * φ + Normal(0, 1).sample()
        return Normal(loc=0., scale=torch.exp(0.5 * (λ + σ * b))).sample()
