import torch
from torch.distributions import *
from ptvi import (VIModel, VITimeSeriesResult, local_param, global_param)


class StochVolModel(VIModel):
    result_class = VITimeSeriesResult
    name = 'Stochastic volatility model'
    b = local_param()
    # λ = global_param(prior=Normal(0, 1e-4))
    σ = global_param(prior=LogNormal(0, 1), rename='α', transform='log')
    φ = global_param(prior=Beta(2, 2), rename='ψ', transform='logit')

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

    def print_status(self, i, elbo_hat):
        self.print(f'{i: 8d}. smoothed elbo_hat ={float(elbo_hat):12.2f}')
        # b, λ, (σ, log_σ), (φ, logit_φ) = self.unpack(self.u)
        b, (σ, log_σ), (φ, logit_φ) = self.unpack(self.u)
        self.print(f'          '  # λ={float(λ):.2f} '
                   f'σ={float(σ):.2f} log(σ)={float(log_σ):.2f} '
                   f'φ={float(φ):.2f} logit(φ)={float(logit_φ):.2f}')
