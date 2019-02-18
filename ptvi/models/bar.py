# import math
# import torch
from torch.distributions import Normal, Bernoulli
from ptvi import Model, NormalPrior, global_param


class BAR2(Model):

    name = "BAR(2) model"
    has_observation_error = False
    μ = global_param(NormalPrior(0, 1))
    ρ1 = global_param(NormalPrior(0, 1), transform="logit", rename="φ1")
    ρ2 = global_param(NormalPrior(0, 1), transform="logit", rename="φ2")

    def ln_joint(self, y, ζ):
        μ, (ρ1, φ1), (ρ2, φ2), = self.unpack(ζ)
        llhood = Bernoulli(μ + y[1:-1] * ρ1 + y[:-2] * ρ2).log_prob(y[2:]).sum()
        lprior = (
            self.μ_prior.log_prob(μ)
            + self.φ1_prior.log_prob(φ1)
            + self.φ2_prior.log_prob(φ2)
        )
        return llhood + lprior

    # def forecast(self, ζ: torch.Tensor, y: torch.Tensor, fc_steps: int) -> torch.Tensor:
    #     assert fc_steps >= 1, "Must forecast at least 1 step."
    #     μ, (ρ1, φ1), (ρ2, φ2), = self.unpack(ζ)
    #     fc = torch.empty((fc_steps,), device=self.device, dtype=self.dtype)
    #     ε = torch.randn((1,), device=self.device, dtype=self.dtype)
    #     fc[0] = μ + ρ1 * y[-1] + ρ2 * y[-2] + σ * ε
    #     if fc_steps > 1:
    #         ε = torch.randn((1,), device=self.device, dtype=self.dtype)
    #         fc[1] = μ + ρ1 * fc[0] + ρ2 * y[-1] + σ * ε
    #     for i in range(2, fc_steps):
    #         ε = torch.randn((1,), device=self.device, dtype=self.dtype)
    #         fc[i] = μ + ρ1 * fc[i - 1] + ρ2 * fc[i - 2] + σ * ε
    #     return fc
