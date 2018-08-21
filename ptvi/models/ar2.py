import math
import torch
from torch.distributions import Normal, LogNormal, Beta
from ptvi import *


class AR2(Model):

    name = "AR(2) model"
    has_observation_error = False
    μ = global_param(NormalPrior(0, 1))
    ρ1 = global_param(BetaPrior(2, 2), transform="logit", rename="φ1")
    ρ2 = global_param(BetaPrior(2, 2), transform="logit", rename="φ2")
    σ = global_param(LogNormalPrior(0, 1), transform="log", rename="η")

    def ln_joint(self, y, ζ):
        μ, (ρ1, φ1), (ρ2, φ2), (σ, η) = self.unpack(ζ)
        llhood = Normal(μ + y[1:-1] * ρ1 + y[:-2] * ρ2, σ).log_prob(y[2:]).sum()
        lprior = (
            self.μ_prior.log_prob(μ)
            + self.φ1_prior.log_prob(φ1)
            + self.φ2_prior.log_prob(φ2)
            + self.η_prior.log_prob(η)
        )
        return llhood + lprior

    def simulate(self, μ, ρ1, ρ2, σ):
        assert self.input_length is not None
        y = torch.empty((self.input_length,))
        y[0] = Normal(
            μ / (1 - ρ1 - ρ2),
            σ * math.sqrt((1 - ρ2) / ((1 + ρ2) * ((1 - ρ2) ** 2 - ρ1 ** 2))),
        ).sample()
        y[1] = μ + ρ1 * y[0] + Normal(μ / (1 - ρ2), σ / math.sqrt(1 - ρ2 ** 2)).sample()
        for i in range(2, self.input_length):
            y[i] = μ + ρ1 * y[i - 1] + ρ2 * y[i - 2] + Normal(0, σ).sample()
        return y.type(self.dtype).to(self.device)

    def forecast(self, ζ: torch.Tensor, y: torch.Tensor, fc_steps: int) -> torch.Tensor:
        assert fc_steps >= 1, "Must forecast at least 1 step."
        μ, (ρ1, φ1), (ρ2, φ2), (σ, η) = self.unpack(ζ)
        fc = torch.empty((fc_steps,), device=self.device, dtype=self.dtype)
        ε = torch.randn((1,), device=self.device, dtype=self.dtype)
        fc[0] = μ + ρ1 * y[-1] + ρ2 * y[-2] + σ * ε
        if fc_steps > 1:
            ε = torch.randn((1,), device=self.device, dtype=self.dtype)
            fc[1] = μ + ρ1 * fc[0] + ρ2 * y[-1] + σ * ε
        for i in range(2, fc_steps):
            ε = torch.randn((1,), device=self.device, dtype=self.dtype)
            fc[i] = μ + ρ1 * fc[i - 1] + ρ2 * fc[i - 2] + σ * ε
        return fc
