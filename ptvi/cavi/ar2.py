import math
import torch
from torch.distributions import Normal, LogNormal, Beta
from ptvi import *
from ptvi import Prior


class GammaPrior(Prior):
    """Gamma prior distribution."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        _device = device or torch.device("cpu")
        _a = torch.tensor(self.a, dtype=dtype).to(_device)
        _b = torch.tensor(self.b, dtype=dtype).to(_device)
        return torch.distributions.Gamma(_a, _b)

    def description(self):
        return f"Gamma(concentration={float(self.a)}, rate={float(self.b)})"


class AR2(Model):

    name = "AR(2) model"
    has_observation_error = False
    μ = global_param(NormalPrior(0, 1))
    ρ1 = global_param(NormalPrior(0, 1), transform="logit", rename="φ1")
    ρ2 = global_param(NormalPrior(0, 1), transform="logit", rename="φ2")
    τ = global_param(GammaPrior(2, 0.5), transform="log", rename="η")

    def ln_joint(self, y, ζ):
        μ, (ρ1, φ1), (ρ2, φ2), (τ, η) = self.unpack(ζ)
        llhood = (
            Normal(μ + y[1:-1] * ρ1 + y[:-2] * ρ2, 1 / torch.sqrt(τ))
            .log_prob(y[2:])
            .sum()
        )
        lprior = (
            self.μ_prior.log_prob(μ)
            + self.φ1_prior.log_prob(φ1)
            + self.φ2_prior.log_prob(φ2)
            + self.η_prior.log_prob(η)
        )
        return llhood + lprior

    def simulate(self, μ, ρ1, ρ2, τ):
        assert self.input_length is not None
        y = torch.empty((self.input_length,))
        y[0] = Normal(
            μ / (1 - ρ1 - ρ2),
            1
            / math.sqrt(τ)
            * math.sqrt((1 - ρ2) / ((1 + ρ2) * ((1 - ρ2) ** 2 - ρ1 ** 2))),
        ).sample()
        y[1] = (
            μ
            + ρ1 * y[0]
            + Normal(μ / (1 - ρ2), 1 / math.sqrt(τ * (1 - ρ2 ** 2))).sample()
        )
        for i in range(2, self.input_length):
            y[i] = (
                μ + ρ1 * y[i - 1] + ρ2 * y[i - 2] + Normal(0, 1 / math.sqrt(τ)).sample()
            )
        return y

    def forecast(self, ζ: torch.Tensor, y: torch.Tensor, fc_steps: int) -> torch.Tensor:
        assert fc_steps >= 1, "Must forecast at least 1 step."
        μ, (ρ1, φ1), (ρ2, φ2), (τ, η) = self.unpack(ζ)
        fc = torch.empty((fc_steps,))
        fc[0] = μ + ρ1 * y[-1] + ρ2 * y[-2] + Normal(0, 1 / math.sqrt(τ)).sample()
        if fc_steps > 1:
            fc[1] = μ + ρ1 * fc[0] + ρ2 * y[-1] + Normal(0, 1 / math.sqrt(τ)).sample()
        for i in range(2, fc_steps):
            fc[i] = (
                μ
                + ρ1 * fc[i - 1]
                + ρ2 * fc[i - 2]
                + Normal(0, 1 / math.sqrt(τ)).sample()
            )
        return fc


def arp_sgvb_forecast(y, p, steps, fit, draws=1000):
    """Forecast an AR(p) `steps` ahead using existing model fit.

    Args:
        y:        data
        p:        lags
        steps:    steps ahead
        fit:      model fit
        draws:    number of draws per time-step
        **kwargs: keyword kwargs to pass to arp_mcmc

    Returns
        Dict of {t: draws from the p(y_{t+steps}| y_{1:N}) marginal}.

    Time indexes are zero-based, so the first forecast has index N, etc.
    """
    N = len(y)
    paths = fit.sample_paths(draws, steps).cpu().numpy()
    return {s: paths[:, s] for s in range(N, N + steps)}
