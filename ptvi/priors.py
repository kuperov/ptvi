"""Wrappers for constructing probability distributions for use as priors.

These wrappers defer construction to allow the model to specify a data type,
such as torch.double, for the parameters. They also serialize nicely for display
in notebooks.
"""

import torch
from torch.distributions import Distribution, Normal, LogNormal
import ptvi.dist


class Prior(object):

    def to_distribution(self, dtype: torch.dtype=torch.float32) -> Distribution:
        """Construct the Distribution corresponding to this prior."""
        raise NotImplementedError

    def description(self):
        raise NotImplementedError

    def __str__(self):
        return self.description()


class NormalPrior(Prior):
    """A normal prior with mean μ and s.d. σ."""

    def __init__(self, μ, σ):
        self.μ, self.σ = μ, σ

    def to_distribution(self, dtype: torch.dtype=torch.float32):
        _μ = torch.tensor(self.μ, dtype=dtype)
        _σ = torch.tensor(self.σ, dtype=dtype)
        return torch.distributions.Normal(loc=_μ, scale=_σ)

    def description(self):
        return f"Normal(μ={self.μ}, σ^2={self.σ}^2)"


class LogNormalPrior(Prior):
    """A log normal prior, where X ~ LogNormal(μ, σ^2) means log(X) ~ Normal(μ, σ^2).
    """

    def __init__(self, μ, σ):
        self.μ, self.σ = μ, σ

    def to_distribution(self, dtype: torch.dtype=torch.float32):
        _μ = torch.tensor(self.μ, dtype=dtype)
        _σ = torch.tensor(self.σ, dtype=dtype)
        return torch.distributions.LogNormal(loc=_μ, scale=_σ)

    def description(self):
        return f"LogNormal(μ={self.μ}, σ^2={self.σ}^2)"


class BetaPrior(Prior):
    """Beta prior distribution."""

    def __init__(self, α, β):
        self.α, self.β = α, β

    def to_distribution(self, dtype: torch.dtype = torch.float32):
        _α = torch.tensor(self.α, dtype=dtype)
        _β = torch.tensor(self.β, dtype=dtype)
        return torch.distributions.LogNormal(loc=_α, scale=_β)

    def description(self):
        return f"Beta(α={self.α}, β={self.β})"
