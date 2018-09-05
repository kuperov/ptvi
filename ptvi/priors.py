"""Wrappers for constructing probability distributions for use as priors.

These wrappers defer construction to allow the model to specify a data type,
such as torch.double, for the parameters. They also serialize nicely for display
in notebooks.
"""

from ptvi import InvGamma, Improper

import torch
from torch.distributions import Distribution


class Prior(object):
    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ) -> Distribution:
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

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        _device = device or torch.device("cpu")
        _μ = torch.tensor(self.μ, dtype=dtype).to(_device)
        _σ = torch.tensor(self.σ, dtype=dtype).to(_device)
        return torch.distributions.Normal(loc=_μ, scale=_σ)

    def description(self):
        return f"Normal(μ={float(self.μ)}, σ²={float(self.σ**2)})"


class LogNormalPrior(Prior):
    """A log normal prior, where X ~ LogNormal(μ, σ^2) means log(X) ~ Normal(μ, σ^2).
    """

    def __init__(self, μ, σ):
        self.μ, self.σ = μ, σ

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        _device = device or torch.device("cpu")
        _μ = torch.tensor(self.μ, dtype=dtype).to(_device)
        _σ = torch.tensor(self.σ, dtype=dtype).to(_device)
        return torch.distributions.LogNormal(loc=_μ, scale=_σ)

    def description(self):
        return f"LogNormal(μ={float(self.μ)}, σ²={float(self.σ**2)})"


class BetaPrior(Prior):
    """Beta prior distribution."""

    def __init__(self, α, β):
        self.α, self.β = α, β

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        _device = device or torch.device("cpu")
        _α = torch.tensor(self.α, dtype=dtype).to(_device)
        _β = torch.tensor(self.β, dtype=dtype).to(_device)
        return torch.distributions.Beta(_α, _β)

    def description(self):
        return f"Beta(α={float(self.α)}, β={float(self.β)})"


class ModifiedBetaPrior(Prior):
    """Beta prior distribution, modified to lie between -1 and 1."""

    def __init__(self, α, β):
        self.α, self.β = α, β

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        _device = device or torch.device("cpu")
        _α = torch.tensor(self.α, dtype=dtype).to(_device)
        _β = torch.tensor(self.β, dtype=dtype).to(_device)
        return torch.distributions.TransformedDistribution(
            torch.distributions.Beta(_α, _β),
            torch.distributions.AffineTransform(loc=-1., scale=2.))

    def description(self):
        return f"2*Beta(α={float(self.α)}, β={float(self.β)})-1"


class InvGammaPrior(Prior):
    """Inverse gamma prior distribution."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        _device = device or torch.device("cpu")
        _a = torch.tensor(self.a, dtype=dtype).to(_device)
        _b = torch.tensor(self.b, dtype=dtype).to(_device)
        return InvGamma(_a, _b)

    def description(self):
        return f"InvGamma(a={float(self.a)}, b={float(self.b)})"


class ImproperPrior(Prior):
    """Improper uniform prior with support over the real line."""

    def to_distribution(
        self, dtype: torch.dtype = torch.float32, device: torch.device = None
    ):
        return Improper()

    def description(self):
        return f"Improper(-∞, +∞)"
