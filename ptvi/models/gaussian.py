from torch.distributions import Normal, LogNormal
from ptvi import Model, global_param


class UnivariateGaussian(Model):
    """Simple univariate Gaussian model.

    For the optimization, we transform σ -> ln(σ) = η to ensure σ > 0.
    """

    name = "Univariate Gaussian model"
    μ = global_param(prior=Normal(0., 10.))
    σ = global_param(prior=LogNormal(0., 10.), rename="η", transform="log")

    def simulate(self, N: int, μ: float, σ: float):
        assert N > 2 and σ > 0
        return Normal(μ, σ).sample((N,))

    def ln_joint(self, y, ζ):
        μ, (σ, η) = self.unpack(ζ)
        ll = Normal(μ, σ).log_prob(y).sum()
        lp = self.μ_prior.log_prob(μ) + self.η_prior.log_prob(η)
        return ll + lp
