import torch
from torch.distributions import Normal, LogNormal
from ptvi import UnivariateGaussian
from .test_util import TorchTestCase


class TestGaussian(TorchTestCase):

    def test_lnσ_transformation(self):
        model = UnivariateGaussian(
            μ_prior=Normal(0, 1),
            σ_prior=LogNormal(0, 1)  # i.e. σ =^d exp(N(0,1))
        )
        two = torch.tensor(2.)
        self.assertClose(model.η_to_σ(two), torch.exp(two))
        self.assertClose(model.η_to_σ.inv(two), torch.log(two))

        σ_prior, η_prior = LogNormal(0, 1), model.η_prior

        # η_prior should just be N(0,1)
        ηs = torch.randn(50)
        self.assertClose(η_prior.log_prob(ηs), Normal(0, 1).log_prob(ηs))

        # Transformed log density should include the log abs determinant of the
        # inverse transform σ = log(η), which is log(η)
        σs = model.η_to_σ(ηs)
        self.assertClose(η_prior.log_prob(ηs),
                         σ_prior.log_prob(σs) + torch.log(σs))
