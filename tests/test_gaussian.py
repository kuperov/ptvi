import torch
from torch.distributions import Normal, LogNormal
from ptvi import UnivariateGaussian, VIResult
from tests.test_util import TorchTestCase


class TestGaussian(TorchTestCase):

    def test_lnσ_transformation(self):
        model = UnivariateGaussian()
        two = torch.tensor(2.)
        self.assertClose(model.σ_to_η.inv(two), torch.exp(two))
        self.assertClose(model.σ_to_η(two), torch.log(two))

        σ_prior, η_prior = LogNormal(0, 10), model.η_prior

        # η_prior should just be N(0,10)
        ηs = torch.randn(50)
        self.assertClose(η_prior.log_prob(ηs), Normal(0, 10).log_prob(ηs))

        # Transformed log density should include the log abs determinant of the
        # inverse transform σ = log(η), which is log(η)
        σs = model.σ_to_η.inv(ηs)
        self.assertClose(
            η_prior.log_prob(ηs), σ_prior.log_prob(σs) + torch.log(σs))

    def test_smoke_test_training_loop(self):
        model = UnivariateGaussian(num_draws=1, stochastic_entropy=True,
                                   quiet=True)
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ0=μ0, σ0=σ0)
        fit = model.training_loop(y, max_iters=2**4)
        self.assertIsInstance(fit, VIResult)
