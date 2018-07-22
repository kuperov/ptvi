import tests.test_util
from ptvi import *
from torch.distributions import (
    Normal, LogNormal, TransformedDistribution, Transform)
import torch

class TestGaussianModel(tests.test_util.TorchTestCase):

    def test_model_attrs(self):
        m = UnivariateGaussian()
        self.assertIsInstance(m.μ_prior, Normal)
        self.assertClose(m.μ_prior.loc, 0.)
        self.assertClose(m.μ_prior.scale, 10.)
        self.assertIsInstance(m.σ_prior, LogNormal)
        self.assertClose(m.σ_prior.loc, 0.)
        self.assertClose(m.σ_prior.scale, 10.)
        self.assertIsInstance(m.η_prior, TransformedDistribution)
        self.assertIsInstance(m.σ_to_η, Transform)

    def test_unpack(self):
        m = UnivariateGaussian()
        ζ = torch.tensor([1.5, 2.5])
        μ, (σ, η) = m.unpack(ζ)
        self.assertClose(1.5, μ)
        self.assertClose(torch.exp(torch.tensor(2.5)), σ)
        self.assertClose(2.5, η)
