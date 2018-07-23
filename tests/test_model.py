import math
import tests.test_util
from ptvi import *
from ptvi.model import (
    _LocalParameter, _ModelParameter, _TransformedModelParameter)
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

    def test_param_attrs(self):

        class TestModel(VIModel):
            z = local_param()
            a = global_param(prior=Normal(0, 1))
            b = global_param(prior=InvGamma(2, 2), transform='log')

        m = TestModel(input_length=10)
        self.assertIsInstance(m.z, _LocalParameter)
        self.assertEqual(m.z.name, 'z')
        self.assertIsInstance(m.z.prior, Improper)
        self.assertIsInstance(m.a, _ModelParameter)

        self.assertIsInstance(m.b, _TransformedModelParameter)
        self.assertEqual(m.b.name, 'b')
        self.assertEqual(m.b.transform_desc, 'log')
        self.assertEqual(m.b.transformed_name, 'log_b')

        self.assertEqual(m.params, [m.z, m.a, m.b])

    def test_map(self):
        model = UnivariateGaussian(quiet=True)
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ0=μ0, σ0=σ0)
        ζ = model.map(y, max_iters=20)
        self.assertIsInstance(ζ, torch.Tensor)
        self.assertEqual(ζ.shape, (model.d,))

    def test_hessian(self):
        model = UnivariateGaussian(quiet=True)
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ0=μ0, σ0=σ0)
        ζ = model.map(y)
        g, H = model.ln_joint_grad_hessian(y, ζ)
        self.assertIsInstance(g, torch.Tensor)
        self.assertEqual(g.shape, (model.d,))
        self.assertIsInstance(H, torch.Tensor)
        self.assertEqual(H.shape, (model.d, model.d))

    def test_initial_conditions(self):
        model = UnivariateGaussian(quiet=True)
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ0=μ0, σ0=σ0)
        u, L = model.initial_conditions(y)
        self.assertIsInstance(u, torch.Tensor)
        self.assertIsInstance(L, torch.Tensor)
        self.assertEqual(u.shape, (model.d,))
        self.assertEqual(L.shape, (model.d, model.d))
