from warnings import warn
import tests.test_util
import pandas as pd
from ptvi import *
from ptvi.model import LocalParameter, ModelParameter, TransformedModelParameter
from torch.distributions import Normal, LogNormal, TransformedDistribution, Transform
import torch


if torch.cuda.is_available():
    cuda = torch.device("cuda")
else:
    warn("WARNING: CUDA unavailable, running tests on CPU")
    cuda = torch.device("cpu")


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

        μ, σ = m.unpack_natural(ζ)
        self.assertClose(1.5, μ)
        self.assertClose(torch.exp(torch.tensor(2.5)), σ)

        μ, η = m.unpack_unrestricted(ζ)
        self.assertClose(1.5, μ)
        self.assertClose(2.5, η)

    def test_param_attrs(self):
        class TestModel(Model):
            z = local_param()
            a = global_param(prior=NormalPrior(0, 1))
            b = global_param(prior=InvGammaPrior(2, 2), transform="log")

        m = TestModel(input_length=10)
        self.assertIsInstance(m.z, LocalParameter)
        self.assertEqual(m.z.name, "z")
        self.assertIsInstance(m.z.prior, ImproperPrior)
        self.assertIsInstance(m.a, ModelParameter)

        self.assertIsInstance(m.b, TransformedModelParameter)
        self.assertEqual(m.b.name, "b")
        self.assertEqual(m.b.transform_desc, "log")
        self.assertEqual(m.b.transformed_name, "log_b")

        self.assertEqual(m.params, [m.z, m.a, m.b])

        # check prior
        ζ = torch.cat([torch.zeros(10), torch.tensor([1., 0.])])
        lp_ζ = Normal(0, 1).log_prob(torch.tensor(1.)) + InvGamma(2, 2).log_prob(
            torch.tensor(1.)
        )
        self.assertEqual(m.ln_prior(ζ), lp_ζ)

    def test_map(self):
        model = UnivariateGaussian()
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = map(model, y, max_iters=20, quiet=True)
        self.assertIsInstance(fit.ζ, torch.Tensor)
        self.assertEqual(fit.ζ.shape, (model.d,))

    def test_sgvb(self):
        model = UnivariateGaussian()
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = sgvb(model, y, max_iters=20, quiet=True)
        self.assertIsInstance(fit, MVNPosterior)
        self.assertIsInstance(fit.summary(), pd.DataFrame)  # good smoke test
        fit = mf_sgvb(model, y, max_iters=20, quiet=True)
        self.assertIsInstance(fit, MVNPosterior)
        self.assertIsInstance(fit.summary(), pd.DataFrame)  # good smoke test


class TestFilteredModel(tests.test_util.TorchTestCase):
    def test_SV_filtered_gpu(self):
        params = dict(a=1., b=0., c=.95)
        T = 200
        model = FilteredStochasticVolatilityModelFreeProposal(
            input_length=T,
            num_particles=3000,
            resample=True,
            device=cuda,
            dtype=torch.float64,
        )
        torch.manual_seed(123)
        y, z_true = model.simulate(**params)
        self.assertEqual(y.device.type, cuda.type)
        self.assertEqual(z_true.device.type, cuda.type)
        ζ0 = torch.zeros((6,), device=cuda, dtype=torch.float64)
        lj = model.ln_joint(y, ζ0)
