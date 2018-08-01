import torch
from torch.distributions import LogNormal, Normal, StudentT
from unittest.mock import patch

from ptvi import map, sgvb
from ptvi.models import *
from ptvi.algos.sgvb import SGVBResult
from ptvi.algos.map import MAPResult

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

    def test_smoke_test_sgvb(self):
        model = UnivariateGaussian()
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = sgvb(model, y, max_iters=2**4, num_draws=1,
            sim_entropy=True, quiet=True)
        self.assertIsInstance(fit, SGVBResult)

    def test_plots(self):
        torch.manual_seed(123)
        m = UnivariateGaussian()
        N, μ0, σ0 = 100, 5., 5.
        y = m.simulate(N=N, μ=μ0, σ=σ0)
        fit = sgvb(m, y, max_iters=100, quiet=True)

        patch("ptvi.model.plt.show", fit.plot_marg_post('μ'))
        patch("ptvi.model.plt.show", fit.plot_data())
        patch("ptvi.model.plt.show", fit.plot_elbos())


class TestLocalLevel(TorchTestCase):

    def test_training_loop(self):
        torch.manual_seed(123)
        m = LocalLevelModel(input_length=20)
        y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        self.assertEqual(20, len(y))
        self.assertEqual(20, len(z))
        fit = sgvb(m, y, max_iters=8, quiet=True)
        self.assertIsInstance(fit, SGVBResult)

    def test_map_inference(self):
        torch.manual_seed(123)
        model = LocalLevelModel(input_length=100)
        y, z = model.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        # numerically unstable as all hell, but this seems to pass ok
        ζ0 = torch.cat([torch.tensor(y), torch.ones((4,))])
        fit = map(model, y, ζ0=ζ0, quiet=True)
        self.assertIsInstance(fit, MAPResult)
        self.assertIsInstance(fit.ζ, torch.Tensor)
        self.assertEqual(fit.ζ.shape, (model.d,))

    def test_outputs(self):
        torch.manual_seed(123)
        m = LocalLevelModel(input_length=20)
        y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        # we can't do much better than smoke test sampling methods
        fit = sgvb(m, y, max_iters=100, quiet=True)
        ss = fit.sample_paths(N=10, fc_steps=0)
        self.assertEqual(ss.shape, (10, 20))
        self.assertEqual(0, torch.sum(torch.isnan(ss)))
        ss = fit.sample_paths(N=10, fc_steps=10)
        self.assertEqual(ss.shape, (10, 20+10))
        self.assertEqual(0, torch.sum(torch.isnan(ss)))
        summ = fit.summary()
        self.assertTrue(all(summ.index == ['γ', 'η', 'σ', 'ρ']))

    def test_plots(self):
        torch.manual_seed(123)
        m = LocalLevelModel(input_length=20)
        y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        fit = sgvb(m, y, max_iters=100, quiet=True)

        patch("ptvi.model.plt.show", fit.plot_sample_paths())
        patch("ptvi.model.plt.show", fit.plot_pred_ci(fc_steps=2, true_y=y))
        patch("ptvi.model.plt.show", fit.plot_marg_post('η'))
        patch("ptvi.model.plt.show", fit.plot_data())
        patch("ptvi.model.plt.show", fit.plot_elbos())
        patch("ptvi.model.plt.show", fit.plot_latent(true_z=z,
                                                     include_data=True))


class TestFilteredLocalLevelModel(TorchTestCase):

    def test_training(self):
        fll = FilteredLocalLevelModel(input_length=50)
        true_params = dict(γ=0., η=2., ρ=0.95, σ=1.5)
        algo_seed, data_seed = 123, 123
        torch.manual_seed(data_seed)
        y, z = fll.simulate(**true_params)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (50,))
        self.assertIsInstance(z, torch.Tensor)
        self.assertEqual(z.shape, (50,))
        torch.manual_seed(algo_seed)
        fit = sgvb(fll, y, max_iters=8, quiet=True)
        self.assertIsInstance(fit, SGVBResult)

    def test_smoothing(self):
        # we should be able to run the kalman smoother over pretty much any
        # parameters without it blowing up
        fll = FilteredLocalLevelModel(input_length=50)
        true_params = dict(γ=0., η=2., ρ=0.95, σ=1.5)
        algo_seed, data_seed = 123, 123
        torch.manual_seed(data_seed)
        y, z = fll.simulate(**true_params)
        for i in range(10):
            ζ = StudentT(df=4, loc=0, scale=10).sample((fll.d,))
            sm = fll.kalman_smoother(y, ζ)
            for k in ['z_upd', 'Σz_upd', 'z_smooth', 'Σz_smooth', 'y_pred',
                      'Σy_pred']:
                self.assertIsInstance(sm[k], torch.Tensor)
                self.assertFalse(any(torch.isnan(sm[k])))


class TestAR2(TorchTestCase):

    def setUp(self):
        μ, ρ1, ρ2, σ = 1.5, 0.2, 0.1, 1.5
        torch.manual_seed(123)
        params = dict(μ=μ, ρ1=ρ1, ρ2=ρ2, σ=σ)
        self.model = AR2(input_length=100)
        self.y = self.model.simulate(**params)

    def test_sgvb(self):
        fit = sgvb(self.model, self.y, max_iters=200, quiet=True)
        summ = fit.summary()

    def test_map(self):
        fit = map(self.model, self.y, quiet=True)
        summ = fit.summary()

    def test_plots_and_forecasts(self):
        fit = sgvb(self.model, self.y, max_iters=200, quiet=True)

        patch("ptvi.model.plt.show", fit.plot_sample_paths())
        patch("ptvi.model.plt.show", fit.plot_sample_paths(fc_steps=2))
        patch("ptvi.model.plt.show", fit.plot_pred_ci(fc_steps=2, true_y=self.y))
        patch("ptvi.model.plt.show", fit.plot_marg_post('σ'))
        patch("ptvi.model.plt.show", fit.plot_data())
        patch("ptvi.model.plt.show", fit.plot_elbos())
