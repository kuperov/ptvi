import unittest
from ptvi import map, sgvb
from ptvi.models.local_level import *
from ptvi.algos.sgvb import SGVBResult
from ptvi.algos.map import MAPResult
from unittest.mock import patch


class TestLocalLevel(unittest.TestCase):

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
