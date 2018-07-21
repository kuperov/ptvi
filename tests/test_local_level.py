import unittest
from ptvi.local_level import *
from unittest.mock import patch


class TestLocalLevel(unittest.TestCase):

    def test_training_loop(self):
        torch.manual_seed(123)
        m = LocalLevelModel(input_length=20, quiet=True)
        y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        self.assertEqual(20, len(y))
        self.assertEqual(20, len(z))
        fit = m.training_loop(y, max_iters=8)
        self.assertIsInstance(fit, VITimeSeriesResult)
        # we can't do much better than smoke test sampling methods
        ss = m.sample_paths(10, 10)
        self.assertEqual(ss.shape, (10, 10+20))
        self.assertEqual(0, torch.sum(torch.isnan(ss)))

    def test_plots(self):
        torch.manual_seed(123)
        m = LocalLevelModel(input_length=20, quiet=True)
        y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        fit = m.training_loop(y, max_iters=100)

        patch("ptvi.model.plt.show", fit.plot_sample_paths())
        patch("ptvi.model.plt.show", fit.plot_pred_ci(fc_steps=2, true_y=y))
        patch("ptvi.model.plt.show", fit.plot_marg_post('η'))
        patch("ptvi.model.plt.show", fit.plot_data())
        patch("ptvi.model.plt.show", fit.plot_elbos())
        patch("ptvi.model.plt.show", fit.plot_latent(true_z=z,
                                                     include_data=True))
