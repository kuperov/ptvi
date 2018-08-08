from unittest.mock import patch

import tests.test_util

from ptvi import sgvb
from ptvi.fivo import *
from ptvi.models.filtered_sv_model import *


class TestFIVO(tests.test_util.TorchTestCase):
    def test_stochvol(self):
        torch.manual_seed(123)
        T = 200
        model = FilteredStochasticVolatilityModel(input_length=T, num_particles=50)
        params = dict(a=1., b=0., c=.95)
        y, z_true = model.simulate(**params)
        ζ = torch.zeros(model.d)
        lj = model.ln_joint(y, ζ)
        self.assertIsInstance(lj, torch.Tensor)

    def test_samples(self):
        torch.manual_seed(123)
        T = 200
        model = FilteredStochasticVolatilityModel(input_length=T, num_particles=50)
        params = dict(a=1., b=0., c=.95)
        y, z_true = model.simulate(**params)
        ζ = torch.zeros(model.d)
        fit = sgvb(model, y, max_iters=20, quiet=True)  # crap fit but whatevs

        patch("ptvi.model.plt.show", fit.plot_marg_post("a"))
        patch("ptvi.model.plt.show", fit.plot_data())
        patch("ptvi.model.plt.show", fit.plot_elbos())
        patch("ptvi.model.plt.show", fit.plot_pred_ci(N=10, fc_steps=2, true_y=y))
        patch("ptvi.model.plt.show", fit.plot_latent(N=10, true_z=z_true))

    def test_log_phatN_gradient(self):
        torch.manual_seed(123)
        T = 200
        model = FilteredStochasticVolatilityModel(input_length=T, num_particles=5)
        params = dict(a=1., b=0., c=.95)
        y, z_true = model.simulate(a=1., b=0., c=.95)

        ζ = torch.tensor(list(params.values()), requires_grad=True)
        ghat = torch.zeros((3,))
        reps = 5
        for i in range(reps):
            if ζ.grad is not None:
                ζ.grad.zero_()
            phatN = model.simulate_log_phatN(y, ζ)
            phatN.backward()
            ghat += ζ.grad
        ghat /= reps
        self.assertFalse(any(torch.isnan(ghat)))
