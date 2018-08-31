"""Test 'fivo' model for particle filtering state-space models.
"""

from unittest.mock import patch

import tests.test_util

from ptvi import *
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
        # check we can do sgvb
        fit = sgvb(model, y, quiet=True, max_iters=8)
        self.assertIsNotNone(fit)

    def test_proposal(self):
        ar1norm = AR1Proposal(μ=0, σ=1, ρ=0.9)
        self.assertIn("η_t ~ Ν(0,1.00)", repr(ar1norm))
        Z = torch.empty((10, 12))
        Z[0, :] = ar1norm.conditional_sample(0, Z, 12)
        Z[1, :] = ar1norm.conditional_sample(1, Z, 12)

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
        y, z_true = model.simulate(**params)

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

    def test_double_precision_gradient(self):
        torch.manual_seed(123)
        T = 200
        model = FilteredStochasticVolatilityModel(
            input_length=T, num_particles=5, dtype=torch.float64
        )
        params = dict(a=1., b=0., c=.95)
        y, z_true = model.simulate(**params)

        ζ = torch.tensor(list(params.values()), requires_grad=True, dtype=torch.float64)
        ghat = torch.zeros((3,), dtype=torch.float64)
        reps = 5
        for i in range(reps):
            if ζ.grad is not None:
                ζ.grad.zero_()
            phatN = model.simulate_log_phatN(y, ζ)
            phatN.backward()
            ghat += ζ.grad
        ghat /= reps
        self.assertFalse(any(torch.isnan(ghat)))
