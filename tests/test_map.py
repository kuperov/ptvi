import unittest
from ptvi import *
import torch


class TestMap(unittest.TestCase):
    def test_hessian(self):
        model = UnivariateGaussian()
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = map(model, y, quiet=True)
        g, H = fit.ln_joint_grad_hessian()
        self.assertIsInstance(g, torch.Tensor)
        self.assertEqual(g.shape, (model.d,))
        self.assertIsInstance(H, torch.Tensor)
        self.assertEqual(H.shape, (model.d, model.d))

    def test_initial_conditions(self):
        model = UnivariateGaussian()
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = map(model, y, quiet=True)
        u, L = fit.initial_conditions()
        self.assertIsInstance(u, torch.Tensor)
        self.assertIsInstance(L, torch.Tensor)
        self.assertEqual(u.shape, (model.d,))
        self.assertEqual(L.shape, (model.d, model.d))


class TestStochOpt(unittest.TestCase):
    def test_basic_optimization(self):
        data_seed, algo_seed = 123, 123
        params = dict(a=1., b=0., c=.95)
        T = 200
        model = FilteredStochasticVolatilityModelFreeProposal(
            input_length=T, num_particles=30, resample=True
        )
        torch.manual_seed(data_seed)
        y, z_true = model.simulate(**params)
        torch.manual_seed(algo_seed)
        fit = stoch_opt(model, y, max_iters=10, quiet=True)
        # if this converges before 10 iters, something is wrong
        self.assertEqual(len(fit.losses), 10)
