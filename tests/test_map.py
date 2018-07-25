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