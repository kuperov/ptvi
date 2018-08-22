from warnings import warn
import unittest
from ptvi import *
import torch


if torch.cuda.is_available():
    cuda = torch.device("cuda")
else:
    warn("WARNING: executing CUDA tests on CPU")
    cuda = torch.device("cpu")


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

    def test_hessian_double_gpu(self):
        model = UnivariateGaussian(dtype=torch.float64, device=cuda)
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = map(model, y, quiet=True)
        # approximating density should also be float64 and on the gpu
        self.assertEqual(fit.q.loc.device.type, "cuda")
        self.assertEqual(fit.q.loc.dtype, torch.float64)
        self.assertEqual(fit.q.variance.device.type, "cuda")
        self.assertEqual(fit.q.variance.dtype, torch.float64)
        # ditto for hessian
        g, H = fit.ln_joint_grad_hessian()
        # TODO

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

    def test_initial_conditions_double_gpu(self):
        model = UnivariateGaussian(dtype=torch.float64, device=torch.device("cuda"))
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = map(model, y, quiet=True)
        u, L = fit.initial_conditions()
        # approximating density should also be float64 and on the gpu
        self.assertEqual(u.device.type, "cuda")
        self.assertEqual(u.dtype, torch.float64)
        self.assertEqual(L.device.type, "cuda")
        self.assertEqual(L.dtype, torch.float64)


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
        self.assertEqual(len(fit.objectives), 10)

    def test_opt_fivo_model_cuda(self):
        data_seed, algo_seed = 1234, 1234
        params = dict(a=1., b=0., c=.95)
        T = 200
        torch.manual_seed(algo_seed)
        model = FilteredStochasticVolatilityModelFreeProposal(
            input_length=T,
            num_particles=10,
            resample=True,
            device=cuda,
            dtype=torch.float64,
        )
        torch.manual_seed(data_seed)
        y, z_true = model.simulate(**params)
        fit = stoch_opt(model, y, max_iters=4, quiet=True)
