from warnings import warn
import unittest
from ptvi import *
import torch


if torch.cuda.is_available():
    cuda = torch.device("cuda")
else:
    warn("WARNING: executing CUDA tests on CPU")
    cuda = torch.device("cpu")


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


class TestDualStochOpt(unittest.TestCase):
    def test_basic_optimization(self):
        data_seed, algo_seed = 123, 123
        params = dict(a=1., b=0., c=.95)
        T = 200
        model = FilteredSVModelDualOpt(input_length=T, num_particles=10, resample=True)
        torch.manual_seed(data_seed)
        y, z_true = model.simulate(**params)
        torch.manual_seed(algo_seed)
        fit = dual_stoch_opt(model, y, max_iters=10, quiet=True)
        # if this converges before 10 iters, something is wrong
        self.assertEqual(len(fit.objectives), 10)

    def test_basic_optimization_double_gpu(self):
        data_seed, algo_seed = 123, 123
        params = dict(a=1., b=0., c=.95)
        T = 200
        model = FilteredSVModelDualOpt(
            input_length=T,
            num_particles=10,
            resample=True,
            dtype=torch.float64,
            device=cuda,
        )
        torch.manual_seed(data_seed)
        y, z_true = model.simulate(**params)
        torch.manual_seed(algo_seed)
        fit = dual_stoch_opt(model, y, max_iters=10, quiet=True)
        # if this converges before 10 iters, something is wrong
        self.assertEqual(len(fit.objectives), 10)
