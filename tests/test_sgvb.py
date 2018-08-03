import unittest
import torch
from ptvi.algos import mf_sgvb
from ptvi import LocalLevelModel, UnivariateGaussian
from ptvi.stopping import SupGrowthStoppingHeuristic


class TestMFSGVB(unittest.TestCase):
    def test_sgvb_gaussian(self):
        model = UnivariateGaussian()
        torch.manual_seed(123)
        N, μ0, σ0 = 100, 5., 5.
        y = model.simulate(N=N, μ=μ0, σ=σ0)
        fit = mf_sgvb(
            model,
            y,
            max_iters=2 ** 4,
            num_draws=1,
            sim_entropy=True,
            quiet=True,
            stop_heur=SupGrowthStoppingHeuristic(),
        )

    def test_sgvb_local_level(self):
        torch.manual_seed(123)
        model = LocalLevelModel(input_length=500)
        y, z = model.simulate(γ=0., η=2., σ=1.5, ρ=0.85)
        stop_heur = SupGrowthStoppingHeuristic(skip=20, min_steps=2_000)
        fit = mf_sgvb(
            model,
            y,
            max_iters=2 ** 4,
            num_draws=1,
            sim_entropy=True,
            quiet=True,
            stop_heur=stop_heur,
        )
