import unittest
import os
import torch
from ptvi.algos import mf_sgvb
from ptvi import LocalLevelModel, UnivariateGaussian, MVNPosterior
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

    def test_load_and_save_posteriors(self):
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
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), 'dummy.posterior')
        fit.save(path)
        fit2 = MVNPosterior.load(path)
        self.assertTrue(torch.allclose(fit.q.loc, fit2.q.loc))
        self.assertTrue(torch.allclose(fit.q.scale_tril, fit2.q.scale_tril))
