import tests.test_util
import numbers
import torch
from torch.distributions import Gumbel, StudentT

from ptvi.algos.fivo import *
from ptvi.algos.sgvb import SGVBResult
from ptvi import sgvb


class TestFIVO(tests.test_util.TorchTestCase):
    def test_stochvol(self):
        torch.manual_seed(123)
        T = 200
        model = FilteredStochasticVolatilityModel(
            input_length=T,
            proposal=AR1Proposal(0, .95),
            num_particles=50,
        )
        params = dict(a=1., b=0., c=.95)
        y, z_true = model.simulate(**params)
        ζ = torch.zeros(model.d)
        lj = model.ln_joint(y, ζ)
        self.assertIsInstance(lj, torch.Tensor)
        # check we can do sgvb
        fit = sgvb(model, y, quiet=True, max_iters=8)
        self.assertIsInstance(fit, SGVBResult)

    def test_proposal(self):
        ar1norm = AR1Proposal(ρ=0.9)
        self.assertIn("ε_t ~ Normal(loc=0.00, scale=1.00)", repr(ar1norm))
        ar1student = AR1Proposal(ρ=0.9, μ=0.5, ε_type=StudentT, df=3)
        Z = torch.empty((100, 50))
        Z[0, :] = ar1student.conditional_sample(0, Z)
        Z[1, :] = ar1student.conditional_sample(1, Z)
        ar1g = AR1Proposal(ρ=0.9, μ=10., ε_type=Gumbel)
        Z = torch.empty((10, 12))
        Z[0, :] = ar1g.conditional_sample(0, Z)
        Z[1, :] = ar1g.conditional_sample(1, Z)
