import tests.test_util
import numbers
import torch

from ptvi.algos.fivo import *
from ptvi import sgvb


class TestFIVO(tests.test_util.TorchTestCase):
    def test_stochvol(self):
        torch.manual_seed(123)
        T = 200
        model = FilteredStochasticVolatilityModel(
            input_length=T, proposal=AR1Proposal(0, .95), num_particles=50
        )
        params = dict(a=1., b=0., c=.95)
        y, z_true = model.simulate(**params)
        ζ = torch.zeros(model.d)
        lj = model.ln_joint(y, ζ)
        self.assertIsInstance(lj, torch.Tensor)
