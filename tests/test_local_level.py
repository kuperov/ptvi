import torch
from ptvi.local_level import *


class TestLocalLevelPytorch(object):

    def test_vectorized_loglik(self):
        torch.manual_seed(1234)
        m = LocalLevelModel(input_length=20)
        y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.85)

        torch.manual_seed(1234)
        ε = torch.randn((m.d,))

        fit = m.training_loop(y, lg_iters=8)
