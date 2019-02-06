import unittest
from .garch import *
from ptvi.cavi.results import print_results


class GarchTest(unittest.TestCase):
    def test_simulate_garch(self):
        torch.random.manual_seed(123)
        N = 1000
        true_params = dict(
            beta=torch.tensor([1.0, 2.0]),
            A_poly=torch.tensor([0.20, 0.25]),
            D_poly=torch.tensor([0.10, 0.05]),
            a0=0.5,
        )
        p, q = len(true_params["D_poly"]), len(true_params["A_poly"])
        y, X = simulate_garch(N, X=None, **true_params, burnin=10)
        self.assertEqual(y.shape[0], N)
        self.assertEqual(X.shape, (N, 2))
        fit = garch_freq(y, X, p, q)
        self.assertIsInstance(fit, dict)
        print_results(fit, true_values=true_params)
