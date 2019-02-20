import unittest
from ptvi.cavi import poisson
import numpy as np
from statsmodels.discrete.discrete_model import Poisson


class TestPoisson(unittest.TestCase):

    def testSimulate(self):
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        self.assertEqual(X.shape, (100, 4))
        self.assertEqual(y.shape, (100,))
        # try to recover params using frequentist regression
        ml_fit = Poisson(y, X).fit()
        self.assertLess(np.linalg.norm(beta0 - ml_fit.params, 2), 0.05)

    def testVariationalPoisson(self):
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        mu_0, C_0 = np.zeros(4), np.eye(4)
        fit = poisson.poisson_vi_reg(y, X, mu_0, C_0)
        self.assertIsInstance(fit, dict)
