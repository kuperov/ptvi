import unittest
from ptvi.cavi import poisson
import numpy as np


class TestPoisson(unittest.TestCase):

    def testSimulate(self):
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        self.assertEqual(X.shape, (100, 4))
        self.assertEqual(y.shape, (100,))
        # try to recover params
        from statsmodels.discrete.discrete_models import Poisson
        ml_fit = Poisson(y, X).fit()
        assert np.linalg.norm(beta0 - ml_fit.params, 2) < 0.05

    