import unittest
from scipy import stats
import numpy as np
from ptvi.cavi import probit


class TestProbit(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.N, self.k = 500, 10
        self.X = np.random.normal(size=[self.N, self.k])
        self.mu_beta, self.Sigma_beta = np.zeros(self.k), np.eye(self.k)
        self.beta0 = np.random.normal(size=self.k)
        Phi = stats.norm().cdf
        self.y = np.random.binomial(n=1, p=Phi(self.X @ self.beta0), size=self.N)

    def test_inference(self):
        fit = probit.vi_probit(self.y, self.X, maxiter=1000, mu_beta=self.mu_beta, Sigma_beta=self.Sigma_beta)
        q_beta = fit['q_beta']
        self.assertIsInstance(fit['q_beta'], stats._multivariate.multivariate_normal_frozen)
        mu_q_beta, Sigma_beta = fit['mu_q_beta'], fit['Sigma_q_beta']
        # posterior should be reasonably close to true values
        self.assertTrue(q_beta.pdf(self.beta0) > 0.1**10)
