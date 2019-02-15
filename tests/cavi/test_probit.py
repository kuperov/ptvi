import unittest
from scipy import stats
import numpy as np


class TestProbit(unittest.TestCase):

    def set_up(self):
        np.random.seed(123)
        self.N, self.k = 500, 10
        self.X = np.random.normal(size=[N, k])
        self.mu_beta, self.Sigma_beta = np.zeros(k), np.eye(k)
        self.beta0 = np.random.normal(size=k)
        Phi = stats.norm().cdf
        self.y = np.random.binomial(n=1, p=Phi(X @ beta0), size=N)

    def test_inference(self):
        fit = vi_probit(self.y, self.X, maxiter=1000, mu_beta=self.mu_beta, Sigma_beta=self.Sigma_beta)
        q_beta = fit['q_beta']
        self.assertIsInstance(fit, stats.rv_continuous)
        mu_q_beta, Sigma_beta = fit['mu_q_beta'], fit['Sigma_q_beta']
        # posterior should be reasonably close to true values
        self.assertTrue(q_beta.pdf(self.beta0) > 0.1**10)

    def test_log_joint():
        q_beta, q_a = fit["q_beta"], fit["q_a"]
        beta, a = q_beta.rvs(), q_a.rvs()
        lj = log_joint(y, X, beta, a, mu_beta, Sigma_beta)
