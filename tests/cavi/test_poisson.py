import unittest
from ptvi.cavi import poisson
import numpy as np
from statsmodels.discrete.discrete_model import Poisson


class TestPoisson(unittest.TestCase):

    def testSimulate(self):
        np.random.seed(123)
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        self.assertEqual(X.shape, (100, 4))
        self.assertEqual(y.shape, (100,))
        # try to recover params using frequentist regression
        ml_fit = Poisson(y, X).fit()
        self.assertLess(np.linalg.norm(beta0 - ml_fit.params, 2), 2.0)

    def testVariationalPoisson(self):
        np.random.seed(123)
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        mu_0, C_0 = np.zeros(4), np.eye(4)
        fit = poisson.vi_reg(y, X, mu_0, C_0)
        self.assertIsInstance(fit, dict)
        self.assertLess(np.linalg.norm(beta0 - fit['x_bar'], 2), 2.0)  # super lax

    def testStanPoisson(self):
        np.random.seed(123)
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        mu_0, C_0 = np.zeros(4), np.eye(4)
        fit = poisson.stan_reg(y, X, mu_0, C_0)
        draws = fit.extract('beta')['beta']
        means = np.mean(draws, axis=0)
        self.assertLess(np.linalg.norm(beta0 - means, 2), 2.0)

    def testMHPoisson(self):
        np.random.seed(123)
        beta0 = np.r_[1.1, 2.2, 3.3, 4.4]
        y, X = poisson.simulate(100, beta0)
        mu_0, C_0 = np.zeros(4), np.eye(4)
        draws = poisson.mh_reg(y, X, mu_0, C_0, warmup=400, num_draws=1_000)
        means = np.mean(draws, axis=0)
        self.assertLess(np.linalg.norm(beta0 - means, 2), 2.0)

    def testForecast(self):
        rs = np.random.RandomState(seed=123)
        beta0, phi0 = np.r_[1.1], np.r_[0.3]
        y, X = poisson.simulate_ar(100, beta=beta0, phi=phi0)
        mu_0, C_0 = np.zeros(2), np.eye(2) * 0.1
        y_, X_ = poisson.ar_design_matrix(y, X, p=1, c=0.2)
        fit = poisson.vi_reg(y_, X_, mu_0=mu_0, C_0=C_0)
        fc = poisson.forecast_arp(y, X, fit, p=1, steps=10, num_draws=500, rs=rs)
        self.assertIsInstance(fc, dict)
        for i in range(100, 110):
            self.assertTrue(i in fc)
            self.assertIsInstance(fc[i], np.ndarray)

    def testMixtureDist(self):
        d = poisson.poisson_mixture(mus=np.r_[1.2, 10.5, 10.2])
        # note we (deliberately) have the strange behavior of simulating one rv per
        # mixture member
        self.assertEqual(len(d.rvs()), 3)
        self.assertIsInstance(d.pmf(2), float)

    def testForecastMixture(self):
        rs = np.random.RandomState(seed=123)
        beta0, phi0 = np.r_[1.1], np.r_[0.3]
        y, X = poisson.simulate_ar(100, beta=beta0, phi=phi0)
        mu_0, C_0 = np.zeros(2), np.eye(2) * 0.1
        y_, X_ = poisson.ar_design_matrix(y, X, p=1, c=0.2)
        fit = poisson.vi_reg(y_, X_, mu_0=mu_0, C_0=C_0)
        fc = poisson.forecast_arp_pmfs(y, X, fit, p=1, steps=10, num_draws=500, rs=rs)
        self.assertIsInstance(fc, dict)
        from scipy import stats
        for i in range(100, 110):
            self.assertTrue(i in fc)
            self.assertIsInstance(fc[i], stats._distn_infrastructure.rv_frozen)
        scores = poisson.score_arp_pmf_forecasts(y, X, beta0, phi0, [fc, fc], p=1, steps=10, num_draws=50)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (10, 2))
        # TODO: verify scores are exactly equal, because same forecast
