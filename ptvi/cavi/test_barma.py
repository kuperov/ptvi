import unittest

import numpy as np
from ptvi.cavi import barma, probit, mcmc


class TestProbitBAR(unittest.TestCase):
    def testSimulate(self):
        np.random.seed(123)
        beta0 = np.r_[0.5, -0.5]
        phi0 = np.r_[0.8, 0.15]
        N = 100
        k, = len(beta0)
        y, X = barma.simulate_bar(N, beta0, phi0)
        self.assertEqual(max(y), 1)
        self.assertEqual(min(y), 0)
        self.assertEqual(X.shape, (N, k))

    def testDesignMatrix(self):
        np.random.seed(123)
        beta0 = np.r_[0.5, -0.5]
        phi0 = np.r_[0.8, 0.15]
        N = 100
        k, p, = len(beta0), len(phi0)
        y, X = barma.simulate_bar(N, beta0, phi0)
        y_, X_ = barma.bar_design_matrix(y, X, p)
        self.assertEqual(X_.shape, (N - p, p + k))
        self.assertEqual(y_.shape, (len(y) - p,))

    def testCombineBarPriors(self):
        mu, sig = barma.combine_bar_priors(
            np.zeros(2), np.ones(3), np.ones(2), np.ones(3)
        )
        self.assertEqual(mu.shape, (5,))
        self.assertEqual(sig.shape, (5, 5))

    def testForecast(self):
        # simulate data
        np.random.seed(123)
        beta0 = np.r_[0.5, -0.5]
        phi0 = np.r_[0.8, 0.15]
        N = 100
        k, p, = len(beta0), len(phi0)
        y, X = barma.simulate_bar(N, beta0, phi0)
        y_, X_ = barma.bar_design_matrix(y, X, p)
        mu_both, Sigma_both = np.zeros(4), np.eye(4)
        # Gibbs
        mcmc_draws = probit.gibbs_probit(y_, X_, mu_beta=mu_both, Sigma_beta=Sigma_both)
        names = [f"beta[{i+1}]" for i in range(k)] + [f"phi[{i+1}]" for i in range(p)]
        mcmc_summary = mcmc.mcmc_summ(mcmc_draws, true=np.r_[beta0, phi0], names=names)
        self.assertIsNotNone(mcmc_summary)
        # NUTS
        # stan_draws = None
        # fit = probit.stan_probit(y_, X_, mu_beta=mu_both, Sigma_beta=Sigma_both)
        # stan_draws = fit.extract("beta")["beta"]
        # VI
        vi_fit = probit.vi_probit(y_, X_, mu_beta=mu_both, Sigma_beta=Sigma_both)
        # forecasts
        fits = [vi_fit, mcmc_draws]
        fc = barma.binary_forecast(y, X, k, p, fits, steps=10)
        self.assertEqual(fc.shape, (3, 10))
        self.assertTrue(np.alltrue(fc != 0.0))
        self.assertTrue(max(fc) < 1.0)
        self.assertTrue(min(fc) > 0.0)
