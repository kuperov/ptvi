import unittest

from scipy import stats
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

from ptvi.cavi.inference import *

# split out a bunch of functions so we can use them to test in the console


def _gen_reg():
    N = 100
    X = sm.add_constant(np.random.normal(size=[N, 3]))
    β0 = np.r_[0.0, 2.0, -1.0, 10]
    noise = stats.norm(scale=np.sqrt(1 / 2.5))
    y = X @ β0 + noise.rvs(size=[N])
    fit = sm.OLS(y, X).fit()
    return y, X, fit


def _gen_ar2(do_fit=True):
    phi = np.r_[0.9, 0.05]
    ar2 = ArmaProcess(ar=np.r_[1, -phi], nobs=500)
    y = ar2.generate_sample()
    assert ar2.isstationary and ar2.isinvertible
    fit = sm.tsa.ARMA(y, (2, 0)).fit() if do_fit else None
    return y, fit


class TestReg(unittest.TestCase):
    """Test MCMC routines by comparing them to the frequentist fits obtained by StatsModels."""

    def setUp(self):
        np.random.seed(123)
        y, X, fit = _gen_reg()
        self.X, self.y = X, y
        self.β_ols = fit.params
        self.β_cov = fit.cov_params()
        self.e_mse = fit.mse_resid
        self.common_args = dict(draws=1_000, warmup=1_000, chains=4, y=self.y, X=self.X, a_0=2, b_0=0.5, c_0=2, d_0=0.5)

    @unittest.skip("Stan borks the whole damn thread in PyCharm")
    def testStanReg(self):
        t = time.perf_counter()
        βs, τs, αs = reg_mcmc(**self.common_args, method='NUTS', verbose=False)
        self.assertEqual(βs.shape, (4000, 4))
        self.assertEqual(τs.shape, (4000,))
        self.assertEqual(αs.shape, (4000,))
        elapsed = time.perf_counter() - t
        print(f'NUTS took {elapsed:.6f}s')

    def testGibbsReg(self):
        t = time.perf_counter()
        βs, τs, αs = reg_mcmc(**self.common_args, method='Gibbs', verbose=False)
        self.assertEqual(βs.shape, (4000, 4))
        self.assertEqual(τs.shape, (4000,))
        self.assertEqual(αs.shape, (4000,))
        elapsed = time.perf_counter() - t
        print(f'Gibbs took {elapsed:.6f}s')
        self.assertLess(elapsed, 5., "Sampler took >5s, that's too long.")
        β_means = np.mean(βs, axis=0)
        self.assertLess(max(abs(β_means - self.β_ols)), 1., "Coeffs differ from OLS by too much")

    def testVB(self):
        np.random.seed(123)
        fit = vb_reg(self.y, self.X, a_0=2, b_0=.5, c_0=2, d_0=.5, verbose=False)
        # check roughly close
        E_τ = fit["a_N"] / fit["b_N"]
        assert abs(E_τ - 2.5) < 1
        assert np.allclose(self.β_ols, fit["w_N"], atol=0.5)


class TestARp(unittest.TestCase):
    """Test AR(p) routines by comparing to StatsModels."""

    def setUp(self):
        np.random.seed(123)
        y, fit = _gen_ar2()
        self.y = y
        self.β_mle = fit.params
        self.common_params = dict(y=self.y, p=2, a_0=2, b_0=0.5, c_0=2, d_0=0.5)

    @unittest.skip('Stan is behaving badly in PyCharm')
    def testStanARp(self):
        βs, τs, αs = arp_mcmc(method="NUTS", draws=1_000, warmup=1_000, **self.common_params)
        β_means = np.mean(βs, axis=0)
        self.assertLess(max(abs(β_means - self.β_mle)), 1., "Coeffs differ from MLE by too much")

    def testGibbsARp(self):
        np.random.seed(123)
        βs, τs, αs = arp_mcmc(method="Gibbs", draws=1_000, warmup=1_000, **self.common_params)
        β_means = np.mean(βs, axis=0)
        self.assertLess(max(abs(β_means - self.β_mle)), 1., "Coeffs differ from MLE by too much")

    def testGibbsForecast(self):
        np.random.seed(123)
        fc = arp_mcmc_forecast(steps=3, draws=10_000, warmup=1_000, method="Gibbs", **self.common_params)
        self.assertEqual(type(fc), dict)
        self.assertEqual(len(fc), 3)
        N = len(self.y)
        # variability of forecasts should be increasing
        sds = [np.std(fc[s]) for s in range(N,N+3)]
        self.assertTrue(sds[0] < sds[1] < sds[2])

    def testVBARp(self):
        np.random.seed(123)
        post = arp_vb(**self.common_params)
        β_means = post['w_N']
        self.assertLess(max(abs(β_means - self.β_mle)), 1., "Coeffs differ from MLE by too much")

    def testVBForecast(self):
        np.random.seed(123)
        fc = arp_vb_forecast(steps=3, draws=10_000, **self.common_params)
        self.assertEqual(type(fc), dict)
        self.assertEqual(len(fc), 3)
        N = len(self.y)
        # variability of forecasts should be increasing
        sds = [np.std(fc[s]) for s in range(N,N+3)]
        self.assertTrue(sds[0] < sds[1] < sds[2])
