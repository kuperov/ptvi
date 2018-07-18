import unittest

import numpy as np

from ptvi.attic.tan_nott_sparse import TanNottSparse


class TestTanNottSparse(unittest.TestCase):

    def testBasics(self):
        rs = np.random.RandomState()
        y, b = TanNottSparse.simulate(τ=5, λ=0.5, σ=0.1, φ=0.95, rs=rs)
        self.assertEqual(len(y), 5)
        self.assertEqual(len(b), 5)
        # check we can evaluate log_h, log_q, and Dθ_log_h
        m = TanNottSparse(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        θ = rs.normal(size=m.d)
        import numbers
        self.assertIsInstance(m.log_q(θ, np.zeros(m.d), np.eye(m.d)),
                              numbers.Number)
        self.assertIsInstance(m.log_h(θ), numbers.Number)
        self.assertIsInstance(m.Dθ_log_h(θ), np.ndarray)

    def testGradients(self):
        rs = np.random.RandomState(seed=123)
        y, _ = TanNottSparse.simulate(τ=5, λ=0.5, σ=0.1, φ=0.95, rs=rs)
        m = TanNottSparse(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        ε = 1e-4
        with np.errstate(divide='raise'):
            for j in range(100):
                θ = rs.normal(size=m.d)
                ad_grad = m.Dθ_log_h(θ)
                for i in range(len(θ)):
                    δ = np.array([ε if i == j else 0 for j in range(m.d)])
                    num_grad = (m.log_h(θ + δ) - m.log_h(θ)) / ε
                    self.assertAlmostEqual(
                        ad_grad[i] / num_grad, 1., 0,
                        msg='{}: Gradient {}: {} != {} @ {}'.format(
                             j, i, ad_grad[i], num_grad, θ[i]))

    def testSparsityMask(self):
        rs = np.random.RandomState(seed=123)
        y, _ = TanNottSparse.simulate(τ=10, λ=0., σ=0.5, φ=0.95, rs=rs)
        m = TanNottSparse(y)
        M1 = m.sparsity_mask(lags=1)
        self.assertEqual(M1.shape, (13, 13))

    def testSparseOuterProduct(self):
        from ptvi.attic.tan_nott_sparse import (
            sparse_outer_product, TanNottSparse)
        rs = np.random.RandomState(seed=123)
        y, _ = TanNottSparse.simulate(τ=4, λ=0., σ=0.5, φ=0.95, rs=rs)
        m = TanNottSparse(y)
        mask = m.sparsity_mask(lags=1)
        #
        n = mask.shape[0]
        a, b = np.arange(n)+1, np.arange(n)+5
        expected = mask.multiply(np.outer(a, b))
        actual = sparse_outer_product(a, b, locals=2, globals=3)
        self.assertFalse(np.any((expected != actual).todense()))

    def testEstimate(self):
        # smoke test for estimating a model
        rs = np.random.RandomState(seed=123)
        y, b_true = TanNottSparse.simulate(τ=50, λ=0., σ=0.5, φ=0.95, rs=rs)
        m = TanNottSparse(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        μ, T = m.adadelta(ω_adj=10., algo=1, num_iters=2 ** 5,
                          quiet=True, rs=rs)
        self.assertTrue(np.all(np.isfinite(μ)))
        self.assertTrue(np.all(np.isfinite(T.data)))

    def testLogLikelihood(self):
        rs = np.random.RandomState(seed=123)
        λ, σ, φ = 0., 0.5, 0.95
        y, b_true = TanNottSparse.simulate(τ=50, λ=λ, σ=σ, φ=φ, rs=rs)
        m = TanNottSparse(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        # llik should be same up to a constant that doesn't depend on θ
        θ = np.ones(m.d)
        C = m.log_h(θ) - m.log_h2(θ)
        for _ in range(10):
            θ = rs.normal(size=m.d)
            ll2 = m.log_h2(θ) + C
            ll = m.log_h(θ)
            self.assertAlmostEqual(ll2, ll)
