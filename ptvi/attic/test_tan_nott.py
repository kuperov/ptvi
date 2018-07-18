# file encoding: utf-8

import unittest

from autograd import numpy as np

from ptvi.attic.tan_nott import TanNott


class TestTanNott(unittest.TestCase):

    def testAutogradBasics(self):
        rs = np.random.RandomState()
        y, b = TanNott.simulate(τ=5, λ=0.5, σ=0.1, φ=0.95, rs=rs)
        # check we can evaluate log_h, log_q, and Dθ_log_h
        m = TanNott(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        θ = rs.normal(size=m.d)
        import numbers
        self.assertIsInstance(m.log_q(θ, np.zeros(m.d), np.eye(m.d)),
                              numbers.Number)
        self.assertIsInstance(m.log_h(θ), numbers.Number)
        self.assertIsInstance(m.Dθ_log_h(θ), np.ndarray)
        ε = 1e-8
        rs = np.random.RandomState(seed=123)
        for _ in range(100):
            θ = rs.normal(size=m.d)
            ad_grad = m.Dθ_log_h(θ)
            for i in range(len(θ)):
                δ = np.array([ε if i == j else 0 for j in range(len(θ))])
                num_grad = (m.log_h(θ + δ) - m.log_h(θ)) / ε
                self.assertAlmostEqual(ad_grad[i] / num_grad, 1, 0)
        # check_grads(m.log_h, θ)

    def testEstimate(self):
        # smoke test for estimating a model
        rs = np.random.RandomState(seed=123)
        λ, σ, φ = 0., 0.5, 0.95
        y, b_true = TanNott.simulate(τ=50, λ=λ, σ=σ, φ=φ, rs=rs)
        m = TanNott(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        μ, T = m.algorithm1b(step_size=0.1, algo=1, draws=1, num_iters=2 ** 5,
                             quiet=True, rs=rs)
        self.assertTrue(np.all(np.isfinite(μ)))
        self.assertTrue(np.all(np.isfinite(T)))

    def testLogLikelihood(self):
        rs = np.random.RandomState(seed=123)
        λ, σ, φ = 0., 0.5, 0.95
        y, b_true = TanNott.simulate(τ=50, λ=λ, σ=σ, φ=φ, rs=rs)
        m = TanNott(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        # llik should be same up to a constant that doesn't depend on θ
        θ = np.ones(m.d)
        C = m.log_h(θ) - m.log_h2(θ)
        for _ in range(10):
            θ = rs.normal(size=m.d)
            ll2 = m.log_h2(θ) + C
            ll = m.log_h(θ)
            self.assertAlmostEqual(ll2, ll)

    def testHandWrittenGradient(self):
        rs = np.random.RandomState(seed=123)
        λ, σ, φ = 0., 0.5, 0.95
        y, b_true = TanNott.simulate(τ=50, λ=λ, σ=σ, φ=φ, rs=rs)
        m = TanNott(y, σ_α=1., σ_λ=1., σ_ψ=1.)
        for _ in range(10):
            θ = rs.normal(size=m.d)
            δll2 = m.Dθ_log_h2(θ)
            δll = m.Dθ_log_h(θ)
            diffs = δll2 - δll
            which = np.bitwise_not(np.isclose(diffs, 0))
            if np.any(which):
                print('mismatches = {}'.format(np.arange(m.d)[which]))
                print('autograd: {}'.format(δll[which]))
                print('handwritten: {}'.format(δll2[which]))
            self.assertTrue(np.allclose(diffs, np.zeros(m.d)))
