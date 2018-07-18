#!/usr/bin/env python3
# file encoding: utf-8

import matplotlib.pyplot as plt
from autograd import numpy as np, primitive, grad
from autograd.scipy.linalg import solve_triangular
from autograd.scipy.stats import multivariate_normal as mvn, norm
import pandas as pd


class TanNott(object):

    def __init__(self, y, σ_α, σ_ψ, σ_λ):
        self.y = y
        self.n = len(y)
        self.d = self.n + 3  # n local values, 3 globals
        self.Dθ_log_h = grad(self.log_h)
        self.σ_λ = σ_λ
        self.σ_ψ = σ_ψ
        self.σ_α = σ_α

    @staticmethod
    @primitive
    def simulate(τ, λ=0., σ=0.5, φ=0.95, rs=None):
        rs = rs or np.random.RandomState()
        assert τ >= 1 and σ > 0 and 0 < φ < 1
        b = np.zeros(τ)
        b[0] = rs.normal(0, 1 / np.sqrt(1 - φ ** 2))
        for t in range(1, τ):
            b[t] = rs.normal(φ * b[t - 1], 1, 1)
        y = rs.normal(loc=0, scale=np.exp(0.5 * (λ + σ * b)))
        return y, b

    def log_q(self, θ, μ, L):
        assert len(θ) == self.d
        return mvn.logpdf(θ, μ, L @ L.T)

    def log_h(self, θ):
        η, b = θ[-3:], θ[:-3]
        α, λ, ψ = η
        σ = np.exp(α)
        φ = 1. / (1. + np.exp(-ψ))
        log_lik = (
                np.sum(norm.logpdf(self.y, 0, np.exp(.5 * (λ + σ * b))))
                + norm.logpdf(b[0], 0, 1 / np.sqrt(1 - φ ** 2))
                + np.sum(norm.logpdf(b[1:], φ * b[:-1], 1))
        )
        log_prior = (
                norm.logpdf(ψ, 0., self.σ_ψ) +
                norm.logpdf(α, 0., self.σ_α) +
                norm.logpdf(λ, 0., self.σ_λ)
        )
        return log_lik + log_prior

    def log_h2(self, θ):
        """A manual implementation of ln h(θ), see p.274 of Tan & Nott."""
        η, b = θ[-3:], θ[:-3]
        α, λ, ψ = η
        σ = np.exp(α)
        φ = 1. / (1. + np.exp(-ψ))
        return (
                - 0.5 * self.n * λ
                - 0.5 * σ * np.sum(b)
                - 0.5 * np.sum(np.exp(-λ - σ * b) * np.square(self.y))
                - 0.5 * np.sum(np.square(b[1:] - φ * b[:-1]))
                + 0.5 * np.log(1 - φ ** 2)
                - 0.5 * (1 - φ ** 2) * b[0] ** 2
                - 0.5 * α ** 2 / self.σ_α ** 2
                - 0.5 * λ ** 2 / self.σ_λ ** 2
                - 0.5 * ψ ** 2 / self.σ_ψ ** 2
        )

    def Dθ_log_h2(self, θ):
        η = θ[-3:]
        α, λ, ψ = η
        b = θ[:-3]
        # transformations as detailed in paper
        σ = np.exp(α)
        φ = 1. / (1. + np.exp(-ψ))
        # numbering is 0-based, unlike in the paper
        δb0 = (-(1 - φ ** 2) * b[0] + φ * (b[1] - φ * b[0])
               - 0.5 * σ + 0.5 * σ * (self.y[0] ** 2) * np.exp(-λ - σ * b[0]))
        δb1_ = (φ * (b[2:] - φ * b[1:-1]) - (b[1:-1] - φ * b[:-2]) - 0.5 * σ
                + 0.5 * σ * self.y[1:-1] ** 2 * np.exp(-λ - σ * b[1:-1]))
        δbn = (-(b[-1] - φ * b[-2]) - 0.5 * σ
               + 0.5 * σ * self.y[-1] ** 2 * np.exp(-λ - σ * b[-1]))
        δα = (0.5 * np.sum(self.y ** 2 * b * np.exp(α - λ - σ * b))
              - 0.5 * σ * np.sum(b)) - α / self.σ_α
        δλ = -0.5 * self.n + 0.5 * np.sum(
            self.y ** 2 * np.exp(-λ - σ * b)) - λ / self.σ_λ
        δψ = ((φ * b[0] ** 2 - φ / (1 - φ ** 2) + np.sum(
            (b[1:] - φ * b[:-1]) * b[:-1])) *
              np.exp(ψ) / (np.exp(ψ) + 1) ** 2 - ψ / self.σ_ψ)
        return np.concatenate([[δb0], δb1_, [δbn, δα, δλ, δψ]])

    @primitive
    def initial_conditions(self, half_bw=2):
        """Estimate rough initial conditions with moving average of y^2.
        """
        y2 = np.square(self.y)
        y2_psum = np.cumsum(y2)
        w = half_bw * 2
        # rolling estimate of variance
        σ2_hat = (y2_psum[w:] - y2_psum[:-w]) / w  # divide by w since mean 0
        # pad ends to match number of latent variables, inc starting value
        σ2_hat_full = np.concatenate([
            [σ2_hat[0]] * (half_bw + 1), σ2_hat, [σ2_hat[-1]] * half_bw])
        ln_σ_hat = np.log(σ2_hat_full) / 2
        λ0 = np.median(ln_σ_hat)
        σb_hat = ln_σ_hat - λ0
        # estimate φ by regressing σb_t = φσb_{t-1} + ε_t
        _X, _Y = σb_hat[:-1], σb_hat[1:]
        φ0 = (_X * _Y).sum() / np.square(_X).sum()
        # use marginal variance of b to estimate σ
        σ0 = np.std(σb_hat) * np.sqrt(1 - φ0 ** 2)
        b0 = σb_hat[:-1] / σ0
        # ξ0 = np.r_[σ0, λ0, φ0]
        # transformations per paper
        α0 = np.log(σ0)
        ψ0 = np.log(φ0 / (1 - φ0))
        η0 = np.r_[α0, λ0, ψ0]
        θ0 = np.concatenate([b0, η0])
        return θ0

    @primitive
    def algorithm1(self, rs=None):
        rs = rs or np.random.RandomState()
        μ = np.zeros(self.d)
        T = np.eye(self.d)
        ρ = 0.01
        for t in range(1000):
            s = rs.normal(size=self.d)
            T_inv = np.linalg.inv(T)
            if not np.all(np.isfinite(T_inv)):
                print('Warning, T_inv not finite')
            θ = μ + T_inv.T @ s
            g_μ = self.Dθ_log_h(θ)
            μ += ρ * g_μ
            g_T = -np.outer(T_inv.T @ s, (T_inv @ g_μ).T) - np.diag(
                1 / np.diag(T))
            T += ρ * g_T
            print(f'{t: 4d}. |μ| = {np.linalg.norm(μ):5.2f}\t'
                  f'|T| = {np.linalg.norm(T):5.2f}\t'
                  f'|T^-1| = {np.linalg.norm(T_inv):5.2f}\t'
                  f'|g_μ| = {np.linalg.norm(g_μ):5.2f}\t'
                  f'|g_Tprime| = {np.linalg.norm(g_T):5.2f}\t')
            if not np.all(np.isfinite(T)):
                print('Warning: T not finite')
                break
            if not np.all(np.isfinite(μ)):
                print('Warning: μ not finite')
                break
            if not np.all(np.isfinite(g_μ)):
                print('Warning: g_μ not finite')
                break
            T = self.ensure_sparse(T)
        return μ, T

    @primitive
    def algorithm1a(self, rs=None, **kwargs):
        from autograd.misc.optimizers import adam
        rs = rs or np.random.RandomState()
        x0 = np.concatenate([np.zeros(self.d), np.eye(self.d).reshape(-1)])

        def unstack(x):
            μ = x[:self.d]
            T = np.reshape(x[self.d:], [self.d, self.d])
            return μ, T

        def stoch_grad(x, t):
            s = rs.normal(size=self.d)
            μ, T = unstack(x)
            T_inv = np.linalg.pinv(T)
            θ = μ + T_inv.T @ s
            g_μ = self.Dθ_log_h(θ)
            g_T = -np.outer(T_inv.T @ s, (T_inv @ g_μ).T) - np.diag(
                1 / np.diag(T))
            g_T = self.ensure_sparse(g_T)
            return np.concatenate([-g_μ, -g_T.reshape(-1)])

        def callback(x, t, g):
            if not t & (t - 1):
                μ, T = unstack(x)
                g_μ, g_T = unstack(g)
                print(f'{t: 4d}. '
                      f'|μ| = {np.linalg.norm(μ):5.2f}\t'
                      f'|T| = {np.linalg.norm(T):5.2f}\t'
                      f'|g_μ| = {np.linalg.norm(g_μ):5.2f}\t'
                      f'|g_T| = {np.linalg.norm(g_T):5.2f}\t')
                print(f'{t: 4d}. {μ.round(2)}')

        x_hat = adam(stoch_grad, x0, callback, **kwargs)
        return unstack(x_hat)

    @primitive
    def algorithm1b(self, μ0=None, algo=1, draws=5, plot_callback=None, rs=None,
                    quiet=False, **kwargs):
        assert draws > 0 and algo in [1, 2]
        from autograd.misc.optimizers import adam
        rs = rs or np.random.RandomState()
        if isinstance(μ0, str) and μ0 == 'guess':
            μ0 = self.initial_conditions()
        elif μ0 is None or isinstance(μ0, str) and μ0 == 'zero':
            μ0 = np.r_[[0] * self.n, 0., 0., 0.]
        x0 = {'μ': μ0, 'T': np.zeros([self.d, self.d])}
        diag_ind = np.diag_indices(self.d)

        @primitive
        def stoch_grad(x, t):
            # T is low-triangular with logged diagonal entries
            μ, T = x['μ'], x['T']
            # temporarily exponentiate T's diagonal entries
            t_diagonal = T[diag_ind]
            T[diag_ind] = np.exp(T[diag_ind])
            g_μ_tot = np.zeros(self.d)
            g_T_tot = np.zeros([self.d, self.d])
            for _ in range(draws):
                s = rs.normal(size=self.d)
                try:
                    with np.errstate(divide='raise'):
                        T_inv_t_s = solve_triangular(T, s, trans='T',
                                                     lower=True)
                        θ = μ + T_inv_t_s
                        if algo == 1:
                            g_μ = self.Dθ_log_h2(θ)
                            U_inv_g_μ = solve_triangular(T, g_μ)
                            g_T = (-np.outer(T_inv_t_s, U_inv_g_μ.T)
                                   - np.diag(1 / np.diag(T)))
                        elif algo == 2:
                            g_μ = self.Dθ_log_h(θ) + T @ s
                            T_inv_g_μ = solve_triangular(T, g_μ)
                            g_T = -np.outer(T_inv_t_s, T_inv_g_μ.T)
                        # multiplying diagonal entries by U gives us derivative wrt T diags
                        g_T[diag_ind] *= T[diag_ind]
                        g_μ_tot += g_μ
                        g_T_tot += g_T
                except (FloatingPointError, ValueError) as e:
                    import traceback, sys
                    print('Freakout on iteration {}'.format(t))
                    traceback.print_exc()
                    sys.exit(-1)
            T[diag_ind] = t_diagonal
            return {'μ': -g_μ_tot / draws,
                    'T': -self.ensure_sparse(g_T_tot) / draws}

        def callback(x, t, g):
            if plot_callback:
                plot_callback(t, x['μ'], x['T'])

            if not quiet and not t & (t - 1):
                μ, T = x['μ'], x['T']
                g_μ, g_T = g['μ'], g['T']
                η = μ[-3:]
                α, λ, ψ = η
                σ = np.exp(α)
                φ = 1. / (1. + np.exp(-ψ))
                print(f'{t: 4d}. '
                      f'σ = {σ:.2f};  '
                      f'φ = {φ:.2f};  '
                      f'λ = {λ:.2f};  '
                )

        x_hat = adam(stoch_grad, x0, callback, **kwargs)
        return x_hat['μ'], x_hat['T']

    # Algorithm 2: modified doubly stochastic variational inference
    @primitive
    def algorithm2(self, rs=None):
        rs = rs or np.random.RandomState()
        μ = np.zeros(self.d)
        T = np.eye(self.d)
        Tprime = np.zeros_like(T)
        diag_ind = np.diag_indices_from(Tprime)
        ρ = 0.01
        for t in range(100):
            s = rs.normal(size=self.d)
            T_inv = np.linalg.inv(T)
            if not np.all(np.isfinite(T_inv)):
                print('Warning, T_inv not finite')
            θ = μ + T_inv.T @ s
            g_μ = self.Dθ_log_h(θ) + T @ s
            μ += ρ * g_μ
            g_Tprime = -np.outer(T_inv.T @ s, (T_inv @ g_μ).T)
            g_Tprime[diag_ind] = np.multiply(np.diag(g_Tprime), np.diag(T))
            Tprime += ρ * g_Tprime
            T = Tprime
            T[diag_ind] = np.exp(np.diag(Tprime))
            print(f'{t: 4d}. |μ| = {np.linalg.norm(μ):5.2f}\t'
                  f'|T| = {np.linalg.norm(T):5.2f}\t'
                  f'|T^-1| = {np.linalg.norm(T_inv):5.2f}\t'
                  f'|Tprime| = {np.linalg.norm(Tprime):5.2f}\t'
                  f'|g_μ| = {np.linalg.norm(g_μ):5.2f}\t'
                  f'|g_Tprime| = {np.linalg.norm(g_Tprime):5.2f}\t')
            if not np.all(np.isfinite(T)):
                print('Warning: T not finite')
                break
            if not np.all(np.isfinite(μ)):
                print('Warning: μ not finite')
                break
            if not np.all(np.isfinite(g_Tprime)):
                print('Warning: g_Tprime not finite')
                break
            if not np.all(np.isfinite(g_μ)):
                print('Warning: g_μ not finite')
                break
            T = self.ensure_sparse(T)
            Tprime = self.ensure_sparse(Tprime)
        return μ, T

    @primitive
    def ensure_sparse(self, A):
        B = np.zeros_like(A)
        n = A.shape[0]
        for i in range(n - 3):
            B[i, i] = A[i, i]
            if i >= 1:
                B[i, i - 1] = A[i, i - 1]
        B[-3:n, :] = A[-3:n, :]
        return B

    @primitive
    def plot_vol_vs_true(self, μ, T, λ_true, σ_true, φ_true, b_true):
        # plotting in terms of vol
        plt.cla()
        η = μ[-3:]
        α, λ, ψ = η
        σ = np.exp(α)
        φ = 1. / (1. + np.exp(-ψ))
        b = μ[:-3]
        vol = np.exp(0.5 * (λ + σ * b))
        xs = np.arange(self.n)
        plt.plot(xs, vol, label=r'$vol_{VI}$')
        # recall: T is cholesky factor of precision matrix
        U = self.ensure_sparse(np.copy(T))
        diag_ind = np.diag_indices(self.d)
        U[diag_ind] = np.exp(T[diag_ind])
        U_inv = np.linalg.pinv(U)
        Λ = U_inv @ U_inv.T
        sds = np.sqrt(np.diag(Λ)[:-3])
        plt.fill_between(
            xs, np.maximum(vol - sds, 0), vol + sds, facecolor='blue',
            alpha=0.1, label='1 SD')
        vol_true = np.exp(0.5 * (λ_true + σ_true * b_true))
        plt.plot(xs, vol_true, label='$vol_{true}$')
        plt.legend()

    @primitive
    def plot_vol(self, μ, T):
        # plotting in terms of vol
        plt.cla()
        η = μ[-3:]
        α, λ, ψ = η
        σ = np.exp(α)
        φ = 1. / (1. + np.exp(-ψ))
        b = μ[:-3]
        vol = np.exp(0.5 * (λ + σ * b))
        xs = np.arange(self.n)
        plt.plot(xs, vol, label=r'$vol_{VI}$')
        # recall: T is cholesky factor of precision matrix
        U = self.ensure_sparse(np.copy(T))
        diag_ind = np.diag_indices(self.d)
        U[diag_ind] = np.exp(T[diag_ind])
        U_inv = np.linalg.pinv(U)
        Λ = U_inv @ U_inv.T
        sds = np.sqrt(np.diag(Λ)[:-3])
        plt.fill_between(
            xs, np.maximum(vol - sds, 0), vol + sds, facecolor='blue',
            alpha=0.1, label='1 SD')
        plt.legend()

    @primitive
    def plot_log_vol_vs_true(self, μ, T):
        # plotting in terms of b
        plt.cla()
        b = μ[:-3]
        plt.plot(b, label='$b$')
        if getattr(self, 'b_true', None) is not None:
            plt.plot(self.b_true, label='$b^T$')
        plt.plot(self.y, label='y')
        plt.legend()
        plt.draw()

    @primitive
    def plot_initial_vs_true(self, λ, σ, φ, b):
        plt.figure()
        plt.subplot(311)
        μ0 = self.initial_conditions(half_bw=2)
        η0 = μ0[-3:]
        α0, λ0, ψ0 = η0
        σ0 = np.exp(α0)
        φ0 = 1. / (1. + np.exp(-ψ0))
        b0 = μ0[:-3]
        plt.plot(b_true, label='$b_{true}$')
        plt.plot(b0, label='$b_{init}$')
        plt.legend()
        plt.title('Log volatility $b_t$')
        plt.subplot(312)
        vol_true = np.exp(0.5 * (λ + σ * b))
        plt.plot(vol_true, label='$vol_{true}$')
        plt.plot(np.exp(0.5 * (λ0 + σ0 * b0)),
                 label='$vol_{init}$')
        plt.legend()
        plt.title('Volatility $(λ + σb_t)/2$')
        plt.subplot(313)
        plt.plot(y, label='y')
        plt.legend()
        plt.title('Observed data')
        plt.suptitle('Data and initial conditions')
        plt.tight_layout()
        plt.show()


def demo():
    rs = np.random.RandomState(seed=123)
    np.set_printoptions(precision=2)
    λ, σ, φ = 0., 0.5, 0.95
    y, b = TanNott.simulate(τ=200, λ=λ, σ=σ, φ=φ, rs=rs)
    m = TanNott(y, σ_α=1., σ_λ=1.e-4, σ_ψ=1.)

    def plot_callback(t, μ, T):
        if t % 100:
            return
        m.plot_vol_vs_true(μ=μ, T=T, λ_true=λ, σ_true=σ, φ_true=φ, b_true=b)
        plt.draw()
        plt.pause(1e-10)

    μ0 = np.r_[[0] * m.n, 0., 0., 0.]

    μ, T = m.algorithm1b(step_size=0.05, algo=1, draws=1, num_iters=2 ** 16,
                         plot_callback=plot_callback, rs=rs)

    m.plot_initial_vs_true(λ=λ, σ=σ, φ=φ, b=b)
    plt.show()

    m.plot_vol_vs_true(μ=μ, T=T, λ_true=λ, σ_true=σ, φ_true=φ, b_true=b)
    plt.show()


def fit_usd_gbp():
    import os
    rs = np.random.RandomState(seed=123)
    np.set_printoptions(precision=2)
    infile = os.path.join(os.path.dirname(__file__), 'data', 'DEXUSUK.csv')
    dat = pd.read_csv(infile, na_values=['.'])
    dat['DEXUSUK'] = dat['DEXUSUK'].astype(np.float64)
    y = np.array(dat[dat['DEXUSUK'].notna()]['DEXUSUK'])
    m = TanNott(y, σ_α=1., σ_λ=1.e-4, σ_ψ=1.)
    def plot_callback(t, μ, T):
        if t % 100:
            return
        m.plot_vol(μ=μ, T=T)
        plt.draw()
        plt.pause(1e-10)

    μ, T = m.algorithm1b(step_size=0.05, algo=1, draws=1,
                         plot_callback=plot_callback, num_iters=2 ** 10, rs=rs)


if __name__ == '__main__':
    # run Tan-Nott SV model on simulated data
    #demo()
    fit_usd_gbp()
