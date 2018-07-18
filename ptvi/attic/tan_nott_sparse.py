#!/usr/bin/env python3
# file encoding: utf-8

import numpy as np
from scipy.sparse.linalg import spsolve_triangular
from scipy.stats import multivariate_normal as mvn, norm
from scipy import sparse
import matplotlib.pyplot as plt
from autograd.misc.optimizers import adam


class TanNottSparse(object):

    def __init__(self, y, σ_α=1., σ_λ=1., σ_ψ=1.):
        self.y = y
        self.n = len(y)
        self.d = self.n + 3  # n local values, 3 globals
        self.σ_λ = σ_λ
        self.σ_ψ = σ_ψ
        self.σ_α = σ_α

    @staticmethod
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

    def sparsity_mask(self, lags=1):
        """Returns a sparse indicator matrix with 1s where (lower triangular) T
        has non-zero entries."""
        ndiags = lags + 1
        M = sparse.spdiags(np.ones([ndiags, self.d]),
                           diags=np.arange(0, -ndiags, -1), m=self.d,
                           n=self.d).tolil()
        # η has dimension 3
        M[-1, :] = np.ones(self.d)
        M[-2, :-1] = np.ones(self.d-1)
        M[-3, :-2] = np.ones(self.d-2)
        return M.tocsr()

    def Dθ_log_h(self, θ):
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
                + 0.5 * σ * (self.y[1:-1] ** 2) * np.exp(-λ - σ * b[1:-1]))
        δbn = (-(b[-1] - φ * b[-2]) - 0.5 * σ
               + 0.5 * σ * (self.y[-1] ** 2) * np.exp(-λ - σ * b[-1]))
        δα = (0.5 * np.sum((self.y ** 2) * b * np.exp(α - λ - σ * b))
              - 0.5 * σ * np.sum(b)) - α / self.σ_α
        δλ = -0.5 * self.n + 0.5 * np.sum(
            self.y ** 2 * np.exp(-λ - σ * b)) - λ / self.σ_λ
        δψ = ((φ * b[0] ** 2 - φ / (1 - φ ** 2) + np.sum(
            (b[1:] - φ * b[:-1]) * b[:-1])) *
              np.exp(ψ) / (np.exp(ψ) + 1) ** 2 - ψ / self.σ_ψ)
        return np.concatenate([[δb0], δb1_, [δbn, δα, δλ, δψ]])

    def initial_conditions(self, half_bw=1):
        """Estimate rough initial conditions μ0 with moving average of y^2.
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
        # transformations per paper
        α0 = np.log(σ0)
        ψ0 = np.log(φ0 / (1 - φ0))
        η0 = np.r_[α0, λ0, ψ0]
        μ0 = np.concatenate([b0, η0])
        return μ0

    def adadelta(self, μ0=None, algo=1, ω_adj=1., ρ=.5, ε=1e-8,
                 plot_callback=None, rs=None, quiet=False, num_iters=100_000):
        assert algo in [1, 2] and 0 < ρ < 1.
        rs = rs or np.random.RandomState()
        if isinstance(μ0, str) and μ0 == 'guess':
            μ0 = self.initial_conditions()
        elif μ0 is None or isinstance(μ0, str) and μ0 == 'zero':
            μ0 = np.r_[[0] * self.n, 0., 0., 0.]
        mask = self.sparsity_mask(lags=1)
        diag_ind = np.diag_indices_from(mask)
        μ, T = μ0, mask*0.
        Eg2_μ, ΕΔ2_μ = np.zeros(self.d), np.zeros(self.d)
        Εg2_T, ΕΔ2_T, Δ_T = T.copy(), T.copy(), T.copy()

        for t in range(num_iters):
            # adadelta as described on p.266
            t_diagonal = T[diag_ind]
            T[diag_ind] = np.exp(T[diag_ind])
            s = rs.normal(size=self.d)
            T_inv_t_s = spsolve_triangular(T.T, s, lower=False)
            θ = μ + T_inv_t_s

            if algo == 1:
                g_μ = self.Dθ_log_h(θ)
            else:
                g_μ = self.Dθ_log_h(θ) + T @ s

            Eg2_μ = ρ * Eg2_μ + (1 - ρ) * g_μ ** 2
            Δ_μ = np.sqrt((ΕΔ2_μ + ε)/(Eg2_μ + ε))*g_μ
            ΕΔ2_μ = ρ * ΕΔ2_μ + (1 - ρ) * Δ_μ ** 2
            μ = μ + ω_adj*Δ_μ

            Τ_inv_g_μ = spsolve_triangular(T, g_μ)
            # the following is inefficient (O(n^2)) because we have to
            # construct a whole dense matrix, then make it sparse
            if algo == 1:
                # g_T = mask.multiply((
                #         -np.outer(T_inv_t_s, Τ_inv_g_μ.T)
                #         - np.diag(np.array(1./T[diag_ind]).squeeze()))).tocsr()
                g_T = (sparse_outer_product(-T_inv_t_s, Τ_inv_g_μ.T, locals=2,
                                     globals=3) - \
                    sparse.spdiags(1. / T[diag_ind], 0, self.d, self.d)).tocsr()
            else:
                # g_T = mask.multiply((
                #         -np.outer(T_inv_t_s, Τ_inv_g_μ.T))).tocsr()
                g_T = sparse_outer_product(-T_inv_t_s, Τ_inv_g_μ.T, locals=2,
                                           globals=3).tocsr()

            # multiplying diagonal entries by T' gives us derivative wrt T diags
            g_T[diag_ind] = np.multiply(g_T[diag_ind], T[diag_ind])

            Εg2_T.data = ρ * Εg2_T.data + (1 - ρ) * g_T.data ** 2
            Δ_T.data = np.sqrt((ΕΔ2_T.data + ε) / (Εg2_T.data + ε)) * g_T.data
            ΕΔ2_T.data = ρ * ΕΔ2_T.data + (1 - ρ) * Δ_T.data ** 2
            T += ω_adj*Δ_T

            T[diag_ind] = t_diagonal

            if plot_callback:
                plot_callback(t, μ, T)

            if not quiet and not t & (t - 1):
                η = μ[-3:]
                α, λ, ψ = η
                σ = np.exp(α)
                φ = 1. / (1. + np.exp(-ψ))
                print(f'{t:4d}. σ = {σ:.2f}; φ = {φ:.2f}; λ = {λ:.2f}.')

        return μ, T

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
        try:
            U = T.copy()
            diag_ind = np.diag_indices(self.d)
            U[diag_ind] = np.exp(T[diag_ind])
            U_inv = np.linalg.pinv(U.todense())
            Λ = U_inv @ U_inv.T
            sds = np.sqrt(np.diag(Λ)[:-3])
            plt.fill_between(
                xs, np.maximum(vol - sds, 0), vol + sds, facecolor='blue',
                alpha=0.1, label='1 SD')
        except Exception as e:
            print('Exception: {}'.format(e))
        vol_true = np.exp(0.5 * (λ_true + σ_true * b_true))
        plt.plot(xs, vol_true, label='$vol_{true}$')
        plt.legend()

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

    def plot_initial_vs_true(self, λ, σ, φ, b):
        plt.figure()
        plt.subplot(311)
        μ0 = self.initial_conditions(half_bw=2)
        η0 = μ0[-3:]
        α0, λ0, ψ0 = η0
        σ0 = np.exp(α0)
        φ0 = 1. / (1. + np.exp(-ψ0))
        b0 = μ0[:-3]
        plt.plot(b, label='$b_{true}$')
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
        plt.plot(self.y, label='y')
        plt.legend()
        plt.title('Observed data')
        plt.suptitle('Data and initial conditions')
        plt.tight_layout()
        plt.show()


def sparse_outer_product(a, b, locals, globals):
    """Compute a@b.T retaining <locals> diagonals, and <globals> final rows and
    columns."""
    from scipy.sparse import spdiags
    n = len(a)
    assert len(b) == n and locals > 0 and globals >= 0
    ds = [np.concatenate([a[i:]*b[:n-i], [-999]*i]) for i in range(locals)]
    M = spdiags(ds, -np.arange(locals), n, n, format='lil')
    for r in range(n-globals, n):
        M[r, :r-locals+1] = a[r]*b[:r-locals+1]
    return M


def main():
    # run Tan-Nott SV model on simulated data
    np.seterr(all='raise')
    rs = np.random.RandomState(seed=123)
    np.set_printoptions(precision=2)
    λ, σ, φ = 0., 0.5, 0.95
    y, b = TanNottSparse.simulate(τ=500, λ=λ, σ=σ, φ=φ, rs=rs)
    m = TanNottSparse(y, σ_α=1., σ_λ=1.e-4, σ_ψ=1.)

    def plot_callback(t, μ, T):
        if t % 500:
            return
        m.plot_vol_vs_true(μ=μ, T=T, λ_true=λ, σ_true=σ, φ_true=φ, b_true=b)
        plt.draw()
        plt.pause(1e-10)

    μ, T = m.adadelta(μ0='zero', ρ=.5, ω_adj=100., algo=1, num_iters=100_000,
                      plot_callback=plot_callback, rs=rs)

    m.plot_initial_vs_true(λ=λ, σ=σ, φ=φ, b=b)
    plt.show()

    m.plot_vol_vs_true(μ=μ, T=T, λ_true=λ, σ_true=σ, φ_true=φ, b_true=b)
    plt.show()


if __name__ == '__main__':
    main()
