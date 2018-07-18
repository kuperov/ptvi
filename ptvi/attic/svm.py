# file encoding: utf-8

from autograd import numpy as np, primitive, jacobian
from autograd.scipy import stats
import matplotlib.pyplot as plt
from ptvi.util import (vec_to_trilpd, tril_to_vec, vec_to_tril,
                       vec_to_pd, pd_to_vec)
from ptvi.attic import invgamma, invwishart


class SVM(object):
    """Unobserved components model with stochastic volatility in mean.

    Chan, Joshua CC. "The stochastic volatility in mean model with time-varying
    parameters: An application to inflation modeling." Journal of Business &
    Economic Statistics 35.1 (2017): 17-28.

    The model is as follows, where B denotes the backshift operator, εₜ and νₜ
    are iid standard normal, ωₜ is iid multivariate standard normal, and Ω^½
    denotes the cholesky factor of Ω.

      ..math::
        yₜ = τₜ + αₜ exp(hₜ) + exp(½hₜ)εₜ
        hₜ = μ + φ(Βhₜ - μ) + βΒyₜ+ σνₜ
        γₜ = (αₜ, τₜ)ᵀ
        γₜ = Βγₜ + Ω^½ᵀ ωₜ
        τ₁ ~ N(0, Vτ)
        h₁ ~ N(μ, σ²/(1 - φ²))
    """
    def __init__(self, μ0, Vμ, φ0, Vφ, νσ2, Sσ2, νΩ, SΩ, γ0, Ω0, β0, Vβ):
        self.μ0, self.Vμ = μ0, Vμ
        self.φ0, self.Vφ = φ0, Vφ
        self.νσ2, self.Sσ2 = νσ2, Sσ2
        self.νΩ, self.SΩ = νΩ, SΩ
        self.β0, self.Vβ = β0, Vβ
        self.γ0, self.Ω0 = γ0, Ω0
        self.log_det_J_Tinv = self.make_log_det_J_Tinv()

    @primitive
    def simulate_prior(self, rs=None):
        """Simulate one draw of global parameters from prior p(μ, φ, β, σ2, Ω).

        Returns a dict.
        """
        rs = rs or np.random.RandomState()
        μ = rs.normal(self.μ0, np.sqrt(self.Vμ))
        φ = None
        while φ is None or not (-1. <= φ <= 1.):
            φ = rs.normal(self.φ0, np.sqrt(self.Vφ))
        β = rs.normal(self.β0, np.sqrt(self.Vβ))
        σ2 = invgamma.rvs(self.νσ2, self.Sσ2, random_state=rs)
        Ω = invwishart.rvs(self.νΩ, self.SΩ, random_state=rs)
        return dict(μ=μ, φ=φ, β=β, σ2=σ2, Ω=Ω)

    @primitive
    def simulate(self, T, μ, φ, β, σ2, Ω, rs=None):
        """Simulate T observations from SVM model p(y, h, γ | μ, φ, β, σ2, Ω).

        Returns:
            tuple of y, h, γ, each with length T
        """
        rs = rs or np.random.RandomState()
        # time-varying parameters evolve as a random walk
        εγ = np.concatenate([
            [rs.multivariate_normal(self.γ0, self.Ω0)],
            rs.multivariate_normal(np.zeros(2), Ω, T - 1)])
        γ = np.cumsum(εγ, axis=0)
        α, τ = γ[:, 0], γ[:, 1]
        # jointly simulate volatility and observed data
        h, y = np.empty(T), np.empty(T)
        for t in range(T):
            if t == 0:
                h[0] = rs.normal(μ, np.sqrt(σ2 / (1 - φ ** 2)))
            else:
                h[t] = rs.normal(μ + φ*(h[t-1] - μ) + β*y[t-1], np.sqrt(σ2))
            if not np.isfinite(h[t]):
                raise Exception('h[{}] not finite.'.format(t))
            y[t] = rs.normal(τ[t] + α[t] * np.exp(h[t]), np.exp(0.5*h[t]))
            if not np.isfinite(y[t]):
                raise Exception('y[{}] not finite.'.format(t))
        return SVMData(y=y, h=h, γ=γ)

    def log_p(self, y, γ, h, μ, φ, β, σ2, Ω):
        """Compute log joint density, log p(y, h, γ, μ, φ, β, σ2, Ω)."""
        α, τ = γ[:, 0], γ[:, 1]
        llik = (
            _norm_ll(τ + α * np.exp(h), y, np.exp(h))
            + _norm_ll(h[0], μ, σ2 / (1 - φ ** 2))
            + _norm_ll(h[1:], μ + φ * (h[:-1] - μ) + β * y[:-1], σ2)
            + _mvn_ll(γ[1:, :], γ[:-1, :], Ω)
        )
        lprior = (
                _norm_ll(μ, self.μ0, self.Vμ)
                + _norm_ll(h, 0, 1.)  # are priors for h given in the paper??
                + _mvn_ll(γ[0], self.γ0, self.Ω0)
                + _trunc_norm_ll(φ, self.φ0, self.Vφ, -1, 1)
                + invgamma.logpdf(σ2, self.νσ2, self.Sσ2)
                + invwishart.logpdf(Ω, self.νΩ, self.SΩ)
        )
        return llik + lprior

    def pack_ζ(self, γ, h, μ, η, β, ln_σ2, ω):
        """Pack parameter by turning into a single vector in unrestricted
        coordinate space."""
        return np.concatenate([γ.flatten(), h, [μ, η, β, ln_σ2], ω])

    def unpack_ζ(self, ζ):
        # local variables
        n = (len(ζ) - 7) // 3
        γ, h = ζ[:2 * n].reshape([n, 2]), ζ[2 * n:3 * n]
        # global variables
        μ, η, β, ln_σ2 = ζ[3 * n], ζ[3 * n + 1], ζ[3 * n + 2], ζ[3 * n + 3]
        ω = ζ[3 * n + 4:3 * n + 7]
        return dict(γ=γ, h=h, μ=μ, η=η, β=β, ln_σ2=ln_σ2, ω=ω)

    def T(self, γ, h, μ, φ, β, σ2, Ω):
        """ζ = T(θ): Transform from user coordinates to optimization coords."""
        η, ln_σ2, ω = np.arctanh(φ), np.log(σ2), pd_to_vec(Ω)
        return dict(γ=γ, h=h, μ=μ, η=η, β=β, ln_σ2=ln_σ2, ω=ω)

    def Tinv(self, γ, h, μ, η, β, ln_σ2, ω):
        """θ = Tinv(ζ): Inverse transform from optimization to user coords."""
        φ = np.tanh(η)         # guarantee φ ∈ (-1, 1)
        σ2 = np.exp(ln_σ2)     # guarantee σ2 > 0
        Ω = vec_to_pd(ω)       # guarantee Ω psd
        return dict(γ=γ, h=h, μ=μ, φ=φ, β=β, σ2=σ2, Ω=Ω)

    def make_log_det_J_Tinv(self):
        def Tinv_star(ζ_star):
            η, ln_σ2, ω = ζ_star[0], ζ_star[1], ζ_star[2:]
            φ = np.tanh(η)      # guarantee φ ∈ (-1, 1)
            σ2 = np.exp(ln_σ2)  # guarantee σ2 > 0
            Ω = vec_to_pd(ω)    # guarantee Ω psd
            return np.concatenate([np.array([φ, σ2]), tril_to_vec(Ω)])
        jac = jacobian(Tinv_star)
        def log_det_J_Tinv(η, ln_σ2, ω):
            ζ_star = np.concatenate([np.array([η, ln_σ2]), ω])
            J = jac(ζ_star)
            return np.abs(np.linalg.slogdet(J)[1])
        return log_det_J_Tinv

    def Tinv_packed(self, ζ):
        return self.Tinv(**self.unpack_ζ(ζ))

    def log_p_packed(self, y, ζ):
        # local variables
        n = len(y)
        assert len(ζ) == 3 * n + 7
        γ, h = ζ[:2 * n].reshape([n, 2]), ζ[2 * n:3 * n]
        # global variables
        μ, η, β, ln_σ2 = ζ[3 * n], ζ[3 * n + 1], ζ[3 * n + 2], ζ[3 * n + 3]
        ω = ζ[3 * n + 4:]
        # reverse transform variables
        θ = self.Tinv(γ, h, μ, η, β, ln_σ2, ω)
        # log density includes the jacobian from Tinv
        logdens = self.log_p(y, **θ) + self.log_det_J_Tinv(η, ln_σ2, ω)
        return logdens

    def log_q(self, φ, μ, L):
        return stats.multivariate_normal.logpdf(φ, μ, L@L.T)

    def elbo_hat(self, y, φ, ηs):
        n = len(y)
        k = 3*n+7
        accum = 0.
        μ, L = φ[:k], vec_to_tril(φ[k:])
        n_draws = ηs.shape[0]
        for i in range(n_draws):
            accum += self.log_p_packed(y, μ + np.dot(L, ηs[i, :]))
        return accum/n_draws - stats.multivariate_normal.entropy(μ, np.dot(L, L.T))

    def sgvb(self, y, ρ=0.01, max_iter=1_000, n_draws=1, noisy=False, quiet=False, rs=None):
        rs = rs or np.random.RandomState()
        n = len(y)
        k = 3*n+7
        φ = np.concatenate([[0]*k, tril_to_vec(np.eye(k))])
        grad_elbo_hat = jacobian(self.elbo_hat, argnum=1)
        for i in range(max_iter):
            print('**** values ****')
            print(self.unpack_ζ(φ[:k]))
            # print('**** gradients ****')
            # print(self.unpack_ζ(grad_φ[:k]))
            # print('**** marginal variances ****')
            # L = vec_to_tril(φ[k:]); V = L@L.T
            # print(self.unpack_ζ(np.diag(V)))
            # print('**** diagonal variance gradients ****')
            # print(self.unpack_ζ(np.diag(vec_to_trilpd(grad_φ[k:]))))
            η = rs.normal(size=[n_draws, k])
            grad_φ = grad_elbo_hat(y, φ, η)
            φ = φ + ρ*grad_φ
            if noisy or not quiet and not i & (i-1):
                try:
                    print(f'{i:4d}. L^ = {self.elbo_hat(y, φ, η)}')
                except np.linalg.LinAlgError:
                    print(f'{i:4d}. L^ could not be estimated, with φ:')
                    print(φ.round(2))
        return dict(μ=φ[:k], L=vec_to_trilpd(φ[k:]))


def _norm_ll(xs, mean, var):
    """Shortcut for computing a normal log likelihood with potentially many
    observations."""
    try:
        return np.sum(stats.norm.logpdf(xs, mean, np.sqrt(var)))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return -np.inf

def _trunc_norm_ll(x, mean, var, lower=-1., upper=1.):
    """Truncated normal log lhood. Returns -inf if outside [lower,upper]."""
    σ = np.sqrt(var)
    if lower <= x <= upper:
        return stats.norm.logpdf(x, mean, σ) - \
               np.log(stats.norm.cdf(upper, mean, σ) -
                      stats.norm.cdf(lower, mean, σ))
    else:
        return -np.inf


def _mvn_ll(Xs, mean, vcv):
    """Shortcut for computing a multivariate normal log likelihood with
    potentially many observations."""
    if np.ndim(Xs) == 1:
        return np.sum(stats.multivariate_normal.logpdf(Xs, mean, vcv))
    else:
        return np.sum([stats.multivariate_normal.logpdf(Xs[i, :], mean[i, :], vcv)
                      for i in range(Xs.shape[0])])


class SVMData(dict):
    """Container for SVM data. Dict with a few convenient functions for viewing
    the data.
    """
    def __getattr__(self, item):
        return self[item] if item in self.keys() else None

    def plot_data(self):
        plt.figure()
        plt.plot(self['y'])
        plt.title('$y_t$')

    def plot_state(self):
        plt.figure()
        # plt.box(False)
        plt.plot(self['h'], linewidth=2, color='black')
        # for i in [0, 1]:
        #     plt.plot(self['hCI'][i, :], '--r', linewidth=2)
        # plt.xlim((min(tid) - 1, 2014))
        plt.title('$h_t$')

    def plot_moving_params(self, names=None):
        names = names or ['α', 'τ']
        γ = self['γ']
        fig, axes = plt.subplots(ncols=γ.shape[1], squeeze=True)
        for i, (ax, name) in enumerate(zip(axes.ravel(), names)):
            ax.plot(γ[:, i], linewidth=2, color='black')
            #plt.xlim((min(tid) - 1, 2014))
            ax.set_title(name)


def main():
    ...


if __name__ == '__main__':
    main()
