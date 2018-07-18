import matplotlib.pyplot as plt
from autograd import numpy as np, jacobian
from autograd.scipy.stats import multivariate_normal as mvn, norm, beta
from ptvi.attic import invgamma
from ptvi.learning_rate import make_advi_step_sequence
import pandas as pd


class UnivariateLocalLevel(object):
    """Simplest possible dense matrix implementation of SVGB for a univariate
    local level model.
    """

    def __init__(self, τ):
        self.τ = τ

    def log_joint(self, y, z, γ, η, σ, ρ):
        """Autograd-differentiable function for log p(y, z, γ, η, σ, ρ)."""
        llik = (
            np.sum(norm.logpdf(y, γ + η*z, σ))
            + np.sum(norm.logpdf(z[1:], ρ*z[:-1], 1))
            + norm.logpdf(z[0], 0, 1./(1 - ρ**2))
        )
        lprior = (
                norm.logpdf(γ, 0, 10) +
                norm.logpdf(η, 4, 10) +
                invgamma.logpdf(σ, 4, 10) +
                beta.logpdf(np.abs(ρ), 2, 2)
        )
        return llik + lprior

    def simulate(self, γ, η, σ, ρ, rs=None):
        rs = rs or np.random.RandomState()
        z = np.empty([self.τ])
        z[0] = rs.normal(0, 1/(1-ρ**2))
        for i in range(1, self.τ):
            z[i] = ρ*z[i-1] + rs.normal()
        y = γ + η*z + σ*rs.normal(size=[self.τ])
        return z, y

    def log_q(self, ζ, μ, L):
        """Approximating density log q(θ | μ, L)"""
        return mvn.logpdf(ζ, μ, np.dot(L, L.T))

    def unstack_ζ(self, ζ):
        _τ = self.τ
        z, γ, η, ς, φ = ζ[:_τ], ζ[_τ], ζ[_τ + 1], ζ[_τ + 2], ζ[_τ + 3]
        return z, γ, η, ς, φ

    def transformed_log_joint(self, y, ζ):
        z, γ, η, ς, φ = self.unstack_ζ(ζ)
        σ, ρ = self.Tinv(ς, φ)
        ζ_sub = np.array([ς, φ])

        def TinvVec(ζ_sub):
            ς, φ = ζ_sub[0], ζ_sub[1]
            return np.array(self.Tinv(ς, φ))
        log_jac_adjust = np.abs(np.linalg.slogdet(jacobian(TinvVec)(ζ_sub))[1])
        return self.log_joint(y, z, γ, η, σ, ρ) + log_jac_adjust

    def T(self, σ, ρ):
        """Transform global variables from user coords into unrestricted
        optimization coords.
        """
        ς, φ = np.log(σ), np.arctanh(ρ)
        return ς, φ

    def Tinv(self, ς, φ):
        σ, ρ = np.exp(ς), np.tanh(φ)
        return σ, ρ

    def advi(self, y, ζ0=None, n_draws=1, ρ_generator=None, quiet=True, noisy=False, n_iter=1_000, callback=None, rs=None):
        rs = rs or np.random.RandomState()
        ρ_generator = ρ_generator or make_advi_step_sequence()
        if ζ0 is None:
            ζ0 = np.zeros(self.τ+4)
        k = len(ζ0)
        μ, L = ζ0, np.eye(k)  # initial conditions
        djoint_dζ = jacobian(self.transformed_log_joint, argnum=1)
        elbo_hat = None
        for i in range(n_iter):
            g_μ, g_L = np.zeros(k), np.zeros([k, k])
            for j in range(n_draws):
                η = rs.normal(size=k)
                ζ = μ + np.dot(L, η)  # reparam. trick: ζ ∼ mvn(μ, Λ)
                dpdζ = djoint_dζ(y, ζ)
                assert np.all(np.isfinite(dpdζ)), f'{i}. dp/dζ not finite'
                g_μ += dpdζ
                g_L += np.outer(dpdζ, η)
            g_μ /= n_draws
            g_L = g_L/n_draws + np.linalg.inv(L).T
            ρ = ρ_generator(dpdζ)
            μ = μ + np.dot(np.diag(ρ), g_μ)
            L = L + np.dot(np.diag(ρ), g_L)

            assert np.all(np.isfinite(μ)), f'{i}. μ not finite'
            assert np.all(np.isfinite(L)), f'{i}. L not finite'

            if callback: callback(i, μ, L)

            try:
                elbo_hat_t = (self.transformed_log_joint(y, ζ)
                              - mvn.entropy(ζ, μ, L@L.T, allow_singular=True))
                if elbo_hat:
                    elbo_hat = 0.1 * elbo_hat_t + 0.9 * elbo_hat
                else:
                    elbo_hat = elbo_hat_t

                if noisy or (not quiet and (not i&(i-1))):
                    print(f'{i:4d}. L^ = {elbo_hat:.2f}')
            except Exception as e:
                import traceback
                print(f'gosh darn')
                traceback.print_exc()
        return μ, L

    def plot_latent(self, t, μ, L):
        Llat = L[:self.τ, :self.τ]
        μlat = μ[:self.τ]
        sd = np.sqrt(np.trace(Llat@Llat.T))
        xs = np.arange(self.τ)
        plt.plot(xs, μlat, label='mean')
        plt.fill_between(xs, μlat-sd, μlat+sd, label='+/- 1 s.d.')
        plt.title(f'Latent states (iter {t})')


def main():
    rs = np.random.RandomState(seed=123)
    model = UnivariateLocalLevel(τ=50)
    y, _ = model.simulate(γ=100., η=2., σ=1.5, ρ=0.8, rs=rs)

    def callback(t, μ, L):
        # if t % 100:
        #     return
        model.plot_latent(t, μ, L)
        plt.draw()
        plt.pause(1e-10)

    μ, L = model.advi(y, n_draws=5, noisy=True, callback=callback)
    sd = np.sqrt(np.trace(L@L.T))
    result = pd.DataFrame({'mean':μ, 'sd': sd})
    print(result)


if __name__ == '__main__':
    main()
