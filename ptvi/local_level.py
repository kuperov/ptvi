import torch
from torch.distributions import *
from time import time
import matplotlib.pyplot as plt
from ptvi.stopping import *
from ptvi.dist import InvGamma
import numpy as np


class LocalLevelModel(object):
    # approximating density: q(u, LL')

    def __init__(self, τ: int, num_draws = 1,
                 stoch_entropy: bool = False,
                 γ_prior: Distribution = None, η_prior: Distribution = None,
                 σ_prior: Distribution = None, ρ_prior: Distribution = None):
        self.τ, self.d = τ, τ + 4
        self.stoch_entropy, self.num_draws = stoch_entropy, num_draws

        # q(ζ): dense matrices are inefficient but we'll keep τ small for now
        self.u = torch.tensor(torch.zeros(self.d), requires_grad=True)
        self.L = torch.tensor(torch.eye(self.d), requires_grad=True)

        # priors are defined wrt user coordinates
        self.γ_prior = γ_prior or Normal(0, 3)
        self.η_prior = η_prior or Normal(0, 3)
        self.σ_prior = σ_prior or InvGamma(1, 5)
        self.ρ_prior = ρ_prior or TransformedDistribution(
            Beta(1, 1), AffineTransform(-1, 2))

        # transformations from optimization to user coordinates
        self.ς_to_σ = ExpTransform()
        self.ψ_to_η = ExpTransform()
        self.φ_to_ρ = ComposeTransform([
            AffineTransform(0, 1), SigmoidTransform(), AffineTransform(-1, 2)
        ])

        # transformed priors automatically include Jacobian transformations
        self.ψ_prior = TransformedDistribution(self.η_prior, self.ψ_to_η.inv)
        self.φ_prior = TransformedDistribution(self.ρ_prior, self.φ_to_ρ.inv)
        self.ς_prior = TransformedDistribution(self.σ_prior, self.ς_to_σ.inv)

    def elbo_hat(self, y):
        L = torch.tril(self.L)  # force gradients for L to be lower triangular
        _τ = self.τ
        ζ = self.u + L@torch.randn((self.d,))  # r16n trick

        z, γ, ψ, ς, φ = ζ[:_τ], ζ[_τ], ζ[_τ + 1], ζ[_τ + 2], ζ[_τ + 3]

        # transform from optimization to user coordinates
        η, σ, ρ = self.ψ_to_η(ψ), self.ς_to_σ(ς), self.φ_to_ρ(φ)

        # log joint = log likelihood + log prior
        ar1_var = torch.pow((1 - torch.pow(ρ, 2)), -0.5)
        log_likelihood = (
            Normal(γ + η * z, σ).log_prob(y).sum()
            + Normal(ρ * z[:-1], 1).log_prob(z[1:]).sum()
            + Normal(0., ar1_var).log_prob(z[0])
        )
        log_prior = (
            self.γ_prior.log_prob(γ)
            + self.η_prior.log_prob(η)
            + self.ς_prior.log_prob(ς)
            + self.φ_prior.log_prob(φ)
        )

        q = MultivariateNormal(loc=self.u, scale_tril=L)
        entropy_hat = q.log_prob(ζ) if self.stoch_entropy else q.entropy()

        return log_likelihood + log_prior - entropy_hat

    def parameters(self):
        return [self.u, self.L]

    def training_loop(self, y,
                      max_iters: int = 2**20,
                      stop_crit: StoppingHeuristic = None) -> None:
        stop_crit = stop_crit or ExponentialStoppingHeuristic(50, 50, .1)
        t, elbo_hats, objective = -time(), [], 0.
        optimizer = torch.optim.Adadelta(self.parameters())
        for i in range(max_iters):
            optimizer.zero_grad()
            objective = -self.elbo_hat(y)
            objective.backward()
            optimizer.step()
            elbo_hats.append(-objective)
            if not i & (i - 1):
                print(f'{i: 8d}. ll ={-objective:8.2f}')
            if stop_crit.early_stop(-objective):
                print('Early stopping criterion satisfied.')
                break
        else:
            print('WARNING: maximum iterations reached')
        t += time()
        print(f'{i: 8d}. ll ={-objective:8.2f}')
        print(f'Completed {i+1} iterations in {t:.2f}s @ {i/(t+1):.2f} i/s.')
        return type(self).Results(self, y, elbo_hats)

    def simulate(self, γ: float, η: float, σ: float, ρ: float):
        z = torch.empty([self.τ])
        z[0] = Normal(0, 1/(1 - ρ**2)**0.5).sample()
        for i in range(1, self.τ):
            z[i] = ρ*z[i-1] + Normal(0, 1).sample()
        y = Normal(γ + η*z, σ).sample()
        return y, z

    def sample_paths(self, N=100, fc_steps=0):
        """Sample N paths from the model, forecasting fc_steps additional steps.

        Args:
            N:        number of paths to sample
            fc_steps: number of steps to project forward

        Returns:
            Nx(τ+fc_steps) tensor of sample paths
        """
        paths = torch.empty((N, self.τ+fc_steps))
        ζs = MultivariateNormal(self.u, scale_tril=self.L).sample((N,))
        for i in range(N):
            ζ = ζs[i]
            γ, ψ, ς, φ = ζ[self.τ], ζ[self.τ + 1], ζ[self.τ + 2], ζ[self.τ + 3]
            z = torch.zeros((self.τ+fc_steps,))
            z[:self.τ] = ζ[:self.τ]
            η, σ, ρ = self.ψ_to_η(ψ), self.ς_to_σ(ς), self.φ_to_ρ(φ)
            # project states forward
            for t in range(self.τ, self.τ+fc_steps):
                z[t] = z[t-1]*ρ + Normal(0, 1).sample()
            paths[i, :] = Normal(γ + η*z, σ).sample()
        return paths

    class Results(object):

        def __init__(self, model, y, elbos):
            self.model, self.y, self.elbos = model, y, elbos
            self.u, self.L = model.u.detach(), model.L.detach()
            # posteriors are transformed from normal distributions
            Σ = self.L @ (self.L.t())
            sds = torch.sqrt(torch.diag(Σ))
            self.γ_marg_post = Normal(self.u[self.τ], sds[self.τ])
            self.η_marg_post = TransformedDistribution(
                Normal(self.u[self.τ+1], sds[self.τ+1]), model.ψ_to_η)
            self.σ_marg_post = TransformedDistribution(
                Normal(self.u[self.τ+2], sds[self.τ+2]), model.ς_to_σ)
            self.ρ_marg_post = TransformedDistribution(
                Normal(self.u[self.τ+3], sds[self.τ+3]), model.φ_to_ρ)

        def __getattr__(self, item):
            return getattr(self.model, item, None)

        def summary(self):
            import pandas as pd
            # transform and simulate from marginal transformed parameters
            params = ['γ', 'η', 'σ', 'ρ']
            means, sds = [], []
            for param in params:
                post = getattr(self, f'{param}_marg_post')
                if isinstance(post, Normal):
                    means.append(float(post.loc.numpy()))
                    sds.append(float(post.scale.numpy()))
                else:  # simulate non-gaussian posteriors
                    xs = post.sample((100,)).numpy()
                    means.append(np.mean(xs))
                    sds.append(np.std(xs))
            return pd.DataFrame({'mean': means, 'sd': sds}, index=params)

        def sds(self):
            Σ = self.L@(self.L.t())
            return torch.sqrt(torch.diag(Σ)).numpy()

        def plot_sampled_paths(self, N=50, fc_steps=0, true_y=None):
            paths = self.model.sample_paths(N, fc_steps=fc_steps)
            plt.figure()
            xs, fxs = range(self.τ), range(self.τ+fc_steps)
            for i in range(N):
                plt.plot(fxs, paths[i, :].numpy(), linewidth=0.5, alpha=0.5)
            if fc_steps > 0:
                plt.axvline(x=self.τ, color='black')
                plt.title(f'{N} posterior samples and {fc_steps}-step forecast')
            else:
                plt.title(f'{N} posterior samples')
            if true_y is not None:
                plt.plot(xs, true_y.numpy(), color='black', linewidth=2,
                         label='y')
                plt.legend()

        def plot_pred_ci(self, N: int = 100, α: float = 0.05, true_y=None,
                         fc_steps: int = 0):
            paths = self.model.sample_paths(N, fc_steps=fc_steps)
            ci_bands = np.empty([self.τ+fc_steps, 2])
            fxs, xs = range(self.τ+fc_steps), range(self.τ)
            perc = 100 * np.array([α * 0.5, 1. - α * 0.5])
            for t in fxs:
                ci_bands[t, :] = np.percentile(paths[:, t], q=perc)
            plt.figure()
            plt.fill_between(fxs, ci_bands[:, 0], ci_bands[:, 1], alpha=0.5,
                             label=f'{(1-α)*100:.0f}% CI')
            if true_y is not None:
                plt.plot(xs, true_y.numpy(), color='black', linewidth=2,
                         label='y')
                plt.legend()
            if fc_steps > 0:
                plt.axvline(x=self.τ, color='black')
                plt.title(f'Posterior credible interval and '
                          f'{fc_steps}-step-ahead forecast')
            else:
                plt.title(f'Posterior credible interval')

        def plot_elbos(self):
            plt.figure()
            plt.plot(self.elbos)
            plt.title(r'$\hat L$ by iteration')

        def plot_latent(self, true_z=None, include_data=False):
            plt.figure()
            zs = self.u[:-4].numpy()
            xs = torch.arange(len(zs)).numpy()
            sds = self.sds()[:-4]
            if include_data:
                plt.subplot(211)
                plt.plot(xs, self.y.numpy())
                plt.title('Observed data')
                plt.subplot(212)
            plt.plot(xs, zs, label=r'$E[z_{1:\tau} | y]$')
            plt.fill_between(xs, zs - sds, zs + sds, label=r'$\pm$ 1 SD',
                             color='blue', alpha=0.1)
            plt.title('Latent state')
            if true_z is not None:
                plt.plot(xs, true_z, label=r'$z_{0,1:\tau}$')
            plt.legend()
            plt.tight_layout()

        def plot_data(self):
            plt.figure()
            plt.plot(self.y.numpy(), label='data')
            plt.title('Data')

        def __repr__(self):
            return repr(self.summary())


if __name__ == '__main__' and '__file__' in globals():
    torch.manual_seed(123)
    m = LocalLevelModel(τ=100, stoch_entropy=False)
    y, z = m.simulate(γ=0., η=2., σ=1.5, ρ=0.98)
    fit = m.training_loop(y, max_iters=2**20)
    print(fit.summary())
    # fit.plot_latent(true_z=z.numpy(), include_data=True)
    # plt.show()
    # fit.plot_elbos()
    # plt.show()
    # fit.plot_sampled_paths(200, true_y=y, fc_steps=10)
    # plt.show()
    # fit.plot_pred_ci(true_y=y, fc_steps=10)
    # plt.show()
