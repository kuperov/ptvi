import torch
from torch.distributions import *
from time import time
import matplotlib.pyplot as plt
from ptvi.stopping import *
from ptvi.dist import InvGamma
from ptvi import VIModel
import numpy as np


class _LocalLevelResult(object):

    def __init__(self, model, elbo_hats, y):
        self.model, self.y, self.elbos = model, y, elbo_hats
        self.u, self.L = model.u.detach(), model.L.detach()
        # posteriors are transformed from normal distributions
        Σ = self.L @ (self.L.t())
        sds = torch.sqrt(torch.diag(Σ))
        self.γ_marg_post = Normal(self.u[self.τ], sds[self.τ])
        self.η_marg_post = TransformedDistribution(
            Normal(self.u[self.τ + 1], sds[self.τ + 1]), model.ψ_to_η)
        self.σ_marg_post = TransformedDistribution(
            Normal(self.u[self.τ + 2], sds[self.τ + 2]), model.ς_to_σ)
        self.ρ_marg_post = TransformedDistribution(
            Normal(self.u[self.τ + 3], sds[self.τ + 3]), model.φ_to_ρ)

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
        Σ = self.L @ (self.L.t())
        return torch.sqrt(torch.diag(Σ)).numpy()

    def plot_sampled_paths(self, N=50, fc_steps=0, true_y=None):
        paths = self.model.sample_paths(N, fc_steps=fc_steps)
        plt.figure()
        xs, fxs = range(self.τ), range(self.τ + fc_steps)
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
        ci_bands = np.empty([self.τ + fc_steps, 2])
        fxs, xs = range(self.τ + fc_steps), range(self.τ)
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


class LocalLevelModel(VIModel):
    # approximating density: q(u, LL')

    result_class = _LocalLevelResult
    global_params = ['γ', 'η', 'σ', 'ρ']
    transformed_params = {'η': 'ψ', 'σ': 'ς', 'ρ': 'φ'}

    def __init__(self, τ: int,
                 num_draws = 1,
                 stochastic_entropy: bool = False,
                 stop_heur: StoppingHeuristic = None,
                 γ_prior: Distribution = None,
                 η_prior: Distribution = None,
                 σ_prior: Distribution = None,
                 ρ_prior: Distribution = None,
                 ):
        self.τ, self.d = τ, τ + 4

        self.stop_heur = stop_heur or ExponentialStoppingHeuristic(50, 50)

        # q(ζ): dense matrices are inefficient but we'll keep τ small for now
        self.u = torch.tensor(torch.zeros(self.d), requires_grad=True)
        self.L = torch.tensor(torch.eye(self.d), requires_grad=True)
        self.parameters = [self.u, self.L]

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

        super().__init__(num_draws=num_draws,
            stochastic_entropy=stochastic_entropy,
            stop_heur=stop_heur)

    def elbo_hat(self, y):
        L = torch.tril(self.L)  # force gradients for L to be lower triangular
        _τ = self.τ
        E_ln_lik_hat, E_ln_pr_hat, H_q_hat = 0., 0., 0.  # accumulators

        q = MultivariateNormal(loc=self.u, scale_tril=L)
        if not self.stochastic_entropy:
            H_q_hat = q.entropy()

        for _ in range(self.num_draws):
            ζ = self.u + L@torch.randn((self.d,))  # reparam trick
            z, γ, ψ, ς, φ = ζ[:_τ], ζ[_τ], ζ[_τ + 1], ζ[_τ + 2], ζ[_τ + 3]
            # transform from optimization to user coordinates
            η, σ, ρ = self.ψ_to_η(ψ), self.ς_to_σ(ς), self.φ_to_ρ(φ)
            ar1_uncond_var = torch.pow((1 - torch.pow(ρ, 2)), -0.5)
            llikelihood = (
                Normal(γ + η * z, σ).log_prob(y).sum()
                + Normal(ρ * z[:-1], 1).log_prob(z[1:]).sum()
                + Normal(0., ar1_uncond_var).log_prob(z[0])
            )
            E_ln_lik_hat += llikelihood/self.num_draws
            lprior = (
                self.γ_prior.log_prob(γ)
                + self.η_prior.log_prob(η)
                + self.ς_prior.log_prob(ς)
                + self.φ_prior.log_prob(φ)
            )
            E_ln_pr_hat += lprior/self.num_draws
            if self.stochastic_entropy:
                H_q_hat += q.log_prob(ζ)/self.num_draws

        return E_ln_lik_hat + E_ln_pr_hat - H_q_hat

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

    def __str__(self):
        entr = 'stochastic' if self.stochastic_entropy else 'analytic'
        draw_s = 's' if self.num_draws > 1 else ''
        return (f"Local level model with τ={self.τ}:\n"
                f"    - {entr} entropy;\n"
                f"    - {self.num_draws} simulation draw{draw_s};\n"
                f"    - {str(self.stop_heur)}")


if __name__ == '__main__' and '__file__' in globals():
    torch.manual_seed(1234)
    γ0, η0, σ0, ρ0 = 0., 2., 1.5, 0.92
    y, z = LocalLevelModel(τ=100).simulate(γ=γ0, η=η0, σ=σ0, ρ=ρ0)

    m = LocalLevelModel(τ=100, stochastic_entropy=True, num_draws=2,
                        stop_heur=NoImprovementStoppingHeuristic())
    opt = torch.optim.Adadelta(m.parameters)
    fit = m.training_loop(y, optimizer=opt)
    print(fit.summary())
    fit.plot_elbos()
    plt.show()
    fit.plot_latent(true_z=z.numpy(), include_data=True)
    plt.show()
    # fit.plot_elbos()
    # plt.show()
    # fit.plot_sampled_paths(200, true_y=y, fc_steps=10)
    # plt.show()
    # fit.plot_pred_ci(true_y=y, fc_steps=10)
    # plt.show()
