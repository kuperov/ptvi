from typing import List

import torch
from torch.distributions import (
    Normal, MultivariateNormal, Distribution, ExpTransform, LogNormal,
    TransformedDistribution)
import matplotlib.pyplot as plt
import numpy as np

from ptvi import VIModel, VIResult, StoppingHeuristic


class GaussianResult(VIResult):
    """Base class for representing model results.
    """

    def __init__(self,
                 model: 'VIModel',
                 elbo_hats: List[float]):
        super().__init__(model=model, elbo_hats=elbo_hats)

        # approximating distribution
        self.q = MultivariateNormal(loc=model.u.detach(),
                                    scale_tril=model.L.detach())

        # marginal approximate posteriors
        self.μ_marg_post = Normal(self.q.mean[0],
                                  torch.sqrt(self.q.variance[0]))
        self.η_marg_post = Normal(self.q.mean[1],
                                  torch.sqrt(self.q.variance[1]))

        # transformed marginal posteriors
        self.σ_marg_post = TransformedDistribution(
            self.η_marg_post, self.η_to_σ
        )

    def summary(self):
        """Return a pandas data frame summarizing model parameters"""
        import pandas as pd
        # transform and simulate from marginal transformed parameters
        means, sds = [], []
        for param in self.global_params:
            post = getattr(self, f'{param}_marg_post')
            if isinstance(post, Normal):
                means.append(float(post.loc.numpy()))
                sds.append(float(post.scale.numpy()))
            else:  # simulate non-gaussian posteriors
                xs = post.sample((100,)).numpy()
                means.append(np.mean(xs))
                sds.append(np.std(xs))
        return pd.DataFrame({'mean': means, 'sd': sds}, index=self.global_params)

    def plot_elbos(self, skip=0):
        plt.figure()
        xs = np.arange(skip, len(self.elbo_hats))
        plt.plot(xs, self.elbo_hats[skip:])
        if skip > 0:
            plt.title(r'$\hat L$ by iteration')
        else:
            plt.title(r'$\hat L$ by iteration ({} skipped)'.format(skip))

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

    def __repr__(self):
        return repr(self.summary())


class UnivariateGaussian(VIModel):
    """
    Fit a simple univariate gaussian to an approximating density: q = N(u, LL').

    For the optimization, we transform σ -> ln(σ) = η to ensure σ > 0.
    """

    global_params = ['μ', 'σ']
    mapped_params = {'η': 'σ'}
    result_class = GaussianResult

    def __init__(self,
                 μ_prior: Distribution=None,
                 σ_prior: Distribution = None,
                 n_draws: int = 1,
                 stochastic_entropy: bool = False,
                 stop_heur: StoppingHeuristic = None
                 ):
        """Create a UnivariateGaussian model object.

        Args:
           n_draws: number of draws for simulating elbo
           μ_prior: prior for μ (default N(0, 10^2) )
           σ_prior: prior for σ (default LN(0, 10^2) )
           stochastic_entropy: simulate entropy term
           stop_heur: rule for stopping the computation
        """
        super().__init__(n_draws, stochastic_entropy, stop_heur)
        self.μ_prior = μ_prior or Normal(0, 10)
        self.σ_prior = σ_prior or LogNormal(0, 10)

        # elements of reverse transformation Tinv()
        self.η_to_σ = ExpTransform()

        # transformed priors include jacobian determinant
        self.η_prior = TransformedDistribution(self.σ_prior, self.η_to_σ.inv)

        # approximation: q = N(u, LL')
        self.d = 2
        self.u = torch.tensor(torch.zeros(self.d), requires_grad=True)
        self.L = torch.tensor(torch.eye(self.d), requires_grad=True)
        self.parameters = [self.u, self.L]

    def simulate(self, N: int, μ0: float, σ0: float, quiet=False):
        assert N > 2 and σ0 > 0
        dgp = Normal(torch.tensor([μ0]), torch.tensor([σ0]))
        y = dgp.sample((N,))
        if not quiet:
            print(f'Simulated {N} observations with mean = {torch.mean(y):.2f},'
                  f' sd = {torch.std(y):.2f}')
        return y

    def elbo_hat(self, y):
        L = torch.tril(self.L)  # ensure L remains lower triangular
        q = MultivariateNormal(self.u, scale_tril=L)  # approximating density
        E_ln_lik_hat, E_ln_pr_hat, H_q_hat = 0., 0., 0.  # accumulators

        for _ in range(self.n_draws):
            ζ = self.u + L@torch.randn((2,))
            μ, η = ζ[0], ζ[1]  # unpack drawn parameter
            σ = self.η_to_σ(η)  # transform to user parameters

            E_ln_lik_hat += Normal(μ, σ).log_prob(y).sum()/self.n_draws
            E_ln_pr_hat += (
                self.μ_prior.log_prob(μ) + self.η_prior.log_prob(η)
                )/self.n_draws

            if self.stochastic_entropy:
                H_q_hat += q.log_prob(ζ)/self.n_draws

        if not self.stochastic_entropy:
            H_q_hat = q.entropy()

        return E_ln_lik_hat + E_ln_pr_hat - H_q_hat

    def print_status(self, i, loss):
        μ_hat, η_hat = self.u[0], self.u[1]
        sds = torch.sqrt(torch.diag(self.L @ self.L.t()))
        μ_sd, η_sd = sds[0], sds[1]
        print(f'{i: 8d}. smoothed loss ={loss:12.2f}  '
              f'μ_hat ={μ_hat: 4.2f} ({μ_sd:4.2f}), '
              f'η_hat ={η_hat: 4.2f} ({η_sd:4.2f}) ')

    def __str__(self):
        entr = 'stochastic' if self.stochastic_entropy else 'analytic'
        draw_s = 's' if self.n_draws > 1 else ''
        return (f"Gaussian model:\n"
                f"    - {entr} entropy;\n"
                f"    - {self.n_draws} simulation draw{draw_s};\n"
                f"    - {str(self.stop_heur)}")


if __name__ == '__main__' and '__file__' in globals():  # ie run as script
    model = UnivariateGaussian(n_draws=1, stochastic_entropy=True)
    torch.manual_seed(123)
    N, μ0, σ0 = 100, 5., 5.
    y = model.simulate(N=N, μ0=μ0, σ0=σ0)
    result = model.training_loop(y)
    # result.plot_elbos()
    # plt.show()
    result.plot_marg('μ', true_val=μ0)
    plt.show()
    result.plot_marg('σ', true_val=σ0)
    plt.show()
