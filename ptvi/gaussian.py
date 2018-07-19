import torch
from torch.distributions import (
    Normal, MultivariateNormal, Distribution, ExpTransform, LogNormal,
    TransformedDistribution)
from time import time
import matplotlib.pyplot as plt
from ptvi import StoppingHeuristic, ExponentialStoppingHeuristic


class UnivariateGaussian(object):
    """
    Fit a simple univariate gaussian to an approximating density: q = N(u, LL').

    For the optimization, we transform σ -> ln(σ) = η to ensure σ > 0.
    """

    def __init__(self,
                 n_draws: int = 1,
                 μ_prior: Distribution=None,
                 σ_prior: Distribution = None,
                 stochastic_entropy: bool = False):
        """Create a UnivariateGaussian model object.

        Args:
           n_draws: number of draws for simulating elbo
           μ_prior: prior for μ (default N(0, 10^2) )
           σ_prior: prior for μ (default LN(0, 10^2) )
           stochastic_entropy: simulate entropy term
        """
        self.stochastic_entropy, self.n_draws = stochastic_entropy, n_draws
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

    def simulate(self, N: int, μ0: float, σ0: float):
        assert N > 2 and σ0 > 0
        dgp = Normal(torch.tensor([μ0]), torch.tensor([σ0]))
        y = dgp.sample((N,))
        print(f'Simulated {N} observations with mean = {torch.mean(y):.2f}, '
              f'sd = {torch.std(y):.2f}')
        return y

    def elbo_hat(self, y):
        L = torch.tril(self.L)
        q = MultivariateNormal(self.u, scale_tril=L)  # approximating density
        E_log_lik_hat, E_l_p_hat, E_ln_q_hat = 0., 0., 0.  # accumulators

        for _ in range(self.n_draws):
            ζ = self.u + L@torch.randn((2,))
            μ, η = ζ[0], ζ[1]  # unpack drawn parameter
            σ = self.η_to_σ(η)  # transform to user parameters

            E_log_lik_hat += Normal(μ, σ).log_prob(y).sum()/self.n_draws
            E_l_p_hat += (self.μ_prior.log_prob(μ) + self.η_prior.log_prob(η)
                         )/self.n_draws

            if self.stochastic_entropy:
                E_ln_q_hat += q.log_prob(ζ)/self.n_draws

        if not self.stochastic_entropy:
            E_ln_q_hat = q.entropy()

        return E_log_lik_hat + E_l_p_hat - E_ln_q_hat

    def print_status(self, i, loss):
        μ_hat, η_hat = self.u[0], self.u[1]
        sds = torch.sqrt(torch.diag(self.L @ self.L.t()))
        μ_sd, η_sd = sds[0], sds[1]
        print(f'{i: 8d}. smoothed loss ={loss:12.2f}  '
              f'μ^ ={μ_hat: 4.2f} ({μ_sd:4.2f}), '
              f'η^ ={η_hat: 4.2f} ({η_sd:4.2f}) ')

    def training_loop(self, y, max_iters: int = 2**20, λ=0.1,
                      stop: StoppingHeuristic=None):
        """Train the model using VI.

        Args:
            y: (a 1-tensor) data vector
            max_iters: maximum number of iterations
            λ: exponential smoothing parameter for displaying estimated elbo
               (display only; does not affect the optimization)
            stop: rule for stopping the computation
        """
        msg = f'{" "*20}Gaussian Model with {len(y)} observations {" "*20}'
        print(f'{"="*len(msg)}\n{msg}\n{"="*len(msg)}')
        print(f'simulation draws = {self.n_draws}; '
              f'stochastic entropy: {self.stochastic_entropy}')
        stop = stop or ExponentialStoppingHeuristic(50, 50)
        print(f'stopping heuristic: {stop}')
        t = -time()
        elbo_hats = []
        optimizer = torch.optim.RMSprop([self.u, self.L])
        smoothed_elbo_hat = -self.elbo_hat(y)
        for i in range(max_iters):
            optimizer.zero_grad()
            objective = -self.elbo_hat(y)
            smoothed_elbo_hat = - λ*objective - (1-λ)*smoothed_elbo_hat
            objective.backward()
            optimizer.step()
            elbo_hats.append(-objective)
            if not i & (i - 1):
                self.print_status(i, smoothed_elbo_hat)
            if stop.early_stop(-objective):
                print('Stopping heuristic criterion satisfied')
                break
        else:
            print('WARNING: maximum iterations reached.')
        t += time()
        self.print_status(i+1, smoothed_elbo_hat)
        print(f'Completed {i+1} iterations in {t:.2f}s @ {i/(t+1):.2f} iter/s.')
        return elbo_hats


if __name__ == '__main__' and '__file__' in globals():  # ie run as script
    model = UnivariateGaussian(n_draws=1, stochastic_entropy=False)
    torch.manual_seed(123)
    y = model.simulate(N=100, μ0=5., σ0=5.)
    elbo_hats = model.training_loop(y)
    plt.plot(elbo_hats)
    plt.show()
