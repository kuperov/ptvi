from time import time
from typing import List

import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np

from ptvi import StoppingHeuristic, ExponentialStoppingHeuristic, plot_dens


class VIResult(object):
    """Base class for representing model results.
    """

    def __init__(self,
                 model: 'VIModel',
                 elbo_hats: List[float]):
        self.elbo_hats, self.model = elbo_hats, model

    def __getattr__(self, item):
        """Forward requests for members to the model. Useful for priors."""
        return getattr(self.model, item, None)

    def summary(self):
        """Return a pandas data frame summarizing model parameters"""
        raise NotImplementedError
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

    def plot_marg(self, variable: str, suffix='', true_val: float=None, **kwargs):
        post_key = f'$p({variable}{suffix} | y)$'
        prior_key = f'$p({variable}{suffix})$'
        args = {
            prior_key: getattr(self, f'{variable}_prior'),
            post_key: getattr(self, f'{variable}_marg_post')
        }
        if true_val is not None:
            args[f'${variable}{suffix}_0$'] = true_val
        if kwargs:
            args.update(kwargs)
        # figure out range by sampling from posterior
        variates = args[post_key].sample((100,))
        a, b = min(variates), max(variates)
        a -= 0.25 * (b-a)
        b += 0.25 * (b-a)
        plot_dens(args, a, b)

    def __repr__(self):
        return repr(self.summary())



class VIModel(object):
    """Abstract class for performing VI with general models (not necessarily
    time series models)"""

    result_class = VIResult

    def __init__(self,
                 n_draws: int = 1,
                 stochastic_entropy: bool = False,
                 stop_heur: StoppingHeuristic = None):
        self.stochastic_entropy, self.n_draws = stochastic_entropy, n_draws
        self.stop_heur = stop_heur or ExponentialStoppingHeuristic(50, 50)

    def simulate(self, *args, quiet=False):
        raise NotImplementedError

    def training_loop(self, y, max_iters: int = 2**20, λ=0.1, quiet=False,
                      optimizer=None):
        """Train the model using VI.

        Args:
            y: (a 1-tensor) data vector
            max_iters: maximum number of iterations
            λ: exponential smoothing parameter for displaying estimated elbo
               (display only; does not affect the optimization)
            quiet: suppress output
            optimizer: override optimizer

        Returns:
            A VariationalResults object with the approximate posterior.
        """
        optimizer = optimizer or torch.optim.RMSprop(self.parameters)
        if not quiet:
            print(f'{"="*80}')
            print(str(self))
            print(f'\n{type(optimizer).__name__} optimizer with param groups:')
            for i, pg in enumerate(optimizer.param_groups):
                desc = ', '.join(f'{k}={v}' for k, v in pg.items()
                                 if k != 'params')
                print(f'    group {i}. {desc}')
            print(f'\nDisplayed loss is smoothed with λ={λ}')
            print(f'{"="*80}')
        t, i = -time(), 0
        elbo_hats = []
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
            if self.stop_heur.early_stop(-objective):
                print('Stopping heuristic criterion satisfied')
                break
        else:
            if not quiet: print('WARNING: maximum iterations reached.')
        t += time()
        if not quiet:
            self.print_status(i + 1, smoothed_elbo_hat)
            r = i/(t+1)
            print(f'Completed {i+1} iterations in {t:.1f}s @ {r:.2f} i/s.')
            print(f'{"="*80}')
        return self.result_class(model=self, elbo_hats=elbo_hats)

    def print_status(self, i, loss):
        print(f'{i: 8d}. smoothed loss ={loss:12.2f}')

    def __str__(self):
        raise NotImplementedError
