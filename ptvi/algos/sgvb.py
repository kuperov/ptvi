import collections
from typing import List
import torch
from time import time
from torch.optim import Adadelta
from torch.distributions import (
    MultivariateNormal, Normal, TransformedDistribution)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ptvi import Model, StoppingHeuristic, NoImprovementStoppingHeuristic
from ptvi.params import TransformedModelParameter, LocalParameter


_DIVIDER = "―"*80


def sgvb(model: Model,
         y: torch.Tensor,
         u0: torch.Tensor=None,
         L0: torch.Tensor=None,
         max_iters: int = 2 ** 20,
         sim_entropy: bool = True,
         stop_heur: StoppingHeuristic = None,
         num_draws: int = 1,
         optimizer_type: type=None,
         quiet=False,
         λ=.1,
         **opt_params):
    """Train the model using VI.

    Args:
        model: a Model instance to fit to the data
        y: (a 1-tensor) data vector
        max_iters: maximum number of iterations
        sim_entropy: if true, simulate entropy term
        quiet: suppress output
        λ: exponential smoothing parameter for displaying estimated elbo
           (display only; does not affect the optimization)

    Returns:
        A SGVBResult object with the approximate posterior.
    """
    stop_heur = stop_heur or NoImprovementStoppingHeuristic()

    # dense approximation: q = N(u, LL')
    u0 = torch.tensor(u0) if u0 is not None else torch.zeros(model.d)
    L0 = torch.tensor(L0) if L0 is not None else torch.eye(model.d)

    u = torch.tensor(u0, requires_grad=True)
    L = torch.tensor(L0, requires_grad=True)

    optimizer = (optimizer_type or Adadelta)([u, L], **opt_params)

    def qprint(s):
        if not quiet: print(s)

    qprint(header(optimizer, sim_entropy, stop_heur, model.name, num_draws, λ))

    def elbo_hat():
        trL = torch.tril(L)
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        q = MultivariateNormal(loc=u, scale_tril=trL)
        if not sim_entropy:
            H_q_hat = q.entropy()
        for _ in range(num_draws):
            ζ = u + trL @ torch.randn((model.d,))  # reparam trick
            E_ln_joint += model.ln_joint(y, ζ) / num_draws
            if sim_entropy:
                H_q_hat += q.log_prob(ζ) / num_draws
        return E_ln_joint - H_q_hat

    t, i = -time(), 0
    elbo_hats = []
    smoothed_objective = -elbo_hat().data
    for i in range(max_iters):
        optimizer.zero_grad()
        objective = -elbo_hat()
        objective.backward()
        if torch.isnan(objective.data):
            raise Exception('Infinite objective; cannot continue.')
        optimizer.step()
        elbo_hats.append(-objective.data)
        smoothed_objective = λ * objective.data + (1. - λ) * smoothed_objective
        if not i & (i - 1):
            qprint(f'{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}')
        if stop_heur.early_stop(-objective.data):
            qprint('Stopping heuristic criterion satisfied')
            break
    else:
        qprint('WARNING: maximum iterations reached.')
    t += time()
    qprint(
        f'{i: 8d}. smoothed elbo ={float(-smoothed_objective):12.2f}\n'
        f'Completed {i+1} iterations in {t:.1f}s @ {(i+1)/(t+1e-10):.2f} i/s.\n'
        f'{_DIVIDER}')
    return SGVBResult(model=model, elbo_hats=elbo_hats, y=y, u=u, L=L)


def header(optimizer, stochastic_entropy, stop_heur, model_name, num_draws, λ):
    if stochastic_entropy:
        title = 'Structured SGVB Inference'
    else:
        title = 'Structured ADVI'
    lines = [_DIVIDER, f"{title} for {model_name}:"]
    lines += [f"  - Estimating elbo with M={num_draws};",
              f"  - {str(stop_heur)}",
              f'  - {type(optimizer).__name__} optimizer with param groups:']
    for i, pg in enumerate(optimizer.param_groups):
        desc = ', '.join(f'{k}={v}' for k, v in pg.items() if k != 'params')
        lines.append(f'    group {i}. {desc}')
    lines.append(f'  - Displayed loss is smoothed with λ={λ}')
    lines.append(_DIVIDER)
    return '\n'.join(lines)


class SGVBResult(object):
    """Base class for representing model results."""

    def __init__(self, model: 'Model', elbo_hats: List[float], y: torch.Tensor,
                 u: torch.Tensor, L: torch.Tensor):
        self.elbo_hats, self.model, self.y = elbo_hats, model, y
        self.u: torch.Tensor = u.detach()
        self.L: torch.Tensor = L.detach()
        self.input_length = len(y)

        # posteriors are transformed from normal distributions
        self.q = MultivariateNormal(self.u, scale_tril=self.L)

        sds = torch.sqrt(self.q.variance)
        for p in model.params:
            setattr(self, p.prior_name, getattr(model, p.prior_name))
            if p.dimension > 1:
                continue
            # construct marginals in optimization space
            if isinstance(p, TransformedModelParameter):
                tfm_post_marg = Normal(self.u[p.index], sds[p.index])
                setattr(self, p.tfm_post_marg_name, tfm_post_marg)

                tfm_prior = getattr(model, p.tfm_prior_name)
                setattr(self, p.tfm_prior_name, tfm_prior)
                tfm = getattr(model, p.tfm_name)
                setattr(self, p.tfm_name, tfm)
                post_marg = TransformedDistribution(tfm_post_marg, tfm.inv)
                setattr(self, p.post_marg_name, post_marg)
            else:
                post_marg = Normal(self.u[p.index], sds[p.index])
                setattr(self, p.post_marg_name, post_marg)

    def sample_paths(self, N=100, fc_steps=0):
        """Sample N paths from the model, forecasting fc_steps additional steps.

        Args:
            N:        number of paths to sample
            fc_steps: number of steps to project forward

        Returns:
            Nx(τ+fc_steps) tensor of sample paths
        """
        paths = torch.empty((N, self.input_length + fc_steps))
        ζs = MultivariateNormal(self.u, scale_tril=self.L).sample((N,))
        for i in range(N):
            ζ = ζs[i]
            paths[i, :] = self.model.sample_observed(ζ, fc_steps=fc_steps)
        return paths

    def summary(self):
        """Return a pandas data frame summarizing model parameters"""
        # transform and simulate from marginal transformed parameters
        names, means, sds = [], [], []
        for p in self.model.params:
            post = getattr(self, p.post_marg_name, None)
            if p.dimension > 1 and post is not None:
                continue
            if isinstance(post, Normal):
                names.append(p.name)
                means.append(float(post.loc))
                sds.append(float(post.scale))
            elif post is not None:  # simulate non-gaussian posteriors
                names.append(p.name)
                xs = post.sample((100,))
                means.append(float(torch.mean(xs)))
                sds.append(float(torch.std(xs)))
        return pd.DataFrame({'mean': means, 'sd': sds}, index=names)

    def plot_elbos(self, skip=0):
        xs = range(skip, len(self.elbo_hats))
        plt.plot(xs, self.elbo_hats[skip:])
        plt.title(r'$\hat L$ by iteration')

    def plot_latent(self, **true_vals):
        true_vals = true_vals or {}
        locals = [p for p in self.model.params
                  if isinstance(p, LocalParameter)]
        fig, axes = plt.subplots(nrows=len(locals), ncols=1)
        if not isinstance(axes, collections.Iterable):
            axes = [axes]
        for ax, p in zip(axes, locals):
            zs = self.q.mean[p.index:p.index+p.dimension].numpy()
            xs = torch.arange(len(zs)).numpy()
            vars = self.q.variance[p.index:p.index+p.dimension]
            sds = torch.sqrt(vars).numpy()
            if p.name in true_vals:
                true = true_vals[p.name]
                if hasattr(true, 'numpy'):
                    true = true.numpy()
                plt.plot(xs, true, label=p.name)
            plt.plot(xs, zs, label=f'$E[{p.name} | data]$')
            plt.fill_between(xs, zs - sds, zs + sds, label=r'$\pm$ 1 SD',
                             color='blue', alpha=0.1)
            plt.title(f'Local variable - {p.name}')
            plt.legend()
        plt.tight_layout()

    def plot_data(self):
        plt.plot(self.y.numpy(), label='data')
        plt.title('Data')

    def plot_marg_post(self, variable: str, suffix='', true_val: float=None,
                       plot_prior=True):
        """Plot marginal posterior distribution, prior, and optionally the
        true value.
        """
        post = getattr(self, f'{variable}_post_marg')
        prior = getattr(self, f'{variable}_prior')
        # figure out range by sampling from posterior
        variates = post.sample((100,))
        a, b = min(variates), max(variates)
        if true_val:
            a, b = min(a, true_val), max(b, true_val)
        xs = torch.linspace(a-(b-a)/4., b+(b-a)/4., 500)

        def plotpdf(p, label=''):
            ys = torch.exp(p.log_prob(xs))
            ys[torch.isnan(ys)] = 0
            plt.plot(xs.numpy(), ys.numpy(), label=label)

        if plot_prior:
            plotpdf(prior, label=f'$p({variable}{suffix})$')
        plotpdf(post, label=f'$p({variable}{suffix} | y)$')
        if true_val is not None:
            plt.axvline(true_val, label=f'${variable}{suffix}$', linestyle='--')
        plt.legend()

    def plot_global_marginals(self, cols=2, **true_vals):
        import math
        params = [p for p in self.model.params
                  if not isinstance(p, LocalParameter)]
        rows = round(math.ceil(len(params) / cols))
        for r in range(rows):
            for c in range(cols):
                plt.subplot(rows, cols, r*cols + c + 1)
                p = params[r * cols + c]
                self.plot_marg_post(p.name, true_val=true_vals.get(p.name, None))
        plt.tight_layout()

    def plot_sample_paths(self, N=50, fc_steps=0, true_y=None):
        paths = self.sample_paths(N, fc_steps=fc_steps)
        xs, fxs = range(self.input_length), range(self.input_length+fc_steps)
        for i in range(N):
            plt.plot(fxs, paths[i, :].numpy(), linewidth=0.5, alpha=0.5)
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color='black')
            plt.title(f'{N} posterior samples and {fc_steps}-step forecast')
        else:
            plt.title(f'{N} posterior samples')
        if true_y is not None:
            plt.plot(xs, true_y.numpy(), color='black', linewidth=2, label='y')
            plt.legend()

    def plot_pred_ci(self, N:int=100, α:float=0.05, true_y=None,
                     fc_steps:int=0):
        paths = self.sample_paths(N, fc_steps=fc_steps)
        ci_bands = np.empty([self.input_length+fc_steps, 2])
        fxs, xs = range(self.input_length+fc_steps), range(self.input_length)
        perc = 100 * np.array([α * 0.5, 1. - α * 0.5])
        for t in fxs:
            ci_bands[t, :] = np.percentile(paths[:, t], q=perc)
        plt.fill_between(fxs, ci_bands[:, 0], ci_bands[:, 1], alpha=0.5,
                         label=f'{(1-α)*100:.0f}% CI')
        if true_y is not None:
            plt.plot(xs, true_y.numpy(), color='black', linewidth=2, label='y')
            plt.legend()
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color='black')
            plt.title(f'Posterior credible interval and '
                      f'{fc_steps}-step-ahead forecast')
        else:
            plt.title(f'Posterior credible interval')

