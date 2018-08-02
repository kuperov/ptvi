import math
import collections
from typing import List
import torch
from time import time
from torch.optim import Adadelta
from torch.distributions import (
    Distribution, MultivariateNormal, Normal, TransformedDistribution,
    Categorical)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ptvi import Model, StoppingHeuristic, SupGrowthStoppingHeuristic
from ptvi.params import TransformedModelParameter, LocalParameter
from ptvi.algos.sgvb import SGVBResult


_DIVIDER = "―"*80
_default_heuristic_type = SupGrowthStoppingHeuristic


def svgb_filt(model: Model,
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
    stop_heur = stop_heur or _default_heuristic_type()

    # dense approximation: q = N(u, LL')
    u0 = torch.tensor(u0) if u0 is not None else torch.zeros(model.d)
    L0 = torch.tensor(L0) if L0 is not None else torch.eye(model.d)

    u = torch.tensor(u0, requires_grad=True)
    L = torch.tensor(L0, requires_grad=True)

    optimizer = (optimizer_type or Adadelta)([u, L], **opt_params)

    def qprint(s):
        if not quiet: print(s)

    qprint(_header('Structured', optimizer, sim_entropy, stop_heur, model.name,
                   num_draws, λ))

    def elbo_hat():
        trL = torch.tril(L)
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        if not sim_entropy:
            q = MultivariateNormal(loc=u, scale_tril=trL)
            H_q_hat = -q.entropy()
        else:
            # don't accumulate gradients; see https://arxiv.org/abs/1703.09194
            q = MultivariateNormal(loc=u.detach(), scale_tril=trL.detach())
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
        f'{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}\n'
        f'Completed {i+1} iterations in {t:.1f}s @ {(i+1)/(t+1e-10):.2f} i/s.\n'
        f'{_DIVIDER}')
    q = MultivariateNormal(u.detach(), scale_tril=L.detach())
    return SGVBResult(model=model, elbo_hats=elbo_hats, y=y, q=q)


def _header(inftype, optimizer, stochastic_entropy, stop_heur, model_name, num_draws, λ):
    if stochastic_entropy:
        title = f'{inftype} SGVB Inference'
    else:
        title = f'{inftype} ADVI'
    lines = [_DIVIDER, f"{title}: {model_name}"]
    lines += [f"  - Estimating elbo with M={num_draws};",
              f"  - {str(stop_heur)}",
              f'  - {type(optimizer).__name__} optimizer with param groups:']
    for i, pg in enumerate(optimizer.param_groups):
        desc = ', '.join(f'{k}={v}' for k, v in pg.items() if k != 'params')
        lines.append(f'    group {i}. {desc}')
    lines.append(f'  - Displayed loss is smoothed with λ={λ}')
    lines.append(_DIVIDER)
    return '\n'.join(lines)


class StateSpaceModel(object):

    def simulate(self, T):
        raise NotImplementedError

    def conditional_log_prob(self, t, x_1t, z_1t):
        raise NotImplementedError


class PFProposal(object):

    def conditional_sample(self, t, Z):
        raise NotImplementedError

    def conditional_log_prob(self, t, Z):
        raise NotImplementedError


class StochasticVolatilityModel(StateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)
    """
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def simulate(self, T):
        """Simulate from p(x, z | θ)"""
        z_true = torch.empty((T,))
        z_true[0] = Normal(self.b, (1-self.c**2)**(-.5)).sample()
        for t in range(1, T):
            z_true[t] = self.b + self.c * z_true[t-1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(self.a) * torch.exp(z_true/2)).sample()
        return x, z_true

    def conditional_log_prob(self, t, x_1t, z_1t):
        """Compute log p(x_t, z_t | x_{1:t}, z_{1:t}, θ)"""
        if t == 0:
            log_pzt = Normal(self.b, (1-self.c**2)**(-.5)).log_prob(z_1t[t])
        else:
            log_pzt = Normal(self.b + self.c * z_1t[t], 1).log_prob(z_1t[t])
        log_pxt = (Normal(0, torch.exp(self.a)
                          * torch.exp(z_1t[t]/2)).log_prob(x_1t[t]))
        return log_pzt + log_pxt

    def __repr__(self):
        return (
            f"Stochastic volatility model\n"
            f"x_t = exp({self.a:.2f} * z_t/2) ε_t\n"
            f"z_t = {self.b:.2f} + {self.c:.2f} * z_{{t-1}} + ν_t\n"
            f"ε_t ~ Ν(0,1)\n"
            f"ν_t ~ Ν(0,1)")


class AR1Proposal(PFProposal):
    """A simple linear/gaussian AR(1) to use as a particle filter proposal.

    .. math::
        z_t = μ + ρ * z_{t-1} + η_t
    """
    def __init__(self, μ, ρ):
        self.μ, self.ρ = μ, ρ

    def conditional_sample(self, t, Z):
        """Simulate z_t from q(z_t | z_{t-1}, φ)

        Z has an extra dimension, of N particles.
        """
        N = Z.shape[1]
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2)**(-.5)).sample((N,))
        else:
            return Normal(self.μ + self.ρ * Z[t-1], 1).sample()

    def conditional_log_prob(self, t, Z):
        """Compute log q(z_t | z_{t-1}, φ)"""
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2) ** (-.5)).log_prob(Z[t])
        else:
            return Normal(self.μ + self.ρ * Z[t - 1], 1).log_prob(Z[t])

    def __repr__(self):
        return ("AR(1) proposals\n"
            f"z_t = {self.μ:.2f} + {self.ρ:.2f} * z_{{t-1}} + η_t\n"
            f"η_t ~ Ν(0,1)"
        )


def simulate_log_phatN(x: torch.Tensor, p: StateSpaceModel, q: PFProposal,
                       N: int, resample=False, rewrite_history=False):
    T, log_phatN = x.shape[0], 0
    log_w = torch.tensor([math.log(1/N)]*N)
    Z, resampled = torch.zeros((T, N)), [False] * T
    for t in range(T):
        Z[t, :] = q.conditional_sample(t, Z)
        log_αt = p.conditional_log_prob(t, x, Z) - q.conditional_log_prob(t, Z)
        log_phatt = torch.logsumexp(log_w + log_αt, dim=0)
        log_phatN += log_phatt
        log_w += log_αt - log_phatt
        ESS = 1./torch.exp(2*log_w).sum()
        if ESS < N and resample:
            resampled[t] = True
            a = Categorical(torch.exp(log_w)).sample((N,))
            if rewrite_history:
                Z = Z[:, a]
            else:
                Z[t, :] = Z[t, a]
            log_w = torch.tensor([math.log(1 / N)] * N)
    return log_phatN, Z, resampled


if __name__ == '__main__' and '__file__' in globals():
    import matplotlib.pyplot as plt
    torch.manual_seed(123)
    T = 200
    p_true = StochasticVolatilityModel(a=1., b=0., c=0.5)
    x, z_true = p_true.simulate(T)
    p = StochasticVolatilityModel(a=1., b=0., c=0.5)
    q = AR1Proposal(μ=0., ρ=0.5)
    log_phatN, Z, resamp = simulate_log_phatN(x, p, q, N=10, resample=True)
    plt.plot(Z.numpy(), alpha=0.1)
    plt.plot(z_true.numpy(), color='black', label='true z')
    plt.legend()
    plt.show()
