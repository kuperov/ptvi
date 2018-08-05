import collections
from typing import List
import torch
from time import time
from torch.optim import Adadelta
from torch.distributions import (
    Distribution,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ptvi import Model, StoppingHeuristic, SupGrowthStoppingHeuristic
from ptvi.params import TransformedModelParameter, LocalParameter


_DIVIDER = "―" * 80
_default_heuristic_type = SupGrowthStoppingHeuristic


def sgvb(
    model: Model,
    y: torch.Tensor,
    u0: torch.Tensor = None,
    L0: torch.Tensor = None,
    max_iters: int = 2 ** 20,
    sim_entropy: bool = True,
    stop_heur: StoppingHeuristic = None,
    num_draws: int = 1,
    optimizer_type: type = None,
    quiet=False,
    λ=.1,
    **opt_params,
):
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
        if not quiet:
            print(s)

    qprint(
        _header(
            "Structured", optimizer, sim_entropy, stop_heur, model.name, num_draws, λ
        )
    )

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
            raise Exception("Infinite objective; cannot continue.")
        optimizer.step()
        elbo_hats.append(-objective.data)
        smoothed_objective = λ * objective.data + (1. - λ) * smoothed_objective
        if not i & (i - 1):
            qprint(f"{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}")
        if stop_heur.early_stop(-objective.data):
            qprint("Stopping heuristic criterion satisfied")
            break
    else:
        qprint("WARNING: maximum iterations reached.")
    t += time()
    qprint(
        f"{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}\n"
        f"Completed {i+1} iterations in {t:.1f}s @ {(i+1)/(t+1e-10):.2f} i/s.\n"
        f"{_DIVIDER}"
    )
    u: torch.Tensor = u.detach()
    L: torch.Tensor = L.detach()
    q = MultivariateNormal(u, scale_tril=L)
    return model.result_type(model=model, elbo_hats=elbo_hats, y=y, q=q)


def mf_sgvb(
    model: Model,
    y: torch.Tensor,
    u0: torch.Tensor = None,
    L0: torch.Tensor = None,
    max_iters: int = 2 ** 20,
    sim_entropy: bool = True,
    stop_heur: StoppingHeuristic = None,
    num_draws: int = 1,
    optimizer_type: type = None,
    quiet=False,
    λ=.1,
    **opt_params,
):
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
    omega0 = torch.tensor(L0) if L0 is not None else torch.zeros(model.d)

    u = torch.tensor(u0, requires_grad=True)
    omega = torch.tensor(omega0, requires_grad=True)

    optimizer = (optimizer_type or Adadelta)([u, omega], **opt_params)

    def qprint(s):
        if not quiet:
            print(s)

    qprint(
        _header(
            "Mean-field", optimizer, sim_entropy, stop_heur, model.name, num_draws, λ
        )
    )

    def elbo_hat():
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        if not sim_entropy:
            q = Normal(loc=u, scale=torch.exp(omega / 2))
            H_q_hat = -q.entropy().sum()
        else:
            # don't accumulate gradients; see https://arxiv.org/abs/1703.09194
            q = Normal(loc=u.detach(), scale=torch.exp(omega.detach() / 2))
        for _ in range(num_draws):
            ζ = u + torch.exp(omega / 2) * torch.randn((model.d,))  # reparam trick
            E_ln_joint += model.ln_joint(y, ζ) / num_draws
            if sim_entropy:
                H_q_hat += q.log_prob(ζ).sum() / num_draws
        return E_ln_joint - H_q_hat

    t, i = -time(), 0
    elbo_hats = []
    smoothed_objective = -elbo_hat().data
    for i in range(max_iters):
        optimizer.zero_grad()
        objective = -elbo_hat()
        objective.backward()
        if torch.isnan(objective.data):
            raise Exception("Infinite objective; cannot continue.")
        optimizer.step()
        elbo_hats.append(-objective.data)
        smoothed_objective = λ * objective.data + (1. - λ) * smoothed_objective
        if not i & (i - 1):
            qprint(f"{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}")
        if stop_heur.early_stop(-objective.data):
            qprint("Stopping heuristic criterion satisfied")
            break
    else:
        qprint("WARNING: maximum iterations reached.")
    t += time()
    qprint(
        f"{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}\n"
        f"Completed {i+1} iterations in {t:.1f}s @ {(i+1)/(t+1e-10):.2f} i/s.\n"
        f"{_DIVIDER}"
    )
    q = Normal(u.detach(), torch.exp(omega.detach() / 2))
    return model.result_type(model, elbo_hats, y, q)


def _header(
    inftype, optimizer, stochastic_entropy, stop_heur, model_name, num_draws, λ
):
    if stochastic_entropy:
        title = f"{inftype} SGVB Inference"
    else:
        title = f"{inftype} ADVI"
    lines = [_DIVIDER, f"{title}: {model_name}"]
    lines += [
        f"  - Estimating elbo with M={num_draws};",
        f"  - {str(stop_heur)}",
        f"  - {type(optimizer).__name__} optimizer with param groups:",
    ]
    for i, pg in enumerate(optimizer.param_groups):
        desc = ", ".join(f"{k}={v}" for k, v in pg.items() if k != "params")
        lines.append(f"    group {i}. {desc}")
    lines.append(f"  - Displayed loss is smoothed with λ={λ}")
    lines.append(_DIVIDER)
    return "\n".join(lines)
