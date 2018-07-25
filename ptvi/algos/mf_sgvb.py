import torch
from time import time
from torch.optim import Adadelta
from torch.distributions import Normal

from ptvi import Model, StoppingHeuristic, NoImprovementStoppingHeuristic
from ptvi.algos.sgvb import SGVBResult


_DIVIDER = "―"*80


def mf_sgvb(model: Model,
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
    omega0 = torch.tensor(L0) if L0 is not None else torch.zeros(model.d)

    u = torch.tensor(u0, requires_grad=True)
    omega = torch.tensor(omega0, requires_grad=True)

    optimizer = (optimizer_type or Adadelta)([u, omega], **opt_params)

    def qprint(s):
        if not quiet: print(s)

    qprint(header(optimizer, sim_entropy, stop_heur, model.name, num_draws, λ))

    def elbo_hat():
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        q = Normal(loc=u, scale=torch.exp(omega/2))
        if not sim_entropy:
            H_q_hat = q.entropy().sum()
        for _ in range(num_draws):
            ζ = u + torch.exp(omega/2) * torch.randn((model.d,)) # reparam trick
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
    q = Normal(u.detach(), torch.exp(omega.detach()/2))
    return SGVBResult(model, elbo_hats, y, q)


def header(optimizer, stochastic_entropy, stop_heur, model_name, num_draws, λ):
    if stochastic_entropy:
        title = 'Mean-field SGVB Inference'
    else:
        title = 'Mean-field ADVI'
    lines = [_DIVIDER, f"{title}: {model_name}:"]
    lines += [f"  - Estimating elbo with M={num_draws};",
              f"  - {str(stop_heur)}",
              f'  - {type(optimizer).__name__} optimizer with param groups:']
    for i, pg in enumerate(optimizer.param_groups):
        desc = ', '.join(f'{k}={v}' for k, v in pg.items() if k != 'params')
        lines.append(f'    group {i}. {desc}')
    lines.append(f'  - Displayed loss is smoothed with λ={λ}')
    lines.append(_DIVIDER)
    return '\n'.join(lines)

