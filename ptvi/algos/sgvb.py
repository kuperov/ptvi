import torch
from time import time
from torch.optim import Adadelta
from torch.distributions import MultivariateNormal, Normal

from ptvi import (
    Model,
    StoppingHeuristic,
    SupGrowthStoppingHeuristic,
    FilteredStateSpaceModelFreeProposal,
    PointEstimateTracer,
    DualPointEstimateTracer,
)


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
    tracer: PointEstimateTracer = None,
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

    # this dance seems to be necessary to silence ugly torch warnings
    u = (
        torch.empty_like(u0, device=model.device, dtype=model.dtype)
        .copy_(u0)
        .requires_grad_(True)
    )
    L = (
        torch.empty_like(L0, device=model.device, dtype=model.dtype)
        .copy_(L0)
        .requires_grad_(True)
    )

    optimizer = (optimizer_type or Adadelta)([u, L], **opt_params)

    def qprint(s):
        if not quiet:
            print(s)

    qprint(
        _header("Structured", optimizer, sim_entropy, stop_heur, model, num_draws, λ)
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
            ε = torch.randn((model.d,), device=model.device, dtype=model.dtype)
            ζ = u + trL @ ε  # reparam trick
            E_ln_joint += model.ln_joint(y, ζ) / num_draws
            if sim_entropy:
                H_q_hat += q.log_prob(ζ) / num_draws
        return E_ln_joint - H_q_hat

    t, i = -time(), 0
    elbo_hats = []
    smoothed_objective = -elbo_hat().detach()
    for i in range(max_iters):
        optimizer.zero_grad()
        objective = -elbo_hat()
        objective.backward()
        if torch.isnan(objective.detach()):
            raise Exception("Infinite objective; cannot continue.")
        optimizer.step()
        elbo_hats.append(-objective.detach())
        smoothed_objective = λ * objective.detach() + (1. - λ) * smoothed_objective
        if not i & (i - 1):
            qprint(f"{i: 8d}. smoothed elbo ={float(-smoothed_objective):8.2f}")
        if stop_heur.early_stop(-objective.detach()):
            qprint("Stopping heuristic criterion satisfied")
            break
        if tracer is not None:
            tracer.append(u.detach().clone(), -objective.detach())
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
    tracer: PointEstimateTracer = None,
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

    u = torch.tensor(u0, requires_grad=True, dtype=model.dtype, device=model.device)
    omega = torch.tensor(
        omega0, requires_grad=True, dtype=model.dtype, device=model.device
    )

    optimizer = (optimizer_type or Adadelta)([u, omega], **opt_params)

    def qprint(s):
        if not quiet:
            print(s)

    qprint(
        _header("Mean-field", optimizer, sim_entropy, stop_heur, model, num_draws, λ)
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
            ε = torch.randn((model.d,), device=model.device, dtype=model.dtype)
            ζ = u + torch.exp(omega / 2) * ε  # reparam trick
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
        if tracer is not None:
            tracer.append(u.detach().clone(), -objective.detach())
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


def dual_sgvb(
    model: FilteredStateSpaceModelFreeProposal,
    y: torch.Tensor,
    um0: torch.Tensor = None,
    Lm0: torch.Tensor = None,
    up0: torch.Tensor = None,
    Lp0: torch.Tensor = None,
    max_iters: int = 2 ** 20,
    sim_entropy: bool = True,
    stop_heur: StoppingHeuristic = None,
    num_draws: int = 1,
    model_optimizer_type: type = None,
    proposal_optimizer_type: type = None,
    quiet=False,
    λ=.1,
    model_opt_params=None,
    proposal_opt_params=None,
    tracer: DualPointEstimateTracer = None,
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
    um0 = um0 if um0 is not None else torch.zeros(model.md)
    Lm0 = Lm0 if Lm0 is not None else torch.eye(model.md)

    # dense approximation: q = N(u, LL')
    up0 = up0 if up0 is not None else torch.zeros(model.pd)
    Lp0 = Lp0 if Lp0 is not None else torch.eye(model.pd)

    um = torch.tensor(um0, requires_grad=True, dtype=model.dtype, device=model.device)
    Lm = torch.tensor(Lm0, requires_grad=True, dtype=model.dtype, device=model.device)

    up = torch.tensor(up0, requires_grad=True, dtype=model.dtype, device=model.device)
    Lp = torch.tensor(Lp0, requires_grad=True, dtype=model.dtype, device=model.device)

    model_opt_params = model_opt_params or {}
    proposal_opt_params = proposal_opt_params or {}

    model_optimizer = (model_optimizer_type or Adadelta)([um, Lm], **model_opt_params)
    prop_optimizer = (proposal_optimizer_type or Adadelta)(
        [up, Lp], **proposal_opt_params
    )

    def qprint(s):
        if not quiet:
            print(s)

    qprint(
        _header(
            "Alternating model/proposal structured",
            model_optimizer,
            sim_entropy,
            stop_heur,
            model,
            num_draws,
            λ,
        )
    )

    def model_elbo_hat():
        trL = torch.tril(Lm)
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        if not sim_entropy:
            q = MultivariateNormal(loc=um, scale_tril=trL)
            H_q_hat = -q.entropy()
        else:
            # don't accumulate gradients; see https://arxiv.org/abs/1703.09194
            q = MultivariateNormal(loc=um.detach(), scale_tril=trL.detach())
        for _ in range(num_draws):
            εm = torch.randn((model.md,), device=model.device, dtype=model.dtype)
            ζ = um + trL @ εm  # reparam trick
            εp = torch.randn((model.pd,), device=model.device, dtype=model.dtype)
            η = up.detach() + Lp.detach() @ εp  # reparam trick
            E_ln_joint += model.ln_joint(y, ζ, η) / num_draws
            if sim_entropy:
                H_q_hat += q.log_prob(ζ) / num_draws
        return E_ln_joint - H_q_hat

    def proposal_elbo_hat():
        trL = torch.tril(Lp)
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        if not sim_entropy:
            q = MultivariateNormal(loc=up, scale_tril=trL)
            H_q_hat = -q.entropy()
        else:
            # don't accumulate gradients; see https://arxiv.org/abs/1703.09194
            q = MultivariateNormal(loc=up.detach(), scale_tril=trL.detach())
        for _ in range(num_draws):
            εm = torch.randn((model.md,), device=model.device, dtype=model.dtype)
            ζ = um.detach() + Lm.detach() @ εm
            εp = torch.randn((model.pd,), device=model.device, dtype=model.dtype)
            η = up + trL @ εp  # reparam trick
            E_ln_joint += model.ln_joint(y, ζ, η) / num_draws
            if sim_entropy:
                H_q_hat += q.log_prob(η) / num_draws
        return E_ln_joint - H_q_hat

    t, i = -time(), 0
    model_elbo_hats, proposal_elbo_hats = [], []
    smoothed_model_objective = -model_elbo_hat().detach()
    smoothed_proposal_objective = -proposal_elbo_hat().detach()
    for i in range(max_iters):
        model_optimizer.zero_grad()
        model_objective = -model_elbo_hat()
        model_objective.backward()
        if torch.isnan(model_objective.detach()):
            raise Exception("Infinite model objective; cannot continue.")
        model_optimizer.step()
        model_elbo_hats.append(-model_objective.detach())
        smoothed_model_objective = (
            λ * model_objective.detach() + (1. - λ) * smoothed_model_objective
        )

        prop_optimizer.zero_grad()
        proposal_objective = -proposal_elbo_hat()
        proposal_objective.backward()
        if torch.isnan(proposal_objective.detach()):
            raise Exception("Infinite proposal objective; cannot continue.")
        prop_optimizer.step()
        proposal_elbo_hats.append(-proposal_objective.detach())
        smoothed_proposal_objective = (
            λ * proposal_objective.detach() + (1. - λ) * smoothed_proposal_objective
        )

        if not i & (i - 1):
            qprint(
                f"{i: 8d}. smoothed model elbo ={float(-smoothed_model_objective):8.2f}, "
                f"proposal elbo ={float(-smoothed_proposal_objective):8.2f}"
            )
        if stop_heur.early_stop(-model_objective.data):
            qprint("Stopping heuristic criterion satisfied for model elbo")
            break
        if tracer is not None:
            tracer.append(
                um.detach().clone(), up.detach().clone(), -model_objective.detach()
            )
    else:
        qprint("WARNING: maximum iterations reached.")
    t += time()
    qprint(
        f"{i: 8d}. smoothed model elbo ={float(-smoothed_model_objective):8.2f}, "
        f"proposal elbo ={float(-smoothed_proposal_objective):8.2f}\n"
        f"Completed {i+1} iterations in {t:.1f}s @ {(i+1)/(t+1e-10):.2f} i/s.\n"
        f"{_DIVIDER}"
    )
    u = torch.cat([um.detach(), up.detach()])
    L = torch.cat(
        [
            torch.cat([Lm.detach(), torch.zeros((model.md, model.pd))], dim=1),
            torch.cat([torch.zeros((model.pd, model.md)), Lp.detach()], dim=1),
        ],
        dim=0,
    )
    q = MultivariateNormal(u, scale_tril=L)
    return model.result_type(model=model, elbo_hats=model_elbo_hats, y=y, q=q)


def sgvb_and_stoch_opt(
    model: FilteredStateSpaceModelFreeProposal,
    y: torch.Tensor,
    um0: torch.Tensor = None,
    Lm0: torch.Tensor = None,
    η0: torch.Tensor = None,
    max_iters: int = 2 ** 20,
    sim_entropy: bool = True,
    stop_heur: StoppingHeuristic = None,
    num_draws: int = 1,
    model_optimizer_type: type = None,
    proposal_optimizer_type: type = None,
    quiet=False,
    λ=.1,
    model_opt_params=None,
    proposal_opt_params=None,
    tracer: DualPointEstimateTracer = None,
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
    um0 = um0 if um0 is not None else torch.zeros(model.md)
    Lm0 = Lm0 if Lm0 is not None else torch.eye(model.md)

    # point estimate only
    η0 = η0 if η0 is not None else torch.zeros(model.pd)
    η = torch.tensor(η0, requires_grad=True, dtype=model.dtype, device=model.device)

    um = torch.tensor(um0, requires_grad=True, dtype=model.dtype, device=model.device)
    Lm = torch.tensor(Lm0, requires_grad=True, dtype=model.dtype, device=model.device)

    model_opt_params = model_opt_params or {}
    proposal_opt_params = proposal_opt_params or {}

    model_optimizer = (model_optimizer_type or Adadelta)([um, Lm], **model_opt_params)
    prop_optimizer = (proposal_optimizer_type or Adadelta)([η], **proposal_opt_params)

    def qprint(s):
        if not quiet:
            print(s)

    qprint(
        _header(
            "Alternating model/proposal structured",
            model_optimizer,
            sim_entropy,
            stop_heur,
            model,
            num_draws,
            λ,
        )
    )

    def sim_elbo_hat():
        trL = torch.tril(Lm)
        E_ln_joint, H_q_hat = 0., 0.  # accumulators
        if not sim_entropy:
            q = MultivariateNormal(loc=um, scale_tril=trL)
            H_q_hat = -q.entropy()
        else:
            # don't accumulate gradients; see https://arxiv.org/abs/1703.09194
            q = MultivariateNormal(loc=um.detach(), scale_tril=trL.detach())
        for _ in range(num_draws):
            εm = torch.randn((model.md,), device=model.device, dtype=model.dtype)
            ζ = um + trL @ εm  # reparam trick
            E_ln_joint += model.ln_joint(y, ζ, η) / num_draws
            if sim_entropy:
                H_q_hat += q.log_prob(ζ) / num_draws
        return E_ln_joint - H_q_hat

    t, i = -time(), 0
    model_elbo_hats, proposal_elbo_hats = [], []
    smoothed_model_objective = None
    for i in range(max_iters):
        model_optimizer.zero_grad()
        prop_optimizer.zero_grad()
        model_objective = -sim_elbo_hat()
        model_objective.backward()
        if torch.isnan(model_objective.detach()):
            raise Exception("Infinite model objective; cannot continue.")
        model_optimizer.step()
        prop_optimizer.step()
        model_elbo_hats.append(-model_objective.detach())
        if smoothed_model_objective is not None:
            smoothed_model_objective = (
                λ * model_objective.detach() + (1. - λ) * smoothed_model_objective
            )
        else:
            smoothed_model_objective = model_objective.detach()

        if not i & (i - 1):
            qprint(
                f"{i: 8d}. smoothed model elbo ={float(-smoothed_model_objective):8.2f}"
            )
        if stop_heur.early_stop(-model_objective.data):
            qprint("Stopping heuristic criterion satisfied for model elbo")
            break
        if tracer is not None:
            tracer.append(
                um.detach().clone(), η.detach().clone(), -model_objective.detach()
            )
    else:
        qprint("WARNING: maximum iterations reached.")
    t += time()
    qprint(
        f"{i: 8d}. smoothed model elbo ={float(-smoothed_model_objective):8.2f}\n"
        f"Completed {i+1} iterations in {t:.1f}s @ {(i+1)/(t+1e-10):.2f} i/s.\n"
        f"{_DIVIDER}"
    )
    u = torch.cat([um.detach(), η.detach()])
    L = torch.cat(
        [
            torch.cat([Lm.detach(), torch.zeros((model.md, model.pd))], dim=1),
            torch.cat([torch.zeros((model.pd, model.md)), torch.eye(model.pd)], dim=1),
        ],
        dim=0,
    )
    q = MultivariateNormal(u, scale_tril=L)
    return model.result_type(model=model, elbo_hats=model_elbo_hats, y=y, q=q)


def _header(inftype, optimizer, stochastic_entropy, stop_heur, model, num_draws, λ):
    if stochastic_entropy:
        title = f"{inftype} SGVB Inference"
    else:
        title = f"{inftype} ADVI"
    lines = [
        _DIVIDER,
        f"{title}: {model.name}",
        f"  - Using {model.dtype} precision on {model.device}",
        f"  - Estimating elbo with M={num_draws}",
        f"  - {str(stop_heur)}",
        f"  - {type(optimizer).__name__} optimizer with param groups:",
    ]
    for i, pg in enumerate(optimizer.param_groups):
        desc = ", ".join(f"{k}={v}" for k, v in pg.items() if k != "params")
        lines.append(f"    group {i}. {desc}")
    lines.append(f"  - Displayed loss is smoothed with λ={λ}")
    lines.append(_DIVIDER)
    return "\n".join(lines)
