import math
import torch
from time import time

import pandas as pd

from ptvi import (
    Model,
    SupGrowthStoppingHeuristic,
    PointEstimateTracer,
    DualPointEstimateTracer,
    FilteredStateSpaceModelFreeProposal,
)
from ptvi.params import TransformedModelParameter


_DIVIDER = "―" * 80


def stoch_opt(
    model: Model,
    y: torch.Tensor,
    ζ0=None,
    max_iters=2 ** 10,
    λ=0.1,
    quiet=False,
    opt_type=None,
    stop_heur=None,
    tracer: PointEstimateTracer = None,
    **kwargs,
):
    """Use stochastic optimization to compute the maximum a postiori (MAP) by maximizing
    the log joint function with respect to the parameter ζ (in optimization space).

    Call self.unpack() to convert parameters in natural coordinates.
    """

    def qprint(s):
        if not quiet:
            print(s)

    opt_type = opt_type or torch.optim.Adam
    if ζ0 is not None:
        ζ = torch.tensor(ζ0, requires_grad=True, dtype=model.dtype, device=model.device)
    else:
        ζ = torch.zeros(
            model.d, requires_grad=True, dtype=model.dtype, device=model.device
        )
    optimizer = opt_type([ζ], **kwargs)
    stop_heur = stop_heur or SupGrowthStoppingHeuristic()

    qprint(_header(optimizer, stop_heur, model, λ))
    t, losses, smooth_loss = -time(), [], None
    for i in range(int(max_iters)):

        def closure():
            optimizer.zero_grad()
            loss = -model.ln_joint(y, ζ)
            loss.backward()
            return loss

        neg_loss = optimizer.step(closure)
        loss_d = -neg_loss.detach()
        smooth_loss = (
            loss_d if smooth_loss is None else λ * loss_d + smooth_loss * (1 - λ)
        )
        if tracer is not None:
            tracer.append(ζ, loss_d)
        if not i & (i - 1):
            qprint(f"{i:8d}. smoothed stochastic loss = {smooth_loss:.1f}")
        if math.isnan(loss_d):
            raise Exception("Non-finite neg_loss encountered.")
        elif stop_heur.early_stop(loss_d):
            qprint("Convergence criterion met.")
            break
        losses.append(loss_d)
    else:
        qprint("WARNING: Maximum iterations reached.")
    qprint(f"{i:8d}. (unsmoothed) stochastic loss = {loss_d:.1f}")
    t += time()
    qprint(f"Completed {i+1:d} iterations in {t:.2f}s @ {(i+1)/t:.2f} i/s.")
    qprint(_DIVIDER)
    return StochOptResult(model=model, y=y, ζ=ζ.detach(), objectives=losses)


def dual_stoch_opt(
    model: FilteredStateSpaceModelFreeProposal,
    y: torch.Tensor,
    ζ0=None,
    η0=None,
    max_iters=2 ** 10,
    λ=0.1,
    quiet=False,
    optimizer_type=None,
    stop_heur=None,
    tracer: DualPointEstimateTracer = None,
    **kwargs,
):
    """Use stochastic optimization to compute the maximum a postiori (MAP) by maximizing
    the log joint function with respect to the parameter ζ (in optimization space).

    Call self.unpack() to convert parameters in natural coordinates.
    """

    def qprint(s):
        if not quiet:
            print(s)

    optimizer_type = optimizer_type or torch.optim.Adam

    ζ0 = ζ0 if ζ0 is not None else torch.zeros(model.md)
    ζ = torch.tensor(ζ0, requires_grad=True, dtype=model.dtype, device=model.device)

    η0 = η0 if η0 is not None else torch.zeros(model.pd)
    η = torch.tensor(η0, requires_grad=True, dtype=model.dtype, device=model.device)

    # one optimizer for each parameter, ie for model and proposals
    model_optimizer = optimizer_type([ζ], **kwargs)
    proposal_optimizer = optimizer_type([η], **kwargs)

    stop_heur = stop_heur or SupGrowthStoppingHeuristic()

    qprint(_header(model_optimizer, stop_heur, model, λ))
    t, losses, smooth_loss = -time(), [], None

    for i in range(int(max_iters)):

        def model_closure():
            model_optimizer.zero_grad()
            loss = -model.ln_joint(y, ζ, η)
            loss.backward()
            return loss

        neg_loss = model_optimizer.step(model_closure)
        loss_d = -neg_loss.detach()
        smooth_loss = (
            loss_d if smooth_loss is None else λ * loss_d + smooth_loss * (1 - λ)
        )

        def proposal_closure():
            proposal_optimizer.zero_grad()
            loss = -model.ln_joint(y, ζ, η)
            loss.backward()
            return loss

        proposal_optimizer.step(proposal_closure)

        if tracer is not None:
            tracer.append(ζ, η, loss_d)
        if not i & (i - 1):
            qprint(f"{i:8d}. smoothed stochastic loss = {smooth_loss:.1f}")
        if math.isnan(loss_d):
            raise Exception("Non-finite neg_loss encountered.")
        elif stop_heur.early_stop(loss_d):
            qprint("Convergence criterion met.")
            break
        losses.append(loss_d)
    else:
        qprint("WARNING: Maximum iterations reached.")
    qprint(f"{i:8d}. (unsmoothed) stochastic loss = {loss_d:.1f}")
    t += time()
    qprint(f"Completed {i+1:d} iterations in {t:.2f}s @ {(i+1)/t:.2f} i/s.")
    qprint(_DIVIDER)
    param = torch.cat([ζ.detach(), η.detach()])  # todo: .cpu()
    return StochOptResult(model=model, y=y, ζ=param, objectives=losses)


def _header(optimizer, stop_heur, model, λ):
    lines = [
        _DIVIDER,
        f"Stochastic optimization for {model.name}",
        f"  - Searching for point estimates only",
        f"  - Using {model.dtype} precision on {model.device}",
        f"  - {str(stop_heur)}",
        f"  - {type(optimizer).__name__} optimizer with param groups:",
    ]
    for i, pg in enumerate(optimizer.param_groups):
        desc = ", ".join(f"{k}={v}" for k, v in pg.items() if k != "params")
        lines.append(f"    group {i}. {desc}")
    lines.append(f"  - Displayed loss is smoothed with λ={λ}")
    lines.append(_DIVIDER)
    return "\n".join(lines)


class StochOptResult(object):
    def __init__(self, model, y, ζ, objectives):
        self.model, self.y, self.ζ, self.objectives = model, y, ζ, objectives

    def plot_objectives(self, skip=0):
        import matplotlib.pyplot as plt

        xs = range(skip, len(self.objectives))
        plt.plot(xs, self.objectives[skip:])
        plt.title(r"Estimated objective by iteration")

    def summary(self, true=None):
        """Return a pandas data frame summarizing model parameters"""
        # transform and simulate from marginal transformed parameters
        names, estimates = [], []
        index = 0
        for p in self.model.params:
            if p.dimension > 1:
                continue
            names.append(p.name)
            if isinstance(p, TransformedModelParameter):
                t = p.transform.inv
                estimates.append(float(t(self.ζ[index : index + p.dimension])))
            else:
                estimates.append(float(self.ζ[index : index + p.dimension]))
            index += p.dimension
        cols = {"estimate": estimates}
        if true is not None:
            cols["true"] = [true.get(n, None) for n in names]
        return pd.DataFrame(cols, index=names)
