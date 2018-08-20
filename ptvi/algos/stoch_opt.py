import math
import torch
from time import time

import pandas as pd

from ptvi import Model, SupGrowthStoppingHeuristic, PointEstimateTracer


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
        ζ = torch.tensor(ζ0, requires_grad=True)
    else:
        ζ = torch.zeros(model.d, requires_grad=True)
    optimizer = opt_type([ζ], **kwargs)
    stop_heur = stop_heur or SupGrowthStoppingHeuristic()

    qprint(_header(optimizer, stop_heur, model, λ))
    t, losses, smooth_loss = -time(), [], None
    for i in range(int(max_iters)):

        if tracer is not None:
            tracer.append(ζ)

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


def _header(optimizer, stop_heur, model, λ):
    lines = [
        _DIVIDER,
        f"Stochastic optimization for {model.name}",
        f"  - Searching for point estimates only",
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
            estimates.append(float(self.ζ[index : index + p.dimension]))
            index += p.dimension
        cols = {"estimate": estimates}
        if true is not None:
            cols["true"] = [true.get(n, None) for n in names]
        return pd.DataFrame(cols, index=names)
