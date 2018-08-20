import math
import torch
from torch.distributions import MultivariateNormal
from time import time

import pandas as pd

from ptvi import Model, MVNPosterior, SupGrowthStoppingHeuristic


_DIVIDER = "―" * 80


def map(
    model: Model, y: torch.Tensor, ζ0=None, max_iters=20, ε=1e-4, quiet=False, **kwargs
):
    """Compute the maximum a postiori (MAP) by maximizing the log joint
    function with respect to the parameter ζ (in optimization space).

    Call self.unpack() to convert parameters in natural coordinates.
    """

    def qprint(s):
        if not quiet:
            print(s)

    qprint(f"{_DIVIDER}\nStochastic optimization: {model.name}\n{_DIVIDER}")
    if ζ0 is not None:
        ζ = torch.tensor(ζ0, requires_grad=True)
    else:
        ζ = torch.zeros(model.d, requires_grad=True)
    optimizer = torch.optim.LBFGS([ζ], **kwargs)
    last_loss, t, losses = None, -time(), []
    for i in range(max_iters):

        def closure():
            optimizer.zero_grad()
            loss = -model.ln_joint(y, ζ)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        qprint(f"{i:8d}. log joint = {-loss:.4f}")
        if math.isnan(loss):
            raise Exception("Non-finite loss encountered.")
        elif last_loss and last_loss < loss + ε:
            qprint("Convergence criterion met.")
            break
        losses.append(loss)
        last_loss = loss
    else:
        qprint("WARNING: maximum iterations reached.")
    qprint(f"{i:8d}. log joint = {-loss:.4f}")
    t += time()
    qprint(f"Completed {i+1:d} iterations in {t:.2f}s @ {(i+1)/t:.2f} i/s.")
    qprint(_DIVIDER)
    return MAPResult(model=model, y=y, ζ=ζ.detach(), losses=losses)


class MAPResult(MVNPosterior):
    def __init__(self, model, y, ζ, losses):
        self.model, self.y, self.ζ = model, y, ζ
        u, L = self.initial_conditions()
        q = MultivariateNormal(u, scale_tril=L)
        super().__init__(model, losses, y, q)

    # https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270#post_3
    def ln_joint_grad_hessian(self):
        z = torch.tensor(self.ζ, requires_grad=True)
        lj = self.model.ln_joint(self.y, z)
        grad = torch.autograd.grad(lj, z, create_graph=True)
        cnt = 0
        for g in grad:
            g_vector = (
                g.contiguous().view(-1)
                if cnt == 0
                else torch.cat([g_vector, g.contiguous().view(-1)])
            )
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], z, create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = (
                    g.contiguous().view(-1)
                    if cnt == 0
                    else torch.cat([g2, g.contiguous().view(-1)])
                )
                cnt = 1
            hessian[idx] = g2
        return grad[0].detach(), hessian.detach()

    def initial_conditions(self, **kwargs):
        """Hacky initial conditions. MAP for initial guess, block-diagonal
        covariance function.
        """
        mask = torch.zeros((self.model.d, self.model.d))
        index = 0
        for p in self.model.params:
            mask[index : index + p.dimension, index : index + p.dimension] = 1
            index += p.dimension
        _, H = self.ln_joint_grad_hessian()
        nHinv = torch.inverse(-mask * H)
        Lv = torch.potrf(nHinv, upper=False)
        return self.ζ, Lv


def stoch_opt(
    model: Model,
    y: torch.Tensor,
    ζ0=None,
    max_iters=20,
    ε=1e-4,
    quiet=False,
    opt_type=None,
    stop_heur=None,
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
    stop_heur = stop_heur or SupGrowthStoppingHeuristic()

    qprint(f"{_DIVIDER}\nMAP inference with L-BFGS: {model.name}\n{_DIVIDER}")
    if ζ0 is not None:
        ζ = torch.tensor(ζ0, requires_grad=True)
    else:
        ζ = torch.zeros(model.d, requires_grad=True)
    optimizer = opt_type([ζ], **kwargs)
    last_loss, t, losses = None, -time(), []
    for i in range(max_iters):

        def closure():
            optimizer.zero_grad()
            loss = -model.ln_joint(y, ζ)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        qprint(f"{i:8d}. loss = {-loss:.4f}")
        if math.isnan(loss):
            raise Exception("Non-finite loss encountered.")
        elif last_loss and last_loss < loss + ε:
            qprint("Convergence criterion met.")
            break
        losses.append(loss)
        last_loss = loss
    else:
        qprint("WARNING: maximum iterations reached.")
    qprint(f"{i:8d}. log joint = {-loss:.4f}")
    t += time()
    qprint(f"Completed {i+1:d} iterations in {t:.2f}s @ {(i+1)/t:.2f} i/s.")
    qprint(_DIVIDER)
    return StochOptResult(model=model, y=y, ζ=ζ.detach(), losses=losses)


class StochOptResult(object):
    def __init__(self, model, y, ζ, losses):
        self.model, self.y, self.ζ = model, y, ζ
        super().__init__(model, losses, y, q)

    def plot_elbos(self, skip=0):
        xs = range(skip, len(self.losses))
        plt.plot(xs, self.losses[skip:])
        plt.title(r"Estimated loss by iteration")

    def summary(self, true=None):
        """Return a pandas data frame summarizing model parameters"""
        # transform and simulate from marginal transformed parameters
        names, estimates = [], []
        index = 0
        for p in self.model.params:
            if p.dimension > 1 and post is not None:
                continue
            names.append(p.name)
            estimates.append(float(self.ζ[index:index+p.dimension]))
            index += p.dimension
        cols = {"estimate": estimates}
        if true is not None:
            cols["true"] = [true.get(n, None) for n in names]
        return pd.DataFrame(cols, index=names)
