import collections
import os
import pickle
from typing import List
import torch
from torch.distributions import (
    Distribution,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ptvi.params import TransformedModelParameter, ModelParameter, LocalParameter



class MVNPosterior(object):
    """Base class for representing model results."""

    def __init__(
        self, model: "Model", elbo_hats: List[float], y: torch.Tensor, q: Distribution
    ):
        self.elbo_hats, self.model, self.y, self.q = elbo_hats, model, y, q
        self.input_length = len(y)

        sds = torch.sqrt(self.q.variance)
        for p in model.params:
            setattr(self, p.prior_name, getattr(model, p.prior_name))
            if p.dimension > 1:
                continue
            # construct marginals in optimization space
            if isinstance(p, TransformedModelParameter):
                tfm_post_marg = Normal(self.q.loc[p.index], sds[p.index])
                setattr(self, p.tfm_post_marg_name, tfm_post_marg)

                tfm_prior = getattr(model, p.tfm_prior_name)
                setattr(self, p.tfm_prior_name, tfm_prior)
                tfm = getattr(model, p.tfm_name)
                setattr(self, p.tfm_name, tfm)
                post_marg = TransformedDistribution(tfm_post_marg, tfm.inv)
                setattr(self, p.post_marg_name, post_marg)
            else:
                post_marg = Normal(self.q.loc[p.index], sds[p.index])
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
        i, dropped = 0, 0
        while i < N:
            ζ = self.q.sample()
            try:
                if self.model.has_observation_error:
                    paths[i, :] = self.model.sample_observed(
                        ζ=ζ, y=self.y, fc_steps=fc_steps
                    )
                else:
                    _τ = self.input_length
                    paths[i, :_τ] = self.y
                    if fc_steps > 0:
                        paths[i, _τ:] = self.model.forecast(
                            ζ, self.y, fc_steps=fc_steps
                        )
            except Exception as e:
                # the ζ we sampled blew stuff up - drop this observation
                dropped += 1
                if dropped > 20:
                    # prevent infinite loop, since N is finite.
                    raise e
                continue
            i += 1
        return paths

    def sample_latent_paths(self, N=100, fc_steps=0):
        """Sample N latent paths from the model, forecasting fc_steps additional steps.

        Args:
            N:        number of paths to sample
            fc_steps: number of steps to project forward

        Returns:
            Nx(τ+fc_steps) tensor of sample paths
        """
        paths = torch.empty((N, self.input_length + fc_steps))
        ζs = self.q.sample((N,))
        for i in range(N):
            ζ = ζs[i]
            paths[i, :] = self.model.sample_unobserved(ζ=ζ, y=self.y, fc_steps=fc_steps)
        return paths

    def summary(self, true=None):
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
        cols = {"mean": means, "sd": sds}
        if true is not None:
            cols["true"] = [true.get(n, None) for n in names]
        return pd.DataFrame(cols, index=names)

    def plot_elbos(self, skip=0):
        xs = range(skip, len(self.elbo_hats))
        plt.plot(xs, self.elbo_hats[skip:])
        plt.title(r"$\hat L$ by iteration")

    def plot_latent(self, **true_vals):
        true_vals = true_vals or {}
        locals = [p for p in self.model.params if isinstance(p, LocalParameter)]
        if len(locals) == 0:
            raise Exception("No local variables")
        fig, axes = plt.subplots(nrows=len(locals), ncols=1)
        if not isinstance(axes, collections.Iterable):
            axes = [axes]
        for ax, p in zip(axes, locals):
            zs = self.q.mean[p.index : p.index + p.dimension].cpu().numpy()
            xs = torch.arange(len(zs)).cpu().numpy()
            vars = self.q.variance[p.index : p.index + p.dimension]
            sds = torch.sqrt(vars).cpu().numpy()
            if p.name in true_vals:
                true = true_vals[p.name]
                if hasattr(true, "numpy"):
                    true = true.cpu().numpy()
                plt.plot(xs, true, label=p.name)
            plt.plot(xs, zs, label=f"$E[{p.name} | data]$")
            plt.fill_between(
                xs, zs - sds, zs + sds, label=r"$\pm$ 1 SD", color="blue", alpha=0.1
            )
            plt.title(f"Local variable - {p.name}")
            plt.legend()
        plt.tight_layout()

    def plot_data(self):
        plt.plot(self.y.cpu().numpy(), label="data")
        plt.title("Data")

    def plot_marg_post(
        self, variable: str, suffix="", true_val: float = None, plot_prior=True
    ):
        """Plot marginal posterior distribution, prior, and optionally the
        true value.
        """
        post = getattr(self, f"{variable}_post_marg")
        prior = getattr(self, f"{variable}_prior")
        # figure out range by sampling from posterior
        variates = post.sample((100,))
        a, b = min(variates), max(variates)
        if true_val:
            a, b = min(a, true_val), max(b, true_val)
        xs = torch.linspace(
            a - (b - a) / 4.,
            b + (b - a) / 4.,
            500,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        xs_np = xs.cpu().numpy()

        def plotpdf(p, label=""):
            ys = torch.exp(p.log_prob(xs))
            ys[torch.isnan(ys)] = 0
            plt.plot(xs_np, ys.cpu().numpy(), label=label)

        if plot_prior:
            plotpdf(prior, label=f"$p({variable}{suffix})$")
        plotpdf(post, label=f"$p({variable}{suffix} | y)$")
        if true_val is not None:
            plt.axvline(true_val, label=f"${variable}{suffix}$", linestyle="--")
        plt.legend()

    def plot_global_marginals(self, cols=2, **true_vals):
        import math

        params = [p for p in self.model.params if not isinstance(p, LocalParameter)]
        rows = round(math.ceil(len(params) / cols))
        for r in range(rows):
            for c in range(cols):
                if r * cols + c >= len(params):
                    break
                plt.subplot(rows, cols, r * cols + c + 1)
                p = params[r * cols + c]
                self.plot_marg_post(p.name, true_val=true_vals.get(p.name, None))
        plt.tight_layout()

    def plot_sample_paths(self, N=50, fc_steps=0, true_y=None):
        paths = self.sample_paths(N, fc_steps=fc_steps)
        xs, fxs = range(self.input_length), range(self.input_length + fc_steps)
        for i in range(N):
            plt.plot(fxs, paths[i, :].cpu().numpy(), linewidth=0.5, alpha=0.5)
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color="black")
            plt.title(f"{N} posterior samples and {fc_steps}-step forecast")
        else:
            plt.title(f"{N} posterior samples")
        if true_y is not None:
            plt.plot(xs, true_y.cpu().numpy(), color="black", linewidth=2, label="y")
            plt.legend()

    def plot_pred_ci(
        self, N: int = 100, α: float = 0.05, true_y=None, fc_steps: int = 0
    ):
        paths = self.sample_paths(N, fc_steps=fc_steps)
        ci_bands = np.empty([self.input_length + fc_steps, 2])
        fxs, xs = range(self.input_length + fc_steps), range(self.input_length)
        perc = 100 * np.array([α * 0.5, 1. - α * 0.5])
        for t in fxs:
            ci_bands[t, :] = np.percentile(paths[:, t], q=perc)
        plt.fill_between(
            fxs, ci_bands[:, 0], ci_bands[:, 1], alpha=0.5, label=f"{(1-α)*100:.0f}% CI"
        )
        if true_y is not None:
            plt.plot(xs, true_y.cpu().numpy(), color="black", linewidth=2, label="y")
            plt.legend()
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color="black")
            plt.title(
                f"Posterior credible interval and " f"{fc_steps}-step-ahead forecast"
            )
        else:
            plt.title(f"Posterior credible interval")

    def save(self, filename):
        """Save this approximate posterior to disk."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load an apprxoimate posterior from disk."""
        with open(filename, "rb") as f:
            return pickle.load(f)
