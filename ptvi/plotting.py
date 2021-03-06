from numbers import Number
import torch
from torch.distributions import Distribution


def plot_dens(dists, a, b):
    """Plot a distribution or list of distributions."""
    import matplotlib.pyplot as plt

    xs = torch.linspace(a, b, 500)

    def plotpdf(p, label=""):
        if isinstance(p, Distribution):
            ys = torch.exp(p.log_prob(xs))
            ys[torch.isnan(ys)] = 0
            plt.plot(xs.cpu().numpy(), ys.cpu().numpy(), label=label)
        elif isinstance(p, Number):
            plt.axvline(p, linestyle="--", label=label)

    if isinstance(dists, dict):
        for lbl, p in dists.items():
            plotpdf(p, lbl)
        plt.legend()
    else:
        plotpdf(dists)
