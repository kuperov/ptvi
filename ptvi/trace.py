import torch

from ptvi.params import TransformedModelParameter
from ptvi import Model, FilteredStateSpaceModelFreeProposal


class PointEstimateTracer(object):
    """Utility for tracing the progress of optimizations. This is just for debugging."""

    def __init__(self, model: Model):
        self.model = model
        self.natural_values, self.unconstrained_values = [], []
        self.natural_varnames, self.unconstrained_varnames = [], []
        self.objectives = []
        self.transforms = []
        index = 0
        for p in model.params:
            if p.dimension != 1:
                continue
            self.natural_varnames.append(p.name)
            if isinstance(p, TransformedModelParameter):
                self.unconstrained_varnames.append(p.transformed_name)
                self.transforms.append(p.transform.inv)
            else:
                self.unconstrained_varnames.append(p.name)
                self.transforms.append(None)
            index += p.dimension

    def append(self, param_value, objective):
        to_add = param_value.detach().clone()
        self.unconstrained_values.append(to_add)
        self.objectives.append(objective)

    def to_unconstrained_array(self):
        return torch.stack(self.unconstrained_values)

    def to_constrained_array(self):
        uarr = self.to_unconstrained_array()
        carr = torch.empty_like(uarr)
        for i in range(uarr.shape[1]):
            t = self.transforms[i]
            if t is None:
                carr[:, i] = uarr[:, i]
            else:
                for j in range(carr.shape[0]):
                    carr[j, i] = t(uarr[j, i])  # slow but idc
        return carr

    def plot(self, true={}, **fig_kw):
        import matplotlib.pyplot as plt

        u_values = self.to_unconstrained_array()
        c_values = self.to_constrained_array()
        n = len(self.natural_varnames)
        fig, axes = plt.subplots(n, 2, **fig_kw)
        for i in range(n):
            nat = self.natural_varnames[i]
            axes[i, 0].plot(c_values[:, i].cpu().numpy(), label=f"$\\hat {nat}$")
            if nat in true:
                axes[i, 0].axhline(y=true[nat], linestyle="--", label=f"${nat}_0$")
            axes[i, 0].legend()
            unc = self.unconstrained_varnames[i]
            axes[i, 1].plot(u_values[:, i].cpu().numpy(), label=f"$\\hat {unc}$")
            if nat in true:
                if self.transforms[i] is not None:
                    tfm_true = self.transforms[i].inv(torch.tensor(true[nat])).numpy()
                else:
                    tfm_true = true[nat]
                axes[i, 1].axhline(y=tfm_true, linestyle="--", label=f"${unc}_0$")
            axes[i, 1].legend()
        plt.suptitle("Optimization trace - point estimates")

    def plot_objectives(self, skip=0):
        import matplotlib.pyplot as plt

        xs = range(skip, len(self.objectives))
        plt.plot(xs, self.objectives[skip:])
        plt.title(r"Estimated objective by iteration")


class DualPointEstimateTracer(PointEstimateTracer):
    """Tracer for dual optimization algorithms."""

    def __init__(self, model: FilteredStateSpaceModelFreeProposal):
        super().__init__(model)

    def append(self, ζ, η, objective):
        super().append(torch.cat([ζ, η]), objective)
