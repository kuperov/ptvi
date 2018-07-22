from time import time
from typing import List, Dict, Union
from warnings import warn

import torch
from torch.distributions import (
    Distribution, Normal, TransformedDistribution, MultivariateNormal,
    Transform)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptvi import StoppingHeuristic, NoImprovementStoppingHeuristic, plot_dens, Improper


class ModelParameter(object):
    """A parameter in a model.

    Attrs:
        name:            param name, usually a greek letter or short identifier
        prior:           a Distribution object
        parameter_index: index of the *start* of this parameter, when it is
                         stacked in optimizaton space
        dimension:       length of parameter when stacked as a vector
    """
    index = -1
    dimension = 1

    def __init__(self, name: str, prior: Distribution):
        self.name, self.prior = name, prior

    def inferred_name(self, name):
        """Provides a guess of the name for this variable. Accepts name if none
        provided."""
        self.name = self.name or name

    @property
    def prior_name(self):
        return '{}_prior'.format(self.name)

    @property
    def post_marg_name(self):
        return '{}_post_marg'.format(self.name)

    def __str__(self):
        return f"{self.name} with prior {self.prior}"


class TransformedModelParameter(ModelParameter):
    """A parameter that has been transformed to an unrestricted space.

    Attrs:
        name:             param name, usually a greek letter or short word
        prior:            a Distribution object
        transformed_name: name to use in optimization space
        transform:        transformation to apply
        parameter_index:  index of the *start* of this parameter, when it is
                          stacked in optimizaton space
        dimension:        length of parameter when stacked as a vector
        transform_desc:   text description of the transform (e.g. 'log')
    """
    def __init__(self, name: str, prior: Distribution, transformed_name: str,
                 transform: Transform, transform_desc: str=None):
        super().__init__(name, prior)
        self.transformed_name = transformed_name
        self.transform = transform
        self.transform_desc = transform_desc

    def inferred_name(self, name):
        """Provides a guess of the name for this variable. Accepts name if none
        provided."""
        if self.name is None:
            self.name = name
        if self.transformed_name is None:
            tfm_desc = self.transform_desc or 'transformed'
            self.transformed_name = f'{tfm_desc}_{self.name}'

    @property
    def tfm_prior_name(self):
        return '{}_prior'.format(self.transformed_name)

    @property
    def tfm_name(self):
        return '{}_to_{}'.format(self.name, self.transformed_name)

    @property
    def tfm_post_marg_name(self):
        return '{}_post_marg'.format(self.transformed_name)

    def __str__(self):
        return (f"{self.name} with prior {self.prior} transformed to "
                f"{self.transformed_name} by {self.transform}")


class LocalParameter(ModelParameter):
    """A local model parameter.

    For now local parameters are assumed not transformable and dimension 1.
    """
    def __init__(self, name: str, prior: Distribution):
        super().__init__(name, prior)

    def __str__(self):
        return f"{self.name} local parameter with prior {self.prior}"


def global_param(prior: Distribution=None, name: str=None,
                 transform: Union[Transform,str]=None, rename: str=None):
    """Define a scalar global model parameter.

    Args:
        prior: parameter prior, a Distribution object
        name: optional, name for parameter
        transform: optional transformation to apply (its domain should be an
                   unconstrained space)
        rename: optional, name of parameter in unconstrained space
    """
    if prior is None:
        prior = Improper()
    if rename is not None and transform is None:
        raise Exception('rename requires a transform')
    if transform is None:
        return ModelParameter(name=name, prior=prior)
    transform_desc = 'transformed'
    if isinstance(transform, str):
        transform_desc = transform
        if transform == 'log':
            transform = torch.distributions.ExpTransform().inv
            if rename is None and name is not None:
                rename = f'log{name}'
        elif transform == 'exp':
            raise Exception("Use 'log' to constrain parameters > 0")
        elif transform == 'logit':
            transform = torch.distributions.SigmoidTransform().inv
            if rename is None and name is not None:
                rename = f'logit{name}'
        elif transform == 'sigmoid':
            raise Exception("Use 'logit' to constrain parameters to (0, 1)")
        else:
            raise Exception(f'Unknown transform {transform}')
    if rename is None and name is not None:
        rename = f'{transform_desc}_{name}'
    return TransformedModelParameter(
        name=name, prior=prior, transform=transform, transformed_name=rename,
        transform_desc=transform_desc)


def local_param(prior: Distribution=None, name: str=None):
    """Define a local (latent) parameter.

    Args:
        prior: parameter prior, a Distribution object
        name: optional, name for parameter
    """
    if prior is None:
        prior = Improper()
    return LocalParameter(name=name, prior=prior)


class VIResult(object):
    """Base class for representing model results."""

    def __init__(self,
                 model: 'VIModel',
                 elbo_hats: List[float],
                 y=None):
        self.elbo_hats, self.model, self.y = elbo_hats, model, y

        self.u, self.L = model.u.data, model.L.data
        # posteriors are transformed from normal distributions
        self.q = MultivariateNormal(self.u, scale_tril=self.L)

        sds = torch.sqrt(self.q.variance)
        for p in model.params:
            setattr(self, p.prior_name, getattr(model, p.prior_name))
            if p.dimension > 1:
                continue
            # construct marginals in optimization space
            if isinstance(p, TransformedModelParameter):
                tfm_post_marg = Normal(self.u[p.index], sds[p.index])
                setattr(self, p.tfm_post_marg_name, tfm_post_marg)

                tfm_prior = getattr(model, p.tfm_prior_name)
                setattr(self, p.tfm_prior_name, tfm_prior)
                tfm = getattr(model, p.tfm_name)
                setattr(self, p.tfm_name, tfm)
                post_marg = TransformedDistribution(tfm_post_marg, tfm.inv)
                setattr(self, p.post_marg_name, post_marg)
            else:
                post_marg = Normal(self.u[p.index], sds[p.index])
                setattr(self, p.post_marg_name, post_marg)

    def summary(self):
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
        return pd.DataFrame({'mean': means, 'sd': sds}, index=names)

    def plot_elbos(self):
        plt.figure()
        plt.plot(self.elbo_hats)
        plt.title(r'$\hat L$ by iteration')

    def plot_latent(self, true_z=None, include_data=False):
        plt.figure()
        zs = self.q.mean[:-4].numpy()
        xs = torch.arange(len(zs)).numpy()
        sds = torch.sqrt(self.q.variance).numpy()[:-4]
        if include_data:
            plt.subplot(211)
            plt.plot(xs, self.y.numpy())
            plt.title('Observed data')
            plt.subplot(212)
        plt.plot(xs, zs, label=r'$E[z_{1:\tau} | y]$')
        plt.fill_between(xs, zs - sds, zs + sds, label=r'$\pm$ 1 SD',
                         color='blue', alpha=0.1)
        plt.title('Latent state')
        if true_z is not None:
            if isinstance(true_z, torch.Tensor):
                true_z = true_z.numpy()
            plt.plot(xs, true_z, label=r'$z_{0,1:\tau}$')
        plt.legend()
        plt.tight_layout()

    def plot_data(self):
        plt.figure()
        plt.plot(self.y.numpy(), label='data')
        plt.title('Data')

    def plot_marg_post(self, variable:str, suffix='', true_val:float=None,
                       new_figure=True, plot_prior=True):
        """Plot marginal posterior distribution, prior, and optionally the
        true value.
        """
        post = getattr(self, f'{variable}_post_marg')
        prior = getattr(self, f'{variable}_prior')
        # figure out range by sampling from posterior
        variates = post.sample((100,))
        a, b = min(variates), max(variates)
        xs = torch.linspace(a-(b-a)/4., b+(b-a)/4., 500)

        def plotpdf(p, label=''):
            ys = torch.exp(p.log_prob(xs))
            ys[torch.isnan(ys)] = 0
            plt.plot(xs.numpy(), ys.numpy(), label=label)

        if plot_prior:
            plotpdf(prior, label=f'$p({variable}{suffix})$')
        plotpdf(post, label=f'$p({variable}{suffix} | y)$')
        if true_val is not None:
            plt.axvline(true_val, label=f'${variable}{suffix}$', linestyle='--')
        plt.legend()


class VITimeSeriesResult(VIResult):
    """Time series result object, which adds plotting functions etc relevant
    for TS models.
    """
    def __init__(self, model: 'VITimeSeriesModel', elbo_hats: List[float], y):
        self.input_length = len(y)
        super().__init__(model=model, elbo_hats=elbo_hats, y=y)

    def plot_sample_paths(self, N=50, fc_steps=0, true_y=None):
        paths = self.model.sample_paths(N, fc_steps=fc_steps)
        xs, fxs = range(self.input_length), range(self.input_length+fc_steps)
        for i in range(N):
            plt.plot(fxs, paths[i, :].numpy(), linewidth=0.5, alpha=0.5)
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color='black')
            plt.title(f'{N} posterior samples and {fc_steps}-step forecast')
        else:
            plt.title(f'{N} posterior samples')
        if true_y is not None:
            plt.plot(xs, true_y.numpy(), color='black', linewidth=2, label='y')
            plt.legend()

    def plot_pred_ci(self, N:int=100, α:float=0.05, true_y=None,
                     fc_steps:int=0):
        paths = self.model.sample_paths(N, fc_steps=fc_steps)
        ci_bands = np.empty([self.input_length+fc_steps, 2])
        fxs, xs = range(self.input_length+fc_steps), range(self.input_length)
        perc = 100 * np.array([α * 0.5, 1. - α * 0.5])
        for t in fxs:
            ci_bands[t, :] = np.percentile(paths[:, t], q=perc)
        plt.fill_between(fxs, ci_bands[:, 0], ci_bands[:, 1], alpha=0.5,
                         label=f'{(1-α)*100:.0f}% CI')
        if true_y is not None:
            plt.plot(xs, true_y.numpy(), color='black', linewidth=2, label='y')
            plt.legend()
        if fc_steps > 0:
            plt.axvline(x=self.input_length, color='black')
            plt.title(f'Posterior credible interval and '
                      f'{fc_steps}-step-ahead forecast')
        else:
            plt.title(f'Posterior credible interval')


class VIModel(object):
    """Abstract class for performing VI with general models (not necessarily
    time series models).
    """

    result_class: classmethod = VIResult
    name = 'VI Model'

    def __init__(self,
                 input_length=None,
                 num_draws: int = 1,
                 quiet=False,
                 stochastic_entropy: bool = False,
                 stop_heur: StoppingHeuristic = None):
        self.stochastic_entropy, self.num_draws = stochastic_entropy, num_draws
        self.stop_heur = stop_heur or NoImprovementStoppingHeuristic()
        self.quiet, self.input_length = quiet, input_length

        self.params: List[ModelParameter] = []
        for attr, a in type(self).__dict__.items():
            if not isinstance(a, ModelParameter):
                continue
            a.inferred_name(attr)
            self.params.append(a)

        index = 0
        self.d = 0
        for p in self.params:
            p.index = index
            prior_name = f'{p.name}_prior'  # e.g. self.σ_prior()
            setattr(self, prior_name, p.prior)

            if isinstance(p, LocalParameter):
                if input_length is None:
                    raise Exception('Data length required for local variables')
                p.dimension = input_length
            elif isinstance(p, TransformedModelParameter):
                tfm_name = f'{p.name}_to_{p.transformed_name}'
                setattr(self, tfm_name, p.transform)  # e.g. self.σ_to_φ()
                tfm_prior_name = f'{p.transformed_name}_prior'
                tfm_prior = TransformedDistribution(p.prior, p.transform)
                setattr(self, tfm_prior_name, tfm_prior)  # e.g. self.φ_prior()
            index += p.dimension
            self.d += p.dimension

        assert self.d > 0, 'No parameters'

        # dense approximation: q = N(u, LL')
        self.u = torch.tensor(torch.zeros(self.d), requires_grad=True)
        self.L = torch.tensor(torch.eye(self.d), requires_grad=True)
        self.parameters = [self.u, self.L]
        self.q = MultivariateNormal(self.u, scale_tril=self.L)

        self.optimizer = torch.optim.Adadelta(self.parameters)

    def unpack(self, ζ:torch.Tensor):
        """Unstack the vector ζ into individual parameters, in the order given
        in self.params. For transformed parameters, both optimization and
        natural parameters are returned as a tuple.
        """
        assert ζ.shape == (self.d,), f'Expected 1-tensor of length {self.d}'
        unpacked = []
        index = 0
        for p in self.params:
            opt_p = ζ[index:index+p.dimension]
            if isinstance(p, TransformedModelParameter):
                # transform parameter *back* to natural coordinates
                nat_p = p.transform.inv(opt_p)
                unpacked.append((nat_p, opt_p))
            else:
                unpacked.append(opt_p)
            index += p.dimension
        return tuple(unpacked)

    def print(self, *args):
        """Print function that only does anything if quiet is False."""
        if not self.quiet:
            print(*args)

    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    def training_loop(self, y, max_iters: int = 2**20, λ=0.1):
        """Train the model using VI.

        Args:
            y: (a 1-tensor) data vector
            max_iters: maximum number of iterations
            λ: exponential smoothing parameter for displaying estimated elbo
               (display only; does not affect the optimization)
            quiet: suppress output

        Returns:
            A VariationalResults object with the approximate posterior.
        """
        assert 0. < λ <= 1., 'λ out of range'
        self.print(f'{"="*80}\n{str(self)}\n\n'
                   f'Displayed loss is smoothed with λ={λ}\n{"="*80}')
        t, i = -time(), 0
        elbo_hats = []
        smoothed_objective = -self.elbo_hat(y).data
        for i in range(max_iters):
            self.optimizer.zero_grad()
            objective = -self.elbo_hat(y)
            objective.backward()
            self.optimizer.step()
            elbo_hats.append(-objective.data)
            smoothed_objective = λ*objective.data + (1. - λ)*smoothed_objective
            if not i & (i - 1):
                self.print_status(i, -smoothed_objective)
            if self.stop_heur.early_stop(-objective.data):
                self.print('Stopping heuristic criterion satisfied')
                break
        else:
            self.print('WARNING: maximum iterations reached.')
        t += time()
        self.print_status(i + 1, -smoothed_objective)
        self.print(
            f'Completed {i+1} iterations in {t:.1f}s @ {i/(t+1):.2f} i/s.\n'
            f'{"="*80}')
        result = self.result_class(model=self, elbo_hats=elbo_hats, y=y)
        return result

    def print_status(self, i, elbo_hat):
        self.print(f'{i: 8d}. smoothed elbo_hat ={float(elbo_hat):12.2f}')

    def __str__(self):
        _entr = 'Stochastic' if self.stochastic_entropy else 'Analytic'
        _oname = type(self.optimizer).__name__
        lines = [f"{self.name}:\n"
                 f"  - {_entr} entropy term with M={self.num_draws};\n"
                 f"  - {str(self.stop_heur)}",
                 f'  - {_oname} optimizer with param groups:']
        for i, pg in enumerate(self.optimizer.param_groups):
            desc = ', '.join(f'{k}={v}' for k, v in pg.items() if k != 'params')
            lines.append(f'    group {i}. {desc}')
        return '\n'.join(lines)
