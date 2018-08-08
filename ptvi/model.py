from typing import List

import torch
from torch.distributions import TransformedDistribution

from ptvi.params import TransformedModelParameter, ModelParameter, LocalParameter
from ptvi.mvn_posterior import MVNPosterior


_DIVIDER = "―" * 80


class Model(object):
    """Abstract class for performing VI with general models (not necessarily
    time series models).
    """

    name = "VI Model"
    has_observation_error = True  # true if outcome is imperfectly observed
    result_type = MVNPosterior

    def __init__(self, input_length=None):
        self.input_length = input_length

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
            prior_name = f"{p.name}_prior"  # e.g. self.σ_prior()
            setattr(self, prior_name, p.prior)

            if isinstance(p, LocalParameter):
                if input_length is None:
                    raise Exception("Data length required for local variables")
                p.dimension = input_length
            elif isinstance(p, TransformedModelParameter):
                tfm_name = f"{p.name}_to_{p.transformed_name}"
                setattr(self, tfm_name, p.transform)  # e.g. self.σ_to_φ()
                tfm_prior_name = f"{p.transformed_name}_prior"
                tfm_prior = TransformedDistribution(p.prior, p.transform)
                setattr(self, tfm_prior_name, tfm_prior)  # e.g. self.φ_prior()
            index += p.dimension
            self.d += p.dimension

        assert self.d > 0, "No parameters"

    def unpack(self, ζ: torch.Tensor):
        """Unstack the vector ζ into individual parameters, in the order given
        in self.params. For transformed parameters, both optimization and
        natural parameters are returned as a tuple.
        """
        assert ζ.shape == (self.d,), f"Expected 1-tensor of length {self.d}"
        unpacked = []
        index = 0
        for p in self.params:
            opt_p = ζ[index : index + p.dimension].squeeze()
            if isinstance(p, TransformedModelParameter):
                # transform parameter *back* to natural coordinates
                nat_p = p.transform.inv(opt_p)
                unpacked.append((nat_p, opt_p))
            else:
                unpacked.append(opt_p)
            index += p.dimension
        return tuple(unpacked)

    def ln_prior(self, ζ: torch.Tensor):
        """Compute log prior at ζ, with respect to transformed parameters (ie including
        jacobian adjustments from transformations into free parameters).
        """
        assert ζ.shape == (self.d,), f"Expected 1-tensor of length {self.d}"
        index, lp = 0, 0.
        for p in self.params:
            opt_p = ζ[index : index + p.dimension].squeeze()
            if isinstance(p, TransformedModelParameter):
                prior_fn = getattr(self, p.tfm_prior_name)
            else:
                prior_fn = getattr(self, p.prior_name)
            lp += prior_fn.log_prob(opt_p)
            index += p.dimension
        return lp

    def ln_joint(self, y, ζ):
        """Computes the log likelihood plus the log prior at ζ."""
        return self.ln_likelihood(y, ζ) + self.ln_prior(ζ)

    def ln_likelihood(self, y, ζ):
        """Computes the log likelihood at ζ."""
        raise NotImplementedError

    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    def sample_observed(self, ζ, y, fc_steps=0):
        """Sample a path from the model, forecasting fc_steps additional steps,
        given parameters ζ (in transformed space).

        Args:
            ζ: parameter to condition on
            y: observed data to optionally condition on
            fc_steps: number of steps to project forward

        Returns:
            1-tensor of sample paths of length (input_length+fc_steps)
        """
        raise NotImplementedError

    def forecast(self, ζ, y, fc_steps):
        """Project a path from the model with fc_steps additional steps,
        given parameters ζ (in transformed space).

        Args:
            ζ: parameter to condition on
            y: observed data to extend
            fc_steps: number of steps to project forward

        Returns:
            1-tensor of length fc_steps
        """
        raise NotImplementedError

    def __str__(self):
        return self.name
