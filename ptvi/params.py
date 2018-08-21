from typing import Union
import torch
from torch.distributions import (
    Distribution,
    Transform,
    ComposeTransform,
    SigmoidTransform,
    ExpTransform,
    AffineTransform,
)
from ptvi.priors import Prior, ImproperPrior


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

    def __init__(self, name: str, prior: Prior):
        self.name, self.prior = name, prior

    def inferred_name(self, name):
        """Provides a guess of the name for this variable. Accepts name if none
        provided."""
        self.name = self.name or name

    @property
    def prior_name(self):
        return "{}_prior".format(self.name)

    @property
    def post_marg_name(self):
        return "{}_post_marg".format(self.name)

    def get_prior_distribution(self, dtype, device):
        return self.prior.to_distribution(dtype=dtype, device=device)

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

    def __init__(
        self,
        name: str,
        prior: Prior,
        transformed_name: str,
        transform: Transform,
        transform_desc: str = None,
    ):
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
            tfm_desc = self.transform_desc or "transformed"
            self.transformed_name = f"{tfm_desc}_{self.name}"

    @property
    def tfm_prior_name(self):
        return "{}_prior".format(self.transformed_name)

    @property
    def tfm_name(self):
        return "{}_to_{}".format(self.name, self.transformed_name)

    @property
    def tfm_post_marg_name(self):
        return "{}_post_marg".format(self.transformed_name)

    def __str__(self):
        return (
            f"{self.name} with prior {self.prior} transformed to "
            f"{self.transformed_name} by {self.transform}"
        )


class LocalParameter(ModelParameter):
    """A local model parameter.

    For now local parameters are assumed not transformable and dimension 1.
    """

    def __init__(self, name: str, prior: Prior):
        super().__init__(name, prior)

    def __str__(self):
        return f"{self.name} local parameter with prior {self.prior}"


def global_param(
    prior: Prior = None,
    name: str = None,
    transform: Union[Transform, str] = None,
    rename: str = None,
):
    """Define a scalar global model parameter.

    Args:
        prior: parameter prior, a Distribution object
        name: optional, name for parameter
        transform: optional transformation to apply (its domain should be an
                   unconstrained space)
        rename: optional, name of parameter in unconstrained space
    """
    if prior is None:
        prior = ImproperPrior()
    if rename is not None and transform is None:
        raise Exception("rename requires a transform")
    if transform is None:
        return ModelParameter(name=name, prior=prior)
    transform_desc = "transformed"
    if isinstance(transform, str):
        transform_desc = transform
        if transform == "log":
            transform = torch.distributions.ExpTransform().inv
            if rename is None and name is not None:
                rename = f"log{name}"
        elif transform == "exp":
            raise Exception("Use 'log' to constrain parameters > 0")
        elif transform == "logit":
            transform = torch.distributions.SigmoidTransform().inv
            if rename is None and name is not None:
                rename = f"logit{name}"
        elif transform == "slogit":
            # nasty and hacky attempt to avoid saturating the logistic transform
            inv_transform = ComposeTransform(
                [AffineTransform(loc=0, scale=1e-4), SigmoidTransform()]
            )
            transform = inv_transform.inv
            if rename is None and name is not None:
                rename = f"slogit{name}"
        elif transform == "sigmoid":
            raise Exception("Use 'logit' to constrain parameters to (0, 1)")
        else:
            raise Exception(f"Unknown transform {transform}")
    if rename is None and name is not None:
        rename = f"{transform_desc}_{name}"
    return TransformedModelParameter(
        name=name,
        prior=prior,
        transform=transform,
        transformed_name=rename,
        transform_desc=transform_desc,
    )


def local_param(prior: Distribution = None, name: str = None):
    """Define a local (latent) parameter.

    Args:
        prior: parameter prior, a Distribution object
        name: optional, name for parameter
    """
    if prior is None:
        prior = ImproperPrior()
    return LocalParameter(name=name, prior=prior)
