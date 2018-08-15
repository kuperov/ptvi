import torch
from torch.distributions import LogNormal, Normal, Beta
from ptvi import FilteredStateSpaceModel, global_param, AR1Proposal, PFProposal


class FilteredSVModelDualOpt(FilteredStateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)

    The proposal density is

    .. math::
        z_t = d + e * z_{t-1} + η_t    η_t ~ Ν(0,1)

    The model parameter ζ covers the parameters used in the SV model, ζ={a, b, c}.

    The alternative parameter η covers the parameters η={d, e}.
    """

    name = "Particle filtered stochastic volatility model"
    a = global_param(prior=LogNormal(0, 1), transform="log", rename="α")
    b = global_param(prior=Normal(0, 1))
    c = global_param(prior=Beta(1, 1), transform="logit", rename="ψ")
    d = global_param(prior=Normal(0, 1))
    e = global_param(prior=Beta(1, 1), transform="logit", rename="ρ")

    def parameters(self):
        return [self.a, self.b, self.c]

    def proposal_parameters(self):
        return [self.d, self.e]

    def unpack_natural_model_parameters(self, ζ):
        a = ζ[0]
        b = ζ[1]
        c = ζ[2]
        return a, b, c

    def unpack_natural_proposal_parameters(self, η):
        d = η[0]
        e = η[1]
        return d, e

