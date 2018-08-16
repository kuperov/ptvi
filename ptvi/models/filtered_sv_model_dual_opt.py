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

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = map(torch.tensor, (a, b, c))
        z_true = torch.empty((self.input_length,))
        z_true[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z_true[t] = b + c * z_true[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(a) * torch.exp(z_true / 2)).sample()
        return x, z_true

    def conditional_log_prob(self, t, y, z, ζ):
        """Compute log p(x_t, z_t | y_{0:t-1}, z_{0:t-1}, ζ).

        Args:
            t: time index (zero-based)
            y: y_{0:t} vector of points observed up to this point (which may
               actually be longer, but should only be indexed up to t)
            z: z_{0:t} vector of unobserved variables to condition on (ditto,
               array may be longer)
            ζ: parameter to condition on; should be unpacked with self.unpack
        """
        a, b, c, d, e = self.unpack_natural(ζ)
        if t == 0:
            log_pzt = Normal(b, (1 - c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(b + c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def model_parameters(self):
        return [self.a, self.b, self.c]

    def proposal_parameters(self):
        return [self.d, self.e]

    def unpack_natural_model_parameters(self, ζ: torch.Tensor):
        α, b, ψ = ζ[0], ζ[1], ζ[2]
        return self.α_to_a(α), b, self.ψ_to_c(ψ)

    def unpack_natural_proposal_parameters(self, η: torch.Tensor):
        d, ρ = η[0], η[1]
        return d, self.ρ_to_e(ρ)
