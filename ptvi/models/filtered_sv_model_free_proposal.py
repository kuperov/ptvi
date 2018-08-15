import torch
from torch.distributions import LogNormal, Normal, Beta

from ptvi import FilteredStateSpaceModel, global_param, AR1Proposal, PFProposal


class FilteredStochasticVolatilityModelFreeProposal(FilteredStateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)

    The proposal density is also an AR(1):

    .. math::
        z_t = d + e * z_{t-1} + η_t    η_t ~ Ν(0,1)
    """

    name = "Particle filtered stochastic volatility model"
    a = global_param(prior=LogNormal(0, 1), transform="log", rename="α")
    b = global_param(prior=Normal(0, 1))
    c = global_param(prior=Beta(1, 1), transform="logit", rename="ψ")
    d = global_param(prior=Normal(0, 1))
    e = global_param(prior=Beta(1, 1), transform="logit", rename="ρ")

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)
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

    def sample_observed(self, ζ, y, fc_steps=0):
        a, _, _, _, _ = self.unpack_natural(ζ)
        z = self.sample_unobserved(ζ, y, fc_steps)
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def sample_unobserved(self, ζ, y, fc_steps=0):
        assert y is not None
        a, b, c, _, _ = self.unpack_natural(ζ)
        # get a sample of states by filtering wrt y
        z = torch.empty((len(y) + fc_steps,))
        self.simulate_log_phatN(y=y, ζ=ζ, sample=z)
        # now project states forward fc_steps
        if fc_steps > 0:
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        _, _, _, d, e = self.unpack_natural(ζ)
        return AR1Proposal(μ=d, ρ=e, σ=1)

    def forecast(self, ζ, y, fc_steps):
        pass

    def __repr__(self):
        return (
            f"Stochastic volatility model:\n"
            f"\tx_t = exp(a * z_t/2) ε_t      t=1, …, {self.input_length}\n"
            f"\tz_t = b + c * z_{{t-1}} + ν_t,  t=2, …, {self.input_length}\n"
            f"\tz_1 = b + 1/√(1 - c^2) ν_1\n"
            f"\twhere ε_t, ν_t ~ Ν(0,1)\n\n"
            f"Particle filter with {self.num_particles} particles, AR(1) proposal:\n"
            f"\tz_t = d + e * z_{{t-1}} + η_t,  t=2, …, {self.input_length}\n"
            f"\tz_1 = d + 1/√(1 - e^2) η_1\n"
            f"\twhere η_t ~ Ν(0,1)\n"
        )
