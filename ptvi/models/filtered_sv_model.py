import torch
from torch.distributions import Normal, Categorical

from ptvi import (
    FilteredStateSpaceModel,
    FilteredStateSpaceModelFreeProposal,
    global_param,
    AR1Proposal,
    PFProposal,
    LogNormalPrior,
    NormalPrior,
    BetaPrior,
)


class FilteredStochasticVolatilityModel(FilteredStateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)
    """

    name = "Particle filtered stochastic volatility model"
    a = global_param(prior=LogNormalPrior(0, 1), transform="log", rename="α")
    b = global_param(prior=NormalPrior(0, 1))
    c = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ψ")

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)
        z_true = torch.empty((self.input_length,))
        z_true[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z_true[t] = b + c * z_true[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(a) * torch.exp(z_true / 2)).sample()
        return x.type(self.dtype), z_true.type(self.dtype)

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
        a, b, c = self.unpack_natural(ζ)
        if t == 0:
            log_pzt = Normal(b, (1 - c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(b + c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def sample_observed(self, ζ, y, fc_steps=0):
        a, b, c = self.unpack_natural(ζ)
        z = self.sample_unobserved(ζ, y, fc_steps)
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def sample_unobserved(self, ζ, y, fc_steps=0):
        assert y is not None
        a, b, c = self.unpack_natural(ζ)
        # get a sample of states by filtering wrt y
        z = torch.empty((len(y) + fc_steps,))
        self.simulate_log_phatN(y=y, ζ=ζ, sample=z)
        # now project states forward fc_steps
        if fc_steps > 0:
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        _, b, c = self.unpack_natural(ζ)
        return AR1Proposal(μ=b, ρ=c, σ=1)

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
            f"\tz_t = b + c * z_{{t-1}} + η_t,  t=2, …, {self.input_length}\n"
            f"\tz_1 = b + 1/√(1 - c^2) η_1\n"
            f"\twhere η_t ~ Ν(0,1)\n"
        )


class FilteredSVModelDualOpt(FilteredStateSpaceModelFreeProposal):
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
    a = global_param(prior=LogNormalPrior(0, 1), transform="log", rename="α")
    b = global_param(prior=NormalPrior(0, 1))
    c = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ψ")
    d = global_param(prior=NormalPrior(0, 1))
    e = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ρ")

    def __init__(self, input_length: int, num_particles: int = 50, resample=True):
        super().__init__(input_length, num_particles, resample)
        self._md = 3
        self._pd = 2  # no σ in proposal yet

    def simulate(self, a, b, c):
        """Simulate from p(y, z | θ)"""
        a, b, c = map(torch.tensor, (a, b, c))
        z_true = torch.empty((self.input_length,))
        z_true[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z_true[t] = b + c * z_true[t - 1] + Normal(0, 1).sample()
        y = Normal(0, torch.exp(a) * torch.exp(z_true / 2)).sample()
        return y, z_true

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
        a, b, c, = self.unpack_natural_model_parameters(ζ)
        if t == 0:
            log_pzt = Normal(b, (1 - c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(b + c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def ln_prior(self, ζ: torch.Tensor) -> float:
        a, b, c = self.unpack_natural_model_parameters(ζ)
        return (
            self.a_prior.log_prob(a)
            + self.b_prior.log_prob(b)
            + self.c_prior.log_prob(c)
        )

    def model_parameters(self):
        return [self.a, self.b, self.c]

    def proposal_parameters(self):
        return [self.d, self.e]

    def unpack_natural_model_parameters(self, ζ: torch.Tensor):
        α, b, ψ = ζ[0], ζ[1], ζ[2]
        return self.a_to_α.inv(α), b, self.c_to_ψ.inv(ψ)

    def unpack_natural_proposal_parameters(self, η: torch.Tensor):
        d, ρ = η[0], η[1]
        return d, self.e_to_ρ.inv(ρ)

    def simulate_log_phatN(
        self,
        y: torch.Tensor,
        ζ: torch.Tensor,
        η: torch.Tensor,
        sample: torch.Tensor = None,
    ):
        """Apply particle filter to estimate marginal likelihood log p^(y | ζ)

        This algorithm is subtly different than the one in fivo.py, because it
        also takes η as a parameter.
        """
        log_phatN = 0.
        log_N = math.log(self.num_particles)
        log_w = torch.full((self.num_particles,), -log_N)
        Z = None
        proposal = self.proposal_for(y, ζ)
        for t in range(self.input_length):
            zt = proposal.conditional_sample(t, Z, self.num_particles).unsqueeze(0)
            Z = torch.cat([Z, zt]) if Z is not None else zt
            log_αt = self.conditional_log_prob(
                t, y, Z, ζ
            ) - proposal.conditional_log_prob(t, Z)
            log_phatt = torch.logsumexp(log_w + log_αt, dim=0)
            log_phatN += log_phatt
            log_w += log_αt - log_phatt
            with torch.no_grad():
                ESS = 1. / torch.exp(2 * log_w).sum()
                if self.resample and ESS < self.num_particles:
                    a = Categorical(torch.exp(log_w)).sample((self.num_particles,))
                    Z = (Z[:, a]).clone()
                    log_w = torch.full((self.num_particles,), -log_N)
        if sample is not None:
            with torch.no_grad():
                # samples should be M * T, where M is the number of samples
                assert sample.shape[0] >= self.input_length
                idxs = Categorical(torch.exp(log_w)).sample()
                sample[: self.input_length] = Z[:, idxs]
        return log_phatN

    def proposal_for(self, y: torch.Tensor, η: torch.Tensor) -> PFProposal:
        """Return the proposal distribution for the given parameters.

        Args:
            y: data vector
            η: proposal parameter vector
        """
        d, e = self.unpack_natural_proposal_parameters(η)
        return AR1Proposal(μ=d, ρ=e, σ=1.)

    @property
    def md(self) -> int:
        """Dimension of the model."""
        return self._md

    @property
    def pd(self) -> int:
        """Dimension of the proposal."""
        return self._pd

    def sample_observed(self, ζ, y, fc_steps=0):
        a, b, c = self.unpack_natural_model_parameters(ζ[:3])
        z = self.sample_unobserved(ζ, y, fc_steps)
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def sample_unobserved(self, ζ, y, fc_steps=0):
        assert y is not None
        a, b, c = self.unpack_natural_model_parameters(ζ[:3])
        # get a sample of states by filtering wrt y
        z = torch.empty((len(y) + fc_steps,))
        self.simulate_log_phatN(y=y, ζ=ζ[:3], η=ζ[3:], sample=z)
        # now project states forward fc_steps
        if fc_steps > 0:
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def __repr__(self):
        return (
            f"Stochastic volatility model for dual optimization of model and proposal:\n"
            f"\tx_t = exp(a * z_t/2) ε_t      t=1, …, {self.input_length}\n"
            f"\tz_t = b + c * z_{{t-1}} + ν_t,  t=2, …, {self.input_length}\n"
            f"\tz_1 = b + 1/√(1 - c^2) ν_1\n"
            f"\twhere ε_t, ν_t ~ Ν(0,1)\n\n"
            f"Particle filter with {self.num_particles} particles, AR(1) proposal:\n"
            f"\tz_t = d + e * z_{{t-1}} + η_t,  t=2, …, {self.input_length}\n"
            f"\tz_1 = d + 1/√(1 - e^2) η_1\n"
            f"\twhere η_t ~ Ν(0,1)\n"
        )


class FilteredStochasticVolatilityModelFixedParams(FilteredStateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)
    """

    name = "Particle filtered stochastic volatility model"
    d = global_param(prior=NormalPrior(0, 1))
    e = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ρ")

    def __init__(self, input_length, num_particles, resample):
        self.a, self.b, self.c = map(torch.tensor, (0.5, 1., 0.95))
        super().__init__(
            input_length=input_length, num_particles=num_particles, resample=resample
        )

    def simulate(self):
        """Simulate from p(x, z | θ)"""
        z_true = torch.empty((self.input_length,))
        z_true[0] = Normal(self.b, (1 - self.c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z_true[t] = self.b + self.c * z_true[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(self.a) * torch.exp(z_true / 2)).sample()
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
        if t == 0:
            log_pzt = Normal(self.b, (1 - self.c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(self.b + self.c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(self.a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def sample_observed(self, ζ, y, fc_steps=0):
        z = self.sample_unobserved(ζ, y, fc_steps)
        return Normal(0, torch.exp(self.a) * torch.exp(z / 2)).sample()

    def sample_unobserved(self, ζ, y, fc_steps=0):
        assert y is not None
        # get a sample of states by filtering wrt y
        z = torch.empty((len(y) + fc_steps,))
        self.simulate_log_phatN(y=y, ζ=ζ, sample=z)
        # now project states forward fc_steps
        if fc_steps > 0:
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = self.b + self.c * z[t - 1] + Normal(0, 1).sample()
        return Normal(0, torch.exp(self.a) * torch.exp(z / 2)).sample()

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        d, e = self.unpack_natural(ζ)
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
    a = global_param(prior=LogNormalPrior(0, 1), transform="log", rename="α")
    b = global_param(prior=NormalPrior(0, 1))
    c = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ψ")
    d = global_param(prior=NormalPrior(0, 1))
    e = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ρ")
    f = global_param(prior=LogNormalPrior(0, 1), transform="log", rename="ι")

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = map(torch.tensor, (a, b, c))
        z = torch.empty((self.input_length,))
        z[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()
        return x.type(self.dtype).to(self.device), z.type(self.dtype).to(self.device)

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
        a, b, c, _, _, _ = self.unpack_natural(ζ)
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
        a, b, c, _, _, _ = self.unpack_natural(ζ)
        # get a sample of states by filtering wrt y
        z = torch.empty((len(y) + fc_steps,))
        self.simulate_log_phatN(y=y, ζ=ζ, sample=z)
        # now project states forward fc_steps
        if fc_steps > 0:
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        _, _, _, d, e, f = self.unpack_natural(ζ)
        return AR1Proposal(μ=d, ρ=e, σ=f)

    def forecast(self, ζ, y, fc_steps):
        pass

    def __repr__(self):
        return (
            f"Stochastic volatility model with parameters {{a, b, c}}:\n"
            f"\tx_t = exp(a * z_t/2) ε_t        t=1,…,{self.input_length}\n"
            f"\tz_t = b + c * z_{{t-1}} + ν_t,    t=2,…,{self.input_length}\n"
            f"\tz_1 = b + 1/√(1 - c^2) ν_1\n"
            f"\twhere ε_t, ν_t ~ Ν(0,1)\n\n"
            f"Filter with {self.num_particles} particles; AR(1) proposal params {{d, e, f}}:\n"
            f"\tz_t = d + e * z_{{t-1}} + f η_t,  t=2,…,{self.input_length}\n"
            f"\tz_1 = d + f/√(1 - e^2) η_1\n"
            f"\twhere η_t ~ Ν(0,1)\n"
        )
