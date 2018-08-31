from warnings import warn
from typing import List

import math
import torch
from torch.distributions import Normal, Categorical, TransformedDistribution

import ptvi
from ptvi.params import TransformedModelParameter, ModelParameter, LocalParameter
from ptvi.mvn_posterior import MVNPosterior

import numpy as np


_DIVIDER = "―" * 80


class Model(object):
    """Abstract class for performing VI with general models (not necessarily
    time series models).
    """

    name = "VI Model"
    has_observation_error = True  # true if outcome is imperfectly observed
    result_type = MVNPosterior

    def __init__(self, input_length=None, dtype=None, device=None):
        """
        Construct a model.

        Args:
            input_length: data input size this model should expect. Especially useful
                          for time series models.
            dtype:        data type (default: torch.float32)
            device:       compute device (default: cpu)
        """
        self.input_length = input_length
        self.dtype = dtype or torch.float32
        self.device = device or torch.device("cpu")

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
            prior = p.get_prior_distribution(dtype=self.dtype, device=self.device)
            setattr(self, f"{p.name}_prior", prior)  # e.g. self.σ_prior()

            if isinstance(p, LocalParameter):
                if input_length is None:
                    raise Exception("Data length required for local variables")
                p.dimension = input_length
            elif isinstance(p, TransformedModelParameter):
                tfm_name = f"{p.name}_to_{p.transformed_name}"
                setattr(self, tfm_name, p.transform)  # e.g. self.σ_to_φ()
                tfm_prior_name = f"{p.transformed_name}_prior"
                tfm_prior = TransformedDistribution(prior, p.transform)
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

    def unpack_unrestricted(self, ζ: torch.Tensor):
        """Unstack the vector ζ into individual parameters, in the order given
        in self.params. For transformed parameters, optimization (transformed)
        coordinates are used.
        """
        assert ζ.shape == (self.d,), f"Expected 1-tensor of length {self.d}"
        unpacked = []
        index = 0
        for p in self.params:
            opt_p = ζ[index : index + p.dimension].squeeze()
            unpacked.append(opt_p)
            index += p.dimension
        return tuple(unpacked)

    def unpack_natural(self, ζ: torch.Tensor):
        """Unstack the vector ζ into individual parameters, in the order given
        in self.params. For transformed parameters, natural parameters are returned (ie
        the parameters are transformed back into parameters appropriate for the log
        likelihood function.)
        """
        assert ζ.shape == (self.d,), f"Expected 1-tensor of length {self.d}"
        unpacked = []
        index = 0
        for p in self.params:
            opt_p = ζ[index : index + p.dimension].squeeze()
            if isinstance(p, TransformedModelParameter):
                # transform parameter *back* to natural coordinates
                nat_p = p.transform.inv(opt_p)
                unpacked.append(nat_p)
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

    def tabulate_priors(self):
        return ["{p.name} ~ {str(p.prior)}" for p in self.params]

    def __str__(self):
        return self.name


class PFProposal(object):
    def conditional_sample(self, t, Z, N):
        raise NotImplementedError

    def conditional_log_prob(self, t, Z):
        raise NotImplementedError


class FilteredStateSpaceModel(Model):
    class FIVOResult(ptvi.MVNPosterior):
        def plot_latent(
            self, N: int = 100, α: float = 0.05, true_z=None, fc_steps: int = 0
        ):
            import matplotlib.pyplot as plt

            paths = self.sample_latent_paths(N, fc_steps=fc_steps)
            ci_bands = np.empty([self.input_length + fc_steps, 2])
            fxs, xs = range(self.input_length + fc_steps), range(self.input_length)
            perc = 100 * np.array([α * 0.5, 1. - α * 0.5])
            for t in fxs:
                ci_bands[t, :] = np.percentile(paths[:, t], q=perc)
            plt.fill_between(
                fxs,
                ci_bands[:, 0],
                ci_bands[:, 1],
                alpha=0.5,
                label=f"{(1 - α)*100:.0f}% CI",
            )
            if true_z is not None:
                plt.plot(
                    xs, true_z.cpu().numpy(), color="black", linewidth=2, label="z"
                )
                plt.legend()
            if fc_steps > 0:
                plt.axvline(x=self.input_length, color="black")
                plt.title(
                    f"Posterior credible interval and "
                    f"{fc_steps}-step-ahead forecast"
                )
            else:
                plt.title(f"Posterior credible interval")

    result_type = FIVOResult

    def __init__(
        self,
        input_length: int,
        num_particles: int = 50,
        resample=True,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        self.num_particles = num_particles
        self.resample = resample
        super().__init__(input_length=input_length, dtype=dtype, device=device)

    def simulate(self, *args, **kwargs):
        """Simulate from p(x_{1:T} | θ)."""
        raise NotImplementedError

    def conditional_log_prob(self, t, y, Z, ζ):
        """Compute log p(x_t, z_t | y_{0:t-1}, z_{0:t-1}, ζ).

        Args:
            t: time index (zero-based)
            y: y_{0:t} vector of points observed up to this point (which may
               actually be longer, but should only be indexed up to t)
            z: z_{0:t} matrix of unobserved variables to condition on, one row
               per trajectory
            ζ: parameter to condition on; should be unpacked with self.unpack

        Return:
            1x1 tensor containing estimated conditional log marginal density
        """
        raise NotImplementedError

    def ln_joint(self, y, ζ):
        llik_hat = self.simulate_log_phatN(y, ζ)
        lprior = self.ln_prior(ζ)
        return llik_hat + lprior

    def simulate_log_phatN(
        self, y: torch.Tensor, ζ: torch.Tensor, sample: torch.Tensor = None
    ):
        """Apply particle filter to estimate marginal likelihood log p^(y | ζ)"""
        log_phatN = 0.
        log_N = math.log(self.num_particles)
        log_w = torch.full(
            (self.num_particles,), -log_N, dtype=self.dtype, device=self.device
        )
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
                    w = torch.exp(log_w)
                    if not all(torch.isfinite(w)):
                        warn(f'Overflow: log_w = {log_w}')
                        w = torch.tensor(1. - torch.isfinite(w), dtype=self.dtype, device=self.device)
                    a = Categorical(w).sample((self.num_particles,))
                    Z = (Z[:, a]).clone()
                    log_w = torch.full(
                        (self.num_particles,),
                        -log_N,
                        dtype=self.dtype,
                        device=self.device,
                    )
        if sample is not None:
            with torch.no_grad():
                # samples should be M * T, where M is the number of samples
                assert sample.shape[0] >= self.input_length
                idxs = Categorical(torch.exp(log_w)).sample()
                sample[: self.input_length] = Z[:, idxs]
        return log_phatN

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        """Return the proposal distribution for the given parameters.

        Args:
            y: data vector
            ζ: parameter vector
        """
        raise NotImplementedError


class AR1Proposal(PFProposal):
    """A simple linear/gaussian AR(1) to use as a particle filter proposal.

    .. math::
        z_t = μ + ρ * z_{t-1} + η_t
    """

    def __init__(self, μ, ρ, σ):
        assert -1 < ρ < 1 and σ > 0, f"ρ={ρ} and σ={σ}"
        self.μ, self.ρ, self.σ = μ, ρ, σ

    def conditional_sample(self, t, Z, N):
        """Simulate z_t from q(z_t | z_{0:t-1}, y_{0:t}, φ)

        Z has an extra dimension, of N particles.
        """
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2) ** (-.5)).rsample((N,))
        else:
            return Normal(self.μ + self.ρ * Z[t - 1], 1).rsample()

    def conditional_log_prob(self, t, Z):
        """Compute log q(z_t | z_{t-1}, φ)"""
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2) ** (-.5)).log_prob(Z[t])
        else:
            return Normal(self.μ + self.ρ * Z[t - 1], 1).log_prob(Z[t])

    def __repr__(self):
        return (
            "AR(1) proposals:\n"
            f"\tz_t = {self.μ:.2f} + {self.ρ:.2f} * z_{{t-1}} + η_t\n"
            f"\tη_t ~ Ν(0,{self.σ:.2f})"
        )


class FilteredStateSpaceModelFreeProposal(FilteredStateSpaceModel):
    """A specialization of FilteredStateSpaceModel with separate parameters for the proposal
    and the model.

    The model parameter ζ covers the parameters used in the model, and the
    alternative parameter η covers the parameters η={d, e}.
    """

    name = "Particle filtered model"

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = map(torch.tensor, (a, b, c))
        z_true = torch.empty((self.input_length,))
        z_true[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z_true[t] = b + c * z_true[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(a) * torch.exp(z_true / 2)).sample()
        return x, z_true

    def ln_joint(self, y, ζ, η):
        llik_hat = self.simulate_log_phatN(y, ζ, η)
        lprior = self.ln_prior(ζ)
        return llik_hat + lprior

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
        a, b, c, d, e = self.unpack_natural_model_parameters(ζ)
        if t == 0:
            log_pzt = Normal(b, (1 - c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(b + c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def model_parameters(self):
        raise NotImplementedError

    def proposal_parameters(self):
        raise NotImplementedError

    def unpack_natural_model_parameters(self, ζ: torch.Tensor):
        raise NotImplementedError

    def unpack_natural_proposal_parameters(self, η: torch.Tensor):
        raise NotImplementedError

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
        proposal = self.proposal_for(y, η)
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
        raise NotImplementedError

    @property
    def md(self) -> int:
        """Dimension of the model."""
        raise NotImplementedError

    @property
    def pd(self) -> int:
        """Dimension of the proposal."""
        raise NotImplementedError
