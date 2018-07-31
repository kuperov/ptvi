import torch
from torch.distributions import Normal, LogNormal, Beta
from ptvi import Model, global_param, InvGamma


class FilteredLocalLevelModel(Model):
    """Local level (linear gaussian) model that exploits the Kalman filter."""

    name = 'Filtered local level model'
    z0 = global_param(prior=Normal(0, 10))
    σz0 = global_param(prior=LogNormal(1, 1), transform='log', rename='ςz0')
    γ = global_param(prior=Normal(1, 3))
    η = global_param(prior=LogNormal(0, 1), transform='log', rename='ψ')
    σ = global_param(prior=InvGamma(1, 5), transform='log', rename='ς')
    ρ = global_param(prior=Beta(1, 1), transform='logit', rename='φ')

    def ln_joint(self, y, ζ):
        """Computes the log likelihood plus the log prior at ζ."""
        z0, (σz0, ςz0), γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)

        # unroll first iteration of loop to set initial conditions

        # prediction step
        z_pred = ρ * z0
        Σz_pred = ρ**2 * σz0**2 + 1
        y_pred = η * z_pred
        Σy_pred = η**2 * Σz_pred + σ**2

        # correction step
        gain = Σz_pred * η / Σy_pred
        z_upd = z_pred + gain * (y[0] - y_pred)
        Σz_upd = Σz_pred - gain**2 * Σy_pred

        llik = Normal(y_pred, torch.sqrt(Σy_pred)).log_prob(y[0])

        for t in range(2, y.shape[0] + 1):
            i = t - 1
            # prediction step
            z_pred = ρ * z_upd
            Σz_pred = ρ**2 * Σz_upd + 1
            y_pred = η * z_pred
            Σy_pred = η**2 * Σz_pred + σ**2

            # correction step
            gain = Σz_pred * η / Σy_pred
            z_upd = z_pred + gain * (y[i] - y_pred)
            Σz_upd = Σz_pred - gain**2 * Σy_pred

            llik += Normal(y_pred, torch.sqrt(Σy_pred)).log_prob(y[i])

        lprior = (
            self.ςz0_prior.log_prob(ςz0)
            + self.z0_prior.log_prob(z0)
            + self.γ_prior.log_prob(γ)
            + self.ψ_prior.log_prob(ψ)
            + self.ς_prior.log_prob(ς)
            + self.φ_prior.log_prob(φ)
        )
        return llik + lprior

    def simulate(self, γ: float, η: float, σ: float, ρ: float, z0: float=None,
                 σz0: float=None):
        if z0 is None: z0 = 0
        if σz0 is None: σz0 = 1./(1 - ρ**2)**0.5
        z = torch.empty([self.input_length])
        z[0] = Normal(z0, σz0).sample()
        for i in range(1, self.input_length):
            z[i] = ρ*z[i-1] + Normal(0, 1).sample()
        y = Normal(γ + η*z, σ).sample()
        return y, z

    def kalman_smoother(self, y, ζ):
        z0, (σz0, ςz0), γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)

        z_pred = torch.zeros((self.input_length,))  # z_{t|t-1}
        z_upd = torch.zeros((self.input_length,))  # z_{t|t}
        Σz_pred = torch.zeros((self.input_length,))  # Sigma_{z_{t|t-1}}
        Σz_upd = torch.zeros((self.input_length,))  # Sigma_{z_{t|t}}
        y_pred = torch.zeros((self.input_length,))  # y_{t|t-1}
        Σy_pred = torch.zeros((self.input_length,))  # Sigma_{y_{t|t-1}}
        z_smooth = torch.zeros((self.input_length,))  # z_{t|T}
        Σz_smooth = torch.zeros((self.input_length,))  # Sigma_{z_{t|T}}

        # prediction step
        z_pred[0] = ρ * z0
        Σz_pred[0] = ρ**2 * σz0**2 + 1
        y_pred[0] = η * z_pred[0]
        Σy_pred[0] = η**2 * Σz_pred[0] + σ**2

        # correction step
        gain = Σz_pred[0] * η / Σy_pred[0]
        z_upd[0] = z_pred[0] + gain * (y[0] - y_pred[0])
        Σz_upd[0] = Σz_pred[0] - gain**2 * Σy_pred[0]

        for i in range(1, y.shape[0]):
            # prediction step
            z_pred[i] = ρ * z_upd[i-1]
            Σz_pred[i] = ρ**2 * Σz_upd[i-1] + 1
            y_pred[i] = η * z_pred[i]
            Σy_pred[i] = η**2 * Σz_pred[i] + σ**2
            # correction step
            gain = Σz_pred[i] * η / Σy_pred[i]
            z_upd[i] = z_pred[i] + gain * (y[i] - y_pred[i])
            Σz_upd[i] = Σz_pred[i] - gain**2 * Σy_pred[i]

        # smoothing step
        z_smooth[self.input_length - 1] = z_upd[self.input_length - 1]
        Σz_smooth[self.input_length - 1] = Σz_upd[self.input_length - 1]
        for i in range(self.input_length - 2, -1, -1):
            smooth = Σz_upd[i]*ρ / Σz_pred[i]
            z_smooth[i] = (
                z_upd[i] + smooth**2 * (Σz_pred[i + 1] - Σz_smooth[i])
            )
            Σz_smooth[i] = (
                Σz_upd[i] - smooth**2 * (Σz_pred[i + 1] - Σz_smooth[i + 1])
            )

        return {
            'z_upd': z_upd, 'Σz_upd': Σz_upd, 'z_smooth': z_smooth,
            'Σz_smooth': Σz_smooth, 'y_pred': y_pred, 'Σy_pred': Σy_pred
        }

    def sample_unobserved(self, ζ, y, fc_steps=0):
        z0, (σz0, ςz0), γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)

        # prediction step
        z_pred = ρ * z0
        Σz_pred = ρ**2 * σz0**2 + 1
        y_pred = η * z_pred
        Σy_pred = η**2 * Σz_pred + σ**2

        # correction step
        gain = Σz_pred * η / Σy_pred
        z_upd = z_pred + gain * (y[0] - y_pred)
        Σz_upd = Σz_pred - gain**2 * Σy_pred

        llik = Normal(y_pred, torch.sqrt(Σy_pred)).log_prob(y[0])

        for t in range(2, y.shape[0] + 1):
            i = t - 1
            # prediction step
            z_pred = ρ * z_upd
            Σz_pred = ρ**2 * Σz_upd + 1
            y_pred = η * z_pred
            Σy_pred = η**2 * Σz_pred + σ**2
            # correction step
            gain = Σz_pred * η / Σy_pred
            z_upd = z_pred + gain * (y[i] - y_pred)
            Σz_upd = Σz_pred - gain**2 * Σy_pred

            llik += Normal(y_pred, torch.sqrt(Σy_pred)).log_prob(y[i])

        return y

    # def sample_observed(self, ζ, fc_steps=0):
    #     z0, (σz0, _), γ, (η, _), (σ, _), (ρ, _) = self.unpack(ζ)
    #     if fc_steps > 0:
    #         z = torch.cat([z, torch.zeros(fc_steps)])
    #     # iteratively project states forward
    #     for t in range(self.input_length, self.input_length + fc_steps):
    #         z[t] = z[t - 1] * ρ + Normal(0, 1).sample()
    #     return Normal(γ + η * z, σ).sample()
