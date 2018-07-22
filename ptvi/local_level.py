import torch
from torch.distributions import *
import matplotlib.pyplot as plt
from ptvi.stopping import *
from ptvi.dist import InvGamma
from ptvi import (VIModel, VITimeSeriesResult, ModelParameter, LocalParameter,
                  TransformedModelParameter)


class LocalLevelModel(VIModel):
    # approximating density: q(u, LL')

    result_class = VITimeSeriesResult
    name = 'Local level model'
    params = [
        LocalParameter(name='z', prior=Beta(0, 0)),
        ModelParameter(name='γ', prior=Normal(0, 3)),
        TransformedModelParameter(
            name='η', prior=Normal(0, 3), transformed_name='ψ',
            transform=ExpTransform().inv),
        TransformedModelParameter(
            name='σ', prior=InvGamma(1, 5), transformed_name='ς',
            transform=ExpTransform().inv),
        TransformedModelParameter(
            name='ρ', prior=Beta(1, 1), transformed_name='φ',
            transform=SigmoidTransform().inv)
    ]

    def elbo_hat(self, y):
        L = torch.tril(self.L)
        E_ln_lik_hat, E_ln_pr_hat, H_q_hat = 0., 0., 0.  # accumulators

        q = MultivariateNormal(loc=self.u, scale_tril=L)
        if not self.stochastic_entropy:
            H_q_hat = q.entropy()

        for _ in range(self.num_draws):
            ζ = self.u + L@torch.randn((self.d,))  # reparam trick
            z, γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)
            ar1_uncond_var = torch.pow((1 - torch.pow(ρ, 2)), -0.5)
            llikelihood = (
                Normal(γ + η * z, σ).log_prob(y).sum()
                + Normal(ρ * z[:-1], 1).log_prob(z[1:]).sum()
                + Normal(0., ar1_uncond_var).log_prob(z[0])
            )
            E_ln_lik_hat += llikelihood/self.num_draws
            lprior = (
                self.γ_prior.log_prob(γ)
                + self.η_prior.log_prob(η)
                + self.ς_prior.log_prob(ς)
                + self.φ_prior.log_prob(φ)
            )
            E_ln_pr_hat += lprior/self.num_draws
            if self.stochastic_entropy:
                H_q_hat += q.log_prob(ζ)/self.num_draws

        return E_ln_lik_hat + E_ln_pr_hat - H_q_hat

    def simulate(self, γ: float, η: float, σ: float, ρ: float):
        z = torch.empty([self.input_length])
        z[0] = Normal(0, 1/(1 - ρ**2)**0.5).sample()
        for i in range(1, self.input_length):
            z[i] = ρ*z[i-1] + Normal(0, 1).sample()
        y = Normal(γ + η*z, σ).sample()
        return y, z

    def sample_paths(self, N=100, fc_steps=0):
        """Sample N paths from the model, forecasting fc_steps additional steps.

        Args:
            N:        number of paths to sample
            fc_steps: number of steps to project forward

        Returns:
            Nx(τ+fc_steps) tensor of sample paths
        """
        paths = torch.empty((N, self.input_length + fc_steps))
        ζs = MultivariateNormal(self.u, scale_tril=self.L).sample((N,))
        _τ = self.input_length
        for i in range(N):
            ζ = ζs[i]
            z, γ, (η, ψ), (σ, ς), (ρ, φ) = self.unpack(ζ)
            if fc_steps > 0:
                z = torch.cat([z, torch.zeros(fc_steps)])
            # project states forward
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = z[t-1]*ρ + Normal(0, 1).sample()
            paths[i, :] = Normal(γ + η*z, σ).sample()
        return paths


if __name__ == '__main__' and '__file__' in globals():
    torch.manual_seed(123)
    γ0, η0, σ0, ρ0 = 0., 2., 1.5, 0.92
    y, z = LocalLevelModel(input_length=100).simulate(γ=γ0, η=η0, σ=σ0, ρ=ρ0)

    m = LocalLevelModel(input_length=100, stochastic_entropy=True, num_draws=1,
                        stop_heur=NoImprovementStoppingHeuristic())
    m.optimizer = torch.optim.Adadelta(m.parameters)
    fit = m.training_loop(y)
    print(fit.summary())
    fit.plot_elbos()
    plt.show()
    fit.plot_latent(true_z=z.numpy(), include_data=True)
    plt.show()
    # fit.plot_elbos()
    # plt.show()
    # fit.plot_sampled_paths(200, true_y=y, fc_steps=10)
    # plt.show()
    # fit.plot_pred_ci(true_y=y, fc_steps=10)
    # plt.show()
