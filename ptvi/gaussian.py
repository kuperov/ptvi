import torch
from torch.distributions import Normal, ExpTransform, LogNormal
from ptvi import VIModel, ModelParameter, TransformedModelParameter


class UnivariateGaussian(VIModel):
    """
    Fit a simple univariate gaussian to an approximating density: q = N(u, LL').

    For the optimization, we transform σ -> ln(σ) = η to ensure σ > 0.
    """
    name = 'Univariate Gaussian model'
    params = [
        ModelParameter(name='μ', prior=Normal(0., 10.)),
        TransformedModelParameter(
            name='σ', prior=LogNormal(0., 10.),
            transformed_name='η', transform=ExpTransform().inv)
    ]

    def simulate(self, N: int, μ0: float, σ0: float):
        assert N > 2 and σ0 > 0
        return μ0 + σ0 * torch.randn((N,))

    def elbo_hat(self, y):
        E_ln_lik_hat, E_ln_pr_hat, H_q_hat = 0., 0., 0.  # accumulators

        for _ in range(self.num_draws):
            ζ = self.u + self.L@torch.randn((2,))
            μ, η = ζ[0], ζ[1]  # unpack drawn parameter
            σ = self.σ_to_η.inv(η)  # transform to user parameters
            E_ln_lik_hat += Normal(μ, σ).log_prob(y).sum()/self.num_draws
            E_ln_pr_hat += (self.μ_prior.log_prob(μ)
                            + self.η_prior.log_prob(η))/self.num_draws
            if self.stochastic_entropy:
                H_q_hat += self.q.log_prob(ζ)/self.num_draws
        if not self.stochastic_entropy:
            H_q_hat = self.q.entropy()
        return E_ln_lik_hat + E_ln_pr_hat - H_q_hat


if __name__ == '__main__' and '__file__' in globals():  # ie run as script
    import matplotlib.pyplot as plt
    torch.manual_seed(123)
    model = UnivariateGaussian(stochastic_entropy=True)
    N, μ0, σ0 = 100, 5., 5.
    y = model.simulate(N=N, μ0=μ0, σ0=σ0)
    result = model.training_loop(y)
    print(result.summary())
    result.plot_elbos()
    plt.show()
    plt.figure()
    result.plot_marg_post('μ', true_val=μ0)
    plt.show()
    result.plot_marg_post('σ', true_val=σ0)
    plt.show()
