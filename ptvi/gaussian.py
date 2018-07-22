import torch
from torch.distributions import (
    Normal, ExpTransform, LogNormal, MultivariateNormal)
from ptvi import VIModel, global_param


class UnivariateGaussian(VIModel):
    """
    Fit a simple univariate gaussian to an approximating density: q = N(u, LL').

    For the optimization, we transform σ -> ln(σ) = η to ensure σ > 0.
    """
    name = 'Univariate Gaussian model'
    μ = global_param(prior=Normal(0., 10.))
    σ = global_param(prior=LogNormal(0., 10.), rename='η', transform='log')

    def simulate(self, N: int, μ0: float, σ0: float):
        assert N > 2 and σ0 > 0
        return Normal(μ0, σ0).sample((N,))

    def elbo_hat(self, y):
        L = torch.tril(self.L)
        E_ln_lik_hat, E_ln_pr_hat, H_q_hat = 0., 0., 0.  # accumulators

        q = MultivariateNormal(loc=self.u, scale_tril=L)
        if not self.stochastic_entropy:
            H_q_hat = q.entropy()

        for _ in range(self.num_draws):
            ζ = self.u + L@torch.randn((2,))
            μ, (σ, η) = self.unpack(ζ)
            E_ln_lik_hat += Normal(μ, σ).log_prob(y).sum()/self.num_draws
            E_ln_pr_hat += (self.μ_prior.log_prob(μ)
                            + self.η_prior.log_prob(η))/self.num_draws
            if self.stochastic_entropy:
                H_q_hat += self.q.log_prob(ζ)/self.num_draws
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
