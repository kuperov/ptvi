import math
import torch
from torch.distributions import Normal, Categorical


class StateSpaceModel(object):

    def simulate(self, T):
        raise NotImplementedError

    def conditional_log_prob(self, t, x_1t, z_1t):
        raise NotImplementedError


class PFProposal(object):

    def conditional_sample(self, t, Z):
        raise NotImplementedError

    def conditional_log_prob(self, t, Z):
        raise NotImplementedError


class StochasticVolatilityModel(StateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)
    """
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def simulate(self, T):
        """Simulate from p(x, z | θ)"""
        z_true = torch.empty((T,))
        z_true[0] = Normal(self.b, (1-self.c**2)**(-.5)).sample()
        for t in range(1, T):
            z_true[t] = self.b + self.c * z_true[t-1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(self.a) * torch.exp(z_true/2)).sample()
        return x, z_true

    def conditional_log_prob(self, t, x_1t, z_1t):
        """Compute log p(x_t, z_t | x_{1:t}, z_{1:t}, θ)"""
        if t == 0:
            log_pzt = Normal(self.b, (1-self.c**2)**(-.5)).log_prob(z_1t[t])
        else:
            log_pzt = Normal(self.b + self.c * z_1t[t], 1).log_prob(z_1t[t])
        log_pxt = Normal(0, torch.exp(self.a) * torch.exp(z_1t[t]/2)).log_prob(x_1t[t])
        return log_pzt + log_pxt

    def __repr__(self):
        return ("Stochastic volatility model\n"
            f"x_t = exp({self.a:.2f} * z_t/2) ε_t\n"
            f"z_t = {self.b:.2f} + {self.c:.2f} * z_{{t-1}} + ν_t\n"
            f"ε_t ~ Ν(0,1)\n"
            f"ν_t ~ Ν(0,1)")


class AR1Proposal(PFProposal):
    """A simple linear/gaussian AR(1) to use as a particle filter proposal.

    .. math::
        z_t = μ + ρ * z_{t-1} + η_t
    """
    def __init__(self, μ, ρ):
        self.μ, self.ρ = μ, ρ

    def conditional_sample(self, t, Z):
        """Simulate z_t from q(z_t | z_{t-1}, φ)

        Z has an extra dimension, of N particles.
        """
        N = Z.shape[1]
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2)**(-.5)).sample((N,))
        else:
            return Normal(self.μ + self.ρ * Z[t-1], 1).sample()

    def conditional_log_prob(self, t, Z):
        """Compute log q(z_t | z_{t-1}, φ)"""
        if t == 0:
            return Normal(0, (1 - self.ρ ** 2) ** (-.5)).log_prob(Z[t])
        else:
            return Normal(self.μ + self.ρ * Z[t - 1], 1).log_prob(Z[t])

    def __repr__(self):
        return ("AR(1) proposals\n"
            f"z_t = {self.μ:.2f} + {self.ρ:.2f} * z_{{t-1}} + η_t\n"
            f"η_t ~ Ν(0,1)"
        )


def simulate_FIVO(x: torch.Tensor, p: StateSpaceModel, q: PFProposal, N: int,
                  resample=False, rewrite_history=False):
    T, log_phatN = x.shape[0], 0
    log_w = torch.tensor([math.log(1/N)]*N)
    Z, resampled = torch.zeros((T, N)), [False] * T
    for t in range(T):
        Z[t, :] = q.conditional_sample(t, Z)
        log_αt = p.conditional_log_prob(t, x, Z) - q.conditional_log_prob(t, Z)
        log_phatt = torch.logsumexp(log_w + log_αt, dim=0)
        log_phatN += log_phatt
        log_w += log_αt - log_phatt
        ESS = 1./torch.exp(2*log_w).sum()
        if ESS < N and resample:
            resampled[t] = True
            a = Categorical(torch.exp(log_w)).sample((N,))
            if rewrite_history:
                Z = Z[:, a]
            else:
                Z[t, :] = Z[t, a]
            log_w = torch.tensor([math.log(1 / N)] * N)
        # resamp = '*' if ESS < N/2 else ''
        # print(f'{t:4d}. log_phatN = {log_phatN:.2f}  ESS={ESS:.1f}{resamp}')
    return log_phatN, Z, resampled


if __name__ == '__main__' and '__file__' in globals():
    import matplotlib.pyplot as plt
    torch.manual_seed(123)
    T = 200
    p_true = StochasticVolatilityModel(a=1., b=0., c=0.5)
    x, z_true = p_true.simulate(T)
    p = StochasticVolatilityModel(a=1., b=0., c=0.5)
    q = AR1Proposal(μ=0., ρ=0.5)
    log_phatN, Z, resamp = simulate_FIVO(x, p, q, N=10, resample=True)
    plt.plot(Z.numpy(), alpha=0.1)
    plt.plot(z_true.numpy(), color='black', label='true z')
    plt.legend()
    plt.show()
