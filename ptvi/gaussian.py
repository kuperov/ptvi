import torch
from torch.distributions import *
from time import time
import matplotlib.pyplot as plt

# just fit μ, σ for a gaussian

torch.manual_seed(123)

N = 5000
dgp = torch.distributions.Normal(torch.tensor([5.0]), torch.tensor([5.0]))
data = dgp.sample((N,))
batches = data[torch.randperm(N)]

print(f'mean = {torch.mean(data):.2f}, sd = {torch.sqrt(torch.var(data)):.2f}')

# approximating density: q(u, LL')
d = 2
u = torch.tensor(torch.zeros(d), requires_grad=True)
L = torch.tensor(torch.eye(d), requires_grad=True)

η_dist = torch.distributions.Normal(0, 1)
μ_prior = Normal(0, 10)
lnσ_prior = Normal(0, 10)


def elbo_hat_analytic_entropy(u, L):
    η = η_dist.sample(torch.Size([2]))
    L = torch.tril(L)
    ζ = u + L@η
    μ, lnσ, σ = ζ[0], ζ[1], torch.exp(ζ[1])
    log_likelihood = Normal(μ, σ).log_prob(data).sum()
    log_prior = μ_prior.log_prob(μ) + lnσ_prior.log_prob(lnσ)
    log_jac_adj = torch.log(N * σ)
    q = MultivariateNormal(u, scale_tril=L)
    return log_likelihood + log_prior + log_jac_adj - q.entropy()


def elbo_hat_stochastic_entropy(u, L):
    η = η_dist.sample(torch.Size([2]))
    L = torch.tril(L)
    ζ = u + L@η
    μ, lnσ, σ = ζ[0], ζ[1], torch.exp(ζ[1])
    log_likelihood = Normal(μ, σ).log_prob(data).sum()
    log_prior = μ_prior.log_prob(μ) + lnσ_prior.log_prob(lnσ)
    log_jac_adj = torch.log(N * σ)
    q = MultivariateNormal(u, scale_tril=L)
    return log_likelihood + log_prior + log_jac_adj - q.log_prob(ζ)


def print_status(i, objective):
    μ_hat, ln_σ_hat, σ_hat = u[0], u[1], torch.exp(u[1])
    sds = torch.sqrt(torch.diag(L @ L.t()))
    μ_sd, ln_σ_sd = sds[0], sds[1]
    σ_sd = torch.sqrt(torch.exp(2 * ln_σ_hat + ln_σ_sd ** 2) *
                      (torch.exp(ln_σ_sd ** 2) - 1))
    print(f'{i: 8d}. ll ={-objective:8.2f}  μ ={μ_hat: 4.2f} ({μ_sd:4.2f}), '
          f'ln σ ={ln_σ_hat: 4.2f} ({ln_σ_sd:4.2f}), '
          f'σ ={σ_hat: 4.2f} ({σ_sd:4.2f})')


def fit(elbo_hat, lg_iters=20):
    t = -time()
    elbo_hats = []
    optimizer = torch.optim.RMSprop([u, L])
    for i in range(2**lg_iters):
        optimizer.zero_grad()
        objective = -elbo_hat(u, L)
        objective.backward()
        optimizer.step()
        elbo_hats.append(-objective)
        if not i & (i - 1):
            print_status(i, objective)
    t += time()
    print_status(i+1, objective)
    print(f'Completed {i+1} iterations in {t:.2f}s @ {i/(t+1):.2f} iter/s.')
    return elbo_hats


print('*** Analytic entropy term ***')
elbo_hats = fit(elbo_hat_analytic_entropy, lg_iters=14)
plt.plot(elbo_hats)
plt.show()


d = 2
u = torch.tensor(torch.zeros(d), requires_grad=True)
L = torch.tensor(torch.eye(d), requires_grad=True)


print('*** Stochastic entropy term ***')
elbo_hats = fit(elbo_hat_stochastic_entropy, lg_iters=14)
plt.plot(elbo_hats)
plt.show()
