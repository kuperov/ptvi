"""Variational probit regression.
"""

from time import perf_counter
import os

import numpy as np
from numpy.linalg import inv
from scipy import stats
from scipy.stats import multivariate_normal as mvn

from ptvi.cavi.tnorm import tnorm


def logdet(x):
    return np.linalg.slogdet(x)[1]


def vi_probit(y, X, mu_beta=None, Sigma_beta=None, maxiter=1000, tol=1e-2):
    """Variational probit regression.

    Args:
        y:          binary response variable (n-array-like of 1 and 0s)
        X:          covariate matrix (n*p-array-like of floats)
        mu_beta:    prior mean for beta (k-array-like of floats)
        Sigma_beta: prior variance for beta (k*k-array-like of floats)
        maxiter:    maximum number of iterations
        tol:        elbo tolerance for detecting convergence

    Returns:
        dict of results, including q(beta) approx posterior, a stats.distribution object
    """
    start_time = perf_counter()
    N, k = X.shape
    assert y.shape == (N,)
    if mu_beta is None:
        mu_beta = np.zeros(k)
    if Sigma_beta is None:
        Sigma_beta = np.eye(k)
    assert Sigma_beta.shape == (k, k) and mu_beta.shape == (k,)
    XX = X.T @ X
    Sigma_beta_inv = inv(Sigma_beta)
    Sigma_q_beta = inv(XX + Sigma_beta_inv)  # constant
    mu_q_a = X @ Sigma_q_beta @ X.T @ y  # start at OLS solution
    old_elbo = None
    std_norm = stats.norm()
    elbo_const = (
        -0.5 * N
        - 0.5 * logdet(2 * np.pi * Sigma_beta)
        + 0.5 * logdet(2 * np.pi * np.e * inv(XX + Sigma_beta_inv))
    )

    for i in range(maxiter):
        # update mean of q(beta)
        mu_q_beta = Sigma_q_beta @ (X.T @ mu_q_a + Sigma_beta_inv @ mu_beta)
        # update mean of q(a)
        mu = X @ mu_q_beta
        phi = std_norm.pdf(-mu)
        Phi = std_norm.cdf(-mu)
        mu_q_a = X @ mu_q_beta + phi / np.where(y == 1, 1.0 - Phi, -Phi)
        # need elbo to detect convergence
        elbo = (
            elbo_const
            - 0.5 * mu_q_beta.T @ XX @ mu_q_beta
            + mu_q_a.T @ mu
            - 0.5
            * np.trace(
                (XX + Sigma_beta_inv) @ (np.outer(mu_q_beta, mu_q_beta) + Sigma_q_beta)
            )
            - np.sum(X @ mu_q_beta * phi / np.where(y == 1, 1.0 - Phi, -Phi))
            + mu_beta.T @ Sigma_beta_inv @ (mu_q_beta - 0.5 * mu_beta)
            + np.sum(np.log(np.where(y == 1, 1.0 - Phi, Phi)))
        )
        print(f"{i+1:4d}. elbo: {elbo:.4f}, E[beta|y]: {str(mu_q_beta.round(2))}")
        if old_elbo and elbo < old_elbo:
            print(f"Bug warning: elbo decreased from {old_elbo:.4f} to {elbo:.4f}.")
        elif old_elbo and elbo < old_elbo + tol:
            elapsed_ms = 1e3 * (perf_counter() - start_time)
            print(f"Convergence achieved in {elapsed_ms:.4f} ms after {i+1} iterations.")
            break
        old_elbo = elbo
    else:
        print("Warning: maximum iterations reached; convergence not detected.")
    sd_q_beta = np.sqrt(np.diag(Sigma_q_beta))
    print(f" E[beta|y]: {str(mu_q_beta.round(2))}")
    print(f"sd[beta|y]: {str(sd_q_beta.round(2))}")
    q_beta = stats.multivariate_normal(mean=mu_q_beta, cov=Sigma_q_beta)
    q_a = tnorm(
        mu=X @ mu_q_beta,
        sigma=1.0,
        a=np.where(y == 1, 0, -np.inf),
        b=np.where(y == 1, np.inf, 0),
    )

    def q_y_hat(x):
        return stats.bernoulli(p=std_norm.cdf(x @ mu_q_beta))  # predictive distribution

    return {
        "q_beta": q_beta,
        "q_a": q_a,
        "elbo": elbo,
        "mu_q_beta": mu_q_beta,
        "Sigma_q_beta": Sigma_q_beta,
        "mu_q_a": mu_q_a,
        "q_y_hat": q_y_hat
    }


def gibbs_probit(y, X, mu_beta=None, Sigma_beta=None, num_draws=10_000, warmup=1000):
    """Gibbs sampler for probit regression.

    Args:
        y:          binary response variable (n-array-like of 1 and 0s)
        X:          covariate matrix (n*p-array-like of floats)
        mu_beta:    prior mean for beta (k-array-like of floats)
        Sigma_beta: prior variance for beta (k*k-array-like of floats)
        num_draws:  number of draws to return
        warmup:     number of warmup draws to discard

    Returns:
        array of draws of beta
    """
    N, k = X.shape
    assert y.shape == (N,)
    XX = X.T @ X
    Sigma_beta_inv = inv(Sigma_beta)
    Sigma_q_beta = inv(XX + Sigma_beta_inv)
    a = np.random.normal(size=N)
    draws = np.empty([num_draws + warmup, k])
    for i in range(num_draws + warmup):
        # update mean of q(beta)
        mu_q_beta = Sigma_q_beta @ (X.T @ a + Sigma_beta_inv @ mu_beta)
        beta = mvn(mu_q_beta, Sigma_q_beta).rvs()
        # update mean of q(a)
        q_a_lower = np.where(y == 1, 0, -np.inf)
        q_a_upper = np.where(y == 1, np.inf, 0)
        a_dist = tnorm(mu=X @ beta, sigma=1.0, a=q_a_lower, b=q_a_upper)
        a = a_dist.rvs()
        # only store beta draws, don't care about a's
        draws[i] = beta
    return draws[warmup:]


def stan_probit(y, X, mu_beta=None, Sigma_beta=None, 
    num_draws=10_000, chains=1, warmup=1000):
    """Gibbs sampler for probit regression.

    Args:
        y:          binary response variable (n-array-like of 1 and 0s)
        X:          covariate matrix (n*p-array-like of floats)
        mu_beta:    prior mean for beta (k-array-like of floats)
        Sigma_beta: prior variance for beta (k*k-array-like of floats)
        num_draws:  number of draws to return
        chains:     number of chains to run
        warmup:     number of warmup draws to discard

    Returns:
        array of draws of beta
    """
    # import pystan loaclly so if the environment is broken it doesn't
    # hose the entire module
    import pystan
    N, k = X.shape
    assert y.shape == (N,)
    XX = X.T @ X
    Sigma_beta_inv = inv(Sigma_beta)
    Sigma_q_beta = inv(XX + Sigma_beta_inv)
    stanfile = os.path.join(os.path.dirname(__file__), 'probit.stan')
    mdl = pystan.StanModel(file=stanfile)
    data = {'N': N, 'k': k, 'y': y, 'X': X, 'mu_beta': mu_beta, 'Sigma_beta': Sigma_beta}
    start_t = perf_counter()
    fit = mdl.sampling(data=data, iter=warmup+num_draws//chains,
        warmup=warmup, chains=chains)
    end_t = perf_counter()
    print(f'Time elapsed excl. compilation: {(end_t - start_t):.4f}s')
    return fit


def log_joint(y, X, beta, a, mu_beta, Sigma_beta):
    """Log joint, for checking analytical ELBO."""
    N, k, = X.shape
    lj = 0
    # contribution from p(y|a) is zero because log(1) = 0
    # p(a|beta)
    lj += stats.multivariate_normal(mean=X @ beta, cov=np.eye(N)).logpdf(a)
    # p(beta)
    lj += stats.multivariate_normal(mean=mu_beta, cov=Sigma_beta).logpdf(beta)
    return lj


if __name__ == "__main__":
    np.random.seed(123)
    N, k = 500, 10
    X = np.random.normal(size=[N, k])
    mu_beta, Sigma_beta = np.zeros(k), np.eye(k)
    beta0 = np.random.normal(size=k)
    eta = X @ beta0
    Phi = stats.norm().cdf
    y = np.random.binomial(n=1, p=Phi(eta), size=N)
    print(f"True beta = {beta0.round(2)}")
    fit = vi_probit(y, X, maxiter=1000, mu_beta=mu_beta, Sigma_beta=Sigma_beta)
    # check lj can be evaluated
    q_beta, q_a = fit["q_beta"], fit["q_a"]
    beta, a = q_beta.rvs(), q_a.rvs()
    lj = log_joint(y, X, beta, a, mu_beta, Sigma_beta)

