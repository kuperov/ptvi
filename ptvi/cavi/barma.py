"""Binary/binomial autoregressive moving average model.

"""
import numpy as np
from .logistic import logit, r_trunc_logistic_sim, l_trunc_logistic_sim


def barma_gibbs(y, X, p, q, ndraws=10000, warmup=1000):
    m, n = max(p, q), len(y)

    # 7.1 - initialization
    eta_star = np.copy(y)
    eta_star[:m] = np.mean(y)
    mu = np.random.uniform(size=y.shape)
    mu[:m] = np.mean(y)
    eta_bar = -np.log((1 - mu) / mu)

    draws = np.empty(shape=[ndraws + warmup, 1 + p + q])
    # fill the 1 and the AR terms in X, populate MA in each iteration
    X = np.empty(shape=[len(y) - m, 1 + p + q])
    X[:, 0] = 1.0
    for lag in range(1, p):
        X[:, lag] = y[m - lag : -lag]

    for i in range(ndraws + warmup):
        # 7.2 - draw BARMA parameters
        XX_inv = np.linalg.inv(X.T @ X)
        b_mean = XX_inv @ X.T @ eta_star
        b_cov = (np.pi ** 2) / 3 * XX_inv
        b_tilde = np.random.multivariate_normal(mean=b_mean, cov=b_cov, size=1)
        beta_0, phi, theta = b_tilde[0], b_tilde[1 : (p + 1)], b_tilde[(p + 1) :]

        # 7.3 - draw mu_star
        # iteratively compute mu_star given parameters
        for t in range(m, n):
            # notice we reverse the direction of the subset of y prior to
            # multiplication by drawn coefficients
            eta_bar[t] = (
                beta_0
                + phi @ y[(t - 1) : (t - m - 1) : -1]
                + mu[(t - 1) : t(t - m - 1)] @ theta
            )
        mu[t] = logit(eta_bar)

        # 7.4 - draw latent eta_star
        eta_star = np.which(
            y == 0,
            r_trunc_logistic_sim(n, mu=eta_bar),
            l_trunc_logistic_sim(n, mu=eta_bar),
        )

        # record drawn parameters and move on
        draws[i, :] = np._r[b_tilde]

    return draws[warmup:, :]
