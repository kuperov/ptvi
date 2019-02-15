"""Binary/binomial autoregressive moving average model.

"""
from math import log
import numpy as np
from .logistic import logit, r_trunc_logistic_sim, l_trunc_logistic_sim
from scipy import stats

from ptvi.cavi import probit


def logit_barma_gibbs(y, X, p, q, ndraws=10000, warmup=1000):
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


def simulate_bar(N, beta0, phi0, X=None):
    """Simulate a BAR(p) process.

    Args:
        N:     length of simulated series
        beta0: true covariate vector
        phi0:  true autoregressive vector
        X:     covariate matrix (optional)

    Returns:
        Tuple of (BAR(p) realization, design matrix)
    """
    k, p = len(beta0), len(phi0)
    if X is not None:
        X_ = X.copy()
        assert X_.shape == (N, k)
    else:
        X_ = np.random.normal(size=[N, k])
        X_[:, 0] = 1.0  # first col is constant, needs 1s
    Phi = stats.norm().cdf  # link function
    y = np.empty(N)
    eta = X_ @ beta0
    for i in range(N):
        for j in range(p):
            if i - j - 1 >= 0:
                eta[i] += y[i - j - 1] * phi0[j]
            y[i] = np.random.binomial(n=1, p=Phi(eta[i]), size=1)
    return (y, X_)


def bar_design_matrix(y, X, p):
    """Construct a BAR(p) design matrix by adding lags of y to X.

    Also shortens y by p observations.

    Args:
        y:  response vector
        X:  contemporaneous explanatory variables
        p:  Number of lags to add

    Returns:
        y, (N-p)*(p+k) design matrix
    """
    y_lags = np.stack([y[p - i - 1 : -i - 1] for i in range(p)], axis=0).T
    X_ = np.block([X[p:,], y_lags])
    y_ = y[p:]
    return (y_, X_)


def combine_bar_priors(mu_beta, mu_phi, Sigma_beta, Sigma_phi):
    """Combine priors for mu and phi for use in probit regression functions.

    Args:
        mu_beta:    mean of beta prior
        mu_phi:     mean of phi prior
        Sigma_beta: covariance of beta prior
        Sigma_phi:  covariance of phi prior

    Returns:
        tuple: mu_both, Sigma_both
    """
    p, k = len(mu_beta), len(mu_phi)
    mu_both = np.r_[mu_beta, mu_phi]
    Sigma_both = np.block(
        [[Sigma_beta, np.zeros([k, p])], [np.zeros([p, k]), Sigma_phi]]
    )
    return mu_both, Sigma_both


def binary_forecast(y, X, p, fits, steps=10, M=1000):
    """Forecast binary series using PROBIT parameterization of BAR model.

    Args:
        y:      observed data
        X:      past explanatory variables (excluding lagged ys)
        p:      number of lags of dependent variable (dimension of phi)
        fits:   list of dicts returned by vi_probit() OR matrix of draws
        steps:  number of steps to project series foreward
        M:      number of simulation draws (ignored for matrices of draws)

    Returns:
        List of dictionary of proportions, indexed by time [N+1, ..., N+steps].
    """
    N, k, = X.shape
    x_bar = np.mean(X[:, :k], axis=0)  # hold x fixed at mean
    y_ext = np.r_[y, np.empty([steps])]  # forecast placeholder
    nfits = len(fits)
    Phi = stats.norm().cdf  # link function
    fc_total = np.zeros([nfits, steps])
    for l in range(nfits):
        if isinstance(fits[l], np.ndarray):
            # mcmc draws
            q_beta = None
            m = fits[l].shape[0]
        else:
            # variational fit
            q_beta = fits[l]["q_beta"]
            m = M
        for j in range(m):
            # m draws, each conditioned on a single draw from q(beta,phi)
            if isinstance(fits[l], np.ndarray):
                param = fits[l][j, :]
            else:
                param = q_beta.rvs()
            beta, phi = param[:k], param[k:]
            for i in range(N, N + steps):
                eta = x_bar.dot(beta)
                for j in range(p):
                    if i - j - 1 >= 0:
                        eta += y_ext[i - j - 1] * phi[j]
                    y_ext[i] = np.random.binomial(n=1, p=Phi(eta), size=1)
            fc_total[l,] += y_ext[-steps:]
        fc_total[l,] /= m
    return fc_total


def score_binary_forecast(y, X, p, beta0, phi0, fcs, labs, M=1000):
    """Score forecasts of binary series using PROBIT parameterization of BAR(p) model.

    The forecasts should be 1-dimensional vectors of length `steps`.

    To ensure results are reproducible, be sure to set the seed before calling this method.

    Args:
        y:      observed data
        X:      past explanatory variables (excluding lagged ys)
        p:      number of lags of dependent variable (dimension of phi)
        fcs:    list of forecasts for i=n+1, dots, n+steps
        steps:  number of steps to project series foreward
        labs:   labels for forecasts
        M:      number of simulation draws (ignored for matrices of draws)

    Returns:
        Pandas data frame of scores, indexed by time [N+1, ..., N+steps].
    """
    import pandas as pd

    N, k, = X.shape
    p = len(phi0)
    steps = fcs[0].shape[0]  # fcs is 1-dimensional
    nfcs = len(fcs)
    y_ext = np.r_[y, np.empty([steps])]  # forecast placeholder
    Phi = stats.norm().cdf  # link function
    scores = np.zeros([nfcs, steps])
    for j in range(M):
        # M times, we project the series forward
        if p > 1:
            x = np.r_[1.0, np.random.normal(p - 1)]  # unpredictable X
        else:
            x = np.r_[1.0]
        for i in range(N, N + steps):
            eta = x @ beta0
            for k in range(p):
                if i - k - 1 >= 0:
                    eta += y_ext[i - k] * phi0[k]
                y_ext[i] = np.random.binomial(n=1, p=Phi(eta), size=1)
        # now compare each forecast to this realization, and add log score
        # to the corresponding score array
        for l in range(nfcs):
            # m draws, each conditioned on a single draw from q(beta,phi)
            fc = fcs[l]
            for i in range(steps):
                prob1, prob0 = fc[i], 1.0 - fc[i]
                scores[l, i] += (
                    np.log(prob1) if (y_ext[N + i] == 1.0) else np.log(prob0)
                )
    scores /= M
    rowlabs = [f"y[N+{i}]" for i in range(1, steps + 1)]
    return pd.DataFrame(scores.T, columns=labs, index=rowlabs)
