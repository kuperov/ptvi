"""Variational Poisson regression.

See: [1] Arridge, S. R., Ito, K., Jin, B., & Zhang, C. (2018). Variational
     Gaussian approximation for Poisson data. Inverse Problems, 34(2).
     https://iopscience.iop.org/article/10.1088/1361-6420/aaa0ab/meta
"""

import numpy as np
from scipy import stats, special
from scipy.stats import multivariate_normal as mvn
# from sklearn.decomposition import TruncatedSVD  # <-- TODO: optimize with random SVD
import time
from ptvi.mvn_posterior import MVNPosterior
from ptvi.cavi.stanutil import cache_stan_model


def simulate(N, beta0, X=None):
    """Generate data for testing a poisson GLM.

    Args:
        N:      number of variates
        beta0:  true regression coefficient vector
        X:      covariate matrix (optional)

    Returns:
        (y, X) as a tuple
    """
    k = beta0.shape[0]
    assert k >= 1
    if X is None:
        X = stats.norm(0, 0.1).rvs(size=[N, k])
        X[:, 0] = 1.0
    else:
        assert X.shape == (N, k)
    eta = X @ beta0
    y = stats.poisson(np.exp(eta)).rvs()
    return (y, X)


def vi_reg(y, A, mu_0, C_0, tol=1e-10, maxiter=1000, maxNRiter=10):
    """Fit a poisson regression using VGA using [1].

    This optimization uses one fixed-point update and one
    newton-Raphson update.

    Args:
        y:       response variable
        A:       design matrix
        mu_0:    prior mean
        C_0:     prior covariance
        tol:     elbo convergence tolerance
        maxiter: maximum number of iterations
        maxNRiter: maximum Newton-Raphson iterations for x_bar

    Return:
        dict of result values
    """
    N, k, = A.shape
    assert y.shape == (N,) 
    assert C_0.shape == (k, k)
    assert mu_0.shape == (k,)
    old_elbo = None
    U, Sigma, V = np.linalg.svd(A, full_matrices=True)  # replace with rSVD
    C_0_inv = np.linalg.inv(C_0)
    C, x_bar = C_0, mu_0  # initial value for C
    m = C.shape[0]
    elbo_const = (
        -0.5 * np.linalg.slogdet(C_0)[1] + 0.5 * m - np.sum(special.gammaln(y + 1))
    )
    start_time = time.perf_counter()

    def print_status():
        print(f"{i:4d}. elbo = {elbo:.4f}, x_bar = {x_bar.round(2)}")

    for i in range(maxiter):
        # Newton-Raphson update for x_bar
        for j in range(maxNRiter):
            G = (
                A.T @ np.exp(A @ x_bar + 0.5 * np.diag(A @ C @ A.T))
                + C_0_inv @ (x_bar - mu_0)
                - A.T @ y
            )
            curlG = (
                A.T @ np.diag(np.exp(A @ x_bar + 0.5 * np.diag(A @ C @ A.T))) @ A + C_0_inv
            )
            delta_x = np.linalg.solve(curlG, -G)
            x_bar = x_bar + delta_x
            if np.linalg.norm(delta_x, 2) < 1e-1:
                break
        elbo = (
            y.T @ A @ x_bar
            - np.sum(np.exp(A @ x_bar + 0.5 * np.diag(A @ C @ A.T)))
            - 0.5 * (x_bar - mu_0).T @ C_0_inv @ (x_bar - mu_0)
            - 0.5 * np.trace(C_0_inv @ C)
            + 0.5 * np.linalg.slogdet(C)[1]
            + elbo_const
        )
        # fixed point update for C
        D = np.diag(np.exp(A @ x_bar + 0.5 * np.diag(A @ C @ A.T)))
        C = np.linalg.inv(C_0_inv + A.T @ D @ A)
        # elbo for monitoring convergence
        elbo = (
            y.T @ A @ x_bar
            - np.sum(np.exp(A @ x_bar + 0.5 * np.diag(A @ C @ A.T)))
            - 0.5 * (x_bar - mu_0).T @ C_0_inv @ (x_bar - mu_0)
            - 0.5 * np.trace(C_0_inv @ C)
            + 0.5 * np.linalg.slogdet(C)[1]
            + elbo_const
        )
        if old_elbo and old_elbo > elbo:
            print(f'BUG WARNING: elbo decreased from {old_elbo:.2f} to {elbo:.2f}.')
        if old_elbo and elbo - old_elbo < tol:
            end_time = time.perf_counter()
            print(f"Convergence detected in {1e3*(end_time - start_time):.3f} ms.")
            break
        old_elbo = elbo
        # if i % 5 == 0:
        print_status()
        if not np.isfinite(elbo):
            print_status()
            raise Exception('Infinite objective. Stopping.')
    else:
        print("WARNING: Maximum iterations reached.")
    print_status()
    return {"elbo": elbo, "C": C, "x_bar": x_bar, 'q': mvn(x_bar, C)}


def stan_reg(y, X, mu_0, C_0, num_draws=10_000, chains=1, warmup=1_000):
    """Fit a poisson regression using NUTS implemented with Stan.

    Args:
        y:         response variable
        X:         design matrix
        mu_0:      prior mean
        C_0:       prior covariance
        num_draws: number of posterior draws
        chains:    number of independent chains to run
        warmup:    number of warmup iterations

    Return:
        ndarray of mcmc draws, with params over dimension 0
    """
    N, k = X.shape
    assert y.shape == (N,)
    mdl = cache_stan_model("poisson.stan")
    y_ = y.astype(int)
    data = {"N": N, "k": k, "y": y_, "X": X, "mu_beta": mu_0, "Sigma_beta": C_0}
    start_t = time.perf_counter()
    fit = mdl.sampling(
        data=data, iter=warmup + num_draws // chains, warmup=warmup, chains=chains
    )
    end_t = time.perf_counter()
    print(f"Time elapsed excl. compilation: {(end_t - start_t):.4f}s")
    return fit


def mh_reg(y, X, mu_0, C_0, num_draws=10_000, warmup=1_000, init_tune=1e-2, autotune=True):
    """Fit a poisson regression using an adaptive Metropolis-Hastings algo.

    The Gaussian proposal density has covariance C_0 * tune. Tune is adjusted each
    100 steps during the warmup.

    Args:
        y:         response variable
        X:         design matrix
        mu_0:      prior mean
        C_0:       prior covariance
        num_draws: number of posterior draws
        warmup:    number of warmup iterations
        init_tune: initial tuning parameter value
        autotune:  if true, autotune during warmup

    Return:
        ndarray of mcmc draws, with params over dimension 0
    """
    N, k = X.shape
    assert y.shape == (N,)
    start_t = time.perf_counter()
    draws = np.empty(shape=[warmup + num_draws, k])
    prior = mvn(mu_0, C_0)
    unif = stats.uniform()
    tune = init_tune  # covariance tuning param for MH proposal
    proposal = mvn(mu_0, tune * C_0)
    beta = proposal.rvs()

    def ln_joint(beta):
        llhood = np.sum(stats.poisson(mu=np.exp(X @ beta)).logpmf(y))
        lprior = prior.logpdf(beta)
        return llhood + lprior

    autotune_s = 100
    for i in range(1, warmup + num_draws):
        in_warmup = (i <= warmup)
        # auto-tune so acceptance rate is near 0.23
        if autotune and in_warmup and i > autotune_s + 1 and i % autotune_s == 0:
            accept_rate = np.mean(np.where(draws[i - autotune_s:i, 0] != draws[i - autotune_s - 1:i - 1, 0], 1, 0))
            old_tune = tune
            tune += tune * (accept_rate - 0.23)
            print(f'Warmup auto-tuning: accept rate = {100*accept_rate:.1f}%, adjusting tune: {old_tune:.6f} -> {tune:.6f}')
        new_proposal = mvn(beta, tune * C_0)
        beta_prop = proposal.rvs()
        ln_alpha = ln_joint(beta_prop) - ln_joint(beta) + new_proposal.logpdf(beta) - proposal.logpdf(beta_prop)
        if np.exp(ln_alpha) > unif.rvs():
            beta = beta_prop
            proposal = new_proposal
        draws[i, ] = beta
    end_t = time.perf_counter()
    accept_rate = np.mean(np.where(draws[warmup + 1:, 0] != draws[warmup:-1, 0], 1, 0))
    print(f"Time elapsed: {(end_t - start_t):.4f}s, acceptance rate {100*accept_rate:.2f}%.")
    return draws[warmup:, :]


def simulate_ar(N, beta, phi, X=None, c=1e-3, rstate=None):
    """Simulate from poisson AR(p) process.

    Args:
        N:      Observations to simulate
        beta:   Regression coefficients
        phi:    Autoregression coefficients
        X:      Covariate matrix
        c:      Scalar threshold value 0<c<1 to replace zeros with
        rstate: Numpy random state

    Returns:
        Tuple of (y, X).
    """
    if rstate is None:
        rstate = np.random.RandomState(seed=123)
    k, p = len(beta), len(phi)
    if X is None:
        X = rstate.normal(size=[N, k])
        X[:, 0] = 1.0
    else:
        assert X.shape == (N, k), 'X should be a N*k array.'
    eta = (X @ beta).astype(np.float64)  # contribution from regression coeffts
    y = np.empty(shape=[N], dtype=np.float64)
    for i in range(N):
        for j in range(min(p, i)):  # autoregression contribution
            eta[i] += phi[j] * np.log(max(c, y[i - j - 1]))
        y[i] = rstate.poisson(np.exp(eta[i]))
    return y, X


def ar_design_matrix(y, X, p, c=1e-3):
    """Construct an AR(p) design matrix by adding lags of log(y) to X.

    Also shortens y by p observations.

    Args:
        y:  response vector
        X:  contemporaneous explanatory variables
        p:  Number of lags to add
        c:  Threshold value to replace zeros with

    Returns:
        y, (N-p)*(p+k) design matrix
    """
    lystar = np.log(np.maximum(y, c))
    y_lags = np.stack([lystar[p - i - 1 : -i - 1] for i in range(p)], axis=0).T
    X_ = np.block([X[p:, ], y_lags])
    y_ = y[p:]
    return (y_, X_)


def forecast_arp(y, X, fit, p, steps, c=0.2, num_draws=10_000, rs=None):
    """Construct forecast by simulation.

    Forecasts are presented as histograms describing the forecast pmf.

    Args:
        y:         observed data
        X:         covariate matrix
        p:         autoregression order
        fit:       inference results to use to construct forecast
        steps:     number of steps for forecast
        c:         threshold 0<c<1
        num_draws: number of simulation draws for forecast
        rs:        numpy random state

    Returns:
        Dict of forecast densities, keyed by observation number.
    """
    if rs is None:
        rs = np.random.RandomState(seed=123)
    assert 0.0 < c < 1.0
    N, k = X.shape
    y_ext = np.r_[y, np.zeros(steps)]
    max_x = np.ceil(np.max(y) * 2).astype(int)
    fc_hist = np.zeros([steps, max_x])
    x_hat = np.mean(X, axis=0)  # hold x at average
    if isinstance(fit, dict):
        q, draws = fit['q'], None
    elif isinstance(fit, np.ndarray):
        draws, q = fit, None
    elif isinstance(fit, MVNPosterior):
        fit_q = getattr(fit, 'q')
        mean = fit_q.mean.cpu().numpy()
        cov = fit_q.covariance_matrix.cpu().numpy()
        draws, q = None, mvn(mean, cov)
    for i in range(num_draws):
        params = q.rvs() if q is not None else draws[i, :]
        beta, phi = params[:k], params[k:]
        for j in range(steps):
            eta = x_hat @ beta
            for k in range(p):
                eta += phi[k] * np.log(max(c, y_ext[N + j - k - 1]))
            try:
                y_hat = rs.poisson(lam=np.exp(eta))
            except ValueError:
                print(f'Value error with eta={eta}, exp(eta)={np.exp(eta)}')
                import traceback
                traceback.print_exc()
            if (y_hat >= fc_hist.shape[1]):
                # double the histogram size (or expand to y_hat)
                new_elem = max(fc_hist.shape[1] * 2, y_hat + 1) - fc_hist.shape[1]
                fc_hist = np.pad(fc_hist, mode='constant', constant_values=0.0,
                                 pad_width=[[0, 0], [0, new_elem]])
            fc_hist[j, y_hat] += 1.0
            y_ext[N + j] = y_hat
    # normalize histogram, indexed by 0-based observation number
    fcs = {N + j: fc_hist[j, :] / np.sum(fc_hist[j, :]) for j in range(steps)}
    return fcs
