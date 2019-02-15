"""Variational regression model [3] in MCMC and CAVI.

We use Stan [0] to implement a NUTS sampler [1].

Cross-check: a Metropolis algorithm produces identical results.

Windows users: see the pystan documentation[2] to get your environment set up.

Refs:
  [0] https://journals.sagepub.com/doi/abs/10.3102/1076998615606113
  [1] https://arxiv.org/abs/1111.4246
  [2] https://pystan.readthedocs.io/en/latest/windows.html
  [3] https://arxiv.org/abs/1310.5438
"""
import os
import pickle
from warnings import warn
import time
from multiprocessing import Pool

import pystan as ps
from os import path
import numpy as np
from numpy import log, sqrt, pi as π
from numpy.linalg import pinv, cholesky as chol, inv, slogdet
from scipy import stats
from scipy.stats import multivariate_normal as Φ, gamma as Γ
from scipy.special import loggamma


# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def _get_stan_model():
    """Load and compile the Stan model.

    This function can take a while the first time it is run. The returned model object
    is reusable for multiple inference runs, including with different data and hyper-
    parameters.
    """
    stan_file = path.join(path.dirname(__file__), "bayesian_reg.stan")
    pickle_file = path.join(path.dirname(__file__), "bayesian_reg.stan.pkl")
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as fp:
            sm = pickle.load(fp)
    else:
        sm = ps.StanModel(file=stan_file, model_name="bayes_reg")
        with open(pickle_file, "wb") as fp:
            pickle.dump(sm, fp)
    return sm


def stan_reg(
    y, X, a_0, b_0, c_0, d_0, warmup=1000, draws=1000, chains=4, verbose=False
):
    """Use NUTS to draw from posterior for simple hierarchical model

    Model is Drugowitsch's non-ARD VB model in https://arxiv.org/abs/1310.5438.

    Args:
        y:        univariate time series
        X:        explanatory variables; typically the first column is 1
        a_0, b_0: prior for τ
        c_0, d_0: prior for α
        warmup:   number of warmup draws
        draws:    number of draws to return (in addition to warmup)
        verbose:  suppress output

    Returns:
        draws
    """
    model = _get_stan_model()
    N, k = X.shape
    assert len(y) == N
    dat = dict(y=y, X=X, a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0, N=N, k=k)
    start_t = time.perf_counter_ns()
    with suppress_stdout_stderr():
        fit = model.sampling(
            data=dat, iter=draws + warmup, warmup=warmup, chains=chains
        )
    elapsed_ms = 1e-6 * (time.perf_counter_ns() - start_t)
    if verbose:
        print(f"Sampling (ex compilation) completed in {elapsed_ms:.2f} ms.")
    return fit


def _gibbs_onechain(args):
    y, X, draws, warmup, = args["y"], args["X"], args["draws"], args["warmup"]
    a_0, b_0, c_0, d_0 = args["a_0"], args["b_0"], args["c_0"], args["d_0"]
    N, k, = X.shape
    assert len(y) == N
    XX, I = X.T @ X, np.eye(k)
    β, τ, α = np.empty([draws, k]), np.empty([draws]), np.empty([draws])
    τ_, α_ = 0.1, 5.0
    for i in range(-warmup, draws):
        R = pinv(α_ * I + XX)
        β_hat = R @ X.T @ y
        β_ = Φ(mean=β_hat, cov=R / τ_, allow_singular=True).rvs()
        e = y - X @ β_
        τ_ = Γ(
            a=0.5 * (N + k) + a_0, scale=2.0 / (e.T @ e + α_ * β_.T @ β_ + 2 * b_0)
        ).rvs()
        α_ = Γ(a=0.5 * k + c_0, scale=2.0 / (τ_ * β_.T @ β_ + d_0)).rvs()
        if i >= 0:
            β[i, :], τ[i], α[i] = β_, τ_, α_
    return β, τ, α


def gibbs_reg(y, X, warmup, draws, chains, a_0, b_0, c_0, d_0):
    """Use Gibbs sampling to draw from posterior for simple hierarchical model

    Model is Drugowitsch's non-ARD LR model in https://arxiv.org/abs/1310.5438.

    Args:
        y:        univariate time series
        X:        explanatory variables; typically the first column is 1
        warmup:   number of warmup draws
        draws:    number of draws to return (in addition to warmup)
        chains:   number of parallel chains to run
        a_0, b_0: prior for τ
        c_0, d_0: prior for α

    Returns:
        tuple (β, τ, α) of vectors of draws
    """
    args = dict(
        y=y, X=X, warmup=warmup, draws=draws, a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0
    )
    with Pool(chains) as p:
        results = p.map(_gibbs_onechain, [args] * chains)
    combo = tuple(
        np.concatenate([results[i][j] for i in range(chains)]) for j in range(3)
    )
    return combo


def reg_mcmc(
    y,
    X,
    draws=1000,
    warmup=1000,
    chains=4,
    a_0=2,
    b_0=0.5,
    c_0=2,
    d_0=0.5,
    verbose=True,
    method="Gibbs",
):
    """Use Gibbs sampling to draw from posterior for simple hierarchical model

    Model is Drugowitsch's non-ARD LR model in https://arxiv.org/abs/1310.5438.

    Args:
        y:        univariate time series
        X:        explanatory variables; typically the first column is 1
        a_0, b_0: prior for τ
        c_0, d_0: prior for α
        warmup:   number of warmup draws
        draws:    number of draws to return (in addition to warmup)

    Returns:
        tuple (β, τ, α) of vectors of draws
    """
    if method == "Gibbs":
        t = time.perf_counter()
        βs, τs, αs = gibbs_reg(
            warmup=warmup,
            draws=draws,
            chains=4,
            y=y,
            X=X,
            a_0=a_0,
            b_0=b_0,
            c_0=c_0,
            d_0=d_0,
        )
        t = 1e3 * (time.perf_counter() - t)
    elif method == "NUTS":
        t = time.perf_counter_ns()
        fit = stan_reg(y, X, a_0, b_0, c_0, d_0, warmup, draws, 4, verbose=verbose)
        t = 1e-6 * (time.perf_counter_ns() - t)
        samples = fit.extract()
        βs, τs, αs = samples["beta"], samples["tau"], samples["alpha"]
    else:
        raise Exception(f"Invalid method {method}.")
    if verbose:
        bmsd = [
            f"{m:.2f} ({s:.2f})"
            for m, s in zip(np.mean(βs, axis=0), np.std(βs, axis=0))
        ]
        out_lines = [
            " " * 10 + "Summary of Marginals",
            "=" * 50,
            f'βs: {", ".join(bmsd)}',
            f"α:  {np.mean(αs):.2f} ({np.std(αs):.2f})",
            f"τ:  {np.mean(τs):.2f} ({np.std(τs):.2f})",
            "=" * 50,
            f"Sampled {draws} draws after {warmup} warmup. Total time {t:.3f} ms.",
        ]
        print("\n".join(out_lines))
    return βs, τs, αs


def arp_mcmc(y, p, method="Gibbs", **kwargs):
    """Use mcmc to draw from posterior for AR(p).

    Args:
        y:        univariate time series
        p:        autoregression order
        a_0, b_0: prior for τ
        c_0, d_0: prior for α
        warmup:   number of warmup draws
        draws:    number of draws to return (in addition to warmup)
        method:   "Gibbs" or "NUTS"

    Returns:
        draws
    """
    assert method in ["Gibbs", "NUTS"]
    N = len(y) - p
    _y = y[p:]  # first p obs unusable in model
    _X = np.empty([N, p + 1])
    _X[:, 0] = 1
    for i in range(p):
        _X[:, i + 1] = y[p - i - 1 : N + p - i - 1]
    βs, τs, αs = reg_mcmc(y=_y, X=_X, **kwargs, method=method)
    return βs, τs, αs


def arp_mcmc_forecast(y, p, steps, draws=1000, **kwargs):
    """Forecast an AR(p) `steps` using MCMC to fit model.

    Note: supply keyword kwargs for priors, warmup, etc. needed by arp_mcmc().

    Args:
        y:        data
        p:        lags
        steps:    steps ahead
        draws:    number of draws per time-step
        **kwargs: keyword kwargs to pass to arp_mcmc

    Returns
        Dict of {t: draws from the p(y_{t+steps}| y_{1:N}) marginal}.

    Time indexes are zero-based, so the first forecast has index N, etc.
    """
    N = len(y)
    βs, τs, _ = arp_mcmc(y=y, p=p, draws=draws, **kwargs)
    res = {s: np.empty([draws]) for s in range(N, N + steps)}
    y_ext = np.r_[y, np.empty([steps])]  # for storing explanatory variables
    for i in range(draws):
        β, τ = βs[i, :], τs[i]
        for s in range(N, N + steps):
            x = np.r_[1, y_ext[s - 1 : s - p - 1 : -1]]  # constant and p lagged values
            y_dist = stats.norm(
                loc=x.T @ β, scale=1 / np.sqrt(τ)
            )  # p(y_s | β, τ, y_{1:s-1})
            y_ext[s] = y_dist.rvs(size=1)
            res[s][i] = y_ext[s]
    return res


class multi_t(stats.rv_continuous):
    """Multivariate student T distribution.

    Parameterized per Bishop (2006) with mean μ, precision Λ, and ν dof.
    """

    def __init__(self, μ, Λ, ν):
        self.μ = μ
        self.Λ = Λ
        self.ν = ν
        self.D = len(μ)
        assert Λ.shape == (self.D, self.D), f"Λ must be {self.D}x{self.D}"

    def logpdf(self, x):
        Δ2 = (x - self.μ).T @ self.Λ @ (x - self.μ)
        return (
            loggamma((self.ν + self.D) / 2)
            - loggamma(self.ν / 2)
            + 0.5 * slogdet(self.Λ)[1]
            - self.D / 2 * log(self.ν * π)
            - (self.ν + self.D) / 2 * log(1 + Δ2 / self.ν)
        )

    def rvs(self):
        """Draw one sample. See p.582 of Gelman."""
        x = stats.wishart(df=self.ν).rvs()
        z = stats.multivariate_normal(mean=np.zeros(self.D), cov=np.eye(self.D)).rvs()
        A = chol(inv(self.Λ))
        return self.μ + A @ z * sqrt(self.ν / x)


def vb_reg(y, X, a_0, b_0, c_0, d_0, maxiter=1000, tol=1e-4, verbose=False):
    """Simple hierarchical VB linear regression without ARD.

    See: Drugowitsch, J. (2013). Variational Bayesian inference for linear and logistic
         regression. https://arxiv.org/abs/1310.5438

    Args:
        y        - response vector
        X        - matrix of explanatory variables
        a_0, b_0 - hyperparameters for prior on τ
        c_0, d_0 - hyperparameters for prior on α
        maxiter  - maximum iterations
        tol      - relative tolerance (for stopping rule)
        verbose  - show extra output if true

    Returns:
        Dict of parameters and variational posteriors
    """
    N, D = X.shape
    assert len(y) == N
    if verbose:
        print(f"Coordinate ascent on {N} observations.")
    Vinv_N, V_N, a_N, b_N, w_N = None, None, None, None, None
    a_N, c_N, d_N = a_0 + 0.5 * N, c_0 + 0.5 * D, d_0
    L, i = -np.inf, 1
    start_time = time.perf_counter_ns()
    while i < maxiter:
        E_α = c_N / d_N
        Vinv_N = E_α * np.eye(D) + X.T @ X
        V_N = np.linalg.inv(Vinv_N)
        w_N = V_N @ X.T @ y
        b_N = b_0 + 0.5 * (y.T @ y - w_N.T @ Vinv_N @ w_N)
        d_N = d_0 + 0.5 * a_N / b_N * w_N.T @ w_N + np.trace(V_N)
        Lold = L
        x_vn_x = np.array([(X[i, :] @ Vinv_N).T @ X[i, :] for i in range(N)])
        L = (
            -0.5 * N * log(2 * π)
            - 0.5 * sum(a_N / b_N * np.square(y - X @ w_N) + x_vn_x)
            - 0.5 * slogdet(Vinv_N)[1]
            + 0.5 * D
            - loggamma(a_0)
            + a_0 * log(b_0)
            - b_0 * a_N / b_N
            + loggamma(a_N)
            - a_N * log(b_N)
            + a_N
            - loggamma(c_0)
            + c_0 * log(d_0)
            + loggamma(c_N)
            - c_N * log(d_N)
        )
        if verbose:
            print(
                f"{i:4d}. L={L:.4f} a_N={a_N:.2f} b_N={b_N:.2f} "
                f"c_N={c_N:.2f} d_N={d_N:.2f} w_N={w_N}"
            )
        if L < Lold:
            warn(f"L decreased from {Lold} to {L}.")
        if i > 1 and (L - Lold) < tol:
            break
        i += 1
    else:
        warn("Maximum iterations reached.")

    time_elapsed_ns = time.perf_counter_ns() - start_time

    def predictive(x):
        """Return variational predictive distribution for y given x."""
        λ = (1 + x.T @ V_N @ x).inv() * a_N / b_N
        return stats.t(loc=w_N @ x, scale=λ ** (-0.5), df=2 * a_N)

    return {
        "q_α": stats.gamma(c_N, scale=1.0 / d_N),
        "q_τ_marg": stats.gamma(a_N, scale=1.0 / b_N),
        "q_w_marg": multi_t(μ=w_N, Λ=Vinv_N, ν=2 * a_N),
        "Vinv_N": Vinv_N,
        "V_N": V_N,
        "w_N": w_N,
        "a_N": a_N,
        "b_N": b_N,
        "c_N": c_N,
        "d_N": d_N,
        "L": L,
        "iter": i,
        "time_ms": time_elapsed_ns * 1e-6,
        "predictive": predictive,
    }


def arp_vb(y, p, **kwargs):
    """Use VB to compute posterior for AR(p).

    Args:
        y:        univariate time series
        p:        autoregression order
        a_0, b_0: prior for τ
        c_0, d_0: prior for α
        maxiter:  maximum number of coordinate ascent iterations
        tol:      numerical tolerance for assessing convergence

    Returns:
        dict of parameters and marginal distributions
    """
    N = len(y) - p
    _y = y[p:]  # first p obs unusable in model
    _X = np.empty([N, p + 1])
    _X[:, 0] = 1
    for i in range(p):
        _X[:, i + 1] = y[p - i - 1 : N + p - i - 1]
    return vb_reg(y=_y, X=_X, **kwargs)


def arp_vb_forecast(y, p, steps, draws=1000, **kwargs):
    """Forecast an AR(p) `steps` using VB to fit model.

    Note: supply keyword args needed by arp_vb().

    Args:
        y:        data
        p:        lags
        steps:    steps ahead
        draws:    number of draws per time-step
        **kwargs: keyword args to pass to arp_mcmc

    Returns
        Dict of {t: draws from the p(y_{t+steps}| y_{1:N}) marginal}.

    Time indexes are zero-based, so the first forecast has index N, etc.
    """
    N = len(y)
    post = arp_vb(y=y, p=p, **kwargs)
    w_N, V_N = post["w_N"], post["V_N"]
    res = {s: np.empty([draws]) for s in range(N, N + steps)}
    y_ext = np.r_[y, np.empty([steps])]  # for storing explanatory variables
    for i in range(draws):
        α, τ = post["q_α"].rvs(), post["q_τ_marg"].rvs()
        # τ = stats.gamma(post['a_N'], scale=1./post['b_N']).rvs()
        β = Φ(mean=w_N, cov=V_N / τ).rvs()  # int(gamma * normal) = student t
        for s in range(N, N + steps):
            x = np.r_[1, y_ext[s - 1 : s - p - 1 : -1]]  # constant and p lagged values
            y_dist = stats.norm(
                loc=x.T @ β, scale=1 / np.sqrt(τ)
            )  # p(y_s | β, τ, y_{1:s-1})
            y_ext[s] = y_dist.rvs(size=1)
            res[s][i] = y_ext[s]
    return res
