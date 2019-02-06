"""Inference scripts for generalized autoregressive conditional
heteroskedasticity (GARCH(p,q)) model.

Model description:

      εₜ | Ψₜ ~ N(0,σ²)
          σ²ₜ = α₀ + δ₁σ²ₜ₋₁ + ⋅⋅⋅ + δₚ σ²ₜ₋ₚ + α₁ε²ₜ₋₁ + ⋅⋅⋅ + α_q ε²ₜ₋_q

or equivalently:

  D(L)σ²ₜz = A(L)ε²ₜ

Strategy here is to lay out the relationships in the display above in matrix
form, and then premultiply by inverse of matrix on left, yielding an
expression for σ² in terms of the ε² vector. We then do standard ML with BFGS.

Apart from being really slow (it scales as O(T²)), the D coefficients seem to
be biased.

See: Greene section 20.10,
     https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity
"""
from warnings import warn
import time
import torch
from ptvi.cavi.results import summary_row


def simulate_garch(T, beta, X, a0, A_poly, D_poly, burnin=None):
    """Simulate GARCH(p,q) model. Caller should set the random seed.

    Args:
        T:      length of series to generate
        beta:      true regressors, k-vector
        X:      T*k design matrix, conforms with beta. If None, simulate one.
        a0:     additive constant α0 for variance
        A_poly: coefficients of lag polynomial A, where A[0] = α₁ (array-like, length q)
        D_poly: coefficients of lag polynomial D, where D[0] = δ₁ (array-like, length p)
        burnin: elements to simulate before start of series (default is 2*max(p,q))

    Returns:
        Tensor of length T.
    """
    q, p, k = len(A_poly), len(D_poly), len(beta)
    if torch.sum(A_poly) + torch.sum(D_poly) >= 1:
        warn("GARCH process is nonstationary")
    if X is None:
        X = torch.distributions.Normal(0, 1).sample((T, k))
    B = burnin if burnin is not None else 2 * max(p, q)
    y, sigsq = torch.empty([T + B]), torch.empty([T + B])
    eps = torch.distributions.Normal(0, 1).sample((T + B,))
    for t in range(T + B):
        sigsq[t] = a0
        for i in range(min(t, p)):
            sigsq[t] += D_poly[i] * sigsq[t - i - 1]
        for i in range(min(t, q)):
            sigsq[t] += A_poly[i] * eps[t - i - 1] ** 2
        eps[t] *= torch.sqrt(sigsq[t])
        y[t] = X[max(0, t - B), :] @ beta + eps[t]  # use first row of X for burnin
    return y[-T:], X


def garch_freq(y, X, p, q, tol=1e-12, maxiter=1000, **kwargs):
    """Fit a GARCH(p,q) model using maximum likelihood.

    Args:
        y:        response variable
        X:        explanatory variables
        p:        number of autoregressors in error variance
        q:        number of moving average components in error variance
        tol:      convergence tolerance for the log likelihood
        maxiter:  maximum iterations
        **kwargs: optional keyword args to pass to optimizer

    Returns:
        dict of results
    """
    T, k = len(y), X.shape[1]
    print(f"Maximum likelihood (L-BFGS) for GARCH({p},{q}) with {T} observations.")

    def nllik(beta, log_a0, log_A_poly_rev, log_D_poly_rev):
        eps = y - X @ beta
        eps_sq = eps ** 2
        a0 = torch.exp(log_a0)
        sigsqs = a0
        A_poly_rev = torch.exp(log_A_poly_rev)
        D_poly_rev = torch.exp(log_D_poly_rev)
        for t in range(1, T):
            sigsq = (
                a0
                + A_poly_rev[: min(t, q)] @ eps_sq[max(0, t - q) : t]
                + D_poly_rev[: min(t, p)] @ sigsqs[max(0, t - p) : t]
            )
            sigsqs = torch.cat([sigsqs, sigsq])
        sig = torch.sqrt(sigsqs)
        lprobs = torch.distributions.Normal(0, sig).log_prob(eps)
        if any(torch.isnan(lprobs)):
            raise Exception(f"Error: infinite loss")
        return -torch.sum(lprobs)

    # autoregressive parameters in log space
    params = dict(
        log_a0=(-torch.ones(1)).requires_grad_(True),
        beta=(torch.zeros(k)).requires_grad_(True),
        log_A_poly_rev=(-torch.ones(q)).requires_grad_(True),
        log_D_poly_rev=(-torch.ones(p)).requires_grad_(True),
    )
    loss = dict(loss=nllik(**params), old_loss=None)
    opt = torch.optim.LBFGS(list(params.values()), **kwargs)
    start_time = time.perf_counter()
    for i in range(1, maxiter + 1):

        def summary():
            print(summary_row(params, iteration=i))

        def closure():
            opt.zero_grad()
            loss["loss"] = nllik(**params)
            loss["loss"].backward()
            return loss["loss"]

        opt.step(closure)

        if i > 1 and abs(loss["old_loss"] - loss["loss"]) < tol:
            elapsed = time.perf_counter() - start_time
            print(f"Convergence reached in {elapsed:.4f}s")
            break
        loss["old_loss"] = loss["loss"]
        if i % 1 == 0:
            summary()
    else:
        elapsed = time.perf_counter() - start_time
        print(f"Maximum iterations reached in {elapsed:.4f}s")
    summary()
    return params
