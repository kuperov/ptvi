"""Variational Poisson regression.

See: [1] Arridge, S. R., Ito, K., Jin, B., & Zhang, C. (2018). Variational 
     Gaussian approximation for Poisson data. Inverse Problems, 34(2).
     https://iopscience.iop.org/article/10.1088/1361-6420/aaa0ab/meta
"""

import numpy as np
from scipy import stats, special
from sklearn.decomposition import TruncatedSVD


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
        X = stats.norm().rvs(size=[N, k])
        X[:, 0] = 1.
    else:
        assert X.shape == (N, k)
    eta = X@beta0
    y = stats.poisson(np.exp(eta)).rvs()
    return (y, X)


def poisson_vi_reg(y, A, mu_0, C_0, tol=1e-5, maxiter=1_000):
    """Fit a poisson regression using VGA using [1].

    This optimization uses one fixed-point update and one
    newton-Raphson update.

    Args:
        y:       response variable
        A:       design matrix
        mu_0:    prior mean
        C_0:     prior variance
        tol:     convergence tolerance
        maxiter: maximum number of iterations

    Return:
        dict of various result values
    """
    N, k, = A.shape
    assert y.shape == (N,)
    old_elbo = None
    beta = np.empty(k)
    U, Sigma, V = np.linalg.svd(A, full_matrices=True)  # replace with rSVD
    C_0_inv = np.linalg.inv(C_0)
    C, x_bar = C_0, mu_0  # initial value for C
    m = C.shape[0]
    elbo_const = (
        - 0.5 * np.linalg.slogdet(C_0)[1] - 0.5 * m
        - np.sum(special.gammaln(y + 1))
    )

    def print_status():
        print(f'{i:4d}. elbo = {elbo}, beta = {beta.round(2)}')

    for i in range(maxiter):
        # Newton-Raphson update for x_bar
        G = A.T @ np.exp(A@x_bar + 0.5 * np.diag(A @ C @ A.T)) + C_0_inv @ (x_bar - mu_0) - A.T@y
        curlG = A.T @ np.diag(np.exp(A @ x_bar + 0.5 * A @ C @ A.T)) @ A + C_0_inv
        delta_x = np.linalg.solve(curlG, -G)
        x_bar = x_bar + delta_x
        # fixed point update for C
        D = np.diag(np.exp(A@x_bar + 0.5 * np.diag(A @ C @ A.T)))
        C = np.linalg.inv(C_0_inv + A.T @ D @ A)
        # elbo for monitoring convergence
        elbo = (
            y.T@A@x_bar - np.sum(np.exp(A@x_bar + 0.5 * np.diag(A@C@A.T))) 
            - 0.5 * (x_bar - mu_0).T @ C_0_inv @ (x_bar - mu_0)
            - 0.5 * np.trace(C_0_inv @ C)
            + 0.5 * np.linalg.slogdet(C)[1]
            + elbo_const
        )
        if old_elbo and elbo - old_elbo < tol:
            print('Convergence detected.')
            break
        old_elbo = elbo
        if i % 5 == 0:
            print_status()
    else:
        print('WARNING: Maximum iterations reached.')
    print_status()
    return {'elbo': elbo, 'beta': beta}
