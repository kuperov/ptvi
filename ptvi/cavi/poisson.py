"""Variational Poisson regression.

See: [1] Arridge, S. R., Ito, K., Jin, B., & Zhang, C. (2018). Variational 
     Gaussian approximation for Poisson data. Inverse Problems, 34(2).
     https://iopscience.iop.org/article/10.1088/1361-6420/aaa0ab/meta
"""

import numpy as np
from scipy import stats
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


def poisson_vi_reg(y, X, tol=1e-5, maxiter=1_000):
    """Fit a poisson regression using VGA using [1].

    This optimization uses one fixed-point update and one
    newton-Raphson update.

    Args:
        y:       response variable
        X:       design matrix
        tol:     convergence tolerance
        maxiter: maximum number of iterations

    Return:
        dict of various result values
    """
    N, k, = X.shape
    assert y.shape == (N,)
    old_elbo = None
    beta = np.empty(k)
    U, Sigma, V = np.linalg.svd(X, full_matrices=True)  # replace with rSVD
    
    def print_status():
        print(f'{i:4d}. elbo = {elbo}, beta = {beta.round(2)}')

    for i in range(maxiter):
        G = ...
        curlG = ...

        elbo = -1
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
