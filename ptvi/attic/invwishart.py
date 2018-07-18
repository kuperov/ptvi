from autograd import numpy as np, primitive
from autograd.scipy import special
from scipy import stats


# TODO: use scipy inverse wishart and explicitly compute our own derivative
# logpdf = primitive(scipy.stats.invwishart.logpdf)


rvs = primitive(stats.invwishart.rvs)


def logpdf(X, df, scale):
    """Inverse-wishart log density.

    Computes density of X, where X ~ W^-1(Psi=scale, nu=df)

    Returns:
      log density of X
    """
    X = np.atleast_2d(X)
    scale = np.atleast_2d(scale)
    p = scale.shape[0]
    X_inv = np.linalg.inv(X)
    return (
            + 0.5 * df * np.linalg.slogdet(scale)[1]
            - 0.5 * df * p * np.log(2)
            - special.multigammaln(0.5 * df, p)
            - 0.5 * (df + p + 1) * np.linalg.slogdet(X)[1]  # |X|^{-(nu+p+1)/2}
            - 0.5 * np.trace(np.dot(scale, X_inv))
    )


def logpdf_cholesky(L, df, Psi):
    """Inverse-wishart log density with Cholesky parameterization.

    Computes p(X | Psi, df) where X ~ W^-1(Psi, df), where X = L@L.T and L is
    lower-triangular

    Returns:
      log density of L
    """
    p = Psi.shape[0]
    L_inv = np.linalg.inv(np.tril(L))  # better to backsolve
    return (
            + 0.5 * df * np.linalg.slogdet(Psi)[1]
            - 0.5 * df * p * np.log(2)
            - special.multigammaln(0.5 * df, p)
            - (df + p + 1) * np.sum(np.log(np.diag(L)))  # |X|^{-(nu+p+1)/2}
            - 0.5 * np.trace(np.dot(np.dot(Psi, L_inv.T), L_inv))
    )
