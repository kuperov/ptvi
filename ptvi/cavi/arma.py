import numpy as np


def ar_design_matrix(y, X, p):
    """Construct an AR(p) design matrix by adding lags of y to X.

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


def combine_ar_priors(mu_beta, mu_phi, Sigma_beta, Sigma_phi):
    """Combine priors for mu and phi for use in AR regression functions.

    Args:
        mu_beta:    mean of beta prior
        mu_phi:     mean of phi prior
        Sigma_beta: covariance of beta prior
        Sigma_phi:  covariance of phi prior

    Returns:
        tuple: mu_both, Sigma_both
    """
    k, p = len(mu_beta), len(mu_phi)
    mu_both = np.r_[mu_beta, mu_phi]
    Sigma_both = np.block(
        [[Sigma_beta, np.zeros([k, p])], [np.zeros([p, k]), Sigma_phi]]
    )
    return mu_both, Sigma_both
