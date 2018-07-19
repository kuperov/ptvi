import math
from scipy import stats, special
import torch
from torch.distributions import Transform, constraints


class Reciprocal(Transform):
    sign = 1

    def _call(self, x):
        return 1/x  # P(x=0) = 0 for diffuse distributions

    def _inverse(self, y):
        return 1./y  # P(y=0) = 0 for diffuse distributions

    def log_abs_det_jacobian(self, x, y):
        return 2*torch.log(y)


def InvGamma(a, b):
    return torch.distributions.TransformedDistribution(
        torch.distributions.Gamma(concentration=a, rate=b), Reciprocal())


class InvWishart(torch.distributions.Distribution):
    """Inverse-Wishart distribution, which is a distribution over real-valued
    positive-definite matrices.

    """

    arg_constraints = {'df': constraints.positive,
                       'scale': constraints.positive_definite}
    support = constraints.real
    has_rsample = False

    def __init__(self, df, scale):
        """Create inverse Wishart distribution.

        Args:
            df:    degrees of freedom parameter
            scale: positive-definite scale matrix
        """
        self.df, self.scale, self.p = df, scale, scale.shape[0]

    @property
    def mean(self):
        return self.scale/(self.df - self.p - 1)

    # @property
    # def variance(self):
    #     pass
    # TODO: implement variance

    def log_prob(self, value):
        _p = self.p
        assert value.shape == (_p, _p), f'value should be {_p}x{_p} psd'
        X_inv = value.inverse()
        _df = self.df
        return (
            + 0.5 * _df * torch.slogdet(self.scale)[1]
            - 0.5 * _df * _p * math.log(2)
            - special.multigammaln(0.5 * _df, _p)
            - 0.5 * (_df + _p + 1) * torch.slogdet(value)[1]  # |X|^{-(nu+p+1)/2}
            - 0.5 * torch.trace(self.scale @ X_inv)
        )

# def logpdf_cholesky(L, df, Psi):
#     """Inverse-wishart log density with Cholesky parameterization.
#
#     Computes p(X | Psi, df) where X ~ W^-1(Psi, df), where X = L@L.T and L is
#     lower-triangular
#
#     Returns:
#       log density of L
#     """
#     p = Psi.shape[0]
#     L_inv = np.linalg.inv(np.tril(L))  # better to backsolve
#     return (
#             + 0.5 * df * np.linalg.slogdet(Psi)[1]
#             - 0.5 * df * p * np.log(2)
#             - special.multigammaln(0.5 * df, p)
#             - (df + p + 1) * np.sum(np.log(np.diag(L)))  # |X|^{-(nu+p+1)/2}
#             - 0.5 * np.trace(np.dot(np.dot(Psi, L_inv.T), L_inv))
#     )
