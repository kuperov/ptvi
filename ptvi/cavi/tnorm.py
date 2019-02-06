"""Truncated normal distribution.

Follows Wikipedia's parameterization[0].

  [0] https://en.wikipedia.org/wiki/Truncated_normal_distribution
"""
from scipy import stats
import numpy as np
from scipy.stats import rv_continuous

_norm = stats.norm()


class tnorm_gen(rv_continuous):
    """Truncated normal random variable.

    .. math::

        f(x; mu, sigma, a, b) = ...
    """

    def _argcheck(self, mu, sigma, a, b):
        self.mu = np.array(mu)
        shape = np.shape(mu)
        self.sigma = np.broadcast_to(sigma, shape=shape)
        self.a = np.broadcast_to(a, shape=shape)
        self.b = np.broadcast_to(b, shape=shape)
        self._alpha = (a - mu) / sigma
        self._beta = (b - mu) / sigma
        self._Z = _norm.cdf(self._beta) - _norm.cdf(self._alpha)
        return np.alltrue(a < b)

    def _pdf(self, x, mu, sigma, a, b):
        return np.exp(self._logpdf(x, mu, sigma, a, b))

    def _logpdf(self, x, mu, sigma, a, b):
        xi = (x - mu) / sigma
        lpdfs = _norm.logpdf(xi) - np.log(sigma) - np.log(self._Z)
        return np.where(x <= a, -np.inf, np.where(x > b, -np.inf, lpdfs))

    def _cdf(self, x, mu, sigma, a, b):
        cdf = (_norm.cdf((x - mu) / sigma) - _norm.cdf(self._alpha)) / self._Z
        return np.where(x <= a, 0.0, np.where(x > b, 1.0, cdf))

    def _entropy(self, mu, sigma, a, b):
        """Return total entropy.

        If this distribution is defined over an array of parameters, the result is summed."""
        # numerator terms are zeroed out when boundaries are infinite
        # we have to fiddle a bit to avoid calling normal pdf() with infinite value
        finite_alpha = np.where(np.isfinite(self._alpha), self._alpha, 0)
        assert np.alltrue(np.isfinite(finite_alpha))
        alpha_x_phi_alpha = np.where(
            np.isfinite(self._alpha), self._alpha * _norm.pdf(finite_alpha), 0
        )
        finite_beta = np.where(np.isfinite(self._beta), self._beta, 0)
        assert np.alltrue(np.isfinite(finite_beta))
        beta_x_phi_beta = np.where(
            np.isfinite(self._beta), self._beta * _norm.pdf(finite_beta), 0
        )
        return np.sum(
            0.5 * np.log(2 * np.e * np.pi)
            + np.log(sigma * self._Z)
            + 0.5 * (alpha_x_phi_alpha - beta_x_phi_beta) / self._Z
        )

    def _ppf(self, q, mu, sigma, a, b):
        return sigma * _norm.ppf(q * self._Z + _norm.cdf(self._alpha)) + mu


tnorm = tnorm_gen(name="tnorm")  # , shapes='mu, sigma, a, b')


if __name__ == "__main__":
    p = tnorm(mu=1.5, sigma=2.0, a=0.0, b=np.inf)
    xs = np.linspace(-10, 10, 1000)
    ys = tnorm.pdf(xs, mu=1.5, sigma=2.0, a=0.0, b=np.inf)
    print(tnorm.pdf(0.5, mu=1.5, sigma=2.0, a=0.0, b=np.inf))
    assert np.all(np.isfinite(ys))
