import numpy as np


def logit(x):
    return np.exp(x) / (1.0 + np.exp(x))


def logistic_pdf(x, mu, s):
    """PDF of logistic distribution with mean mu and scale s.
    """
    z = np.exp(-(x - mu) / s)
    return z / (1.0 + z) ** 2.0 / s


def logistic_cdf(x, mu, s):
    """CDF of logistic distribution with mean mu and scale s.
    """
    z = np.exp(-(x - mu) / s)
    return 1.0 / (1.0 + z)


def logistic_icdf(q, mu, s):
    """Inverse CDF of logistic distribution with mean mu and scale s.
    """
    is_valid = np.logical_and(q < 1, q > 0)
    alternative = np.broadcast_to(np.nan, np.shape(q))
    return np.where(is_valid, s * np.log((1 - q) / q) + mu, alternative)


def logistic_sim(n, mu, s):
    """Simulate from logistic distribution with mean mu and scale s.
    """
    us = np.random.uniform(size=n)
    return logistic_icdf(us, mu, s)


def l_trunc_logistic_pdf(x, mu, s, a):
    """PDF of logistic distribution left-truncated at a."""
    q_a = logistic_cdf(a, mu, s)
    return np.where(x < a, 0.0, logistic_pdf(x, mu, s) / (1 - q_a))


def r_trunc_logistic_pdf(x, mu, s, a):
    """PDF of logistic distribution right-truncated at a."""
    q_a = logistic_cdf(a, mu, s)
    return np.where(x > a, 0.0, logistic_pdf(x, mu, s) / q_a)


def l_trunc_logistic_cdf(x, mu, s, a):
    """CDF of logistic distribution left-truncated at a."""
    q_a = logistic_cdf(a, mu, s)
    return np.where(x < a, 0.0, (logistic_cdf(x, mu, s) - q_a) / (1.0 - q_a))


def r_trunc_logistic_cdf(q, mu, s, a):
    """CDF of logistic distribution right-truncated at a."""
    q_a = logistic_cdf(a, mu, s)
    return np.where(q > a, 1.0, logistic_cdf(q, mu, s) / q_a)


def l_trunc_logistic_icdf(q, mu, s, a):
    """Inverse CDF of logistic distribution left-truncated at a."""
    q_a = logistic_cdf(a, mu, s)
    return np.where(
        q == 1.0,
        np.inf,
        np.where(
            q == 0.0,
            a,
            mu + s * (np.log(q * (1 - q_a) + q_a) - np.log(1 - q * (1 - q_a) - q_a)),
        ),
    )


def r_trunc_logistic_icdf(q, mu, s, a):
    """Inverse CDF of logistic distribution right-truncated at a."""
    q_a = logistic_cdf(a, mu, s)
    return np.where(
        q == 1.0,
        np.inf,
        np.where(q == 0.0, a, mu + s * np.log(q_a * q / (1 - q_a * q))),
    )


def l_trunc_logistic_sim(n, mu=0, s=1, a=0):
    """Simulate from logistic distribution left-truncated at a."""
    u = np.random.uniform(size=n)
    return l_trunc_logistic_icdf(u, mu, s, a)


def r_trunc_logistic_sim(n, mu=0, s=1, a=0):
    """Simulate from logistic distribution right-truncated at a."""
    u = np.random.uniform(size=n)
    return r_trunc_logistic_icdf(u, mu, s, a)
