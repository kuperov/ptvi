import torch
from torch.distributions import Transform


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
    @property
    def support(self):
        pass

    @property
    def arg_constraints(self):
        pass

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def rsample(self, sample_shape=torch.Size()):
        pass

    def log_prob(self, value):
        pass

    def cdf(self, value):
        pass

    def icdf(self, value):
        pass

    def enumerate_support(self):
        pass

    def entropy(self):
        pass
