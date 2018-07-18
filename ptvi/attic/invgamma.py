from autograd.scipy import special
from autograd import numpy as np, primitive
from scipy import stats


rvs = primitive(stats.invgamma.rvs)


# differentiable inverse gamma logpdf
def logpdf(x, a, scale=1):
    return (a * np.log(scale) - special.gammaln(a) - (a + 1) * np.log(x)
            - scale / x)
