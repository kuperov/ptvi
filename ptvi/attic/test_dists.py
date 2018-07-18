import unittest
from functools import partial
from autograd import numpy as np
from autograd.numpy import random as npr
from autograd.scipy.stats import multivariate_normal as mvn
from autograd.test_util import combo_check, check_grads

from ptvi.attic import invgamma, invwishart


# a bunch of this stuff is copied from autograd; here to check we have the
# appropriate bug fix in place and that our own gradients are right

def symmetrize_matrix_arg(fun, argnum):
    def T(X): return np.swapaxes(X, -1, -2) if np.ndim(X) > 1 else X
    def symmetrize(X): return 0.5 * (X + T(X))
    def symmetrized_fun(*args, **kwargs):
        args = list(args)
        args[argnum] = symmetrize(args[argnum])
        return fun(*args, **kwargs)
    return symmetrized_fun

R = npr.randn
U = npr.uniform
def make_psd(mat):
    return np.dot(mat.T, mat) + np.eye(mat.shape[0])
combo_check = partial(combo_check, modes=['rev'])
check_grads = partial(check_grads, modes=['rev'])


class TestMVN(unittest.TestCase):

    def test_mvn_pdf(self):
        combo_check(symmetrize_matrix_arg(mvn.pdf, 2), [0, 1, 2])(
            [R(4)], [R(4)], [make_psd(R(4, 4))], allow_singular=[False])

    def test_mvn_logpdf(self):
        combo_check(symmetrize_matrix_arg(mvn.logpdf, 2), [0, 1, 2])(
            [R(4)], [R(4)], [make_psd(R(4, 4))], allow_singular=[False])

    def test_mvn_entropy(self):
        combo_check(mvn.entropy, [0, 1])([R(4)], [make_psd(R(4, 4))])


class TestInvWishart(unittest.TestCase):

    def test_invwishart_logpdf(self):
        combo_check(symmetrize_matrix_arg(invwishart.logpdf, 2), [0, 1, 2])(
            [R(4, 4)], [U(5, 10)], [make_psd(R(4, 4))])


class TestInvGamma(unittest.TestCase):

    def test_invgamma_logpdf(self):
        combo_check(invgamma.logpdf, [0, 1, 2])([U()], [U()], [U()])
