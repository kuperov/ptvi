import unittest
from ptvi.util import (tril_to_vec, vec_to_tril, trilpd_to_vec,
                       vec_to_trilpd, pd_to_vec, vec_to_pd)
from autograd import numpy as np, jacobian, primitive
from autograd.test_util import check_grads

class TestUtil(unittest.TestCase):

    def test_tril_to_vec(self):
        l = np.r_[1., 2., 3., 4., 5., 6.]  # for a 3x3 matrix
        L = vec_to_tril(l)
        l2 = tril_to_vec(L)
        L2 = np.array([[1., 0, 0], [2., 3., 0], [4., 5., 6.]])
        self.assertTrue(np.all(l == l2))
        self.assertTrue(np.all(L == L2))
        # 1x1 edge case
        self.assertTrue(np.all(
            np.array([1.]) == tril_to_vec(vec_to_tril(np.array([1.])))))
        self.assertTrue(np.array([[1.]]) ==
                        vec_to_tril(tril_to_vec(np.array([[1.]]))))

    def test_vec_to_tril_gradients(self):
        def f(x):
            return vec_to_tril(np.array([0, x, x**2]))  # gradients 0, 1, 2x
        self.assertTrue(np.all(jacobian(f)(2.) == np.array([[0.,0.], [1.,4.]])))

    def test_tril_to_vec_gradients(self):
        def f(x):
            return tril_to_vec(np.array([[0, 0], [x, x**2]]))  # 0, 1, 2x
        self.assertTrue(np.all(jacobian(f)(3.) == np.array([0., 1., 6.])))

    def test_vec_to_trilpd_and_back(self):
        l = np.r_[0., 2., np.log(3.), 4., 5., np.log(6.)]  # for a 3x3 matrix
        L = vec_to_trilpd(l)
        l2 = trilpd_to_vec(L)
        L2 = np.array([[1., 0, 0], [2., 3., 0], [4., 5., 6.]])
        self.assertTrue(np.allclose(l, l2))
        self.assertTrue(np.allclose(L, L2))
        # 1x1 edge case
        self.assertTrue(np.allclose(
            np.array([1.]), trilpd_to_vec(vec_to_trilpd(np.array([1.])))))
        self.assertTrue(np.allclose(
            np.array([[1.]]), vec_to_trilpd(trilpd_to_vec(np.array([[1.]])))))

    def test_trilpd_gradients(self):
        check_grads(vec_to_trilpd)(np.array([1., 2., 3.]))
        check_grads(trilpd_to_vec, modes=['rev'])(
            np.array([[0.5, 0.], [1, np.log(3.)]]))

        def f(x: float):
            return trilpd_to_vec(np.array([[x, 0.], [x**2, np.log(x**3)]]))
        check_grads(f, modes=['rev'])(1.5)

    def test_pd_vec_gradients(self):
        rs = np.random.RandomState(seed=123)
        check_grads(vec_to_pd, modes=['rev'])(np.array([1., 2., 3.]))
        L = rs.normal(size=[4, 4])
        A = L@L.T + 5*np.eye(4)
        assert np.all(np.linalg.eig(A)[0] > 0)
        # only defined for psd matrices, soo...
        check_grads(lambda B: pd_to_vec(B@B.T), modes=['rev'])(A)
