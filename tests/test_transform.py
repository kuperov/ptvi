import unittest
from ptvi.transform import (
    tril_to_vec,
    vec_to_tril,
    trilpd_to_vec,
    vec_to_trilpd,
    pd_to_vec,
    vec_to_pd,
)
import torch
from torch.autograd import gradcheck
import numpy as np


class TestUtil(unittest.TestCase):
    def test_tril_to_vec(self):
        l = torch.tensor([1., 2., 3., 4., 5., 6.])  # for a 3x3 matrix
        L = vec_to_tril(l)
        l2 = tril_to_vec(L)
        L2 = torch.tensor([[1., 0, 0], [2., 3., 0], [4., 5., 6.]])
        self.assertTrue(torch.allclose(l, l2))
        self.assertTrue(torch.allclose(L, L2))
        # 1x1 edge case
        self.assertTrue(
            torch.allclose(
                torch.tensor([1.]), tril_to_vec(vec_to_tril(torch.tensor([1.])))
            )
        )
        self.assertTrue(
            torch.tensor([[1.]]) == vec_to_tril(tril_to_vec(torch.tensor([[1.]])))
        )

    def test_vec_to_tril_gradients(self):
        def f(x):
            return vec_to_tril(torch.tensor([0, x, x ** 2]))  # gradients 0, 1, 2x

        gradcheck(f, (torch.tensor(2.),))

    def test_tril_to_vec_gradients(self):
        def f(x):
            return tril_to_vec(torch.tensor([[0, 0], [x, x ** 2]]))  # 0, 1, 2x

        gradcheck(f, (torch.tensor(3.),))

    def test_vec_to_trilpd_and_back(self):
        # for a 3x3 matrix
        l = torch.tensor([0., 2., np.log(3.), 4., 5., np.log(6.)])
        L = vec_to_trilpd(l)
        l2 = trilpd_to_vec(L)
        L2 = torch.tensor([[1., 0, 0], [2., 3., 0], [4., 5., 6.]])
        self.assertTrue(torch.allclose(l, l2))
        self.assertTrue(torch.allclose(L, L2))
        # 1x1 edge case
        self.assertTrue(
            torch.allclose(
                torch.tensor([1.]), trilpd_to_vec(vec_to_trilpd(torch.tensor([1.])))
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor([[1.]]), vec_to_trilpd(trilpd_to_vec(torch.tensor([[1.]])))
            )
        )

    def test_trilpd_gradients(self):
        gradcheck(vec_to_trilpd, (torch.tensor([1., 2., 3.]),))
        gradcheck(trilpd_to_vec, (torch.tensor([[0.5, 0.], [1, np.log(3.)]]),))

        def f(x: float):
            return trilpd_to_vec(torch.tensor([[x, 0.], [x ** 2, np.log(x ** 3)]]))

        gradcheck(f, (1.5,))

    def test_pd_vec_gradients(self):
        torch.manual_seed(123)
        gradcheck(vec_to_pd, (torch.tensor([1., 2., 3.]),))
        L = torch.randn((4, 4))
        A = L @ L.t() + 5 * torch.eye(4)
        assert all(torch.symeig(A)[0] > 0)
        # only defined for psd matrices, soo...
        gradcheck(lambda B: pd_to_vec(B @ B.t()), (A,))
