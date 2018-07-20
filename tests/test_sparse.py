import unittest
from ptvi import sparse_prec_chol
import torch


class TestSparseMatrix(unittest.TestCase):

    def test_sparse_prec_chol(self):
        for n, d, g in [(5, 1, 1), (10, 3, 3), (3, 1, 1)]:
            S = sparse_prec_chol(dim=n, diags=d, globals=g)
            Sd = S.to_dense()
            self.assertTrue(torch.allclose(Sd, torch.eye(n)))
            self.assertTrue(S.requires_grad)
