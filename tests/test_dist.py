import unittest
import torch
import numpy as np
from ptvi.dist import MVNPrecisionTril


class TestMVNPrecisionTril(unittest.TestCase):

    def test_logprob_dense(self):
        k = 5
        L = torch.zeros((k, k))
        torch.manual_seed(123)
        L[np.tril_indices(k)] = torch.randn(k*(k+1)//2)
        Λ = torch.matmul(L, L.t())
        Σ = Λ.inverse()
        μ = torch.randn(k)
        dist = MVNPrecisionTril(loc=μ, precision_tril=L)
        dist_equiv = torch.distributions.MultivariateNormal(
            loc=μ, covariance_matrix=Σ)
        H1 = dist.entropy()
        H2 = dist_equiv.entropy()
        self.assertTrue(torch.allclose(H1, H2, rtol=1e-2))
        # draw variates and check densities match
        torch.manual_seed(123)
        for i in range(5):
            x = dist.sample()
            logdens = dist.log_prob(x)
            logdens2 = dist_equiv.log_prob(x)
            # surprising amount of error creeps in between these two...
            self.assertTrue(torch.allclose(logdens, logdens2, rtol=1e-2))

    def test_logprob_sparse(self):
        from ptvi.sparse import sparse_prec_chol
        k = 5
        L = sparse_prec_chol(k, 2, 1, requires_grad=False)
        Σ = torch.eye(k)
        μ = torch.randn(k)
        dist = MVNPrecisionTril(loc=μ, precision_tril=L)
        dist_equiv = torch.distributions.MultivariateNormal(
            loc=μ, covariance_matrix=Σ)
        H1 = dist.entropy()
        H2 = dist_equiv.entropy()
        self.assertTrue(torch.allclose(H1, H2, rtol=1e-2))
        # draw variates and check densities match
        torch.manual_seed(123)
        for _ in range(5):
            x = dist.sample()
            logdens = dist.log_prob(x)
            logdens2 = dist_equiv.log_prob(x)
            self.assertTrue(torch.allclose(logdens, logdens2, rtol=1e-2))
