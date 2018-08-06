"""Pytorch-differentiable functions for reshaping and transforming data.

The main use of these transforms is to allow useful random objects (eg random
psd matrices) to be represented as random vectors that can be optimized.
"""

import torch
import math
import numpy as np

#      |      >>> class Exp(Function):
#      |      >>>
#      |      >>>     @staticmethod
#      |      >>>     def forward(ctx, i):
#      |      >>>         result = i.exp()
#      |      >>>         ctx.save_for_backward(result)
#      |      >>>         return result
#      |      >>>
#      |      >>>     @staticmethod
#      |      >>>     def backward(ctx, grad_output):
#      |      >>>         result, = ctx.saved_tensors
#      |      >>>         return grad_output * result


def vec_to_tril(l):
    assert l.ndimension() == 1, "l should have dim=1"
    # lower triangle has k(k+1)//2 entries
    k = round(-1 + math.sqrt(1 + 8 * len(l))) // 2
    L = torch.zeros((k, k))
    L[np.tril_indices(k)] = l
    return L


def tril_to_vec(L):
    """Convert a lower-triangular matrix L to a vector l."""
    assert L.ndimension() == 2, "L should have dimension 2"
    return L[np.tril_indices(L.shape[0])]


# this function is differentiable as-is
def vec_to_trilpd(l):
    """Convert a vector ls to a lower-triangular matrix L and exponentiate the
    diagonal entries."""
    assert l.ndimension() == 1, "l should have dimension 1"
    # lower triangle has k(k+1)//2 entries
    k = round(-1 + math.sqrt(1 + 8 * len(l))) // 2
    L = torch.zeros((k, k))
    L[np.tril_indices(k)] = l
    L[np.diag_indices(k)] = torch.exp(L[np.diag_indices(k)])
    return L


def trilpd_to_vec(L):
    """Convert a lower triangular factor with positive diagonal to a vector,
    after first taking elementwise logs of the diagonals. Autodiffable.
    """
    k = L.shape[0]
    assert L.shape == (k, k) and all(torch.diag(L) > 0)
    L2 = L * (1 - torch.eye(k)) + torch.diag(torch.log(torch.diag(L)))
    return L2[np.tril_indices(k)]


def pd_to_vec(A):
    """Convert a positive-definite matrix A to a vector l of entries from
    its cholesky factor. Diagonal entries are logged so they occupy the full
    real line, and still map back to positive values.
    """
    L = torch.potrf(A, upper=False)
    return trilpd_to_vec(L)


def vec_to_pd(l):
    """Convert a vector created by pd_to_vec back to a positive-definite matrix.
    """
    L = vec_to_trilpd(l)
    return torch.matmul(L, L.t())
