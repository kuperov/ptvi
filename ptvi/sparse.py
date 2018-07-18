import math
from typing import List, Tuple

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from torch.distributions.distribution import Distribution


def sparse_prec_chol(dim: int, diags: int, globals: int, requires_grad=True):
    assert dim > diags + globals
    """Create the lower-triangular cholesky factor of a sparse precision matrix
    with dimension dim, diags diagonals, and globals global variables.
    """
    # rows and cols jointly define coordinates, values defines values
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    # first: diagonals for the local variables
    for d in range(diags):
        # note if we were using scipy we would be numbering the (lower)
        # diagonals with negative numbers, unlike here
        rows += list(range(d, dim))
        cols += list(range(dim - d))
        values += [1. if d == 0 else 0] * (dim - d)
    # second: bottom rows are for the globals
    for r in range(dim-globals, dim):
        rowlen = r + 1 - diags  # exclude the diagonals
        rows += [r] * rowlen
        cols += list(range(rowlen))
        values += [0.] * rowlen
    idx_tens = torch.LongTensor([rows, cols])
    val_tens = torch.FloatTensor(values)
    st = torch.sparse.FloatTensor(
        idx_tens, val_tens, torch.Size([dim, dim]))
    return st.requires_grad_(requires_grad)

def _get_batch_shape(bmat, bvec):
    r"""
    Given a batch of matrices and a batch of vectors, compute the combined `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError("Incompatible batch shapes: vector {}, matrix {}".format(bvec.shape, bmat.shape))
    return torch.Size(vec_shape[:-1])


class SparseMultivariateNormal(torch.distributions.Distribution):
    """
    Multivariate normal distribution parameterized by a mean vector and lower-
    triangular cholesky factor of the precision matrix.

    Args:
        loc (Tensor):            mean of the distribution
        precision_tril (Tensor): lower-triangular factor of precision, with
                                 positive-valued diagonal
    """
    support = constraints.real
    arg_constraints = {'precision_tril': constraints.lower_cholesky}
    has_rsample = False

    def __init__(self, loc, precision_tril, validate_args=None):
        self.loc, self.precision_tril = loc, precision_tril
        event_shape = torch.Size(loc.shape[-1:])
        if precision_tril.dim() < 2:
            raise ValueError(
                "precision_tril matrix must be at least two-dimensional, "
                "with optional leading batch dimensions")
        batch_shape = _get_batch_shape(precision_tril, loc)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        return torch.potri(self.loc, upper=False)  # is this diffable?

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        term = self.precision_tril.dot(diff)
        M = term.dot(term)
        log_det = torch.diag(self.scale_tril).abs().log().sum(-1)
        return -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det

    def entropy(self):
        log_det = torch.diag(self.scale_tril).abs().log().sum()
        H = 0.5 * (1.0 + math.log(2 * math.pi)) * self._event_shape[0] + log_det
        return H

    def rsample(self, sample_shape=torch.Size()):
        pass
