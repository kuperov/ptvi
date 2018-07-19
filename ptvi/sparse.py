import math
from typing import List, Tuple

import torch
from torch.distributions import constraints
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

