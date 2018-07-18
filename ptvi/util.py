import itertools
from autograd import numpy as np, primitive
from autograd.extend import defvjp


@primitive
def vec_to_tril(l):
    """Convert a vector ls to a lower-triangular matrix L."""
    assert np.ndim(l) == 1, '{} should have dim=1'.format(l)
    k = int(np.round((-1 + np.sqrt(1 + 8*len(l)))/2))
    entries = [[l[r*(r+1)//2 + c] for c in range(r+1)] + [0]*(k-r-1)
               for r in range(k)]
    return np.array(entries)

@primitive
def tril_to_vec(L):
    """Convert a lower-triangular matrix L to a vector l."""
    assert np.ndim(L) == 2
    return L[np.tril_indices_from(L)]

# for automatic differentiation
defvjp(vec_to_tril, lambda ans, x: lambda g: tril_to_vec(g))
defvjp(tril_to_vec, lambda ans, x: lambda g: vec_to_tril(g))


def vec_to_trilpd(l):
    """Convert a vector ls to a lower-triangular matrix L and exponentiate the
    diagonal entries."""
    assert np.ndim(l) == 1, '{} should have dim=1'.format(l)
    k = int(np.round((-1 + np.sqrt(1 + 8*len(l)))/2))
    entries = [[l[r*(r+1)//2 + c] for c in range(r)]
               + [np.exp(l[r*(r+1)//2 + r])]
               + [0]*(k-r-1)
               for r in range(k)]
    return np.array(entries)

@primitive
def trilpd_to_vec(L):
    """Convert a lower-triangular matrix L to a vector l, after first logging
    the diagonal entries."""
    k = np.shape(L)[0]
    assert L.shape == (k, k) and np.all(np.diag(L) > 0)
    entries = ([[np.log(L[0, 0])]] +         # first diagonal entry
               [[L[r, c] for c in range(r)]  # lower off-diagonals
                + [np.log(L[r, r])]          # diagonals
                for r in range(1, k)])
    return np.concatenate(entries)

def make_trilpd_to_vec_gradient(ans, x):
    #k = int(np.round((-1 + np.sqrt(1 + 8 * len(ans))) / 2))
    k = np.shape(x)[0]
    def vjp(g):
        entries = [[g[r*(r+1)//2 + c] for c in range(r)]
                   + [1./x[r, r] * g[r*(r+1)//2 + r]]
                   + [0]*(k-r-1)
                   for r in range(k)]
        return np.array(entries)
    return vjp

defvjp(trilpd_to_vec, make_trilpd_to_vec_gradient)


def pd_to_vec(A):
    """Convert a positive-definite matrix A to a vector l of entries from
    its cholesky factor. Diagonal entries are logged so they occupy the full
    real line, and still map back to positive values.
    """
    L = np.linalg.cholesky(A)
    return trilpd_to_vec(L)


def vec_to_pd(l):
    """Convert a vector created by pd_to_vec back to a positive-definite matrix.
    """
    L = vec_to_trilpd(l)
    return np.dot(L, L.T)
