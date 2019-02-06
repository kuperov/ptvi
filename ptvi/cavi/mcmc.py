"""Utility functions for summarizing MCMC output.
"""
import pandas as pd
import numpy as np


def mcmc_summ(draws, names=None, true=None):
    """Summarize MCMC draws with given parameter names.

    Draw axis should be axis 0, variables on axis 1.
    
    Args:
        draws: 2-matrix of draws
        names: list of names of variables
        true:  vector of true values for variables
    
    Returns:
        pandas DataFrame summarizing draws
    """
    ndraws, k = draws.shape
    means = np.mean(draws, axis=0)
    qs = np.quantile(draws, axis=0, q=[0.05, 0.25, 0.5, 0.75, 0.95])
    to_join = [np.expand_dims(means,0), qs]
    rows = names.copy() if names else [f'beta[{i}]' for i in range(k)]
    cols = ['Mean','5%','25%','50%','75%','95%']
    if true is not None:
        to_join.insert(1, np.expand_dims(true,0))
        cols.insert(1, 'True')
    res = np.concatenate(to_join, axis=0).T
    frame = pd.DataFrame(res, index=rows, columns=cols)
    return frame