"""
Functions for splitting data into train and test sets
"""

import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    

    Parameters
    ----------
    X : array-like
        The features of your dataset.
    y : array-like
        The labels of you dataset.
    test_size : float, optional
        The percentage of your dataset you want to be in the test set. The default is 0.2.
    random_state : int, optional
        The random seed that the user wants to use. The default is None.

    Returns
    -------
    array-like
        The train set of X.
    array-like
        The test set of X.
    array-like
        The train set of y..
    array-like
        The test set of y..

    """
    
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    shift = int(n_samples * (1 - test_size))
    
    train_idx, test_idx = indices[:shift], indices[shift:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
