"""
Functions for scaling data
"""
import numpy as np

def scale(X):
    """

    Parameters
    ----------
    X : array-like
        The data that you want to scale.

    Returns
    -------
    array-like
        The scaled data.

    """
    X = np.array(X)
    X_mean = np.mean(X, axis = 0)
    X_std = np.std(X, axis = 0)
    return (X-X_mean)/X_std

def normalise(X):
    """
    
    Parameters
    ----------
    X : array-like
        The data you want to scale.

    Returns
    -------
    array-like
        Data scaled between 0 and 1.

    """
    X = np.array(X)
    
    X_min = np.min(X, axis = 0)
    X_max = np.max(X, axis = 0)
    
    return (X-X_min) / (X_max-X_min)
    