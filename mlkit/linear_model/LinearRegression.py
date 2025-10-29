"""
Module used for Linear Regression
"""

import numpy as np

class LinearRegression2d:
    """
    A class that can be used for Linear Regression when both features and labels are 1 dimensional
    """
    def __init__(self):
        self.slope = None
        self.y_int = None
    def fit(self, X, y):
        """
        

        Parameters
        ----------
        X : np array
            Features of the dataset
        y : np array
            Labels of the dataset

        Returns
        -------
        Changes parameters inplace. Calculates slope and intercept of line of best fit when features and labels are 1 dimensional data

        """
        X = np.array(X)
        y = np.array(y)
        self.slope = ((np.mean(X)*np.mean(y))-(np.mean(X*y)))/((np.mean(X)*np.mean(X))-(np.mean(X**2)))
        self.y_int = np.mean(y)- self.slope*np.mean(X)
        return 
    def predict(self, x):
        """
        

        Parameters
        ----------
        x : float or numpy array
            A point or group of points that the user wants to predict at

        Returns
        -------
            The model's prediction at this point/ group of points.

        """
        x = np.array(x)
        return x * self.slope + self.y_int
    
    def score(self, X, y):
        """
        

        Parameters
        ----------
        X : np array
            Features.
        y : np array
            Labels.

        Returns
        -------
        The coefficient of determination of the model on these datasets

        """
        X = np.array(X)
        y = np.array(y)
        y_hat_error = np.mean((self.predict(X)-y)**2)
        y_mean_error = np.mean((np.mean(y)- y)**2)
        self.accuracy = 1 - (y_hat_error/y_mean_error)
        return self.accuracy
    

