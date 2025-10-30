"""
Module used for Linear Regression
"""

import numpy as np

class LinearRegression:
    """
    A class that can be used for Linear Regression
    """
    def __init__(self):
        self.slopes = None
        self.intercept = None
        self.parameters = None
        
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
        Changes parameters inplace. Calculates parameters of model based on data

        """
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshapes if X is 1D to allow for the calculations to work
        
        X = np.insert(X, 0, 1, axis=1) # Adding extra column for an intercept
        
        self.parameters, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.intercept = self.parameters[0]
        self.slopes = self.parameters[1:]
        
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
        
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Single point becomes 1 row
            
        y_pred = x @ self.slopes + self.intercept
        
        if y_pred.shape[0] == 1:
            return y_pred[0]   # Returns a float instead of a one element array
        
        return y_pred
    
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
    

