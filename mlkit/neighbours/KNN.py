"""
Module implementing K Nearest Neighbours (KNN) classification
"""


import numpy as np
from collections import Counter

class KNeighboursClassifier:
    """
    A class that can be used for classification with K Nearest Neighbours
    """
    def __init__(self, k=5):
        """
        

        Parameters
        ----------
        k : int, optional
            How many points we will compare to. The default is 5.

        Returns
        -------
        None.

        """
        self.k = k
    
    def fit(self, data, classification):
        """
        

        Parameters
        ----------
        data : np array or list
            Datapoints without their class, only the features
        classification : np array or list
            The classes of the datapoints. The classes should map up directly with the data

        Returns
        -------
        None. Creates variables inplace

        """
        self.data = np.array(data)
        self.classification = np.array(classification)
        
        
    def predict(self, point):
        
        """

        Parameters
        ----------
        point : np.array or list
            The point that you want to classify.

        Returns
        -------
        vote_result : str
            The class of your point as determined by the KNN algorithm.

        """
    
        distances = []
        for i in range(len(self.data)):
            euclidean_distance = np.linalg.norm(np.array(self.data[i]) - np.array(point))
            distances.append([euclidean_distance, self.classification[i]])
        
        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        
        return vote_result
    def score(self, X, y):
        """
        

        Parameters
        ----------
        X : np.array or list
            Datapoints without their class, only the features
        y : np.array or list
            The classes of the datapoints, what we want to predict. The classes should map up directly with the data.

        Returns
        -------
        accuracy : float
            The accuracy of our model in classifying the points

        """
        correct = 0
        total = 0
        for i in range(len(X)):
            vote = self.predict(X[i])
            if vote == y[i]:
                correct +=1
            total +=1
        self.accuracy = correct / total
        return self.accuracy * 100

