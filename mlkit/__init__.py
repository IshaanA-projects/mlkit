"""
mlkit: A simple and modular machine learning library in Python.

Overview
--------
mlkit provides a collection of machine learning models.
It follows a modular structure, allowing users to import only the components
they need.

Submodules
----------
- linear_models: Implements linear models such as LinearRegression.
- neighbours: Implements nearest neighbours models such as KNN
"""

from . import linear_model
from . import neighbours
from . import utils