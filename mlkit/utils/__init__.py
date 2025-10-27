"""
The utils subpackage provides generally helpful functions for data splitting and processing

Submodules
----------
- model_selection: Functions for splitting datasets.
- preprocessing: Functions for scaling and normalizing data.
"""


from .model_selection import train_test_split
from .preprocessing import scale, normalise