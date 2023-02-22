"""
Performance analysis of predictive (alpha) stock factors

Functions
---------
from_pipeline
    Create a full tear sheet from a zipline Pipeline.

Modules
-------
tears
    Functions for creating tear sheets.

utils
    Utility functions for formatting factor data.
"""
from . import performance
from . import plotting
from . import tears
from . import utils
from .pipeline import from_pipeline

from ._version import get_versions


__version__ = get_versions()['version']
del get_versions

__all__ = [
    'from_pipeline',
    'tears',
    'utils'
]
