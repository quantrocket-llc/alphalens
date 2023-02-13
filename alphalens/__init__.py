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
    'performance',
    'plotting',
    'tears',
    'utils'
]
