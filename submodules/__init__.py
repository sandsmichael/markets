"""
Submodules Package
==================

Core data retrieval and analysis modules for the markets package.

This package contains:
- Market data APIs (EODHD)
- Economic data APIs (FRED)
- Technical analysis tools
- Fundamental analysis tools
- Options analysis
- Statistical helpers
- Visualization utilities
"""

from .eodhd import EODHD, Fundamentals, Index, Security, Chain
from .fred import get_matrix as get_fred_data, get_series, get_observations, fetch_transform_fred
from .ta import TechnicalAnalysis
from .fa import FundamentalAnalysis

__all__ = [
    'EODHD',
    'Fundamentals', 
    'Index',
    'Security',
    'Chain',
    'get_fred_data',
    'get_series',
    'get_observations',
    'fetch_transform_fred',
    'TechnicalAnalysis',
    'FundamentalAnalysis',
]
