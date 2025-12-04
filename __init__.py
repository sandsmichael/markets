"""
Markets - Comprehensive Market Analysis Toolkit
================================================

A Python package for market data retrieval, technical analysis, fundamental analysis,
and macroeconomic data integration.

Modules:
--------
- submodules.eodhd: EODHD market data API wrapper
- submodules.fred: Federal Reserve Economic Data (FRED) API wrapper
- submodules.ta: Technical analysis indicators
- submodules.fa: Fundamental analysis tools
- submodules.options: Options analysis toolkit
- submodules.assist: Statistical analysis helpers
- submodules.plot: Visualization utilities
- submodules.calendar: Calendar utilities
- submodules.fixed_income: Fixed income analysis
- submodules.fammafrench: Fama-French factor models

Example Usage:
--------------
>>> from markets.submodules import Security, get_fred_data
>>> 
>>> # Get stock prices
>>> aapl = Security('AAPL.US', start='2024-01-01')
>>> prices = aapl.eoddf
>>> 
>>> # Get macro data
>>> gdp = get_fred_data(['GDP', 'CPIAUCSL'])
"""

__version__ = "0.2.0"
__author__ = "Michael Sands"

# Import key classes and functions for convenience
try:
    from .submodules.eodhd import EODHD, Fundamentals, Index, Security, Chain
    from .submodules.fred import get_matrix, get_series, get_observations, fetch_transform_fred
    from .submodules.ta import TechnicalAnalysis
    from .submodules.fa import FundamentalAnalysis
    
    __all__ = [
        # EODHD classes
        'EODHD',
        'Fundamentals',
        'Index',
        'Security',
        'Chain',
        
        # FRED functions
        'get_matrix',
        'get_series',
        'get_observations',
        'fetch_transform_fred',
        
        # Analysis classes
        'TechnicalAnalysis',
        'FundamentalAnalysis',
    ]
except ImportError:
    # If imports fail (e.g., during installation), just set version
    __all__ = []
