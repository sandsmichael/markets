"""
Markets Package Setup
=====================
Combined markets analysis and submodules package for pip installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "readme.md").read_text(encoding='utf-8')

setup(
    name="markets",
    version="0.2.0",
    author="Michael Sands",
    author_email="",
    description="Comprehensive market analysis toolkit with data retrieval, technical analysis, and fundamental analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandsmichael/submodules",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "wip", "output", "img"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "requests>=2.26.0",
        
        # Technical analysis
        "TA-Lib>=0.4.24",
        "yfinance>=0.2.0",
        
        # Statistical analysis
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        
        # Date handling
        "python-dateutil>=2.8.0",
        
        # Visualization (optional)
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        
        # Progress bars
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "options": [
            "QuantLib>=1.28",
        ],
        "full": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "QuantLib>=1.28",
        ],
    },
    include_package_data=True,
    package_data={
        "markets": ["*.json"],
        "markets.submodules": ["*.json"],
    },
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here
            # "markets-cli=markets.cli:main",
        ],
    },
)
