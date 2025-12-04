# Building and Installing the Markets Package

## Prerequisites

1. **Move submodules into markets folder:**
   ```bash
   # Navigate to your code directory
   cd C:\Users\msands\OneDrive\Documents\code\markets
   
   # Copy all Python files from submodules to markets/submodules
   # (Keep the original submodules folder as backup)
   ```

2. **Ensure all required files are in place:**
   ```
   markets/
   ├── setup.py
   ├── pyproject.toml
   ├── MANIFEST.in
   ├── __init__.py
   ├── readme.md
   ├── submodules/
   │   ├── __init__.py
   │   ├── eodhd.py
   │   ├── fred.py
   │   ├── ta.py
   │   ├── fa.py
   │   ├── options.py
   │   ├── assist.py
   │   ├── plot.py
   │   ├── calendar.py
   │   ├── fixed_income.py
   │   ├── fammafrench.py
   │   └── secrets.json (optional, for development)
   ├── blacklitterman.ipynb
   ├── fundamentals.ipynb
   └── ... (other notebooks and folders)
   ```

## Building the Package

### Method 1: Build Wheel File (Recommended)

```bash
# Navigate to markets directory
cd C:\Users\msands\OneDrive\Documents\code\markets

# Install build tools (if not already installed)
pip install build wheel

# Build the package (creates both .whl and .tar.gz)
python -m build

# This creates files in dist/ directory:
# - markets-0.2.0-py3-none-any.whl
# - markets-0.2.0.tar.gz
```

### Method 2: Build using setup.py directly

```bash
# Navigate to markets directory
cd C:\Users\msands\OneDrive\Documents\code\markets

# Build wheel
python setup.py bdist_wheel

# Build source distribution
python setup.py sdist
```

## Installing the Package

### Option 1: Install from Wheel File

```bash
# Install the built wheel
pip install dist/markets-0.2.0-py3-none-any.whl

# Or install in editable mode for development
pip install -e .
```

### Option 2: Install Directly from Directory

```bash
# Install from current directory
cd C:\Users\msands\OneDrive\Documents\code\markets
pip install .

# Or in editable mode (recommended for development)
pip install -e .
```

### Option 3: Install with Optional Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with options support (QuantLib)
pip install -e ".[options]"

# Install with all dependencies
pip install -e ".[full]"
```

## Verifying Installation

```python
# In Python, verify the package is installed
import markets
print(markets.__version__)

# Import specific modules
from markets.submodules import Security, get_fred_data

# Test basic functionality
security = Security('AAPL.US')
print(f"Latest price: {security.last_close}")

# Get FRED data
gdp = get_fred_data(['GDP'])
print(gdp.tail())
```

## Editable Mode (Development)

When developing, use editable mode so changes are immediately reflected:

```bash
cd C:\Users\msands\OneDrive\Documents\code\markets
pip install -e ".[dev]"
```

This allows you to:
- Edit code without reinstalling
- Test changes immediately
- Develop and use simultaneously

## Updating the Package

After making changes:

```bash
# If installed in editable mode, no action needed
# Changes are automatically available

# If installed normally, rebuild and reinstall
python -m build
pip install --upgrade --force-reinstall dist/markets-0.2.0-py3-none-any.whl
```

## Using in Other Projects

Once installed, you can use it in any Python project:

```python
# In any Python file or notebook
from markets.submodules import Security, Index, get_fred_data
from markets.submodules.ta import TechnicalAnalysis
from markets.submodules.fa import FundamentalAnalysis

# Use the classes
aapl = Security('AAPL.US', start='2024-01-01')
sp500 = Index('GSPC.INDX')
macro_data = get_fred_data(['GDP', 'CPIAUCSL'])
```

## For GPTrader Integration

Update your gptrader tools.py imports:

```python
# Old (before packaging)
sys.path.append(r"C:\Users\msands\OneDrive\Documents\code\submodules")
from eodhd import Security
from fred import get_matrix

# New (after packaging and installation)
from markets.submodules import Security
from markets.submodules.fred import get_matrix as get_fred_data
```

## Uninstalling

```bash
pip uninstall markets
```

## Distribution

To share the package:

1. **Share the wheel file:**
   ```bash
   # Send the .whl file from dist/ directory
   # Others can install with: pip install markets-0.2.0-py3-none-any.whl
   ```

2. **Publish to PyPI (optional):**
   ```bash
   # Install twine
   pip install twine
   
   # Upload to PyPI (requires account)
   python -m twine upload dist/*
   ```

3. **Install from GitHub:**
   ```bash
   pip install git+https://github.com/sandsmichael/submodules.git
   ```

## Troubleshooting

### TA-Lib Installation Issues

TA-Lib requires binary dependencies. On Windows:

```bash
# Download the appropriate .whl from:
# https://github.com/cgohlke/talib-build/releases

# Install the downloaded wheel
pip install TA_Lib-0.4.XX-cpXXX-cpXXX-win_amd64.whl

# Then install markets
pip install -e .
```

### Import Errors

If you get import errors after installation:

```python
# Check installation
pip show markets

# Check Python path
import sys
print(sys.path)

# Reinstall in editable mode
pip uninstall markets
pip install -e .
```

### Secrets File

The `secrets.json` file is excluded from the package. Each installation needs its own:

```json
{
  "eodhd_api_key": "your_key_here",
  "fred_api_key": "your_key_here",
  "openai_api_key": "your_key_here"
}
```

Place this file in:
- Development: `markets/submodules/secrets.json`
- After install: In your working directory where you run scripts
