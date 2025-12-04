"""
Setup script for secrets.json file
Run this after installing the package to configure your API keys
"""

import os
import json
from pathlib import Path

# Determine the markets directory (where this script is located)
MARKETS_DIR = Path(__file__).parent
SECRETS_PATH = MARKETS_DIR / "secrets.json"
SUBMODULES_SECRETS = MARKETS_DIR / "submodules" / "secrets.json"
ORIGINAL_SUBMODULES = Path(r"C:\Users\msands\OneDrive\Documents\code\submodules\secrets.json")

def setup_secrets():
    """Set up secrets.json in the markets directory."""
    
    print("=" * 80)
    print("SECRETS.JSON SETUP")
    print("=" * 80)
    
    # Check if secrets already exists in markets root
    if SECRETS_PATH.exists():
        print(f"\n✓ secrets.json already exists at: {SECRETS_PATH}")
        
        # Verify it has the required keys
        try:
            with open(SECRETS_PATH, 'r') as f:
                secrets = json.load(f)
            
            required_keys = ['eodhd_api_key', 'fred_api_key', 'openai_api_key']
            missing_keys = [k for k in required_keys if k not in secrets]
            
            if missing_keys:
                print(f"\n⚠ Warning: Missing keys in secrets.json: {missing_keys}")
                print("Please add these keys to your secrets.json file")
            else:
                print("✓ All required API keys are present")
            
            return True
            
        except json.JSONDecodeError:
            print("✗ Error: secrets.json exists but is not valid JSON")
            return False
    
    # Try to copy from submodules
    if SUBMODULES_SECRETS.exists():
        print(f"\n⚠ secrets.json found in submodules directory")
        print(f"  Source: {SUBMODULES_SECRETS}")
        print(f"  Copying to: {SECRETS_PATH}")
        
        import shutil
        shutil.copy2(SUBMODULES_SECRETS, SECRETS_PATH)
        print("✓ Copied successfully")
        return True
    
    # Try to copy from original submodules folder
    if ORIGINAL_SUBMODULES.exists():
        print(f"\n⚠ secrets.json found in original submodules directory")
        print(f"  Source: {ORIGINAL_SUBMODULES}")
        print(f"  Copying to: {SECRETS_PATH}")
        
        import shutil
        shutil.copy2(ORIGINAL_SUBMODULES, SECRETS_PATH)
        print("✓ Copied successfully")
        return True
    
    # Create a template if no secrets found
    print(f"\n✗ No secrets.json file found")
    print("\nCreating template secrets.json...")
    
    template = {
        "eodhd_api_key": "YOUR_EODHD_API_KEY_HERE",
        "fred_api_key": "YOUR_FRED_API_KEY_HERE",
        "openai_api_key": "YOUR_OPENAI_API_KEY_HERE"
    }
    
    with open(SECRETS_PATH, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Template created at: {SECRETS_PATH}")
    print("\n⚠ IMPORTANT: Edit secrets.json and add your actual API keys!")
    print("\nGet your API keys from:")
    print("  - EODHD: https://eodhd.com/")
    print("  - FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("  - OpenAI: https://platform.openai.com/api-keys")
    
    return False

def verify_package_installation():
    """Check if the markets package is installed."""
    print("\n" + "=" * 80)
    print("PACKAGE INSTALLATION CHECK")
    print("=" * 80)
    
    try:
        import markets
        print(f"\n✓ markets package is installed")
        print(f"  Version: {markets.__version__}")
        print(f"  Location: {markets.__file__}")
        return True
    except ImportError:
        print("\n✗ markets package is not installed")
        print("\nTo install the package, run:")
        print("  cd " + str(MARKETS_DIR))
        print("  pip install -e .")
        return False

def check_git_ignore():
    """Verify secrets.json is in .gitignore."""
    print("\n" + "=" * 80)
    print("GIT IGNORE CHECK")
    print("=" * 80)
    
    gitignore_path = MARKETS_DIR / ".gitignore"
    
    if not gitignore_path.exists():
        print("\n⚠ .gitignore file not found")
        print("Creating .gitignore with secrets.json...")
        
        with open(gitignore_path, 'w') as f:
            f.write("# Secrets\n")
            f.write("secrets.json\n")
            f.write("*/secrets.json\n")
            f.write("**/secrets.json\n")
            f.write("\n# Python\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write("*.pyo\n")
            f.write("*.egg-info/\n")
            f.write("dist/\n")
            f.write("build/\n")
        
        print("✓ .gitignore created")
        return True
    
    # Check if secrets.json is ignored
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    if 'secrets.json' in content:
        print("\n✓ secrets.json is already in .gitignore")
        return True
    else:
        print("\n⚠ secrets.json is NOT in .gitignore")
        print("Adding it now...")
        
        with open(gitignore_path, 'a') as f:
            f.write("\n# Secrets\n")
            f.write("secrets.json\n")
            f.write("*/secrets.json\n")
            f.write("**/secrets.json\n")
        
        print("✓ Added secrets.json to .gitignore")
        return True

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MARKETS PACKAGE SETUP" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Run all checks
    secrets_ok = setup_secrets()
    package_ok = verify_package_installation()
    git_ok = check_git_ignore()
    
    # Final summary
    print("\n" + "=" * 80)
    print("SETUP SUMMARY")
    print("=" * 80)
    
    if secrets_ok and package_ok and git_ok:
        print("\n✓ All setup complete! You're ready to use the markets package.")
        print("\nYou can now:")
        print("  1. Open any notebook in implementations/")
        print("  2. Import with: from markets.submodules import Security, Index, ...")
        print("  3. Make changes to code - they'll be immediately available (editable install)")
    else:
        print("\n⚠ Setup incomplete. Please address the issues above.")
        
        if not secrets_ok:
            print("\n  ☐ Configure your API keys in secrets.json")
        if not package_ok:
            print("\n  ☐ Install package: pip install -e .")
        
    print("\n" + "=" * 80)
