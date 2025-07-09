#!/usr/bin/env python3
"""
check_python_deps.py
Quick dependency checker for Knock-Knock Python analysis tools
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"Python {version.major}.{version.minor} detected")
        print("   Required: Python 3.7+")
        return False
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro}")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is available"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"{package_name}: {version}")
            return True
        else:
            print(f"{package_name}: Not found")
            return False
    except ImportError:
        print(f"{package_name}: Import error")
        return False

def main():
    print("Checking Knock-Knock Python Analysis Dependencies\n")
    
    # Check Python version
    python_ok = check_python_version()
    print()
    
    # Required packages
    required_packages = [
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'), 
        ('Matplotlib', 'matplotlib'),
        ('SciPy', 'scipy'),
        ('Galois', 'galois'),
    ]
    
    # Optional packages
    optional_packages = [
        ('Seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('Jupyter', 'jupyter'),
        ('IPython', 'IPython'),
    ]
    
    print("Required packages:")
    required_ok = all(check_package(name, import_name) for name, import_name in required_packages)
    
    print("\nOptional packages:")
    optional_ok = all(check_package(name, import_name) for name, import_name in optional_packages)
    
    print("\n" + "="*50)
    
    if python_ok and required_ok:
        print("All required dependencies are satisfied!")
        print("   You can run: python full_analysis.py <csv_file> --thresh <threshold>")
        if not optional_ok:
            print("ℹSome optional packages are missing (enhanced features)")
    else:
        print("Missing required dependencies!")
        print("\n  To install missing packages:")
        print("   Option 1: ./setup_python_env.sh  (recommended)")
        print("   Option 2: pip3 install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
