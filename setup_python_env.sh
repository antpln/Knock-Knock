#!/bin/bash

# setup_python_env.sh
# Portable Python virtual environment setup for Knock-Knock analysis tools

set -e  # Exit on any error

echo "Setting up Python virtual environment for Knock-Knock analysis..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ before running this script"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $PYTHON_VERSION detected, but Python $REQUIRED_VERSION+ is required"
    exit 1
fi

echo "Python $PYTHON_VERSION detected"

# Create virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment directory already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment"
        echo "To activate: source venv/bin/activate"
        exit 0
    fi
fi

echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."

# Core scientific computing packages
pip install numpy>=1.19.0
pip install pandas>=1.3.0
pip install matplotlib>=3.3.0
pip install scipy>=1.7.0

# Galois field operations (essential for null-space analysis)
pip install galois>=0.3.0

# Optional but useful packages
pip install seaborn>=0.11.0  # Enhanced plotting
pip install tqdm>=4.60.0     # Progress bars

# Create requirements.txt for reference
echo "Creating requirements.txt..."
pip freeze > requirements.txt

# Create activation script
echo "Creating activation script..."
cat > activate_analysis.sh << 'EOF'
#!/bin/bash
# Activation script for Knock-Knock Python analysis environment

if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Run ./setup_python_env.sh first"
    exit 1
fi

echo "Activating Knock-Knock analysis environment..."
source venv/bin/activate

echo "Environment activated! Available analysis tools:"
echo "   - python full_analysis.py <csv_file> --thresh <threshold>"
echo "   - python -c 'import numpy, pandas, galois; print(\"All packages ready!\")'"
echo ""
echo "To deactivate: deactivate"
EOF

chmod +x activate_analysis.sh

# Test the installation
echo "Testing installation..."
python -c "import numpy, pandas, matplotlib, galois, scipy; print('✅ All core packages imported successfully')"

# Display package versions
echo ""
echo "Installed package versions:"
python -c "
import numpy as np
import pandas as pd
import matplotlib
import galois
import scipy

print(f'  NumPy: {np.__version__}')
print(f'  Pandas: {pd.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
print(f'  Galois: {galois.__version__}')
print(f'  SciPy: {scipy.__version__}')
"

deactivate

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run analysis: python full_analysis.py <csv_file> --thresh <threshold>"
echo "  3. Deactivate when done: deactivate"
echo ""
echo "Quick start: ./activate_analysis.sh"
echo "Requirements saved to: requirements.txt"
