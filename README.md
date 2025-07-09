# Knock-Knock: Blind, Platform-Agnostic DRAM Address-Mapping Reverse Engineering

This repository contains the implementation of **Knock-Knock**, a novel methodology for reverse engineering DRAM memory controller address mappings using timing-based side-channel attacks. The tool provides an automated approach to recover bank and row addressing functions without requiring platform-specific knowledge.

# /!\ Disclaimer
This is a Proof-of-Concept of Knock-Knock. While it yields expected results, the fully automated pipeline is not yet implemented at the time of submission. However, the missing automated features, mainly automatic threshold detection and the second timing analysis from the found masks, will be implemented for the artifact evaluation and publication.

## Building and Running

### Prerequisites
- Linux system with root privileges (for `/proc/self/pagemap` access)
- GCC with C++11 support
- Make build system
- Python 3

### Performance Counter Setup (ARMv8 only)
**Note**: The performance counter kernel module is only required for ARMv8 devices. On x86_64 systems, the program will work without additional setup.

**For ARMv8 devices only**: Install the kernel module provided in the `enable_arm_pmu/` directory:

```bash
cd enable_arm_pmu
make
sudo ./load-module
```

### Compilation
```bash
# Build the project
make clean && make

# The binary will be created as obj/tester
```

### Python Analysis Prerequisites

The Python analysis component requires specific packages for mathematical operations and data processing:

#### Option 1: Using Portable Virtual Environment (Recommended)
A portable Python virtual environment is provided in the repository for easy setup:

```bash
# Set up the Python virtual environment (one-time setup)
./setup_python_env.sh

# Activate the virtual environment
source venv/bin/activate

# Run analysis (virtual environment active)
python full_analysis.py access_module_1024.csv --thresh 150

# Deactivate when done
deactivate
```

#### Option 2: Manual Installation
If you prefer to install dependencies manually:

```bash
# Install Python dependencies
pip3 install --user numpy pandas matplotlib galois scipy

# Verify installation
python3 -c "import numpy, pandas, matplotlib, galois, scipy; print('All dependencies installed successfully')"
```

#### Required Python Packages
- **Python 3.7+**: Base interpreter
- **NumPy**: Numerical computing and matrix operations
- **Pandas**: Data manipulation and CSV processing  
- **Matplotlib**: Plotting and visualization
- **galois**: Galois field (GF(2)) operations for linear algebra
- **SciPy**: Scientific computing and statistical functions

#### Dependency Check
To verify your Python environment is ready:

```bash
# Quick dependency check
python3 check_python_deps.py

# Should output all green checkmarks for required packages
```

### Usage
```bash
# Basic usage (requires root for pagemap access)
sudo ./obj/tester

# Specify memory size (25GB default)
sudo ./obj/tester -m 8192  # Use 8GB

# Use percentage of total memory
sudo ./obj/tester -p 50    # Use 50% of system memory

# Run bitflip analysis instead of timing measurement
sudo ./obj/tester --bitflip

# Customize number of measurements and rounds
sudo ./obj/tester -n 50000 -r 100

# Specify output file
sudo ./obj/tester -o custom_output.csv
```

### Command-Line Options
- `-h`: Show help message
- `-m <size_mb>`: Memory size in MB (default: 25600)
- `-p <percentage>`: Memory size as percentage of total system memory
- `-r <rounds>`: Number of timing measurement rounds (default: 50)
- `-n <measurements>`: Number of measurements to perform (default: 100000)
- `-o <file>`: Output file for results (default: output.csv)
- `--timing`: Run timing measurement mode (default)
- `--bitflip`: Run bitflip probing analysis
- `-v`: Verbose output

## Output Files

The tool generates CSV files containing timing measurements and address mappings:

### Timing Measurement Output (`access_module_<size>.csv`)
```
a1,a2,elapsed_cycles,v_a1,v_a2
1a2b3c4d,5e6f7890,234,7f8e9d0c,1b2a3948
...
```
- `a1`, `a2`: Physical addresses (hexadecimal)
- `elapsed_cycles`: Measured access time in CPU cycles
- `v_a1`, `v_a2`: Virtual addresses (hexadecimal)

### Bitflip Analysis Output (`bitflip_probe_<size>.csv`)
```
anchor_va,a1,delta,probe_va,a2,elapsed_cycles
deadbeef,12345678,00000400,cafebabe,12341278,156
...
```
- `anchor_va`, `probe_va`: Virtual addresses
- `a1`, `a2`: Physical addresses
- `delta`: XOR difference applied
- `elapsed_cycles`: Measured timing

## Architecture Support

### ARM64 (AArch64)
- Uses `DC CIVAC` for cache line eviction
- Employs `PMCCNTR_EL0` performance counter for high-precision timing
- Includes anti-speculation techniques with memory barriers

### x86/x86_64
- Uses `clflush` instruction for cache eviction
- Employs `rdtsc` instruction for timing measurements
- Includes appropriate memory fences and barriers

## Features

### Data Generation (C++)
- **Timing Measurements**: Perform precise memory access timing analysis using hardware performance counters
- **Bitflip Probing**: Optional memory mapping analysis through controlled bit manipulation
- **Flexible Memory Allocation**: Support for both fixed memory sizes and percentage-based allocation
make
```

The executable will be created as `obj/tester`.

## Usage

### How the Program Works

The program performs memory analysis through two main operations:

1. **Timing Measurements** (Default): Analyzes memory access patterns by measuring timing differences between memory accesses to different physical addresses.
2. **Bitflip Probing** (Optional): Performs controlled bit manipulation to analyze memory mapping and identify relationships between virtual and physical addresses.

### Running the Program

The program must be run with root privileges to access system memory information and performance counters.

#### Basic Workflow:
1. **Build the program** (one-time setup)
2. **Install ARM PMU module** (ARMv8 only, one-time setup)
3. **Run measurements** with desired parameters
4. **Analyze output** CSV files

### Command Line Usage

### Basic Usage
```bash
sudo ./obj/tester [OPTIONS]
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h` | Show help message | - |
| `-o <file>` | Output file for memory profiling | `output.csv` |
| `-m <size_mb>` | Memory size in MB | `25600` (25GB) |
| `-p <percent>` | Memory size as percentage of total (overrides `-m`) | - |
| `-r <rounds>` | Number of rounds | `50` |
| `-n <measurements>` | Number of measurements | `100000` |
| `--timing` | Run timing measurement instead of rev_mc | - |
| `--bitflip` | Run mapping bitflip probe | - |
| `-v` | Verbose output | - |

### Examples

#### Step-by-Step: First Run
1. **Build the program:**
   ```bash
   make clean && make
   ```

2. **For ARMv8 systems only - Install PMU module:**
   ```bash
   make
   sudo ./load-module
   cd ..
   ```

3. **Run basic timing measurement:**
   ```bash
   sudo ./obj/tester
   ```
   This uses default settings: 25GB memory, 50 rounds, 100,000 measurements

4. **Check output:**
   ```bash
   ls -la access_module_*.csv
   ```

### Memory Size Configuration

**Use percentage of total memory (recommended):**
```bash
# Use 10% of total system memory
sudo ./obj/tester -p 10

# Use 1% for quick testing
sudo ./obj/tester -p 1
```

**Use fixed memory size:**
```bash
# Use 1GB of memory
sudo ./obj/tester -m 1024

# Use 512MB of memory
sudo ./obj/tester -m 512
```

### Performance Tuning

**Quick test run:**
```bash
sudo ./obj/tester -p 1 -r 5 -n 1000
```

**High-precision measurement:**
```bash
sudo ./obj/tester -p 20 -r 100 -n 500000
```

### Advanced Usage

**Run bitflip analysis:**
```bash
sudo ./obj/tester -p 5 --bitflip
```

**Custom output file:**
```bash
sudo ./obj/tester -p 10 -o my_experiment.csv
```

**Verbose output for debugging:**
```bash
sudo ./obj/tester -p 1 -n 100 -v
```

## Analysis

### Prerequisites Check
Before running analysis, verify your Python environment:

```bash
# Check all dependencies are installed
python3 check_python_deps.py

# If using virtual environment
source venv/bin/activate
python check_python_deps.py
```

### Running the Analysis Tool

After collecting timing data with the data generation tool, use the Python analysis script to extract memory patterns:

```bash
# If using virtual environment
source venv/bin/activate

# Run analysis
python3 full_analysis.py <csv_file> --thresh <threshold> [OPTIONS]
```

### Analysis Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `csv_file` | Path to CSV file generated by data collection tool | Required |
| `--thresh <cycles>` | Threshold latency separating conflicts from non-conflicts | Required |
| `--subsample <size>` | Subsample size for nullspace analysis | `1000` |
| `--repeat <count>` | Number of repetitions for subsampling | `50` |
| `--limit <count>` | Limit number of pairs to process (for testing) | None |
| `--verbose`, `-v` | Enable verbose output | False |

### Complete Workflow Example

```bash
# 1. Generate timing data
sudo ./obj/tester -p 5 -n 50000 -r 50

# 2. Analyze the results (adjust threshold based on your data)
python3 full_analysis.py access_module_<size>.csv --thresh 150 --verbose

# 3. For high-precision analysis
python3 full_analysis.py access_module_<size>.csv --thresh 150 --subsample 2000 --repeat 100
```