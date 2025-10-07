# Knock-Knock: Black-Box, Platform-Agnostic DRAM Address-Mapping Reverse Engineering [uASC '26]
## Antoine Plin, Lorenzo Casalino, Thomas Rokicki, Ruben Salvador 

*Automated DRAM address mapping reverse engineering for x86_64, ARMv8, and POWER9/10.*
Preprint available at https://arxiv.org/abs/2509.19568

## Overview

Knock-Knock provides two workflows for DRAM reverse engineering:

1. **Automatic Pipeline** (`main --full-analysis`): Complete C++ implementation that automatically detects thresholds, discovers bank masks, and derives row masks in one run.

2. **Manual Pipeline** (`main --timing` + `full_analysis.py`): Two-stage process where C++ collects timing data, then Python performs offline analysis with full control over parameters.

**Platform Support:** x86_64, ARMv8 (AArch64), POWER9/10

**Core Techniques:**
- Timing-based row buffer conflict detection
- GF(2) nullspace analysis for mask discovery
- Minimal-weight basis optimization
- Automated threshold detection using histogram analysis

## Installation

### 1. Build
```bash
make
```
This produces `main` (automatic pipeline) and supporting components.

### 2. ARM64 PMU Setup (ARM platforms only)
```bash
cd enable_arm_pmu
make
sudo ./load-module
```

### 3. Python Dependencies (for manual pipeline only)
```bash
./setup_python_env.sh           # Automated setup
source venv/bin/activate

# Or install manually:
pip3 install numpy pandas matplotlib galois scipy
python3 check_python_deps.py
```

---

## Workflow 1: Automatic Pipeline (Recommended)

The automatic pipeline runs all three stages in C++ with a single command. It performs threshold detection, bank mask inference, and row mask derivation automatically.

### Basic Usage
```bash
sudo ./main --full-analysis -p 50
```

This runs the complete analysis using 50% of system memory.

### Pipeline Stages

**Stage 1: Threshold Detection**
- Collects random memory access latencies
- Builds histogram and applies smoothing
- Uses "Find the Bump's Left Foot" heuristic to separate hits from conflicts
- Outputs: `latencies.dat`, `smoothed_histogram.dat`, `analysis_points.dat`

**Stage 2: Bank Mask Discovery**
- Collects conflict samples (same-bank pairs)
- Performs GF(2) nullspace analysis with subsampling
- Finds minimal-weight basis of candidate masks
- Evaluates accuracy with confusion matrix

**Stage 3: Row Mask Derivation**
- Collects same-bank pairs with known timing behavior
- Applies invariance filtering (bits constant in hits, varying in conflicts)
- Derives row masks from bank mask nullspace
- Validates against collected samples

### Automatic Pipeline Parameters

#### Memory Configuration
- `-m <MB>`: Memory size in megabytes (default: 25600 MB / 25 GB)
- `-p <percent>`: Portion of system DRAM to allocate (overrides `-m`)
- `-d, --delay <us>`: Inter-pair delay in microseconds to avoid tFAW violations (default: 0)

#### Stage 1: Threshold Detection Options
- `--threshold <cycles>`: Manually set threshold, skip auto-detection (default: auto)
  - Example: `--threshold 260`
- `--threshold-samples <n>`: Samples for histogram (default: 100000)
  - Example: `--threshold-samples 200000`

#### Stage 2: Bank Mask Detection Options
- `--bank-conflicts <n>`: Target conflict samples (default: 5000)
- `--bank-measurements <n>`: Max measurements (default: 3000000)
- `--bank-subsample <n>`: Subsample size for nullspace (default: 1000)
- `--bank-rounds <n>`: Subsampling iterations (default: 35)
- `--bank-attempts <n>`: Max retry attempts (default: 3)

#### Stage 3: Row Mask Detection Options
- `--row-pairs <n>`: Target same-bank pairs (default: 8000)
- `--row-min-hits <n>`: Minimum hit samples (default: 1000)
- `--row-min-conflicts <n>`: Minimum conflict samples (default: 1000)
- `--row-max-attempts <n>`: Max sampling attempts (default: 2400000)

#### Other Options
- `-v`: Verbose output
- `-d, --delay <us>`: Delay between measurements (µs) for tFAW mitigation (default: 0)
- `--force-multiple-rounds`: Force multiple rounds for validation

### Examples

#### Basic Run
```bash
sudo ./main --full-analysis -p 25
```

#### High Precision
```bash
sudo ./main --full-analysis -p 50 \
  --bank-conflicts 10000 --bank-rounds 60 --row-pairs 15000
```

#### Manual Threshold (Skip Stage 1)
```bash
sudo ./main --full-analysis --threshold 260 -p 25
```

#### Difficult/Noisy Systems
```bash
sudo ./main --full-analysis -p 50 \
  --threshold-samples 200000 \
  --bank-conflicts 15000 --bank-subsample 2000 --bank-rounds 80 \
  --row-pairs 20000 --row-min-hits 3000
```

#### With tFAW Mitigation
```bash
sudo ./main --full-analysis -p 50 --delay 10
```

### Output Files
- `latencies.dat`: Raw latency samples for threshold analysis
- `smoothed_histogram.dat`: Histogram with smoothing applied
- `analysis_points.dat`: Auto-detected threshold, peaks, confidence score
- `threshold_analysis_summary.txt`: Detailed threshold detection report

---

## Workflow 2: Manual Pipeline (Python Analytics)

The manual pipeline separates data collection (C++) from analysis (Python), giving you full control over the analysis parameters and allowing iterative experimentation.

### Step 1: Collect Timing Data (C++)

Use the legacy timing measurement mode to generate CSV files:

```bash
sudo ./main --timing -a -p 50 -n 100000 -r 50
```

#### Data Collection Parameters
- `-a`: Auto-name output as `data/<hostname>_<mem>.csv`
- `-o <file>`: Specify custom output file
- `-p <percent>`: Memory allocation (% of system RAM)
- `-m <MB>`: Memory allocation (absolute MB, overridden by `-p`)
- `-n <count>`: Number of address pairs to measure (default: 100000)
- `-r <rounds>`: Timing rounds per pair for median calculation (default: 50)
- `-d <us>`: Inter-measurement delay in microseconds

#### Alternative: Bitflip Probe Mode
```bash
sudo ./main --bitflip -p 50 -r 50
```
Systematically flips bits in physical addresses to probe bank mapping functions.

### Step 2: Analyze with Python

```bash
python3 full_analysis.py data/<hostname>_<mem>.csv --thresh <value> [options]
```

#### Required Parameter
- `--thresh <cycles>`: Latency threshold separating hits from conflicts
  - Determine this by examining the timing distribution
  - Typical values: 150-300 cycles depending on system

#### Analysis Parameters
- `--subsample <n>`: Subsample size for GF(2) nullspace (default: 1000)
  - Controls memory usage and convergence speed
  - Larger = slower but potentially more accurate
  
- `--repeat <n>`: Number of subsampling rounds (default: 50)
  - More rounds = better mask frequency statistics
  - Increase for difficult systems (60-100)

- `--sensitivity <float>`: Sensitivity for row bit analysis (default: 0.05)
  - Range: 0.0 to 1.0
  - Lower = stricter filtering

- `--limit <n>`: Limit pairs processed (for testing)
  - Useful for quick validation runs

- `--verbose, -v`: Detailed progress output

#### Python Pipeline Output
The script performs:
1. **Data loading** with consistency filtering
2. **Binary difference matrix** construction
3. **GF(2) nullspace analysis** with subsampling
4. **Minimal-weight basis** optimization
5. **Accuracy evaluation** (precision, recall, F1 score)
6. **Bank-separated timing** analysis
7. **Row buffer analysis** with invariant detection

### Visualization

After either workflow, visualize threshold detection:

```bash
python3 plot_histogram.py [--interactive] [--save output.png]
```

Options:
- `--interactive`: Click to select manual threshold
- `--save <file>`: Save plot instead of displaying

### CSV Output Format
```csv
a1,a2,elapsed_cycles,v_a1,v_a2
1a2b3c4d,5e6f7890,234,7f8e9d0c,1b2a3948
```
- `a1`, `a2`: Physical addresses (hex)
- `elapsed_cycles`: Access latency in CPU cycles
- `v_a1`, `v_a2`: Virtual addresses (hex)

---

## Platform-Specific Implementation

### x86/x86_64
- **Cache eviction:** `clflush` instruction
- **Timing:** `rdtsc`/`rdtscp` timestamp counter
- **Barriers:** `mfence`, `lfence`

### ARM64 (AArch64)
- **Cache eviction:** `DC CIVAC` (clean & invalidate)
- **Timing:** `PMCCNTR_EL0` cycle counter (requires PMU kernel module)
- **Barriers:** `DSB`, `ISB`
- **Anti-speculation:** Dependency chains to prevent speculative loads

### POWER (ppc64le)
- **Cache eviction:** `dcbf` (data cache block flush)
- **Timing:** Time-base register
- **Barriers:** `sync`, `isync`

---

## Troubleshooting

### Automatic Pipeline

**Bank masks fail to converge:**
- Increase conflict samples: `--bank-conflicts 10000`
- More subsampling rounds: `--bank-rounds 80`
- Larger subsample: `--bank-subsample 2000`
- Verify threshold: `python3 plot_histogram.py`

**Row masks not found:**
- Increase same-bank pairs: `--row-pairs 15000`
- Raise minimums: `--row-min-hits 2000`
- More attempts: `--row-max-attempts 3000000`

**Threshold detection fails:**
- Manual threshold: `--threshold <value>`
- More samples: `--threshold-samples 200000`
- Visualize distribution: `python3 plot_histogram.py`

### Manual Pipeline

**Determining threshold:**
1. Generate histogram: `python3 plot_histogram.py --interactive`
2. Look for the "left foot" where high-latency bump begins
3. Choose value in the valley between two modes

**Low accuracy (<70%):**
- Verify threshold is correct
- Increase Python analysis rounds: `--repeat 100`
- Larger subsample: `--subsample 2000`
- Collect more data with longer C++ runs

**Python analysis tips:**
- Start with `--verbose` to see detailed progress
- Use `--limit 50000` for quick testing
- Increase `--repeat` if masks are inconsistent
- Check consistency filtering output

### General Issues

**System-specific:**
- High contention: Add `--delay 10`
- Noisy timings: Double all sample counts
- Slow systems: Reduce memory `-p 10`
- Ensure system is idle during measurements

## Requirements
- Linux with root access (for `/proc/self/pagemap` and PMU control)
- GCC with C++11 support, GNU make
- Python 3.8+ (optional, for analytics)

## Contributing
Issues and PRs welcome. Please document your hardware platform, kernel version, and parameter configuration. Keep patches focused and well-tested.

## Citation
If you use this tool in your research, please cite the Knock-Knock paper and let us know about your findings.
```
@misc{plin2025knockknock,
      title={Knock-Knock: Black-Box, Platform-Agnostic DRAM Address-Mapping Reverse Engineering}, 
      author={Antoine Plin and Lorenzo Casalino and Thomas Rokicki and Ruben Salvador},
      year={2025},
      eprint={2509.19568},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2509.19568}, 
}
```
