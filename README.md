# SAGE-PSO: Semi-Analytic Galaxy Evolution Model Particle Swarm Optimization Package

A robust, modular Python package for PSO-based parameter optimization and analysis in galaxy evolution modeling using the SAGE semi-analytic model.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Package Structure](#package-structure)
- [Usage](#usage)
  - [Basic Workflow](#basic-workflow)
  - [Command-Line Arguments](#command-line-arguments)
  - [Batch and HPC Usage](#batch-and-hpc-usage)
- [Data Files and Formats](#data-files-and-formats)
- [Constraints System](#constraints-system)
- [Diagnostics and Output](#diagnostics-and-output)
- [Testing and Validation](#testing-and-validation)
- [Extending and Customization](#extending-and-customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- Particle Swarm Optimization (PSO) for parameter fitting and model calibration
- Constraint-based analysis (SMF, BHMF, BHBM, CSFRDH, HIMF, etc.)
- Automatic diagnostics and output management, including plots and summary statistics
- Self-healing workflow: missing `sage_*.csv` data files are automatically regenerated from SAGE output
- Modular codebase for easy extension and customization
- All data files accessed via the local `data/` folder; no hardcoded paths
- Supports batch and HPC workflows (SLURM integration)
- Comprehensive logging and reproducibility

---

## Requirements

- Python 3.8 or newer
- SAGE binary (must be provided by the user)
- SAGE input `.par` file (must be provided by the user)
- Python dependencies (see `requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - scipy
  - h5py
  - seaborn
  - scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sage-optim.git
cd sage-optim
pip install -r requirements.txt
```

---

## Package Structure

```
sage-optim/
  analysis.py              # Statistical tests and search space loading
  common.py                # Shared utilities and configuration parsing
  constraints.py           # Constraint classes and parsing logic
  diagnostics.py           # Diagnostic plots and PSO visualization
  diagnostics2.py          # Additional diagnostics and plotting
  execution.py             # SAGE execution and workflow management
  main.py                  # Main entry point for PSO workflow
  pso.py                   # PSO algorithm implementation
  pso_uncertainty.py       # Uncertainty analysis and reporting
  routines.py              # Data processing and helper routines
  redshift_utils.py        # Redshift handling utilities
  data/                    # All required observational and model data files
  plots_and_random/        # Plotting scripts and utilities
  test_constraint_data.py  # Quick constraint data tests
  test_pso_benchmarks.py   # PSO benchmark tests
  quick_pso_test.py        # Minimal PSO sanity check
  requirements.txt         # Python dependencies
  README.md                # This documentation
  run_pso.sh               # Batch script for running PSO
  run_multiple_pso.sh      # Batch script for multiple runs
  run_multiple_pso_slurm.sh# SLURM batch script for HPC
  space.txt                # Default search space specification
  space_massSF.txt         # Alternative search space file
```

---

## Usage

### Basic Workflow

Run the main optimization script from the repo root:

```bash
python main.py --sage-binary /path/to/sage --config /path/to/input.par [other options]
```

#### Workflow Steps

1. **CSV Data Check:** At startup, the workflow checks for required `sage_*.csv` files in the `data/` folder.
2. **Automatic CSV Generation:** If any are missing, the workflow:
   - Runs SAGE using the provided binary and `.par` file
   - Parses the output directory from the `.par` file (`OutputDir`)
   - Locates the output `.hdf5` file
   - Extracts the required properties and recreates the CSV files in the correct format
3. **PSO Execution:** The PSO algorithm runs using the available data and configuration, writing results and diagnostics to the output directory.
4. **Diagnostics:** After PSO, diagnostic plots and summary statistics are generated.

---

### Command-Line Arguments

#### Required Arguments

- `--sage-binary` (`-b`): Path to the SAGE binary to use
- `--config` (`-c`): Path to the SAGE input `.par` file

#### Common Options

- `--outdir` (`-o`): Output directory (default: current directory)
- `--space-file` (`-S`): Search space specification file (default: `space.txt`)
- `--constraints` (`-x`): Constraints to use (default: `BHMF,SMF_z0,BHBM`)
- `--csv-output` (`-csv`): Path to save PSO results as CSV
- `--swarm-size` (`-s`): Size of the particle swarm
- `--max-iterations` (`-m`): Maximum PSO iterations
- `--stat-test` (`-t`): Statistical test for evaluation (`student-t`, `chi2`)
- `--snapshot` (`-sn`): Comma-separated list of snapshot numbers
- `--age-alist-file`: Path to age list file (if required)
- `--random-seed` (`-r`): Random seed for reproducibility

#### PSO Hyperparameters

- `--omega`: PSO inertia weight (default: 0.729)
- `--phip`: PSO cognitive parameter (default: 1.49445)
- `--phig`: PSO social parameter (default: 1.49445)

#### HPC Options

- `--hpc-mode` (`-H`): Enable HPC mode
- `--cpus` (`-C`): Number of CPUs per SAGE instance
- `--memory` (`-M`): Memory per SAGE instance
- `--nodes` (`-N`): Number of nodes to use
- `--account` (`-a`): SLURM account
- `--queue` (`-q`): SLURM queue
- `--walltime` (`-w`): Walltime per submission
- `--username` (`-u`): Username for SLURM

#### Example Command

```bash
python main.py --sage-binary ./sage --config ./input/millennium.par --outdir ./output --space-file space.txt --constraints BHMF,SMF_z0,BHBM --csv-output results.csv --swarm-size 20 --max-iterations 30
```

---

### Batch and HPC Usage

Automate runs using the provided batch scripts:

- `run_pso.sh`: Run the PSO workflow with your chosen configuration
- `run_multiple_pso.sh`: Run multiple PSO jobs in sequence
- `run_multiple_pso_slurm.sh`: Submit multiple jobs to SLURM for HPC environments

All scripts will:
- Run the PSO workflow with your configuration
- Automatically regenerate missing CSV files from SAGE output
- Use the output directory specified in your `.par` file

---

## Data Files and Formats

All required data files are stored in the `data/` folder. The workflow will automatically generate missing files if possible.

**Required property names in the .hdf5 file:**
- For `sage_bhmf_all_redshifts.csv`: `LogMass`, `BHMF`
- For `sage_smf_all_redshifts.csv`: `LogMass`, `SMF`
- For `sage_smf_extra_redshifts.csv`: `LogMass`, `SMF_extra`
- For `sage_bhbm_all_redshifts.csv`: `LogMass`, `LogBHM`, `Error`, `Count`
- For `sage_halostellar_all_redshifts.csv`: `LogHaloMass`, `LogStellarMass`, `Error`, `Count`

**File Format:**
- All files are tab-separated, with no header.
- Each row represents a snapshot or bin, with properties grouped in sets (usually 2 or 4).
- NaN values are present where data is missing.

---

## Constraints System

Constraints are defined in `constraints.py` and parsed from the command line. Supported constraints include:

- `SMF_z0`, `SMF_z05`, `SMF_z10`, `SMF_z20`, `SMF_z30`, `SMF_z40`: Stellar Mass Function at various redshifts
- `BHMF_z0`, `BHMF_z10`: Black Hole Mass Function
- `BHBM`: Black Hole - Bulge Mass relation
- `CSFRDH`: Cosmic Star Formation Rate Density History
- `HIMF`: HI Mass Function

**Constraint Syntax:**
- Specify domain: `SMF_z0(8-11)`
- Specify weight: `BHMF*6,SMF_z0(8-11)*10`

---

## Diagnostics and Output

After PSO, the package generates:

- Diagnostic plots (parameter evolution, likelihood, swarm movement)
- Pairplots and KDE plots for parameter distributions
- Constraint grid plots and summary statistics
- GIF animations of swarm evolution (if possible)
- All outputs are saved to the specified output directory

---

## Testing and Validation

- `test_constraint_data.py`: Quick test to verify constraint data loading
- `test_pso_benchmarks.py`: PSO benchmark tests with standard functions
- `quick_pso_test.py`: Minimal PSO sanity check

Run tests with:

```bash
python test_constraint_data.py
python test_pso_benchmarks.py --test all
python quick_pso_test.py
```

---

## Extending and Customization

- Add new constraints or diagnostics by editing the relevant Python modules
- To add new automatic CSV generation, update the extraction logic in `main.py`
- Modular codebase allows for easy integration with other analysis tools

---

## Troubleshooting

- **Missing CSV files:** The workflow will automatically regenerate them if possible. Ensure your `.hdf5` output contains the required properties.
- **SAGE binary not found:** Provide the correct path using `--sage-binary`.
- **Input file errors:** Ensure your `.par` file is valid and contains the correct `OutputDir`.
- **Property extraction errors:** Check that your `.hdf5` file contains the expected datasets (see above).
- **HPC issues:** Check SLURM configuration and resource availability.

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open a GitHub issue or pull request.

---

## License

MIT

---

## Contact

For issues, questions, or collaboration, open a GitHub issue or pull request, or contact the maintainer directly.

---

Let me know if you want further customization or additional details added!
