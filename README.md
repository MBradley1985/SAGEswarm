# SAGE-PSO: Semi-Analytic Galaxy Evolution Model Particle Swarm Optimization

A Python package for PSO-based parameter optimization in galaxy evolution modeling using the SAGE semi-analytic model.

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
- [Search Space Configuration](#search-space-configuration)
- [Constraints System](#constraints-system)
- [Simulation Support](#simulation-support)
- [Data Files and Formats](#data-files-and-formats)
- [Diagnostics and Output](#diagnostics-and-output)
- [Testing](#testing)
- [License](#license)

---

## Features

- Particle Swarm Optimization (PSO) for SAGE parameter calibration
- Multiple constraint types: SMF, BHMF, BHBM, CSFRDH, HIMF, H2MF, MZR, SHMR, SMD
- Red/blue galaxy stellar mass function discrimination
- Multi-simulation support: miniUchuu, miniMillennium, MTNG
- Automatic CSV data generation from SAGE HDF5 output
- HPC/SLURM integration for parallel execution
- Diagnostic plots, animations, and parameter uncertainty analysis

---

## Requirements

- Python 3.8+
- SAGE binary (user-provided)
- SAGE input `.par` file (user-provided)
- Merger tree age list file (simulation-specific)

**Python dependencies:**
- numpy
- pandas
- matplotlib
- scipy
- h5py

Install with:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourusername/sage-pso.git
cd sage-pso
pip install -r requirements.txt
```

---

## Package Structure

```
SAGE-PSO/
├── main.py                 # Entry point: SAGE execution, CSV generation, PSO orchestration
├── space.txt               # Parameter search space specification
├── requirements.txt        # Python dependencies
├── run_pso.sh              # Single PSO run script
│
├── src/
│   ├── pso.py              # PSO algorithm implementation
│   ├── constraints.py      # Constraint definitions and data loading
│   ├── analysis.py         # Statistical tests (chi-squared, Student's t)
│   ├── execution.py        # SAGE binary execution, SLURM job submission
│   ├── diagnostics.py      # Post-PSO plots and animations
│   ├── simulation_config.py # Simulation parameters and snapshot mappings
│   ├── routines.py         # HDF5 reading and data extraction
│   ├── common.py           # Utility functions
│   ├── redshift_utils.py   # Redshift/snapshot conversion
│   └── pso_uncertainty.py  # Parameter uncertainty analysis
│
├── run_types/
│   ├── run_multiple_pso.sh       # Sequential multiple PSO runs
│   ├── run_multiple_pso_slurm.sh # SLURM parallel PSO runs
│   ├── submit_pso_array.sh       # SLURM array job submission
│   ├── analyze_pso_array.sh      # Analyze array job results
│   ├── analyze_multiple_pso.py   # Multi-run analysis script
│   └── make_comparison_plot.sh   # Generate comparison plots
│
├── tests/
│   ├── test_constraint_data.py   # Constraint data loading tests
│   ├── test_pso_benchmarks.py    # PSO algorithm validation
│   ├── quick_pso_test.py         # Minimal sanity check
│   └── visual_pso_test.py        # Visual convergence test
│
└── data/
    └── (constraint observational data files)
```

---

## Usage

### Basic Workflow

1. **CSV Data Check:** At startup, required `sage_*.csv` files are checked in the output directory
2. **Automatic Generation:** Missing CSVs are regenerated from SAGE HDF5 output
3. **PSO Execution:** The optimizer runs using constraints and configuration
4. **Diagnostics:** Plots and statistics are generated after PSO completes

### Command-Line Arguments

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `-c, --config` | Path to SAGE input `.par` file |
| `-b, --sage-binary` | Path to SAGE binary |

#### Common Options

| Argument | Default | Description |
|----------|---------|-------------|
| `-o, --outdir` | `.` | Output directory |
| `-v, --subvolumes` | `0` | Subvolumes to process |
| `-k, --keep` | off | Keep temporary output files |
| `-sn, --snapshot` | auto | Snapshot numbers to analyze |

#### Simulation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--sim` | `0` | Simulation type: 0=miniUchuu, 1=miniMillennium, 2=MTNG |
| `--boxsize` | sim-specific | Simulation box size in Mpc/h |
| `--vol-frac` | `1.0` | Volume fraction of simulation box |
| `--age-alist-file` | sim-specific | Path to merger tree age list file |
| `--Omega0` | sim-specific | Matter density parameter |
| `--h0` | sim-specific | Hubble parameter (H0/100) |

#### PSO Options

| Argument | Default | Description |
|----------|---------|-------------|
| `-s, --swarm-size` | `10 + 2*sqrt(D)` | Number of particles |
| `-m, --max-iterations` | `20` | Maximum iterations |
| `-S, --space-file` | `space.txt` | Search space specification |
| `-t, --stat-test` | `student-t` | Statistical test (`student-t`, `chi2`) |
| `-x, --constraints` | `BHMF,SMF_z0,BHBM` | Constraints to use |
| `-csv, --csv-output` | none | Save results to CSV |
| `-r, --random-seed` | random | Seed for reproducibility |
| `--omega` | `0.729` | PSO inertia weight |
| `--phip` | `1.49445` | Cognitive parameter |
| `--phig` | `1.49445` | Social parameter |

#### HPC Options

| Argument | Default | Description |
|----------|---------|-------------|
| `-H, --hpc-mode` | off | Enable HPC mode |
| `-C, --cpus` | `1` | CPUs per SAGE instance |
| `-M, --memory` | `1500m` | Memory per instance |
| `-N, --nodes` | auto | Number of nodes |
| `-a, --account` | none | SLURM account |
| `-q, --queue` | none | SLURM queue |
| `-w, --walltime` | `1:00:00` | Walltime per job |
| `-u, --username` | none | SLURM username |

#### Example

```bash
python main.py \
  -b ./sage \
  -c ./input/millennium.par \
  -o ./output \
  --sim 1 \
  -x "SMF_z0(8-11)*5,BHMF_z0,BHBM" \
  -s 20 \
  -m 30 \
  -csv results.csv
```

---

## Search Space Configuration

The `space.txt` file defines the parameter search space:

```
SfrEfficiency,eSFR,1,0.01,0.1
FeedbackReheatingEpsilon,eReheat,0,0.0,6.0
FeedbackEjectionEfficiency,eEject,0,0.1,1.0
ReIncorporationFactor,eReinc,0,0.05,0.3
RadioModeEfficiency,eRadio,1,0.001,1.0
QuasarModeEfficiency,eQuasar,1,0.001,0.5
BlackHoleGrowthRate,eBHgrowth,1,0.0001,0.5
```

**Format:** `ParameterName,Label,IsLog,LowerBound,UpperBound`

- `IsLog`: 1 = logarithmic space, 0 = linear space

---

## Constraints System

### Available Constraints

| Constraint | Description |
|------------|-------------|
| `SMF_z0`, `SMF_z05`, `SMF_z10`, `SMF_z20`, `SMF_z30`, `SMF_z40` | Stellar Mass Function at z=0, 0.5, 1, 2, 3, 4 |
| `SMF_Red_z0`, `SMF_Blue_z0` | Red (quiescent) and blue (star-forming) galaxy SMF at z=0 |
| `BHMF_z0`, `BHMF_z10` | Black Hole Mass Function |
| `BHBM` | Black Hole - Bulge Mass relation |
| `CSFRDH` | Cosmic Star Formation Rate Density History |
| `HIMF` | HI Mass Function |
| `H2MF` | H2 Mass Function |
| `MZR` | Mass-Metallicity Relation |
| `SHMR` | Stellar-Halo Mass Relation |
| `SMD` | Stellar Mass Density history |

### Constraint Syntax

```bash
-x "SMF_z0(8-11)*5,BHMF_z0*10,BHBM"
```

- **Domain restriction:** `SMF_z0(8-11)` limits to log(M/M☉) = 8-11
- **Weighting:** `BHMF_z0*10` applies weight of 10

---

## Simulation Support

### miniUchuu (SIM_ID=0)
- 50 snapshots (0-49), snapshot 49 ≈ z=0
- Box size: 400 Mpc/h
- Cosmology: h=0.6774, Ω₀=0.3089

### miniMillennium (SIM_ID=1)
- 64 snapshots (0-63), snapshot 63 = z=0
- Box size: 62.5 Mpc/h
- Cosmology: h=0.73, Ω₀=0.25

### MTNG (SIM_ID=2)
- 100 snapshots (0-99), snapshot 99 = z=0
- Box size: 500 Mpc/h
- Cosmology: h=0.6774, Ω₀=0.3089

Each simulation has its own snapshot-to-redshift mapping defined in `src/simulation_config.py`.

---

## Data Files and Formats

SAGE output is automatically converted to CSV files:

| File | Contents |
|------|----------|
| `sage_smf_all_redshifts.csv` | Stellar Mass Function |
| `sage_smf_red_all_redshifts.csv` | Red galaxy SMF |
| `sage_smf_blue_all_redshifts.csv` | Blue galaxy SMF |
| `sage_bhmf_all_redshifts.csv` | Black Hole Mass Function |
| `sage_bhbm_all_redshifts.csv` | BHBM relation (median, std, counts) |
| `sage_halostellar_all_redshifts.csv` | Halo-Stellar mass relation |
| `sage_himf_all_redshifts.csv` | HI Mass Function |
| `sage_h2mf_all_redshifts.csv` | H2 Mass Function |
| `sage_mzr_all_redshifts.csv` | Mass-Metallicity Relation |
| `sage_history.csv` | Cosmic history (CSFRDH, SMD) |

Files are tab-separated with no headers.

---

## Diagnostics and Output

After PSO completion:

- `sage_pso.log` - Run log
- `tracks/track_*_pos.npy`, `tracks/track_*_fx.npy` - Particle trajectories
- Parameter evolution plots
- Likelihood curves
- Swarm movement visualizations
- Pairplots and KDE distributions
- Constraint comparison grids
- GIF animations of swarm evolution (optional)

---

## Testing

Run tests from the project root:

```bash
# Constraint data loading
python tests/test_constraint_data.py

# PSO algorithm benchmarks
python tests/test_pso_benchmarks.py --test all

# Quick sanity check
python tests/quick_pso_test.py

# Visual convergence test
python tests/visual_pso_test.py
```

---

## License

MIT
