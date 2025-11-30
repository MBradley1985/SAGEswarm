
# sage-optim

A fully self-contained Python package for PSO-based analysis and optimization in galaxy evolution modeling using SAGE.

## Features
- Particle Swarm Optimization (PSO) for parameter fitting
- Constraint-based analysis (SMF, BHMF, BHBM, etc.)
- Diagnostics and output management
- Designed to work with the SAGE binary and input `.par` files
- All data files accessed via the local `data/` folder
- No hardcoded paths or dependencies outside the repo

## Requirements
- Python 3.8+
- SAGE binary (must be provided by user)
- Input `.par` file (must be provided by user)
- See `requirements.txt` for Python dependencies

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sage-optim.git
cd sage-optim
pip install -r requirements.txt
```


## Usage
Run the main optimization script from the repo root:

```bash
python main.py --sage-binary /path/to/sage --config /path/to/input.par [other options]
```

### Automatic CSV Generation
If any required `sage_*.csv` files are missing from the `data/` folder, the workflow will:
1. Run SAGE using the provided binary and `.par` file.
2. Parse the output directory from the `.par` file (`OutputDir`).
3. Locate the output `.hdf5` file.
4. Extract the required properties and recreate the CSV files in the correct format.

**Required property names in the .hdf5 file:**
- For `sage_bhmf_all_redshifts.csv`: `LogMass`, `BHMF`
- For `sage_smf_all_redshifts.csv`: `LogMass`, `SMF`
- For `sage_smf_extra_redshifts.csv`: `LogMass`, `SMF_extra`
- For `sage_bhbm_all_redshifts.csv`: `LogMass`, `LogBHM`, `Error`, `Count`
- For `sage_halostellar_all_redshifts.csv`: `LogHaloMass`, `LogStellarMass`, `Error`, `Count`

All files are tab-separated, with no header. Each row represents a snapshot or bin, with properties grouped in sets (usually 2 or 4).

### Required Arguments
- `--sage-binary` (`-b`): Path to the SAGE binary to use
- `--config` (`-c`): Path to the SAGE input `.par` file

# sage-optim

## Overview
sage-optim is a fully self-contained Python package for Particle Swarm Optimization (PSO) and analysis in galaxy evolution modeling using the SAGE semi-analytic model. It is designed to be robust, reproducible, and easy to use for both standalone and automated workflows.

## Features
- Particle Swarm Optimization (PSO) for parameter fitting and model calibration
- Constraint-based analysis (e.g., SMF, BHMF, BHBM, CSFRDH)
- Automatic diagnostics and output management
- Self-healing workflow: missing `sage_*.csv` data files are automatically regenerated from SAGE output
- Modular codebase for easy extension and customization
- All data files accessed via the local `data/` folder
- No hardcoded paths or dependencies outside the repository

## Requirements
- Python 3.8 or newer
- SAGE binary (must be provided by the user)
- SAGE input `.par` file (must be provided by the user)
- See `requirements.txt` for Python dependencies (numpy, pandas, matplotlib, scipy, h5py, etc.)

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sage-optim.git
cd sage-optim
pip install -r requirements.txt
```

## Usage
Run the main optimization script from the repo root:

```bash
python main.py --sage-binary /path/to/sage --config /path/to/input.par [other options]
```

### Workflow Details
1. **CSV Data Check:** At startup, the workflow checks for required `sage_*.csv` files in the `data/` folder.
2. **Automatic CSV Generation:** If any are missing, the workflow:
	 - Runs SAGE using the provided binary and `.par` file
	 - Parses the output directory from the `.par` file (`OutputDir`)
	 - Locates the output `.hdf5` file
	 - Extracts the required properties and recreates the CSV files in the correct format
3. **PSO Execution:** The PSO algorithm runs using the available data and configuration, writing results and diagnostics to the output directory.

### Data File Formats
- All `sage_*.csv` files are tab-separated, with no header.
- Each row represents a snapshot or bin, with properties grouped in sets (usually 2 or 4).
- NaN values are present where data is missing.
- Example property groupings:
	- `sage_bhmf_all_redshifts.csv`: `LogMass`, `BHMF`
	- `sage_smf_all_redshifts.csv`: `LogMass`, `SMF`
	- `sage_smf_extra_redshifts.csv`: `LogMass`, `SMF_extra`
	- `sage_bhbm_all_redshifts.csv`: `LogMass`, `LogBHM`, `Error`, `Count`
	- `sage_halostellar_all_redshifts.csv`: `LogHaloMass`, `LogStellarMass`, `Error`, `Count`

### Required Arguments
- `--sage-binary` (`-b`): Path to the SAGE binary to use
- `--config` (`-c`): Path to the SAGE input `.par` file

### Common Options
- `--outdir` (`-o`): Output directory (default: current directory)
- `--space-file` (`-S`): Search space specification file (default: `space.txt`)
- `--constraints` (`-x`): Constraints to use (default: `BHMF,SMF_z0,BHBM`)
- `--csv-output` (`-csv`): Path to save PSO results as CSV
- `--swarm-size` (`-s`): Size of the particle swarm
- `--max-iterations` (`-m`): Maximum PSO iterations
- `--stat-test` (`-t`): Statistical test for evaluation
- `--snapshot` (`-sn`): Comma-separated list of snapshot numbers
- `--age-alist-file`: Path to age list file (if required)

### Example Command
```bash
python main.py --sage-binary ./sage --config ./input/millennium.par --outdir ./output --space-file space.txt --constraints BHMF,SMF_z0,BHBM --csv-output results.csv --swarm-size 20 --max-iterations 30
```

## Batch Usage
You can automate runs using the provided `run_pso.sh` batch script. This script will:
- Run the PSO workflow with your chosen configuration
- Automatically regenerate missing CSV files from SAGE output
- Use the output directory specified in your `.par` file

## Output
- All output files and diagnostics are written to the specified output directory (`--outdir`)
- The output `.hdf5` file is located in the directory specified by the `OutputDir` line in your `.par` file
- Results, logs, and diagnostics are saved for reproducibility

## Troubleshooting
- **Missing CSV files:** The workflow will automatically regenerate them if possible. Ensure your `.hdf5` output contains the required properties.
- **SAGE binary not found:** Provide the correct path using `--sage-binary`.
- **Input file errors:** Ensure your `.par` file is valid and contains the correct `OutputDir`.
- **Property extraction errors:** Check that your `.hdf5` file contains the expected datasets (see above).

## Extending and Customization
- Add new constraints or diagnostics by editing the relevant Python modules
- To add new automatic CSV generation, update the extraction logic in `main.py`
- Modular codebase allows for easy integration with other analysis tools

## Contributing
Contributions, bug reports, and feature requests are welcome! Please open a GitHub issue or pull request.

## License
MIT

## Contact
For issues, questions, or collaboration, open a GitHub issue or pull request, or contact the maintainer directly.
