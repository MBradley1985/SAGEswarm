#!/bin/bash

import argparse
import logging
import math
import multiprocessing
import os
import sys
import time

def _abspath(p):
    return os.path.normpath(os.path.abspath(p))




import analysis
import common
import constraints
import execution
import pso
import glob
import diagnostics


logger = logging.getLogger('main')

def setup_logging(outdir):
    log_fname = os.path.join(outdir, 'sage_pso.log')
    fmt = '%(asctime)-15s %(name)s#%(funcName)s:%(lineno)s %(message)s'
    fmt = logging.Formatter(fmt)
    fmt.converter = time.gmtime
    
    # Set root logger level
    logging.root.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    logging.root.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_fname)
    file_handler.setFormatter(fmt)
    logging.root.addHandler(file_handler)
    
    # Ensure all loggers inherit these settings
    logging.getLogger('diagnostics').setLevel(logging.INFO)
    logging.getLogger('main').setLevel(logging.INFO)

def get_required_snapshots(constraints_str):
    """Get all unique snapshots needed for constraints"""
    # Map of constraint classes to their snapshots 
    snapshot_map = {
        'SMF_z0': [63],
        'SMF_z05': [48],
        'SMF_z10': [40], 
        'SMF_z20': [32],
        'SMF_z30': [27],
        'SMF_z40': [23],
        'BHMF_z0': [63],
        'BHMF_z10': [40],
        'BHBM': [63],
        'CSFRDH': [23, 27, 32, 36, 40, 44, 48, 52, 56, 60, 63],  # Snapshots spanning cosmic history
        'HIMF': [63]
    }
    
    snapshots = set()
    print(f"Parsing constraints string: {constraints_str}")
    for constraint in constraints_str.split(','):
        # Remove any weight/domain specifications
        base_constraint = constraint.split('(')[0].split('*')[0]
        print(f"Processing constraint: {constraint}")
        print(f"Base constraint: {base_constraint}")
        if base_constraint in snapshot_map:
            print(f"Found snapshot mapping: {snapshot_map[base_constraint]}")
            snapshots.update(snapshot_map[base_constraint])
        else:
            print(f"Warning: No snapshot mapping found for {base_constraint}")
    
    result = sorted(list(snapshots))
    print(f"Final snapshots list: {result}")
    return result

def cleanup_files(opts):
    """Clean up dump files and track files after PSO run"""
    
    # Define patterns for files to delete
    patterns = {
        'smf_dumps': os.path.join(opts.outdir, 'SMF_z*_dump.txt'),
        'bhmf_dump': os.path.join(opts.outdir, 'BHMF_z*_dump.txt'),
        'bhbm_dump': os.path.join(opts.outdir, 'BHBM_dump.txt'),
        'hsmr_dump': os.path.join(opts.outdir, 'HSMR_z*_dump.txt'),
        'CSFRDH_dump': os.path.join(opts.outdir, 'CSFRDH_dump.txt')
    }

    # Delete dump files
    for pattern_name, pattern in patterns.items():
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                os.remove(file_path)
                print(f"Deleted {os.path.basename(file_path)}")
            except OSError as e:
                print(f"Error deleting {os.path.basename(file_path)}: {e}")
    """
    # Clean up tracks folder
    tracks_folder = os.path.join(opts.outdir, 'tracks')
    if os.path.exists(tracks_folder):
        for ext in ['*.npy', '*.par']:
            for file in glob.glob(os.path.join(tracks_folder, ext)):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error deleting {os.path.basename(file)}: {e}")
    """
    # Clean up tracks folder
    tracks_folder = os.path.join(opts.outdir, 'tracks')
    par_folder = opts.outdir
    
    # Define extensions to clean up
    extensions = ['.npy', '.par']
    """
    # Clean up tracks folder
    if os.path.exists(tracks_folder) and os.path.isdir(tracks_folder):
        for ext in extensions:
            for file in glob.glob(os.path.join(tracks_folder, f'*{ext}')):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error deleting {os.path.basename(file)}: {e}")
    """
    # Clean up par folder
    if os.path.exists(par_folder) and os.path.isdir(par_folder):
        for ext in extensions:
            for file in glob.glob(os.path.join(par_folder, f'*{ext}')):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error deleting {os.path.basename(file)}: {e}")

def main():

    # Argument parsing moved to top to ensure opts is always assigned
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Configuration (.par) file for SAGE input', type=_abspath)
    parser.add_argument('-v', '--subvolumes', help='Comma- and dash-separated list of subvolumes to process', default='0')
    parser.add_argument('-b', '--sage-binary', required=True, help='Path to the SAGE binary to use', type=_abspath)
    parser.add_argument('-o', '--outdir', help='Auxiliary output directory, defaults to .', default=_abspath('.'),
                        type=_abspath)
    parser.add_argument('-k', '--keep', help='Keep temporary output files', action='store_true')
    parser.add_argument('-sn', '--snapshot', help='Comma-separated list of snapshot numbers to analyze', 
                   type=lambda x: [int(i) for i in x.split(',')], default=None)
    parser.add_argument('--sim', help='Simulation to use (0=miniUchuu, 1=miniMillennium, 2=MTNG)', 
                   type=int, default=0)
    parser.add_argument('--boxsize', help='Size of the simulation box in Mpc/h', 
                    type=float, default=400.0)
    parser.add_argument('--vol-frac', help='Volume fraction of the simulation box', 
                    type=float, default=0.0019)
    parser.add_argument('--age-alist-file', help='Path to the age list file, match with .par file',
                   default=None, type=_abspath)
    parser.add_argument('--Omega0', help='Omega0 value for the simulation', 
                    type=float, default=0.3089)
    parser.add_argument('--h0', help='H0 value for the simulation', 
                    type=float, default=0.677400)

    pso_opts = parser.add_argument_group('PSO options')
    pso_opts.add_argument('-s', '--swarm-size', help='Size of the particle swarm. Defaults to 10 + sqrt(D) * 2 (D=number of dimensions)',
                          type=int, default=None)
    pso_opts.add_argument('-m', '--max-iterations', help='Maximum number of iterations to reach before giving up, defaults to 20',
                          default=10, type=int)
    pso_opts.add_argument('-S', '--space-file', help='File with the search space specification, defaults to space.txt',
                          default='space.txt', type=_abspath)
    pso_opts.add_argument('-t', '--stat-test', help='Stat function used to calculate the value of a particle, defaults to student-t',
                          default='student-t', choices=list(analysis.stat_tests.keys()))
    pso_opts.add_argument('-x', '--constraints', default='BHMF,SMF_z0,BHBM',
                          help=("Comma-separated list of constraints, any of BHMF, SMF_z0 or BHBM, defaults to 'BHMF,SMF_z0,BHBM'. "
                                "Can specify a domain range after the name (e.g., 'SMF_z0(8-11)')"
                                "and/or a relative weight (e.g. 'BHMF*6,SMF_z0(8-11)*10)'") )
    pso_opts.add_argument('-csv', '--csv-output', help='Path to save PSO results as CSV file. If not specified, no CSV will be generated.',
                      type=_abspath, default=None)
    pso_opts.add_argument('-r', '--random-seed', help='Random seed for reproducibility. If not specified, PSO will use random initialization.',
                      type=int, default=None)
    pso_opts.add_argument('--omega', help='PSO inertia weight (default: 0.729). Standard constriction coefficient from Clerc & Kennedy (2002).',
                      type=float, default=0.729)
    pso_opts.add_argument('--phip', help='PSO cognitive parameter (default: 1.49445). Particle learning from own best. Standard value ~1.5-2.0.',
                      type=float, default=1.49445)
    pso_opts.add_argument('--phig', help='PSO social parameter (default: 1.49445). Particle learning from swarm best. Standard value ~1.5-2.0.',
                      type=float, default=1.49445)

    hpc_opts = parser.add_argument_group('HPC options')
    hpc_opts.add_argument('-H', '--hpc-mode', help='Enable HPC mode', action='store_true')
    hpc_opts.add_argument('-C', '--cpus', help='Number of CPUs per sage instance', default=1, type=int)
    hpc_opts.add_argument('-M', '--memory', help='Memory needed by each sage instance', default='1500m')
    hpc_opts.add_argument('-N', '--nodes', help='Number of nodes to use', default=None, type=int)
    hpc_opts.add_argument('-a', '--account', help='Submit jobs using this account', default=None)
    hpc_opts.add_argument('-q', '--queue', help='Submit jobs to this queue', default=None)
    hpc_opts.add_argument('-w', '--walltime', help='Walltime for each submission, defaults to 1:00:00', default='1:00:00')
    hpc_opts.add_argument('-u', '--username', help='Username for SLURM job submission', default=None)

    opts = parser.parse_args()

    if not opts.config:
        parser.error('-c option is mandatory but missing')

    # ... [Previous imports and setup] ...

    print("\nStarting SAGE-PSO Main Program\n")
    print("="*60)

    print(f"Checking for required SAGE output CSV files in {os.path.join(opts.outdir, 'data')}...")

    # --- Check for required sage_*.csv files ---
    required_csvs = [
        'sage_bhbm_all_redshifts.csv',
        'sage_bhmf_all_redshifts.csv',
        'sage_halostellar_all_redshifts.csv',
        'sage_smf_all_redshifts.csv',
        'sage_smf_extra_redshifts.csv'
    ]
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    missing_csvs = [f for f in required_csvs if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_csvs:
        print(f"Missing CSV files: {missing_csvs}\nRunning SAGE to generate them...")
        
        # 1. Parse OutputDir from .par file
        output_dir = None
        try:
            with open(opts.config, 'r') as parfile:
                for line in parfile:
                    if line.strip().startswith('OutputDir'):
                        output_dir = line.split()[1].strip()
                        break
        except Exception as e:
            print(f"Error reading config file: {e}")
            sys.exit(1)

        # 2. Force Run SAGE
        import subprocess
        import h5py
        import numpy as np
        from scipy import stats

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"Executing SAGE: {opts.sage_binary} {opts.config}")
        sage_cmd = [opts.sage_binary, opts.config]
        subprocess.run(sage_cmd, cwd=opts.outdir, check=True)

        # 3. Find HDF5 files
        if output_dir and os.path.exists(output_dir):
            hdf5_files = [os.path.join(output_dir, fname) for fname in os.listdir(output_dir)
                          if fname.startswith('model_') and fname.endswith('.hdf5')]
        else:
            hdf5_files = []
        
        if not hdf5_files:
            raise FileNotFoundError("No model_*.hdf5 files found after SAGE run.")

        # 4. Read Simulation Parameters & Detect Structure
        print("Reading simulation parameters...")
        with h5py.File(hdf5_files[0], 'r') as f:
            # Detect Header
            if 'Header' in f: header = f['Header'].attrs
            elif 'Core_0' in f and 'Header' in f['Core_0']: header = f['Core_0']['Header'].attrs
            else: header = {}

            h = header.get('HubbleParam', opts.h0)
            box = header.get('BoxSize', opts.boxsize)
            n_files_total = header.get('NumFilesPerSnapshot', 1)
            n_files_processed = len(hdf5_files)
            vol_frac = n_files_processed / n_files_total
            
            # Calculate Volume (Mpc^3)
            volume = (box / h)**3 * vol_frac
            print(f"  Volume: {volume:.2e} Mpc^3 (h={h}, Box={box}, Frac={vol_frac:.3f})")

            # Detect Structure Type
            top_keys = list(f.keys())
            if any(k.startswith('Core_') for k in top_keys):
                structure_type = 'core_level'
            elif any(k.startswith('Snap_') for k in top_keys):
                structure_type = 'snap_level'
            else:
                raise ValueError("Unknown HDF5 structure")

        # 5. Define Target Snapshots (z=0, 0.5, 1.0, 2.0, 3.0, 4.0)
        # Corresponds to Millennium snapshots: 63, 48, 40, 32, 27, 23
        target_snapshots = [63, 48, 40, 32, 27, 23]
        print(f"Target Snapshots: {target_snapshots}")

        # Initialize Data Containers
        smf_data_columns = [] 
        bhmf_data_columns = []
        bhbm_data_columns = []
        halostellar_data_columns = []

        # Define Bins
        binwidth = 0.1
        smf_bins = np.arange(6.0, 13.0, binwidth)
        bhmf_bins = np.arange(5.0, 11.0, binwidth)
        bhbm_bins = np.arange(8.0, 12.5, 0.2)
        hs_bins = np.arange(9.0, 15.0, 0.2)

        # 6. Process Specific Snapshots
        for snap_num in target_snapshots:
            snap_key = f"Snap_{snap_num}"
            print(f"  Processing {snap_key}...")
            
            g_stellar = []
            g_bhole = []
            g_bulge = []
            g_mvir = []
            
            for file_path in hdf5_files:
                with h5py.File(file_path, 'r') as f:
                    def get_props(loc):
                        s = np.array(loc['StellarMass']) * 1.0e10 / h if 'StellarMass' in loc else []
                        bh = np.array(loc['BlackHoleMass']) * 1.0e10 / h if 'BlackHoleMass' in loc else []
                        b = np.array(loc['BulgeMass']) * 1.0e10 / h if 'BulgeMass' in loc else []
                        m = np.array(loc['CentralMvir']) * 1.0e10 / h if 'CentralMvir' in loc else \
                            (np.array(loc['Mvir']) * 1.0e10 / h if 'Mvir' in loc else [])
                        return s, bh, b, m

                    if structure_type == 'core_level':
                        for core in f.keys():
                            if snap_key in f[core]:
                                s, bh, b, m = get_props(f[core][snap_key])
                                g_stellar.extend(s)
                                g_bhole.extend(bh)
                                g_bulge.extend(b)
                                g_mvir.extend(m)
                    else:
                        if snap_key in f:
                            s, bh, b, m = get_props(f[snap_key])
                            g_stellar.extend(s)
                            g_bhole.extend(bh)
                            g_bulge.extend(b)
                            g_mvir.extend(m)

            g_stellar = np.array(g_stellar)
            g_bhole = np.array(g_bhole)
            g_bulge = np.array(g_bulge)
            g_mvir = np.array(g_mvir)

            # --- SMF ---
            valid = g_stellar > 0
            if np.sum(valid) > 0:
                hist, edges = np.histogram(np.log10(g_stellar[valid]), bins=smf_bins)
                phi = hist / (volume * binwidth)
                centers = edges[:-1] + binwidth / 2
            else:
                centers = smf_bins[:-1] + binwidth / 2
                phi = np.zeros_like(centers)
            smf_data_columns.extend([centers, phi])

            # --- BHMF ---
            valid = g_bhole > 0
            if np.sum(valid) > 0:
                hist, edges = np.histogram(np.log10(g_bhole[valid]), bins=bhmf_bins)
                phi = hist / (volume * binwidth)
                centers = edges[:-1] + binwidth / 2
            else:
                centers = bhmf_bins[:-1] + binwidth / 2
                phi = np.zeros_like(centers)
            bhmf_data_columns.extend([centers, phi])

            # --- BHBM ---
            valid = (g_bulge > 0) & (g_bhole > 0)
            if np.sum(valid) > 0:
                x, y = np.log10(g_bulge[valid]), np.log10(g_bhole[valid])
                mean_y, _, _ = stats.binned_statistic(x, y, 'mean', bins=bhbm_bins)
                std_y, _, _ = stats.binned_statistic(x, y, 'std', bins=bhbm_bins)
                count_y, edges, _ = stats.binned_statistic(x, y, 'count', bins=bhbm_bins)
                centers = edges[:-1] + (edges[1] - edges[0])/2
            else:
                centers = bhbm_bins[:-1] + (bhbm_bins[1]-bhbm_bins[0])/2
                mean_y = std_y = count_y = np.zeros_like(centers)
            bhbm_data_columns.extend([centers, np.nan_to_num(mean_y), np.nan_to_num(std_y), count_y])

            # --- Halo-Stellar ---
            valid = (g_mvir > 0) & (g_stellar > 0)
            if np.sum(valid) > 0:
                x, y = np.log10(g_mvir[valid]), np.log10(g_stellar[valid])
                mean_y, _, _ = stats.binned_statistic(x, y, 'mean', bins=hs_bins)
                std_y, _, _ = stats.binned_statistic(x, y, 'std', bins=hs_bins)
                count_y, edges, _ = stats.binned_statistic(x, y, 'count', bins=hs_bins)
                centers = edges[:-1] + (edges[1] - edges[0])/2
            else:
                centers = hs_bins[:-1] + (hs_bins[1]-hs_bins[0])/2
                mean_y = std_y = count_y = np.zeros_like(centers)
            halostellar_data_columns.extend([centers, np.nan_to_num(mean_y), np.nan_to_num(std_y), count_y])

        # --- Write Files ---
        def write_wide_csv(filename, columns):
            if not columns: return
            path = os.path.join(data_dir, filename)
            data_matrix = np.column_stack(columns)
            np.savetxt(path, data_matrix, delimiter='\t', fmt='%.6e')
            print(f"Generated {path}")

        if 'sage_smf_all_redshifts.csv' in missing_csvs:
            write_wide_csv('sage_smf_all_redshifts.csv', smf_data_columns)
        if 'sage_smf_extra_redshifts.csv' in missing_csvs:
            write_wide_csv('sage_smf_extra_redshifts.csv', smf_data_columns)
        if 'sage_bhmf_all_redshifts.csv' in missing_csvs:
            write_wide_csv('sage_bhmf_all_redshifts.csv', bhmf_data_columns)
        if 'sage_bhbm_all_redshifts.csv' in missing_csvs:
            write_wide_csv('sage_bhbm_all_redshifts.csv', bhbm_data_columns)
        if 'sage_halostellar_all_redshifts.csv' in missing_csvs:
            write_wide_csv('sage_halostellar_all_redshifts.csv', halostellar_data_columns)


### HEAVILY modified to be SAGE specific
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Configuration (.par) file for SAGE input', type=_abspath)
    parser.add_argument('-v', '--subvolumes', help='Comma- and dash-separated list of subvolumes to process', default='0')
    parser.add_argument('-b', '--sage-binary', required=True, help='Path to the SAGE binary to use', type=_abspath)
    parser.add_argument('-o', '--outdir', help='Auxiliary output directory, defaults to .', default=_abspath('.'),
                        type=_abspath)
    parser.add_argument('-k', '--keep', help='Keep temporary output files', action='store_true')
    parser.add_argument('-sn', '--snapshot', help='Comma-separated list of snapshot numbers to analyze', 
                   type=lambda x: [int(i) for i in x.split(',')], default=None)
    parser.add_argument('--sim', help='Simulation to use (0=miniUchuu, 1=miniMillennium, 2=MTNG)', 
                   type=int, default=0)
    parser.add_argument('--boxsize', help='Size of the simulation box in Mpc/h', 
                    type=float, default=400.0)
    parser.add_argument('--vol-frac', help='Volume fraction of the simulation box', 
                    type=float, default=0.0019)
    parser.add_argument('--age-alist-file', help='Path to the age list file, match with .par file',
                   default=None, type=_abspath)
    parser.add_argument('--Omega0', help='Omega0 value for the simulation', 
                    type=float, default=0.3089)
    parser.add_argument('--h0', help='H0 value for the simulation', 
                    type=float, default=0.677400)
###

    pso_opts = parser.add_argument_group('PSO options')
    pso_opts.add_argument('-s', '--swarm-size', help='Size of the particle swarm. Defaults to 10 + sqrt(D) * 2 (D=number of dimensions)',
                          type=int, default=None)
    pso_opts.add_argument('-m', '--max-iterations', help='Maximum number of iterations to reach before giving up, defaults to 20',
                          default=10, type=int)
    pso_opts.add_argument('-S', '--space-file', help='File with the search space specification, defaults to space.txt',
                          default='space.txt', type=_abspath)
    pso_opts.add_argument('-t', '--stat-test', help='Stat function used to calculate the value of a particle, defaults to student-t',
                          default='student-t', choices=list(analysis.stat_tests.keys()))
    pso_opts.add_argument('-x', '--constraints', default='BHMF,SMF_z0,BHBM',
                          help=("Comma-separated list of constraints, any of BHMF, SMF_z0 or BHBM, defaults to 'BHMF,SMF_z0,BHBM'. "
                                "Can specify a domain range after the name (e.g., 'SMF_z0(8-11)')"
                                "and/or a relative weight (e.g. 'BHMF*6,SMF_z0(8-11)*10)'"))
    pso_opts.add_argument('-csv', '--csv-output', help='Path to save PSO results as CSV file. If not specified, no CSV will be generated.',
                      type=_abspath, default=None)
    pso_opts.add_argument('-r', '--random-seed', help='Random seed for reproducibility. If not specified, PSO will use random initialization.',
                      type=int, default=None)
    pso_opts.add_argument('--omega', help='PSO inertia weight (default: 0.729). Standard constriction coefficient from Clerc & Kennedy (2002).',
                      type=float, default=0.729)
    pso_opts.add_argument('--phip', help='PSO cognitive parameter (default: 1.49445). Particle learning from own best. Standard value ~1.5-2.0.',
                      type=float, default=1.49445)
    pso_opts.add_argument('--phig', help='PSO social parameter (default: 1.49445). Particle learning from swarm best. Standard value ~1.5-2.0.',
                      type=float, default=1.49445)

### 
    hpc_opts = parser.add_argument_group('HPC options')
    hpc_opts.add_argument('-H', '--hpc-mode', help='Enable HPC mode', action='store_true')
    hpc_opts.add_argument('-C', '--cpus', help='Number of CPUs per sage instance', default=1, type=int)
    hpc_opts.add_argument('-M', '--memory', help='Memory needed by each sage instance', default='1500m')
    hpc_opts.add_argument('-N', '--nodes', help='Number of nodes to use', default=None, type=int)
    hpc_opts.add_argument('-a', '--account', help='Submit jobs using this account', default=None)
    hpc_opts.add_argument('-q', '--queue', help='Submit jobs to this queue', default=None)
    hpc_opts.add_argument('-w', '--walltime', help='Walltime for each submission, defaults to 1:00:00', default='1:00:00')
    hpc_opts.add_argument('-u', '--username', help='Username for SLURM job submission', default=None)
###

    opts = parser.parse_args()

    if not opts.config:
        parser.error('-c option is mandatory but missing')

    if opts.snapshot:
        snapshots = opts.snapshot
    else:
        snapshots = get_required_snapshots(opts.constraints)
        opts.snapshot = snapshots

    # Create the output directory if it doesn't exist
    os.makedirs(opts.outdir, exist_ok=True)

    if not opts.sage_binary or not common.has_program(opts.sage_binary):
        parser.error("SAGE binary '%s' not found, specify a correct one via --sage-binary" % opts.sage_binary)

#    _, _, _, redshift_file = common.read_configuration(opts.config)
#    redshift_table = common._redshift_table(redshift_file)
    subvols = common.parse_subvolumes(opts.subvolumes)

    setup_logging(opts.outdir)

    opts.constraints = constraints.parse(opts.constraints, snapshot=opts.snapshot,
                                    boxsize=opts.boxsize,
                                    sim=opts.sim,
                                    vol_frac=opts.vol_frac,
                                    age_alist_file=opts.age_alist_file,
                                    Omega0=opts.Omega0, h0=opts.h0,
                                    output_dir=opts.outdir)
#    for c in opts.constraints:
#        c.redshift_table = redshift_table

    # Read search space specification, which is a comma-separated multiline file,
    # each line containing the following elements:
    #
    # param_name, plot_label, is_log, lower_bound, upper_bound
    space = analysis.load_space(opts.space_file)

    ss = opts.swarm_size
    if ss is None:
        ss = 10 + int(2 * math.sqrt(len(space)))

    args = (opts, space, subvols, analysis.stat_tests[opts.stat_test])

    if opts.hpc_mode:
        procs = 0
        f = execution.run_sage_hpc
#    else:
#        n_cpus = multiprocessing.cpu_count()
#        procs = min(n_cpus, ss)
#        f = execution.run_shark
    else:
        n_cpus = multiprocessing.cpu_count()
        print('seeing', n_cpus, 'CPUs')
        procs = min(n_cpus, ss)        
        f = execution.run_sage



    logger.info('-----------------------------------------------------')
    logger.info('Runtime information')
    logger.info('    SAGE binary: %s', opts.sage_binary)
    logger.info('    Base configuration file: %s', opts.config)
    logger.info('    Subvolumes to use: %r', subvols)
    logger.info('    Output directory: %s', opts.outdir)
    logger.info('    Simulation Type: %d (0=miniUchuu, 1=miniMillennium, 2=MTNG)', opts.sim)
    logger.info('    Box Size: %.1f', opts.boxsize)
    logger.info('    Volume Fraction: %.4f', opts.vol_frac)
    logger.info('    Age List File: %s', opts.age_alist_file if opts.age_alist_file else 'Using default')
    logger.info('    Omega0: %.4f', opts.Omega0)
    logger.info('    h0: %.4f', opts.h0)
    logger.info('    Keep temporary output files: %d', opts.keep)
    logger.info('    Snapshot Number: %s', opts.snapshot)
    logger.info("PSO information:")
    logger.info('    Search space parameters: %s', ' '.join(space['name']))
    logger.info('    Swarm size: %d', ss)
    logger.info('    Maximum iterations: %d', opts.max_iterations)
    logger.info('    PSO Hyperparameters:')
    logger.info('        omega (inertia): %.3f', opts.omega)
    logger.info('        phip (cognitive): %.3f', opts.phip)
    logger.info('        phig (social): %.3f', opts.phig)
    logger.info('    Lower bounds: %r', space['lb'])
    logger.info('    Upper bounds: %r', space['ub'])
    logger.info('    Test function: %s', opts.stat_test)
    logger.info('Constraints:')
    for c in opts.constraints:
        logger.info('    %s', c)
    logger.info('    CSV Output Path: %s', opts.csv_output if opts.csv_output else 'Not specified')
    logger.info('    Random Seed: %s', opts.random_seed if opts.random_seed is not None else 'Not specified (random initialization)')
    logger.info('HPC mode: %d', opts.hpc_mode)
    if opts.hpc_mode:
        logger.info('    Account used to submit: %s', opts.account if opts.account else '')
        logger.info('    Queue to submit: %s', opts.queue if opts.queue else '')
        logger.info('    Walltime per submission: %s', opts.walltime)
        logger.info('    CPUs per instance: %d', opts.cpus)
        logger.info('    Memory per instance: %s', opts.memory)
        logger.info('    Nodes to use: %s', opts.nodes)
        logger.info('    Username to use: %s', opts.username if opts.username else '')
    """
    raw_input = input
    while True:
        answer = raw_input('\nAre these parameters correct? (Yes/no): ')
        if answer:
            if answer.lower() in ('n', 'no'):
                logger.info('Not starting PSO, check your configuration and try again')
                return
            print("Please answer 'yes' or 'no'")
            continue
        break
    """
    # Directory where we store the intermediate results
    tracksdir = os.path.join(opts.outdir, 'tracks')
    try:
        os.makedirs(tracksdir)
    except OSError:
        pass

    # Go, go, go!
    logger.info('Starting PSO now')
    tStart = time.time()
    # No changing directories outside repo; all output is local
    xopt, fopt = pso.pso(f, space['lb'], space['ub'], args=args, swarmsize=ss,
                         maxiter=opts.max_iterations, processes=procs,
                         omega=opts.omega, phip=opts.phip, phig=opts.phig,
                         dumpfile_prefix=os.path.join(tracksdir, 'track_%03d'),
                         csv_output_path=opts.csv_output,
                         random_seed=opts.random_seed)
    tEnd = time.time()

    global count
    #logger.info('Number of iterations = %d', count)
    logger.info('xopt = %r', xopt)
    logger.info('fopt = %r', fopt)
    logger.info('PSO finished in %.3f [s]', tEnd - tStart)
    logger.info('Checking for SMF, BHBM and BHMF dump files...')
    dump_files = glob.glob(os.path.join(opts.outdir, 'SMF_z*_dump.txt'))
    logger.info('Found SMF dump files: %s', dump_files)
    dump_files2 = glob.glob(os.path.join(opts.outdir, 'BHMF_z*_dump.txt'))
    logger.info('Found BHMF dump files: %s', dump_files2)
    dump_files3 = glob.glob(os.path.join(opts.outdir, 'BHBM_z*_dump.txt'))
    logger.info('Found BHBM dump files: %s', dump_files3)

    logger.info('Running diagnostics...')
    diagnostics.main(
        tracks_dir=os.path.join(opts.outdir, 'tracks'),
        space_file=opts.space_file, 
        output_dir=opts.outdir,
        config_opts=opts
    )

    # Clean up all files
    cleanup_files(opts)
    
if __name__ == '__main__':
    main()

