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
    # --- Check for required sage_*.csv files ---
    required_csvs = [
        'sage_bhbm_all_redshifts.csv',
        'sage_bhmf_all_redshifts.csv',
        'sage_halostellar_all_redshifts.csv',
        'sage_smf_all_redshifts.csv',
        'sage_smf_extra_redshifts.csv'
    ]
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    missing_csvs = [f for f in required_csvs if not os.path.exists(os.path.join(data_dir, f))]
    if missing_csvs:
        print(f"Missing CSV files: {missing_csvs}\nRunning SAGE to generate them...")
        # Parse OutputDir from .par file
        output_dir = None
        with open(opts.config, 'r') as parfile:
            for line in parfile:
                if line.strip().startswith('OutputDir'):
                    output_dir = line.split()[1].strip()
                    break
        if not output_dir or not os.path.isdir(output_dir):
            raise FileNotFoundError(f"OutputDir '{output_dir}' from .par file not found.")
        # Run SAGE using the provided binary and .par file
        import subprocess
        sage_cmd = [opts.sage_binary, opts.config]
        subprocess.run(sage_cmd, cwd=output_dir, check=True)
        # Find the .hdf5 output file in output_dir
        hdf5_file = None
        for fname in os.listdir(output_dir):
            if fname.endswith('.h5') or fname.endswith('.hdf5'):
                hdf5_file = os.path.join(output_dir, fname)
                break
        if not hdf5_file:
            raise FileNotFoundError("No .hdf5 output file found after running SAGE.")
        # Extract required properties and create CSVs
        import h5py as h5
        import numpy as np
        def read_hdf(filename, snap_num, param):
            with h5.File(filename, 'r') as property:
                return np.array(property[snap_num][param])

        # Example: snapshot list and property extraction
        # You may need to adjust snapshot numbers and property names to match your simulation
        snapshots = list(h5.File(hdf5_file, 'r').keys())

        # --- sage_bhmf_all_redshifts.csv ---
        if 'sage_bhmf_all_redshifts.csv' in missing_csvs:
            with open(os.path.join(data_dir, 'sage_bhmf_all_redshifts.csv'), 'w') as f:
                for snap in snapshots:
                    log_mass = read_hdf(hdf5_file, snap, 'LogMass')
                    value = read_hdf(hdf5_file, snap, 'BHMF')
                    for m, v in zip(log_mass, value):
                        f.write(f"{m}\t{v}\t")
                    f.write("\n")

        # --- sage_smf_all_redshifts.csv ---
        if 'sage_smf_all_redshifts.csv' in missing_csvs:
            with open(os.path.join(data_dir, 'sage_smf_all_redshifts.csv'), 'w') as f:
                for snap in snapshots:
                    log_mass = read_hdf(hdf5_file, snap, 'LogMass')
                    value = read_hdf(hdf5_file, snap, 'SMF')
                    for m, v in zip(log_mass, value):
                        f.write(f"{m}\t{v}\t")
                    f.write("\n")

        # --- sage_smf_extra_redshifts.csv ---
        if 'sage_smf_extra_redshifts.csv' in missing_csvs:
            with open(os.path.join(data_dir, 'sage_smf_extra_redshifts.csv'), 'w') as f:
                for snap in snapshots:
                    log_mass = read_hdf(hdf5_file, snap, 'LogMass')
                    value = read_hdf(hdf5_file, snap, 'SMF_extra')
                    for m, v in zip(log_mass, value):
                        f.write(f"{m}\t{v}\t")
                    f.write("\n")

        # --- sage_bhbm_all_redshifts.csv ---
        if 'sage_bhbm_all_redshifts.csv' in missing_csvs:
            with open(os.path.join(data_dir, 'sage_bhbm_all_redshifts.csv'), 'w') as f:
                for snap in snapshots:
                    # Example: extract multiple properties per bin
                    # You may need to adjust property names and grouping
                    props = ['LogMass', 'LogBHM', 'Error', 'Count']
                    arrays = [read_hdf(hdf5_file, snap, p) for p in props]
                    for row in zip(*arrays):
                        f.write("\t".join(str(x) for x in row) + "\t")
                    f.write("\n")

        # --- sage_halostellar_all_redshifts.csv ---
        if 'sage_halostellar_all_redshifts.csv' in missing_csvs:
            with open(os.path.join(data_dir, 'sage_halostellar_all_redshifts.csv'), 'w') as f:
                for snap in snapshots:
                    props = ['LogHaloMass', 'LogStellarMass', 'Error', 'Count']
                    arrays = [read_hdf(hdf5_file, snap, p) for p in props]
                    for row in zip(*arrays):
                        f.write("\t".join(str(x) for x in row) + "\t")
                    f.write("\n")

        print(f"Generated missing CSV files: {missing_csvs}")

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

