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

from src import analysis
from src import common
from src import constraints
from src import execution
from src import pso
import glob
from src import diagnostics
from src.simulation_config import (
    get_snapshot_map, get_target_snapshots, get_history_snapshots,
    resolve_target_snapshots, resolve_history_snapshots, identify_simulation,
    get_simulation_config, SIM_MINIUCHUU, SIM_MINIMILLENNIUM, SIM_MTNG,
    read_output_snapshot_redshifts, build_snapshot_map, describe_snapshot_map
)


logger = logging.getLogger('main')

def setup_logging(outdir):
    log_fname = os.path.join(outdir, 'sage_pso.log')

    # Two formats on purpose: the console gets the message only, because a
    # timestamp and source location on every line buries the information you
    # actually came to read.  The log file keeps the full detail for debugging.
    console_fmt = logging.Formatter('%(message)s')
    file_fmt = logging.Formatter(
        '%(asctime)-15s %(name)s#%(funcName)s:%(lineno)s %(message)s')
    file_fmt.converter = time.gmtime

    # Root at DEBUG with the levels set per handler: the console stays at INFO
    # so it reads as before, while the file records the per-constraint score
    # breakdown that execution.py logs at DEBUG.  With the root itself at INFO
    # those records were dropped before either handler saw them, and the
    # full-detail formatter below had nothing to format.
    logging.root.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_fname)
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    
    # Ensure all loggers inherit these settings
    logging.getLogger('diagnostics').setLevel(logging.INFO)
    logging.getLogger('main').setLevel(logging.INFO)

def get_required_snapshots(constraints_str, sim=0, snapshot_map=None):
    """Get all unique snapshots needed for constraints.

    Parameters:
    -----------
    constraints_str : str
        Comma-separated list of constraint names
    sim : int
        Simulation ID (0=miniUchuu, 1=miniMillennium, 2=MTNG)

    Returns:
    --------
    list : Sorted list of unique snapshot numbers needed
    """
    # Prefer the map resolved from the SAGE output; fall back to the
    # hard-coded per-simulation table only if that was unavailable.
    if snapshot_map is None:
        snapshot_map = get_snapshot_map(sim)

    snapshots = set()
    for constraint in constraints_str.split(','):
        base_constraint = constraint.split('(')[0].split('*')[0]
        if base_constraint in snapshot_map:
            snapshots.update(snapshot_map[base_constraint])

    result = sorted(list(snapshots))
    return result

def cleanup_files(opts):
    """Clean up dump files and track files after PSO run"""
    
    # Define patterns for files to delete
    patterns = {
        'smf_dumps': os.path.join(opts.outdir, 'SMF_z*_dump.txt'),
        'bhmf_dump': os.path.join(opts.outdir, 'BHMF_z*_dump.txt'),
        'bhbm_dump': os.path.join(opts.outdir, 'BHBM_dump.txt'),
        'CSFRDH_dump': os.path.join(opts.outdir, 'CSFRDH_dump.txt'),
        'himf_dump': os.path.join(opts.outdir, 'HIMF_dump.txt'),
        'h2mf_dump': os.path.join(opts.outdir, 'H2MF_dump.txt'),
        'mzr_dump': os.path.join(opts.outdir, 'MZR_dump.txt'),
        'shmr_dump': os.path.join(opts.outdir, 'SHMR_dump.txt'),
        'smd_dump': os.path.join(opts.outdir, 'SMD_dump.txt'),
        'fics_dump': os.path.join(opts.outdir, 'FICS_dump.txt'),
        'fics_mvir_dump': os.path.join(opts.outdir, 'FICS_Mvir_dump.txt'),
        'mlf_dump': os.path.join(opts.outdir, 'MLF_dump.txt')
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
    # Inferred from the number of snapshots in the SAGE output.  It survives only
    # as a fallback for output with no Header/snapshot_redshifts table, and as a
    # label; everything that matters is resolved from the output itself.
    parser.add_argument('--sim', help='Snapshot grid (0=Uchuu, 1=Millennium, '
                   '2=MTNG). Inferred from the SAGE output if omitted; only '
                   'needed for output with no snapshot redshift table.',
                   type=int, default=None)
    # These four are read from the SAGE output's own header by default -- SAGE
    # records the box size, cosmology and processed volume fraction it actually
    # ran with, so passing them by hand only creates the opportunity to disagree
    # with the simulation.  Give one explicitly only to override, and it must
    # still match the file or the run is refused.
    parser.add_argument('--boxsize', help='Size of the simulation box in Mpc/h. '
                    'Read from the SAGE output header if omitted.',
                    type=float, default=None)
    parser.add_argument('--vol-frac', help='Fraction of the simulation box processed. '
                    'Read from the SAGE output header if omitted.',
                    type=float, default=None)
    parser.add_argument('--age-alist-file', help='Path to the age list file. Read from '
                   'the .par file\'s FileWithSnapList, or the SAGE output header, '
                   'if omitted.', default=None, type=_abspath)
    parser.add_argument('--Omega0', help='Matter density of the simulation. '
                    'Read from the SAGE output header if omitted.',
                    type=float, default=None)
    parser.add_argument('--h0', help='Hubble parameter of the simulation. '
                    'Read from the SAGE output header if omitted.',
                    type=float, default=None)

    pso_opts = parser.add_argument_group('PSO options')
    pso_opts.add_argument('-s', '--swarm-size', help='Size of the particle swarm. Defaults to 10 + sqrt(D) * 2 (D=number of dimensions)',
                          type=int, default=None)
    pso_opts.add_argument('-m', '--max-iterations', help='Maximum number of iterations before giving up (default 10). '
                               'NOTE: this said "defaults to 20" while the code used 10, '
                               'so runs that did not pass -m explicitly were half as long '
                               'as the documentation implied. 10 iterations x ~15 particles '
                               'is only ~150 evaluations for an 8-D space; use several '
                               'hundred at minimum for a production calibration.',
                          default=10, type=int)
    pso_opts.add_argument('--max-stagnation', help='Stop if no improvement for this many consecutive iterations, defaults to 15',
                          default=15, type=int)
    pso_opts.add_argument('-S', '--space-file', help='File with the search space specification, defaults to space.txt',
                          default='space.txt', type=_abspath)
    pso_opts.add_argument('-t', '--stat-test', help='Stat function used to calculate the value of a particle, defaults to student-t',
                          default='student-t', choices=list(analysis.stat_tests.keys()))
    pso_opts.add_argument('-x', '--constraints', default='BHMF_z0,SMF_z0,BHBM,HIMF',
                          help=("Comma-separated list of constraints. Valid names: BHMF_z0, BHMF_z10, SMF_z0, SMF_z05, SMF_z10, SMF_z20, SMF_z30, SMF_z40, BHBM, HIMF, CSFRDH, H2MF, SMD, MZR, SHMR, FICS, FICS_Mvir, MLF. "
                                "Can specify a domain range after the name (e.g., 'SMF_z0(8-11)') "
                                "and/or a relative weight (e.g. 'BHMF_z0*6,SMF_z0(8-11)*10)')") )
    pso_opts.add_argument('-csv', '--csv-output', help='Path to save PSO results as CSV file. If not specified, no CSV will be generated.',
                      type=_abspath, default=None)
    pso_opts.add_argument('-r', '--random-seed', help='Random seed for reproducibility. If not specified, PSO will use random initialization.',
                      type=int, default=None)
    pso_opts.add_argument('--omega', help='PSO inertia weight (default: 0.5). Lower values damp velocity faster, promoting convergence.',
                      type=float, default=0.5)
    pso_opts.add_argument('--phip', help='PSO cognitive parameter (default: 0.5). Particle learning from own best.',
                      type=float, default=0.5)
    pso_opts.add_argument('--phig', help='PSO social parameter (default: 0.5). Particle learning from swarm best.',
                      type=float, default=0.5)

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

    # Refuse to start if any search parameter is absent from the base .par.
    # execution.py substitutes into lines that already exist and never appends,
    # so such a parameter never reaches SAGE: every particle would then run
    # identical physics, the objective would be flat, and the swarm would report
    # the same best fit for the whole run.  Checked here, before the reference
    # CSVs are rebuilt, so it costs a second rather than an hour.
    _absent = execution.missing_from_config(
        analysis.load_space(opts.space_file), opts.config)
    if _absent:
        parser.error(
            'these search-space parameters are not in %s:\n'
            '    %s\n'
            'PSO would sample them and silently discard every value, so every '
            'particle would run an identical SAGE configuration and the fit '
            'would never improve. Add them to the parameter file (any value -- '
            'PSO overwrites it) or point -c at a file that has them.'
            % (opts.config, '\n    '.join(_absent)))

    # ... [Previous imports and setup] ...

    print("\nStarting SAGE-PSO Main Program\n")
    print("="*60)

    print(f"Generating SAGE reference CSV files in {os.path.join(os.path.dirname(__file__), 'data')}...")

    # --- Always regenerate sage_*.csv files from SAGE output ---
    required_csvs = [
        'sage_bhbm_all_redshifts.csv',
        'sage_bhmf_all_redshifts.csv',
        'sage_halostellar_all_redshifts.csv',
        'sage_smf_all_redshifts.csv',
        'sage_smf_extra_redshifts.csv',
        'sage_himf_all_redshifts.csv',
        'sage_h2mf_all_redshifts.csv',
        'sage_mzr_all_redshifts.csv',
        'sage_history.csv'
    ]
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Always regenerate all CSV files
    print("Regenerating all SAGE reference CSV files...")

    # 1. Parse OutputDir from .par file
    output_dir = None
    try:
        with open(opts.config, 'r') as parfile:
            for line in parfile:
                if line.strip().startswith('OutputDir'):
                    raw_dir = line.split()[1].strip()
                    sage_dir = os.path.dirname(opts.sage_binary)
                    output_dir = os.path.join(sage_dir, raw_dir) if not os.path.isabs(raw_dir) else raw_dir
                    break
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    # 2. Check for existing HDF5 files
    import subprocess
    import h5py
    import numpy as np
    from scipy import stats

    hdf5_files = []
    if output_dir and os.path.exists(output_dir):
        hdf5_files = [os.path.join(output_dir, fname) for fname in os.listdir(output_dir)
                      if fname.startswith('model_') and fname.endswith('.hdf5')]

    # 3. Run SAGE only if HDF5 files are missing
    if hdf5_files:
        print(f"Found {len(hdf5_files)} existing HDF5 files in '{output_dir}'. Skipping SAGE binary run.")
    else:
        print(f"No existing HDF5 files found. Running SAGE to generate them...")

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"Executing SAGE: {opts.sage_binary} {opts.config}")
        sage_cmd = [opts.sage_binary, opts.config]
        subprocess.run(sage_cmd, cwd=os.path.dirname(opts.sage_binary), check=True)

        # Find the newly generated files
        if output_dir and os.path.exists(output_dir):
            hdf5_files = [os.path.join(output_dir, fname) for fname in os.listdir(output_dir)
                          if fname.startswith('model_') and fname.endswith('.hdf5')]

    if not hdf5_files:
        raise FileNotFoundError("No model_*.hdf5 files found (even after attempting SAGE run).")

    # 4. Read Simulation Parameters & Detect Structure
    print("Reading simulation parameters...")
    with h5py.File(hdf5_files[0], 'r') as f:
        # SAGE writes the box size and cosmology it actually ran with into
        # Header/Simulation.  The old lookup read f['Header'].attrs, which is
        # empty, and silently fell back to the command-line values -- so a
        # mismatched --boxsize was never noticed.
        sim_attrs = {}
        for grp in ('Header/Simulation', 'Core_0/Header/Simulation'):
            if grp in f:
                sim_attrs = dict(f[grp].attrs)
                break
        # Older outputs kept them as attributes on Header itself.
        if not sim_attrs:
            if 'Header' in f:
                sim_attrs = dict(f['Header'].attrs)
            elif 'Core_0' in f and 'Header' in f['Core_0']:
                sim_attrs = dict(f['Core_0']['Header'].attrs)

        def _attr(*names, default=None):
            for n in names:
                if n in sim_attrs:
                    return float(sim_attrs[n])
            return default

        # frac_volume_processed lives on Header/Runtime, not Header/Simulation:
        # SAGE derives it from FirstFile, LastFile and num_simulation_tree_files.
        runtime_attrs = {}
        for grp in ('Header/Runtime', 'Core_0/Header/Runtime'):
            if grp in f:
                runtime_attrs = dict(f[grp].attrs)
                break

        def _rt_attr(*names, default=None):
            for n in names:
                if n in runtime_attrs:
                    return float(runtime_attrs[n])
            return default

        detected = {
            'boxsize': _attr('box_size', 'BoxSize'),
            'h0': _attr('hubble_h', 'HubbleParam'),
            'Omega0': _attr('omega_matter', 'Omega'),
            'vol_frac': _rt_attr('frac_volume_processed'),
        }

        # Resolve each: use the header value, or the explicit flag if the header
        # does not carry it.  An explicit flag that contradicts the header is an
        # error rather than an override -- getting the box size wrong rescales
        # every volume-dependent constraint (the SMF, the mass functions, the
        # densities) by (ratio)^3, silently, and --sim does not set it (that only
        # picks the snapshot-to-redshift map).
        wrong, missing = [], []
        for name, flag in (('boxsize', '--boxsize'), ('h0', '--h0'),
                           ('Omega0', '--Omega0'), ('vol_frac', '--vol-frac')):
            given = getattr(opts, name)
            found = detected[name]
            if found is None:
                if given is None:
                    missing.append('%s (not in the header either)' % flag)
                continue
            if given is not None and abs(found - given) > 1e-6 * max(abs(found), 1.0):
                wrong.append('%s %g   (the SAGE output says %g)' % (flag, given, found))
            setattr(opts, name, found)

        if wrong:
            parser.error(
                'these contradict the SAGE output in %s:\n    %s\n'
                'They are read from the output header automatically, so the '
                'simplest fix is to drop the flags entirely.'
                % (output_dir, '\n    '.join(wrong)))
        if missing:
            parser.error(
                'the SAGE output in %s does not record these, so they must be '
                'passed explicitly:\n    %s'
                % (output_dir, '\n    '.join(missing)))

        # The scale-factor list is recorded in both places: the header of the
        # output being scored (FileWithSnapList, written by SAGE at run time)
        # and the .par file SAGE is being run with.  Prefer the header -- it is
        # the authority on what actually produced this output, consistent with
        # how the box size and cosmology are resolved above.  The .par is the
        # fallback for when the recorded path does not exist on this machine
        # (output copied from a cluster, say).
        alist_par = None
        try:
            with open(opts.config) as pf:
                for line in pf:
                    parts = line.split()
                    if parts and parts[0] == 'FileWithSnapList':
                        alist_par = parts[1]
                        break
        except OSError:
            pass

        alist_header = sim_attrs.get('FileWithSnapList')
        if isinstance(alist_header, bytes):
            alist_header = alist_header.decode('utf-8', 'replace')

        if opts.age_alist_file is None:
            for candidate, source in ((alist_header, 'the SAGE output header'),
                                      (alist_par, os.path.basename(opts.config))):
                if candidate and os.path.exists(candidate):
                    opts.age_alist_file = os.path.abspath(candidate)
                    print('  Scale-factor list: %s [from %s]'
                          % (opts.age_alist_file, source))
                    break
            else:
                tried = [c for c in (alist_par, alist_header) if c]
                parser.error(
                    'could not find the scale-factor list automatically, so '
                    '--age-alist-file must be passed.%s'
                    % ('' if not tried else
                       ' Tried:\n    ' + '\n    '.join(tried)))
        else:
            recorded = next((c for c in (alist_header, alist_par)
                             if c and os.path.exists(c)), None)
            if recorded and os.path.abspath(recorded) != opts.age_alist_file:
                logger.warning(
                    '--age-alist-file is %s but the SAGE output/parameter file '
                    'records %s; using the flag. Omit it to follow the '
                    'simulation.', opts.age_alist_file, recorded)

        h = opts.h0
        box = opts.boxsize
        vol_frac = opts.vol_frac

        # Calculate Volume (Mpc^3)
        volume = (box / h)**3 * vol_frac
        print(f"  Volume: {volume:.2e} Mpc^3 (h={h}, Box={box}, Frac={vol_frac:.3f}, "
              f"Omega0={opts.Omega0}) [from the SAGE output header]")

        # Snapshot-to-redshift mapping, straight from Header/snapshot_redshifts.
        # Each constraint then gets the snapshot closest to the redshift its
        # observations were measured at, so the mapping is correct for any
        # simulation without a per-simulation table to maintain.
        snap_z, snap_available = read_output_snapshot_redshifts(output_dir)
        if snap_z is None or not snap_available:
            resolved_snapshot_map = None
            if opts.sim is None:
                parser.error(
                    'the SAGE output in %s has no snapshot redshift table, so '
                    'the snapshot-to-redshift mapping cannot be resolved. Pass '
                    '--sim (0=Uchuu grid, 1=Millennium grid, 2=MTNG) to use the '
                    'legacy hard-coded table.' % output_dir)
            print('  WARNING: no snapshot redshift table in the output; falling '
                  'back to the hard-coded map for --sim %d' % opts.sim)
        else:
            resolved_snapshot_map = build_snapshot_map(snap_z, snap_available)
            # --sim is now only a label and a fallback: identify it from the
            # snapshot count so it need not be passed.
            if opts.sim is None:
                opts.sim = identify_simulation(len(snap_available))
                if opts.sim is None:
                    opts.sim = 0
                    print('  Simulation: %d snapshots, not a known grid; '
                          '--sim defaulted to 0 (only affects the legacy '
                          'fallback, which is unused here)'
                          % len(snap_available))
                else:
                    print('  Simulation: %s [inferred from %d snapshots]'
                          % (get_simulation_config(opts.sim)['name'],
                             len(snap_available)))
            print('  Snapshots: %d available, z = %.4f to %.2f [from the SAGE '
                  'output header]' % (len(snap_available),
                                      snap_z[snap_available[-1]],
                                      snap_z[snap_available[0]]))

        # Detect Structure Type
        top_keys = list(f.keys())
        if any(k.startswith('Core_') for k in top_keys):
            structure_type = 'core_level'
        elif any(k.startswith('Snap_') for k in top_keys):
            structure_type = 'snap_level'
        else:
            raise ValueError("Unknown HDF5 structure")

    # 5. Define Target Snapshots (z=0, 0.5, 1.0, 2.0, 3.0, 4.0)
    # Use simulation-specific snapshots
    if snap_z is not None and snap_available:
        target_snapshots = resolve_target_snapshots(snap_z, snap_available)
        print('Reference epochs: %s  (z = %s)'
              % (target_snapshots,
                 ', '.join('%.2f' % snap_z[s] for s in target_snapshots)))
    else:
        target_snapshots = get_target_snapshots(opts.sim)
        print(f"Target Snapshots for sim={opts.sim}: {target_snapshots}")

    # Snapshots for History (CSFRDH, SMD) - spanning z=0 to z~4
    # Use simulation-specific snapshots
    if snap_z is not None and snap_available:
        history_snapshots = resolve_history_snapshots(snap_z, snap_available)
    else:
        history_snapshots = get_history_snapshots(opts.sim)

    # Initialize Data Containers
    smf_data_columns = []
    smf_red_data_columns = []
    smf_blue_data_columns = []
    bhmf_data_columns = []
    bhbm_data_columns = []
    halostellar_data_columns = []
    mzr_data_columns = []
    himf_data_columns = []
    h2mf_data_columns = []

    # History Containers
    history_z = []
    history_t = []
    history_sfrd = []
    history_smd = []
    history_fics = []

    # Define Bins
    binwidth = 0.1
    smf_bins = np.arange(6.0, 13.0, binwidth)
    bhmf_bins = np.arange(5.0, 11.0, binwidth)
    bhbm_bins = np.arange(8.0, 12.5, 0.2)
    hs_bins = np.arange(9.0, 15.0, 0.2)
    mzr_bins = np.arange(8.5, 11.5, 0.2)
    himf_bins = np.arange(7.0, 11.5, binwidth)
    h2mf_bins = np.arange(7.0, 11.0, binwidth)

    # 6. Process Specific Snapshots
    for snap_num in target_snapshots:
        snap_key = f"Snap_{snap_num}"
        print(f"  Processing {snap_key}...")

        g_stellar = []
        g_bhole = []
        g_bulge = []
        g_mvir = []
        g_coldgas = []
        g_metals = []
        g_h1gas = []
        g_h2gas = []
        g_sfrdisk = []
        g_sfrbulge = []

        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                def get_props(loc):
                    s = np.array(loc['StellarMass']) * 1.0e10 / h if 'StellarMass' in loc else []
                    bh = np.array(loc['BlackHoleMass']) * 1.0e10 / h if 'BlackHoleMass' in loc else []
                    b = np.array(loc['BulgeMass']) * 1.0e10 / h if 'BulgeMass' in loc else []
                    m = np.array(loc['Mvir']) * 1.0e10 / h if 'Mvir' in loc else []
                    cg = np.array(loc['ColdGas']) * 1.0e10 / h if 'ColdGas' in loc else []
                    met = np.array(loc['MetalsColdGas']) * 1.0e10 / h if 'MetalsColdGas' in loc else []
                    h1 = np.array(loc['H1gas']) * 1.0e10 / h if 'H1gas' in loc else []
                    h2 = np.array(loc['H2gas']) * 1.0e10 / h if 'H2gas' in loc else []
                    sfrd = np.array(loc['SfrDisk']) if 'SfrDisk' in loc else []
                    sfrb = np.array(loc['SfrBulge']) if 'SfrBulge' in loc else []
                    return s, bh, b, m, cg, met, h1, h2, sfrd, sfrb

                if structure_type == 'core_level':
                    for core in f.keys():
                        if snap_key in f[core]:
                            s, bh, b, m, cg, met, h1, h2, sfrd, sfrb = get_props(f[core][snap_key])
                            g_stellar.extend(s)
                            g_bhole.extend(bh)
                            g_bulge.extend(b)
                            g_mvir.extend(m)
                            g_coldgas.extend(cg)
                            g_metals.extend(met)
                            g_h1gas.extend(h1)
                            g_h2gas.extend(h2)
                            g_sfrdisk.extend(sfrd)
                            g_sfrbulge.extend(sfrb)
                else:
                    if snap_key in f:
                        s, bh, b, m, cg, met, h1, h2, sfrd, sfrb = get_props(f[snap_key])
                        g_stellar.extend(s)
                        g_bhole.extend(bh)
                        g_bulge.extend(b)
                        g_mvir.extend(m)
                        g_coldgas.extend(cg)
                        g_metals.extend(met)
                        g_h1gas.extend(h1)
                        g_h2gas.extend(h2)
                        g_sfrdisk.extend(sfrd)
                        g_sfrbulge.extend(sfrb)

        g_stellar = np.array(g_stellar)
        g_bhole = np.array(g_bhole)
        g_bulge = np.array(g_bulge)
        g_mvir = np.array(g_mvir)
        g_coldgas = np.array(g_coldgas)
        g_metals = np.array(g_metals)
        g_h1gas = np.array(g_h1gas)
        g_h2gas = np.array(g_h2gas)
        g_sfrdisk = np.array(g_sfrdisk)
        g_sfrbulge = np.array(g_sfrbulge)

        # --- SMF ---
        valid = g_stellar > 0
        if np.sum(valid) > 0:
            hist, edges = np.histogram(np.log10(g_stellar[valid]), bins=smf_bins)
            phi = hist / (volume * binwidth)
            centers = edges[:-1] + binwidth / 2
            phi[phi == 0] = np.nan
        else:
            centers = smf_bins[:-1] + binwidth / 2
            phi = np.zeros_like(centers)
        smf_data_columns.extend([centers, phi])

        # --- SMF Red (Quiescent, sSFR < -11) ---
        sSFRcut = -11.0
        valid_sfr = (g_stellar > 0) & (g_sfrdisk + g_sfrbulge > 0)
        if np.sum(valid_sfr) > 0:
            sSFR = np.log10((g_sfrdisk[valid_sfr] + g_sfrbulge[valid_sfr]) / g_stellar[valid_sfr])
            logM = np.log10(g_stellar[valid_sfr])
            # Red galaxies: sSFR < -11
            red_mask = sSFR < sSFRcut
            if np.sum(red_mask) > 0:
                hist_red, edges = np.histogram(logM[red_mask], bins=smf_bins)
                phi_red = hist_red / (volume * binwidth)
                centers = edges[:-1] + binwidth / 2
                phi_red[phi_red == 0] = np.nan
            else:
                centers = smf_bins[:-1] + binwidth / 2
                phi_red = np.full_like(centers, np.nan)
            # Blue galaxies: sSFR > -11
            blue_mask = sSFR > sSFRcut
            if np.sum(blue_mask) > 0:
                hist_blue, edges = np.histogram(logM[blue_mask], bins=smf_bins)
                phi_blue = hist_blue / (volume * binwidth)
                centers = edges[:-1] + binwidth / 2
                phi_blue[phi_blue == 0] = np.nan
            else:
                centers = smf_bins[:-1] + binwidth / 2
                phi_blue = np.full_like(centers, np.nan)
        else:
            centers = smf_bins[:-1] + binwidth / 2
            phi_red = np.full_like(centers, np.nan)
            phi_blue = np.full_like(centers, np.nan)
        smf_red_data_columns.extend([centers, phi_red])
        smf_blue_data_columns.extend([centers, phi_blue])

        # --- BHMF ---
        valid = g_bhole > 0
        if np.sum(valid) > 0:
            hist, edges = np.histogram(np.log10(g_bhole[valid]), bins=bhmf_bins)
            phi = hist / (volume * binwidth)
            centers = edges[:-1] + binwidth / 2
            phi[phi == 0] = np.nan
        else:
            centers = bhmf_bins[:-1] + binwidth / 2
            phi = np.zeros_like(centers)
        bhmf_data_columns.extend([centers, phi])

        # --- BHBM ---
        valid = (g_bulge > 0) & (g_bhole > 0)
        if np.sum(valid) > 0:
            x, y = np.log10(g_bulge[valid]), np.log10(g_bhole[valid])
            median_y, _, _ = stats.binned_statistic(x, y, 'median', bins=bhbm_bins)
            std_y, _, _ = stats.binned_statistic(x, y, 'std', bins=bhbm_bins)
            count_y, edges, _ = stats.binned_statistic(x, y, 'count', bins=bhbm_bins)
            centers = edges[:-1] + (edges[1] - edges[0])/2
        else:
            centers = bhbm_bins[:-1] + (bhbm_bins[1]-bhbm_bins[0])/2
            median_y = np.full_like(centers, np.nan)
            std_y = np.full_like(centers, np.nan)
            count_y = np.zeros_like(centers)
        bhbm_data_columns.extend([centers, median_y, std_y, count_y])

        # --- Halo-Stellar (SHMR) ---
        valid = (g_mvir > 0) & (g_stellar > 0)
        if np.sum(valid) > 0:
            x, y = np.log10(g_mvir[valid]), np.log10(g_stellar[valid])
            median_y, _, _ = stats.binned_statistic(x, y, 'median', bins=hs_bins)
            std_y, _, _ = stats.binned_statistic(x, y, 'std', bins=hs_bins)
            count_y, edges, _ = stats.binned_statistic(x, y, 'count', bins=hs_bins)
            centers = edges[:-1] + (edges[1] - edges[0])/2
        else:
            centers = hs_bins[:-1] + (hs_bins[1]-hs_bins[0])/2
            median_y = np.full_like(centers, np.nan)
            std_y = np.full_like(centers, np.nan)
            count_y = np.zeros_like(centers)
        halostellar_data_columns.extend([centers, median_y, std_y, count_y])

        # --- MZR (Mass-Metallicity) ---
        if len(g_coldgas) > 0:
            valid = (g_coldgas > 0) & (g_stellar > 0) & (g_metals > 0)
            if np.sum(valid) > 0:
                Z = np.log10((g_metals[valid] / g_coldgas[valid]) / 0.02) + 9.0
                logM = np.log10(g_stellar[valid])
                median_Z, _, _ = stats.binned_statistic(logM, Z, 'median', bins=mzr_bins)
                centers = mzr_bins[:-1] + (mzr_bins[1] - mzr_bins[0])/2
            else:
                centers = mzr_bins[:-1] + (mzr_bins[1] - mzr_bins[0])/2
                median_Z = np.full_like(centers, np.nan)
            mzr_data_columns.extend([centers, median_Z])

        # --- HIMF (HI Mass Function) ---
        # HI mass is now output directly by SAGE as H1gas
        if len(g_h1gas) > 0:
            valid = g_h1gas > 0
            if np.sum(valid) > 0:
                hist, edges = np.histogram(np.log10(g_h1gas[valid]), bins=himf_bins)
                phi = hist / (volume * binwidth)
                centers = edges[:-1] + binwidth / 2
                phi[phi == 0] = np.nan
            else:
                centers = himf_bins[:-1] + binwidth / 2
                phi = np.full_like(centers, np.nan)
        else:
            centers = himf_bins[:-1] + binwidth / 2
            phi = np.full_like(centers, np.nan)
        himf_data_columns.extend([centers, phi])

        # --- H2MF (H2 Mass Function) ---
        if len(g_h2gas) > 0:
            valid = g_h2gas > 0
            if np.sum(valid) > 0:
                hist, edges = np.histogram(np.log10(g_h2gas[valid]), bins=h2mf_bins)
                phi = hist / (volume * binwidth)
                centers = edges[:-1] + binwidth / 2
                phi[phi == 0] = np.nan
            else:
                centers = h2mf_bins[:-1] + binwidth / 2
                phi = np.full_like(centers, np.nan)
        else:
            centers = h2mf_bins[:-1] + binwidth / 2
            phi = np.full_like(centers, np.nan)
        h2mf_data_columns.extend([centers, phi])

    # --- 6b. Process History Snapshots (CSFRDH, SMD & FICS) ---
    print(f"  Processing History ({len(history_snapshots)} snapshots)...")
    from src import routines as r
    from src.constraints import fics_median

    # f_ICS is a per-halo quantity, so unlike the SFRD and SMD sums it needs the
    # galaxy arrays of a whole snapshot together (satellites must be matched to
    # their central).  Collect them per file and concatenate before measuring.
    fics_fields = ['StellarMass', 'IntraClusterStars', 'Mvir', 'Type',
                   'GalaxyIndex', 'CentralGalaxyIndex']

    for snap_num in history_snapshots:
        snap_key = f"Snap_{snap_num}"
        total_sfr = 0.0
        total_sm = 0.0
        current_z = -1.0
        fics_chunks = {field: [] for field in fics_fields}

        def _accumulate(loc):
            """Add one snapshot group's contribution to the running totals."""
            nonlocal total_sfr, total_sm
            if 'SfrDisk' in loc: total_sfr += np.sum(loc['SfrDisk'])
            if 'SfrBulge' in loc: total_sfr += np.sum(loc['SfrBulge'])
            if 'StellarMass' in loc: total_sm += np.sum(loc['StellarMass']) * 1.0e10 / h
            if all(field in loc for field in fics_fields):
                for field in fics_fields:
                    fics_chunks[field].append(np.array(loc[field]))

        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                if structure_type == 'core_level':
                    first_core = [k for k in f.keys() if k.startswith('Core_')][0]
                    if snap_key in f[first_core]:
                        # Read redshift directly from snapshot attributes
                        current_z = f[first_core][snap_key].attrs.get('redshift', -1.0)
                        for core in f.keys():
                            if core.startswith('Core_') and snap_key in f[core]:
                                _accumulate(f[core][snap_key])
                else:
                    if snap_key in f:
                        loc = f[snap_key]
                        current_z = loc.attrs.get('redshift', -1.0)
                        _accumulate(loc)

        if current_z >= 0:
            z = current_z
            tL = r.z2tL(z, h, opts.Omega0, 1.0-opts.Omega0)
        else:
            z = 0; tL = 0

        # Median f_ICS over the groups and clusters of this snapshot.  NaN (too
        # few haloes, or a SAGE build without IntraClusterStars) is written as
        # the same -99 sentinel the other history columns use.
        f_ics = np.nan
        if fics_chunks['StellarMass']:
            G_snap = {field: np.concatenate(chunks)
                      for field, chunks in fics_chunks.items()}
            f_ics, _ = fics_median(G_snap, h)

        history_z.append(z)
        history_t.append(tL)
        history_sfrd.append(np.log10(total_sfr / volume) if total_sfr > 0 else -99)
        history_smd.append(np.log10(total_sm / volume) if total_sm > 0 else -99)
        history_fics.append(f_ics if np.isfinite(f_ics) else -99)

    # --- Write All CSV Files (always regenerate) ---
    def write_wide_csv(filename, columns):
        if not columns: return
        path = os.path.join(data_dir, filename)
        data_matrix = np.column_stack(columns)
        np.savetxt(path, data_matrix, delimiter='\t', fmt='%.6e')
        print(f"Generated {path}")

    write_wide_csv('sage_smf_all_redshifts.csv', smf_data_columns)
    write_wide_csv('sage_smf_extra_redshifts.csv', smf_data_columns)
    write_wide_csv('sage_smf_red_all_redshifts.csv', smf_red_data_columns)
    write_wide_csv('sage_smf_blue_all_redshifts.csv', smf_blue_data_columns)
    write_wide_csv('sage_bhmf_all_redshifts.csv', bhmf_data_columns)
    write_wide_csv('sage_bhbm_all_redshifts.csv', bhbm_data_columns)
    write_wide_csv('sage_halostellar_all_redshifts.csv', halostellar_data_columns)
    write_wide_csv('sage_mzr_all_redshifts.csv', mzr_data_columns)
    write_wide_csv('sage_himf_all_redshifts.csv', himf_data_columns)
    write_wide_csv('sage_h2mf_all_redshifts.csv', h2mf_data_columns)

    # Write History File (Z, Time, SFRD, SMD, f_ICS)
    # f_ICS is a linear fraction, not a log, and -99 marks snapshots with too
    # few groups and clusters to measure it.
    hist_data = np.column_stack((history_z, history_t, history_sfrd, history_smd, history_fics))
    np.savetxt(os.path.join(data_dir, 'sage_history.csv'), hist_data,
            delimiter='\t', header='Redshift\tLookbackTime\tlogSFRD\tlogSMD\tfICS', comments='')
    print(f"Generated {os.path.join(data_dir, 'sage_history.csv')}")

    # Determine snapshots needed for constraints
    if opts.snapshot is not None:
        snapshots = opts.snapshot
    else:
        snapshots = get_required_snapshots(opts.constraints, sim=opts.sim,
                                           snapshot_map=resolved_snapshot_map)
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
                                    snapshot_map=resolved_snapshot_map,
                                    boxsize=opts.boxsize,
                                    sim=opts.sim,
                                    vol_frac=opts.vol_frac,
                                    age_alist_file=opts.age_alist_file,
                                    Omega0=opts.Omega0, h0=opts.h0,
                                    output_dir=opts.outdir)
    # Load the search space
    space = analysis.load_space(opts.space_file)

    ss = opts.swarm_size
    if ss is None:
        ss = 10 + int(2 * math.sqrt(len(space)))

    args = (opts, space, subvols, analysis.stat_tests[opts.stat_test])

    if opts.hpc_mode:
        procs = 0
        f = execution.run_sage_hpc
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
    logger.info('    Simulation: %s (snapshots and redshifts resolved from the '
                'SAGE output header)',
                get_simulation_config(opts.sim)['name'] if opts.sim is not None
                else 'inferred from the output')
    logger.info('    Box Size: %.1f', opts.boxsize)
    logger.info('    Volume Fraction: %.4f', opts.vol_frac)
    logger.info('    Age List File: %s', opts.age_alist_file if opts.age_alist_file else 'Using default')
    logger.info('    Omega0: %.4f', opts.Omega0)
    logger.info('    h0: %.4f', opts.h0)
    logger.info('    Keep temporary output files: %d', opts.keep)
    logger.info('    Snapshot Number: %s', opts.snapshot)
    if resolved_snapshot_map is not None:
        logger.info("    Snapshot resolution (nearest to each observation's redshift):")
        active = {c.__class__.__name__ for c in opts.constraints}
        for line in describe_snapshot_map(
                {k: v for k, v in resolved_snapshot_map.items() if k in active},
                snap_z):
            logger.info('        %s', line)
    logger.info("PSO information:")
    logger.info('    Search space parameters: %s', ' '.join(space['name']))
    logger.info('    Swarm size: %d', ss)
    logger.info('    Maximum iterations: %d', opts.max_iterations)
    logger.info('    Max stagnation: %d', opts.max_stagnation)
    logger.info('    PSO Hyperparameters:')
    logger.info('        omega (inertia): %.3f', opts.omega)
    logger.info('        phip (cognitive): %.3f', opts.phip)
    logger.info('        phig (social): %.3f', opts.phig)
    logger.info('    Lower bounds: %r', space['lb'])
    logger.info('    Upper bounds: %r', space['ub'])
    switches = [space['name'][i] for i in np.flatnonzero(space['is_int'])]
    if switches:
        levels = 1
        for i in np.flatnonzero(space['is_int']):
            levels *= int(round(space['ub'][i] - space['lb'][i])) + 1
        logger.info('    Integer switches: %s (%d level combinations)',
                    ', '.join(switches), levels)
        logger.info('    Note: a switch makes the objective piecewise constant '
                    'along that axis, and particles that round to the same '
                    'level run identical SAGE configurations. For a switch with '
                    'few levels, separate runs per level are more informative.')
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
    logger.info('-----------------------------------------------------')

    
    # Directory where we store the intermediate results
    tracksdir = os.path.join(opts.outdir, 'tracks')
    try:
        os.makedirs(tracksdir)
    except OSError:
        pass

    # Go, go, go!
    logger.info('Starting PSO now')
    tStart = time.time()
    is_log = space['is_log'].astype(bool)
    is_int = space['is_int'].astype(bool)
    lb_pso, ub_pso = analysis.pso_bounds(space)
    xopt, fopt = pso.pso(f, lb_pso, ub_pso, args=args, swarmsize=ss,
                         maxiter=opts.max_iterations, processes=procs,
                         omega=opts.omega, phip=opts.phip, phig=opts.phig,
                         dumpfile_prefix=os.path.join(tracksdir, 'track_%03d'),
                         csv_output_path=opts.csv_output,
                         random_seed=opts.random_seed,
                         is_log=is_log,
                         is_int=is_int,
                         int_lb=space['lb'], int_ub=space['ub'],
                         max_stagnation=opts.max_stagnation)
    tEnd = time.time()

    global count
    #logger.info('Number of iterations = %d', count)
    logger.info('')
    logger.info('=' * 62)
    logger.info(' Best fit   (%s = %.6g,  %d particles x %d iterations,  %.1f s)',
                opts.stat_test, fopt, ss, opts.max_iterations, tEnd - tStart)
    logger.info('=' * 62)
    _lb, _ub = space['lb'], space['ub']
    for _i, _name in enumerate(space['name']):
        _v = xopt[_i]
        # flag a best fit sitting on a bound: that is a bound, not a fit
        _span = max(_ub[_i] - _lb[_i], 1e-30)
        _at = ('  <-- at lower bound' if _v <= _lb[_i] + 0.01 * _span else
               '  <-- at upper bound' if _v >= _ub[_i] - 0.01 * _span else '')
        _fmt = '%d' % _v if space['is_int'][_i] else '%-12.6g' % _v
        logger.info('   %-28s %-12s  [%g, %g]%s',
                    _name, _fmt, _lb[_i], _ub[_i], _at)
    logger.info('=' * 62)
    logger.info('')
    dump_files = glob.glob(os.path.join(opts.outdir, 'SMF_z*_dump.txt'))
    dump_files2 = glob.glob(os.path.join(opts.outdir, 'BHMF_z*_dump.txt'))
    dump_files3 = glob.glob(os.path.join(opts.outdir, 'BHBM_z*_dump.txt'))

    logger.info('Producing diagnostics...')
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

