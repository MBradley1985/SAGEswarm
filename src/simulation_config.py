#!/bin/bash

"""
Simulation configuration module for SAGE-PSO.
Contains simulation-specific parameters including snapshots, redshifts, and cosmology.
"""

import os

import numpy as np

# Simulation type constants
SIM_MINIUCHUU = 0
SIM_MINIMILLENNIUM = 1
SIM_MTNG = 2

# miniUchuu simulation configuration
MINIUCHUU_CONFIG = {
    'name': 'miniUchuu',
    'FirstSnap': 0,
    'LastSnap': 49,
    'redshifts': np.array([
        13.9334, 12.67409, 11.50797, 10.44649, 9.480752, 8.58543, 7.77447, 7.032387, 6.344409, 5.721695,
        5.153127, 4.629078, 4.26715, 3.929071, 3.610462, 3.314082, 3.128427, 2.951226, 2.77809, 2.616166,
        2.458114, 2.309724, 2.16592, 2.027963, 1.8962, 1.770958, 1.65124, 1.535928, 1.426272, 1.321656,
        1.220303, 1.124166, 1.031983, 0.9441787, 0.8597281, 0.779046, 0.7020205, 0.6282588, 0.5575475, 0.4899777,
        0.4253644, 0.3640053, 0.3047063, 0.2483865, 0.1939743, 0.1425568, 0.09296665, 0.0455745, 0.02265383, 0.0001130128
    ]),
    # Cosmology
    'h0': 0.6774,
    'Omega0': 0.3089,
    'boxsize': 400.0,
    # Default paths
    'age_alist_file': '/fred/oz004/msinha/simulations/uchuu_suite/miniuchuu/mergertrees/u400_planck2016_50.a_list',
}

# miniMillennium simulation configuration (placeholder - adjust values as needed)
MINIMILLENNIUM_CONFIG = {
    'name': 'miniMillennium',
    'FirstSnap': 0,
    'LastSnap': 63,
    'redshifts': None,  # Will be computed from a_list file if needed
    # Cosmology (Millennium)
    'h0': 0.73,
    'Omega0': 0.25,
    'boxsize': 62.5,
    'age_alist_file': None,
}

# MTNG simulation configuration (placeholder - adjust values as needed)
MTNG_CONFIG = {
    'name': 'MTNG',
    'FirstSnap': 0,
    'LastSnap': 99,
    'redshifts': None,  # Will be computed from a_list file if needed
    # Cosmology
    'h0': 0.6774,
    'Omega0': 0.3089,
    'boxsize': 500.0,
    'age_alist_file': None,
}

# Map simulation ID to config
SIMULATION_CONFIGS = {
    SIM_MINIUCHUU: MINIUCHUU_CONFIG,
    SIM_MINIMILLENNIUM: MINIMILLENNIUM_CONFIG,
    SIM_MTNG: MTNG_CONFIG,
}


def get_simulation_config(sim_id):
    """Get the configuration for a given simulation ID."""
    if sim_id not in SIMULATION_CONFIGS:
        raise ValueError(f"Unknown simulation ID: {sim_id}. Must be 0 (miniUchuu), 1 (miniMillennium), or 2 (MTNG)")
    return SIMULATION_CONFIGS[sim_id]


def find_snapshot_for_redshift(sim_id, target_z, tolerance=0.1):
    """
    Find the snapshot number closest to a target redshift for a given simulation.

    Parameters:
    -----------
    sim_id : int
        Simulation ID (0=miniUchuu, 1=miniMillennium, 2=MTNG)
    target_z : float
        Target redshift value
    tolerance : float
        Maximum allowed difference between target and actual redshift

    Returns:
    --------
    int : Snapshot number closest to target redshift, or None if no match within tolerance
    """
    config = get_simulation_config(sim_id)
    redshifts = config['redshifts']

    if redshifts is None:
        return None

    # Find closest redshift
    diff = np.abs(redshifts - target_z)
    min_idx = np.argmin(diff)

    if diff[min_idx] <= tolerance:
        return min_idx
    return None


def get_snapshot_redshift(sim_id, snapshot):
    """
    Get the redshift for a given snapshot number.

    Parameters:
    -----------
    sim_id : int
        Simulation ID
    snapshot : int
        Snapshot number

    Returns:
    --------
    float : Redshift at that snapshot, or None if invalid
    """
    config = get_simulation_config(sim_id)
    redshifts = config['redshifts']

    if redshifts is None or snapshot < 0 or snapshot >= len(redshifts):
        return None

    return redshifts[snapshot]


def get_snapshot_map(sim_id):
    """
    Get the constraint-to-snapshot mapping for a given simulation.

    This maps constraint names to the appropriate snapshot numbers for each simulation.

    Parameters:
    -----------
    sim_id : int
        Simulation ID (0=miniUchuu, 1=miniMillennium, 2=MTNG)

    Returns:
    --------
    dict : Mapping of constraint names to snapshot lists
    """
    config = get_simulation_config(sim_id)

    if sim_id == SIM_MINIUCHUU:
        # miniUchuu: 50 snapshots (0-49), snapshot 49 is z~0
        # Find snapshots closest to target redshifts
        return {
            'SMF_z0': [49],      # z ~ 0
            'SMF_z05': [38],     # z ~ 0.56
            'SMF_z10': [32],     # z ~ 1.03
            'SMF_z20': [23],     # z ~ 2.03
            'SMF_z30': [17],     # z ~ 2.95
            'SMF_z40': [12],     # z ~ 4.27
            'SMF_Red_z0': [49],
            'SMF_Blue_z0': [49],
            'BHMF_z0': [49],
            'BHMF_z10': [32],
            'BHBM': [49],
            'CSFRDH': [12, 15, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 49],  # z~4 to z~0
            'HIMF': [49],
            'H2MF': [49],
            'MZR': [49],
            'SHMR': [49],
            'SMD': [5, 8, 10, 12, 15, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 49],  # Extended history
            'FICS': [23, 26, 29, 32, 35, 38, 41, 44, 47, 49],  # z~2.03 to z~0
            'FICS_Mvir': [49],   # z ~ 0
        }
    elif sim_id == SIM_MINIMILLENNIUM:
        # miniMillennium: 64 snapshots (0-63), snapshot 63 is z=0
        # snap 19 ≈ z=5.72, snap 18 ≈ z=6.20, snap 16 ≈ z=7.27
        return {
            'SMF_z0': [63],
            'SMF_z05': [48],
            'SMF_z10': [40],
            'SMF_z20': [32],
            'SMF_z30': [27],
            'SMF_z40': [23],
            'SMF_z50': [19],
            'SMF_z60': [18],
            'SMF_z70': [16],
            'SMF_z80': [15],
            'SMF_z100': [12],
            'SMF_Red_z0': [63],
            'SMF_Blue_z0': [63],
            'BHMF_z0': [63],
            'BHMF_z10': [40],
            'BHBM': [63],
            'CSFRDH': [23, 27, 32, 36, 40, 44, 48, 52, 56, 60, 63],
            'HIMF': [63],
            'H2MF': [63],
            'MZR': [63],
            'SHMR': [63],
            'SMD': [10, 14, 18, 23, 27, 32, 36, 40, 44, 48, 52, 56, 60, 63],
            'FICS': [32, 34, 36, 38, 40, 44, 48, 52, 56, 60, 63],  # z~2.07 to z=0
            'FICS_Mvir': [63],   # z = 0
        }
    else:  # MTNG
        # MTNG: 100 snapshots (0-99), snapshot 99 is z=0
        return {
            'SMF_z0': [99],
            'SMF_z05': [78],
            'SMF_z10': [67],
            'SMF_z20': [50],
            'SMF_z30': [40],
            'SMF_z40': [33],
            'SMF_Red_z0': [99],
            'SMF_Blue_z0': [99],
            'BHMF_z0': [99],
            'BHMF_z10': [67],
            'BHBM': [99],
            'CSFRDH': [33, 40, 50, 55, 60, 67, 72, 78, 84, 90, 95, 99],
            'HIMF': [99],
            'H2MF': [99],
            'MZR': [99],
            'SHMR': [99],
            'SMD': [20, 25, 30, 33, 40, 50, 55, 60, 67, 72, 78, 84, 90, 95, 99],
            'FICS': [50, 55, 60, 67, 72, 78, 84, 90, 95, 99],  # z~2 to z=0
            'FICS_Mvir': [99],   # z = 0
        }


def get_target_snapshots(sim_id):
    """
    Get the target snapshots for mass function processing (z=0, 0.5, 1.0, 2.0, 3.0, 4.0).

    Parameters:
    -----------
    sim_id : int
        Simulation ID

    Returns:
    --------
    list : List of snapshot numbers corresponding to z=0, 0.5, 1.0, 2.0, 3.0, 4.0
    """
    if sim_id == SIM_MINIUCHUU:
        # miniUchuu snapshots for z ~ [0, 0.5, 1.0, 2.0, 3.0, 4.0]
        return [49, 38, 32, 23, 17, 12]
    elif sim_id == SIM_MINIMILLENNIUM:
        # Millennium snapshots for 11 redshifts — column pairs 0–10 in sage_smf_all_redshifts.csv:
        # z≈[0, 0.18, 0.51, 0.76, 1.08, 1.50, 2.07, 2.42, 3.06, 3.58, 4.18]
        # Pairs 2,6,8,10 map to SMF_z05, SMF_z20, SMF_z30, SMF_z40 column expectations.
        return [63, 56, 48, 44, 40, 36, 32, 30, 27, 25, 23]
    else:  # MTNG
        return [99, 78, 67, 50, 40, 33]


def get_history_snapshots(sim_id):
    """
    Get the history snapshots for CSFRDH and SMD constraints (spanning cosmic history).

    Parameters:
    -----------
    sim_id : int
        Simulation ID

    Returns:
    --------
    list : List of snapshot numbers spanning from high-z to z=0
    """
    if sim_id == SIM_MINIUCHUU:
        # miniUchuu: snapshots from z~8 to z~0
        return [5, 8, 10, 12, 15, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 49]
    elif sim_id == SIM_MINIMILLENNIUM:
        # Millennium snapshots for cosmic history
        return [10, 14, 18, 23, 27, 32, 36, 40, 44, 48, 52, 56, 60, 63]
    else:  # MTNG
        return [20, 25, 30, 33, 40, 50, 55, 60, 67, 72, 78, 84, 90, 95, 99]


def get_z0_snapshot(sim_id):
    """Get the z=0 snapshot number for a simulation."""
    if sim_id == SIM_MINIUCHUU:
        return 49
    elif sim_id == SIM_MINIMILLENNIUM:
        return 63
    else:  # MTNG
        return 99


def get_csfrdh_snapshots(sim_id):
    """Get the snapshots used for CSFRDH constraint."""
    snapshot_map = get_snapshot_map(sim_id)
    return snapshot_map['CSFRDH']


def get_smd_snapshots(sim_id):
    """Get the snapshots used for SMD constraint."""
    snapshot_map = get_snapshot_map(sim_id)
    return snapshot_map['SMD']


def get_fics_snapshots(sim_id):
    """Get the snapshots used for the FICS constraint (z = 0 to z = 2)."""
    snapshot_map = get_snapshot_map(sim_id)
    return snapshot_map['FICS']


def get_fics_mvir_snapshot(sim_id):
    """Get the z = 0 snapshot used for the FICS_Mvir constraint."""
    snapshot_map = get_snapshot_map(sim_id)
    return snapshot_map['FICS_Mvir']

# ---------------------------------------------------------------------------
# Resolving snapshots from the SAGE output rather than from hard-coded tables
# ---------------------------------------------------------------------------
#
# A snapshot number means nothing on its own -- it is only a label for a
# redshift, and the mapping differs between simulations.  What a constraint
# actually needs is "the snapshot closest to the redshift my observations were
# measured at".  SAGE writes the full mapping into Header/snapshot_redshifts, so
# that question can be answered from the output file instead of from a table
# that has to be maintained per simulation (and which had SMF_z50 pointing at
# z = 5.72 for observations measured at z = 4.8-5.3).
#
# The only thing that must be declared here is each constraint's TARGET
# REDSHIFT, because that is a property of the observations, not of the
# simulation.  Everything else is read from the file.

# Single-epoch constraints: the redshift their observations represent.
CONSTRAINT_TARGET_REDSHIFT = {
    'SMF_z0':      0.0,    # paper_smf_observations.csv z bin 0.0-0.2;
                           # pinned to the z=0 snapshot, not the bin centre,
                           # since the SMF barely evolves over 0 < z < 0.2
    'SMF_z05':     0.65,   # z bin 0.5-0.8
    'SMF_z10':     1.00,   # z bin 0.8-1.2
    'SMF_z20':     2.05,   # z bin 1.8-2.3
    'SMF_z30':     3.05,   # z bin 2.8-3.3
    'SMF_z40':     4.05,   # z bin 3.8-4.3
    'SMF_z50':     5.05,   # z bin 4.8-5.3
    'SMF_z60':     6.05,   # z bin 5.8-6.3
    'SMF_z70':     7.05,   # z bin 6.8-7.3
    'SMF_z80':     8.05,   # z bin 7.8-8.3
    'SMF_z100':   10.25,   # z bin 9.5-11.0
    'SMF_Red_z0':  0.0,    # GAMA morphological SMF
    'SMF_Blue_z0': 0.0,
    'BHMF_z0':     0.0,    # fig4_bhmf_z0.1.txt; as SMF_z0, pinned to z=0
    'BHMF_z10':    1.00,   # fig4_bhmf_z1.0.txt
    'BHBM':        0.0,
    'HIMF':        0.0,    # Jones+18
    'H2MF':        0.0,    # Fletcher+21
    'MZR':         0.0,    # Curti+20 (SDSS)
    'SHMR':        0.0,    # Correa & Schaye 19
    'FICS_Mvir':   0.0,    # Contini 2021, z ~ 0
    'MLF':         0.0,    # outflow compilation, z ~ 0 (Sugahara+17 reaches z~2)
}

# History constraints: the redshift range they span, and how many epochs to
# sample across it.  The count is a compromise -- enough to trace the shape,
# few enough that one constraint does not dominate the objective by point count.
CONSTRAINT_TARGET_REDSHIFT_RANGE = {
    'FICS':   (0.0, 2.0, 11),    # ICL fraction compilation spans z = 0-2
    'CSFRDH': (0.0, 6.0, 14),    # domain is lookback time to 12.62 Gyr ~ z = 6
    'SMD':    (0.0, 8.0, 16),    # SMD.ecsv spans z = 0.35-11
}


def read_output_snapshot_redshifts(modeldir):
    """Snapshot-to-redshift mapping and available snapshots, from SAGE output.

    Returns (redshifts, available) where `redshifts` is indexed by snapshot
    number and `available` is the sorted list of snapshots the file contains.
    Returns (None, None) if no readable output is present.
    """
    import glob
    import h5py

    files = sorted(glob.glob(os.path.join(modeldir, 'model_*.hdf5')))
    if not files:
        return None, None

    with h5py.File(files[0], 'r') as f:
        redshifts = None
        for path in ('Header/snapshot_redshifts', 'Core_0/Header/snapshot_redshifts'):
            if path in f:
                redshifts = np.array(f[path], dtype=float)
                break

        available = sorted(int(k.split('_')[1]) for k in f.keys()
                           if k.startswith('Snap_'))
        if not available and 'Core_0' in f:
            available = sorted(int(k.split('_')[1]) for k in f['Core_0'].keys()
                               if k.startswith('Snap_'))

        # Fall back to the per-snapshot redshift attribute if the header
        # dataset is absent (older output).
        if redshifts is None and available:
            n = max(available) + 1
            redshifts = np.full(n, np.nan)
            for snap in available:
                grp = f.get('Snap_%d' % snap) or f['Core_0']['Snap_%d' % snap]
                z = grp.attrs.get('redshift')
                if z is not None:
                    redshifts[snap] = float(z)

    return redshifts, available


def nearest_snapshot(z_target, redshifts, available):
    """Snapshot whose redshift is closest to z_target."""
    usable = [s for s in available
              if s < len(redshifts) and np.isfinite(redshifts[s])]
    if not usable:
        return None
    return min(usable, key=lambda s: abs(redshifts[s] - z_target))


def build_snapshot_map(redshifts, available):
    """Constraint-to-snapshot mapping resolved against a real output file.

    Each constraint is given the snapshot closest to the redshift its
    observations were measured at, so the mapping is correct for any simulation
    without a hard-coded table.
    """
    mapping = {}

    for name, z_target in CONSTRAINT_TARGET_REDSHIFT.items():
        snap = nearest_snapshot(z_target, redshifts, available)
        if snap is not None:
            mapping[name] = [snap]

    for name, (z_lo, z_hi, count) in CONSTRAINT_TARGET_REDSHIFT_RANGE.items():
        snaps = []
        for z_target in np.linspace(z_lo, z_hi, count):
            snap = nearest_snapshot(z_target, redshifts, available)
            if snap is not None and snap not in snaps:
                snaps.append(snap)
        if snaps:
            mapping[name] = sorted(snaps)

    return mapping


def describe_snapshot_map(mapping, redshifts):
    """Human-readable lines for a resolved snapshot map, for the run log."""
    lines = []
    for name in sorted(mapping):
        snaps = mapping[name]
        target = CONSTRAINT_TARGET_REDSHIFT.get(name)
        if len(snaps) == 1:
            z = redshifts[snaps[0]]
            note = '' if target is None else '  (observations at z = %.2f)' % target
            lines.append('%-12s snap %-4d z = %.4f%s' % (name, snaps[0], z, note))
        else:
            zs = [redshifts[s] for s in snaps]
            lines.append('%-12s %2d snapshots, z = %.3f to %.3f'
                         % (name, len(snaps), min(zs), max(zs)))
    return lines

# Reference-CSV epochs, also resolved from the output rather than tabulated.
# These drive the sage_*.csv files the diagnostics overlay, so they only need to
# span the range the constraints use.
# These drive the column layout of sage_smf_all_redshifts.csv and friends, and
# the SMF_z* constraints' get_sage_x_y() reads fixed column PAIRS out of it
# (SMF_z20 reads pair 6, SMF_z30 pair 8, SMF_z40 pair 10).  Shortening this list
# silently puts those indices out of range, so it has to stay 11 epochs in this
# order until get_sage_x_y is taught to look up its own column.
REFERENCE_TARGET_REDSHIFTS = [0.0, 0.18, 0.51, 0.76, 1.08, 1.50,
                              2.07, 2.42, 3.06, 3.58, 4.18]
HISTORY_TARGET_REDSHIFTS = (0.0, 8.0, 17)


def resolve_target_snapshots(redshifts, available,
                             targets=REFERENCE_TARGET_REDSHIFTS):
    """Snapshots nearest a list of target redshifts, in target order."""
    out = []
    for z in targets:
        snap = nearest_snapshot(z, redshifts, available)
        if snap is not None:
            out.append(snap)
    return out


def resolve_history_snapshots(redshifts, available,
                              span=HISTORY_TARGET_REDSHIFTS):
    """Snapshots spanning a redshift range, low z first, duplicates removed."""
    z_lo, z_hi, count = span
    out = []
    for z in np.linspace(z_lo, z_hi, count):
        snap = nearest_snapshot(z, redshifts, available)
        if snap is not None and snap not in out:
            out.append(snap)
    return sorted(out)


def identify_simulation(n_snapshots):
    """Best-guess simulation id from the number of snapshots in the output.

    Only used for the legacy fallback path and for labelling: 50 snapshots is
    the Uchuu grid, 64 the Millennium grid, 100 MTNG.  Everything that matters
    is now resolved from the output's own redshift table, so a wrong guess here
    changes nothing except a log line.
    """
    return {50: SIM_MINIUCHUU, 64: SIM_MINIMILLENNIUM,
            100: SIM_MTNG}.get(int(n_snapshots))
