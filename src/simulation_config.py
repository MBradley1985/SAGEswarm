#!/bin/bash

"""
Simulation configuration module for SAGE-PSO.
Contains simulation-specific parameters including snapshots, redshifts, and cosmology.
"""

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
        }
    elif sim_id == SIM_MINIMILLENNIUM:
        # miniMillennium: 64 snapshots (0-63), snapshot 63 is z=0
        return {
            'SMF_z0': [63],
            'SMF_z05': [48],
            'SMF_z10': [40],
            'SMF_z20': [32],
            'SMF_z30': [27],
            'SMF_z40': [23],
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
        # Millennium snapshots for z = [0, 0.5, 1.0, 2.0, 3.0, 4.0]
        return [63, 48, 40, 32, 27, 23]
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
