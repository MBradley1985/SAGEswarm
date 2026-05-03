#!/usr/bin/env python3
"""
Unit tests for constraint observation-loading methods.

Tests that get_obs_x_y_err() returns valid, physically reasonable data
for each active constraint. No SAGE model output required.

Usage:
    python3 tests/test_constraint_data.py
    python3 -m pytest tests/test_constraint_data.py -v  (if pytest installed)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.constraints import SMF_z0, SMF_z05, SMF_z10, SMF_z20, SMF_z30, SMF_z40
from src.constraints import BHMF_z0, BHMF_z10, BHBM, HIMF, MZR, SHMR

# miniMillennium parameters
SIM_PARAMS = dict(sim=1, boxsize=62.5, vol_frac=1.0, h0=0.73, Omega0=0.25,
                  age_alist_file=None)

CONSTRAINTS = [
    ('SMF_z0',   SMF_z0,   {'domain': (8.5, 12.0), 'y_range': (-7, 0)}),
    ('SMF_z05',  SMF_z05,  {'domain': (8.5, 12.0), 'y_range': (-7, 0)}),
    ('SMF_z10',  SMF_z10,  {'domain': (8.5, 12.0), 'y_range': (-7, 0)}),
    ('SMF_z20',  SMF_z20,  {'domain': (8.5, 12.0), 'y_range': (-7, 0)}),
    ('SMF_z30',  SMF_z30,  {'domain': (8.5, 12.0), 'y_range': (-7, 0)}),
    ('SMF_z40',  SMF_z40,  {'domain': (8.5, 12.0), 'y_range': (-7, 0)}),
    ('BHMF_z0',  BHMF_z0,  {'domain': (7.0, 10.5), 'y_range': (-9, 0)}),
    ('BHMF_z10', BHMF_z10, {'domain': (7.0, 10.5), 'y_range': (-9, 0)}),
    ('BHBM',     BHBM,     {'domain': (9.0, 12.0), 'y_range': (4, 12)}),
    ('HIMF',     HIMF,     {'domain': (8.0, 10.75), 'y_range': (-7, 0)}),
    ('MZR',      MZR,      {'domain': (8.0, 11.0), 'y_range': (7.0, 10.0)}),
    ('SHMR',     SHMR,     {'domain': (11.0, 15.0), 'y_range': (6, 13)}),
]


def check_constraint(name, cls, expected):
    errors = []

    try:
        c = cls(snapshot=63, **SIM_PARAMS)
        x_obs, y_obs, y_dn, y_up = c.get_obs_x_y_err()
    except Exception as e:
        return [f"get_obs_x_y_err() raised {type(e).__name__}: {e}"]

    # Must return numpy arrays
    for arr_name, arr in [('x_obs', x_obs), ('y_obs', y_obs), ('y_dn', y_dn), ('y_up', y_up)]:
        if not isinstance(arr, np.ndarray):
            errors.append(f"{arr_name} is not ndarray (got {type(arr).__name__})")

    if errors:
        return errors

    # Must have at least 3 data points
    n = len(x_obs)
    if n < 3:
        errors.append(f"only {n} data points (need >= 3)")

    # All arrays must be the same length
    if not (len(y_obs) == n and len(y_dn) == n and len(y_up) == n):
        errors.append(f"array length mismatch: x={n} y={len(y_obs)} dn={len(y_dn)} up={len(y_up)}")

    # No NaNs or Infs
    for arr_name, arr in [('x_obs', x_obs), ('y_obs', y_obs), ('y_dn', y_dn), ('y_up', y_up)]:
        if not np.all(np.isfinite(arr)):
            n_bad = np.sum(~np.isfinite(arr))
            errors.append(f"{arr_name} has {n_bad} non-finite values")

    # Errors must be non-zero (sign convention varies — some datasets store signed errors)
    if np.any(np.abs(y_dn) == 0):
        errors.append(f"y_dn has zero values")
    if np.any(np.abs(y_up) == 0):
        errors.append(f"y_up has zero values")

    # At least some x values must overlap with the expected domain.
    # Raw data may extend outside the domain — get_data() applies domain filtering.
    dom = expected['domain']
    in_domain = (x_obs >= dom[0]) & (x_obs <= dom[1])
    if np.sum(in_domain) < 3:
        errors.append(f"fewer than 3 x_obs points within domain {dom} "
                      f"(got {np.sum(in_domain)}, x range: [{x_obs.min():.2f}, {x_obs.max():.2f}])")

    # y values must be in physically plausible range
    yr = expected['y_range']
    if np.any(y_obs < yr[0] - 2.0) or np.any(y_obs > yr[1] + 2.0):
        errors.append(f"y_obs out of expected range {yr}: [{y_obs.min():.2f}, {y_obs.max():.2f}]")

    # MZR specific: solar metallicity anchor check (after fix, solar → 8.69)
    if name == 'MZR':
        plateau = np.max(y_obs)
        if plateau > 9.5:
            errors.append(f"MZR plateau {plateau:.2f} > 9.5 — likely wrong +9.0 formula (should be +8.69)")
        if plateau < 8.0:
            errors.append(f"MZR plateau {plateau:.2f} < 8.0 — suspiciously low")

    # SMF specific: phi values should be in log space (negative values)
    if name.startswith('SMF') or name.startswith('BHMF') or name == 'HIMF':
        if np.any(y_obs > 1.0):
            errors.append(f"y_obs has values > 1.0 — may not be in log space")

    return errors


def run_all():
    print("=" * 65)
    print("Constraint observation data unit tests")
    print("=" * 65)

    passed, failed = 0, 0
    for name, cls, expected in CONSTRAINTS:
        errors = check_constraint(name, cls, expected)
        if errors:
            print(f"  FAIL  {name}")
            for e in errors:
                print(f"        - {e}")
            failed += 1
        else:
            print(f"  PASS  {name}")
            passed += 1

    print("=" * 65)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} constraints")
    print("=" * 65)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
