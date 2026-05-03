#!/usr/bin/env python3
"""
Constraint consistency diagnostic.

Two modes:

1. OBS-ONLY (no SAGE output needed)
   Plots observational data for all active constraints side by side.
   If any obs dataset looks pathological (wrong scale, implausible range)
   that flags a bug before running a single model.

   Usage:
       python3 tests/test_constraint_consistency.py --obs-only

2. MODEL vs OBS (requires a SAGE output directory)
   Loads a model output directory from a previous PSO evaluation and
   computes the reduced chi² for each constraint individually.
   A well-calibrated SAGE model should give chi² ≈ 1 per constraint.
   chi² >> 1 for a specific constraint → that constraint is buggy
   or genuinely inconsistent with the model.

   Usage:
       python3 tests/test_constraint_consistency.py \
           --model-dir /path/to/model/output \
           --subvols 0 1 2 3 \
           --sim 1 --boxsize 62.5 --vol-frac 1.0 --h0 0.73 --omega0 0.25

The obs-only mode runs locally. The model mode needs HPC model output.
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.constraints import (
    SMF_z0, SMF_z05, SMF_z10, SMF_z20, SMF_z30, SMF_z40,
    BHMF_z0, BHMF_z10, BHBM, HIMF, MZR, SHMR
)

# ── Constraint registry ────────────────────────────────────────────────────────

ALL_CONSTRAINTS = {
    'SMF_z0':   SMF_z0,
    'SMF_z05':  SMF_z05,
    'SMF_z10':  SMF_z10,
    'SMF_z20':  SMF_z20,
    'SMF_z30':  SMF_z30,
    'SMF_z40':  SMF_z40,
    'BHMF_z0':  BHMF_z0,
    'BHMF_z10': BHMF_z10,
    'BHBM':     BHBM,
    'HIMF':     HIMF,
    'MZR':      MZR,
    'SHMR':     SHMR,
}

DEFAULT_ACTIVE = ['SMF_z0', 'BHBM', 'MZR', 'HIMF']

# ── Obs-only mode ──────────────────────────────────────────────────────────────

def check_obs_sanity(sim_params):
    """Load and report key statistics for each constraint's obs data."""
    print("=" * 65)
    print("Observational data sanity check")
    print("=" * 65)

    EXPECTED = {
        'SMF_z0':   {'x_range': (8.0, 12.5), 'y_range': (-6.0, 0.0),  'label': 'log10(M*/Msun) vs log10(phi)'},
        'SMF_z05':  {'x_range': (8.0, 12.5), 'y_range': (-6.0, 0.0),  'label': 'log10(M*/Msun) vs log10(phi)'},
        'SMF_z10':  {'x_range': (8.0, 12.5), 'y_range': (-6.0, 0.0),  'label': 'log10(M*/Msun) vs log10(phi)'},
        'SMF_z20':  {'x_range': (8.0, 12.5), 'y_range': (-6.0, 0.0),  'label': 'log10(M*/Msun) vs log10(phi)'},
        'SMF_z30':  {'x_range': (8.0, 12.5), 'y_range': (-6.0, 0.0),  'label': 'log10(M*/Msun) vs log10(phi)'},
        'SMF_z40':  {'x_range': (8.0, 12.5), 'y_range': (-6.0, 0.0),  'label': 'log10(M*/Msun) vs log10(phi)'},
        # TRINITY data includes model-extrapolated low-mass BHs (log M < 7); domain filter handles this
        'BHMF_z0':  {'x_range': (4.0, 11.0), 'y_range': (-9.0, 0.0),  'label': 'log10(MBH/Msun) vs log10(phi)'},
        'BHMF_z10': {'x_range': (4.0, 11.0), 'y_range': (-9.0, 0.0),  'label': 'log10(MBH/Msun) vs log10(phi)'},
        'BHBM':     {'x_range': (9.0, 12.5), 'y_range': (5.0, 11.0),  'label': 'log10(Mbulge) vs log10(MBH)'},
        'HIMF':     {'x_range': (7.0, 11.5), 'y_range': (-5.0, 0.0),  'label': 'log10(MHI/Msun) vs log10(phi)'},
        'MZR':      {'x_range': (8.0, 12.0), 'y_range': (7.5, 9.5),   'label': 'log10(M*/Msun) vs 12+log(O/H)'},
        'SHMR':     {'x_range': (10.5, 15.5),'y_range': (6.0, 13.0),  'label': 'log10(Mhalo) vs log10(M*)'},
    }

    results = {}
    all_ok = True
    for name, cls in ALL_CONSTRAINTS.items():
        try:
            c = cls(snapshot=63, **sim_params)
            x, y, y_dn, y_up = c.get_obs_x_y_err()
            exp = EXPECTED[name]

            issues = []
            # Check x range
            if x.min() < exp['x_range'][0] - 2 or x.max() > exp['x_range'][1] + 2:
                issues.append(f"x range [{x.min():.2f}, {x.max():.2f}] outside expected {exp['x_range']}")
            # Check y range
            if y.min() < exp['y_range'][0] - 2 or y.max() > exp['y_range'][1] + 2:
                issues.append(f"y range [{y.min():.2f}, {y.max():.2f}] outside expected {exp['y_range']}")
            # Check for NaN/Inf
            for arr_name, arr in [('x', x), ('y', y), ('y_dn', y_dn), ('y_up', y_up)]:
                if not np.all(np.isfinite(arr)):
                    issues.append(f"{arr_name} has non-finite values")

            xm, xM = x.min(), x.max()
            ym, yM = y.min(), y.max()
            status = '  PASS' if not issues else '  FAIL'
            print(f"{status}  {name:<12}  x=[{xm:.2f},{xM:.2f}]  y=[{ym:.2f},{yM:.2f}]  n={len(x)}")
            if issues:
                for iss in issues:
                    print(f"         ↳ {iss}")
                all_ok = False
            results[name] = {'x': x, 'y': y, 'y_dn': y_dn, 'y_up': y_up, 'ok': not issues}

        except Exception as e:
            print(f"  FAIL  {name:<12}  ERROR: {e}")
            all_ok = False
            results[name] = {'ok': False}

    print("=" * 65)
    print(f"{'All obs data look sane' if all_ok else 'Some issues found — fix before running PSO'}")
    print("=" * 65)
    return results


# ── Model vs obs mode ──────────────────────────────────────────────────────────

def chi2_per_constraint(constraint_names, model_dir, subvols, sim_params, stat_test='chi2'):
    """
    Given a SAGE model output directory, compute chi² per constraint.

    Returns dict: {name: chi2_reduced}
    """
    try:
        from src.execution import _evaluate
        from src import analysis
    except ImportError as e:
        print(f"ERROR: could not import required module: {e}")
        return {}

    stat_fn = analysis.stat_tests.get(stat_test)
    if stat_fn is None:
        print(f"ERROR: unknown stat_test '{stat_test}'. Choose from: {list(analysis.stat_tests.keys())}")
        return {}

    print("=" * 65)
    print(f"Per-constraint chi² — model dir: {model_dir}")
    print("=" * 65)

    scores = {}
    for name in constraint_names:
        cls = ALL_CONSTRAINTS.get(name)
        if cls is None:
            print(f"  SKIP  {name}  (not in registry)")
            continue
        try:
            c = cls(snapshot=63, **sim_params)
            chi2_val = _evaluate(c, stat_fn, model_dir, subvols)
            scores[name] = chi2_val
            flag = ''
            if chi2_val > 5:
                flag = '  ← HIGH (buggy or inconsistent?)'
            elif chi2_val > 2:
                flag = '  ← elevated'
            print(f"  chi²={chi2_val:7.3f}  {name}{flag}")
        except Exception as e:
            print(f"  ERROR  {name}: {e}")
            scores[name] = float('nan')

    if scores:
        total = sum(v for v in scores.values() if np.isfinite(v))
        print(f"\n  Total score = {total:.4f}")
        print("\n  Interpretation:")
        print("  chi² ≈ 1.0  → model matches this constraint well")
        print("  chi² >> 1   → model fails this constraint")
        print("                (check bug fixes first; then consider constraint inconsistency)")
    print("=" * 65)
    return scores


# ── Obs-vs-obs internal consistency checks ────────────────────────────────────

def cross_check_obs(sim_params):
    """
    Check that SMF + BHMF + BHBM are mutually consistent:

    If phi_BH(MBH) and phi_*(M*) are both given, and BHBM gives MBH(Mbulge),
    we can estimate the implied BH mass density and compare to the BHMF.

    Also check that MZR plateau (< 9.0) is consistent with solar metallicity.
    """
    print("\n" + "=" * 65)
    print("Cross-constraint obs consistency checks")
    print("=" * 65)

    # ── MZR plateau check ──────────────────────────────────────────
    try:
        c = MZR(snapshot=63, **sim_params)
        x, y, _, _ = c.get_obs_x_y_err()
        plateau = np.max(y)
        solar_anchor = 8.69  # Asplund+2009
        ok = (8.0 < plateau < 9.5)
        flag = '  PASS' if ok else '  FAIL'
        print(f"{flag}  MZR plateau = {plateau:.2f}  (solar anchor = {solar_anchor}, expect 8.5–9.0)")
        if plateau > 9.0:
            print("       ↳ Plateau > 9.0 suggests wrong solar metallicity convention (should be 8.69, not 9.0)")
    except Exception as e:
        print(f"  ERROR  MZR: {e}")

    # ── SMF knee consistency ───────────────────────────────────────
    # All SMF redshift bins should have similar or decreasing M* with z
    smf_classes = [('z=0', SMF_z0), ('z=0.5', SMF_z05), ('z=1', SMF_z10), ('z=2', SMF_z20)]
    print("\n  SMF characteristic mass by redshift (should decrease with z):")
    prev_mstar = None
    for label, cls in smf_classes:
        try:
            c = cls(snapshot=63, **sim_params)
            x, y, _, _ = c.get_obs_x_y_err()
            # Rough M* estimate: mass at knee (where phi changes slope)
            # Approximate by median of highest-mass points with y > -4
            mask = y > -4
            if np.sum(mask) > 0:
                mstar_approx = np.median(x[mask])
                note = ''
                if prev_mstar is not None and mstar_approx > prev_mstar + 0.3:
                    note = '  ← increasing with z (unexpected!)'
                print(f"    {label}: M*_approx ≈ {mstar_approx:.2f}{note}")
                prev_mstar = mstar_approx
        except Exception as e:
            print(f"    {label}: ERROR {e}")

    # ── HIMF vs SMF check (HI fraction) ────────────────────────────
    # At M* ~ 10^10 Msun, typical HI fraction is 0.1-1
    # If HIMF peaks at MHI >> M*, that's suspicious
    try:
        c_smf = SMF_z0(snapshot=63, **sim_params)
        c_himf = HIMF(snapshot=63, **sim_params)
        x_smf, y_smf, _, _ = c_smf.get_obs_x_y_err()
        x_himf, y_himf, _, _ = c_himf.get_obs_x_y_err()
        # SMF and HIMF peaks should be within ~1 dex of each other in mass
        # Both are mass functions; their peaks should be in the 9-11 Msun range
        smf_peak = x_smf[np.argmax(y_smf)]
        himf_peak = x_himf[np.argmax(y_himf)]
        offset = abs(smf_peak - himf_peak)
        ok = offset < 2.5
        flag = '  PASS' if ok else '  WARN'
        print(f"\n{flag}  SMF peak mass ≈ {smf_peak:.2f}, HIMF peak mass ≈ {himf_peak:.2f}  (diff = {offset:.2f} dex)")
        if offset > 2.5:
            print("       ↳ Large offset may indicate h-factor mismatch between SMF and HIMF")
    except Exception as e:
        print(f"  ERROR  SMF/HIMF comparison: {e}")

    print("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--obs-only', action='store_true',
                        help='Only check observational data (no model needed)')
    parser.add_argument('--model-dir', default=None,
                        help='Path to SAGE model output directory')
    parser.add_argument('--subvols', nargs='+', type=int, default=[0],
                        help='Subvolume indices (default: 0)')
    parser.add_argument('--constraints', nargs='+', default=DEFAULT_ACTIVE,
                        help=f'Constraints to evaluate (default: {DEFAULT_ACTIVE})')
    parser.add_argument('--sim', type=int, default=1,
                        help='Simulation ID (0=miniUchuu, 1=miniMillennium)')
    parser.add_argument('--boxsize', type=float, default=62.5,
                        help='Box size in h^-1 Mpc (default: 62.5 for miniMillennium)')
    parser.add_argument('--vol-frac', type=float, default=1.0,
                        help='Volume fraction (default: 1.0)')
    parser.add_argument('--h0', type=float, default=0.73,
                        help='Hubble parameter (default: 0.73 for Millennium)')
    parser.add_argument('--omega0', type=float, default=0.25,
                        help='Omega_matter (default: 0.25 for Millennium)')
    parser.add_argument('--age-alist', default=None,
                        help='Path to age/alist file (optional)')
    args = parser.parse_args()

    sim_params = dict(
        sim=args.sim, boxsize=args.boxsize, vol_frac=args.vol_frac,
        h0=args.h0, Omega0=args.omega0, age_alist_file=args.age_alist
    )

    check_obs_sanity(sim_params)
    cross_check_obs(sim_params)

    if not args.obs_only and args.model_dir:
        chi2_per_constraint(args.constraints, args.model_dir, args.subvols, sim_params)
    elif not args.obs_only and args.model_dir is None:
        print("\nTip: pass --model-dir /path/to/sage/output to compute per-constraint chi²")


if __name__ == '__main__':
    main()
