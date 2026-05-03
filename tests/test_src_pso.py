#!/usr/bin/env python3
"""
Tests for src/pso.py features not covered by the benchmark tests:

  1. Batch interface (processes=0) — func receives all particles as (S,D) matrix
  2. is_log — PSO searches in log-space; returned positions must be physical
  3. max_stagnation — terminates after N non-improving iterations, not maxiter
  4. is_log + batch combined — what run_sage_hpc actually uses

Known analytic optima are used throughout so failures are unambiguous.

Usage:
    python3 tests/test_src_pso.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pso import pso

PASS, FAIL = '  PASS', '  FAIL'


# ---------------------------------------------------------------------------
# Test 1: Batch interface (processes=0)
# func(x, *args) receives x as (S, D) array, must return (S,) array.
# Minimum of sum-of-squares at origin.
# ---------------------------------------------------------------------------

def test_batch_interface():
    """Batch mode: function receives all particles, returns array of scores."""

    def batch_sphere(particles):
        # particles is (S, D); return (S,) of scores
        assert particles.ndim == 2, f"Expected 2D array, got shape {particles.shape}"
        return np.sum(particles**2, axis=1)

    lb = [-5.0, -5.0]
    ub = [5.0, 5.0]
    best_pos, best_val = pso(batch_sphere, lb, ub,
                             swarmsize=20, maxiter=100, processes=0,
                             omega=0.5, phip=0.5, phig=0.5,
                             random_seed=42, debug=False, max_stagnation=20)

    ok = best_val < 0.01
    return ok, f"f={best_val:.4f} at {np.round(best_pos, 4)} (want f≈0 at [0,0])"


# ---------------------------------------------------------------------------
# Test 2: is_log — log-space search, physical-space output
#
# The PSO receives log-space bounds and searches in log-space.
# (main.py converts bounds to log before calling pso(), matching production.)
#
# Setup:
#   - 2D: dim 0 log-space, dim 1 linear
#   - Bounds passed to pso() already in log-space for dim 0: [-2, 1] (phys 0.01–10)
#   - True minimum: log-space (-1, 3.0) → physical (0.1, 3.0)
#   - Function takes log-space x[0], linear x[1]
# Expected: returned position is physical [0.1, 3.0], NOT log-space [-1, 3.0]
# ---------------------------------------------------------------------------

def test_is_log_output_is_physical():
    """is_log: returned best position must be in physical space."""

    # Minimum in log-space at (-1, 3.0); function takes log-space x[0]
    def func(x):
        return (x[0] - (-1.0))**2 + (x[1] - 3.0)**2

    is_log = np.array([True, False])
    lb = np.array([-2.0, 1.0])   # log-space lb for dim 0, linear for dim 1
    ub = np.array([1.0, 5.0])

    best_pos, best_val = pso(func, lb, ub,
                             swarmsize=20, maxiter=150,
                             omega=0.5, phip=0.5, phig=0.5,
                             is_log=is_log, random_seed=7,
                             debug=False, max_stagnation=30)

    # best_pos[0] should be physical 10^(-1) = 0.1, not log-space -1
    physical_x0_expected = 0.1
    physical_x1_expected = 3.0

    x0_ok = abs(best_pos[0] - physical_x0_expected) < 0.05
    x1_ok = abs(best_pos[1] - physical_x1_expected) < 0.05
    not_logspace = abs(best_pos[0] - (-1.0)) > 0.5   # returned -1 would be wrong

    ok = x0_ok and x1_ok and not_logspace
    return ok, (f"pos={np.round(best_pos, 4)} f={best_val:.4f} "
                f"(want [{physical_x0_expected}, {physical_x1_expected}]); "
                f"x0_ok={x0_ok}, x1_ok={x1_ok}, not_in_logspace={not_logspace}")


# ---------------------------------------------------------------------------
# Test 3: max_stagnation termination
#
# A constant function can never improve → PSO must stop after exactly
# max_stagnation iterations, not maxiter.
# Batch call count = 1 (init) + max_stagnation (iterations) exactly.
# ---------------------------------------------------------------------------

def test_max_stagnation():
    """max_stagnation: PSO stops after N non-improving iterations."""

    MAX_STAGNATION = 5
    MAXITER = 100
    SWARMSIZE = 10

    call_count = [0]

    def batch_constant(particles):
        call_count[0] += 1
        return np.full(len(particles), 42.0)

    lb = [-1.0, -1.0]
    ub = [1.0, 1.0]
    best_pos, best_val = pso(batch_constant, lb, ub,
                             swarmsize=SWARMSIZE, maxiter=MAXITER,
                             processes=0, is_log=None,
                             max_stagnation=MAX_STAGNATION,
                             random_seed=0, debug=False)

    # Expected: 1 init call + MAX_STAGNATION iteration calls = MAX_STAGNATION + 1
    expected_calls = MAX_STAGNATION + 1
    calls_ok = call_count[0] == expected_calls
    val_ok = abs(best_val - 42.0) < 1e-9
    terminated_early = call_count[0] < MAXITER  # definitely didn't run to maxiter

    ok = calls_ok and val_ok and terminated_early
    return ok, (f"batch calls={call_count[0]} (expected {expected_calls}), "
                f"f={best_val}, early_stop={terminated_early}")


# ---------------------------------------------------------------------------
# Test 4: is_log + batch interface combined
#
# This is the exact mode used by run_sage_hpc (processes=0, is_log set).
# Minimum of a log-sphere at physical [0.01, 100.0]:
#   f(x) = sum((log10(x) - log10(x*))^2) in log-space
# Bounds already in log-space (matching main.py convention):
#   dim 0: [-4, 0]  (physical 1e-4 to 1)
#   dim 1: [0, 4]   (physical 1 to 1e4)
# True minimum: log-space (-2, 2) → physical (0.01, 100)
# ---------------------------------------------------------------------------

def test_is_log_batch_combined():
    """is_log + batch interface: the production mode used by run_sage_hpc."""

    log_optimum = np.array([-2.0, 2.0])  # physical: [0.01, 100]

    def batch_log_sphere(particles):
        # particles is (S, D) in log-space; return (S,) of scores
        assert particles.ndim == 2
        return np.sum((particles - log_optimum)**2, axis=1)

    is_log = np.array([True, True])
    lb = np.array([-4.0, 0.0])  # log-space bounds (as main.py passes them)
    ub = np.array([0.0, 4.0])

    best_pos, best_val = pso(batch_log_sphere, lb, ub,
                             swarmsize=20, maxiter=150,
                             processes=0, is_log=is_log,
                             omega=0.5, phip=0.5, phig=0.5,
                             random_seed=99, debug=False, max_stagnation=30)

    # best_pos must be physical: [10^-2, 10^2] = [0.01, 100]
    physical_expected = np.array([0.01, 100.0])
    pos_err = np.abs(best_pos - physical_expected)

    x0_ok = pos_err[0] < 0.005
    x1_ok = pos_err[1] < 5.0
    ok = x0_ok and x1_ok and best_val < 0.01

    return ok, (f"pos={np.round(best_pos, 4)} f={best_val:.6f} "
                f"(want {physical_expected}); x0_ok={x0_ok}, x1_ok={x1_ok}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Batch interface (processes=0)",          test_batch_interface),
    ("is_log: output is physical not log",     test_is_log_output_is_physical),
    ("max_stagnation terminates early",        test_max_stagnation),
    ("is_log + batch combined (production)",   test_is_log_batch_combined),
]


def run_all():
    print("=" * 65)
    print("src/pso.py feature tests")
    print("=" * 65)

    passed, failed = 0, 0
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"raised {type(e).__name__}: {e}"

        status = PASS if ok else FAIL
        print(f"{status}  {name}")
        if not ok:
            print(f"        {detail}")
        else:
            print(f"        {detail}")
        passed += ok
        failed += not ok

    print("=" * 65)
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 65)
    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
