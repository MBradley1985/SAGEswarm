#!/usr/bin/env python3
"""
Unit tests for the goodness-of-fit statistics.

These are the objective function: a defect here does not crash anything, it
just optimises the wrong thing.  Two have bitten already -- studentT returned
NaN over the whole well-fitting regime, and npsum silently dropped keyword
arguments so cash's count_scale was unreachable.

Usage:
    python3 tests/test_stat_tests.py
    python3 -m pytest tests/test_stat_tests.py -v  (if pytest installed)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import analysis

SCALE = 0.1 * (100.0 / 0.6774) ** 3     # dm * V for a 100/h Mpc box


def _series(n=20, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.linspace(-4, -1, n)
    err = np.full(n, 0.15)
    mod = obs + offset * err * rng.standard_normal(n)
    return obs, mod, err


# ── every test must be finite and monotonic in fit quality ──────────────────

def test_all_finite_across_fit_quality():
    """No statistic may return NaN or inf for any fit quality.

    studentT used to return NaN for every fit better than the errors, which
    made PSO reject good particles outright.
    """
    bad = []
    for name, fn in analysis.stat_tests.items():
        for offset in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
            obs, mod, err = _series(offset=offset)
            kw = {'count_scale': SCALE} if getattr(fn, 'needs_count_scale', False) else {}
            v = fn(obs, mod, err, **kw)
            if not np.isfinite(v):
                bad.append('%s at offset %g -> %r' % (name, offset, v))
    assert not bad, bad
    return 'all %d statistics finite over 8 fit qualities' % len(analysis.stat_tests)


def test_all_monotonic_in_fit_quality():
    """A worse fit must never score better."""
    bad = []
    for name, fn in analysis.stat_tests.items():
        kw = {'count_scale': SCALE} if getattr(fn, 'needs_count_scale', False) else {}
        obs = np.linspace(-4, -1, 20)
        err = np.full(20, 0.15)
        prev = None
        for offset in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            mod = obs + offset * err          # deterministic, same sign
            v = fn(obs, mod, err, **kw)
            if prev is not None and v < prev - 1e-9:
                bad.append('%s: %g -> %g at offset %g' % (name, prev, v, offset))
            prev = v
    assert not bad, bad
    return 'all statistics increase monotonically as the fit worsens'


def test_perfect_fit_is_minimal():
    """A perfect fit must be the minimum, and zero where that is meaningful."""
    obs = np.linspace(-4, -1, 20)
    err = np.full(20, 0.15)
    results = {}
    for name, fn in analysis.stat_tests.items():
        kw = {'count_scale': SCALE} if getattr(fn, 'needs_count_scale', False) else {}
        perfect = fn(obs, obs, err, **kw)
        worse = fn(obs, obs + err, err, **kw)
        assert perfect <= worse + 1e-12, '%s: perfect %g > offset %g' % (name, perfect, worse)
        results[name] = perfect
    for name in ('chi2', 'huber', 'cauchy', 'abs', 'cash'):
        assert abs(results[name]) < 1e-6, '%s perfect fit = %g, expected 0' % (name, results[name])
    return 'perfect fit minimal for all; exactly zero for %s' % (
        ', '.join(n for n in results if abs(results[n]) < 1e-6))


# ── robustness ordering ─────────────────────────────────────────────────────

def test_outlier_cost_ordering():
    """Robust statistics must charge less for an outlier than chi2 does."""
    n = 20
    obs = np.linspace(-4, -1, n)
    err = np.full(n, 0.15)
    mod = obs.copy()
    mod[7] += 10 * err[7]          # one 10-sigma point, rest perfect

    c = analysis.chi2(obs, mod, err)
    h = analysis.huber(obs, mod, err)
    t = analysis.studentT(obs, mod, err)
    q = analysis.cauchy(obs, mod, err)

    assert q < t, 'cauchy (%g) should be cheaper than student-t (%g)' % (q, t)
    assert h < c, 'huber (%g) should be cheaper than chi2 (%g)' % (h, c)
    assert q < h, 'cauchy (%g) should be cheaper than huber (%g)' % (q, h)
    return 'cauchy %.2f < huber %.2f < chi2 %.2f (10-sigma outlier)' % (q, h, c)


def test_huber_matches_chi2_for_good_points():
    """Huber is exactly chi2 inside delta, which is the point of it."""
    obs = np.linspace(-4, -1, 15)
    err = np.full(15, 0.15)
    mod = obs + 1.0 * err          # 1 sigma, well inside delta = 2
    assert abs(analysis.huber(obs, mod, err) - analysis.chi2(obs, mod, err)) < 1e-9
    return 'huber == chi2 for residuals within 2 sigma'


# ── Student-t degrees of freedom ────────────────────────────────────────────

def test_student_dof_never_invalid():
    """The dof estimate must stay in the valid range for any residual variance.

    2*var/(var-1) is negative for var < 1 -- the regime a calibration is trying
    to reach -- and that produced gamma() of a negative argument and a NaN.
    """
    for var in (1e-8, 0.01, 0.5, 0.999, 1.0, 1.001, 2.0, 100.0, 1e6):
        nu = float(analysis.student_dof(var))
        assert analysis.STUDENT_NU_MIN <= nu <= analysis.STUDENT_NU_MAX, (var, nu)
    # continuous across var = 1: both sides go to the Gaussian limit
    assert analysis.student_dof(0.999) == analysis.STUDENT_NU_MAX
    assert float(analysis.student_dof(1.001)) > 100
    return 'dof clamped to [%.1f, %.0f] for all variances' % (
        analysis.STUDENT_NU_MIN, analysis.STUDENT_NU_MAX)


# ── Cash / Poisson ─────────────────────────────────────────────────────────

def test_cash_requires_count_scale():
    obs = np.linspace(-4, -1, 5)
    try:
        analysis.cash(obs, obs, np.full(5, 0.1))
    except ValueError as e:
        assert 'count_scale' in str(e)
        return 'raises without count_scale: %s' % str(e)[:56]
    raise AssertionError('cash accepted a call with no count_scale')


def test_npsum_forwards_keyword_arguments():
    """The decorator must not drop kwargs, or cash and huber lose their options."""
    obs = np.linspace(-4, -1, 5)
    err = np.full(5, 0.1)
    # distinct deltas must give distinct huber values for a 3-sigma residual
    mod = obs + 3.0 * err
    assert analysis.huber(obs, mod, err, delta=1.0) != analysis.huber(obs, mod, err, delta=5.0)
    # and cash must actually see its scale
    assert analysis.cash(obs, obs - 0.1, err, count_scale=1e3) != \
        analysis.cash(obs, obs - 0.1, err, count_scale=1e5)
    return 'huber delta and cash count_scale both reach the function'


def test_cash_weights_by_object_count():
    """Cash must weight a bin by how many objects it holds, unlike chi2.

    This is the whole reason for it: the same 0.1 dex discrepancy is highly
    significant in a 10^4-galaxy bin and meaningless in a 1-galaxy bin, and
    chi2 on log10(phi) cannot tell the difference.
    """
    terms = []
    for N in (1e4, 1e3, 1e2, 10, 1):
        obs = np.array([np.log10(N / SCALE)])
        mod = obs - 0.1
        err = np.array([0.15])
        terms.append(float(analysis.cash(obs, mod, err, count_scale=SCALE)))
        chi_term = float(analysis.chi2(obs, mod, err))
    # strictly decreasing with count
    assert all(a > b for a, b in zip(terms, terms[1:])), terms
    # chi2 by contrast is identical for every one of them
    chi = [float(analysis.chi2(np.array([np.log10(N / SCALE)]),
                               np.array([np.log10(N / SCALE)]) - 0.1,
                               np.array([0.15]))) for N in (1e4, 1e3, 1e2, 10, 1)]
    assert max(chi) - min(chi) < 1e-9, chi
    return 'cash %.1f -> %.2f across N = 1e4 -> 1; chi2 flat at %.3f' % (
        terms[0], terms[-1], chi[0])


def test_cash_handles_empty_model_bins():
    """N = 0 must contribute mu, not NaN: the N ln N term vanishes."""
    obs = np.array([np.log10(100.0 / SCALE)])
    mod = np.array([-30.0])                  # phi -> 0, so N -> 0
    v = float(analysis.cash(obs, mod, np.array([0.15]), count_scale=SCALE))
    assert np.isfinite(v) and v > 0, v
    assert abs(v - 2.0 * 100.0) < 1.0, 'expected ~2*mu = 200, got %g' % v
    return 'empty model bin costs 2*mu = %.1f, finite' % v


TESTS = [
    ('all statistics finite',                 test_all_finite_across_fit_quality),
    ('all statistics monotonic',              test_all_monotonic_in_fit_quality),
    ('perfect fit is minimal',                test_perfect_fit_is_minimal),
    ('outlier cost ordering',                 test_outlier_cost_ordering),
    ('huber == chi2 inside delta',            test_huber_matches_chi2_for_good_points),
    ('student-t dof never invalid',           test_student_dof_never_invalid),
    ('cash requires count_scale',             test_cash_requires_count_scale),
    ('npsum forwards kwargs',                 test_npsum_forwards_keyword_arguments),
    ('cash weights by object count',          test_cash_weights_by_object_count),
    ('cash handles empty model bins',         test_cash_handles_empty_model_bins),
]


def run_all():
    print('=' * 70)
    print('Goodness-of-fit statistic unit tests')
    print('=' * 70)
    passed = failed = 0
    for label, fn in TESTS:
        try:
            detail = fn()
            print('  PASS  %s' % label)
            if detail:
                print('        %s' % detail)
            passed += 1
        except AssertionError as e:
            print('  FAIL  %s' % label); print('        %s' % e); failed += 1
        except Exception as e:
            print('  FAIL  %s' % label); print('        %s: %s' % (type(e).__name__, e)); failed += 1
    print('=' * 70)
    print('Results: %d passed, %d failed out of %d tests' % (passed, failed, passed + failed))
    print('=' * 70)
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
