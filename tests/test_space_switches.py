#!/usr/bin/env python3
"""
Unit tests for integer-switch support in the search space.

Covers sampling code 2, which lets PSO vary a SAGE parameter that its reader
registers as INT (DynamicDisruptionSplit, ConcentrationOn, ...).  The failure
mode being guarded against is severe: SAGE parses those tags with strtol and
aborts the entire run on a leftover character, so a continuous particle value
like 1.37 kills the optimisation rather than scoring badly.

Usage:
    python3 tests/test_space_switches.py
    python3 -m pytest tests/test_space_switches.py -v  (if pytest installed)
"""

import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import analysis, execution, pso


def write_space(content):
    fd, path = tempfile.mkstemp(suffix='.txt', text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def load_quiet(path):
    """load_space without the SAGE_INT_PARAMS advisory warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return analysis.load_space(path)


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_sampling_codes_expand():
    """Codes 0/1/2 expand into separate is_log and is_int flags."""
    path = write_space('FeedbackReheatingEpsilon,eDisk,0,0.5,5.0\n'
                       'SfrEfficiency,aSF,1,0.005,0.2\n'
                       'ConcentrationOn,cOn,2,0,3\n')
    space = load_quiet(path)
    assert list(space['is_log']) == [0, 1, 0], list(space['is_log'])
    assert list(space['is_int']) == [0, 0, 1], list(space['is_int'])
    return 'is_log=%s is_int=%s' % (list(space['is_log']), list(space['is_int']))


def test_existing_space_files_unchanged():
    """Every space file in the repo still parses, with no parameter marked int."""
    root = os.path.join(os.path.dirname(__file__), '..')
    checked = []
    for name in sorted(os.listdir(root)):
        if not (name.startswith('space') and name.endswith('.txt')):
            continue
        space = load_quiet(os.path.join(root, name))
        lb, ub = analysis.pso_bounds(space)
        assert len(lb) == len(space) and np.all(np.isfinite(lb))
        assert np.all(np.isfinite(ub)) and np.all(ub > lb)
        checked.append('%s(n=%d)' % (name, len(space)))
    assert checked, 'no space files found'
    return ', '.join(checked)


def test_switch_bounds_widened_by_half_step():
    """A switch is searched over [lb-0.5, ub+0.5] so levels have equal width."""
    path = write_space('DynamicDisruptionSplit,split,2,0,2\n')
    space = load_quiet(path)
    lb, ub = analysis.pso_bounds(space)
    assert (lb[0], ub[0]) == (-0.5, 2.5), (lb[0], ub[0])

    rng = np.random.default_rng(0)
    draws = rng.uniform(lb[0], ub[0], 200000)
    levels = np.clip(np.rint(draws), 0, 2).astype(int)
    freq = np.bincount(levels, minlength=3) / len(levels)
    assert np.allclose(freq, 1 / 3, atol=0.01), freq
    return 'pso bounds [%.1f, %.1f], level frequencies %s' % (
        lb[0], ub[0], np.round(freq, 3))


def test_log_bound_of_zero_does_not_warn():
    """A switch bound of 0 must not be fed through log10."""
    path = write_space('ConcentrationOn,cOn,2,0,3\n'
                       'SfrEfficiency,aSF,1,0.005,0.2\n')
    space = load_quiet(path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lb, ub = analysis.pso_bounds(space)
    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), (lb, ub)
    return 'bounds finite: %s' % np.round(lb, 3)


# ── Value conversion ──────────────────────────────────────────────────────────

def test_switch_rounds_clamps_and_formats_as_integer():
    """Any particle position yields an in-range int written without a decimal."""
    path = write_space('DynamicDisruptionSplit,split,2,0,2\n')
    space = load_quiet(path)

    for pos in [-9.0, -0.5, 0.2, 0.6, 1.0, 1.4, 1.9, 2.5, 11.0]:
        phys, text = execution._particle_physical(space, np.array([pos]), 0)
        assert isinstance(phys, int), (pos, type(phys))
        assert 0 <= phys <= 2, (pos, phys)
        assert '.' not in text and 'e' not in text, (pos, text)
        assert int(text) == phys, (text, phys)
    return 'rounded, clamped to [0, 2] and formatted without a decimal point'


def test_continuous_conversion_unaffected():
    """Linear and log parameters convert exactly as before."""
    path = write_space('FeedbackReheatingEpsilon,eDisk,0,0.5,5.0\n'
                       'SfrEfficiency,aSF,1,0.005,0.2\n')
    space = load_quiet(path)
    particle = np.array([2.9, np.log10(0.05)])

    lin, lin_text = execution._particle_physical(space, particle, 0)
    log, log_text = execution._particle_physical(space, particle, 1)
    assert abs(lin - 2.9) < 1e-12, lin
    assert abs(log - 0.05) < 1e-12, log
    assert lin_text == '2.9' and log_text == '0.05', (lin_text, log_text)
    return 'linear -> %s, log -> %s' % (lin_text, log_text)


def test_display_reports_rounded_levels():
    """Tracks, CSV and the logged best fit report the level SAGE actually ran."""
    is_log = np.array([False, True])
    is_int = np.array([True, False])
    lb = np.array([0.0, 0.005])
    ub = np.array([2.0, 0.2])

    x = np.array([[1.4, np.log10(0.05)],
                  [-0.3, np.log10(0.02)],
                  [2.49, np.log10(0.1)]])
    out = pso._to_display(x, is_log, is_int=is_int, lb=lb, ub=ub)

    assert list(out[:, 0]) == [1.0, 0.0, 2.0], list(out[:, 0])
    assert abs(out[0, 1] - 0.05) < 1e-12, out[0, 1]
    return 'switch column %s, log column back-transformed' % list(out[:, 0])


# ── Guards ────────────────────────────────────────────────────────────────────

def test_bad_sampling_code_rejected():
    path = write_space('SfrEfficiency,aSF,7,0.005,0.2\n')
    try:
        load_quiet(path)
    except ValueError as e:
        assert 'sampling code' in str(e), str(e)
        return 'raised: %s' % str(e).split('.')[0]
    raise AssertionError('sampling code 7 was accepted')


def test_non_integral_switch_bounds_rejected():
    path = write_space('ConcentrationOn,cOn,2,0.5,3\n')
    try:
        load_quiet(path)
    except ValueError as e:
        assert 'non-integral' in str(e), str(e)
        return 'raised: %s' % e
    raise AssertionError('non-integral switch bounds were accepted')


def test_inverted_switch_bounds_rejected():
    path = write_space('ConcentrationOn,cOn,2,3,0\n')
    try:
        load_quiet(path)
    except ValueError as e:
        assert 'ub < lb' in str(e), str(e)
        return 'raised: %s' % e
    raise AssertionError('inverted switch bounds were accepted')


def test_int_param_declared_continuous_warns():
    """The trap that aborts SAGE mid-run must be caught at load time."""
    path = write_space('ConcentrationOn,cOn,1,1,3\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        analysis.load_space(path)
    messages = [str(w.message) for w in caught]
    assert any('integer parameter in SAGE' in m for m in messages), messages
    return 'warned about ConcentrationOn declared with code 1'


def test_continuous_param_declared_switch_warns():
    path = write_space('SfrEfficiency,aSF,2,0,1\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        analysis.load_space(path)
    messages = [str(w.message) for w in caught]
    assert any('not a known SAGE integer parameter' in m for m in messages), messages
    return 'warned about SfrEfficiency declared with code 2'


def test_sage_int_params_matches_sage_source():
    """SAGE_INT_PARAMS should match SAGE's own REG(..., INT, ...) registry.

    Skipped when the SAGE source is not checked out alongside this repo.
    """
    import re
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'SAGE26',
                     'src', 'core_read_parameter_file.c'),
    ]
    src = next((p for p in candidates if os.path.exists(p)), None)
    if src is None:
        return 'SKIP: SAGE source not found alongside this repo'

    text = open(src).read()
    found = set(re.findall(r'REG\("([A-Za-z0-9_]+)"\s*,[^;]*?\bINT\b', text))
    missing = found - analysis.SAGE_INT_PARAMS
    assert not missing, 'SAGE_INT_PARAMS is missing: %s' % sorted(missing)
    return '%d INT parameters, all present in SAGE_INT_PARAMS' % len(found)


# ── End to end ────────────────────────────────────────────────────────────────

def test_pso_optimises_over_a_switch():
    """A full PSO run explores every level and returns an integral best value."""
    path = write_space('DynamicDisruptionSplit,split,2,0,2\n'
                       'SfrEfficiency,aSF,1,0.005,0.2\n')
    space = load_quiet(path)
    lb, ub = analysis.pso_bounds(space)

    seen = set()

    def objective(particle, *args):
        split, _ = execution._particle_physical(space, particle, 0)
        asf, _ = execution._particle_physical(space, particle, 1)
        assert isinstance(split, int), type(split)
        seen.add(split)
        # Level 1 is best; the switch axis is genuinely piecewise constant.
        return {0: 3.0, 1: 0.0, 2: 1.5}[split] + (np.log10(asf) - np.log10(0.05)) ** 2

    xopt, fopt = pso.pso(objective, lb, ub, swarmsize=12, maxiter=12,
                         random_seed=11,
                         is_log=space['is_log'].astype(bool),
                         is_int=space['is_int'].astype(bool),
                         int_lb=space['lb'], int_ub=space['ub'])

    assert seen == {0, 1, 2}, 'levels explored: %s' % sorted(seen)
    assert xopt[0] == 1.0, 'best switch level %r, expected 1' % xopt[0]
    assert abs(xopt[1] - 0.05) < 0.02, xopt[1]
    return 'explored %s, best level %d, aSF %.4f' % (
        sorted(seen), int(xopt[0]), xopt[1])


TESTS = [
    ('sampling codes expand to is_log/is_int', test_sampling_codes_expand),
    ('existing space files unchanged',         test_existing_space_files_unchanged),
    ('switch bounds widened by half a step',   test_switch_bounds_widened_by_half_step),
    ('switch bound of 0 does not hit log10',   test_log_bound_of_zero_does_not_warn),
    ('switch rounds, clamps, formats as int',  test_switch_rounds_clamps_and_formats_as_integer),
    ('continuous conversion unaffected',       test_continuous_conversion_unaffected),
    ('display reports rounded levels',         test_display_reports_rounded_levels),
    ('bad sampling code rejected',             test_bad_sampling_code_rejected),
    ('non-integral switch bounds rejected',    test_non_integral_switch_bounds_rejected),
    ('inverted switch bounds rejected',        test_inverted_switch_bounds_rejected),
    ('int param declared continuous warns',    test_int_param_declared_continuous_warns),
    ('continuous param declared switch warns', test_continuous_param_declared_switch_warns),
    ('SAGE_INT_PARAMS matches SAGE source',    test_sage_int_params_matches_sage_source),
    ('PSO optimises over a switch',            test_pso_optimises_over_a_switch),
]


def run_all():
    print('=' * 70)
    print('Integer-switch search space unit tests')
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
            print('  FAIL  %s' % label)
            print('        %s' % e)
            failed += 1
        except Exception as e:
            print('  FAIL  %s' % label)
            print('        %s: %s' % (type(e).__name__, e))
            failed += 1

    print('=' * 70)
    print('Results: %d passed, %d failed out of %d tests' % (passed, failed, passed + failed))
    print('=' * 70)
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
