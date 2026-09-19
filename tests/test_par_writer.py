#!/usr/bin/env python3
"""
Unit tests for execution.write_particle_par.

This is the code that turns a PSO particle into a SAGE parameter file, so every
defect here is silent: the run completes and produces plausible-looking numbers
that answer the wrong question.  The two defects it replaced were

  * an early exit that could leave OutputDir unrewritten, sending every
    particle's output into the same directory, and
  * a prefix tag match that would overwrite OmegaLambda when searching Omega.

Usage:
    python3 tests/test_par_writer.py
    python3 -m pytest tests/test_par_writer.py -v  (if pytest installed)
"""

import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import analysis, execution

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def write_tmp(content, suffix='.txt'):
    fd, path = tempfile.mkstemp(suffix=suffix, text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def load_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return analysis.load_space(path)


def write(par, space, particle, output_dir='/particle/7/'):
    out = write_tmp('', suffix='.par')
    execution.write_particle_par(par, out, output_dir, space, particle)
    return open(out).read()


# ── OutputDir ─────────────────────────────────────────────────────────────────

def test_outputdir_rewritten_when_it_follows_the_parameters():
    """The regression that would collapse all particles into one directory."""
    space = load_quiet(write_tmp(
        'SfrEfficiency,aSF,1,0.005,0.2\n'
        'ReIncorporationFactor,eReinc,0,0.05,0.5\n'))
    par = write_tmp(
        'SfrEfficiency               0.05    %comment\n'
        'ReIncorporationFactor       0.15    %comment\n'
        'OutputDir   /original/place/\n'
        'FirstFile 0\n', suffix='.par')

    text = write(par, space, np.array([np.log10(0.11), 0.42]))
    assert 'OutputDir              /particle/7/' in text, text
    assert '/original/place/' not in text, text
    assert 'SfrEfficiency          0.11' in text, text
    assert 'ReIncorporationFactor          0.42' in text, text
    return 'OutputDir rewritten and both parameters substituted'


def test_outputdir_appended_never_silently_missing():
    """Every parameter is substituted even when OutputDir comes first."""
    space = load_quiet(write_tmp('SfrEfficiency,aSF,1,0.005,0.2\n'))
    par = write_tmp('OutputDir   /orig/\nSfrEfficiency 0.05\n', suffix='.par')
    text = write(par, space, np.array([np.log10(0.07)]))
    assert 'OutputDir              /particle/7/' in text, text
    assert 'SfrEfficiency          0.07' in text, text
    return 'ordinary ordering still correct'


# ── Tag matching ──────────────────────────────────────────────────────────────

def test_prefix_collision_does_not_corrupt_longer_parameter():
    """Searching Omega must not overwrite OmegaLambda (SAGE's only such pair)."""
    space = load_quiet(write_tmp('Omega,Om,0,0.1,0.5\n'))
    par = write_tmp('OutputDir   /orig/\n'
                    'Omega             0.25   %matter density\n'
                    'OmegaLambda       0.75   %dark energy\n', suffix='.par')
    text = write(par, space, np.array([0.31]))
    assert 'Omega          0.31' in text, text
    assert 'OmegaLambda       0.75   %dark energy' in text, text
    return 'Omega substituted, OmegaLambda left intact'


def test_commented_lines_are_not_definitions():
    """A commented-out tag must be neither rewritten nor counted as present."""
    space = load_quiet(write_tmp('SfrEfficiency,aSF,1,0.005,0.2\n'))
    par = write_tmp('%OutputDir  /template/path/\n'
                    'OutputDir   /orig/\n'
                    '%SfrEfficiency  0.99  %old value\n'
                    'SfrEfficiency  0.05\n', suffix='.par')
    text = write(par, space, np.array([np.log10(0.07)]))
    assert '%OutputDir  /template/path/' in text, text
    assert '%SfrEfficiency  0.99' in text, text
    assert 'SfrEfficiency          0.07' in text, text

    only_commented = write_tmp('OutputDir /o/\n%SfrEfficiency 0.05\n', suffix='.par')
    missing = execution.missing_from_config(space, only_commented)
    assert missing == ['SfrEfficiency'], missing
    return 'commented lines preserved and reported as absent'


# ── Guard ─────────────────────────────────────────────────────────────────────

def test_missing_parameters_are_reported():
    space = load_quiet(write_tmp('SfrEfficiency,aSF,1,0.005,0.2\n'
                                 'NotAParameter,nope,0,0,1\n'))
    par = write_tmp('OutputDir /o/\nSfrEfficiency 0.05\n', suffix='.par')
    assert execution.missing_from_config(space, par) == ['NotAParameter']
    return 'missing_from_config flags exactly the absent parameter'


def test_writer_warns_on_dropped_parameter():
    """A parameter absent from the .par must not vanish quietly."""
    space = load_quiet(write_tmp('SfrEfficiency,aSF,1,0.005,0.2\n'
                                 'NotAParameter,nope,0,0,1\n'))
    par = write_tmp('OutputDir /o/\nSfrEfficiency 0.05\n', suffix='.par')
    out = write_tmp('', suffix='.par')

    import logging
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Catch()
    execution.logger.addHandler(handler)
    try:
        execution.write_particle_par(par, out, '/p/0/', space,
                                     np.array([np.log10(0.07), 0.5]))
    finally:
        execution.logger.removeHandler(handler)

    assert any('NotAParameter' in m for m in records), records
    return 'warned: %s' % records[0][:70]


# ── Regression against the previous implementation ────────────────────────────

def _old_writer(config_path, output_dir, space, particle):
    """The pre-fix logic, verbatim: prefix match plus the Ndone early break."""
    lines = open(config_path).readlines()
    Np = len(space['name'])
    Ndone = 0
    for l, line in enumerate(lines):
        if line[:9] == 'OutputDir':
            lines[l] = 'OutputDir              ' + output_dir + '\n'
        for p in range(Np):
            if line[:len(space['name'][p])] == space['name'][p]:
                _, txt = execution._particle_physical(space, particle, p)
                lines[l] = space['name'][p] + '          ' + txt + '\n'
                Ndone += 1
        if Ndone == Np:
            break
    return lines


def test_byte_identical_to_old_writer_on_real_configs():
    """The fix must not change output for any config that already worked.

    Skipped when the SAGE parameter files are not checked out alongside.
    """
    par = os.path.join(REPO, '..', 'SAGE26', 'input', 'millennium.par')
    if not os.path.exists(par):
        return 'SKIP: SAGE26 input/millennium.par not found'

    rng = np.random.default_rng(0)
    checked = []
    for name in sorted(os.listdir(REPO)):
        if not (name.startswith('space') and name.endswith('.txt')):
            continue
        space = load_quiet(os.path.join(REPO, name))
        if execution.missing_from_config(space, par):
            continue                      # cannot be compared: not in this .par
        lb, ub = analysis.pso_bounds(space)
        out = write_tmp('', suffix='.par')
        for _ in range(25):
            particle = rng.uniform(lb, ub)
            old = _old_writer(par, '/p/9/', space, particle)
            execution.write_particle_par(par, out, '/p/9/', space, particle)
            new = open(out).readlines()
            assert old == new, '%s differs:\n%r\n%r' % (
                name, [x for x in old if x not in new],
                [x for x in new if x not in old])
        checked.append(name)
    if not checked:
        return 'SKIP: no space file matches millennium.par'
    return 'identical over 25 particles each: %s' % ', '.join(checked)


TESTS = [
    ('OutputDir rewritten when it follows params', test_outputdir_rewritten_when_it_follows_the_parameters),
    ('ordinary ordering still correct',            test_outputdir_appended_never_silently_missing),
    ('prefix collision leaves OmegaLambda alone',  test_prefix_collision_does_not_corrupt_longer_parameter),
    ('commented lines are not definitions',        test_commented_lines_are_not_definitions),
    ('missing parameters reported',                test_missing_parameters_are_reported),
    ('writer warns on dropped parameter',          test_writer_warns_on_dropped_parameter),
    ('byte-identical to old writer (real config)', test_byte_identical_to_old_writer_on_real_configs),
]


def run_all():
    print('=' * 70)
    print('SAGE parameter-file writer unit tests')
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
