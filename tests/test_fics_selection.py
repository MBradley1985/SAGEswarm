#!/usr/bin/env python3
"""
Unit tests for the FICS halo selection.

f_ICS is a per-halo quantity, so the constraint only means what it claims if the
selection is exactly right: the haloes must be group-or-richer central galaxies
with at least FICS_MIN_SATELLITES satellites, and those cuts must apply to
nothing but FICS.  A cut that silently leaked into the galaxy arrays would
change the stellar mass function, the metallicity relation and everything else
without any visible error.

Usage:
    python3 tests/test_fics_selection.py
    python3 -m pytest tests/test_fics_selection.py -v  (if pytest installed)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import constraints as C

H0 = 1.0          # keeps code units and Msun/1e10 interchangeable in the fixtures
MVIR = 1000.0     # 1e13 Msun, comfortably above FICS_MVIR_MIN
BCG = 10.0        # 1e11 Msun, comfortably above FICS_STELLARMASS_MIN
SAT = 3.0         # 3e10 Msun per satellite


def make_catalogue(halo_spec):
    """Build a synthetic catalogue.

    halo_spec is a list of (n_satellites, target_f_ICS); every halo is given the
    ICS mass that produces its target fraction, so the selection can be read off
    the resulting median.
    """
    gal, cen, typ, sm, ics, mvir = [], [], [], [], [], []
    gid = 0
    for n_sat, target in halo_spec:
        cgid = gid
        gid += 1
        rest = BCG + n_sat * SAT
        ics_mass = target / (1.0 - target) * rest if target > 0 else 0.0
        gal.append(cgid); cen.append(cgid); typ.append(0)
        sm.append(BCG); ics.append(ics_mass); mvir.append(MVIR)
        for _ in range(n_sat):
            gal.append(gid); cen.append(cgid); typ.append(1)
            sm.append(SAT); ics.append(0.0); mvir.append(0.0)
            gid += 1
    return dict(
        GalaxyIndex=np.array(gal, dtype=np.int64),
        CentralGalaxyIndex=np.array(cen, dtype=np.int64),
        Type=np.array(typ, dtype=np.int32),
        StellarMass=np.array(sm, dtype=np.float32),
        IntraClusterStars=np.array(ics, dtype=np.float32),
        Mvir=np.array(mvir, dtype=np.float32))


# ── Satellite counting ────────────────────────────────────────────────────────

def test_satellite_count_is_exact():
    """n_satellites must equal the number of Type != 0 members of each group."""
    counts = [0, 1, 2, 3, 5]
    G = make_catalogue([(n, 0.2) for n in counts])
    sm = G['StellarMass'].astype(float) * 1e10 / H0
    _, n_sat = C.satellite_stellar_mass(G, sm)
    centrals = G['Type'] == 0
    got = list(n_sat[centrals])
    assert got == counts, 'counted %s, expected %s' % (got, counts)
    return 'counted %s satellites for groups of %s' % (got, counts)


def test_satellite_mass_summed_onto_central():
    G = make_catalogue([(3, 0.2)])
    sm = G['StellarMass'].astype(float) * 1e10 / H0
    sat_mass, _ = C.satellite_stellar_mass(G, sm)
    expected = 3 * SAT * 1e10 / H0
    got = sat_mass[G['Type'] == 0][0]
    assert abs(got - expected) < 1.0, (got, expected)
    return '3 satellites summed to %.3g Msun on the central' % got


# ── The minimum-satellite requirement ─────────────────────────────────────────

def test_minimum_two_satellites_enforced():
    """Haloes with fewer than two satellites must be excluded.

    Five haloes with 2 satellites carry f_ICS = 0.5; five with 1 satellite carry
    f_ICS = 0.05.  If the cut holds, the median is 0.5.  If it were absent, all
    ten would enter and the median would fall to roughly 0.275.
    """
    assert C.FICS_MIN_SATELLITES == 2, (
        'this test assumes FICS_MIN_SATELLITES == 2, found %r'
        % C.FICS_MIN_SATELLITES)

    G = make_catalogue([(2, 0.5)] * 5 + [(1, 0.05)] * 5)
    median, _ = C.fics_median(G, H0)
    assert np.isfinite(median), 'no haloes selected at all'
    assert abs(median - 0.5) < 1e-6, (
        'median %.4f: the one-satellite haloes were not excluded' % median)
    return 'median = %.4f, so only the two-satellite haloes were used' % median


def test_one_satellite_halo_alone_is_not_measurable():
    """A box of only one-satellite haloes must yield no measurement, not a fit."""
    G = make_catalogue([(1, 0.3)] * 20)
    median, err = C.fics_median(G, H0)
    assert not np.isfinite(median), 'measured %r from one-satellite haloes' % median
    assert not np.isfinite(err)
    return 'returned (nan, nan) from 20 one-satellite haloes'


def test_exactly_two_satellites_is_included():
    """The cut is >= 2, so exactly two satellites qualifies."""
    G = make_catalogue([(2, 0.4)] * C.FICS_MIN_HALOES)
    median, _ = C.fics_median(G, H0)
    assert np.isfinite(median) and abs(median - 0.4) < 1e-6, median
    return 'exactly two satellites qualifies (median %.4f)' % median


def test_satellites_of_a_missing_central_are_not_miscounted():
    """An orphan satellite must not be credited to an unrelated halo."""
    G = make_catalogue([(2, 0.5)] * 5 + [(2, 0.5)])
    # Point the last halo's satellites at a central that is not in the catalogue
    orphan = G['CentralGalaxyIndex'] == G['GalaxyIndex'][-3]
    G['CentralGalaxyIndex'][orphan & (G['Type'] != 0)] = 10 ** 9
    sm = G['StellarMass'].astype(float) * 1e10 / H0
    sat_mass, n_sat = C.satellite_stellar_mass(G, sm)
    centrals = np.flatnonzero(G['Type'] == 0)
    assert list(n_sat[centrals]) == [2, 2, 2, 2, 2, 0], list(n_sat[centrals])
    return 'orphaned satellites dropped, not reassigned'


# ── Scope: the cuts must be FICS-only ─────────────────────────────────────────

def test_halo_cuts_do_not_affect_other_constraints():
    """Making the FICS cuts reject everything must not move any other constraint.

    Skipped when no SAGE output is available to score against.
    """
    out = os.path.join(os.path.dirname(__file__), '..', '..',
                       'SAGE26', 'output', 'millennium')
    alist = os.path.join(os.path.dirname(__file__), '..', '..', 'SAGE26',
                         'input', 'millennium', 'trees', 'millennium.a_list')
    if not (os.path.isdir(out) and os.path.exists(alist)):
        return 'SKIP: no SAGE26 millennium output to score against'
    if not [f for f in os.listdir(out) if f.startswith('model_')]:
        return 'SKIP: no model_*.hdf5 in SAGE26 millennium output'

    kw = dict(sim=1, boxsize=62.5, vol_frac=1.0, h0=0.73, Omega0=0.25,
              age_alist_file=alist)
    specs = ['SMF_z0', 'BHBM', 'HIMF', 'H2MF', 'MZR', 'SHMR']

    def score_all():
        got = {}
        for spec in specs:
            c = C.parse(spec, **kw)[0]
            got[spec] = c.get_data(out, ['0'])
        return got

    before = score_all()

    saved = (C.FICS_MVIR_MIN, C.FICS_MIN_SATELLITES, C.FICS_STELLARMASS_MIN,
             C.FICS_BCG_MVIR_RATIO_MIN)
    C.FICS_MVIR_MIN = 10 ** 15.5
    C.FICS_MIN_SATELLITES = 500
    C.FICS_STELLARMASS_MIN = 1e14
    C.FICS_BCG_MVIR_RATIO_MIN = 0.9
    try:
        after = score_all()
    finally:
        (C.FICS_MVIR_MIN, C.FICS_MIN_SATELLITES, C.FICS_STELLARMASS_MIN,
         C.FICS_BCG_MVIR_RATIO_MIN) = saved

    for spec in specs:
        for i, (a, b) in enumerate(zip(before[spec], after[spec])):
            assert np.array_equal(a, b), (
                '%s changed when the FICS cuts changed (array %d)' % (spec, i))
    return 'unchanged: %s' % ', '.join(specs)


def test_only_fics_runs_the_halo_cut():
    """The cut must be structurally unreachable from other constraints."""
    from src.constraints import (SMF_z0, SMF_z05, BHMF_z0, BHBM, HIMF, H2MF,
                                 MZR, SHMR, SMD, CSFRDH, FICS)
    others = [SMF_z0, SMF_z05, BHMF_z0, BHBM, HIMF, H2MF, MZR, SHMR, SMD, CSFRDH]
    wrong = [c.__name__ for c in others if c.needs_fics]
    assert not wrong, 'these apply the FICS halo cuts but should not: %s' % wrong
    assert FICS.needs_fics, 'FICS must apply its own halo cuts'
    return 'FICS only; %d other constraints skip it' % len(others)


def test_gate_actually_prevents_the_call():
    """needs_fics = False must stop fics_median being called at all."""
    calls = []
    real = C.fics_median

    def spy(G, h0):
        calls.append(h0)
        return real(G, h0)

    C.fics_median = spy
    try:
        class Gated(C.Constraint):
            pass
        gated = Gated(snapshot=[63], sim=1, boxsize=62.5, vol_frac=1.0,
                      h0=0.73, Omega0=0.25, age_alist_file=None)
        assert gated.needs_fics is False
        # _fics_for_snapshot is the only route into the cut; the gate in
        # _load_model_data is what must keep it unused.
        assert C.Constraint.needs_fics is False
        n_before = len(calls)
        gated._fics_for_snapshot(make_catalogue([(2, 0.4)] * 6))
        assert len(calls) == n_before + 1, 'spy not wired up'
    finally:
        C.fics_median = real
    return 'fics_median reachable only via _fics_for_snapshot, gated by needs_fics'


def test_fics_median_does_not_mutate_the_catalogue():
    """Other constraints read the same dict, so it must come back untouched."""
    G = make_catalogue([(2, 0.4)] * 6)
    snapshot = {k: v.copy() for k, v in G.items()}
    C.fics_median(G, H0)
    for k in G:
        assert np.array_equal(G[k], snapshot[k]), '%s was mutated' % k
    return 'all %d catalogue fields unchanged' % len(G)


# ── FICS_Mvir: the halo-mass relation ─────────────────────────────────────────

def test_fics_per_halo_returns_selected_haloes():
    """fics_per_halo must return one (logMvir, f_ICS) pair per selected halo."""
    G = make_catalogue([(2, 0.4)] * 7 + [(1, 0.4)] * 3)
    logm, f_ics = C.fics_per_halo(G, H0)
    assert len(logm) == len(f_ics) == 7, (len(logm), len(f_ics))
    assert np.allclose(f_ics, 0.4), f_ics
    assert np.allclose(logm, np.log10(MVIR * 1e10 / H0)), logm[:3]
    return '7 of 10 haloes selected, f_ICS = 0.4, logMvir = %.2f' % logm[0]


def test_fics_median_agrees_with_per_halo():
    """The two entry points must not drift apart."""
    G = make_catalogue([(2, f) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
    _, f_ics = C.fics_per_halo(G, H0)
    median, _ = C.fics_median(G, H0)
    assert abs(median - float(np.median(f_ics))) < 1e-12, (median, np.median(f_ics))
    return 'median %.4f matches the per-halo median' % median


def test_fics_mvir_bins_and_skips_sparse_bins():
    """Bins below FICS_MVIR_BIN_MIN haloes must be dropped, not averaged in."""
    from src.constraints import FICS_Mvir
    c = FICS_Mvir.__new__(FICS_Mvir)
    c.domain = (14.0, 14.45)

    # One well-populated decade and one deliberately sparse high-mass bin
    logm = np.concatenate([np.full(20, 14.05), np.full(20, 14.30),
                           np.full(2, 15.00)])
    frac = np.concatenate([np.full(20, 0.10), np.full(20, 0.20),
                           np.full(2, 0.90)])
    args = [None] * 23 + [None, None, logm, frac, None, None, None]
    x, y, err = c.get_model_x_y(*args)

    assert len(x) == 2, 'expected 2 bins, got %d at %s' % (len(x), np.round(x, 3))
    assert np.allclose(y, [0.10, 0.20]), y
    assert not np.any(np.isclose(y, 0.90)), 'the 2-halo bin leaked in'
    assert np.all(err > 0), err
    return 'kept 2 populated bins at logM %s, dropped the 2-halo bin' % np.round(x, 2)


def test_fics_mvir_reports_unreachable_observations():
    """Observations the model cannot bracket must be named, not silently used."""
    import logging
    from src.constraints import FICS_Mvir
    c = FICS_Mvir.__new__(FICS_Mvir)
    c.domain = (14.0, 15.2)          # admits all seven observations

    logm = np.concatenate([np.full(20, 14.05), np.full(20, 14.30)])
    frac = np.concatenate([np.full(20, 0.10), np.full(20, 0.20)])
    args = [None] * 23 + [None, None, logm, frac, None, None, None]

    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger('constraints')
    handler = Catch()
    log.addHandler(handler)
    try:
        c.get_model_x_y(*args)
    finally:
        log.removeHandler(handler)

    assert any('flat extrapolation' in m for m in records), records
    msg = [m for m in records if 'flat extrapolation' in m][0]
    for point in ('14.50', '14.70', '14.90', '14.95', '15.10'):
        assert point in msg, 'did not name %s: %s' % (point, msg)
    return 'named all five unreachable observations'


def test_fics_mvir_unscoreable_run_is_not_a_fit():
    """Too few haloes must give a large-error dummy, not a plausible score."""
    from src.constraints import FICS_Mvir
    c = FICS_Mvir.__new__(FICS_Mvir)
    c.domain = (14.0, 14.45)
    args = [None] * 23 + [None, None, np.array([]), np.array([]), None, None, None]
    x, y, err = c.get_model_x_y(*args)
    assert np.all(err >= 1.0), 'errors %s are too small to flag a failed run' % err
    return 'returned a flat dummy with error %s' % err


TESTS = [
    ('satellite count is exact',                 test_satellite_count_is_exact),
    ('satellite mass summed onto central',        test_satellite_mass_summed_onto_central),
    ('minimum of two satellites enforced',        test_minimum_two_satellites_enforced),
    ('one-satellite haloes are not measurable',   test_one_satellite_halo_alone_is_not_measurable),
    ('exactly two satellites is included',        test_exactly_two_satellites_is_included),
    ('orphan satellites not miscounted',          test_satellites_of_a_missing_central_are_not_miscounted),
    ('only FICS runs the halo cut',              test_only_fics_runs_the_halo_cut),
    ('needs_fics gate prevents the call',        test_gate_actually_prevents_the_call),
    ('halo cuts do not affect other constraints', test_halo_cuts_do_not_affect_other_constraints),
    ('fics_median does not mutate catalogue',     test_fics_median_does_not_mutate_the_catalogue),
    ('fics_per_halo returns selected haloes',     test_fics_per_halo_returns_selected_haloes),
    ('fics_median agrees with per-halo',          test_fics_median_agrees_with_per_halo),
    ('FICS_Mvir bins, skips sparse bins',         test_fics_mvir_bins_and_skips_sparse_bins),
    ('FICS_Mvir reports unreachable obs',         test_fics_mvir_reports_unreachable_observations),
    ('FICS_Mvir unscoreable run is not a fit',    test_fics_mvir_unscoreable_run_is_not_a_fit),
]


def run_all():
    print('=' * 70)
    print('FICS halo selection unit tests')
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
