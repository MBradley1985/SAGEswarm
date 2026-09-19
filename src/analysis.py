#!/bin/bash

"""
Contains the methods used to quantify the "goodness" of fit
between model and observed parameters
"""

import functools
import math
import sys
import warnings

import numpy as np # type: ignore
import scipy.special # type: ignore


if sys.version_info[0] == 3:
    b2s = lambda b: b.decode('ascii')
else:
    b2s = lambda b: b

# Sampling codes for the third column of a space file.  0 and 1 are the
# original meaning of the column, so every existing space file keeps working.
SAMPLING_LINEAR = 0   # continuous, searched directly
SAMPLING_LOG = 1      # continuous, searched in log10 and reported physically
SAMPLING_INT = 2      # discrete switch: searched continuously, rounded to the
                      # nearest integer before it reaches SAGE
SAMPLING_CODES = (SAMPLING_LINEAR, SAMPLING_LOG, SAMPLING_INT)

# SAGE parameters its reader registers as INT, taken from the REG(..., INT, ...)
# entries in SAGE26 src/core_read_parameter_file.c.  These must be declared with
# SAMPLING_INT in a space file: SAGE parses them with strtol and aborts the run
# outright if anything is left over, so a continuous particle value like "1.37"
# kills the whole optimisation rather than just that evaluation.  Purely
# informational -- a different SAGE build may register a different set, so a
# mismatch warns rather than raises.
SAGE_INT_PARAMS = frozenset((
    'AGNrecipeOn', 'BulgeSizeOn', 'CGMDensityProfile', 'CGMrecipeOn',
    'ColdStreamCeilingOn', 'ConcentrationOn', 'DiskInstabilityOn',
    'DynamicDisruptionSplit', 'FFBIgnoreRegime', 'FFBRandomMode', 'FIREmodeOn',
    'FeedbackFreeModeOn', 'FirstFile', 'H2DiskAreaOption',
    'H2RadialIntegrationOn', 'H2RadialNBins', 'LastFile', 'LastSnapshotNr',
    'NumOutputs', 'NumSimulationTreeFiles', 'PrecipCriterionOn',
    'RamPressureStrippingOn', 'RegimeRandomMode', 'ReionizationOn',
    'SFprescription', 'SNEnergyConservationOn', 'SaveFullSFH',
    'StarburstColdGasOn', 'SupernovaRecipeOn', 'TrackICSAssembly',
))


def load_space(fname):
    """Loads the search space from a csv file.

    Columns are: name, plot_label, sampling code, lower bound, upper bound.

    The sampling code is 0 (linear), 1 (log10) or 2 (integer switch).  It is
    expanded here into two separate boolean fields, `is_log` and `is_int`, so
    that code reading `space['is_log']` cannot mistake a switch for a
    log-sampled parameter.
    """
    space = np.genfromtxt(fname, delimiter=',',
              dtype=[('name', 'object'), ('plot_label', 'object'),
                     ('sampling', 'int'),
                     ('lb', 'float'), ('ub', 'float')])
    # genfromtxt returns a 0-d scalar for a single-row file, which cannot be
    # iterated over below and breaks every caller.  A one-parameter search space
    # is legitimate (e.g. scanning a single ICS split parameter), so promote it.
    space = np.atleast_1d(space)

    names = [b2s(n) for n in space['name']]
    sampling = space['sampling']

    bad = [(names[i], int(c)) for i, c in enumerate(sampling)
           if int(c) not in SAMPLING_CODES]
    if bad:
        raise ValueError(
            'Invalid sampling code in %s: %s. Column 3 must be 0 (linear), '
            '1 (log10) or 2 (integer switch).'
            % (fname, ', '.join('%s=%d' % b for b in bad)))

    is_int = sampling == SAMPLING_INT
    # An integer switch must have integral, ordered bounds, or the rounding and
    # clamping in execution._to_physical has nothing sensible to target.
    for i in np.flatnonzero(is_int):
        lb, ub = space['lb'][i], space['ub'][i]
        if lb != int(lb) or ub != int(ub):
            raise ValueError(
                'Integer switch %s in %s has non-integral bounds (%g, %g)'
                % (names[i], fname, lb, ub))
        if ub < lb:
            raise ValueError(
                'Integer switch %s in %s has ub < lb (%g < %g)'
                % (names[i], fname, ub, lb))

    out = np.zeros(len(space), dtype=[
        ('name', 'object'), ('plot_label', 'object'),
        ('is_log', 'int'), ('is_int', 'int'),
        ('lb', 'float'), ('ub', 'float')])
    out['name'] = names
    out['plot_label'] = [b2s(n) for n in space['plot_label']]
    out['is_log'] = (sampling == SAMPLING_LOG).astype(int)
    out['is_int'] = is_int.astype(int)
    out['lb'] = space['lb']
    out['ub'] = space['ub']

    # Catch the two ways of getting the sampling code wrong for a switch, both
    # of which fail late and confusingly (a mid-run SAGE abort, or a switch
    # pinned to one level for the entire optimisation).
    for i, name in enumerate(names):
        if name in SAGE_INT_PARAMS and not out['is_int'][i]:
            warnings.warn(
                '%s in %s is an integer parameter in SAGE but is declared with '
                'sampling code %d; SAGE will abort on the first non-integer '
                'value. Use code %d.'
                % (name, fname, int(sampling[i]), SAMPLING_INT),
                stacklevel=2)
        elif out['is_int'][i] and name not in SAGE_INT_PARAMS:
            warnings.warn(
                '%s in %s is declared as an integer switch but is not a known '
                'SAGE integer parameter; it will be rounded to whole numbers, '
                'which is probably not intended for a continuous parameter.'
                % (name, fname),
                stacklevel=2)

    return out


def pso_bounds(space):
    """Bounds to hand to pso.pso for a given search space.

    Log-sampled parameters are converted to log10.  Integer switches are widened
    by half a step: sampling uniformly in [lb, ub] and rounding would give the
    two end values half the probability of the interior ones, so the search runs
    over [lb-0.5, ub+0.5] and every level gets equal width.
    """
    is_log = space['is_log'].astype(bool)
    is_int = space['is_int'].astype(bool)

    lb = np.array(space['lb'], dtype=float)
    ub = np.array(space['ub'], dtype=float)

    # Take the log only where it is asked for: a switch may legitimately have a
    # bound of 0, and log10 of the whole column would warn and produce -inf even
    # though np.where would discard it.
    lb[is_log] = np.log10(lb[is_log])
    ub[is_log] = np.log10(ub[is_log])

    lb[is_int] = space['lb'][is_int] - 0.5
    ub[is_int] = space['ub'][is_int] + 0.5
    return lb, ub

def npsum(f):
    """Return the array-wise sum of the returned array.

    Keyword arguments are forwarded: the statistics take options (huber's
    delta, cash's count_scale) and dropping them silently made those
    unreachable through the decorated name.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return np.sum(f(*args, **kwargs))
    return wrapper

@npsum
def chi2(obs, mod, err):
    """
    Calculate chi-squared statistic.
    
    Parameters:
    -----------
    obs : array
        Observed values
    mod : array
        Model values
    err : array
        1-sigma uncertainties on observations
        
    Returns:
    --------
    chi2 : float
        Sum of squared normalized residuals
    """
    # Ensure errors are valid (avoid division by zero)
    err = np.maximum(err, 1e-8)
    
    # Standard chi-squared: sum of squared normalized residuals
    chi2 = ((mod - obs) / err)**2
    
    return chi2

# Degrees of freedom bounds for the Student-t objective.  A t-distribution only
# has finite variance for nu > 2, and nu above a few hundred is numerically
# indistinguishable from a Gaussian.
STUDENT_NU_MIN = 2.1
STUDENT_NU_MAX = 1.0e4


def student_dof(var):
    """Degrees of freedom for the Student-t objective, from the residual variance.

    Method of moments: a t-distribution with nu degrees of freedom has variance
    nu/(nu-2), so nu = 2*var/(var-1).  That inversion is only valid for var > 1.

    The previous version applied it unconditionally, which made the objective
    NaN over the entire regime a calibration is trying to reach: var < 1 means
    the residuals are smaller than the errors, and it returns a NEGATIVE nu,
    hence gamma(nu/2) of a negative argument and a NaN score.  PSO then rejected
    every well-fitting particle, because `fx < fp` is False for NaN -- so the
    optimiser actively avoided good fits.

    For var <= 1 there are no outliers to down-weight, so the heavy tail is not
    wanted and the Gaussian limit (large nu) is the right answer.  The estimate
    is continuous across var = 1, where nu -> infinity from both sides.
    """
    var = np.asarray(var, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = np.where(var > 1.0, 2.0 * var / np.maximum(var - 1.0, 1e-12),
                      STUDENT_NU_MAX)
    return np.clip(np.nan_to_num(nu, nan=STUDENT_NU_MAX,
                                 posinf=STUDENT_NU_MAX),
                   STUDENT_NU_MIN, STUDENT_NU_MAX)


@npsum
def studentT(obs, mod, err):
    """Negative log Student-t density of the standardised residuals.

    A heavy-tailed alternative to chi2: a handful of badly discrepant points
    (a wrong observation, a mass bin the model cannot reach) cost far less than
    they would under a Gaussian, so they do not hijack the fit.  As the fit
    improves the estimated degrees of freedom rise and the objective tends to
    the Gaussian result.
    """
    # Ensure errors are valid
    err = np.where(err == 0, np.std(obs), err)
    err = np.maximum(err, 1e-8)

    # Standardised residuals and their variance
    sigma = (obs - mod) / err
    var = np.mean(sigma ** 2)
    var = np.maximum(var, 1e-8)  # Just prevent division by zero

    nu = float(student_dof(var))

    # The t-statistic is the SQUARED standardised residual.  This previously
    # divided by err rather than err**2, which is dimensionally inconsistent
    # with the variance above and mis-scaled every residual by a factor of err.
    x = sigma ** 2
    safe_x = np.maximum(1.0 + x / nu, 1e-300)

    # log density, computed with gammaln so large nu does not overflow
    log_t = (scipy.special.gammaln((nu + 1.0) / 2.0)
             - scipy.special.gammaln(nu / 2.0)
             - 0.5 * np.log(nu * math.pi)
             - 0.5 * (nu + 1.0) * np.log(safe_x))

    # Return the negative log of the density
    return -log_t


@npsum
def huber(obs, mod, err, delta=2.0):
    """Huber loss on the standardised residuals.

    Quadratic within `delta` sigma and linear beyond it, so a discrepant point
    contributes in proportion to its distance rather than its distance squared.
    A middle ground between chi2 and student-t: it down-weights outliers like
    student-t but has no degrees-of-freedom to estimate, so it cannot misbehave
    when the fit is good, and its scale is directly comparable to chi2 in the
    well-fitting regime (they agree exactly for residuals below delta).

    delta = 2 sigma is the usual choice: points inside 2 sigma are treated as
    Gaussian, points beyond are treated as suspect.
    """
    err = np.maximum(err, 1e-8)
    r = np.abs((obs - mod) / err)
    return np.where(r <= delta, r ** 2, delta * (2.0 * r - delta))


@npsum
def cauchy(obs, mod, err):
    """Negative log Cauchy (Lorentzian) likelihood of the standardised residuals.

    The heaviest-tailed sensible choice -- Student-t with one degree of freedom.
    A badly wrong point costs only logarithmically, so a single bad observation
    or a mass bin the model fundamentally cannot reach will not steer the fit.
    Use it when you suspect some observations are simply wrong; prefer huber or
    student-t when you only suspect they are noisy.
    """
    err = np.maximum(err, 1e-8)
    r = (obs - mod) / err
    return np.log1p(r ** 2)


@npsum
def absolute(obs, mod, err):
    """Sum of absolute standardised residuals (L1).

    The most robust of the simple options and the easiest to reason about: the
    total is the mean number of sigma the model is away from the data, so a
    value of 1 means "typically one sigma out".  Insensitive to outliers but
    also less sensitive to genuinely good agreement than chi2.
    """
    err = np.maximum(err, 1e-8)
    return np.abs((obs - mod) / err)


@npsum
def cash(obs, mod, err, count_scale=None):
    """Cash (1979) C statistic for a mass function, from galaxy counts.

    A mass function is a histogram: the model value in a bin is a COUNT of
    simulated galaxies, N = phi * dm * V, and counts are Poisson, not Gaussian.
    At the massive end a bin holds a handful of objects -- in a 100/h Mpc box the
    top occupied SMF bins hold 1, 2 and 4 galaxies -- and there a Gaussian error
    on log10(phi) is badly wrong: N = 1 has a 68% Poisson interval of
    [-3.0, +0.3] dex, wildly asymmetric, and no symmetric error bar represents
    it.  chi2 on log10(phi) therefore mis-weights exactly the bins that
    discriminate hardest between models, and needs the phi_empty sentinel hack
    to cope with empty ones at all.

    C = 2 * sum[ mu - N + N ln(N/mu) ]

    which is -2 ln of the Poisson likelihood ratio, asymptotically distributed
    like chi2, so it can be read on the same scale.

    Direction matters and is deliberate: the MODEL is treated as the Poisson
    realisation (N) and the OBSERVATION as the underlying rate (mu).  That is
    the right way round here -- a survey covers far more volume than a mini-box
    simulation, so phi_obs is much better determined and the simulation is the
    noisy side.  Reversing it would attribute the shot noise to the survey.

    `err` is accepted for interface compatibility and deliberately ignored: the
    Poisson likelihood supplies its own uncertainty, which is the point.
    `count_scale` is dm * V for the constraint, so counts can be recovered from
    log10(phi); without it the statistic is undefined and this raises.
    """
    if count_scale is None:
        raise ValueError(
            'cash requires count_scale (bin width x volume) to recover counts '
            'from log10(phi); it is only defined for counting statistics such '
            'as the mass functions')

    N = np.power(10.0, np.asarray(mod, dtype=float)) * count_scale
    mu = np.power(10.0, np.asarray(obs, dtype=float)) * count_scale

    # Guard the logarithms: mu must be positive, and N ln(N/mu) -> 0 as N -> 0.
    mu = np.maximum(mu, 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(N > 0, N * np.log(N / mu), 0.0)
    return 2.0 * (mu - N + term)


# Statistics that need the constraint's count scale, so callers know to supply
# it and to fall back for constraints that are not counting statistics.
cash.needs_count_scale = True


stat_tests = {
    'student-t': studentT,
    'chi2': chi2,
    'huber': huber,
    'cauchy': cauchy,
    'abs': absolute,
    'cash': cash,
}
