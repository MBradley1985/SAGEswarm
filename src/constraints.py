#!/bin/bash

"""
Constraints for optimizers to evaluate shark models against observations
"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt # type: ignore
import os
from random import seed
from src import common
import numpy as np # type: ignore
import re
from scipy.interpolate import interp1d # type: ignore
from src import routines as r
import h5py as h5 # type: ignore
import pandas as pd # type: ignore
from scipy.stats import binned_statistic # type: ignore
import logging
import scipy.stats as stats
from src.simulation_config import get_csfrdh_snapshots, get_smd_snapshots, get_fics_snapshots

warnings.filterwarnings("ignore")
logging.getLogger('constraints').setLevel(logging.INFO)

GyrToYr = 1e9
#######################
# Binning configuration
mupp = 12.0
dm = 0.1
mlow = 6.5
mbins = np.arange(mlow, mupp, dm)
xmf = mbins[:-1] + dm/2.0

mupp2 = 10.5
dm2 = 0.1
mlow2 = 4
mbins2 = np.arange(mlow2,mupp2,dm2)
xmf2 = mbins2 + dm2/2.0

mupp_h1 = 11.0
dm_h1 = 0.1
mlow_h1 = 8
mbins_h1 = np.arange(mlow_h1, mupp_h1, dm_h1)
xmf_h1 = mbins_h1 + dm_h1/2.0

mupp_h2 = 10
dm_h2 = 0.1
mlow_h2 = 8
mbins_h2 = np.arange(mlow_h2,mupp_h2,dm_h2)
xmf_h2 = mbins_h2 + dm_h2/2.0

ssfrlow = -6
ssfrupp = 4
dssfr = 0.2
ssfrbins = np.arange(ssfrlow,ssfrupp,dssfr)

Nmin = 10 # minimum number of galaxies expected in a mass bin for the simulation volume, based on observations, to warrant fitting to that bin for mass functions

#######################
# Intracluster-star (ICS) fraction selection.
#
# f_ICS = m_ICS / M_*,halo, with M_*,halo = m_ICS + m_BCG + sum(m_satellites),
# measured for central galaxies of groups and clusters.  The cuts below mirror
# those used for the same quantity in the SAGE26 analysis
# (SAGE26/plotting/random_plotting_scripts/BCG_ICS_fraction.py).
#
# The halo-mass floor is deliberately low so the model averages over the same
# mix of systems as the compilation it is scored against: that compilation keeps
# its group-scale measurements (Ragusa+23 VEGAS groups, Ahad+25 KiDS+GAMA
# groups, ~25 of 86 points, probing 10^12.5-10^13.5) alongside the cluster
# samples, so restricting the model to cluster haloes would compare unlike
# populations.
#
# It also buys redshift coverage that a cluster-scale floor cannot give in a
# small box.  In miniMillennium (62.5/h Mpc) this cut selects 270-510 haloes at
# every snapshot from z = 0 to z = 2, where a 10^13.5 floor leaves 0-2 above
# z ~ 1 and drops those snapshots below FICS_MIN_HALOES entirely.
#
# Note that the other cuts raise the effective floor above 10^12 on their own:
# requiring a resolved BCG and at least two satellites means the selected haloes
# actually span 10^12.0-10^14.2 at z = 0.
FICS_MVIR_MIN = 10 ** 12.0          # Msun; groups and richer
FICS_STELLARMASS_MIN = 8.6e8        # Msun; BCG must be properly resolved
FICS_MIN_SATELLITES = 2             # must actually be a group/cluster
FICS_BCG_MVIR_RATIO_MIN = 10 ** -3.5  # reject pathological undersized centrals
                                      # (merger-tree edge cases with
                                      #  m_BCG/M_vir ~ 10^-5 instead of ~10^-2)
FICS_MIN_HALOES = 5                 # fewest selected haloes for a usable median
FICS_MVIR_BIN_WIDTH = 0.25          # dex, for the f_ICS-vs-halo-mass relation
FICS_MVIR_BIN_MIN = 5               # fewest haloes in a halo-mass bin to use it

#######################
# Galactic-wind mass loading, eta = Mdot_outflow / SFR, against circular
# velocity.  Applies to the MLF constraint only, on the same principle as the
# FICS_* cuts above: read nowhere but mlf_binned().
#
# SAGE only assigns MassLoading inside the FIRE branch of the supernova
# feedback (model_starformation_and_feedback.c:741), so the field is identically
# zero for FIREmodeOn = 0 and the constraint reports such a run as not
# applicable rather than as a fit.
MLF_SFR_MIN = 1e-3                  # Msun/yr; eta is undefined without star formation
MLF_VVIR_MIN = 20.0                 # km/s; below this the haloes are unresolved
MLF_BIN_WIDTH = 0.15                # dex in log10 v_circ
MLF_BIN_MIN = 20                    # fewest galaxies in a velocity bin to use it
#
# These cuts apply to the FICS constraint ONLY.  They are read nowhere but
# fics_median(), which returns a single number per snapshot and never touches the
# galaxy arrays that the mass functions, the MZR and the SHMR are built from.  If
# you ever need a halo cut for another constraint, add it in that constraint --
# do not filter the shared arrays in _load_model_data, which every constraint
# reads.  tests/test_fics_selection.py asserts this separation holds.

# These are two easily create variables of these different shapes without
# actually storing a reference ourselves; we don't need it
zeros1 = lambda: np.zeros(shape=(1, 3, len(xmf)))
zeros2 = lambda: np.zeros(shape=(1, 3, len(xmf2)))
zeros3 = lambda: np.zeros(shape=(1, len(mbins)))
zeros4 = lambda: np.empty(shape=(1), dtype=np.bool_)
zeros5 = lambda: np.zeros(shape=(1, len(ssfrbins)))
zeros6 = lambda: np.zeros(shape=(1, len(mbins2)))
zeros7 = lambda: np.zeros(shape=(1, len(mbins_h1)))
zeros8 = lambda: np.zeros(shape=(1, len(mbins_h2)))
zeros9 = lambda: np.zeros(shape=(1, len(mbins)))
zeros10 = lambda: np.zeros(shape=(1, 3, len(xmf_h1)))
zeros11 = lambda: np.zeros(shape=(1, 3, len(xmf_h2)))

def satellite_stellar_mass(G, stellar_mass):
    """Sum satellite stellar mass onto the central galaxy of each FOF group.

    Returns an array the same length as the catalogue holding, for every
    central, the summed stellar mass of its satellites (and zero for
    satellites themselves).  Satellites are matched to their host through
    CentralGalaxyIndex -> GalaxyIndex; satellites whose central is missing
    from the catalogue are dropped rather than mis-assigned.
    """
    galaxy_index = np.asarray(G['GalaxyIndex'])
    central_index = np.asarray(G['CentralGalaxyIndex'])
    is_satellite = np.asarray(G['Type']) != 0

    satellite_mass = np.zeros(len(stellar_mass))
    n_satellites = np.zeros(len(stellar_mass), dtype=np.int64)
    if len(galaxy_index) == 0 or not np.any(is_satellite):
        return satellite_mass, n_satellites

    order = np.argsort(galaxy_index)
    sorted_ids = galaxy_index[order]
    wanted = central_index[is_satellite]

    pos = np.searchsorted(sorted_ids, wanted)
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    matched = sorted_ids[pos] == wanted
    host = np.where(matched, order[pos], -1)
    good = host >= 0

    np.add.at(satellite_mass, host[good], stellar_mass[is_satellite][good])
    np.add.at(n_satellites, host[good], 1)
    return satellite_mass, n_satellites


def fics_per_halo(G, h0):
    """Per-halo ICS mass fraction for the groups and clusters of one snapshot.

    f_ICS = m_ICS / (m_ICS + m_BCG + sum m_satellites), evaluated for central
    galaxies passing the FICS_* cuts defined at the top of this module.
    Returns (log10 M_vir, f_ICS) for the selected haloes, both possibly empty.

    An empty return is also what happens for a SAGE build that does not output
    IntraClusterStars: read_sage_hdf fills absent fields with zeros and the
    m_ICS > 0 cut then rejects everything.  The FICS constraints report such a
    run as not applicable rather than as a fit.
    """
    stellar_mass = np.asarray(G['StellarMass'], dtype=np.float64) * 1e10 / h0
    ics_mass = np.asarray(G['IntraClusterStars'], dtype=np.float64) * 1e10 / h0
    mvir = np.asarray(G['Mvir'], dtype=np.float64) * 1e10 / h0
    is_central = np.asarray(G['Type']) == 0

    satellite_mass, n_satellites = satellite_stellar_mass(G, stellar_mass)
    total_stellar = stellar_mass + satellite_mass + ics_mass

    # m_BCG/M_vir guards against merger-tree edge cases that leave a central
    # orders of magnitude too small for its halo; those would otherwise
    # inflate f_ICS towards 1.
    with np.errstate(divide='ignore', invalid='ignore'):
        bcg_mvir_ratio = np.where(mvir > 0, stellar_mass / mvir, 0.0)

    sel = (is_central
           & (mvir >= FICS_MVIR_MIN)
           & (ics_mass > 0)
           & (stellar_mass >= FICS_STELLARMASS_MIN)
           & (total_stellar > 0)
           & (n_satellites >= FICS_MIN_SATELLITES)
           & (bcg_mvir_ratio >= FICS_BCG_MVIR_RATIO_MIN))

    if not np.any(sel):
        return np.array([]), np.array([])
    return np.log10(mvir[sel]), ics_mass[sel] / total_stellar[sel]


def mlf_binned(G):
    """Median mass-loading factor in bins of circular velocity.

    Returns (log10 v_circ, log10 eta, error on log10 eta) for the bins with
    enough star-forming galaxies to measure, all possibly empty.  Empty is also
    what a run with FIREmodeOn = 0 gives, since MassLoading is never assigned
    there.

    v_circ is Vvir, which SAGE stores in km/s directly (no h scaling), and eta
    is dimensionless, so neither needs converting.
    """
    eta = np.asarray(G['MassLoading'], dtype=np.float64)
    vvir = np.asarray(G['Vvir'], dtype=np.float64)
    sfr = (np.asarray(G['SfrDisk'], dtype=np.float64)
           + np.asarray(G['SfrBulge'], dtype=np.float64))

    sel = (np.isfinite(eta) & (eta > 0)
           & np.isfinite(vvir) & (vvir >= MLF_VVIR_MIN)
           & (sfr >= MLF_SFR_MIN))
    if not np.any(sel):
        return np.array([]), np.array([]), np.array([])

    log_v = np.log10(vvir[sel])
    log_eta = np.log10(eta[sel])

    edges = np.arange(np.floor(log_v.min() / MLF_BIN_WIDTH) * MLF_BIN_WIDTH,
                      log_v.max() + MLF_BIN_WIDTH, MLF_BIN_WIDTH)
    centres, medians, errors = [], [], []
    for i in range(len(edges) - 1):
        in_bin = (log_v >= edges[i]) & (log_v < edges[i + 1])
        n = int(np.count_nonzero(in_bin))
        if n < MLF_BIN_MIN:
            continue
        vals = log_eta[in_bin]
        # x at the median velocity of the bin's members: the velocity function
        # falls steeply, so members crowd the low-velocity edge.
        centres.append(float(np.median(log_v[in_bin])))
        medians.append(float(np.median(vals)))
        errors.append(max(1.253 * float(np.std(vals)) / np.sqrt(n), 1e-3))

    return np.array(centres), np.array(medians), np.array(errors)


def fics_median(G, h0):
    """Median ICS mass fraction over the groups and clusters of one snapshot.

    Returns (median, standard error of the median), or (nan, nan) when fewer
    than FICS_MIN_HALOES haloes qualify.
    """
    _, f_ics = fics_per_halo(G, h0)
    n_sel = len(f_ics)
    if n_sel < FICS_MIN_HALOES:
        return np.nan, np.nan

    median = float(np.median(f_ics))
    # Standard error of the median for a roughly normal sample; the 1.253
    # factor is sqrt(pi/2).  Floored so a coincidentally tight sample cannot
    # produce a zero model error.
    err = 1.253 * float(np.std(f_ics)) / np.sqrt(n_sel)
    return median, max(err, 1e-3)

class Constraint(object):
    """Base classes for constraint objects"""

    # Whether this constraint needs the per-halo ICS fraction.  Measuring it
    # costs an argsort over the whole catalogue per snapshot, and it applies a
    # halo-mass and satellite-count selection that is meaningful only for FICS,
    # so constraints that do not use it skip the work entirely.
    needs_fics = False

    # Whether this constraint needs the per-galaxy mass-loading factor.  Same
    # principle as needs_fics: the MLF_* cuts are meaningful only for MLF, and
    # a constraint that does not use them should not pay for the binning.
    needs_mlf = False

    # Bin width in dex for constraints whose y value is a number density built
    # by counting objects per bin (the mass functions).  Set it and count_scale
    # below recovers the underlying counts from log10(phi), which is what the
    # Poisson/Cash statistic needs.  None means "not a counting statistic", and
    # cash falls back to chi2 for such a constraint.
    bin_width = None

    @property
    def count_scale(self):
        """dm * V, converting phi [Mpc^-3 dex^-1] back into object counts.

        A mass function's model value is a histogram: N = phi * dm * V.  Anything
        that wants to treat those counts as Poisson has to be able to undo the
        normalisation, and this is the factor that does it.
        """
        if self.bin_width is None:
            return None
        return self.bin_width * self.vol

    def __init__(self, snapshot=None, sim=None, boxsize=None, vol_frac=None, age_alist_file=None, Omega0=None, h0=None, output_dir=None):
        self.redshift_table = None
        self.weight = 1
        self.rel_weight = 1
        self.snapshot = snapshot
        self.output_dir = output_dir

        # Set defaults if not provided
        self.sim = 0 if sim is None else sim
        self.boxsize = 400.0 if sim is None else boxsize
        self.vol_frac = 0.0019 if vol_frac is None else vol_frac
        self.Omega0 = 0.3089 if Omega0 is None else Omega0
        self.h0 = 0.677400 if h0 is None else h0
        self.age_alist_file = '/fred/oz004/msinha/simulations/uchuu_suite/miniuchuu/mergertrees/u400_planck2016_50.a_list' if age_alist_file is None else age_alist_file

        # Set simulation parameters
        if sim == 0:  # miniUchuu
            self.h0 = h0
            self.Omega0 = Omega0
            self.vol_frac = vol_frac
            self.vol = (boxsize/h0)**3 * vol_frac
            self.age_alist_file = age_alist_file
        elif sim == 1:  # miniMillennium 
            self.h0 = h0
            self.Omega0 = Omega0
            self.vol_frac = vol_frac
            self.vol = (boxsize/h0)**3 * vol_frac
            self.age_alist_file = age_alist_file
        else:  # MTNG
            self.h0 = h0
            self.Omega0 = Omega0
            self.vol_frac = vol_frac
            self.vol = (boxsize/h0)**3 * vol_frac
            self.age_alist_file = age_alist_file

    def _fics_for_snapshot(self, G):
        """Median ICS mass fraction over the groups and clusters of one snapshot."""
        return fics_median(G, self.h0)

    def _load_model_data(self, modeldir, subvols):
        # Allow snapshots to be a list
        if not isinstance(self.snapshot, list):
            self.snapshot = [self.snapshot]
        
        # For constraints that need multiple snapshots (like CSFRDH and SMD), we'll collect SFRD and SMD for each
        Nage = 14
        num_snapshots = len(self.snapshot)
        SFRbyAge = np.zeros(num_snapshots)
        SMD_history = np.zeros(num_snapshots) # [FIX] Array to store SMD history
        # ICS mass fraction history: median f_ICS over the selected groups and
        # clusters at each snapshot, plus the standard error on that median.
        # NaN marks a snapshot with too few haloes to measure (see FICS).
        FICS_history = np.full(num_snapshots, np.nan)
        FICS_history_err = np.full(num_snapshots, np.nan)
        # Per-halo f_ICS at the reference snapshot, for the f_ICS-vs-halo-mass
        # relation.  Only that snapshot is needed, so unlike FICS_history these
        # are filled once rather than per snapshot.
        FICS_halo_mass = np.array([])
        FICS_halo_frac = np.array([])
        # Mass-loading relation at the reference snapshot, for MLF.
        MLF_x = np.array([])
        MLF_y = np.array([])
        MLF_err = np.array([])
        SnapshotTimes = np.zeros(num_snapshots)  # Store lookback time for each snapshot
        
        # For hist_smf, hist_bhmf, etc., always use the last (most recent) snapshot
        # This ensures SMF_z0 uses snapshot 63 even when combined with CSFRDH
        snap_to_use = self.snapshot[-1]
        
        # Loop through snapshots to build SFRD and SMD history
        for snap_idx, snap in enumerate(self.snapshot):
            if len(subvols) > 1:
                subvols = ["multiple_batches"]

            seed(2222)
            fields = ['StellarMass', 'BlackHoleMass', 'Len', 'SfrBulge', 'BulgeMass', 'Mvir', 'SfrDisk', 'ColdGas', 'H1gas', 'H2gas', 'MetalsColdGas', 'IntraClusterStars', 'Type', 'GalaxyIndex', 'CentralGalaxyIndex', 'MassLoading', 'Vvir']
            snap_num = f'Snap_{snap}'
            sSFRcut = -11.0

            # Get list of model files in directory
            model_files = [f for f in os.listdir(modeldir) if f.startswith('model_') and f.endswith('.hdf5')]
            model_files.sort()

            if len(model_files) > 1:
                combined_properties = {}
                for model_file in model_files:
                    G = r.read_sage_hdf(os.path.join(modeldir, model_file), snap_num=snap_num, fields=fields)
                    
                    # Combine properties
                    for field in fields:
                        if field not in combined_properties:
                            combined_properties[field] = G[field]
                        else:
                            combined_properties[field] = np.concatenate((combined_properties[field], G[field]))
                
                G = combined_properties
            else:
                G = r.read_sage_hdf(os.path.join(modeldir, 'model_0.hdf5'), snap_num=snap_num, fields=fields)

            # Calculate SFRD for this snapshot
            total_SFR = np.sum(G['SfrBulge'] + G['SfrDisk'])
            SFRbyAge[snap_idx] = total_SFR / self.vol  # Msun/yr/Mpc^3
            
            # [FIX] Calculate SMD for THIS snapshot (for history constraints)
            stellar_mass_total_snap = np.sum(G['StellarMass'] * 1e10 / self.h0)
            SMD_history[snap_idx] = stellar_mass_total_snap / self.vol # Msun / Mpc^3

            # Calculate the ICS mass fraction for THIS snapshot (for the FICS
            # history constraint).  f_ICS is a per-halo quantity, so it needs the
            # stellar mass of every satellite summed onto its central before the
            # ratio can be formed.  Only FICS asks for this: its halo-mass and
            # satellite-count cuts have no bearing on any other constraint, and
            # skipping it avoids an argsort over the catalogue.
            if self.needs_fics:
                FICS_history[snap_idx], FICS_history_err[snap_idx] = \
                    self._fics_for_snapshot(G)

            # Store snapshot number to calculate time later (after alist is loaded)
            # We'll calculate the times after loading the alist properly below
            
            # For the reference snapshot (used by SMF, BHMF, etc.), save the properties
            if snap == snap_to_use:
                # Process properties - Use self.h0 instead of h0
                BlackHoleMass = np.log10(G['BlackHoleMass'] * 1e10 / self.h0)
                BlackHoleMass[~np.isfinite(BlackHoleMass)] = -20

                BulgeMass = np.log10(G['BulgeMass'] * 1e10 / self.h0)
                BulgeMass[~np.isfinite(BulgeMass)] = -20

                HaloMass = np.log10(G['Mvir'] * 1e10 / self.h0)
                HaloMass[~np.isfinite(HaloMass)] = -20

                StellarMass = np.log10(G['StellarMass'] * 1e10 / self.h0)
                StellarMass[~np.isfinite(StellarMass)] = -20

                logSM = np.log10(G['StellarMass'] * 1e10 / self.h0)
                logSM[~np.isfinite(logSM)] = -20

                logBHM = np.log10(G['BlackHoleMass'] * 1e10 / self.h0)
                logBHM[~np.isfinite(logBHM)] = -20

                smass = (G['StellarMass'] * 1e10 / self.h0)
                SfrDisk = G['SfrDisk']
                SfrBulge = G['SfrBulge']
                
                # calculate all
                w = np.where(smass > 0.0)[0]
                mass = np.log10(smass[w])
                # sSFR = SFR / M_stellar, so log(sSFR) = log(SFR) - log(M)
                # Use smass (linear) not StellarMass (which is already log)
                total_sfr = SfrDisk[w] + SfrBulge[w]
                # Handle zero SFR by setting to very low sSFR
                sSFR = np.full(len(w), -20.0)  # default to very low (quiescent)
                sfr_positive = total_sfr > 0
                sSFR[sfr_positive] = np.log10(total_sfr[sfr_positive] / smass[w][sfr_positive])

                # additionally calculate red (sSFR < -11 is quiescent/red)
                w_red = np.where(sSFR < sSFRcut)[0]
                massRED = mass[w_red]
                (hist_smf_red_counts, binedges) = np.histogram(massRED, bins=mbins)
                hist_smf_red = hist_smf_red_counts / dm / self.vol

                # additionally calculate blue (sSFR > -11 is star-forming/blue)
                w_blue = np.where(sSFR > sSFRcut)[0]
                massBLU = mass[w_blue]
                (hist_smf_blue_counts, binedges) = np.histogram(massBLU, bins=mbins)
                hist_smf_blue = hist_smf_blue_counts / dm / self.vol

                # Calculate SMF and track galaxy counts for Poisson errors
                hist_smf_counts, _ = np.histogram(logSM, bins=mbins)
                hist_smf = hist_smf_counts / dm / self.vol

                # Calculate BHMF and track BH counts for Poisson errors
                hist_bhmf_counts, _ = np.histogram(logBHM, bins=mbins2)
                hist_bhmf = hist_bhmf_counts / dm2 / self.vol
                
                # Calculate HIMF (HI Mass Function)
                # HI mass is now output directly by SAGE as H1gas
                HI_mass = G['H1gas'] * 1e10 / self.h0  # Convert to Msun
                logHI = np.log10(HI_mass)
                logHI[~np.isfinite(logHI)] = -20
                
                hist_himf_counts, _ = np.histogram(logHI, bins=mbins_h1)
                hist_himf = hist_himf_counts / dm_h1 / self.vol

                # Calculate H2MF (H2 Mass Function)
                H2_mass = G['H2gas'] * 1e10 / self.h0  # Convert to Msun
                logH2 = np.log10(H2_mass)
                logH2[~np.isfinite(logH2)] = -20

                hist_h2mf_counts, _ = np.histogram(logH2, bins=mbins_h2)
                hist_h2mf = hist_h2mf_counts / dm_h2 / self.vol
                # [FIX] smd is now calculated per snapshot above as SMD_history. 
                # For compatibility, we can set a scalar 'smd' variable here for the last snapshot
                # but we will return SMD_history.
                smd_scalar = SMD_history[snap_idx] 

                # Calculate Mass-Metallicity Relation (MZR)
                # Z = log10((MetalsColdGas / ColdGas) / 0.02) + 9.0
                w_mzr = np.where((G['ColdGas'] > 0) & (G['MetalsColdGas'] > 0) & (G['StellarMass'] > 0))[0]
                if len(w_mzr) > 0:
                    metallicity = np.log10((G['MetalsColdGas'][w_mzr] / G['ColdGas'][w_mzr]) / 0.02) + 8.69
                    stellar_mass_mzr = np.log10(G['StellarMass'][w_mzr] * 1e10 / self.h0)
                else:
                    metallicity = np.array([])
                    stellar_mass_mzr = np.array([])

                # Per-halo f_ICS at this (reference) snapshot, for FICS_Mvir.
                # Gated the same way as FICS_history: only a constraint that
                # asks for it pays the cost, and the FICS_* halo cuts stay
                # confined to those constraints.
                if self.needs_fics:
                    FICS_halo_mass, FICS_halo_frac = fics_per_halo(G, self.h0)

                # Mass-loading vs circular velocity, gated the same way.
                if self.needs_mlf:
                    MLF_x, MLF_y, MLF_err = mlf_binned(G)

                # Calculate Stellar-Halo Mass Relation (SHMR)
                w_shmr = np.where((G['Mvir'] > 0) & (G['StellarMass'] > 0))[0]
                if len(w_shmr) > 0:
                    halo_mass_shmr = np.log10(G['Mvir'][w_shmr] * 1e10 / self.h0)
                    stellar_mass_shmr = np.log10(G['StellarMass'][w_shmr] * 1e10 / self.h0)
                else:
                    halo_mass_shmr = np.array([])
                    stellar_mass_shmr = np.array([])

        # Get the edges of the age bins (after the loop).
        # alist is only required for SFRD/SMD history constraints; skip gracefully when absent.
        if self.age_alist_file is not None:
            alist_full = np.loadtxt(self.age_alist_file)

            for snap_idx, snap in enumerate(self.snapshot):
                if snap < len(alist_full):
                    redshift = 1.0 / alist_full[snap] - 1.0
                    SnapshotTimes[snap_idx] = r.z2tL(redshift, self.h0, self.Omega0, 1.0-self.Omega0)

            if Nage >= len(alist_full) - 1:
                alist = alist_full[::-1]
                RedshiftBinEdge = 1./ alist - 1.
            else:
                indices_float = np.arange(Nage+1) * (len(alist_full)-1.0) / Nage
                indices = indices_float.astype(np.int32)
                alist = alist_full[indices][::-1]
                RedshiftBinEdge = 1./ alist - 1.
            TimeBinEdge = np.array([r.z2tL(redshift, self.h0, self.Omega0, 1.0-self.Omega0) for redshift in RedshiftBinEdge])
            dT = np.diff(TimeBinEdge)
            TimeBinCentre = TimeBinEdge[:-1] + 0.5*dT
        else:
            TimeBinEdge = np.zeros(Nage + 1)
            dT = np.ones(Nage)
            TimeBinCentre = np.zeros(Nage)

        #########################
        # Calculate Poisson errors before taking logs.
        # For empty bins (N=0), use the Poisson upper limit: treating 0 detections as
        # N_eff=0.5 gives phi_upper = 0.5/(dm*vol) — physically "fewer than 1 galaxy
        # in the box at this mass".  The corresponding log-space error is log10(2)~0.3 dex.
        # This replaces the old -20 / 999 sentinels that corrupted np.interp and gave
        # the PSO a nonsensical (huge or zero) chi² signal for high-z empty bins.
        phi_empty     = 0.5 / (dm * self.vol)          # Poisson UL for 0 detections
        phi_empty_log = np.log10(phi_empty)             # typically ~ -5 for Millennium
        phi_empty_err = np.log10(2.0)                   # ~0.3 dex

        # SMF errors
        hist_smf_err = np.zeros_like(hist_smf)
        for i in range(len(hist_smf)):
            if hist_smf_counts[i] >= 1:
                # Calculate error from Poisson statistics
                phi_upper = (hist_smf_counts[i] + np.sqrt(hist_smf_counts[i])) / dm / self.vol
                phi_lower = np.maximum((hist_smf_counts[i] - np.sqrt(hist_smf_counts[i])), 0.5) / dm / self.vol
                if hist_smf[i] > 0:
                    # Symmetric error in log space (average of upper and lower)
                    err_up = np.log10(phi_upper) - np.log10(hist_smf[i])
                    err_dn = np.log10(hist_smf[i]) - np.log10(phi_lower)
                    hist_smf_err[i] = (err_up + err_dn) / 2.0
                else:
                    hist_smf_err[i] = phi_empty_err
            else:
                hist_smf_err[i] = phi_empty_err
        
        # BHMF errors
        hist_bhmf_err = np.zeros_like(hist_bhmf)
        for i in range(len(hist_bhmf)):
            if hist_bhmf_counts[i] >= 1:
                # Calculate error from Poisson statistics
                phi_upper = (hist_bhmf_counts[i] + np.sqrt(hist_bhmf_counts[i])) / dm2 / self.vol
                phi_lower = np.maximum((hist_bhmf_counts[i] - np.sqrt(hist_bhmf_counts[i])), 0.5) / dm2 / self.vol
                if hist_bhmf[i] > 0:
                    # Symmetric error in log space (average of upper and lower)
                    err_up = np.log10(phi_upper) - np.log10(hist_bhmf[i])
                    err_dn = np.log10(hist_bhmf[i]) - np.log10(phi_lower)
                    hist_bhmf_err[i] = (err_up + err_dn) / 2.0
                else:
                    hist_bhmf_err[i] = phi_empty_err
            else:
                hist_bhmf_err[i] = phi_empty_err

        # HIMF errors
        hist_himf_err = np.zeros_like(hist_himf)
        for i in range(len(hist_himf)):
            if hist_himf_counts[i] >= 1:
                # Calculate error from Poisson statistics
                phi_upper = (hist_himf_counts[i] + np.sqrt(hist_himf_counts[i])) / dm_h1 / self.vol
                phi_lower = np.maximum((hist_himf_counts[i] - np.sqrt(hist_himf_counts[i])), 0.5) / dm_h1 / self.vol
                if hist_himf[i] > 0:
                    # Symmetric error in log space (average of upper and lower)
                    err_up = np.log10(phi_upper) - np.log10(hist_himf[i])
                    err_dn = np.log10(hist_himf[i]) - np.log10(phi_lower)
                    hist_himf_err[i] = (err_up + err_dn) / 2.0
                else:
                    hist_himf_err[i] = phi_empty_err
            else:
                hist_himf_err[i] = phi_empty_err

        # H2MF errors
        hist_h2mf_err = np.zeros_like(hist_h2mf)
        for i in range(len(hist_h2mf)):
            if hist_h2mf_counts[i] >= 1:
                phi_upper = (hist_h2mf_counts[i] + np.sqrt(hist_h2mf_counts[i])) / dm_h2 / self.vol
                phi_lower = np.maximum((hist_h2mf_counts[i] - np.sqrt(hist_h2mf_counts[i])), 0.5) / dm_h2 / self.vol
                if hist_h2mf[i] > 0:
                    err_up = np.log10(phi_upper) - np.log10(hist_h2mf[i])
                    err_dn = np.log10(hist_h2mf[i]) - np.log10(phi_lower)
                    hist_h2mf_err[i] = (err_up + err_dn) / 2.0
                else:
                    hist_h2mf_err[i] = phi_empty_err
            else:
                hist_h2mf_err[i] = phi_empty_err

        # SMF Red errors
        hist_smf_red_err = np.zeros_like(hist_smf_red)
        for i in range(len(hist_smf_red)):
            if hist_smf_red_counts[i] >= 1:
                phi_upper = (hist_smf_red_counts[i] + np.sqrt(hist_smf_red_counts[i])) / dm / self.vol
                phi_lower = np.maximum((hist_smf_red_counts[i] - np.sqrt(hist_smf_red_counts[i])), 0.5) / dm / self.vol
                if hist_smf_red[i] > 0:
                    err_up = np.log10(phi_upper) - np.log10(hist_smf_red[i])
                    err_dn = np.log10(hist_smf_red[i]) - np.log10(phi_lower)
                    hist_smf_red_err[i] = (err_up + err_dn) / 2.0
                else:
                    hist_smf_red_err[i] = phi_empty_err
            else:
                hist_smf_red_err[i] = phi_empty_err

        # SMF Blue errors
        hist_smf_blue_err = np.zeros_like(hist_smf_blue)
        for i in range(len(hist_smf_blue)):
            if hist_smf_blue_counts[i] >= 1:
                phi_upper = (hist_smf_blue_counts[i] + np.sqrt(hist_smf_blue_counts[i])) / dm / self.vol
                phi_lower = np.maximum((hist_smf_blue_counts[i] - np.sqrt(hist_smf_blue_counts[i])), 0.5) / dm / self.vol
                if hist_smf_blue[i] > 0:
                    err_up = np.log10(phi_upper) - np.log10(hist_smf_blue[i])
                    err_dn = np.log10(hist_smf_blue[i]) - np.log10(phi_lower)
                    hist_smf_blue_err[i] = (err_up + err_dn) / 2.0
                else:
                    hist_smf_blue_err[i] = phi_empty_err
            else:
                hist_smf_blue_err[i] = phi_empty_err

        #########################
        # take logs — empty bins get Poisson UL (phi_empty_log) instead of -20 sentinel
        ind = (hist_smf > 0.)
        hist_smf[ind] = np.log10(hist_smf[ind])
        hist_smf[~ind] = phi_empty_log

        ind = (hist_smf_red > 0.)
        hist_smf_red[ind] = np.log10(hist_smf_red[ind])
        hist_smf_red[~ind] = phi_empty_log

        ind = (hist_smf_blue > 0.)
        hist_smf_blue[ind] = np.log10(hist_smf_blue[ind])
        hist_smf_blue[~ind] = phi_empty_log

        phi_empty_bhmf = np.log10(0.5 / (dm2 * self.vol))
        ind = (hist_bhmf > 0.)
        hist_bhmf[ind] = np.log10(hist_bhmf[ind])
        hist_bhmf[~ind] = phi_empty_bhmf

        phi_empty_himf = np.log10(0.5 / (dm_h1 * self.vol))
        ind = (hist_himf > 0.)
        hist_himf[ind] = np.log10(hist_himf[ind])
        hist_himf[~ind] = phi_empty_himf

        phi_empty_h2mf = np.log10(0.5 / (dm_h2 * self.vol))
        ind = (hist_h2mf > 0.)
        hist_h2mf[ind] = np.log10(hist_h2mf[ind])
        hist_h2mf[~ind] = phi_empty_h2mf

        SFRD_Age = np.log10(SFRbyAge)
        SFRD_Age[~np.isfinite(SFRD_Age)] = -20

        hist_bhmf = hist_bhmf[np.newaxis]
        hist_smf = hist_smf[np.newaxis]
        hist_smf_red = hist_smf_red[np.newaxis]
        hist_smf_blue = hist_smf_blue[np.newaxis]
        hist_smf_err = hist_smf_err[np.newaxis]
        hist_smf_red_err = hist_smf_red_err[np.newaxis]
        hist_smf_blue_err = hist_smf_blue_err[np.newaxis]
        hist_bhmf_err = hist_bhmf_err[np.newaxis]
        hist_himf = hist_himf[np.newaxis]
        hist_himf_err = hist_himf_err[np.newaxis]
        hist_h2mf = hist_h2mf[np.newaxis]
        hist_h2mf_err = hist_h2mf_err[np.newaxis]

        # [FIX] Return SMD_history instead of scalar 'smd'
        if num_snapshots > 1:
            return self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, SnapshotTimes, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, SMD_history, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, FICS_history, FICS_history_err, FICS_halo_mass, FICS_halo_frac, MLF_x, MLF_y, MLF_err
        else:
            return self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, SMD_history, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, FICS_history, FICS_history_err, FICS_halo_mass, FICS_halo_frac, MLF_x, MLF_y, MLF_err


    def load_observation(self, *args, **kwargs):
        obsdir = os.path.normpath(os.path.abspath(os.path.join(__file__, '..')))
        return common.load_observation(obsdir, *args, **kwargs)

    def plot_diagnostic(self, x_obs, y_obs, obs_err_dn, obs_err_up,
                       x_mod, y_mod, y_mod_interp, x_obs_sel, y_obs_sel, y_mod_sel, err, output_dir):
        """Generic diagnostic plot showing all interpolation/selection steps"""
        from src import analysis
        
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111)
        
        # Domain boundaries
        ax.axvline(self.domain[0], ls='dotted', c='red', alpha=0.5, label='Domain')
        ax.axvline(self.domain[1], ls='dotted', c='red', alpha=0.5)
        
        # Plot data (raw model last so it renders on top of interp line)
        ax.plot(x_obs_sel, y_obs_sel, marker='v', ls='None', c='blue', markersize=6, label="Selected obs")
        ax.plot(x_obs, y_mod_interp, ls='solid', c='green', linewidth=2, label="Interp model")
        ax.plot(x_obs_sel, y_mod_sel, ls='None', marker='o', c='brown', markersize=5, label="Selected model")
        ax.plot(x_mod, y_mod, marker='^', ls='None', c='orange', markersize=7, zorder=5, label="Raw model")
        
        # Error bars - ensure errors are positive
        for i, x_val in enumerate(x_obs):
            ax.errorbar(x_val, y_obs[i], 
                       yerr=[[np.abs(obs_err_dn[i])], [np.abs(obs_err_up[i])]], 
                       fmt='none', c='black', alpha=0.5, capsize=3)
        
        # Calculate metrics
        chi2 = analysis.chi2(y_obs_sel, y_mod_sel, err)
        st = analysis.studentT(y_obs_sel, y_mod_sel, err)
        
        ax.set_title(r'%s' % str(self) + '\n$\\chi^2$ = %.2f, student-t = %.2f' % (chi2, st), fontsize=10)
        ax.legend(loc='best', fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        
        plotfile = os.path.join(output_dir, f'{self.__class__.__name__}_diagnostic.png')
        plt.savefig(plotfile, dpi=100, bbox_inches='tight')
        plt.close()
        return

    def _get_raw_data(self, modeldir, subvols):
        """Gets the model and observational data for further analysis.
        The model data is interpolated to match the observation's X values."""

        self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err = self._load_model_data(modeldir, subvols)
        x_obs, y_obs, y_dn, y_up = self.get_obs_x_y_err()
        x_sage, y_sage = self.get_sage_x_y()
        x_mod, y_mod, y_mod_err = self.get_model_x_y(hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err)
        return x_obs, y_obs, y_dn, y_up, x_sage, y_sage, x_mod, y_mod, y_mod_err

    def get_data(self, modeldir, subvols):

        self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err = self._load_model_data(modeldir, subvols)
        x_obs, y_obs, y_dn, y_up = self.get_obs_x_y_err()
        x_sage, y_sage = self.get_sage_x_y()
        x_mod, y_mod, y_mod_err = self.get_model_x_y(hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err)

        # Both observations and model values don't come necessarily in order,
        # but if at the end of the day we want to perform array-wise operations
        # over them (to calculate chi2 or student-t) then they should be both in
        # ascending order
        sorted_obs = np.argsort(x_obs)
        x_obs = x_obs[sorted_obs]
        y_obs = y_obs[sorted_obs]
        y_dn = y_dn[sorted_obs]
        y_up = y_up[sorted_obs]
        
        sorted_mod = np.argsort(x_mod)
        x_mod = x_mod[sorted_mod]
        y_mod = y_mod[sorted_mod]
        y_mod_err = y_mod_err[sorted_mod]
        
        # Model errors now come from proper Poisson statistics calculated in _load_model_data
        # (already computed above from get_model_x_y)

        # Linearly interpolate model Y values respect to the observations'
        # X values, and only take those within the domain. We do the same 
        # for the errors of the model.
        # We also consider the biggest relative error as "the" error, in case
        # they are different and add it in quadrature to the Poisson error
        # of the model.
        y_mod_interp = np.interp(x_obs, x_mod, y_mod)
        y_mod_err_interp = np.interp(x_obs, x_mod, y_mod_err)
        
        sel = np.where((x_obs >= self.domain[0]) & (x_obs <= self.domain[1]))
        x_obs_sel = x_obs[sel]
        y_obs_sel = y_obs[sel]
        y_mod_sel = y_mod_interp[sel]
        y_mod_err_sel = y_mod_err_interp[sel]
        y_obs_err_sel = np.maximum(np.abs(y_dn[sel]), np.abs(y_up[sel]))
        err = np.sqrt(y_obs_err_sel ** 2.0 + y_mod_err_sel ** 2.0)
        err = np.maximum(err, 1e-4)   # guard against zero errors causing inf chi²

        # Write dump file for tracking
        if self.output_dir:
            constraint_name = self.__class__.__name__
            filename = os.path.join(self.output_dir, f"{constraint_name}_dump.txt")
            with open(filename, 'a') as f:
                # err is written too so diagnostics can recompute the score with
                # either stat test (chi2 or student-t) from the dump alone,
                # rather than only being able to plot the curves.
                f.write(f"# New Data Block\n")
                for x_val, y_val, mod_y_val, err_val in zip(
                        x_obs_sel, y_obs_sel, y_mod_sel, err):
                    f.write(f"{x_val}\t{y_val}\t{mod_y_val}\t{err_val}\n")

            self.plot_diagnostic(x_obs, y_obs, y_dn, y_up,
                               x_mod, y_mod, y_mod_interp,
                               x_obs_sel, y_obs_sel, y_mod_sel, err,
                               self.output_dir)

        return y_obs_sel, y_mod_sel, err

    def __str__(self):
        s = '%s, low=%.1f, up=%.1f, weight=%.2f, rel_weight=%.2f'
        args = self.__class__.__name__, self.domain[0], self.domain[1], self.weight, self.rel_weight
        return s % args

class BHMF(Constraint):
    """Common logic for BHMF constraints"""

    bin_width = dm2   # counts per dm2 dex bin -> Poisson/Cash

    domain = (7.0, 10.5)  # below 10^7 Msun is TRINITY extrapolation, dominated by SAGE seed BHs
    # How many points to keep from the TRINITY curve.  It is a model, sampled
    # every 0.1 dex; scoring all ~36 rows inside the domain would give the BHMF
    # six times the point count of the black hole-bulge relation for no extra
    # information, since neighbouring rows are not independent.
    n_obs_points = 14

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        y = hist_bhmf[0]
        yerr = hist_bhmf_err[0]
        ind = np.where(y < 0.)

        return xmf2[ind], y[ind], yerr[ind]
    
class BHMF_z0(BHMF):
    """The BHMF constraint at z=0"""
    
    z = [0]

    def get_obs_x_y_err(self):
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        # TRINITY Paper IV (arXiv:2309.07210), z=0.1, Planck 2015 cosmology (h=0.6774)
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'fig4_bhmf_z0.1.txt'))
        logm = obs_data[:, 0]           # log10(Mbh [Msun]) — physical (file header)
        phi = obs_data[:, 1]            # BHMF_best [Mpc^-3 dex^-1] — physical (file header)
        phi_16th = obs_data[:, 2]
        phi_84th = obs_data[:, 3]

        # Data is in physical Msun / Mpc^-3 at h_obs=0.6774 (Planck 2015).
        # Convert physical-at-h_obs to physical-at-h0: M ∝ H0^-2, phi ∝ H0^3
        h_obs = 0.6774
        x_obs = logm + 2.0 * np.log10(h_obs / self.h0)
        logphi      = np.log10(phi)      + 3.0 * np.log10(self.h0 / h_obs)
        logphi_16th = np.log10(phi_16th) + 3.0 * np.log10(self.h0 / h_obs)
        logphi_84th = np.log10(phi_84th) + 3.0 * np.log10(self.h0 / h_obs)

        y_dn = logphi - logphi_16th
        y_up = logphi_84th - logphi

        valid_mask = ~np.isnan(x_obs) & ~np.isnan(logphi) & ~np.isnan(y_dn) & ~np.isnan(y_up)
        x_obs, logphi = x_obs[valid_mask], logphi[valid_mask]
        y_dn, y_up = y_dn[valid_mask], y_up[valid_mask]

        # Thin the smooth TRINITY curve inside the scored domain: it is sampled
        # every 0.1 dex, so the ~36 rows in (7.0, 10.5) are perfectly correlated
        # model points, not independent measurements.
        in_domain = (x_obs >= self.domain[0]) & (x_obs <= self.domain[1])
        idx = np.flatnonzero(in_domain)[
            _subsample_curve(x_obs[in_domain], self.n_obs_points)]
        return x_obs[idx], logphi[idx], y_dn[idx], y_up[idx]

    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_bhmf_all_redshifts.csv', cols=[0,1])
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage

class BHMF_z10(BHMF):
    """The BHMF constraint at z=1.0"""

    z = [1.0]

    def get_obs_x_y_err(self):
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        # TRINITY Paper IV (arXiv:2309.07210), z=1.0, Planck 2015 cosmology (h=0.6774)
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'fig4_bhmf_z1.0.txt'))
        logm = obs_data[:, 0]           # log10(Mbh [Msun]) — physical (file header)
        phi = obs_data[:, 1]            # BHMF_best [Mpc^-3 dex^-1] — physical (file header)
        phi_16th = obs_data[:, 2]
        phi_84th = obs_data[:, 3]

        # Data is in physical Msun / Mpc^-3 at h_obs=0.6774 (Planck 2015).
        h_obs = 0.6774
        x_obs = logm + 2.0 * np.log10(h_obs / self.h0)
        logphi      = np.log10(phi)      + 3.0 * np.log10(self.h0 / h_obs)
        logphi_16th = np.log10(phi_16th) + 3.0 * np.log10(self.h0 / h_obs)
        logphi_84th = np.log10(phi_84th) + 3.0 * np.log10(self.h0 / h_obs)

        y_dn = logphi - logphi_16th
        y_up = logphi_84th - logphi

        valid_mask = ~np.isnan(x_obs) & ~np.isnan(logphi) & ~np.isnan(y_dn) & ~np.isnan(y_up)
        x_obs, logphi = x_obs[valid_mask], logphi[valid_mask]
        y_dn, y_up = y_dn[valid_mask], y_up[valid_mask]

        # Thin the smooth TRINITY curve inside the scored domain: it is sampled
        # every 0.1 dex, so the ~36 rows in (7.0, 10.5) are perfectly correlated
        # model points, not independent measurements.
        in_domain = (x_obs >= self.domain[0]) & (x_obs <= self.domain[1])
        idx = np.flatnonzero(in_domain)[
            _subsample_curve(x_obs[in_domain], self.n_obs_points)]
        return x_obs[idx], logphi[idx], y_dn[idx], y_up[idx]
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_bhmf_all_redshifts.csv', cols=[4,5])
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage



# ---------------------------------------------------------------------------
# Modern observational loaders (added 29 Aug 2026).
#
# The calibration previously constrained against older data than the paper
# compares its figures to -- COSMOS2020 in the constraints vs COSMOS-Web in
# Figure 12, D'Silva+2023 vs COSMOS-Web in Figure 3, Li+2009 at z=0 vs GAMA in
# Figure 6.  These close that gap so the model is fitted to the same data it is
# shown against.
#
# On COMBINING datasets: concatenating surveys that OVERLAP in mass or redshift
# double-weights the overlapping bins and inflates chi2 with inter-survey
# systematic offsets, which are not model error.  Combining is therefore done
# only where ranges are complementary; elsewhere the most recent appropriate
# survey is used on its own.
# ---------------------------------------------------------------------------

def _subsample_curve(x, n_target):
    """Indices of ~n_target evenly spaced points along x.

    The TRINITY black hole mass functions are smooth model curves sampled every
    0.1 dex, not independent measurements.  Scoring every row gives one
    constraint tens of perfectly correlated points and a correspondingly
    inflated share of the objective (the same reason CSFRDH subsamples its
    4001-row table).  Thinning to a handful of points keeps the shape without
    the double-counting.
    """
    x = np.asarray(x, dtype=float)
    if len(x) <= n_target:
        return np.arange(len(x))
    targets = np.linspace(x.min(), x.max(), n_target)
    return np.unique([int(np.argmin(np.abs(x - t))) for t in targets])


def _np_load_cols(relpath, cols, skiprows=0):
    """Load whitespace columns relative to src/.

    ECSV files carry an uncommented column-name row after the '#'-prefixed
    header.  np.loadtxt's skiprows counts every line including comments, so it
    cannot skip that row reliably; instead drop '#' lines and any leading row
    that will not parse as numbers.
    """
    import os as _os
    f = _os.path.join(_os.path.dirname(__file__), relpath)
    rows = [l.split() for l in open(f)
            if not l.lstrip().startswith('#') and l.strip()]
    rows = rows[skiprows:]
    while rows:
        try:
            [float(rows[0][i]) for i in cols]
            break
        except (ValueError, IndexError):
            rows = rows[1:]
    arr = np.array([[float(rw[i]) for i in cols] for rw in rows])
    return tuple(arr[:, k] for k in range(len(cols)))


def _load_paper_smf(zlo, zhi, h_model, labels=None):
    """SMF observations exactly as plotted in the paper's Figure 12.

    Reads data/paper_smf_observations.csv, generated from the paper's own
    loader by data/build_paper_observations.py, so the calibration is fitted to
    the same data the model is shown against.  log_phi and the errors are
    already in dex and already carry the h and IMF corrections paper_plots
    applies, so nothing further is done to them here.

    zlo, zhi  select datasets whose quoted redshift falls in [zlo, zhi).
    labels    optional whitelist, e.g. ('Baldry+08',), for redshift bins where
              several surveys overlap and combining them would double-weight
              shared mass bins.
    """
    import os as _os
    f = _os.path.join(_os.path.dirname(__file__), '../data/paper_smf_observations.csv')
    rows = [l.split() for l in open(f)
            if not l.lstrip().startswith('#') and l.strip()][1:]
    lm, lp, el, eh = [], [], [], []
    seen = set()
    for rw in rows:
        lab, z = rw[0], float(rw[1])
        if not (zlo <= z < zhi):
            continue
        if labels is not None and lab.replace('_', ' ') not in labels:
            continue
        seen.add(lab)
        lm.append(float(rw[2])); lp.append(float(rw[3]))
        el.append(float(rw[4])); eh.append(float(rw[5]))
    if not lm:
        raise ValueError('no paper SMF observations in %g <= z < %g (labels=%r)'
                         % (zlo, zhi, labels))
    o = np.argsort(lm)
    return (np.array(lm)[o], np.array(lp)[o], np.array(el)[o], np.array(eh)[o])


def _load_cosmosweb_smf(z_label, h_model):
    """COSMOS-Web SMF for one redshift bin -- the table Figure 12 plots.

    data/cosmosweb_smf.ecsv columns: Redshift (quoted bin label), M_star
    (log10), Phi (linear Mpc^-3 dex^-1), dPhi (linear, symmetric).  h_obs = 0.7.
    """
    import os as _os
    f = _os.path.join(_os.path.dirname(__file__), '../data/cosmosweb_smf.ecsv')
    rows = [l.rstrip('\n') for l in open(f) if not l.startswith('#') and l.strip()]
    want = z_label.replace(' ', '')
    lm, phi, dphi = [], [], []
    for line in rows[1:]:
        m = re.match(r'\s*"([^"]+)"\s+(\S+)\s+(\S+)\s+(\S+)', line)
        if m and m.group(1).replace(' ', '') == want:
            lm.append(float(m.group(2)))
            phi.append(float(m.group(3)))
            dphi.append(float(m.group(4)))
    if not lm:
        labels = sorted({re.match(r'\s*"([^"]+)"', l).group(1)
                         for l in rows[1:] if re.match(r'\s*"', l)})
        raise ValueError('COSMOS-Web has no bin %r (available: %s)'
                         % (z_label, '; '.join(labels)))
    lm, phi, dphi = np.array(lm), np.array(phi), np.array(dphi)
    ok = phi > 0
    lm, phi, dphi = lm[ok], phi[ok], dphi[ok]

    h_obs = 0.7
    x_obs = lm + 2.0 * np.log10(h_obs / h_model)
    y_obs = np.log10(phi) + 3.0 * np.log10(h_model / h_obs)
    lo = np.maximum(phi - dphi, phi * 1e-6)     # guard dphi >= phi
    return x_obs, y_obs, np.log10(phi) - np.log10(lo), np.log10(phi + dphi) - np.log10(phi)


def _load_gama_driver2022(h_model):
    """z=0 SMF, Driver et al. (2022) Table 6 (GAMA), h_obs = 0.7.

    Replaces Li+2009 (SDSS), which the constraint used while Figure 6 compares
    against GAMA.  Columns 0-2 are logM*, log phi, and its uncertainty in dex.
    """
    lm, lphi, elphi = _np_load_cols('../data/GAMA_SMF.dat', (0, 1, 2))
    ok = np.isfinite(lm) & np.isfinite(lphi) & np.isfinite(elphi) & (elphi > 0)
    lm, lphi, elphi = lm[ok], lphi[ok], elphi[ok]
    h_obs = 0.7
    return (lm + 2.0 * np.log10(h_obs / h_model),
            lphi + 3.0 * np.log10(h_model / h_obs), elphi, elphi)


class SMF(Constraint):
    """Common logic for SMF constraints"""

    bin_width = dm   # counts per dm dex bin -> Poisson/Cash

    domain = (8.5, 12)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        y = hist_smf[0,:]
        yerr = hist_smf_err[0,:]
        ind = np.where(y < 0.)
        return xmf[ind], y[ind], yerr[ind]

class SMF_z0(SMF):
    """The SMF constraint at z=0"""

    z = [0]

    def get_obs_x_y_err(self):
        return _load_paper_smf(0.0, 0.2, self.h0, labels=('Baldry+08',))

    def get_sage_x_y(self):
        # Diagnostic reference curve only -- it is not scored.  The file is
        # regenerated per run and its width depends on the reference epoch
        # list, so a stale or narrower file must not break the objective.
        try:
            logm, phi = self.load_observation(
                '../data/sage_smf_all_redshifts.csv', cols=[0, 1])
        except Exception:
            return np.zeros(1), np.zeros(1)
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage
    
class SMF_z05(SMF):
    """The SMF constraint at z=0.5"""

    z = [0.5]

    def get_obs_x_y_err(self):
        return _load_paper_smf(0.5, 0.8, self.h0)

    def get_sage_x_y(self):
        # Diagnostic reference curve only -- it is not scored.  The file is
        # regenerated per run and its width depends on the reference epoch
        # list, so a stale or narrower file must not break the objective.
        try:
            logm, phi = self.load_observation(
                '../data/sage_smf_all_redshifts.csv', cols=[4, 5])
        except Exception:
            return np.zeros(1), np.zeros(1)
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage
    
class SMF_z10(SMF):
    """The SMF constraint at z=1.0"""

    z = [1.0]

    def get_obs_x_y_err(self):
        return _load_paper_smf(0.8, 1.2, self.h0)

    def get_sage_x_y(self):
        # Diagnostic reference curve only -- it is not scored.  The file is
        # regenerated per run and its width depends on the reference epoch
        # list, so a stale or narrower file must not break the objective.
        try:
            logm, phi = self.load_observation(
                '../data/sage_smf_all_redshifts.csv', cols=[4, 5])
        except Exception:
            return np.zeros(1), np.zeros(1)
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage
    
class SMF_z20(SMF):
    """The SMF constraint at z=2.0"""

    z = [2.0]

    def get_obs_x_y_err(self):
        return _load_paper_smf(1.8, 2.3, self.h0)

    def get_sage_x_y(self):
        # Diagnostic reference curve only -- it is not scored.  The file is
        # regenerated per run and its width depends on the reference epoch
        # list, so a stale or narrower file must not break the objective.
        try:
            logm, phi = self.load_observation(
                '../data/sage_smf_all_redshifts.csv', cols=[12, 13])
        except Exception:
            return np.zeros(1), np.zeros(1)
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage
    
class SMF_z30(SMF):
    """The SMF constraint at z=3.0"""

    z = [3.0]

    def get_obs_x_y_err(self):
        return _load_paper_smf(2.8, 3.3, self.h0)

    def get_sage_x_y(self):
        # Diagnostic reference curve only -- it is not scored.  The file is
        # regenerated per run and its width depends on the reference epoch
        # list, so a stale or narrower file must not break the objective.
        try:
            logm, phi = self.load_observation(
                '../data/sage_smf_all_redshifts.csv', cols=[16, 17])
        except Exception:
            return np.zeros(1), np.zeros(1)
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage

class SMF_z40(SMF):
    """The SMF constraint at z=4.0"""

    z = [4.0]

    def get_obs_x_y_err(self):
        return _load_paper_smf(3.8, 4.3, self.h0)

    def get_sage_x_y(self):
        # Diagnostic reference curve only -- it is not scored.  The file is
        # regenerated per run and its width depends on the reference epoch
        # list, so a stale or narrower file must not break the objective.
        try:
            logm, phi = self.load_observation(
                '../data/sage_smf_all_redshifts.csv', cols=[20, 21])
        except Exception:
            return np.zeros(1), np.zeros(1)
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage

def _load_stefanon2021(redshift_bin, h_model):
    """Load Stefanon et al. (2021) SMF for a given integer redshift bin.

    Returns (x_obs, y_obs, y_dn, y_up) with all h0 corrections applied.
    phi is given in units of 1e-4 Mpc^-3 dex^-1; h_obs=0.7.
    """
    import os as _os
    data_file = _os.path.join(_os.path.dirname(__file__), '../data/stefanon_smf_2021.ecsv')
    raw = np.genfromtxt(data_file, comments='#', skip_header=1,
                        names=['z', 'lm', 'dlm', 'phi', 'eu', 'el'])
    sel = raw['z'] == redshift_bin
    if not np.any(sel):
        available = sorted(set(raw['z'][np.isfinite(raw['z'])].tolist()))
        raise ValueError(
            'Stefanon+2021 has no z=%g bin (available: %s). SMF_z50 requests '
            'z=5, which this dataset does not cover -- no shipped high-z SMF '
            'does. Returning an empty observation silently contributes 0 to '
            'the objective while still consuming its weight, so this raises '
            'instead.' % (redshift_bin, ', '.join('%g' % z for z in available)))
    lm  = raw['lm'][sel]
    phi = raw['phi'][sel] * 1e-4        # → Mpc^-3 dex^-1
    eu  = raw['eu'][sel]  * 1e-4
    el  = raw['el'][sel]  * 1e-4

    # Keep only points with positive phi and valid lower errors
    valid = (phi > 0) & (el > 0) & (el < phi)
    lm, phi, eu, el = lm[valid], phi[valid], eu[valid], el[valid]

    h_obs = 0.7
    x_obs = lm  - 2.0 * np.log10(h_obs / h_model)
    y_obs = np.log10(phi) + 3.0 * np.log10(h_obs / h_model)
    y_dn  = np.log10(phi) - np.log10(phi - el)   # asymmetric lower error in dex
    y_up  = np.log10(phi + eu) - np.log10(phi)   # asymmetric upper error in dex
    return x_obs, y_obs, y_dn, y_up


def _load_song2016(redshift_bin, h_model):
    """Load Song et al. (2016) SMF for a given integer redshift bin.

    Wide-format ECSV: one row per log_M, with phi_z4..phi_z8 columns.  Unlike
    Stefanon+2021 the phi columns are ALREADY log10(phi) and the errors are
    already in dex, so no logarithm is taken here.  h_obs = 0.7.

    Used for SMF_z50 because Stefanon+2021 -- and every other high-z SMF shipped
    with SAGE26 (Navarro-Carrera, Weibel, Kikuchihara) -- starts at z=6.  Song is
    the only shipped dataset covering z=5, and it is the same one Figure 12 uses
    in its 4.5 < z < 5.5 panel, so the constraint and the figure agree.
    """
    import os as _os
    data_file = _os.path.join(_os.path.dirname(__file__), '../data/song_smf_2016.ecsv')
    rows = [l.split() for l in open(data_file)
            if not l.startswith('#') and l.strip()]
    hdr, rows = rows[0], rows[1:]
    try:
        i_m = hdr.index('log_M')
        i_p = hdr.index('phi_z%d' % redshift_bin)
        i_u = hdr.index('phi_z%d_err_up' % redshift_bin)
        i_l = hdr.index('phi_z%d_err_lo' % redshift_bin)
    except ValueError:
        avail = sorted(int(c[5:]) for c in hdr if c.startswith('phi_z') and c[5:].isdigit())
        raise ValueError('Song+2016 has no z=%g column (available: %s)'
                         % (redshift_bin, ', '.join(str(z) for z in avail)))

    arr = np.array([[float(r[i]) for i in (i_m, i_p, i_u, i_l)] for r in rows])
    lm, lphi, eu, el = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    valid = np.isfinite(lm) & np.isfinite(lphi) & np.isfinite(eu) & np.isfinite(el)
    lm, lphi, eu, el = lm[valid], lphi[valid], eu[valid], el[valid]

    h_obs = 0.7
    x_obs = lm - 2.0 * np.log10(h_obs / h_model)
    y_obs = lphi + 3.0 * np.log10(h_obs / h_model)
    return x_obs, y_obs, el, eu      # errors already in dex


class SMF_z50(SMF):
    """SMF at z~5, Song et al. (2016)  [data: logM 7.25–11.25]

    Stefanon+2021 was used here originally but has no z=5 bin -- it covers
    z = 6-10 -- so this constraint silently returned an empty observation,
    which scored 0.0 while still consuming its share of the objective weight.
    """

    z = [5.0]
    domain = (7.5, 10.5)

    def get_obs_x_y_err(self):
        return _load_paper_smf(4.8, 5.3, self.h0)

    def get_sage_x_y(self):
        return np.zeros(0), np.zeros(0)


class SMF_z60(SMF):
    """SMF at z~6, Stefanon et al. (2021)  [data: logM 7.8–10.6]"""

    z = [6.0]
    domain = (7.5, 11.0)

    def get_obs_x_y_err(self):
        return _load_paper_smf(5.8, 6.3, self.h0)

    def get_sage_x_y(self):
        return np.zeros(0), np.zeros(0)


class SMF_z70(SMF):
    """SMF at z~7, Stefanon et al. (2021)  [data: logM 7.75–10.3]"""

    z = [7.0]
    domain = (7.5, 10.5)

    def get_obs_x_y_err(self):
        return _load_paper_smf(6.8, 7.3, self.h0)

    def get_sage_x_y(self):
        return np.zeros(0), np.zeros(0)


class SMF_z80(SMF):
    """SMF at z~8, Stefanon et al. (2021)  [data: logM 7.9–10.15]"""

    z = [8.0]
    domain = (7.5, 10.5)

    def get_obs_x_y_err(self):
        return _load_paper_smf(7.8, 8.3, self.h0)

    def get_sage_x_y(self):
        return np.zeros(0), np.zeros(0)


class SMF_z100(SMF):
    """SMF at z~10, Stefanon et al. (2021)  [data: logM 7.65–8.75]"""

    z = [10.0]
    domain = (7.5, 10.0)   # widened: the combined
                           # Stefanon+COSMOS-Web sample reaches logM 9.77

    def get_obs_x_y_err(self):
        return _load_paper_smf(9.5, 11.0, self.h0)

    def get_sage_x_y(self):
        return np.zeros(0), np.zeros(0)


class SMF_Red(Constraint):
    """Base class for Red/Quiescent SMF constraints"""

    bin_width = dm   # counts per dm dex bin -> Poisson/Cash

    domain = (8.5, 12)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        y = hist_smf_red[0,:]
        yerr = hist_smf_red_err[0,:]
        ind = np.where(y < 0.)
        return xmf[ind], y[ind], yerr[ind]

class SMF_Red_z0(SMF_Red):
    """Red/Quiescent SMF constraint at z=0 using GAMA E+HE data"""

    z = [0]

    def get_obs_x_y_err(self):
        # Load GAMA morphological SMF data
        # E_HE = Elliptical + High-mass Early-type (Red/Quiescent)
        # skiprows=15 to skip 14 comment lines + 1 header row in the ECSV file
        gama_mass, gama_E_HE, gama_E_HE_err = self.load_observation('../data/gama_smf_morph.ecsv', cols=[0,1,2], skiprows=15)

        # Filter out NaN values
        valid = ~np.isnan(gama_E_HE)
        x_obs = gama_mass[valid]
        y_obs = gama_E_HE[valid]
        y_err = gama_E_HE_err[valid]

        return x_obs, y_obs, y_err, y_err

    def get_sage_x_y(self):
        # Load SAGE red SMF data
        logm, phi = self.load_observation('../data/sage_smf_red_all_redshifts.csv', cols=[0,1])
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]
        return x_sage, y_sage

class SMF_Blue(Constraint):
    """Base class for Blue/Star-forming SMF constraints"""

    bin_width = dm   # counts per dm dex bin -> Poisson/Cash

    domain = (8.5, 12)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        y = hist_smf_blue[0,:]
        yerr = hist_smf_blue_err[0,:]
        ind = np.where(y < 0.)
        return xmf[ind], y[ind], yerr[ind]

class SMF_Blue_z0(SMF_Blue):
    """Blue/Star-forming SMF constraint at z=0 using GAMA D data"""

    z = [0]

    def get_obs_x_y_err(self):
        # Load GAMA morphological SMF data
        # D = Disk (Blue/Star-forming)
        # skiprows=15 to skip 14 comment lines + 1 header row in the ECSV file
        gama_mass, gama_D, gama_D_err = self.load_observation('../data/gama_smf_morph.ecsv', cols=[0,7,8], skiprows=15)

        # Filter out NaN values
        valid = ~np.isnan(gama_D)
        x_obs = gama_mass[valid]
        y_obs = gama_D[valid]
        y_err = gama_D_err[valid]

        return x_obs, y_obs, y_err, y_err

    def get_sage_x_y(self):
        # Load SAGE blue SMF data
        logm, phi = self.load_observation('../data/sage_smf_blue_all_redshifts.csv', cols=[0,1])
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]
        return x_sage, y_sage

class CSFRDH(Constraint):

    # Snapshots are now simulation-specific, set in __init__
    z = None  # Will be set dynamically based on simulation
    domain = (0, 12.62)  # look-back time in Gyr; 12.62 Gyr = z=6
                         # (12.0 was only z=4.02, while the paper
                         # states the CSFRD constrains to z=6)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The snapshots come from parse(), which resolved them against the SAGE
        # output's own redshift table; re-deriving them here from a simulation
        # id would discard that and reintroduce the hard-coded mapping.
        if self.snapshot is None:
            self.snapshot = get_csfrdh_snapshots(self.sim)
        elif not isinstance(self.snapshot, list):
            self.snapshot = [self.snapshot]
        self.z = self.snapshot
    
    def get_obs_x_y_err(self):
        """COSMOS-Web cosmic SFR density, inferred from the stellar mass density.

        Replaces D'Silva+2023, which this constraint used while Figure 3 of the
        paper compares against COSMOS-Web.  The source table is a densely
        sampled curve (4001 rows, z = 0 to 17.6); scoring every row would hand
        this one constraint ~4000 of the objective's data points and swamp
        every other constraint, so it is subsampled onto evenly spaced
        lookback-time points.  Errors are the 16th/84th percentile band.
        """
        z_all, s50, s16, s84 = _np_load_cols(
            '../data/CSFRD_inferred_from_SMD.ecsv', (0, 1, 2, 3))
        ok = np.isfinite(z_all) & (s50 > 0) & (s16 > 0) & (s84 > 0)
        z_all, s50, s16, s84 = z_all[ok], s50[ok], s16[ok], s84[ok]

        tL = np.array([r.z2tL(z, self.h0, self.Omega0, 1.0 - self.Omega0)
                       for z in z_all])
        lo, hi = self.domain
        sel = (tL >= lo) & (tL <= hi)
        tL, s50, s16, s84 = tL[sel], s50[sel], s16[sel], s84[sel]

        targets = np.linspace(tL.min(), tL.max(), 25)
        idx = np.unique([int(np.argmin(np.abs(tL - tt))) for tt in targets])

        y_obs = np.log10(s50[idx])
        return tL[idx], y_obs, y_obs - np.log10(s16[idx]), np.log10(s84[idx]) - y_obs

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        # TimeBinEdge now contains the actual snapshot times (lookback times in Gyr) for CSFRDH
        # Ignore the hist_smf and hist_bhmf arrays which contain -20 values
        # For CSFRDH, errors are not well-defined, return zeros
        yerr = np.zeros_like(SFRD_Age)
        return TimeBinEdge, SFRD_Age, yerr
    
    def get_sage_x_y(self):
        # Load data from SAGE
        # logm, phi = self.load_observation('../data/sage_smf_all_redshifts.csv', cols=[20,21])
        logm, phi = np.zeros(1), np.zeros(1)

        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage
        
class BHBM(Constraint):
    """The Black hole-Bulge mass relation constraint"""

    domain = (9.0, 12.0)
    z = [0]

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):

        mask = (BlackHoleMass > 0) & (BulgeMass > 0) & np.isfinite(BlackHoleMass) & np.isfinite(BulgeMass)
        y = BlackHoleMass[mask]
        x = BulgeMass[mask]

        # Only keep galaxies that fall within the binning range
        range_mask = (x >= 9.0) & (x <= 12.0)
        x_in_range = x[range_mask]

        if len(x_in_range) < 3:
            return np.array([9.0, 12.0]), np.array([6.0, 9.0]), np.array([0.3, 0.3])

        # Create bins for bulge mass and calculate median black hole mass in each bin
        bin_edges = np.arange(9.0, 12.1, 0.2)  # Bins every 0.2 dex
        bin_centers = []
        median_bh_mass = []
        bin_errors = []

        for i in range(len(bin_edges) - 1):
            bin_mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
            if np.sum(bin_mask) >= 3:  # At least 3 galaxies in bin
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2.0)
                median_bh_mass.append(np.median(y[bin_mask]))
                N_bin = np.sum(bin_mask)
                # Error on the median, because the observations are binned to
                # medians too (see get_obs_x_y_err).  Using the full intrinsic
                # scatter here instead would make the total error large enough
                # that chi2/n sits near 1 whatever the relation does, and the
                # constraint would stop discriminating between models.
                std_bin = np.std(y[bin_mask]) if N_bin > 1 else 0.3
                bin_errors.append(max(std_bin / np.sqrt(N_bin), 0.05))

        if len(bin_centers) < 1:
            return np.array([9.0, 12.0]), np.array([6.0, 9.0]), np.array([0.3, 0.3])

        return np.array(bin_centers), np.array(median_bh_mass), np.array(bin_errors)

    def get_obs_x_y_err(self):
        """Compilation binned to medians, to match the model's median relation.

        Scott+2013 (75) + Davis+2019 (41 spirals) + Sahu+2019 (41 E/S0) = 157
        galaxies.  They are binned rather than used individually because
        get_model_x_y returns a median relation, not a population: comparing a
        cloud of individual galaxies against a single median curve makes the
        residuals dominated by the intrinsic scatter of the relation rather
        than by whether the relation itself is right.  Widening the errors to
        absorb that scatter does not help -- it drives chi2/n towards 1 for any
        model and the constraint stops discriminating.  Binning both sides to
        medians keeps the comparison sensitive to the quantity being
        calibrated, at the cost of the per-galaxy errors, which are smaller
        than the bin-to-bin scatter anyway.
        """
        log_mbulge, log_mbh, _, _ = self.load_observation(
            '../data/bhbm_obs_combined.csv', cols=[0, 1, 2, 3], skiprows=3)

        bin_edges = np.arange(9.0, 12.1, 0.4)  # coarser than model bins -- ~7 obs bins
        bin_centers, median_bh, bin_err = [], [], []
        for i in range(len(bin_edges) - 1):
            mask = (log_mbulge >= bin_edges[i]) & (log_mbulge < bin_edges[i + 1])
            if np.sum(mask) >= 3:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                median_bh.append(np.median(log_mbh[mask]))
                N = np.sum(mask)
                scatter = np.std(log_mbh[mask]) if N > 1 else 0.3
                bin_err.append(max(scatter / np.sqrt(N), 0.05))

        x_obs = np.array(bin_centers)
        y_obs = np.array(median_bh)
        bin_err = np.array(bin_err)
        err = np.array(bin_err)
        return x_obs, y_obs, err, err
    
    def get_sage_x_y(self):
        # Load data from SAGE
        bulgemass, blackholemass = self.load_observation('../data/sage_bhbm_all_redshifts.csv', cols=[0,1])
        x_sage = bulgemass
        y_sage = blackholemass

        return x_sage, y_sage

class HIMF(Constraint):
    """The HI Mass Function constraint"""

    bin_width = dm_h1   # counts per dm_h1 dex bin -> Poisson/Cash

    domain = (8.0, 10.75)
    z = [0]

    def get_obs_x_y_err(self):
        # Jones et al. (2018) ALFALFA 100%, measured with h=0.7
        # Cols: log(MHI/Msun), log(phi [Mpc^-3 dex^-1]), log(phi_low), log(phi_up)
        lmHI, pHI, pHI_low, pHI_up = self.load_observation('../data/HIMF_Jones18.dat', cols=[0,1,2,3])

        h_obs = 0.7
        x_obs = lmHI + np.log10(pow(h_obs, 2) / pow(self.h0, 2))
        y_obs = pHI + np.log10(pow(self.h0, 3) / pow(h_obs, 3))
        y_dn = pHI - pHI_low   # asymmetric lower error in log space
        y_up = pHI_up - pHI   # asymmetric upper error in log space

        return x_obs, y_obs, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        y = hist_himf[0]
        yerr = hist_himf_err[0]
        ind = np.where(y < 0.)
        return xmf_h1[ind], y[ind], yerr[ind]

    def get_sage_x_y(self):
        # Placeholder - add SAGE HI MF data if available
        logm, phi = np.zeros(1), np.zeros(1)
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]
        return x_sage, y_sage


class H2MF(Constraint):
    """The H2 Mass Function constraint"""

    bin_width = dm_h2   # counts per dm_h2 dex bin -> Poisson/Cash

    domain = (8.25, 10.1)
    z = [0]

    def get_obs_x_y_err(self):
        # Load Fletcher 2021 data
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'H2MF_Fletcher21_DetNonDet.dat'), comments='#')
        logm = obs_data[:, 0]      # log10(MH2/Msun)
        logphi = obs_data[:, 1]    # log10(phi)
        logphi_dn = obs_data[:, 2] # 16th percentile
        logphi_up = obs_data[:, 3] # 84th percentile

        # Calculate asymmetric errors
        y_dn = logphi - logphi_dn  # Lower error (positive value)
        y_up = logphi_up - logphi  # Upper error (positive value)

        return logm, logphi, np.abs(y_dn), np.abs(y_up)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        y = hist_h2mf[0]
        yerr = hist_h2mf_err[0]
        ind = np.where(y < 0.)
        return xmf_h2[ind], y[ind], yerr[ind]

    def get_sage_x_y(self):
        # Placeholder - no SAGE H2MF data available
        logm, phi = np.zeros(1), np.zeros(1)
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]
        return x_sage, y_sage


def _load_madau_dickinson_smd(z_max=None):
    """Stellar mass density compilation from Madau & Dickinson (2014).

    Their table gives a redshift interval per measurement rather than a single
    redshift, so the interval midpoint is used.  Errors are asymmetric and
    sometimes absent; a missing error becomes the median of the quoted ones so
    the point still carries information without dominating.

    Pass z_max to keep only points below that redshift, which is how SMD avoids
    double-counting the range COSMOS2020 already covers.
    """
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    path = os.path.join(DATA_DIR, 'MandD_smd_2014.ecsv')
    if not os.path.exists(path):
        return (np.array([]),) * 4

    z_mid, rho, err_up, err_dn = [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('Reference'):
            continue
        # Reference is a quoted string that contains spaces
        tail = line.rsplit('"', 1)[-1] if '"' in line else line
        parts = tail.split()
        if len(parts) < 4:
            continue
        try:
            zlo, zhi, lr = float(parts[0]), float(parts[1]), float(parts[2])
            eu = float(parts[3]) if len(parts) > 3 else np.nan
            ed = float(parts[4]) if len(parts) > 4 else np.nan
        except ValueError:
            continue
        if not np.isfinite(lr):
            continue
        z_mid.append(0.5 * (zlo + zhi))
        rho.append(lr)
        err_up.append(eu)
        err_dn.append(ed)

    z_mid = np.array(z_mid); rho = np.array(rho)
    err_up = np.array(err_up); err_dn = np.array(err_dn)
    if len(z_mid) == 0:
        return (np.array([]),) * 4

    quoted = np.concatenate([err_up[np.isfinite(err_up)],
                             err_dn[np.isfinite(err_dn)]])
    fallback = float(np.median(quoted)) if len(quoted) else 0.15
    err_up = np.where(np.isfinite(err_up), err_up, fallback)
    err_dn = np.where(np.isfinite(err_dn), err_dn, fallback)

    if z_max is not None:
        keep = z_mid < z_max
        z_mid, rho, err_up, err_dn = z_mid[keep], rho[keep], err_up[keep], err_dn[keep]

    order = np.argsort(z_mid)
    return z_mid[order], rho[order], np.abs(err_dn[order]), np.abs(err_up[order])


class SMD(Constraint):
    """Stellar Mass Density constraint vs redshift"""

    domain = (0.0, 12.0)
    # Snapshots are now simulation-specific, set in __init__
    z = None  # Will be set dynamically based on simulation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The snapshots come from parse(), which resolved them against the SAGE
        # output's own redshift table; re-deriving them here from a simulation
        # id would discard that and reintroduce the hard-coded mapping.
        if self.snapshot is None:
            self.snapshot = get_smd_snapshots(self.sim)
        elif not isinstance(self.snapshot, list):
            self.snapshot = [self.snapshot]
        self.z = self.snapshot

    def get_obs_x_y_err(self):
        # ... (Keep your existing observation loading code) ...
        # Load SMD data from Weaver et al. 2023 (COSMOS2020)
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        # Read ECSV file - skip header lines
        with open(os.path.join(DATA_DIR, 'SMD.ecsv'), 'r') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and 'z rho' not in line:
                data_start = i
                break

        z_obs = []
        rho_50 = []
        rho_16 = []
        rho_84 = []
        for line in lines[data_start:]:
            if line.strip():
                parts = line.split()
                z_obs.append(float(parts[0]))
                rho_50.append(float(parts[1]))
                rho_16.append(float(parts[2]))
                rho_84.append(float(parts[3]))

        z_obs = np.array(z_obs)
        rho_50 = np.array(rho_50)
        rho_16 = np.array(rho_16)
        rho_84 = np.array(rho_84)

        log_rho = np.log10(rho_50)
        log_rho_16 = np.log10(rho_16)
        log_rho_84 = np.log10(rho_84)

        y_dn = log_rho - log_rho_16
        y_up = log_rho_84 - log_rho

        # Weaver+23 (COSMOS2020) starts at z = 0.35, but the constraint samples
        # snapshots from z = 0, so the lowest-redshift model points had nothing
        # to be scored against.  Madau & Dickinson (2014) compile measurements
        # from z = 0.01 upward; take only their points below the COSMOS2020 floor
        # so the two sets do not double-count the same redshifts.
        z_lo, rho_lo, dn_lo, up_lo = _load_madau_dickinson_smd(z_max=z_obs.min())
        if len(z_lo):
            z_obs = np.concatenate([z_lo, z_obs])
            log_rho = np.concatenate([rho_lo, log_rho])
            y_dn = np.concatenate([dn_lo, y_dn])
            y_up = np.concatenate([up_lo, y_up])

        return z_obs, log_rho, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        # <--- NEW: 'smd' is now the SMD_history array from the loader
        
        # Calculate redshifts corresponding to the snapshots
        alist_full = np.loadtxt(self.age_alist_file)
        z_model = np.zeros(len(self.snapshot))
        
        for i, snap in enumerate(self.snapshot):
            if snap < len(alist_full):
                z_model[i] = 1.0 / alist_full[snap] - 1.0
        
        # Calculate log of SMD history
        log_smd = np.zeros_like(smd)
        valid = smd > 0
        log_smd[valid] = np.log10(smd[valid])
        log_smd[~valid] = -20
        
        yerr = np.zeros_like(log_smd) 
        
        return z_model, log_smd, yerr

    def get_sage_x_y(self):
        logm, phi = np.zeros(1), np.zeros(1)
        return logm, phi


class FICS(Constraint):
    """Intracluster-star mass fraction vs redshift.

    f_ICS = m_ICS / M_*,halo, where M_*,halo = m_ICS + m_BCG + sum(m_satellites)
    is the total stellar mass inside the halo.  Constrained from z = 0 to z = 2,
    the range over which the observed ICL fractions in
    data/ICL_fraction_compilation.dat are measured.

    The measurement is a median over the groups and clusters of each snapshot
    (selection cuts: FICS_* at the top of this module), so it is scored in linear
    f_ICS rather than in a log, and the observations are binned in redshift with
    the scatter between independent measurements taken as the error -- the
    individual literature values carry no published uncertainty, and their
    disagreement over how to split ICL from BCG light is the dominant term in
    the error budget anyway.
    """

    # Redshift range scored, matching the span of the observations.  This is
    # only safe while the model can measure f_ICS across the whole range:
    # np.interp in Constraint.get_data extrapolates flat beyond the model's
    # range, so a floor high enough to empty the high-z snapshots would score
    # the upper observation bins against fabricated values.  get_model_x_y
    # warns, naming the bins, if that ever happens.
    domain = (0.0, 2.0)  # redshift
    z = None             # set dynamically per simulation, as for CSFRDH and SMD
    needs_fics = True    # the only constraint that applies the FICS_* halo cuts

    # Redshift bin edges for the observational compilation.  Uneven by design:
    # measurements crowd into z < 0.5 and thin out beyond it, so the high-z bins
    # are wide enough to keep at least a few independent points each.
    obs_z_bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.55, 0.9, 2.05])
    obs_min_per_bin = 3     # bins with fewer points are dropped as unmeasured
    obs_err_floor = 0.02    # absolute floor on the f_ICS error, in fraction units

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The snapshots come from parse(), which resolved them against the SAGE
        # output's own redshift table; re-deriving them here from a simulation
        # id would discard that and reintroduce the hard-coded mapping.
        if self.snapshot is None:
            self.snapshot = get_fics_snapshots(self.sim)
        elif not isinstance(self.snapshot, list):
            self.snapshot = [self.snapshot]
        self.z = self.snapshot

    def get_obs_x_y_err(self):
        """Compilation of observed ICL/ICS mass fractions, binned in redshift.

        Each bin returns the median f_ICS of its points, with the 16th/84th
        percentiles as asymmetric errors.  Points with f_ICS <= 0 are dropped:
        the one such entry in the compilation is a digitisation artefact, not a
        cluster with no ICL.
        """
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'ICL_fraction_compilation.dat'),
                              comments='#', usecols=(0, 1))

        z_obs = obs_data[:, 0]
        f_obs = obs_data[:, 1]

        valid = np.isfinite(z_obs) & np.isfinite(f_obs) & (f_obs > 0)
        z_obs, f_obs = z_obs[valid], f_obs[valid]

        x, y, y_dn, y_up = [], [], [], []
        edges = self.obs_z_bins
        for i in range(len(edges) - 1):
            in_bin = (z_obs >= edges[i]) & (z_obs < edges[i + 1])
            if np.count_nonzero(in_bin) < self.obs_min_per_bin:
                continue
            f_bin = f_obs[in_bin]
            median = np.median(f_bin)
            x.append(np.median(z_obs[in_bin]))
            y.append(median)
            y_dn.append(max(median - np.percentile(f_bin, 16), self.obs_err_floor))
            y_up.append(max(np.percentile(f_bin, 84) - median, self.obs_err_floor))

        return np.array(x), np.array(y), np.array(y_dn), np.array(y_up)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        # fics_history holds the median f_ICS per snapshot, NaN where too few
        # groups and clusters qualified to measure it.
        alist_full = np.loadtxt(self.age_alist_file)
        z_model = np.zeros(len(self.snapshot))

        for i, snap in enumerate(self.snapshot):
            if snap < len(alist_full):
                z_model[i] = 1.0 / alist_full[snap] - 1.0

        f_model = np.asarray(fics_history, dtype=np.float64)
        f_err = np.asarray(fics_history_err, dtype=np.float64)

        measured = np.isfinite(f_model) & (f_model > 0)
        if np.count_nonzero(measured) < 2:
            # Nothing measurable -- either the box holds no groups or the SAGE
            # build does not output IntraClusterStars.  Return a flat dummy with
            # a large error so the constraint cannot masquerade as a good fit.
            logging.getLogger('constraints').warning(
                'FICS: fewer than two snapshots yielded a measurable ICS '
                'fraction; this run cannot be scored on FICS')
            return np.array([0.0, 2.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])

        z_measured = z_model[measured]

        # Guard the flat extrapolation described above.  What matters is not
        # whether the measured range covers the domain exactly, but whether any
        # observation bin that will actually be scored lies outside it -- those
        # are the ones np.interp would fabricate a model value for.
        lo, hi = self.domain
        x_obs = self.get_obs_x_y_err()[0]
        scored = x_obs[(x_obs >= lo) & (x_obs <= hi)]
        beyond = scored[(scored < z_measured.min()) | (scored > z_measured.max())]
        if len(beyond) and not getattr(self, '_warned_coverage', False):
            self._warned_coverage = True
            logging.getLogger('constraints').warning(
                'FICS: model measures f_ICS only over z = %.2f-%.2f, so the '
                'observation bins at z = %s are scored against a flat '
                'extrapolation. Narrow the domain to FICS(%.2f-%.2f) or use a '
                'box with cluster statistics to higher z.',
                z_measured.min(), z_measured.max(),
                ', '.join('%.2f' % b for b in beyond),
                max(lo, z_measured.min()), min(hi, z_measured.max()))

        return z_measured, f_model[measured], f_err[measured]

    def get_sage_x_y(self):
        logm, phi = np.zeros(1), np.zeros(1)
        return logm, phi


class FICS_Mvir(Constraint):
    """Intracluster-star mass fraction as a function of host halo mass, at z = 0.

    f_ICS = m_ICS / M_*,halo, where M_*,halo = m_ICS + m_BCG + sum(m_satellites),
    binned by log10 M_vir and compared with the Contini (2021) compilation in
    data/Contini2021_ICL_fraction_vs_Mvir.dat.

    This is the observable that separates the SAGE26 disruption-split
    mechanisms.  They differ in how the ICS/BCG split depends on the
    satellite-to-host mass ratio, so their distinguishing prediction is the
    slope of f_ICS with halo mass: the fixed-fraction mode has none by
    construction, while the mass-ratio and concentration-weighted modes each
    predict a different one.  FICS constrains f_ICS(z), which marginalises over
    halo mass and so cannot tell them apart -- all three can reproduce the same
    mean fraction.  Use the two together.

    The observations are rich clusters (log10 M_vir = 14.2-15.1), so the box has
    to contain such haloes: microUchuu (100/h Mpc) reaches ~10^14.6 and covers
    the lower half, miniMillennium (62.5/h Mpc) reaches ~10^14.2 and covers
    none.  get_model_x_y warns, naming the points, when the model cannot reach
    them rather than letting np.interp fabricate values for them.
    """

    # Full log10 M_vir range the observations span.  parse() can only narrow a
    # domain, never widen it, so this has to be the outer bound.
    #
    # NARROW IT TO MATCH YOUR BOX.  np.interp in Constraint.get_data extrapolates
    # flat, not along a slope, so an observation above the model's richest
    # halo-mass bin is scored against that bin's value.  microUchuu (100/h Mpc)
    # tops out around log10 M_vir ~ 14.42, so it wants FICS_Mvir(14-14.45),
    # which admits the two lowest points (Krick & Bernstein 2007, Zibetti+05).
    # miniUchuu (400/h Mpc, 64x the volume) reaches past 10^15 and can use the
    # full range.  get_model_x_y warns, naming the points, whenever the model
    # cannot bracket what the domain admits -- do not ignore that warning.
    domain = (14.0, 15.2)
    z = [0]
    needs_fics = True

    def get_obs_x_y_err(self):
        """Contini (2021) compilation of f_ICL against host halo mass."""
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs = np.loadtxt(os.path.join(DATA_DIR,
                                      'Contini2021_ICL_fraction_vs_Mvir.dat'),
                         comments='#', usecols=(0, 1, 2))
        logm, f_ics, err = obs[:, 0], obs[:, 1], obs[:, 2]
        valid = np.isfinite(logm) & np.isfinite(f_ics) & (f_ics > 0)
        logm, f_ics, err = logm[valid], f_ics[valid], err[valid]
        return logm, f_ics, err.copy(), err.copy()

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        # fics_halo_mass / fics_halo_frac are the per-halo values at the z = 0
        # snapshot; bin them by halo mass the way MZR and SHMR bin theirs.
        logm = np.asarray(fics_halo_mass, dtype=np.float64)
        f_ics = np.asarray(fics_halo_frac, dtype=np.float64)

        if len(logm) < FICS_MVIR_BIN_MIN:
            logging.getLogger('constraints').warning(
                'FICS_Mvir: only %d haloes passed the FICS selection; this run '
                'cannot be scored on the f_ICS-halo mass relation', len(logm))
            return np.array([14.0, 15.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])

        edges = np.arange(np.floor(logm.min() / FICS_MVIR_BIN_WIDTH)
                          * FICS_MVIR_BIN_WIDTH,
                          logm.max() + FICS_MVIR_BIN_WIDTH,
                          FICS_MVIR_BIN_WIDTH)
        centres, medians, errors = [], [], []
        for i in range(len(edges) - 1):
            in_bin = (logm >= edges[i]) & (logm < edges[i + 1])
            n = int(np.count_nonzero(in_bin))
            if n < FICS_MVIR_BIN_MIN:
                continue
            vals = f_ics[in_bin]
            # x is the median halo mass of the bin's members rather than the bin
            # centre: the mass function falls steeply, so members cluster near
            # the low-mass edge and the centre would misplace the point.
            centres.append(float(np.median(logm[in_bin])))
            medians.append(float(np.median(vals)))
            errors.append(max(1.253 * float(np.std(vals)) / np.sqrt(n), 1e-3))

        if len(centres) < 2:
            logging.getLogger('constraints').warning(
                'FICS_Mvir: fewer than two usable halo-mass bins; this run '
                'cannot be scored on the f_ICS-halo mass relation')
            return np.array([14.0, 15.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])

        centres = np.array(centres)

        # Guard the flat extrapolation in Constraint.get_data: report any
        # observation that will be scored but that the model cannot bracket.
        lo, hi = self.domain
        x_obs = self.get_obs_x_y_err()[0]
        scored = x_obs[(x_obs >= lo) & (x_obs <= hi)]
        beyond = scored[(scored < centres.min()) | (scored > centres.max())]
        if len(beyond) and not getattr(self, '_warned_coverage', False):
            self._warned_coverage = True
            logging.getLogger('constraints').warning(
                'FICS_Mvir: model haloes span log10 Mvir = %.2f-%.2f, so the '
                'observations at log10 Mvir = %s are scored against a flat '
                'extrapolation. Narrow the domain to FICS_Mvir(%.2f-%.2f) or '
                'use a box containing richer clusters.',
                centres.min(), centres.max(),
                ', '.join('%.2f' % b for b in beyond),
                max(lo, centres.min()), min(hi, centres.max()))

        return centres, np.array(medians), np.array(errors)

    def get_sage_x_y(self):
        logm, phi = np.zeros(1), np.zeros(1)
        return logm, phi


class MLF(Constraint):
    """Galactic-wind mass-loading factor against circular velocity, at z = 0.

    eta = Mdot_outflow / SFR, binned in log10 v_circ and compared with the
    compilation in data/mass_loading_compilation.dat (Rupke+05, Heckman+15,
    Chisholm+17, Sugahara+17).

    This is the observable that speaks most directly to the supernova feedback
    parameters.  FeedbackReheatingEpsilon sets eta's normalisation and
    RedshiftPowerLawExponent its redshift scaling, but the stellar mass function
    constrains them only indirectly, through the stellar mass that survives the
    outflow.  Constraining eta itself separates "the right amount of gas is
    being ejected" from "the right amount of stellar mass happens to be left".

    Requires FIREmodeOn = 1: SAGE assigns MassLoading only inside the FIRE
    branch of the feedback, so the field is identically zero otherwise and this
    constraint reports the run as not applicable rather than as a fit.
    """

    # log10 v_circ range scored, covering the compilation (53-459 km/s).
    domain = (1.9, 2.7)
    z = [0]
    needs_mlf = True

    obs_bin_width = MLF_BIN_WIDTH
    obs_min_per_bin = 3      # bins with fewer independent measurements are dropped
    obs_err_floor = 0.10     # dex; no published errors survive the digitisation

    def get_obs_x_y_err(self):
        """Compilation of measured mass-loading factors, binned in log v_circ.

        The digitised measurements carry no uncertainties, so as for FICS the
        error is the scatter between independent measurements in each velocity
        bin -- which for outflow rates is the dominant term anyway: the papers
        disagree about which gas phase and which aperture define the outflow,
        and two galaxies at the same v_circ can differ by an order of magnitude.
        """
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs = np.loadtxt(os.path.join(DATA_DIR, 'mass_loading_compilation.dat'),
                         comments='#', usecols=(0, 1))
        v, eta = obs[:, 0], obs[:, 1]
        valid = np.isfinite(v) & np.isfinite(eta) & (v > 0) & (eta > 0)
        log_v, log_eta = np.log10(v[valid]), np.log10(eta[valid])

        edges = np.arange(np.floor(log_v.min() / self.obs_bin_width) * self.obs_bin_width,
                          log_v.max() + self.obs_bin_width, self.obs_bin_width)
        x, y, y_dn, y_up = [], [], [], []
        for i in range(len(edges) - 1):
            in_bin = (log_v >= edges[i]) & (log_v < edges[i + 1])
            if np.count_nonzero(in_bin) < self.obs_min_per_bin:
                continue
            vals = log_eta[in_bin]
            median = float(np.median(vals))
            x.append(float(np.median(log_v[in_bin])))
            y.append(median)
            y_dn.append(max(median - np.percentile(vals, 16), self.obs_err_floor))
            y_up.append(max(np.percentile(vals, 84) - median, self.obs_err_floor))

        return np.array(x), np.array(y), np.array(y_dn), np.array(y_up)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        x = np.asarray(mlf_x, dtype=np.float64)
        y = np.asarray(mlf_y, dtype=np.float64)
        err = np.asarray(mlf_err, dtype=np.float64)

        if len(x) < 2:
            logging.getLogger('constraints').warning(
                'MLF: no measurable mass loading (SAGE only sets MassLoading '
                'for FIREmodeOn = 1); this run cannot be scored on the '
                'mass-loading relation')
            return (np.array([self.domain[0], self.domain[1]]),
                    np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        # Same coverage guard as FICS/FICS_Mvir: np.interp extrapolates flat,
        # so report any scored observation the model cannot bracket.
        lo, hi = self.domain
        x_obs = self.get_obs_x_y_err()[0]
        scored = x_obs[(x_obs >= lo) & (x_obs <= hi)]
        beyond = scored[(scored < x.min()) | (scored > x.max())]
        if len(beyond) and not getattr(self, '_warned_coverage', False):
            self._warned_coverage = True
            logging.getLogger('constraints').warning(
                'MLF: model covers log10 v_circ = %.2f-%.2f, so the '
                'observations at %s are scored against a flat extrapolation. '
                'Narrow the domain to MLF(%.2f-%.2f).',
                x.min(), x.max(),
                ', '.join('%.2f' % b for b in beyond),
                max(lo, x.min()), min(hi, x.max()))

        return x, y, err

    def get_sage_x_y(self):
        logm, phi = np.zeros(1), np.zeros(1)
        return logm, phi


class MZR(Constraint):
    """Mass-Metallicity Relation constraint"""

    domain = (8.0, 11.0)  # Stellar mass range in log10(Msun)
    z = [0]

    def get_obs_x_y_err(self):
        # Curti et al. (2020) MZR — 75 stacked data points, asymmetric errors
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'Curti2020.dat'), comments='#')

        logm = obs_data[:, 0]         # log10(Mstar/Msun)
        metallicity = obs_data[:, 1]  # 12+log(O/H)
        met_low = obs_data[:, 2]      # lower bound
        met_high = obs_data[:, 3]     # upper bound

        y_dn = metallicity - met_low
        y_up = met_high - metallicity

        return logm, metallicity, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        # Bin the metallicity data by stellar mass
        if len(stellar_mass_mzr) < 10:
            yerr_dummy = np.array([0.3, 0.3])
            return np.array([9.0, 11.0]), np.array([8.5, 9.0]), yerr_dummy

        bin_edges = np.arange(8.0, 11.5, 0.25)
        bin_centers = []
        median_met = []
        bin_errors = []

        for i in range(len(bin_edges) - 1):
            bin_mask = (stellar_mass_mzr >= bin_edges[i]) & (stellar_mass_mzr < bin_edges[i+1])
            if np.sum(bin_mask) >= 10:
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2.0)
                median_met.append(np.median(metallicity[bin_mask]))
                N_bin = np.sum(bin_mask)
                std_bin = np.std(metallicity[bin_mask])
                bin_errors.append(std_bin / np.sqrt(N_bin))

        if len(bin_centers) < 3:
            yerr_dummy = np.array([0.3, 0.3])
            return np.array([9.0, 11.0]), np.array([8.5, 9.0]), yerr_dummy

        return np.array(bin_centers), np.array(median_met), np.array(bin_errors)

    def get_sage_x_y(self):
        logm, phi = np.zeros(1), np.zeros(1)
        return logm, phi


class SHMR(Constraint):
    """Stellar-to-Halo Mass Relation constraint"""

    domain = (11.0, 15.0)  # Halo mass range in log10(Msun)
    z = [0]

    def get_obs_x_y_err(self):
        # Correa & Schaye (2019) SHMR — combined LTGs + ETGs, actual measurements
        # Cols: log10(M200), log10(Mstar), lower_bound, upper_bound
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        ltg = np.loadtxt(os.path.join(DATA_DIR, 'LTGs_Correa19.dat'), comments='#')
        etg = np.loadtxt(os.path.join(DATA_DIR, 'ETGs_Correa19.dat'), comments='#')
        obs_data = np.vstack([ltg, etg])

        logm_halo = obs_data[:, 0]
        logm_stellar = obs_data[:, 1]
        stellar_low = obs_data[:, 2]
        stellar_high = obs_data[:, 3]

        valid_mask = ~np.isnan(logm_halo) & ~np.isnan(logm_stellar)
        logm_halo = logm_halo[valid_mask]
        logm_stellar = logm_stellar[valid_mask]
        y_dn = logm_stellar - stellar_low[valid_mask]
        y_up = stellar_high[valid_mask] - logm_stellar

        return logm_halo, logm_stellar, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr, fics_history, fics_history_err, fics_halo_mass, fics_halo_frac, mlf_x, mlf_y, mlf_err):
        # Bin the SHMR data by halo mass
        if len(halo_mass_shmr) < 10:
            yerr_dummy = np.array([0.3, 0.3])
            return np.array([11.0, 14.0]), np.array([8.0, 11.0]), yerr_dummy

        bin_edges = np.arange(10.5, 15.1, 0.25)
        bin_centers = []
        median_stellar = []
        bin_errors = []

        for i in range(len(bin_edges) - 1):
            bin_mask = (halo_mass_shmr >= bin_edges[i]) & (halo_mass_shmr < bin_edges[i+1])
            if np.sum(bin_mask) >= 10:
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2.0)
                median_stellar.append(np.median(stellar_mass_shmr[bin_mask]))
                N_bin = np.sum(bin_mask)
                std_bin = np.std(stellar_mass_shmr[bin_mask])
                bin_errors.append(std_bin / np.sqrt(N_bin))

        if len(bin_centers) < 3:
            yerr_dummy = np.array([0.3, 0.3])
            return np.array([11.0, 14.0]), np.array([8.0, 11.0]), yerr_dummy

        return np.array(bin_centers), np.array(median_stellar), np.array(bin_errors)

    def get_sage_x_y(self):
        # Load SAGE SHMR data if available
        try:
            logm_halo, logm_stellar = self.load_observation('../data/sage_halostellar_all_redshifts.csv', cols=[0,1])
            valid_mask = ~np.isnan(logm_halo) & ~np.isnan(logm_stellar)
            return logm_halo[valid_mask], logm_stellar[valid_mask]
        except:
            logm, phi = np.zeros(1), np.zeros(1)
            return logm, phi


_constraint_re = re.compile((r'([0-9_a-zA-Z]+)' # name
                              r'(?:\(([0-9\.]+)-([0-9\.]+)\))?' # domain boundaries
                              r'(?:\*([0-9\.]+))?')) # weight
def parse(spec, snapshot=None, sim=None, boxsize=None, vol_frac=None, age_alist_file=None, Omega0=None, h0=None, output_dir=None, snapshot_map=None):
    """Parses a comma-separated string of constraint names into a list of
    Constraint objects. Specific domain values can be specified in `spec`.

    `snapshot_map` is the constraint-to-snapshot mapping resolved from the SAGE
    output (simulation_config.build_snapshot_map), so each constraint gets the
    snapshot closest to the redshift its observations were measured at.  If it
    is not supplied the legacy hard-coded per-simulation table is used, which is
    only correct for the three simulations it was written for.
    """
    if snapshot_map is None:
        from src.simulation_config import get_snapshot_map as _get_snapshot_map
        _snapshot_map = _get_snapshot_map(sim if sim is not None else 0)
    else:
        _snapshot_map = snapshot_map

    _constraints = {
        'BHMF_z0': BHMF_z0,
        'BHMF_z10': BHMF_z10,
        'SMF_z0': SMF_z0,
        'SMF_z05': SMF_z05,
        'SMF_z10': SMF_z10,
        'SMF_z20': SMF_z20,
        'SMF_z30': SMF_z30,
        'SMF_z40': SMF_z40,
        'SMF_z50': SMF_z50,
        'SMF_z60': SMF_z60,
        'SMF_z70': SMF_z70,
        'SMF_z80': SMF_z80,
        'SMF_z100': SMF_z100,
        'SMF_Red_z0': SMF_Red_z0,
        'SMF_Blue_z0': SMF_Blue_z0,
        'BHBM': BHBM,
        'CSFRDH': CSFRDH,
        'HIMF': HIMF,
        'H2MF': H2MF,
        'SMD': SMD,
        'MZR': MZR,
        'SHMR': SHMR,
        'FICS': FICS,
        'FICS_Mvir': FICS_Mvir,
        'MLF': MLF
    }

    def _parse(s,output_dir):
        m = _constraint_re.match(s)
        if not m or m.group(1) not in _constraints:
            raise ValueError('Constraint does not specify a valid constraint: %s' % s)
        # Use the per-constraint snapshot from the simulation map, not the combined list
        constraint_name = m.group(1)
        per_constraint_snapshot = _snapshot_map.get(constraint_name, snapshot)
        c = _constraints[constraint_name](snapshot=per_constraint_snapshot, sim=sim, boxsize=boxsize, vol_frac=vol_frac, age_alist_file=age_alist_file,
                                     Omega0=Omega0, h0=h0, output_dir=output_dir)
        if m.group(2):
            dn, up = float(m.group(2)), float(m.group(3))
            if dn < c.domain[0]:
                raise ValueError('Constraint low boundary is lower than lowest value possible (%f < %f)' % (dn, c.domain[0]))
            if up > c.domain[1]:
                raise ValueError('Constraint up boundary is higher than lowest value possible (%f > %f)' % (up, c.domain[1]))
            c.domain = (dn, up)
        if m.group(4):
            c.weight = float(m.group(4))
        return c

    constraints = [_parse(s, output_dir) for s in spec.split(',')]
    total_weight = sum([c.weight for c in constraints])
    for c in constraints:
        c.rel_weight = c.weight / total_weight
    return constraints
