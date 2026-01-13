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
from src.simulation_config import get_csfrdh_snapshots, get_smd_snapshots

warnings.filterwarnings("ignore")
logging.getLogger('constraints').setLevel(logging.INFO)

GyrToYr = 1e9
#######################
# Binning configuration
mupp = 12.0
dm = 0.1
mlow = 8
mbins = np.arange(mlow, mupp, dm)
xmf = mbins + dm/2.0

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

class Constraint(object):
    """Base classes for constraint objects"""

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

    def _load_model_data(self, modeldir, subvols):
        # Allow snapshots to be a list
        if not isinstance(self.snapshot, list):
            self.snapshot = [self.snapshot]
        
        # For constraints that need multiple snapshots (like CSFRDH and SMD), we'll collect SFRD and SMD for each
        Nage = 14
        num_snapshots = len(self.snapshot)
        SFRbyAge = np.zeros(num_snapshots)
        SMD_history = np.zeros(num_snapshots) # [FIX] Array to store SMD history
        SnapshotTimes = np.zeros(num_snapshots)  # Store lookback time for each snapshot
        
        # For hist_smf, hist_bhmf, etc., always use the last (most recent) snapshot
        # This ensures SMF_z0 uses snapshot 63 even when combined with CSFRDH
        snap_to_use = self.snapshot[-1]
        
        # Loop through snapshots to build SFRD and SMD history
        for snap_idx, snap in enumerate(self.snapshot):
            if len(subvols) > 1:
                subvols = ["multiple_batches"]

            seed(2222)
            fields = ['StellarMass', 'BlackHoleMass', 'Len', 'SfrBulge', 'BulgeMass', 'Mvir', 'SfrDisk', 'ColdGas', 'H1gas', 'H2gas', 'MetalsColdGas']
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
                    metallicity = np.log10((G['MetalsColdGas'][w_mzr] / G['ColdGas'][w_mzr]) / 0.02) + 9.0
                    stellar_mass_mzr = np.log10(G['StellarMass'][w_mzr] * 1e10 / self.h0)
                else:
                    metallicity = np.array([])
                    stellar_mass_mzr = np.array([])

                # Calculate Stellar-Halo Mass Relation (SHMR)
                w_shmr = np.where((G['Mvir'] > 0) & (G['StellarMass'] > 0))[0]
                if len(w_shmr) > 0:
                    halo_mass_shmr = np.log10(G['Mvir'][w_shmr] * 1e10 / self.h0)
                    stellar_mass_shmr = np.log10(G['StellarMass'][w_shmr] * 1e10 / self.h0)
                else:
                    halo_mass_shmr = np.array([])
                    stellar_mass_shmr = np.array([])

        # Get the edges of the age bins (after the loop)
        alist_full = np.loadtxt(self.age_alist_file)
        
        # Calculate lookback times for the actual snapshots we loaded
        for snap_idx, snap in enumerate(self.snapshot):
            if snap < len(alist_full):
                redshift = 1.0 / alist_full[snap] - 1.0
                SnapshotTimes[snap_idx] = r.z2tL(redshift, self.h0, self.Omega0, 1.0-self.Omega0)
        
        # Also calculate generic age bins for other constraints
        if Nage>=len(alist_full)-1:
            alist = alist_full[::-1]
            RedshiftBinEdge = 1./ alist - 1.
        else:
            indices_float = np.arange(Nage+1) * (len(alist_full)-1.0) / Nage
            indices = indices_float.astype(np.int32)
            alist = alist_full[indices][::-1]
            RedshiftBinEdge = 1./ alist - 1.
        TimeBinEdge = np.array([r.z2tL(redshift, self.h0, self.Omega0, 1.0-self.Omega0) for redshift in RedshiftBinEdge]) # look-back time [Gyr]
        dT = np.diff(TimeBinEdge) # time step for each bin
        TimeBinCentre = TimeBinEdge[:-1] + 0.5*dT

        #########################
        # Calculate Poisson errors before taking logs
        # For a number count N, the Poisson error is sqrt(N)
        # In log space: sigma_log(phi) ≈ 0.434 / sqrt(N) for N >> 1
        # We use the more accurate formula: sigma_log = |log10(phi) - log10(phi ± sqrt(N)/Volume/dm)|
        
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
                    hist_smf_err[i] = 999  # Large error for empty bins
            else:
                hist_smf_err[i] = 999  # Large error for empty bins
        
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
                    hist_bhmf_err[i] = 999  # Large error for empty bins
            else:
                hist_bhmf_err[i] = 999  # Large error for empty bins
        
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
                    hist_himf_err[i] = 999  # Large error for empty bins
            else:
                hist_himf_err[i] = 999  # Large error for empty bins

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
                    hist_h2mf_err[i] = 999
            else:
                hist_h2mf_err[i] = 999

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
                    hist_smf_red_err[i] = 999
            else:
                hist_smf_red_err[i] = 999

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
                    hist_smf_blue_err[i] = 999
            else:
                hist_smf_blue_err[i] = 999

        #########################
        # take logs
        ind = (hist_smf > 0.)
        hist_smf[ind] = np.log10(hist_smf[ind])
        hist_smf[~ind] = -20

        ind = (hist_smf_red > 0.)
        hist_smf_red[ind] = np.log10(hist_smf_red[ind])
        hist_smf_red[~ind] = -20

        ind = (hist_smf_blue > 0.)
        hist_smf_blue[ind] = np.log10(hist_smf_blue[ind])
        hist_smf_blue[~ind] = -20

        ind = (hist_bhmf > 0.)
        hist_bhmf[ind] = np.log10(hist_bhmf[ind])
        hist_bhmf[~ind] = -20

        ind = (hist_himf > 0.)
        hist_himf[ind] = np.log10(hist_himf[ind])
        hist_himf[~ind] = -20

        ind = (hist_h2mf > 0.)
        hist_h2mf[ind] = np.log10(hist_h2mf[ind])
        hist_h2mf[~ind] = -20

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
            return self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, SnapshotTimes, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, SMD_history, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr
        else:
            return self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, SMD_history, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr


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
        
        # Plot data
        ax.plot(x_obs_sel, y_obs_sel, marker='v', ls='None', c='blue', markersize=6, label="Selected obs")
        ax.plot(x_mod, y_mod, marker='^', ls='solid', c='orange', markersize=4, alpha=0.7, label="Raw model")
        ax.plot(x_obs, y_mod_interp, ls='solid', c='green', linewidth=2, label="Interp model")
        ax.plot(x_obs_sel, y_mod_sel, ls='None', marker='o', c='brown', markersize=5, label="Selected model")
        
        # Error bars - ensure errors are positive
        for i, x_val in enumerate(x_obs):
            ax.errorbar(x_val, y_obs[i], 
                       yerr=[[np.abs(obs_err_dn[i])], [np.abs(obs_err_up[i])]], 
                       fmt='none', c='black', alpha=0.5, capsize=3)
        
        # Calculate metrics
        chi2 = analysis.chi2(y_obs_sel, y_mod_sel, err)
        st = analysis.studentT(y_obs_sel, y_mod_sel, err)
        
        ax.set_title('%s\n$\chi^2$ = %.2f, student-t = %.2f' % (str(self), chi2, st), fontsize=10)
        ax.legend(loc='best', fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        
        plotfile = os.path.join(output_dir, f'{self.__class__.__name__}_diagnostic.pdf')
        plt.savefig(plotfile, dpi=100, bbox_inches='tight')
        plt.close()
        return

    def _get_raw_data(self, modeldir, subvols):
        """Gets the model and observational data for further analysis.
        The model data is interpolated to match the observation's X values."""

        self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr = self._load_model_data(modeldir, subvols)
        x_obs, y_obs, y_dn, y_up = self.get_obs_x_y_err()
        x_sage, y_sage = self.get_sage_x_y()
        x_mod, y_mod, y_mod_err = self.get_model_x_y(hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr)
        return x_obs, y_obs, y_dn, y_up, x_sage, y_sage, x_mod, y_mod, y_mod_err

    def get_data(self, modeldir, subvols):

        self.h0, self.Omega0, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr = self._load_model_data(modeldir, subvols)
        x_obs, y_obs, y_dn, y_up = self.get_obs_x_y_err()
        x_sage, y_sage = self.get_sage_x_y()
        x_mod, y_mod, y_mod_err = self.get_model_x_y(hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr)

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
        
        print('in get_data:')
        print('obs x:', np.round(x_obs_sel, 2))
        print('obs y:', np.round(y_obs_sel, 2))
        print('mod y:', np.round(y_mod_sel, 2))
        print('obs errors:', np.round(y_obs_err_sel, 2))
        print('mod errors:', np.round(y_mod_err_sel, 2))
        print('combined errors:', np.round(err, 2))
        print('differences (mod-obs):', np.round(y_mod_sel - y_obs_sel, 2))
        print('normalized differences:', np.round((y_mod_sel - y_obs_sel) / err, 2))

        # Get the constraint name and create filename directly in outdir
        constraint_name = self.__class__.__name__
        filename = os.path.join(self.output_dir, f"{constraint_name}_dump.txt")

        # Append data to dump file
        with open(filename, 'a') as f:
            f.write(f"# New Data Block\n")
            for x_val, y_val, mod_y_val in zip(x_obs_sel, y_obs_sel, y_mod_sel):
                f.write(f"{x_val}\t{y_val}\t{mod_y_val}\n")
            
        # Get constraint name for appropriate plotting function
        # constraint_name = self.__class__.__name__
        
        # # Create specialized publication-quality plots
        # if 'SMF' in constraint_name:
        #     self.plot_smf(x_obs_sel, y_obs_sel, y_mod_sel, x_sage, y_sage, y_dn[sel], y_up[sel], self.output_dir)
        # elif 'BHMF' in constraint_name:
        #     self.plot_bhmf(x_obs_sel, y_obs_sel, y_mod_sel, x_sage, y_sage, y_dn[sel], y_up[sel], self.output_dir)
        # elif 'HIMF' in constraint_name:
        #     self.plot_himf(x_obs_sel, y_obs_sel, y_mod_sel, x_sage, y_sage, y_dn[sel], y_up[sel], self.output_dir)
        # elif 'BHBM' in constraint_name:
        #     self.plot_bhbm(x_obs_sel, y_obs_sel, y_mod_sel, x_sage, y_sage, y_dn[sel], y_up[sel], BlackHoleMass, BulgeMass, self.output_dir)
        # elif 'CSFRDH' in constraint_name:
        #     self.plot_CSFRDH(x_obs_sel, y_obs_sel, y_mod_sel, x_sage, y_sage, y_dn[sel], y_up[sel], TimeBinEdge, SFRD_Age, self.output_dir)
        
        # Always create diagnostic plot showing interpolation/selection steps
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

    domain = (4, 10.5)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
        y = hist_bhmf[0]
        yerr = hist_bhmf_err[0]
        ind = np.where(y < 0.)

        return xmf2[ind], y[ind], yerr[ind]
    
class BHMF_z0(BHMF):
    """The BHMF constraint at z=0"""
    
    z = [0]

    def get_obs_x_y_err(self):
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        # Load data from TRINITY PaperIV (z=0.1)
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'fig4_bhmf_z0.1.txt'))
        logm = obs_data[:, 0]           # log10(Mbh [Msun])
        phi = obs_data[:, 1]            # BHMF_best [Mpc^-3 dex^-1]
        phi_16th = obs_data[:, 2]       # BHMF_16th [Mpc^-3 dex^-1]
        phi_84th = obs_data[:, 3]       # BHMF_84th [Mpc^-3 dex^-1]
        
        # Convert to log10 space
        logphi = np.log10(phi)
        logphi_16th = np.log10(phi_16th)
        logphi_84th = np.log10(phi_84th)
        
        # Calculate asymmetric errors in log space
        y_dn = logphi - logphi_16th  # Lower error (positive value)
        y_up = logphi_84th - logphi  # Upper error (positive value)
        
        # Remove NaN values
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi) & ~np.isnan(y_dn) & ~np.isnan(y_up)
        x_obs = logm[valid_mask]
        y_obs = logphi[valid_mask]
        y_dn = y_dn[valid_mask]
        y_up = y_up[valid_mask]
    
        return x_obs, y_obs, y_dn, y_up
    
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
        # Load data from TRINITY PaperIV (z=1.0)
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'fig4_bhmf_z1.0.txt'))
        logm = obs_data[:, 0]           # log10(Mbh [Msun])
        phi = obs_data[:, 1]            # BHMF_best [Mpc^-3 dex^-1]
        phi_16th = obs_data[:, 2]       # BHMF_16th [Mpc^-3 dex^-1]
        phi_84th = obs_data[:, 3]       # BHMF_84th [Mpc^-3 dex^-1]
        
        # Convert to log10 space
        logphi = np.log10(phi)
        logphi_16th = np.log10(phi_16th)
        logphi_84th = np.log10(phi_84th)
        
        # Calculate asymmetric errors in log space
        y_dn = logphi - logphi_16th  # Lower error (positive value)
        y_up = logphi_84th - logphi  # Upper error (positive value)
        
        # Remove NaN values
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi) & ~np.isnan(y_dn) & ~np.isnan(y_up)
        x_obs = logm[valid_mask]
        y_obs = logphi[valid_mask]
        y_dn = y_dn[valid_mask]
        y_up = y_up[valid_mask]
    
        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_bhmf_all_redshifts.csv', cols=[4,5])
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage


class SMF(Constraint):
    """Common logic for SMF constraints"""

    domain = (8.5, 12)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
        y = hist_smf[0,:]
        yerr = hist_smf_err[0,:]
        ind = np.where(y < 0.)
        return xmf[ind], y[ind], yerr[ind]

class SMF_z0(SMF):
    """The SMF constraint at z=0"""

    z = [0]

    def get_obs_x_y_err(self):
        # SMF from Li & White (2009)
        lm, p, dpdn, dpup = self.load_observation('../data/SMF_Li2009.dat', cols=[0,1,2,3])
        hobs = self.h0
        x_obs = lm - 2.0 * np.log10(hobs) + 2.0 * np.log10(hobs/self.h0)
        y_obs = p + 3.0 * np.log10(hobs) - 3.0 * np.log10(hobs/self.h0)
        y_dn = dpdn
        y_up = dpup

        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_smf_all_redshifts.csv', cols=[0,1])
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
        # SMF from Weaver et al. (2022)
        lm, pD, dn, du = self.load_observation('../data/COSMOS2020/SMF_Farmer_v2.1_0.2z0.5_total.txt', cols=[0,2,3,4])
        hobs = self.h0
        y_obs = np.log10(pD) + 3.0 * np.log10(hobs/self.h0)
        y_dn = np.log10(pD) - np.log10(dn)
        y_up = np.log10(du) - np.log10(pD)
        x_obs = lm - 2.0 * np.log10(hobs/self.h0)

        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_smf_all_redshifts.csv', cols=[4,5])
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
        # SMF from Weaver et al. (2022)
        lm, pD, dn, du = self.load_observation('../data/COSMOS2020/SMF_Farmer_v2.1_0.8z1.1_total.txt', cols=[0,2,3,4])
        hobs = self.h0
        y_obs = np.log10(pD) + 3.0 * np.log10(hobs/self.h0)
        y_dn = np.log10(pD) - np.log10(dn)
        y_up = np.log10(du) - np.log10(pD)
        x_obs = lm - 2.0 * np.log10(hobs/self.h0)

        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_smf_extra_redshifts.csv', cols=[4,5])
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
        # SMF from Weaver et al. (2022)
        lm, pD, dn, du = self.load_observation('../data/COSMOS2020/SMF_Farmer_v2.1_1.5z2.0_total.txt', cols=[0,2,3,4])
        hobs = self.h0
        y_obs = np.log10(pD) + 3.0 * np.log10(hobs/self.h0)
        y_dn = np.log10(pD) - np.log10(dn)
        y_up = np.log10(du) - np.log10(pD)
        x_obs = lm - 2.0 * np.log10(hobs/self.h0)

        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_smf_all_redshifts.csv', cols=[12,13])
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
        # SMF from Weaver et al. (2022)
        lm, pD, dn, du = self.load_observation('../data/COSMOS2020/SMF_Farmer_v2.1_2.5z3.0_total.txt', cols=[0,2,3,4])
        hobs = self.h0
        y_obs = np.log10(pD) + 3.0 * np.log10(hobs/self.h0)
        y_dn = np.log10(pD) - np.log10(dn)
        y_up = np.log10(du) - np.log10(pD)
        x_obs = lm - 2.0 * np.log10(hobs/self.h0)

        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_smf_all_redshifts.csv', cols=[16,17])
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
        # SMF from Weaver et al. (2022)
        lm, pD, dn, du = self.load_observation('../data/COSMOS2020/SMF_Farmer_v2.1_3.5z4.5_total.txt', cols=[0,2,3,4])
        hobs = self.h0
        y_obs = np.log10(pD) + 3.0 * np.log10(hobs/self.h0)
        y_dn = np.log10(pD) - np.log10(dn)
        y_up = np.log10(du) - np.log10(pD)
        x_obs = lm - 2.0 * np.log10(hobs/self.h0)

        return x_obs, y_obs, y_dn, y_up
    
    def get_sage_x_y(self):
        # Load data from SAGE
        logm, phi = self.load_observation('../data/sage_smf_all_redshifts.csv', cols=[20,21])
        # Remove NaN values
        logphi = np.log10(phi)
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        x_sage = logm[valid_mask]
        y_sage = logphi[valid_mask]

        return x_sage, y_sage

class SMF_Red(Constraint):
    """Base class for Red/Quiescent SMF constraints"""

    domain = (8.5, 12)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
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

    domain = (8.5, 12)

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
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
    domain = (0, 12) # look-back time in Gyr

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get simulation-specific CSFRDH snapshots
        self.z = get_csfrdh_snapshots(self.sim)
        self.snapshot = self.z
    
    def get_obs_x_y_err(self):
#        zmin, zmax, logSFRD, err1, err2, err3 = self.load_observation('../data/Driver_SFRD.dat', cols=[1,2,3,5,6,7])
        my_cosmo = [100*self.h0, 0.0, self.Omega0, 1.0-self.Omega0]
        D18_cosmo = [70.0, 0., 0.3, 0.7]

        D23_0, D23_1, D23_2, D23_3, D23_4, D23_5 = self.load_observation('../data/CSFH_DSILVA+23_Ver_Final.csv', cols=[0,1,2,3,4,5])
        D23 = np.column_stack((D23_0, D23_1, D23_2, D23_3, D23_4, D23_5))
        z_D23 = D23[:,3]
        tLB_D23 = np.array([r.z2tL(z, self.h0, self.Omega0,  1.0-self.Omega0) for z in z_D23])
        CSFH_D23 = D23[:,0]
        for i in range(len(z_D23)):
            CSFH_D23[i] += np.log10( r.z2dA(z_D23[i], *my_cosmo) / r.z2dA(z_D23[i], *D18_cosmo) )*2 # adjust for assumed-cosmology influence on SFR calculations
            CSFH_D23[i] += np.log10( (r.comoving_distance(D23[i,3]+D23[i,4], *D18_cosmo)**3 - r.comoving_distance(D23[i,3]-D23[i,5], *D18_cosmo)**3) / (r.comoving_distance(D23[i,3]+D23[i,4], *my_cosmo)**3 - r.comoving_distance(D23[i,3]-D23[i,5], *my_cosmo)**3) )# adjust for assumed-cosmology influence on comoving volume
#        
#        Np = len(logSFRD)
#        x_obs = np.zeros(Np)
#        y_obs = np.zeros(Np)
#        for i in range(Np):
#            z_av = 0.5*(zmin[i]+zmax[i])
#            x_obs[i] = r.z2tL(z_av, h0, Omega0,  1.0-Omega0)
#            y_obs[i] = logSFRD[i] + \
#                        np.log10( pow(r.comoving_distance(zmax[i], *D18_cosmo), 3.0) - pow(r.comoving_distance(zmin[i], *D18_cosmo), 3.0) ) - \
#                        np.log10( pow(r.comoving_distance(zmax[i], *my_cosmo), 3.0) - pow(r.comoving_distance(zmin[i], *my_cosmo), 3.0) ) + \
#                        np.log10( r.z2dA(z_av, *my_cosmo) / r.z2dA(z_av, *D18_cosmo) ) * 2.0 # adjust for cosmology on comoving volume and luminosity of objects
#            
#        err_total = err1 + err2 + err3
        
        return tLB_D23, CSFH_D23, D23[:,2], D23[:,1]
        
    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
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

    domain = (8.0, 11.5)
    z = [0]

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):

        mask = (BlackHoleMass > 0) & (BulgeMass > 0) & np.isfinite(BlackHoleMass) & np.isfinite(BulgeMass)
        y = BlackHoleMass[mask]
        x = BulgeMass[mask]
        
        if len(x) < 10:  # Not enough points for reliable median
            # Return dummy arrays that will result in poor fit
            yerr_dummy = np.array([999.0, 999.0])
            return np.array([8.0, 12.0]), np.array([6.0, 8.0]), yerr_dummy
        
        # Create bins for bulge mass and calculate median black hole mass in each bin
        bin_edges = np.arange(8.0, 12.1, 0.2)  # Bins every 0.2 dex
        bin_centers = []
        median_bh_mass = []
        bin_errors = []
        
        for i in range(len(bin_edges) - 1):
            bin_mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
            if np.sum(bin_mask) >= 10:  # At least 10 galaxies in bin
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2.0)
                median_bh_mass.append(np.median(y[bin_mask]))
                # Error on median is approximately 1.25 * std / sqrt(N) for Gaussian
                # For scatter in BHBM relation, use standard deviation / sqrt(N)
                N_bin = np.sum(bin_mask)
                std_bin = np.std(y[bin_mask])
                bin_errors.append(std_bin / np.sqrt(N_bin))
        
        if len(bin_centers) < 3:  # Not enough bins for reliable relation
            yerr_dummy = np.array([999.0, 999.0])
            return np.array([8.0, 12.0]), np.array([6.0, 8.0]), yerr_dummy
        
        return np.array(bin_centers), np.array(median_bh_mass), np.array(bin_errors)

    def get_obs_x_y_err(self):
        
        # Häring & Rix 2004 relation
        w_bulge = 10. ** np.arange(8.0, 12.5, 0.1)  # More reasonable range
        BHdata_haring = 10. ** (8.2 + 1.12 * np.log10(w_bulge / 1.0e11))
        
        # Convert to log space
        x_points = np.log10(w_bulge)
        y_points = np.log10(BHdata_haring)
        
        # Typical scatter is ~0.3-0.4 dex
        scatter_dex = 0.34  # Häring & Rix 2004 intrinsic scatter
        
        # Use scatter as symmetric error
        err = np.ones_like(y_points) * scatter_dex
        
        return x_points, y_points, err, err
    
    def get_sage_x_y(self):
        # Load data from SAGE
        bulgemass, blackholemass = self.load_observation('../data/sage_bhbm_all_redshifts.csv', cols=[0,1])
        x_sage = bulgemass
        y_sage = blackholemass

        return x_sage, y_sage

class HIMF(Constraint):
    """The HI Mass Function constraint"""

    domain = (8.0, 10.75)
    z = [0]

    def get_obs_x_y_err(self):
        # Load Zwaan05 data and correct data for their choice of cosmology
        lmHI, pHI, dpHIdn, dpHIup = self.load_observation('../data/HIMF_Zwaan2005.dat', cols=[0,1,2,3])

        # Correct data for their choice of cosmology
        hobs = self.h0 
        x_obs = lmHI + np.log10(pow(hobs, 2) / pow(self.h0, 2))
        y_obs = pHI + np.log10(pow(self.h0, 3) / pow(hobs, 3))
        y_dn = dpHIdn
        y_up = dpHIup

        return x_obs, y_obs, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
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

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
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


class SMD(Constraint):
    """Stellar Mass Density constraint vs redshift"""

    domain = (0.0, 12.0)
    # Snapshots are now simulation-specific, set in __init__
    z = None  # Will be set dynamically based on simulation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get simulation-specific SMD snapshots
        self.z = get_smd_snapshots(self.sim)
        self.snapshot = self.z

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

        return z_obs, log_rho, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
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


class MZR(Constraint):
    """Mass-Metallicity Relation constraint"""

    domain = (8.5, 11.0)  # Stellar mass range in log10(Msun)
    z = [0]

    def get_obs_x_y_err(self):
        # Load Tremonti et al. 2004 data
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'Tremonti04.dat'), comments='#')

        logm = obs_data[:, 0]        # log(Mstars/Msun)
        metallicity = obs_data[:, 1]  # 12+log(O/H)
        met_16th = obs_data[:, 2]     # 16th percentile
        met_84th = obs_data[:, 3]     # 84th percentile

        # Calculate asymmetric errors
        y_dn = metallicity - met_16th
        y_up = met_84th - metallicity

        return logm, metallicity, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
        # Bin the metallicity data by stellar mass
        if len(stellar_mass_mzr) < 10:
            yerr_dummy = np.array([999.0, 999.0])
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
            yerr_dummy = np.array([999.0, 999.0])
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
        # Load Moster et al. 2013 data (z=0 columns)
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        obs_data = np.loadtxt(os.path.join(DATA_DIR, 'Moster_2013.csv'), delimiter='\t')

        # First two columns are z=0: halo mass, stellar mass
        logm_halo = obs_data[:, 0]
        logm_stellar = obs_data[:, 1]

        # Remove NaN values
        valid_mask = ~np.isnan(logm_halo) & ~np.isnan(logm_stellar)
        logm_halo = logm_halo[valid_mask]
        logm_stellar = logm_stellar[valid_mask]

        # Assume ~0.15 dex scatter
        scatter = 0.15
        y_dn = np.ones_like(logm_stellar) * scatter
        y_up = np.ones_like(logm_stellar) * scatter

        return logm_halo, logm_stellar, y_dn, y_up

    def get_model_x_y(self, hist_smf, hist_bhmf, hist_himf, TimeBinEdge, SFRD_Age, BlackHoleMass, BulgeMass, HaloMass, StellarMass, hist_smf_red, hist_smf_blue, hist_smf_err, hist_smf_red_err, hist_smf_blue_err, hist_bhmf_err, hist_himf_err, hist_h2mf, hist_h2mf_err, smd, metallicity, stellar_mass_mzr, halo_mass_shmr, stellar_mass_shmr):
        # Bin the SHMR data by halo mass
        if len(halo_mass_shmr) < 10:
            yerr_dummy = np.array([999.0, 999.0])
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
            yerr_dummy = np.array([999.0, 999.0])
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
def parse(spec, snapshot=None, sim=None, boxsize=None, vol_frac=None, age_alist_file=None, Omega0=None, h0=None, output_dir=None):
    """Parses a comma-separated string of constraint names into a list of
    Constraint objects. Specific domain values can be specified in `spec`"""

    _constraints = {
        'BHMF_z0': BHMF_z0,
        'BHMF_z10': BHMF_z10,
        'SMF_z0': SMF_z0,
        'SMF_z05': SMF_z05,
        'SMF_z10': SMF_z10,
        'SMF_z20': SMF_z20,
        'SMF_z30': SMF_z30,
        'SMF_z40': SMF_z40,
        'SMF_Red_z0': SMF_Red_z0,
        'SMF_Blue_z0': SMF_Blue_z0,
        'BHBM': BHBM,
        'CSFRDH': CSFRDH,
        'HIMF': HIMF,
        'H2MF': H2MF,
        'SMD': SMD,
        'MZR': MZR,
        'SHMR': SHMR
    }

    def _parse(s,output_dir):
        m = _constraint_re.match(s)
        if not m or m.group(1) not in _constraints:
            raise ValueError('Constraint does not specify a valid constraint: %s' % s)
        c = _constraints[m.group(1)](snapshot=snapshot, sim=sim, boxsize=boxsize, vol_frac=vol_frac, age_alist_file=age_alist_file,
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
