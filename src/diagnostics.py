#!/bin/bash

"""
This script produces the diagnostic plots used to help visualise the PSO process
One is a 3D representation of the swarm movement in the parameter space the other
is the Log Liklihood over iteration for each particle
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import itertools
import os
import re
from src import common
import logging
import matplotlib # type: ignore
import matplotlib.animation as anim # type: ignore
import matplotlib.cm as cmx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
from matplotlib import cm # type: ignore
from matplotlib.colors import Normalize # type: ignore
import numpy as np # type: ignore
from src import routines as r
from src import analysis
import glob
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import matplotlib.colors # type: ignore
from src.redshift_utils import get_redshift_info, get_all_redshifts
from scipy import stats
# from sklearn.linear_model import LinearRegression
import matplotlib.colors as cols
from src.pso_uncertainty import analyze_pso_uncertainties, plot_parameter_distributions, \
                         create_uncertainty_report, analyze_and_plot

logger = logging.getLogger('diagnostics')

pos_re = re.compile('track_[0-9]+_pos.npy')
fx_re = re.compile('track_[0-9]+_fx.npy')

def load_observation(*args, **kwargs):
    obsdir = os.path.dirname(os.path.abspath(__file__))
    return common.load_observation(obsdir, *args, **kwargs)

def load_space_and_particles(tracks_dir, space_file):
    """Loads the PSO pos/fx information stored within a directory, plus the
    original search space file"""

    all_fnames = list(os.listdir(tracks_dir))
    pos_fnames = list(filter(lambda x: pos_re.match(x), all_fnames))
    fx_fnames = list(filter(lambda x: fx_re.match(x), all_fnames))

    if len(pos_fnames) != len(fx_fnames):
        logger.info("Different number of pos/fx files, using common files only")
        l = min(len(pos_fnames), len(fx_fnames))
        del pos_fnames[l:]
        del fx_fnames[l:]

    logger.info("Loading %d pos/fx files" % len(pos_fnames))

    # Read files in filename order and populate pos/fx np arrays
    pos_fnames.sort()
    fx_fnames.sort()
    pos = []
    fx = []
    for pos_fname, fx_fname in zip(pos_fnames, fx_fnames):
        pos.append(np.load(os.path.join(tracks_dir, pos_fname)))
        fx.append(np.load(os.path.join(tracks_dir, fx_fname)))

    # after this fx dims are (S, L), pos dims are (S, D, L)
    pos, fx = np.asarray(pos), np.asarray(fx)
    pos = np.moveaxis(pos, 0, -1)
    fx = np.moveaxis(fx, 0, -1)
    logger.debug('Position shape: %s, Fitness shape: %s', pos.shape, fx.shape)
    #logger.info(fx)
    #logger.info(pos)

    space = analysis.load_space(space_file)
    if space.shape[0] != pos.shape[1]:
        raise ValueError("Particles have different dimensionality than space")
        
    #logger.info('Ordered fits:\n -logLikelihood,', space['name'])
    #logger.info(np.column_stack((np.log10(np.sort(fx, axis=None)), np.moveaxis(pos,1,-1)[np.unravel_index(np.argsort(fx, axis=None), fx.shape)])))
    #logger.info(np.column_stack(((np.sort(fx, axis=None)), np.moveaxis(pos,1,-1)[np.unravel_index(np.argsort(fx, axis=None), fx.shape)])))

    return space, pos, fx

def plot_pairplot_with_contours(space, pos, fx, cmap='plasma', hist_edgecolor='k', hist_bins=10):
    """
    Produce a pairplot showing parameter distributions and correlations.
    
    Parameters:
    - space: Dictionary with plot labels for axes
    - pos: 3D array of particle positions (iterations, dimensions, particles)
    - cmap: Colormap for density plots
    - hist_edgecolor: Color of histogram edges
    - hist_bins: Number of bins in histograms
    
    Returns:
    - A seaborn PairGrid object
    """
    
    # Rearrange positions to match DataFrame expectations
    S, D, L = pos.shape
    pos = np.swapaxes(pos, 0, 1)
    
    # Create DataFrame of all positions across iterations
    df = pd.DataFrame(pos.reshape((D, S * L)).T, columns=space['plot_label'])
    
    # Create pairplot
    g = sns.pairplot(df, corner=True, 
                     diag_kind="hist",
                     plot_kws={'alpha': 0.5},
                     diag_kws={'edgecolor': hist_edgecolor, 
                              'bins': hist_bins})
    
    # Add KDE plots to show density
    for i, ax in enumerate(g.axes.flat):
        if ax is None:
            continue
            
        row, col = divmod(i, len(space['plot_label']))
        if row > col:  # Lower triangle
            x_label = space['plot_label'][col]
            y_label = space['plot_label'][row]
            
            x_data = df[x_label]
            y_data = df[y_label]
            
            # Add KDE contours
            sns.kdeplot(data=df,
                       x=x_label,
                       y=y_label,
                       ax=ax,
                       cmap=cmap,
                       levels=10,
                       alpha=0.5)
            
            # Add correlation coefficient
            corr = df[x_label].corr(df[y_label])
            ax.text(0.05, 0.95, f'r = {corr:.2f}',
                   transform=ax.transAxes,
                   ha='left', va='top')
    
    return g

def load_sage_history():
    """Load SAGE history (CSFRDH/SMD)"""
    try:
        # Columns: Redshift, LookbackTime, logSFRD, logSMD
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        data = np.loadtxt(os.path.join(DATA_DIR, 'sage_history.csv'), skiprows=1)
        return data
    except Exception as e:
        print("Could not load sage_history.csv:", e)
        return None

def create_iteration_plot(filename, num_particles, num_iterations, obs_data, sage_data, track_folder, plot_type='SMF'):
    """
    Generic function to create iteration plots with customized settings per plot type.
    """
    # Plot settings dictionary for different constraint types
    plot_settings = {
        'SMF': {
            'xlabel': r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$',
            'ylabel': r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
            'xlim': [8.0, 12.2],
            'ylim': [1.0e-6, 1.0e-1],
            'yscale': 'log',
            'legend_loc': 'lower left',
            'transform_y': lambda y: 10**np.array(y)
        },
        'SMF_Red': {
            'xlabel': r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$',
            'ylabel': r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
            'xlim': [8.0, 12.2],
            'ylim': [1.0e-6, 1.0e-1],
            'yscale': 'log',
            'legend_loc': 'lower left',
            'transform_y': lambda y: 10**np.array(y)
        },
        'SMF_Blue': {
            'xlabel': r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$',
            'ylabel': r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
            'xlim': [8.0, 12.2],
            'ylim': [1.0e-6, 1.0e-1],
            'yscale': 'log',
            'legend_loc': 'lower left',
            'transform_y': lambda y: 10**np.array(y)
        },
        'BHMF': {
            'xlabel': r'$\log_{10} M_{\mathrm{bh}}\ (M_{\odot})$',
            'ylabel': r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
            'xlim': [6.0, 10.2],
            'ylim': [1.0e-6, 1.0e-1],
            'yscale': 'log',
            'legend_loc': 'upper right',
            'transform_y': lambda y: 10**np.array(y)
        },
        'BHBM': {
            'xlabel': r'$\log_{10} M_{\mathrm{bulge}}\ (M_{\odot})$',
            'ylabel': r'$\log_{10} M_{\mathrm{bh}}\ (M_{\odot})$',
            'xlim': [9.0, 12.5],
            'ylim': [5.0, 11.0],
            'yscale': 'linear',
            'legend_loc': 'upper left',
            'transform_y': lambda y: y
        },
        'HIMF': {
            'xlabel': r'$\log_{10} M_{\mathrm{HI}}\ (M_{\odot})$',
            'ylabel': r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
            'xlim': [8.0, 11.5],
            'ylim': [1.0e-6, 1.0e-1],
            'yscale': 'log',
            'legend_loc': 'upper right',
            'transform_y': lambda y: 10**np.array(y)
        },
        'H2MF': {
            'xlabel': r'$\log_{10} M_{\mathrm{H2}}\ (M_{\odot})$',
            'ylabel': r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
            'xlim': [8.0, 10.5],
            'ylim': [1.0e-5, 1.0e-1],
            'yscale': 'log',
            'legend_loc': 'upper right',
            'transform_y': lambda y: 10**np.array(y)
        },
        'MZR': {
            'xlabel': r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$',
            'ylabel': r'$12 + \log(\mathrm{O/H})$',
            'xlim': [8.5, 11.0],
            'ylim': [7.0, 9.5],
            'yscale': 'linear',
            'legend_loc': 'lower right',
            'transform_y': lambda y: y
        },
        'SHMR': {
            'xlabel': r'$\log_{10} M_{\mathrm{halo}}\ (M_{\odot})$',
            'ylabel': r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$',
            'xlim': [11.0, 15.0],
            'ylim': [8.0, 12.0],
            'yscale': 'linear',
            'legend_loc': 'lower right',
            'transform_y': lambda y: y
        },
        'SMD': {
            'xlabel': r'$z$',
            'ylabel': r'$\log_{10} \rho_{\star}\ (M_{\odot}\ \mathrm{Mpc}^{-3})$',
            'xlim': [0.0, 12.0],
            'ylim': [5.0, 9.0],
            'yscale': 'linear',
            'legend_loc': 'lower left',
            'transform_y': lambda y: y
        },
        'FICS': {
            'xlabel': r'$z$',
            'ylabel': r'$f_{\mathrm{ICS}} = m_{\mathrm{ICS}} / M_{\star,\mathrm{halo}}$',
            'xlim': [0.0, 2.0],
            'ylim': [0.0, 0.5],
            'yscale': 'linear',
            'legend_loc': 'upper right',
            'transform_y': lambda y: y
        },
        'FICS_Mvir': {
            'xlabel': r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$',
            'ylabel': r'$f_{\mathrm{ICS}} = m_{\mathrm{ICS}} / M_{\star,\mathrm{halo}}$',
            'xlim': [13.5, 15.3],
            'ylim': [0.0, 0.6],
            'yscale': 'linear',
            'legend_loc': 'upper left',
            'transform_y': lambda y: y
        },
        'MLF': {
            'xlabel': r'$\log_{10} v_{\mathrm{circ}}\ (\mathrm{km\ s}^{-1})$',
            'ylabel': r'$\log_{10} \eta\ (\dot{M}_{\mathrm{out}}/\mathrm{SFR})$',
            'xlim': [1.85, 2.75],
            'ylim': [-1.5, 1.5],
            'yscale': 'linear',
            'legend_loc': 'upper right',
            'transform_y': lambda y: y
        },
        'CSFRDH': {
            'xlabel': r'Lookback Time (Gyr)',
            'ylabel': r'$\log_{10}$ SFRD $(M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3})$',
            'xlim': [0.0, 14.0],
            'ylim': [-3.0, 0.0],
            'yscale': 'linear',
            'legend_loc': 'upper right',
            'transform_y': lambda y: y
        }
    }

    settings = plot_settings[plot_type]
    x_obs, y_obs, obs_label = obs_data
    x_sage, y_sage, sage_label = sage_data
    
    # Convert y values based on plot type
    if plot_type in ['SMF', 'SMF_Red', 'SMF_Blue', 'BHMF', 'HIMF', 'H2MF']:
        y_obs_converted = [10**y for y in y_obs]
        y_sage_converted = y_sage # Changed this line
    else:
        y_obs_converted = y_obs
        y_sage_converted = y_sage

    # Process data
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    blocks_by_iteration = []
    current_iteration = []
    current_block_y = []
    x_values = []
    capture_x = True

    # Parse blocks
    for line in lines:
        line = line.strip()
        if line.startswith("# New Data Block"):
            if current_block_y:
                current_iteration.append(current_block_y)
                current_block_y = []
                capture_x = False
                if len(current_iteration) == num_particles:
                    blocks_by_iteration.append(current_iteration)
                    current_iteration = []
                    if len(blocks_by_iteration) == num_iterations:
                        break
        elif line:
            values = list(map(float, line.split('\t')))
            if capture_x:
                x_values.append(values[0])
            current_block_y.append(values[2])

    # Process scores
    track_files = sorted(glob.glob(f"{track_folder}/track_*_fx.npy"))
    fit_scores = [np.load(file) for file in track_files[:num_iterations]]
    all_scores = np.concatenate(np.log10(fit_scores))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    lowest_score = np.inf
    lowest_score_line = None
    
    # Setup colormap
    colormap = cm.get_cmap('plasma_r', num_iterations)
    norm = Normalize(vmin=0, vmax=num_iterations - 1)

    # Plot iterations
    transform_y = settings['transform_y']
    for iteration_index, (blocks, scores) in enumerate(zip(blocks_by_iteration, fit_scores)):
        color = colormap(iteration_index)
        for particle_index, y in enumerate(blocks):
            transformed_y = transform_y(y)
            ax.plot(x_values, transformed_y, color=color, alpha=0.1, linewidth=0.75)
            
            score = np.log10(scores[particle_index])
            if score < lowest_score:
                lowest_score = score
                lowest_score_line = (x_values, transformed_y)

    ax.scatter(x_obs, y_obs_converted, color='k', s=50, marker='d', label=obs_label)
    ax.plot(x_sage, y_sage_converted, 'r--', linewidth=2.25, label=sage_label)
    if lowest_score_line is not None:
        ax.plot(lowest_score_line[0], lowest_score_line[1], 'b-', linewidth=2.25, label='PSO Best Fit')

    # Add SHARK data if available
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if plot_type == 'SMF' and 'SMF_z0_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_SMF.csv'), cols=[0,1])
        ax.plot(mass, transform_y(phi), 'g--', label='SHARK')

    elif plot_type == 'SMF' and 'SMF_z05_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_SMF.csv'), cols=[2,3])
        ax.plot(mass, transform_y(phi), 'g--', label='SHARK')

    elif plot_type == 'SMF' and 'SMF_z10_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_SMF.csv'), cols=[4,5])
        ax.plot(mass, transform_y(phi), 'g--', label='SHARK')

    elif plot_type == 'SMF' and 'SMF_z20_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_SMF.csv'), cols=[6,7])
        ax.plot(mass, transform_y(phi), 'g--', label='SHARK')

    elif plot_type == 'SMF' and 'SMF_z31_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_SMF.csv'), cols=[8,9])
        ax.plot(mass, transform_y(phi), 'g--', label='SHARK')

    elif plot_type == 'SMF' and 'SMF_z46_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_SMF.csv'), cols=[10,11])
        ax.plot(mass, transform_y(phi), 'g--', label='SHARK')

    elif plot_type == 'BHBM' and 'BHBM_z0_dump.txt' in filename:
        mass, phi = load_observation(os.path.join(DATA_DIR, 'SHARK_BHBM_z0.csv'), cols=[0,1])
        ax.plot(mass, phi, 'g--', label='SHARK')

        

    # Setup axes
    ax.set_xlabel(settings['xlabel'], fontsize=12)
    ax.set_ylabel(settings['ylabel'], fontsize=12)
    ax.set_xlim(settings['xlim'])
    ax.set_ylim(settings['ylim'])
    if settings['yscale'] == 'log':
        ax.set_yscale('log')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Iteration Number')

    # Add legend
    leg = ax.legend(loc=settings['legend_loc'])
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')

    plt.tight_layout()
    return fig

def smf_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='SMF')

def bhmf_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='BHMF')

def bhbm_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='BHBM')

def himf_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='HIMF')

def h2mf_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='H2MF')

def mzr_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='MZR')

def shmr_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='SHMR')

def smd_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='SMD')

def csfrdh_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='CSFRDH')

def fics_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='FICS')

def fics_mvir_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='FICS_Mvir')

def mlf_processing_iteration(*args, **kwargs):
    return create_iteration_plot(*args, **kwargs, plot_type='MLF')

def load_all_params(directory, param_names, redshifts):
    """Load parameter values from CSV files"""
    particle_data = {}
    best_params = {}
    best_scores = {}
    
    for z in redshifts:
        _, z_str = get_redshift_info(z=z)
        if z_str is None:
            logger.info(f"Unknown redshift mapping for z={z}")
            continue
            
        filename = os.path.join(directory, f'params_z{z_str}.csv')
        
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename, delimiter='\t', header=None, 
                               names=param_names + ['score'])
                
                particle_data[z] = df.iloc[:-2]
                best_params[z] = df.iloc[-2].values[:-1]  
                best_scores[z] = float(df.iloc[-1].iloc[0])
                logger.info(f"Successfully loaded data for z={z}")
            else:
                logger.info(f"File not found for z={z}: {filename}")
                continue
                
        except Exception as e:
            logger.error(f"Error processing file for z={z}: {str(e)}")
            continue
    
    return particle_data, best_params, best_scores

def setup_plot_style():
    """Setup consistent plot styling"""
    sns.set_theme(style="white", font_scale=1.2)
    sns.set_palette("deep")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    #plt.rcParams['grid.color'] = '#EEEEEE'
    #plt.rcParams['grid.linestyle'] = '-'

def extract_redshift(filename):
    """Extract redshift value from filename for sorting."""
    z, _ = get_redshift_info(filename=filename)
    return z if z is not None else float('inf')

def create_combined_constraint_grids(output_dir='parameter_plots', png_dir=None):
    """
    Creates combined grid figures for each plot type across different constraints (SMF, BHMF, BHBM).
    
    Parameters:
    -----------
    output_dir : str
        Directory where the grid plots will be saved
    png_dir : str
        Directory containing the source PNG files
    """
    if png_dir is None:
        png_dir = '.'
    elif not os.path.exists(png_dir):
        logger.info(f"PNG directory {png_dir} does not exist!")
        return
    
    from math import ceil, sqrt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define patterns to match different plot types
    plot_patterns = {
        'iterations': '*_all.png'
    }
    
    # Define constraints to look for
    constraints = ['SMF', 'BHMF', 'BHBM', 'SHMR']
    
    # Process each plot type
    for plot_type, pattern in plot_patterns.items():
        constraint_files = {}
        for file in glob.glob(os.path.join(png_dir, f'*{pattern}')):
            constraint = os.path.basename(file).split('_')[0]
            z = extract_redshift(os.path.basename(file))
            if constraint not in constraint_files:
                constraint_files[constraint] = {}
            constraint_files[constraint][z] = file
        
        if not constraint_files:
            continue
        
        constraints = sorted(constraint_files.keys())
        redshifts = sorted(set(z for c in constraints for z in constraint_files[c]))

        # Layout: 1 row if <=3 constraints, else 2 rows
        if len(constraints) <= 3:
            n_rows = 1
            n_cols = len(constraints)
        else:
            n_rows = 2
            n_cols = (len(constraints) + 1) // 2

        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

        # Flatten constraints for grid
        constraint_list = constraints.copy()
        while len(constraint_list) < n_rows * n_cols:
            constraint_list.append(None)

        for idx, constraint in enumerate(constraint_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            if constraint is not None:
                # Pick first available redshift for each constraint
                z_list = sorted(constraint_files[constraint].keys())
                file = constraint_files[constraint][z_list[0]] if z_list else None
                if file:
                    img = mpimg.imread(file)
                    ax.imshow(img)
                ax.set_title(constraint)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'combined_{plot_type}_grid.png'), dpi=300)
        plt.close()
        logger.info("Created combined grid figure")

def create_parameter_gif(param_matrix, param_names, output_path, scores=None, particles=None):
    from mpl_toolkits.mplot3d import Axes3D
    
    # Ensure 3D shape: (Iterations, Particles, Params)
    if param_matrix.ndim == 2 and particles is not None:
        iterations = param_matrix.shape[0] // particles
        n_params = param_matrix.shape[1]
        param_matrix = param_matrix.reshape((iterations, particles, n_params))
    
    iterations, n_particles, n_params = param_matrix.shape
    
    fig = plt.figure()
    
    # Check if we have enough parameters for 3D
    is_3d = n_params >= 3
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    def update(i):
        ax.clear()
        xs = param_matrix[i, :, 0]
        ys = param_matrix[i, :, 1]
        
        # Handle Scores (Color)
        if scores is not None:
            # scores should be (Iterations, Particles)
            cs = scores[i] if scores.ndim == 2 else scores
            
            if is_3d:
                zs = param_matrix[i, :, 2]
                sc = ax.scatter(xs, ys, zs, c=cs, cmap='plasma', alpha=0.7)
                ax.set_zlabel(param_names[2])
            else:
                sc = ax.scatter(xs, ys, c=cs, cmap='plasma', alpha=0.7)
        else:
            if is_3d:
                zs = param_matrix[i, :, 2]
                ax.scatter(xs, ys, zs, color='b', alpha=0.7)
                ax.set_zlabel(param_names[2])
            else:
                ax.scatter(xs, ys, color='b', alpha=0.7)

        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_title(f"Iteration {i}")

    # Use Pillow writer to avoid ImageMagick dependency
    ani = anim.FuncAnimation(fig, update, frames=iterations, repeat=False)
    ani.save(output_path, writer='pillow', fps=2)

def load_sage_data():
    """Load SMF data from SAGE (Matched to main.py output)"""
    # We have 6 snapshots (z=0, 0.5, 1.0, 2.0, 3.0, 4.0) -> 12 Columns
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    sage_data = load_observation(os.path.join(DATA_DIR, 'sage_smf_all_redshifts.csv'), cols=list(range(12)))

    data_by_z = {}

    # These must match the 'target_snapshots' order in main.py
    # Column order corresponds to z=0, 0.5, 1.0, 2.0, 3.0, 4.0 (simulation-specific snapshots)
    redshifts = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    
    # Column indices pairs (Mass, Phi) for each redshift
    col_indices = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]
    
    for z, (mass_idx, phi_idx) in zip(redshifts, col_indices):
        logm = sage_data[mass_idx]
        logphi = sage_data[phi_idx]
        
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        data_by_z[z] = (logm[valid_mask], logphi[valid_mask], f'SAGE (z={z})')
        
    return data_by_z

def load_sage_data_forBHMF():
    """Load BHMF data from SAGE (Matched to main.py output)"""
    # We have 6 snapshots (z=0, 0.5, 1.0, 2.0, 3.0, 4.0) -> 12 Columns
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    sage_data = load_observation(os.path.join(DATA_DIR, 'sage_bhmf_all_redshifts.csv'), cols=list(range(12)))
    
    data_by_z = {}
    
    redshifts = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    col_indices = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]
    
    for z, (mass_idx, phi_idx) in zip(redshifts, col_indices):
        logm = sage_data[mass_idx]
        logphi = sage_data[phi_idx]
        
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        data_by_z[z] = (logm[valid_mask], logphi[valid_mask], f'SAGE (z={z})')
        
    return data_by_z

def load_bhbm_data():
    """Load BHBM data from SAGE-miniUchuu and observations"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    # Load combined individual galaxy observations
    # H&R 2004 + Davis 2019 (spirals) + Sahu 2019 (ellipticals)
    log_bulgemass, log_blackholemass = load_observation(
        os.path.join(DATA_DIR, 'bhbm_obs_combined.csv'), cols=[0, 1], skiprows=3)

    # Load SAGE data
    sage_bhbm_data = load_observation(os.path.join(DATA_DIR, 'sage_bhbm_all_redshifts.csv'), cols=[0,1,2,3,4,5,6,7,8,9,10,11])

    data_by_z = {}
    for z, (mass_idx, bh_idx) in zip([0.0], [(0, 1)]):
        logm_sage = sage_bhbm_data[mass_idx]
        logbh_sage = sage_bhbm_data[bh_idx]
        valid_mask = ~np.isnan(logm_sage) & ~np.isnan(logbh_sage)
        data_by_z[z] = (
            (log_bulgemass, log_blackholemass, 'H&R04 + Davis19 + Sahu19'),
            (logm_sage[valid_mask], logbh_sage[valid_mask], f'SAGE (z={z})')
        )

    return data_by_z

def load_shuntov_data():
    """Load SMF data from Shuntov et al. 2024"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    shuntov_data = load_observation(os.path.join(DATA_DIR, 'shuntov_2024_all.csv'), cols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                                                              15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # Dictionary to store data for each redshift
    data_by_z = {}
    
    # List of redshifts and their corresponding column indices
    redshifts = [0.2, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0, 2.4, 3.1, 3.6, 4.6, 5.7, 6.3, 7.7, 8.5, 10.4]
    col_indices = [(0,1), (2,3), (4,5), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15), 
                   (16,17), (18,19), (20,21), (22,23), (24,25), (26,27), (28,29)]
    
    # Process data for each redshift
    for z, (mass_idx, phi_idx) in zip(redshifts, col_indices):
        logm = shuntov_data[mass_idx]
        logphi = shuntov_data[phi_idx]
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        data_by_z[z] = (logm[valid_mask], logphi[valid_mask], f'Shuntov et al., 2024 (z={z})')
    
    return data_by_z

def load_zhang_data():
    """Load SMF data from Zhang et al. 2024"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    zhang_data = load_observation(os.path.join(DATA_DIR, 'zhang_data.csv'), cols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                                                              15,16,17,18,19])
    # Dictionary to store data for each redshift
    data_by_z = {}
    
    # List of redshifts and their corresponding column indices
    redshifts = [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    col_indices = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15), 
                   (16,17), (18,19)]
    
    # Process data for each redshift
    for z, (mass_idx, phi_idx) in zip(redshifts, col_indices):
        logm = zhang_data[mass_idx]
        logphi = zhang_data[phi_idx]
        valid_mask = ~np.isnan(logm) & ~np.isnan(logphi)
        data_by_z[z] = (logm[valid_mask], logphi[valid_mask], f'Zhang et al., 2024 (z={z})')
    
    return data_by_z

def load_gama_data(config_opts):
    """Load GAMA data for z=0"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    logm, logphi, dlogphi = load_observation(os.path.join(DATA_DIR, 'GAMA_SMF_highres.csv'), cols=[0,1,2])
    
    # Use passed parameters
    cosmology_correction_median = np.log10(r.comoving_distance(0.079, 100*config_opts.h0, 0, config_opts.Omega0, 1.0-config_opts.Omega0) / 
                                         r.comoving_distance(0.079, 70.0, 0, 0.3, 0.7))
    cosmology_correction_maximum = np.log10(r.comoving_distance(0.1, 100*config_opts.h0, 0, config_opts.Omega0, 1.0-config_opts.Omega0) / 
                                          r.comoving_distance(0.1, 70.0, 0, 0.3, 0.7))
    
    x_obs = logm + 2.0 * cosmology_correction_median 
    y_obs = logphi - 3.0 * cosmology_correction_maximum + 0.0807
    
    return x_obs, y_obs, 'Driver et al., 2022 (GAMA)'

def load_ilbert_data(config_opts):
    """Load GAMA data for z=0"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    logm, logphi = load_observation(os.path.join(DATA_DIR, 'Ilbert_2010_z1.csv'), cols=[0,1])

    x_obs = logm
    y_obs = logphi

    return x_obs, y_obs, 'Ilbert et al., 2010'

def load_gama_red_data(config_opts):
    """Load GAMA morphological SMF data for Red/Quiescent (E+HE) galaxies at z=0"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    path = os.path.join(DATA_DIR, 'gama_smf_morph.ecsv')
    if not os.path.exists(path):
        logger.warning(f"gama_smf_morph.ecsv not found at {path}, skipping red SMF diagnostic")
        return None
    gama_mass, gama_E_HE, _ = np.loadtxt(path, usecols=[0,1,2], unpack=True, skiprows=15)
    valid = ~np.isnan(gama_E_HE)
    return gama_mass[valid], gama_E_HE[valid], 'GAMA E+HE (Red)'

def load_gama_blue_data(config_opts):
    """Load GAMA morphological SMF data for Blue/Star-forming (D) galaxies at z=0"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    path = os.path.join(DATA_DIR, 'gama_smf_morph.ecsv')
    if not os.path.exists(path):
        logger.warning(f"gama_smf_morph.ecsv not found at {path}, skipping blue SMF diagnostic")
        return None
    gama_mass, gama_D, _ = np.loadtxt(path, usecols=[0,7,8], unpack=True, skiprows=15)
    valid = ~np.isnan(gama_D)
    return gama_mass[valid], gama_D[valid], 'GAMA Disk (Blue)'

def load_sage_smf_red_data():
    """Load SAGE Red SMF data"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    sage_data = load_observation(os.path.join(DATA_DIR, 'sage_smf_red_all_redshifts.csv'), cols=[0,1])

    data_by_z = {}
    # z=0 is columns 0,1
    logm = sage_data[0]
    phi = sage_data[1]
    valid_mask = ~np.isnan(logm) & ~np.isnan(phi) & (phi > 0)
    data_by_z[0.0] = (logm[valid_mask], phi[valid_mask], 'SAGE Red (z=0)')

    return data_by_z

def load_sage_smf_blue_data():
    """Load SAGE Blue SMF data"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    sage_data = load_observation(os.path.join(DATA_DIR, 'sage_smf_blue_all_redshifts.csv'), cols=[0,1])

    data_by_z = {}
    # z=0 is columns 0,1
    logm = sage_data[0]
    phi = sage_data[1]
    valid_mask = ~np.isnan(logm) & ~np.isnan(phi) & (phi > 0)
    data_by_z[0.0] = (logm[valid_mask], phi[valid_mask], 'SAGE Blue (z=0)')

    return data_by_z

def load_wright_z1_data(config_opts):
    """Load Wright data for z=1"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    logm, logphi = load_observation(os.path.join(DATA_DIR, 'Wright_2018_z1_z2.csv'), cols=[0,1])
    
    x_obs = logm
    y_obs = logphi
    
    return x_obs, y_obs, 'Wright et al., 2018'

def load_wright_z2_data(config_opts):
    """Load Wright data for z=2"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    logm, logphi = load_observation(os.path.join(DATA_DIR, 'Wright_2018_z1_z2.csv'), cols=[2,3])
    
    x_obs = logm
    y_obs = logphi
    
    return x_obs, y_obs, 'Wright et al., 2018'

def get_smf_files_map(config_opts):
    """Create mapping of SMF dump files to their corresponding observational data"""
    # Load all observational data
    shuntov_data = load_shuntov_data()
    gama_data = load_gama_data(config_opts)
    sage_data = load_sage_data()
    ilbert_data = load_ilbert_data(config_opts)
    wright_z1_data = load_wright_z1_data(config_opts)
    wright_z2_data = load_wright_z2_data(config_opts)
    
    logger.debug("Checking for SMF dump files in directory...")
    smf_files = {}
    
    # Handle z=0 case specially since it uses GAMA data
    filename = f'SMF_z0_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        smf_files[filename] = (gama_data, sage_data[0.0])
    else:
        logger.debug(f"Not found: {filename}")

    # Handle z=0 case specially since it uses GAMA data
    filename = f'SMF_z10_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        smf_files[filename] = (wright_z1_data, sage_data[1.0])
    else:
        logger.debug(f"Not found: {filename}")

    # Handle z=0 case specially since it uses GAMA data
    filename = f'SMF_z20_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        smf_files[filename] = (wright_z2_data, sage_data[2.0])
    else:
        logger.debug(f"Not found: {filename}")
    
    # Handle all other redshifts using Shuntov data
    for z in get_all_redshifts():
        if z in [0.0,1.0,2.0]:  # Skip z=0 as it's handled above
            continue
        
        _, z_str = get_redshift_info(z=z)
        if z_str is None:
            continue
        filename = f'SMF_z{z_str}_dump.txt'
        filepath = os.path.join(config_opts.outdir, filename)
        if os.path.exists(filepath):
            if z not in shuntov_data:
                logger.warning(f"No Shuntov observational data for z={z}, skipping {filename}")
                continue
            if z not in sage_data:
                logger.warning(f"No SAGE reference data for z={z}, skipping {filename}")
                continue
            logger.debug(f"Found: {filename}")
            smf_files[filename] = (shuntov_data[z], sage_data[z])
        else:
            logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(smf_files)} SMF files to process")
    return smf_files

def get_smf_red_files_map(config_opts):
    """Create mapping of SMF Red dump files to their corresponding observational data"""
    gama_red_data = load_gama_red_data(config_opts)
    if gama_red_data is None:
        return {}
    sage_red_data = load_sage_smf_red_data()

    logger.debug("Checking for SMF_Red dump files in directory...")
    smf_red_files = {}

    filename = 'SMF_Red_z0_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        smf_red_files[filename] = (gama_red_data, sage_red_data[0.0])
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(smf_red_files)} SMF_Red files to process")
    return smf_red_files

def get_smf_blue_files_map(config_opts):
    """Create mapping of SMF Blue dump files to their corresponding observational data"""
    gama_blue_data = load_gama_blue_data(config_opts)
    if gama_blue_data is None:
        return {}
    sage_blue_data = load_sage_smf_blue_data()

    logger.debug("Checking for SMF_Blue dump files in directory...")
    smf_blue_files = {}

    filename = 'SMF_Blue_z0_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        smf_blue_files[filename] = (gama_blue_data, sage_blue_data[0.0])
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(smf_blue_files)} SMF_Blue files to process")
    return smf_blue_files

def get_bhmf_files_map(config_opts):
    """Create mapping of BHMF dump files to their corresponding observational data"""
    # Load all observational data
    zhang_data = load_zhang_data()
    sage_data = load_sage_data_forBHMF()
    
    logger.debug("Checking for BHMF dump files in directory...")
    bhmf_files = {}
    
    for z in get_all_redshifts():
        _, z_str = get_redshift_info(z=z)
        if z_str is None:
            continue
            
        filename = f'BHMF_z{z_str}_dump.txt'
        filepath = os.path.join(config_opts.outdir, filename)
        if os.path.exists(filepath):
            if z not in zhang_data or z not in sage_data:
                logger.warning(f"No data for z={z}, skipping {filename}")
                continue
            logger.debug(f"Found: {filename}")
            bhmf_files[filename] = (zhang_data[z], sage_data[z])
        else:
            logger.debug(f"Not found: {filename}")
    
    logger.debug(f"Found {len(bhmf_files)} BHMF files to process")
    return bhmf_files

def get_bhbm_files_map(config_opts):
    """Create mapping of BHBM dump files to their corresponding observational data"""
    # Load observational and SAGE data
    bhbm_data = load_bhbm_data()
    
    logger.debug("Checking for BHBM dump files in directory...")
    bhbm_files = {}
    
    for z in get_all_redshifts():
        _, z_str = get_redshift_info(z=z)
        if z_str is None:
            continue
        if z not in bhbm_data:
            logger.debug(f"No BHBM data for z={z}, skipping.")
            continue
        filename = f'BHBM_dump.txt'
        filepath = os.path.join(config_opts.outdir, filename)
        if os.path.exists(filepath):
            logger.debug(f"Found: {filename}")
            obs_data, sage_data = bhbm_data[z]
            bhbm_files[filename] = (obs_data, sage_data)
        else:
            logger.debug(f"Not found: {filename}")
  
    
    logger.debug(f"Found {len(bhbm_files)} BHBM files to process")
    return bhbm_files



# Cosmology used when re-loading observations for plotting.  Some loaders
# h-scale their data (HIMF, CSFRDH), so they need it; processing() sets this
# from the run's own values so the plotted points match the scored ones exactly.
_PLOT_COSMOLOGY = {'h0': 0.6774, 'Omega0': 0.3089}


def set_plot_cosmology(h0=None, Omega0=None):
    """Record the run's cosmology for observation loaders that h-scale."""
    if h0:
        _PLOT_COSMOLOGY['h0'] = float(h0)
    if Omega0:
        _PLOT_COSMOLOGY['Omega0'] = float(Omega0)


def constraint_obs(constraint_name, label=None, fallback=None):
    """Observations a constraint is actually scored against, for plotting.

    Delegates to the constraint's own get_obs_x_y_err(), so a diagnostic plot
    can never show a different dataset from the one the objective used.  Four
    of these loaders had drifted: the HIMF plot showed Zwaan+2005 while the fit
    used Jones+2018, the MZR plot showed Tremonti+2004 while the fit used
    Curti+2020, the SHMR plot showed Moster+2013 while the fit used
    Correa & Schaye 2019, and the CSFRDH plot pointed at a file that does not
    exist and silently fell back to a two-point dummy.  Looking at those plots
    would have told you the fit was better or worse than it was.

    __new__ bypasses __init__ because only the observation loader is needed and
    it does not depend on any simulation parameter.
    """
    from src import constraints as _C
    try:
        cls = getattr(_C, constraint_name)
        c = cls.__new__(cls)
        # Attributes the loaders may need; __init__ is skipped because it also
        # resolves snapshots, which plotting does not require.
        c.h0 = _PLOT_COSMOLOGY['h0']
        c.Omega0 = _PLOT_COSMOLOGY['Omega0']
        c.snapshot = None
        x_obs, y_obs, _, _ = c.get_obs_x_y_err()
        x_obs, y_obs = np.asarray(x_obs), np.asarray(y_obs)

        # get_data scores only the points inside the domain, so those are the
        # only ones the plot should show.  Without this the figure includes
        # points the objective never saw -- the HIMF file reaches down to
        # log M_HI = 6.5 but is only scored above 8.0, and the eye reads the
        # unscored points as part of the fit.
        lo, hi = getattr(cls, 'domain', (-np.inf, np.inf))
        keep = (x_obs >= lo) & (x_obs <= hi)
        if np.any(keep):
            x_obs, y_obs = x_obs[keep], y_obs[keep]

        return (x_obs, y_obs,
                label or ('%s observations' % constraint_name))
    except Exception as e:
        logger.warning('could not load %s observations for plotting: %s',
                       constraint_name, e)
        if fallback is not None:
            return fallback
        return (np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
                '%s observations (unavailable)' % constraint_name)

def load_himf_obs_data():
    """HIMF observations, as scored (see constraint_obs)."""
    return constraint_obs('HIMF', label='Jones et al. 2018')



def load_h2mf_obs_data():
    """H2MF observations, as scored (see constraint_obs)."""
    return constraint_obs('H2MF', label='Fletcher et al. 2021')



def load_mzr_obs_data():
    """MZR observations, as scored (see constraint_obs)."""
    return constraint_obs('MZR', label='Curti et al. 2020')



def load_shmr_obs_data():
    """SHMR observations, as scored (see constraint_obs)."""
    return constraint_obs('SHMR', label='Correa & Schaye 2019')



def load_smd_obs_data():
    """SMD observations, as scored (see constraint_obs)."""
    return constraint_obs('SMD', label='Weaver et al. 2023 + Madau & Dickinson 2014')



def load_sage_himf_data():
    """Load HIMF data from SAGE CSV file"""
    try:
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        himf_data = np.loadtxt(os.path.join(DATA_DIR, 'sage_himf_all_redshifts.csv'))
        # z=0 data is in first two columns (mass, phi)
        logm = himf_data[:, 0]
        phi = himf_data[:, 1]
        # Filter out NaN and zero/negative values
        valid_mask = ~np.isnan(logm) & ~np.isnan(phi) & (phi > 0)
        if np.sum(valid_mask) > 0:
            return (logm[valid_mask], phi[valid_mask], 'SAGE')
        else:
            logger.warning("No valid HIMF SAGE data found")
            return (np.array([9.0, 10.0]), np.array([1e-2, 1e-3]), 'SAGE')
    except Exception as e:
        logger.warning(f"Could not load sage_himf_all_redshifts.csv: {e}")
        return (np.array([9.0, 10.0]), np.array([1e-2, 1e-3]), 'SAGE')


def get_himf_files_map(config_opts):
    """Create mapping of HIMF dump files to their corresponding observational data"""
    obs_data = load_himf_obs_data()
    sage_data = load_sage_himf_data()

    logger.debug("Checking for HIMF dump files in directory...")
    files = {}

    filename = 'HIMF_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} HIMF files to process")
    return files


def load_sage_h2mf_data():
    """Load H2MF data from SAGE CSV file"""
    try:
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        h2mf_data = np.loadtxt(os.path.join(DATA_DIR, 'sage_h2mf_all_redshifts.csv'))
        # z=0 data is in first two columns (mass, phi)
        logm = h2mf_data[:, 0]
        phi = h2mf_data[:, 1]
        # Filter out NaN and zero/negative values
        valid_mask = ~np.isnan(logm) & ~np.isnan(phi) & (phi > 0)
        if np.sum(valid_mask) > 0:
            return (logm[valid_mask], phi[valid_mask], 'SAGE')
        else:
            logger.warning("No valid H2MF SAGE data found")
            return (np.array([9.0, 10.0]), np.array([1e-2, 1e-3]), 'SAGE')
    except Exception as e:
        logger.warning(f"Could not load sage_h2mf_all_redshifts.csv: {e}")
        return (np.array([9.0, 10.0]), np.array([1e-2, 1e-3]), 'SAGE')


def get_h2mf_files_map(config_opts):
    """Create mapping of H2MF dump files to their corresponding observational data"""
    obs_data = load_h2mf_obs_data()
    sage_data = load_sage_h2mf_data()

    logger.debug("Checking for H2MF dump files in directory...")
    files = {}

    filename = 'H2MF_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} H2MF files to process")
    return files


def get_mzr_files_map(config_opts):
    """Create mapping of MZR dump files to their corresponding observational data"""
    obs_data = load_mzr_obs_data()
    # LOAD ACTUAL SAGE DATA
    # We saved it as 'sage_mzr_all_redshifts.csv' in main.py
    # Columns 0,1 correspond to z=0 (first in target_snapshots, simulation-specific)
    try:
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        mzr_data = np.loadtxt(os.path.join(DATA_DIR, 'sage_mzr_all_redshifts.csv'))
        x_sage = mzr_data[:, 0]  # log10(Mstellar)
        y_sage = mzr_data[:, 1]  # 12+log(O/H)
        # Filter out NaN values
        valid = ~np.isnan(x_sage) & ~np.isnan(y_sage)
        sage_data = (x_sage[valid], y_sage[valid], 'SAGE')
    except Exception as e:
        logger.warning(f"Could not load sage_mzr_all_redshifts.csv: {e}")
        sage_data = (np.array([9.0, 11.0]), np.array([8.5, 9.0]), 'SAGE')

    logger.debug("Checking for MZR dump files in directory...")
    files = {}

    filename = 'MZR_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} MZR files to process")
    return files


def get_shmr_files_map(config_opts):
    """Create mapping of SHMR dump files to their corresponding observational data"""
    obs_data = load_shmr_obs_data()
    
    # LOAD ACTUAL SAGE DATA
    # Use a robust loader to handle '' headers and whitespace
    x_sage_vals = []
    y_sage_vals = []
    
    # Resolve path to data directory
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    sage_file = os.path.join(DATA_DIR, 'sage_halostellar_all_redshifts.csv')
    
    try:
        if os.path.exists(sage_file):
            with open(sage_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines, metadata headers like '', etc.
                    if not line or line.startswith('[') or 'source:' in line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                        
                    try:
                        # Columns 0 (Halo) and 1 (Stellar)
                        val_x = float(parts[0])
                        val_y = float(parts[1])
                        
                        # Only keep valid finite numbers
                        if np.isfinite(val_x) and np.isfinite(val_y):
                            x_sage_vals.append(val_x)
                            y_sage_vals.append(val_y)
                    except ValueError:
                        continue
                        
            if len(x_sage_vals) > 0:
                sage_data = (np.array(x_sage_vals), np.array(y_sage_vals), 'SAGE')
            else:
                logger.warning(f"File {sage_file} exists but no valid data found.")
                sage_data = (np.array([12.0, 14.0]), np.array([10.0, 11.0]), 'SAGE (Empty)')
        else:
            logger.warning(f"SHMR SAGE file not found: {sage_file}")
            sage_data = (np.array([12.0, 14.0]), np.array([10.0, 11.0]), 'SAGE (Missing)')

    except Exception as e:
        logger.warning(f"Could not load sage_halostellar_all_redshifts.csv: {e}")
        sage_data = (np.array([12.0, 14.0]), np.array([10.0, 11.0]), 'SAGE (Error)')

    logger.debug("Checking for SHMR dump files in directory...")
    files = {}

    filename = 'SHMR_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} SHMR files to process")
    return files


def get_smd_files_map(config_opts):
    """Create mapping of SMD dump files to their corresponding observational data"""
    obs_data = load_smd_obs_data()
    # LOAD ACTUAL SAGE DATA
    hist_data = load_sage_history()
    if hist_data is not None:
        # Col 0 is Redshift, Col 3 is logSMD
        x_sage = hist_data[:, 0]  # Redshift
        y_sage = hist_data[:, 3]  # logSMD
        # Filter out invalid values (NaN and -99 placeholders)
        valid = ~np.isnan(x_sage) & ~np.isnan(y_sage) & (y_sage > -50)
        sage_data = (x_sage[valid], y_sage[valid], 'SAGE')
    else:
        sage_data = (np.array([0.0, 2.0]), np.array([8.0, 7.5]), 'SAGE')

    logger.debug("Checking for SMD dump files in directory...")
    files = {}

    filename = 'SMD_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} SMD files to process")
    return files


def load_fics_obs_data():
    """Load the binned ICS mass fraction compilation used by the FICS constraint.

    Delegates to FICS.get_obs_x_y_err() so the diagnostic plot shows exactly the
    binned points the optimiser is scored against, rather than a second,
    independently binned version of the same compilation.
    """
    try:
        from src.constraints import FICS
        c = FICS.__new__(FICS)          # bypass __init__: only the obs loader is needed
        x_obs, y_obs, _, _ = c.get_obs_x_y_err()
        return (x_obs, y_obs, 'ICL fraction compilation')
    except Exception as e:
        logger.warning(f"Could not load ICL_fraction_compilation.dat: {e}")
        return (np.array([0.05, 0.5, 1.2]), np.array([0.18, 0.06, 0.06]),
                'ICL fraction compilation')


def get_fics_files_map(config_opts):
    """Create mapping of FICS dump files to their corresponding observational data"""
    obs_data = load_fics_obs_data()
    # LOAD ACTUAL SAGE DATA
    hist_data = load_sage_history()
    if hist_data is not None and hist_data.shape[1] > 4:
        # Col 0 is Redshift, Col 4 is f_ICS (linear fraction, -99 where unmeasured)
        x_sage = hist_data[:, 0]
        y_sage = hist_data[:, 4]
        valid = (np.isfinite(x_sage) & np.isfinite(y_sage) & (y_sage > 0)
                 & (x_sage <= 2.0))
        if np.any(valid):
            sage_data = (x_sage[valid], y_sage[valid], 'SAGE')
        else:
            sage_data = (np.array([0.0, 2.0]), np.array([np.nan, np.nan]), 'SAGE (no ICS)')
    else:
        sage_data = (np.array([0.0, 2.0]), np.array([np.nan, np.nan]), 'SAGE (Missing)')

    logger.debug("Checking for FICS dump files in directory...")
    files = {}

    filename = 'FICS_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} FICS files to process")
    return files


def load_fics_mvir_obs_data():
    """Load the Contini (2021) f_ICL-vs-halo-mass compilation.

    Delegates to FICS_Mvir.get_obs_x_y_err() so the diagnostic plot shows the
    same points the optimiser is scored against.
    """
    try:
        from src.constraints import FICS_Mvir
        c = FICS_Mvir.__new__(FICS_Mvir)   # bypass __init__: only obs needed
        x_obs, y_obs, _, _ = c.get_obs_x_y_err()
        return (x_obs, y_obs, 'Contini 2021 compilation')
    except Exception as e:
        logger.warning(f"Could not load Contini2021_ICL_fraction_vs_Mvir.dat: {e}")
        return (np.array([14.2, 14.7, 15.1]), np.array([0.12, 0.24, 0.12]),
                'Contini 2021 compilation')


def get_fics_mvir_files_map(config_opts):
    """Create mapping of FICS_Mvir dump files to their observational data"""
    obs_data = load_fics_mvir_obs_data()
    # No SAGE reference curve for this relation: sage_history.csv is a function
    # of redshift, not halo mass, so there is nothing to overlay.
    sage_data = (np.array([14.0, 15.0]), np.array([np.nan, np.nan]), 'SAGE (n/a)')

    logger.debug("Checking for FICS_Mvir dump files in directory...")
    files = {}
    filename = 'FICS_Mvir_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")
    logger.debug(f"Found {len(files)} FICS_Mvir files to process")
    return files


def load_mlf_obs_data():
    """MLF observations, as scored (see constraint_obs)."""
    return constraint_obs('MLF', label='outflow compilation (Rupke+05, '
                                       'Heckman+15, Chisholm+17, Sugahara+17)')


def get_mlf_files_map(config_opts):
    """Create mapping of MLF dump files to their observational data"""
    obs_data = load_mlf_obs_data()
    sage_data = (np.array([1.9, 2.7]), np.array([np.nan, np.nan]), 'SAGE (n/a)')
    files = {}
    filename = 'MLF_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        files[filename] = (obs_data, sage_data)
    return files


def load_csfrdh_obs_data():
    """CSFRDH observations, as scored (see constraint_obs)."""
    return constraint_obs('CSFRDH', label='COSMOS-Web, inferred from the SMD')



def get_csfrdh_files_map(config_opts):
    """Create mapping of CSFRDH dump files to their corresponding observational data"""
    obs_data = load_csfrdh_obs_data()
    # LOAD ACTUAL SAGE DATA
    hist_data = load_sage_history()
    if hist_data is not None:
        # Col 1 is Lookback Time, Col 2 is logSFRD
        x_sage = hist_data[:, 1]  # Lookback Time
        y_sage = hist_data[:, 2]  # logSFRD
        # Filter out invalid values (NaN and -99 placeholders)
        valid = ~np.isnan(x_sage) & ~np.isnan(y_sage) & (y_sage > -50)
        sage_data = (x_sage[valid], y_sage[valid], 'SAGE')
    else:
        # Fallback only if file missing
        sage_data = (np.array([0.0, 10.0]), np.array([-1.0, -2.0]), 'SAGE')

    logger.debug("Checking for CSFRDH dump files in directory...")
    files = {}

    filename = 'CSFRDH_dump.txt'
    filepath = os.path.join(config_opts.outdir, filename)
    if os.path.exists(filepath):
        logger.debug(f"Found: {filename}")
        files[filename] = (obs_data, sage_data)
    else:
        logger.debug(f"Not found: {filename}")

    logger.debug(f"Found {len(files)} CSFRDH files to process")
    return files


def file_exists_and_not_empty(filepath):
    """Check if a file exists and is not empty."""
    if not os.path.exists(filepath):
        return False
    return os.path.getsize(filepath) > 0

def read_smf_dump_file(filename, n_particles, skip_iterations):
    """
    Read SMF dump file and extract SMF values for all particles.
    Each particle is mark_ed by "# New Data Block" in the file.
    Skips the first skip_iterations iterations.
    
    Parameters:
    -----------
    filename : str
        Path to the SMF dump file
    n_particles : int
        Number of particles per iteration  
    skip_iterations : int
        Number of iterations to skip at the start (default=5)
        
    Returns:
    --------
    x_values : array
        Mass bins
    smf_values : array 
        SMF values for all particles after skipped iterations
    """
    logger = logging.getLogger('diagnostics')
    x_values = None 
    current_block = []
    all_blocks = []
    current_iteration = []
    iteration_blocks = []
    
    logger.info(f"Reading file: {filename}")
    logger.info(f"Skipping first {skip_iterations} iterations ({skip_iterations * n_particles} blocks)")
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("# New Data Block"):
                    if current_block:
                        values = np.array(current_block)
                        if x_values is None:
                            x_values = values[:, 0]  # Mass bins
                            
                        current_iteration.append(values[:, 2])
                        if len(current_iteration) == n_particles:
                            iteration_blocks.append(np.array(current_iteration))
                            current_iteration = []
                        
                        current_block = []
                elif line:
                    try:
                        values = list(map(float, line.split('\t')))
                        current_block.append(values)
                    except (ValueError, IndexError):
                        continue
                        
        # Process last block if needed
        if current_block:
            values = np.array(current_block)
            current_iteration.append(values[:, 2])
            if len(current_iteration) == n_particles:
                iteration_blocks.append(np.array(current_iteration))
        
        # Skip iterations and process remaining ones
        if skip_iterations < len(iteration_blocks):
            iteration_blocks = iteration_blocks[skip_iterations:]
            smf_values = np.vstack(iteration_blocks)
        else:
            logger.error(f"Not enough iterations to skip {skip_iterations}")
            return None, None
            
        if smf_values is not None:
            logger.info(f"Mass bins shape: {x_values.shape}")
            logger.info(f"SMF values shape: {smf_values.shape}")
            logger.info(f"Skipped {skip_iterations} iterations, kept {len(iteration_blocks)} iterations")
        
        return x_values, smf_values
        
    except Exception as e:
        logger.error(f"Error reading SMF dump file: {str(e)}")
        return None, None

def get_aligned_parameter_values(track_files, n_particles, n_iterations, skip_iterations):
    """
    Load parameter values from track files and align them with SMF data iterations.
    
    Parameters:
    -----------
    track_files : list
        List of sorted track file paths
    n_particles : int
        Number of particles per iteration
    n_iterations : int  
        Number of iterations to load after skipping
    skip_iterations : int
        Number of iterations to skip at start (default=5)
        
    Returns:
    --------
    param_values : array
        Parameter values aligned with SMF data after skipped iterations
        Shape: (n_iterations * n_particles, n_params)
    """
    logger = logging.getLogger('diagnostics')
    
    start_idx = skip_iterations
    end_idx = skip_iterations + n_iterations
    
    # Verify we have enough files
    if end_idx > len(track_files):
        logger.error(f"Not enough track files for requested iterations")
        return None
        
    # Load only the iterations we need
    all_positions = []
    for pos_file in track_files[start_idx:end_idx]:
        try:
            pos = np.load(pos_file)
            if pos.shape[0] != n_particles:
                logger.warning(f"Unexpected number of particles in {pos_file}: {pos.shape[0]} vs expected {n_particles}")
                continue
            all_positions.append(pos)
        except Exception as e:
            logger.error(f"Error loading track file {pos_file}: {e}")
            continue
            
    if not all_positions:
        logger.error("No valid position data loaded")
        return None
        
    # Stack positions into single array
    param_values = np.vstack(all_positions)
    
    logger.info(f"Loaded parameter values shape: {param_values.shape}")
    logger.info(f"Skipped first {skip_iterations} iterations, using {len(all_positions)} iterations")
    
    return param_values

def _kde_contour_levels(ZZ, x_grid, y_grid, fractions=(0.68, 0.95)):
    """Return Z thresholds enclosing the given probability mass fractions."""
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    z_sorted = np.sort(ZZ.ravel())[::-1]
    cumulative = np.cumsum(z_sorted) * dx * dy
    cumulative /= cumulative[-1]
    levels = []
    for f in sorted(fractions, reverse=True):
        idx = np.searchsorted(cumulative, f)
        idx = min(idx, len(z_sorted) - 1)
        levels.append(z_sorted[idx])
    return sorted(set(levels))


def plot_pso_corner(pos, fx, space, output_dir):
    """
    Traditional corner plot of all PSO evaluations.
    Off-diagonal: weighted 2-D KDE contours at 68% and 95% credible regions.
    Diagonal: weighted 1-D KDE curve with best-fit marked.
    Weights: exp(-Δchi²/2) so better-fitting particles contribute more.
    """
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    S, D, L = pos.shape

    # Flatten all (particle, iteration) evaluations → (N, D)
    all_pos = pos.transpose(2, 0, 1).reshape(S * L, D)
    all_fx  = fx.T.reshape(S * L)

    # Drop penalty / non-finite scores
    good = np.isfinite(all_fx) & (all_fx < 1e9)
    all_pos = all_pos[good]
    all_fx  = all_fx[good]

    if len(all_fx) == 0:
        logger.warning("No valid evaluations for corner plot — skipping")
        return

    labels   = [str(lbl) for lbl in space['plot_label']]
    best_idx = np.argmin(all_fx)

    # Bayesian-style weights: exp(-Δchi²/2), normalised
    weights = np.exp(-0.5 * (all_fx - all_fx.min()))
    weights /= weights.sum()

    fig, axes = plt.subplots(D, D, figsize=(2.8 * D, 2.8 * D))
    if D == 1:
        axes = np.array([[axes]])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    FILL_95  = '#c6dbef'   # light blue  — outer 95% region
    FILL_68  = '#6baed6'   # mid blue    — inner 68% region
    LINE_95  = '#2171b5'
    LINE_68  = '#084594'

    for row in range(D):
        for col in range(D):
            ax = axes[row, col]

            if col > row:
                ax.set_visible(False)
                continue

            x_data = all_pos[:, col]

            if row == col:
                # Diagonal: weighted 1-D KDE
                try:
                    kde = gaussian_kde(x_data, weights=weights)
                    x_grid = np.linspace(x_data.min(), x_data.max(), 300)
                    dens = kde(x_grid)
                    ax.fill_between(x_grid, dens, alpha=0.35, color=FILL_68)
                    ax.plot(x_grid, dens, color=LINE_68, lw=1.5)
                except Exception:
                    ax.hist(x_data, bins=25, weights=weights * len(weights),
                            color=FILL_68, edgecolor='white', lw=0.5, density=True)
                ax.axvline(all_pos[best_idx, col], color='red', lw=1.5, ls='--')
                ax.set_yticks([])

            else:
                # Off-diagonal: 2-D weighted KDE contours
                y_data = all_pos[:, row]
                try:
                    kde2d = gaussian_kde(np.vstack([x_data, y_data]), weights=weights)
                    nx, ny = 80, 80
                    x_grid = np.linspace(x_data.min(), x_data.max(), nx)
                    y_grid = np.linspace(y_data.min(), y_data.max(), ny)
                    XX, YY = np.meshgrid(x_grid, y_grid)
                    ZZ = kde2d(np.vstack([XX.ravel(), YY.ravel()])).reshape(ny, nx)

                    levels = _kde_contour_levels(ZZ, x_grid, y_grid)

                    if len(levels) >= 2:
                        ax.contourf(XX, YY, ZZ,
                                    levels=[levels[0], ZZ.max()],
                                    colors=[FILL_95], alpha=0.7)
                        ax.contourf(XX, YY, ZZ,
                                    levels=[levels[1], ZZ.max()],
                                    colors=[FILL_68], alpha=0.8)
                        ax.contour(XX, YY, ZZ, levels=levels,
                                   colors=[LINE_95, LINE_68], linewidths=1.0)
                    else:
                        ax.contourf(XX, YY, ZZ,
                                    levels=[levels[0], ZZ.max()],
                                    colors=[FILL_68], alpha=0.8)
                        ax.contour(XX, YY, ZZ, levels=levels,
                                   colors=[LINE_68], linewidths=1.0)

                    ax.scatter(all_pos[best_idx, col], all_pos[best_idx, row],
                               marker='*', s=140, c='red', zorder=10,
                               edgecolors='black', linewidths=0.8)
                except Exception as e:
                    logger.debug(f"KDE failed for ({col},{row}): {e} — using scatter fallback")
                    ax.scatter(x_data, y_data, c=FILL_68, s=4, alpha=0.4, linewidths=0)
                    ax.scatter(all_pos[best_idx, col], all_pos[best_idx, row],
                               marker='*', s=140, c='red', zorder=10)

            # Labels on outer edges only
            if row == D - 1:
                ax.set_xlabel(labels[col], fontsize=10)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(labels[row], fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    # Legend
    legend_elements = [
        Patch(facecolor=FILL_95, alpha=0.7, edgecolor=LINE_95, label='95%'),
        Patch(facecolor=FILL_68, alpha=0.8, edgecolor=LINE_68, label='68%'),
        Line2D([0], [0], color='red', lw=1.5, ls='--', label='Best fit'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=8, framealpha=0.9)

    # Best-fit annotation
    best_str = '  '.join(f'{labels[k]}={all_pos[best_idx, k]:.3g}' for k in range(D))
    fig.text(0.12, 0.97,
             f'Best (\u2605): {best_str}\n'
             f'$\\chi^2_\\mathrm{{red}}$ = {all_fx[best_idx]:.3f}   ({len(all_fx)} evaluations)',
             fontsize=8, va='top', family='monospace')

    outfile = os.path.join(output_dir, 'pso_corner.png')
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Corner plot saved to {outfile}")


def processing(tracks_dir, space_file, output_dir, config_opts, space=None):
    logger.info("Diagnostics:")

    # Load particle data
    space, pos, fx = load_space_and_particles(tracks_dir, space_file)
    S, D, L = pos.shape
    logger.debug('S=%d particles, D=%d dimensions, L=%d iterations' % (S, D, L))

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    num_particles = S
    num_iterations = L

    # Corner plot — always produced regardless of which constraints were active
    try:
        plot_pso_corner(pos, fx, space, output_dir)
    except Exception as e:
        logger.error(f"Error creating corner plot: {e}")

    # Get SMF and BHMF files mapping with observational data
    # PSO health: convergence, per-constraint breakdown, parameter collapse and
    # bound contact.  Produced first because it is what tells you whether the
    # run is worth reading the constraint plots for.
    stat_test = getattr(config_opts, 'stat_test', 'chi2')
    set_plot_cosmology(getattr(config_opts, 'h0', None),
                       getattr(config_opts, 'Omega0', None))
    try:
        health = create_pso_health_figure(
            space, pos, fx, tracks_dir, output_dir, num_particles,
            os.path.join(output_dir, 'pso_health.png'), stat_test=stat_test)
        logger.info('  pso_health.png        convergence, constraint breakdown, '
                    'parameter collapse, bound contact')
    except Exception as e:
        logger.warning('  could not build pso_health.png: %s', e)

    try:
        if plot_degeneracy(space, pos, fx,
                           os.path.join(output_dir, 'pso_degeneracy.png')):
            logger.info('  pso_degeneracy.png    parameter pairs coloured by fit quality')
    except Exception as e:
        logger.warning('  could not build pso_degeneracy.png: %s', e)

    try:
        if plot_residuals(output_dir, num_particles,
                          os.path.join(output_dir, 'pso_residuals.png'),
                          stat_test=stat_test):
            logger.info('  pso_residuals.png     normalised residuals per constraint')
    except Exception as e:
        logger.warning('  could not build pso_residuals.png: %s', e)

    logger.debug("Looking for dump files...")
    smf_files = get_smf_files_map(config_opts)
    logger.debug(f"Found {len(smf_files)} SMF files to process")

    smf_red_files = get_smf_red_files_map(config_opts)
    logger.debug(f"Found {len(smf_red_files)} SMF_Red files to process")

    smf_blue_files = get_smf_blue_files_map(config_opts)
    logger.debug(f"Found {len(smf_blue_files)} SMF_Blue files to process")

    bhmf_files = get_bhmf_files_map(config_opts)
    logger.debug(f"Found {len(bhmf_files)} BHMF files to process")

    bhbm_files = get_bhbm_files_map(config_opts)
    logger.debug(f"Found {len(bhbm_files)} BHBM files to process")

    himf_files = get_himf_files_map(config_opts)
    logger.debug(f"Found {len(himf_files)} HIMF files to process")

    h2mf_files = get_h2mf_files_map(config_opts)
    logger.debug(f"Found {len(h2mf_files)} H2MF files to process")

    mzr_files = get_mzr_files_map(config_opts)
    logger.debug(f"Found {len(mzr_files)} MZR files to process")

    shmr_files = get_shmr_files_map(config_opts)
    logger.debug(f"Found {len(shmr_files)} SHMR files to process")

    smd_files = get_smd_files_map(config_opts)
    logger.debug(f"Found {len(smd_files)} SMD files to process")

    csfrdh_files = get_csfrdh_files_map(config_opts)
    logger.debug(f"Found {len(csfrdh_files)} CSFRDH files to process")

    fics_files = get_fics_files_map(config_opts)
    logger.debug(f"Found {len(fics_files)} FICS files to process")

    mlf_files = get_mlf_files_map(config_opts)
    logger.debug(f"Found {len(mlf_files)} MLF files to process")

    fics_mvir_files = get_fics_mvir_files_map(config_opts)
    logger.debug(f"Found {len(fics_mvir_files)} FICS_Mvir files to process")

    # Process SMF files
    processed_any_smf = False
    for filename, (obs_data, sage_data) in smf_files.items():
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_smf = True
            
            # Create iteration plot
            try:
                logger.debug("Creating iteration plot...")
                fig = smf_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_smf:
        logger.debug("Warning: No SMF files were found to process!")
        logger.debug(f"Expected files in: {output_dir}")
        logger.debug("Expected files: %s", list(smf_files.keys()))

    # Process SMF_Red files
    processed_any_smf_red = False
    for filename, (obs_data, sage_data) in smf_red_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_smf_red = True

            # Create iteration plot
            try:
                logger.debug("Creating iteration plot...")
                fig = create_iteration_plot(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir,
                    plot_type='SMF_Red'
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_smf_red:
        logger.debug("Warning: No SMF_Red files were found to process!")
        logger.debug(f"Expected files in: {output_dir}")
        logger.debug("Expected files: %s", list(smf_red_files.keys()))

    # Process SMF_Blue files
    processed_any_smf_blue = False
    for filename, (obs_data, sage_data) in smf_blue_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_smf_blue = True

            # Create iteration plot
            try:
                logger.debug("Creating iteration plot...")
                fig = create_iteration_plot(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir,
                    plot_type='SMF_Blue'
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_smf_blue:
        logger.debug("Warning: No SMF_Blue files were found to process!")
        logger.debug(f"Expected files in: {output_dir}")
        logger.debug("Expected files: %s", list(smf_blue_files.keys()))

    # Process BHMF files
    processed_any_bhmf = False
    for filename, (obs_data, sage_data) in bhmf_files.items():
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_bhmf = True
            
            # Create iteration plot
            try:
                logger.debug("Creating iteration plot...")
                fig = bhmf_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")
 
    if not processed_any_bhmf:
        logger.debug("Warning: No BHMF files were found to process!")
        logger.debug(f"Expected files in: {output_dir}")
        logger.debug("Expected files: %s", list(bhmf_files.keys())) 

    # Process BHBM files
    processed_any_bhbm = False
    for filename, (obs_data, sage_data) in bhbm_files.items():
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_bhbm = True
            
            # Create iteration plot
            try:
                logger.debug("Creating iteration plot...")
                fig = bhbm_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_bhbm:
        logger.debug("Warning: No BHBM files were found to process!")
        logger.debug(f"Expected files in: {output_dir}")
        logger.debug("Expected files: %s", list(bhbm_files.keys()))

    # Process HIMF files
    processed_any_himf = False
    for filename, (obs_data, sage_data) in himf_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_himf = True

            try:
                logger.debug("Creating iteration plot...")
                fig = himf_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_himf:
        logger.debug("Warning: No HIMF files were found to process!")

    # Process H2MF files
    processed_any_h2mf = False
    for filename, (obs_data, sage_data) in h2mf_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_h2mf = True

            try:
                logger.debug("Creating iteration plot...")
                fig = h2mf_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_h2mf:
        logger.debug("Warning: No H2MF files were found to process!")

    # Process MZR files
    processed_any_mzr = False
    for filename, (obs_data, sage_data) in mzr_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_mzr = True

            try:
                logger.debug("Creating iteration plot...")
                fig = mzr_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_mzr:
        logger.debug("Warning: No MZR files were found to process!")

    # Process SHMR files
    processed_any_shmr = False
    for filename, (obs_data, sage_data) in shmr_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_shmr = True

            try:
                logger.debug("Creating iteration plot...")
                fig = shmr_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_shmr:
        logger.debug("Warning: No SHMR files were found to process!")

    # Process SMD files
    processed_any_smd = False
    for filename, (obs_data, sage_data) in smd_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_smd = True

            try:
                logger.debug("Creating iteration plot...")
                fig = smd_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_smd:
        logger.debug("Warning: No SMD files were found to process!")

    # Process FICS files
    processed_any_fics = False
    for filename, (obs_data, sage_data) in fics_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_fics = True

            try:
                logger.debug("Creating iteration plot...")
                fig = fics_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_fics:
        logger.debug("Warning: No FICS files were found to process!")

    # Process FICS_Mvir files
    processed_any_fics_mvir = False
    for filename, (obs_data, sage_data) in fics_mvir_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_fics_mvir = True

            try:
                logger.debug("Creating iteration plot...")
                fig = fics_mvir_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_fics_mvir:
        logger.debug("Warning: No FICS_Mvir files were found to process!")

    # Process MLF files
    for filename, (obs_data, sage_data) in mlf_files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                fig = mlf_processing_iteration(filepath, num_particles,
                                               num_iterations, obs_data,
                                               sage_data, tracks_dir)
                outfile = os.path.join(output_dir,
                                       f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating MLF iteration plot: {e}")

    # Process CSFRDH files
    processed_any_csfrdh = False
    for filename, (obs_data, sage_data) in csfrdh_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.debug(f"Processing {filename}...")
            processed_any_csfrdh = True

            try:
                logger.debug("Creating iteration plot...")
                fig = csfrdh_processing_iteration(
                    filepath,
                    num_particles,
                    num_iterations,
                    obs_data,
                    sage_data,
                    tracks_dir
                )
                outfile = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_all.png')
                fig.savefig(outfile, dpi=300)
                logger.info("  %s", os.path.basename(outfile))
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating iteration plot: {str(e)}")

    if not processed_any_csfrdh:
        logger.debug("Warning: No CSFRDH files were found to process!")

    # Load parameter values
    logger.debug("Processing parameter evolution...")
    param_names = ['SFR efficiency', 'Reheating epsilon', 'Ejection efficiency', 'Reincorporation efficiency',
                   'Radio Mode', 'Quasar Mode', 'Black Hole growth', 'Baryon Fraction']
    
    # Map the file numbers to actual redshifts
    redshift_map = {
        0.0: '0',
        0.2: '02',
        0.5: '05',
        0.8: '08',
        1.0: '10',
        1.1: '11',
        1.5: '15',
        2.0: '20',
        2.4: '24',
        3.1: '31',
        3.6: '36',
        4.6: '46',
        5.7: '57',
        6.3: '63',
        7.7: '77',
        8.5: '85',
        10.4: '104'
    }
    
    processed_redshifts = []
    for z, z_str in redshift_map.items():
        param_file = os.path.join(output_dir, f'params_z{z_str}.csv')
        if os.path.exists(param_file):
            processed_redshifts.append(z)
            logger.info(f"Found parameter file for z={z}")
        else:
            logger.debug(f"No parameter file found for z={z}")

    if not processed_redshifts:
        logger.debug("No per-redshift parameter files found (only produced by the multi-redshift workflow); skipping that plot.")
        return

    processed_redshifts.sort()
    logger.info(f"Found parameter files for redshifts: {processed_redshifts}")
    
    particle_data, best_params, best_scores = load_all_params(output_dir, param_names, processed_redshifts)
    
    if particle_data:

        track_files = sorted(glob.glob(os.path.join(tracks_dir, 'track_*_pos.npy')))
        if not track_files:
            logger.error("Missing track files")
            return
        
        results, scores = analyze_and_plot(
            tracks_dir=tracks_dir,
            space_file=space_file,
            output_dir=output_dir,
            csv_output_path=output_dir
        )

        # Create .gif if possible
        try:
            gif_path = os.path.join(output_dir, 'parameter_evolution.gif')
            create_swarm_gif_from_tracks(tracks_dir, space_file, gif_path)
            logger.info(f"Parameter evolution GIF saved to {gif_path}")
        except Exception as e:
            logger.error(f"Error creating GIF animation: {str(e)}")

        # After creating all individual plots
        logger.info("Creating combined constraint grid plots...")
        try:
            create_combined_constraint_grids(output_dir=output_dir, png_dir=output_dir)
            logger.info("Successfully created combined constraint grid plots")
        except Exception as e:
            logger.error(f"Error creating combined constraint grid plots: {str(e)}")
        
        logger.info("All plots have been saved to: %s", output_dir)
    else:
        logger.debug("No per-redshift parameter files for visualisation; skipping.")
    print(particle_data, best_params, best_scores)
    return particle_data, best_params, best_scores

def create_swarm_gif_from_tracks(tracks_dir, space_file, output_path):
    """
    Loads positions and fitness from tracks files and creates swarm GIF.
    """
    space, pos, fx = load_space_and_particles(tracks_dir, space_file)
    
    # Input pos is (S, D, L) -> (Particles, Dimensions, Iterations)
    # We want (L, S, D) -> (Iterations, Particles, Dimensions) for the animation
    pos = pos.transpose(2, 0, 1)
    
    # fx is (S, L) -> (Particles, Iterations)
    # We want (L, S) -> (Iterations, Particles) to match the frames
    fx = fx.T

    # FIX: Check dtype.names to avoid "structured array" comparison error
    if space.dtype.names and 'plot_label' in space.dtype.names:
        param_names = list(space['plot_label'])
    else:
        param_names = [f'param{i}' for i in range(pos.shape[2])]

    # Pass the corrected data to the generator
    # pos.shape[1] is now 'Particles' (S)
    create_parameter_gif(pos, param_names, output_path, scores=fx, particles=pos.shape[1])
def main(tracks_dir, space_file, output_dir, config_opts, space=None):
    
    processing(tracks_dir, space_file, output_dir, config_opts, space=space)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--space-file',
        help='File with the search space specification, defaults to space.txt',
        default='space.txt')
    parser.add_argument('-o', '--output-dir',
        help='Output directory for plots',
        required=True)
    parser.add_argument('tracks_dir',
        help='Directory containing PSO tracks')
    opts = parser.parse_args()

    # Create minimal config_opts object for standalone usage
    class ConfigOpts:
        def __init__(self):
            self.h0 = 0.677400
            self.Omega0 = 0.3089
            self.username = None
    
    config_opts = ConfigOpts()

    # Load space file for parameter bounds
    space_obj = analysis.load_space(opts.space_file)
    
    main(opts.tracks_dir, opts.space_file, opts.output_dir, config_opts, space=space_obj)

    main(opts.tracks_dir, opts.space_file, opts.output_dir)

# =============================================================================
# PSO health and constraint diagnostics
#
# The plots above show one constraint's curves per figure.  These show the
# optimisation itself: whether it converged, which constraint is driving the
# objective, which parameters the data actually constrains, and where the model
# is systematically off.  Every score here is recomputed from the dump files
# with a chosen stat test, so chi2 and student-t are interchangeable without
# rerunning anything.
# =============================================================================

def read_dump_blocks(filepath):
    """Parse a <Constraint>_dump.txt into per-evaluation blocks.

    Returns a list of (x_obs, y_obs, y_mod, err) tuples, one per particle
    evaluation, in the order written.  Dumps written before err was added have
    three columns; those come back with err = None.
    """
    blocks, cur = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('# New Data Block'):
                if cur:
                    blocks.append(cur)
                cur = []
            elif line:
                try:
                    cur.append([float(v) for v in line.split('\t')])
                except ValueError:
                    continue
    if cur:
        blocks.append(cur)

    out = []
    for b in blocks:
        arr = np.array(b, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        err = arr[:, 3] if arr.shape[1] > 3 else None
        out.append((arr[:, 0], arr[:, 1], arr[:, 2], err))
    return out


# One-shot flag so the count-scale notice is said once per run, not once per
# constraint per figure.
_COUNT_SCALE_NOTICE = []


def score_dump_blocks(blocks, stat_test='chi2', reduced=True):
    """Score each evaluation block with the named stat test.

    stat_test is a key of analysis.stat_tests ('chi2' or 'student-t'), so the
    same dumps can be re-scored either way after the fact.  Blocks with no
    recorded err fall back to the scatter of the observations, which is what
    studentT does internally for zero errors.
    """
    from src import analysis
    fn = analysis.stat_tests[stat_test]

    # A counting statistic needs the constraint's dm * V, which the dump does
    # not record (it stores log10(phi), not counts).  Score with chi2 instead:
    # these scores only rank particles so the plots can pick the best one, and
    # chi2 ranks them faithfully enough for that.  Noted rather than silent,
    # because the number shown on a residual panel is then a chi2.
    if getattr(fn, 'needs_count_scale', False):
        if not _COUNT_SCALE_NOTICE:
            _COUNT_SCALE_NOTICE.append(True)
            logger.info('  (dump files record log10(phi), not counts, so the '
                        'diagnostic panels rank particles by chi2 rather '
                        'than %s)', stat_test)
        fn = analysis.stat_tests['chi2']
        stat_test = 'chi2'

    scores = []
    for x_obs, y_obs, y_mod, err in blocks:
        e = err if err is not None else np.full_like(y_obs, max(np.std(y_obs), 1e-3))
        try:
            s = float(fn(y_obs, y_mod, e))
        except Exception:
            s = np.nan
        scores.append(s / len(y_obs) if reduced and len(y_obs) else s)
    return np.array(scores)


def _dump_files(output_dir):
    """Constraint name -> dump path for every dump in an output directory."""
    found = {}
    for path in sorted(glob.glob(os.path.join(output_dir, '*_dump.txt'))):
        name = os.path.basename(path)[:-len('_dump.txt')]
        found[name] = path
    return found


def _reshape_by_iteration(values, num_particles):
    """(n_iterations, num_particles) view of a flat per-evaluation series."""
    values = np.asarray(values, dtype=float)
    n_iter = len(values) // num_particles
    if n_iter < 1:
        return values.reshape(1, -1)
    return values[:n_iter * num_particles].reshape(n_iter, num_particles)


# --- 1. convergence trace ----------------------------------------------------

def plot_convergence(tracks_dir, ax=None):
    """Best-so-far and per-iteration best/median/worst objective vs iteration.

    Answers the first question about any optimisation: did it converge, or did
    it run out of iterations still descending?  A best-so-far curve that is
    still dropping at the last iteration means the run was cut short.
    """
    files = sorted(glob.glob(os.path.join(tracks_dir, 'track_*_fx.npy')))
    if not files:
        return None
    fx = [np.load(f) for f in files]

    it = np.arange(len(fx))
    best = np.array([np.nanmin(v) for v in fx])
    med = np.array([np.nanmedian(v) for v in fx])
    worst = np.array([np.nanmax(v) for v in fx])
    best_so_far = np.minimum.accumulate(best)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(it, best, worst, color='steelblue', alpha=0.18,
                    label='swarm range')
    ax.plot(it, med, color='steelblue', lw=1.5, label='swarm median')
    ax.plot(it, best, color='darkorange', lw=1.5, label='iteration best')
    ax.plot(it, best_so_far, color='crimson', lw=2.5, label='best so far')
    ax.set_xlabel('iteration')
    ax.set_ylabel('objective')
    ax.set_yscale('log' if np.all(best_so_far > 0) else 'linear')
    ax.grid(alpha=0.25)
    leg = ax.legend(fontsize=9); leg.draw_frame(False)

    # Flag a run that never flattened
    if len(best_so_far) > 3:
        tail = best_so_far[-max(3, len(best_so_far) // 4):]
        if tail[0] > 0 and (tail[0] - tail[-1]) / tail[0] > 0.02:
            ax.set_title('still improving at the last iteration -- '
                         'consider more iterations', fontsize=10, color='firebrick')
        else:
            ax.set_title('converged', fontsize=10, color='darkgreen')
    return ax.figure if own_fig else ax


# --- 2. per-constraint breakdown ---------------------------------------------

def plot_constraint_breakdown(output_dir, num_particles, stat_test='chi2', ax=None):
    """Each constraint's reduced score vs iteration, recomputed from the dumps.

    Shows which constraint is actually driving the objective and which is
    already satisfied -- the thing a single total score hides.  Because the
    dumps carry err, the same run can be re-scored with student-t by passing
    stat_test='student-t'.
    """
    dumps = _dump_files(output_dir)
    if not dumps:
        return None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    colours = plt.cm.tab10(np.linspace(0, 1, max(len(dumps), 2)))
    plotted = 0
    for (name, path), colour in zip(sorted(dumps.items()), colours):
        blocks = read_dump_blocks(path)
        if not blocks:
            continue
        scores = score_dump_blocks(blocks, stat_test=stat_test)
        per_iter = _reshape_by_iteration(scores, num_particles)
        best = np.nanmin(per_iter, axis=1)
        ax.plot(np.arange(len(best)), best, lw=2, color=colour,
                label='%s (n=%d pts)' % (name, len(blocks[0][0])))
        plotted += 1

    if not plotted:
        if own_fig:
            plt.close(fig)
        return None

    ax.set_xlabel('iteration')
    ax.set_ylabel('reduced %s of the iteration-best particle'
                  % ('chi$^2$' if stat_test == 'chi2' else 'student-t'))
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    leg = ax.legend(fontsize=8, ncol=2); leg.draw_frame(False)
    ax.set_title('which constraint is driving the fit', fontsize=10)
    return ax.figure if own_fig else ax


# --- 3. parameter traces / swarm collapse ------------------------------------

def plot_parameter_traces(space, pos, fx, ax=None):
    """Swarm spread per parameter vs iteration, normalised to its bounds.

    A parameter whose swarm collapses to a narrow band is constrained by the
    data; one still spanning the full range at the last iteration is not, and
    its best-fit value should not be quoted as a measurement.
    """
    labels = [str(l) for l in space['plot_label']]
    is_log = space['is_log'].astype(bool)
    lb = np.where(is_log, np.log10(space['lb']), space['lb'])
    ub = np.where(is_log, np.log10(space['ub']), space['ub'])

    # pos is (D, S, L): dimensions, particles, iterations
    D, S, L = pos.shape
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    colours = plt.cm.viridis(np.linspace(0, 0.9, D))
    for d in range(D):
        vals = pos[d]
        if is_log[d]:
            with np.errstate(divide='ignore'):
                vals = np.log10(np.where(vals > 0, vals, np.nan))
        frac = (vals - lb[d]) / max(ub[d] - lb[d], 1e-30)
        spread = np.nanpercentile(frac, 84, axis=0) - np.nanpercentile(frac, 16, axis=0)
        ax.plot(np.arange(L), spread, lw=2, color=colours[d], label=labels[d])

    ax.axhline(0.1, ls='dotted', c='0.4', lw=1)
    ax.text(0.01, 0.105, 'collapsed (< 10% of range)', fontsize=8, color='0.4',
            transform=ax.get_yaxis_transform())
    ax.set_xlabel('iteration')
    ax.set_ylabel('swarm 16-84 spread / bound range')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    leg = ax.legend(fontsize=8, ncol=2); leg.draw_frame(False)
    ax.set_title('which parameters the data constrains', fontsize=10)
    return ax.figure if own_fig else ax


# --- 4. bound contact --------------------------------------------------------

def plot_bound_contact(space, pos, ax=None):
    """Fraction of particle-iterations spent within 1% of each bound.

    A best fit pinned against a bound is not a fit, it is a bound: the optimum
    lies outside the search space and the range needs widening.  Persistent
    contact also wastes evaluations, since PSO clamps position but not velocity.
    """
    labels = [str(l) for l in space['plot_label']]
    is_log = space['is_log'].astype(bool)
    lb = np.where(is_log, np.log10(space['lb']), space['lb'])
    ub = np.where(is_log, np.log10(space['ub']), space['ub'])

    D = pos.shape[0]
    at_lb, at_ub = np.zeros(D), np.zeros(D)
    for d in range(D):
        vals = pos[d].ravel()
        if is_log[d]:
            with np.errstate(divide='ignore'):
                vals = np.log10(np.where(vals > 0, vals, np.nan))
        rng = max(ub[d] - lb[d], 1e-30)
        tol = 0.01 * rng
        finite = np.isfinite(vals)
        n = max(np.count_nonzero(finite), 1)
        at_lb[d] = np.count_nonzero(vals[finite] <= lb[d] + tol) / n
        at_ub[d] = np.count_nonzero(vals[finite] >= ub[d] - tol) / n

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(D)
    ax.barh(y - 0.2, at_lb, height=0.38, color='steelblue', label='at lower bound')
    ax.barh(y + 0.2, at_ub, height=0.38, color='indianred', label='at upper bound')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('fraction of particle-iterations within 1% of the bound')
    ax.axvline(0.2, ls='dotted', c='0.4', lw=1)
    ax.grid(alpha=0.25, axis='x')
    leg = ax.legend(fontsize=8); leg.draw_frame(False)
    flagged = [labels[d] for d in range(D) if max(at_lb[d], at_ub[d]) > 0.2]
    ax.set_title('bounds to reconsider: %s' % (', '.join(flagged) if flagged else 'none'),
                 fontsize=10, color='firebrick' if flagged else 'darkgreen')
    return ax.figure if own_fig else ax


# --- 5. degeneracy, coloured by score ----------------------------------------

def plot_degeneracy(space, pos, fx, output_path, max_params=6):
    """Parameter-pair scatter coloured by objective, to expose degeneracies.

    The existing corner plot shows where particles went, which is density; this
    shows how good they were.  A degenerate direction appears as an elongated
    low-score valley -- two parameters trading off, only their combination
    constrained.
    """
    labels = [str(l) for l in space['plot_label']]
    D = min(pos.shape[0], max_params)
    if D < 2:
        return None

    flat = [pos[d].ravel() for d in range(D)]
    score = np.asarray(fx).ravel()
    n = min(len(score), min(len(v) for v in flat))
    flat = [v[:n] for v in flat]
    score = score[:n]
    ok = np.isfinite(score) & (score > 0)
    if np.count_nonzero(ok) < 10:
        return None
    c = np.log10(score[ok])

    fig, axes = plt.subplots(D - 1, D - 1, figsize=(2.5 * (D - 1), 2.5 * (D - 1)),
                             squeeze=False)
    sc = None
    for i in range(D - 1):
        for j in range(D - 1):
            ax = axes[i][j]
            if j > i:
                ax.axis('off')
                continue
            xi, yi = j, i + 1
            sc = ax.scatter(flat[xi][ok], flat[yi][ok], c=c, s=6,
                            cmap='viridis_r', alpha=0.65, linewidths=0)
            # mark the best particle
            b = np.argmin(score[ok])
            ax.plot(flat[xi][ok][b], flat[yi][ok][b], marker='*', ms=14,
                    color='crimson', mec='k', mew=0.5)
            if i == D - 2:
                ax.set_xlabel(labels[xi], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[yi], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label(r'$\log_{10}$ objective  (dark = better)', fontsize=9)
    fig.suptitle('parameter degeneracies, coloured by fit quality; star = best',
                 fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


# --- 6. residual panels ------------------------------------------------------

def plot_residuals(output_dir, num_particles, output_path, stat_test='chi2'):
    """Normalised residuals of the best particle, one panel per constraint.

    (model - obs)/sigma against the constraint's own x axis.  A systematic
    trend -- all high at one end, all low at the other -- is a shape error the
    optimiser cannot fix by rescaling, and is far easier to see here than in a
    log-log overlay of the curves themselves.
    """
    dumps = _dump_files(output_dir)
    usable = {}
    for name, path in sorted(dumps.items()):
        blocks = read_dump_blocks(path)
        if blocks and blocks[0][3] is not None:
            usable[name] = blocks
    if not usable:
        return None

    n = len(usable)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             squeeze=False)

    from src import analysis as _a
    if getattr(_a.stat_tests.get(stat_test), 'needs_count_scale', False):
        stat_test = 'chi2'      # see score_dump_blocks

    for k, (name, blocks) in enumerate(usable.items()):
        ax = axes[k // ncol][k % ncol]
        scores = score_dump_blocks(blocks, stat_test=stat_test)
        best = int(np.nanargmin(scores))
        x_obs, y_obs, y_mod, err = blocks[best]
        resid = (y_mod - y_obs) / np.maximum(err, 1e-12)

        ax.axhline(0, c='k', lw=1)
        ax.axhspan(-1, 1, color='0.85', zorder=0)
        ax.axhspan(-2, 2, color='0.93', zorder=-1)
        ax.plot(x_obs, resid, 'o-', ms=4, lw=1.2, color='crimson')
        ax.set_title('%s   (reduced %s = %.2f)'
                     % (name, stat_test, scores[best]), fontsize=9)
        ax.set_ylabel(r'$(\mathrm{model}-\mathrm{obs})/\sigma$', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        lim = max(3.0, float(np.nanmax(np.abs(resid))) * 1.1)
        ax.set_ylim(-lim, lim)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')

    fig.suptitle('normalised residuals of the best-fit particle '
                 '(grey bands: 1 and 2 sigma)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


# --- combined PSO health figure ---------------------------------------------

def create_pso_health_figure(space, pos, fx, tracks_dir, output_dir,
                             num_particles, output_path, stat_test='chi2'):
    """Convergence, per-constraint breakdown, parameter collapse and bound
    contact in one figure -- the four things worth checking after every run."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    try:
        plot_convergence(tracks_dir, ax=axes[0][0])
    except Exception as e:
        axes[0][0].text(0.5, 0.5, 'convergence: %s' % e, ha='center', fontsize=8)
    try:
        if plot_constraint_breakdown(output_dir, num_particles,
                                     stat_test=stat_test, ax=axes[0][1]) is None:
            axes[0][1].text(0.5, 0.5, 'no constraint dumps found',
                            ha='center', fontsize=9)
            axes[0][1].axis('off')
    except Exception as e:
        axes[0][1].text(0.5, 0.5, 'breakdown: %s' % e, ha='center', fontsize=8)
    try:
        plot_parameter_traces(space, pos, fx, ax=axes[1][0])
    except Exception as e:
        axes[1][0].text(0.5, 0.5, 'traces: %s' % e, ha='center', fontsize=8)
    try:
        plot_bound_contact(space, pos, ax=axes[1][1])
    except Exception as e:
        axes[1][1].text(0.5, 0.5, 'bounds: %s' % e, ha='center', fontsize=8)

    fig.suptitle('PSO health  (scored with %s)' % stat_test, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path
