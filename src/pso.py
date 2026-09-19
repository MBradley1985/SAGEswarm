#!/bin/bash

# Adapted from pyswarm verison 0.7 for optimising the shark SAM on a HPC system
#
# https://github.com/tisimst/pyswarm/tree/master/pyswarm   (original pyswarms code)

from functools import partial
import numpy as np # type: ignore
import csv
import os
import logging
import sys
import time
import threading

def _write_results_to_csv(csv_path, iteration_history, final_positions, particle_fitness, best_position, best_fitness, max_retries=3):
    """
    Write PSO results to CSV, including full iteration history.
    
    Parameters:
    -----------
    csv_path : str
        Path to save the CSV file
    iteration_history : list of tuples
        List containing (iteration, positions, fitness) for each iteration
    final_positions : array
        Final positions of all particles
    particle_fitness : array
        Final fitness values of all particles
    best_position : array
        Best position found
    best_fitness : float
        Best fitness value found
    max_retries : int
        Maximum number of retry attempts for writing CSV
    """
    
    # Logging setup
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s',
                       filename='pso_csv_log.txt')
    
    # Thread-safe lock
    csv_write_lock = threading.Lock()
    
    with csv_write_lock:
        for attempt in range(max_retries):
            try:
                # Check directory permissions
                directory = os.path.dirname(csv_path)
                if not os.access(directory, os.W_OK):
                    logging.error(f"No write permissions for directory: {directory}")
                    return
                
                # Ensure directory exists
                os.makedirs(directory, exist_ok=True)
                
                # Open file in write mode
                with open(csv_path, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter='\t')
                    
                    # Write iteration history
                    for it, positions, fitness in iteration_history:
                        for particle_idx in range(len(positions)):
                            row = list(positions[particle_idx])
                            row.append(fitness[particle_idx])
                            csvwriter.writerow(row)
                    
                    # Add blank line before best position
                    csvwriter.writerow([])
                    # Write final best position and score format at the end (last 2 rows)
                    csvwriter.writerow(list(best_position))  # Second to last row: best position
                    csvwriter.writerow([best_fitness])       # Last row: best fitness score
                
                logging.debug(f"CSV successfully written to {csv_path}")
                return
                
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
                time.sleep(0.1)  # Small delay between attempts
        
        logging.error(f"Failed to write CSV after {max_retries} attempts")

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _indexed_obj_wrapper(func, args, kwargs, ix):
    i, x = ix
    return i, func(x, *args, **kwargs)

def _to_display(x, is_log, is_int=None, lb=None, ub=None):
    """Back-transform particle positions to physical space for output.

    Log-sampled dimensions are raised back out of log10; integer switches are
    rounded and clamped exactly as execution._to_physical does, so the tracks,
    the CSV and the logged best-fit report the switch level that SAGE actually
    ran rather than the particle's continuous position.
    """
    has_log = is_log is not None and np.any(is_log)
    has_int = is_int is not None and np.any(is_int)
    if not has_log and not has_int:
        return x
    out = np.array(x, dtype=float)
    if has_log:
        out[..., is_log] = 10.0 ** out[..., is_log]
    if has_int:
        rounded = np.rint(out[..., is_int])
        if lb is not None:
            rounded = np.maximum(rounded, np.rint(np.asarray(lb)[is_int]))
        if ub is not None:
            rounded = np.minimum(rounded, np.rint(np.asarray(ub)[is_int]))
        out[..., is_int] = rounded
    return out

def _fmt_params(values):
    """Compact parameter vector for a single log line."""
    out = []
    for v in np.atleast_1d(np.asarray(values, dtype=float)):
        out.append('%d' % v if float(v).is_integer() else '%.4g' % v)
    return '[' + ', '.join(out) + ']'


def _report_iteration(label, x_phys, fx, fg):
    """Report what this iteration's swarm actually scored.

    The running best changes only when it improves, so a line repeating it every
    iteration carries no information.  What is informative is the spread of the
    current swarm: if best/median/worst are still moving the search is live, and
    if they have collapsed together the swarm has converged (or stalled).
    Improvements to the global best are announced separately by the caller.
    """
    fx = np.asarray(fx, dtype=float).ravel()
    finite = np.isfinite(fx)
    if not np.any(finite):
        print('{:}  all {:d} particles failed to evaluate'.format(label, fx.size))
        return

    masked = np.where(finite, fx, np.inf)
    i_best = int(np.argmin(masked))
    vals = fx[finite]
    failed = int(np.count_nonzero(~finite))

    msg = ('{:}  best={:<11.6g} median={:<11.6g} worst={:<11.6g}'
           .format(label, fx[i_best], float(np.median(vals)),
                   float(np.max(vals))))
    msg += (' swarm best={:<11.6g}'.format(fg) if np.isfinite(fg)
            else ' ' * 24)
    msg += ' at ' + _fmt_params(x_phys[i_best])
    if failed:
        msg += '   ({:d} failed)'.format(failed)
    print(msg)


def _print_progress(done, total, iteration):
    bar_width = 30
    filled = int(bar_width * done / total) if total > 0 else 0
    bar = '\u2588' * filled + '\u2591' * (bar_width - filled)
    label = 'Init' if iteration == 0 else f'Iter {iteration}'
    sys.stdout.write(f'\r  {label}  [{bar}]  {done}/{total} particles complete  ')
    sys.stdout.flush()

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))
    
def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=True, processes=1,
        particle_output=False, dumpfile_prefix=None, csv_output_path=None,
        random_seed=None, is_log=None, max_stagnation=15,
        is_int=None, int_lb=None, int_ub=None):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor / inertia weight (Default: 0.729)
        Constriction coefficient from Clerc & Kennedy (2002).
        Standard range: 0.7-0.9. Controls exploration vs exploitation balance.
    phip : scalar
        Cognitive parameter - scaling factor to search away from the particle's 
        best known position (Default: 1.49445 ≈ 2.05 × 0.729). 
        Standard range: 1.5-2.5. Higher values increase personal best attraction.
    phig : scalar
        Social parameter - scaling factor to search away from the swarm's 
        best known position (Default: 1.49445 ≈ 2.05 × 0.729). 
        Standard range: 1.5-2.5. Higher values increase global best attraction.
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints. If processes = 0 then all particles are given to a single
        handling function to deal with them all at once (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
    csv_output_path : str, optional
        Path to save CSV file with best positions and their objective values
        (Default: None)
    random_seed : int, optional
        Random seed for reproducibility (Default: None)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p
   
    """
   
    # Set random seed if provided for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'

    def _display(pos):
        """Particle position(s) as the physical values SAGE was given."""
        return _to_display(pos, is_log, is_int=is_int, lb=int_lb, ub=int_ub)
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)

    # Initialize dumping function if required
    if dumpfile_prefix:
        def dump(i, x, fx):
            np.save(dumpfile_prefix % i + "_fx", fx)
            np.save(dumpfile_prefix % i + "_pos", x)
    else:
        dump = lambda *_: None

    # Check for constraint function(s)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                pass  # no inequality constraints; this is the normal path
                       # (the SAGE constraints are the objective, not ieqcons)
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)

    # Initialize the particle swarm
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value

    # Initialize the particle's position
    x = lb + x*(ub - lb)

    # Calculate objective and constraints for each particle
    if processes > 1:
        indexed_obj = partial(_indexed_obj_wrapper, func, args, kwargs)
        _print_progress(0, S, 0)
        buf = {}
        for i, val in mp_pool.imap_unordered(indexed_obj, enumerate(x)):
            buf[i] = val
            _print_progress(len(buf), S, 0)
        sys.stdout.write('\n'); sys.stdout.flush()
        fx = np.array([buf[i] for i in range(S)])
        fs = np.array(mp_pool.map(is_feasible, x))
    elif processes != 0:
        _print_progress(0, S, 0)
        for i in range(S):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])
            _print_progress(i + 1, S, 0)
        sys.stdout.write('\n'); sys.stdout.flush()
    else:
        fx = obj(x)
        fs = is_feasible(x)
    x_phys = _display(x)
    dump(0, x_phys, fx)
    if debug:
        _report_iteration('Init    ', x_phys, fx, np.inf)

    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()

    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)

    # Initialize iteration history
    iteration_history = []
    stagnation = 0

    # Iterate until termination criterion met
    it = 1
    while it < maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        if processes > 1:
            indexed_obj = partial(_indexed_obj_wrapper, func, args, kwargs)
            _print_progress(0, S, it)
            buf = {}
            for i, val in mp_pool.imap_unordered(indexed_obj, enumerate(x)):
                buf[i] = val
                _print_progress(len(buf), S, it)
            sys.stdout.write('\n'); sys.stdout.flush()
            fx = np.array([buf[i] for i in range(S)])
            fs = np.array(mp_pool.map(is_feasible, x))
        elif processes != 0:
            _print_progress(0, S, it)
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
                _print_progress(i + 1, S, it)
            sys.stdout.write('\n'); sys.stdout.flush()
        else:
            fx = obj(x)
            fs = is_feasible(x)

        # Store current iteration data (physical space for output)
        x_phys = _display(x)
        iteration_history.append((it, x_phys.copy(), fx.copy()))
        dump(it, x_phys, fx)

        # Reported here, before the swarm best is updated, so an improvement
        # this iteration is announced underneath the iteration that produced it
        # and 'swarm best' is the value this iteration had to beat.
        if debug:
            _report_iteration('Iter {:<3d}'.format(it), x_phys, fx, fg)

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('  -> New swarm best: {:.6g} at {:}'.format(
                    fp[i_min], _fmt_params(_display(p[i_min, :]))))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                
                # Write CSV before returning
                if csv_output_path:
                    _write_results_to_csv(csv_output_path, iteration_history, _display(p), fp, _display(p_min), fp[i_min])
                p_min_phys = _display(p_min)
                if particle_output:
                    return p_min_phys, fp[i_min], _display(p), fp
                else:
                    return p_min_phys, fp[i_min]

            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'.format(minstep))

                if csv_output_path:
                    _write_results_to_csv(csv_output_path, iteration_history, _display(p), fp, _display(p_min), fp[i_min])
                p_min_phys = _display(p_min)
                if particle_output:
                    return p_min_phys, fp[i_min], _display(p), fp
                else:
                    return p_min_phys, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]
                stagnation = 0
        else:
            stagnation += 1
            if stagnation >= max_stagnation:
                print(f'Stopping search: no improvement for {max_stagnation} iterations')
                if csv_output_path:
                    _write_results_to_csv(csv_output_path, iteration_history, _display(p), fp, _display(g), fg)
                if particle_output:
                    return _display(g), fg, _display(p), fp
                else:
                    return _display(g), fg

        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if csv_output_path:
        _write_results_to_csv(csv_output_path, iteration_history, _display(p), fp, _display(g), fg)

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return _display(g), fg, _display(p), fp
    else:
        return _display(g), fg