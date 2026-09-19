#!/bin/bash

import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import numpy as np # type: ignore
from src import common


logger = logging.getLogger(__name__)

def count_jobs(job_name, username=None):
    """
    Returns how many jobs are currently queued or running.
    If username is provided, checks all jobs for that user.
    Otherwise, checks for jobs matching job_name.
    """
    try:
        if username:
            # Check jobs for specific user
            out, err, code = common.exec_command(["squeue", "-u", username])
        else:
            # Fall back to checking all jobs and filtering by name
            out, err, code = common.exec_command("squeue")
            
        if code:
            raise RuntimeError(f"squeue failed with code {code}: stdout: {out}, stderr: {err}")
        
        # Convert bytes to string if necessary
        if isinstance(out, bytes):
            out = out.decode('utf-8')
            
        if username:
            # For username query, just count lines (minus header)
            return max(0, len(out.splitlines()) - 1)
        else:
            # For job name, filter lines containing the name
            lines_with_jobname = [l for l in out.splitlines() if job_name in l]
            return len(lines_with_jobname)
            
    except OSError:
        raise RuntimeError("Couldn't run squeue, is it installed?")

#This is the original function and is just fine
def _exec_sage(msg, cmdline, cwd=None):
    logger.debug('%s with command line: %s', msg, subprocess.list2cmdline(cmdline))
    out, err, code = common.exec_command(cmdline, cwd=cwd)
    if code != 0:
        logger.error('Error while executing %s (exit code %d):\n' +
                     'stdout:\n%s\nstderr:\n%s', cmdline[0], code,
                     common.b2s(out), common.b2s(err))
        raise RuntimeError('%s error' % cmdline[0])

def _to_physical(value, is_log, is_int=False, lb=None, ub=None):
    """Convert a PSO particle value back to physical space.

    An integer switch is rounded to the nearest level and clamped to its
    declared bounds.  The clamp matters: SAGE validates switches against a
    fixed allowed range and aborts the whole run on an out-of-range value, so a
    particle that drifts past the half-step margin must not be passed through.
    """
    if is_int:
        value = int(np.rint(value))
        if lb is not None:
            value = max(value, int(np.rint(lb)))
        if ub is not None:
            value = min(value, int(np.rint(ub)))
        return value
    return 10.0 ** value if is_log else value


def _format_physical(value, is_int):
    """Render a physical parameter value for a SAGE parameter file.

    Switches must be written without a decimal point: SAGE reads them with
    strtol and rejects the value outright if any character is left over, so
    '2.0' aborts the run where '2' is fine.
    """
    return '%d' % value if is_int else '%.6g' % value


def _particle_physical(space, particle, p):
    """Physical value and formatted string for parameter `p` of a particle."""
    phys = _to_physical(particle[p], space['is_log'][p],
                        is_int=bool(space['is_int'][p]),
                        lb=space['lb'][p], ub=space['ub'][p])
    return phys, _format_physical(phys, bool(space['is_int'][p]))

def _config_tag(line):
    """The parameter tag a SAGE parameter-file line defines, or '' if none.

    The tag is the first whitespace-delimited token.  Commented-out lines begin
    with '%', so their token differs from the bare parameter name and they
    correctly do not count as defining it.
    """
    parts = line.split()
    return parts[0] if parts else ''


def config_tags(config_path):
    """Set of parameter tags defined in a SAGE parameter file."""
    with open(config_path) as f:
        return {tag for tag in (_config_tag(line) for line in f) if tag}


def missing_from_config(space, config_path):
    """Space parameters that do not appear in the SAGE parameter file.

    write_particle_par substitutes values into lines that already exist and
    never appends, so a parameter absent from the base .par is silently
    discarded: every particle then runs identical physics, the objective is
    flat, and the swarm reports the same best fit for the whole run.  Callers
    should refuse to start rather than burn hours on that.

    Uses the same tag rule as the writer, so this cannot disagree with what the
    writer will actually do.
    """
    tags = config_tags(config_path)
    return [name for name in space['name'] if name not in tags]


def write_particle_par(config_path, out_path, output_dir, space, particle):
    """Write the SAGE parameter file for one particle.

    Substitutes this particle's value for every search-space parameter and
    points OutputDir at `output_dir`.

    Replaces three hand-copied loops that shared two defects:

      * They stopped scanning as soon as they had made as many substitutions as
        there are search parameters.  A search parameter positioned before
        OutputDir in the file therefore left OutputDir unrewritten, and every
        particle wrote its output into the same directory.  The same early exit
        could also fire before all parameters were substituted, because the
        counter counted substitutions rather than distinct parameters.

      * They matched the parameter tag by prefix.  SAGE has one parameter whose
        name is a prefix of another (Omega / OmegaLambda), so searching over the
        shorter name would also overwrite the longer one's line.

    This scans the whole file and matches tags exactly.  The file is a hundred
    or so lines, so there is nothing to gain from stopping early.
    """
    with open(config_path) as f:
        lines = f.readlines()

    index_of = {name: p for p, name in enumerate(space['name'])}
    substituted = set()

    for l, line in enumerate(lines):
        tag = _config_tag(line)
        if tag == 'OutputDir':
            lines[l] = f'OutputDir              {output_dir}\n'
        elif tag in index_of:
            _, phys_str = _particle_physical(space, particle, index_of[tag])
            lines[l] = f'{tag}          {phys_str}\n'
            substituted.add(tag)

    with open(out_path, 'w') as f:
        f.writelines(lines)

    # main.py checks this up front, but the config could change under a long
    # run; a silently dropped parameter flattens the objective, so say so.
    dropped = set(index_of) - substituted
    if dropped:
        logger.warning(
            'parameters absent from %s, so these particle values were '
            'discarded: %s', config_path, ', '.join(sorted(dropped)))

    return substituted


def _print_progress(done, total, iteration):
    bar_width = 30
    filled = int(bar_width * done / total) if total > 0 else 0
    bar = '\u2588' * filled + '\u2591' * (bar_width - filled)
    sys.stdout.write(f'\r  Iteration {iteration + 1}  [{bar}]  {done}/{total} particles complete  ')
    sys.stdout.flush()

_warned_no_counts = set()


def _evaluate(constraint, stat_test, modeldir, subvols):
    y_obs, y_mod, err = constraint.get_data(modeldir, subvols)

    # A counting statistic (cash) needs the constraint's dm * V so it can
    # recover object counts from log10(phi).  Constraints that are not
    # histograms -- the black hole-bulge relation, the MZR, the ICS fractions,
    # the mass loading -- have no counts, so Poisson is undefined for them and
    # they fall back to chi2.  Said once per constraint rather than per
    # particle, or it would be thousands of lines.
    if getattr(stat_test, 'needs_count_scale', False):
        scale = getattr(constraint, 'count_scale', None)
        if scale is None:
            name = type(constraint).__name__
            if name not in _warned_no_counts:
                _warned_no_counts.add(name)
                logger.warning(
                    '%s is not a counting statistic (no bin width), so the '
                    'Poisson/Cash test is undefined for it; scoring it with '
                    'chi2 instead. The counting constraints in this run still '
                    'use Cash.', name)
            from src import analysis as _analysis
            score = _analysis.chi2(y_obs, y_mod, err)
        else:
            score = stat_test(y_obs, y_mod, err, count_scale=scale)
    else:
        score = stat_test(y_obs, y_mod, err)
    n = len(y_obs)
    if n == 0:
        # Previously this returned the raw score of empty arrays, i.e. 0.0, so a
        # constraint with no observations in its domain looked like a perfect
        # fit AND still consumed its share of the relative weight -- silently
        # diluting the objective for every other constraint.  Fail instead.
        raise ValueError(
            '%s produced no data points in its domain %s. Check that the '
            'observation covers this redshift and mass range.'
            % (type(constraint).__name__, getattr(constraint, 'domain', '?')))
    return score / n   # reduced chi² — equal weight per data point

count = 0
def run_sage_hpc(particles, *args):
    """Modified version for SAGE with parallel particle execution."""
    global count, opts
    opts, space, subvols, statTest = args

    # Create base output directory for all particles
    modeldir = os.path.join(opts.outdir, f'DS_output_{count}/')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    fx = np.zeros(len(particles))
    processes = []

    # Determine if we should use SLURM
    use_slurm = opts.cpus > 4

    if use_slurm:
        # Prepare for SLURM submission
        job_name = f'PSOSMF_{count}'
        
        for i, particle in enumerate(particles):
            # Create particle subdirectory
            particle_dir = os.path.join(modeldir, f"{i}/")
            os.makedirs(particle_dir, exist_ok=True)
            
            # Get JOBFS path from environment
            jobfs = os.environ.get('JOBFS', '/var/tmp')
            work_dir = os.path.join(jobfs, 'jobfs')
            
            # Create particle-specific parameter file
            slash = len(opts.config) - opts.config[::-1].find('/') - 1
            temp_filename = os.path.join(opts.outdir, f'{opts.config[slash+1:-4]}_{count}_{i}_temp.par')
            
            # Modify parameter file with JOBFS path
            write_particle_par(opts.config, temp_filename, work_dir,
                               space, particle)

            logger.info(f'Submitting SAGE SLURM job for: {temp_filename}')

            # Create SLURM batch script
            batch_script = os.path.join(opts.outdir, f'slurm_script_{count}_{i}.slurm')
            with open(batch_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={job_name}_{i}\n")
                f.write("#SBATCH --output=/dev/null\n")
                f.write("#SBATCH --error=/dev/null\n\n")
                f.write(f"#SBATCH --ntasks={opts.cpus}\n")
                f.write(f"#SBATCH --mem-per-cpu={opts.memory}\n")
                f.write("#SBATCH --tmp=200GB\n")

                if opts.walltime:
                    f.write(f"#SBATCH --time={opts.walltime}\n")
                if opts.account:
                    f.write(f"#SBATCH --account={opts.account}\n")
                if opts.queue:
                    f.write(f"#SBATCH --partition={opts.queue}\n")
                
                # Create work directory in JOBFS
                f.write("\n# Setup working directory\n")
                f.write(f'mkdir -p {work_dir}\n')
                
                # Show initial JOBFS state
                f.write('\necho "Initial JOBFS status:"\n')
                f.write('df -h $JOBFS\n')
                
                f.write("\nml purge\n")
                f.write("ml restore basic\n\n")
                
                # Run SAGE
                f.write(f"echo 'Starting SAGE job with {opts.cpus} CPUs'\n")
                f.write(f"mpirun -np {opts.cpus} {opts.sage_binary} {temp_filename}\n")
                
                # Copy results back with error checking
                f.write("\necho 'Copying results to permanent storage...'\n")
                f.write(f'if [ "$(ls -A {work_dir})" ]; then\n')
                f.write(f'    cp -r {work_dir}/* {particle_dir}\n')
                f.write('    echo "Copy completed successfully"\n')
                f.write('else\n')
                f.write('    echo "Error: No files found in JOBFS working directory"\n')
                f.write('    exit 1\n')
                f.write('fi\n\n')

            # Submit job using the batch script
            cmdline = ['sbatch', batch_script]
            process = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"SLURM submission failed: {err.decode()}")
                
            # Extract job ID from sbatch output
            jobid = out.decode().strip().split()[-1]
            processes.append((i, jobid, temp_filename, particle_dir, batch_script))

        # Wait for all SLURM jobs to complete
        total_particles = len(processes)
        while True:
            remaining = count_jobs(job_name)
            _print_progress(total_particles - remaining, total_particles, count)
            if remaining == 0:
                time.sleep(2)
                break
            time.sleep(10)
        sys.stdout.write('\n')
        sys.stdout.flush()

    else:
        # Use MPI implementation for small CPU requests
        for i, particle in enumerate(particles):
            # Create particle subdirectory
            particle_dir = os.path.join(modeldir, f"{i}/")
            os.makedirs(particle_dir, exist_ok=True)
            
            # Create particle-specific parameter file
            slash = len(opts.config) - opts.config[::-1].find('/') - 1
            temp_filename = os.path.join(opts.outdir, f'{opts.config[slash+1:-4]}_{count}_{i}_temp.par')
            
            # Modify parameter file
            write_particle_par(opts.config, temp_filename, particle_dir,
                               space, particle)

            # Launch SAGE with MPI for each particle
            cmdline = [
                'mpirun',
                '-np', str(opts.cpus),
                opts.sage_binary,
                temp_filename
            ]

            # Launch process and store handle
            process = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=os.path.dirname(opts.sage_binary))
            processes.append((i, process, temp_filename, particle_dir))
            
        # Wait for all MPI processes to complete
        total_particles = len(processes)
        _print_progress(0, total_particles, count)
        for done, (i, process, temp_filename, particle_dir) in enumerate(processes, start=1):
            out, err = process.communicate()
            _print_progress(done, total_particles, count)
            if process.returncode != 0:
                logger.error(f"SAGE instance {i} failed with return code {process.returncode}")
                logger.error(f"stdout: {out.decode()}")
                logger.error(f"stderr: {err.decode()}")
                raise RuntimeError(f"SAGE instance {i} failed")
        sys.stdout.write('\n')
        sys.stdout.flush()

    # Process results and clean up
    for item in processes:
        if use_slurm:
            try:
                i, jobid, temp_filename, particle_dir, batch_script = item
            except ValueError:
                # Handle case where we might not have batch_script
                i, jobid, temp_filename, particle_dir = item
                batch_script = None
        else:
            i, process, temp_filename, particle_dir = item
            
        # Process results with retries
        max_retries = 1
        retry_delay = 10
        success = False
        #logger.info("Processing everything, combining HDF5 files if needed, etc. etc. etc.")
        
        for retry in range(max_retries):
            try:
                scores = {c: _evaluate(c, statTest, particle_dir, subvols) for c in opts.constraints}
                total_score = sum(scores[c] * c.rel_weight for c in opts.constraints)
                score_parts = '  '.join(f'{c.__class__.__name__}={scores[c]:.3f}' for c in opts.constraints)
                logger.debug(f'Particle {i} scores: {score_parts}  total={total_score:.4f}')
                fx[i] = total_score
                success = True
                break
            except Exception as e:
                logger.warning(f"Attempt {retry+1}: Error evaluating particle: {e}")
                time.sleep(retry_delay)
                
        if not success:
            logger.warning(f"Failed to process outputs for particle {i}  - assigning penalty score")
            logger.warning("This usually means SAGE is unhappy, bad parameter combination")
            fx[i] = 1e10

        # Clean up parameter file and batch script
        try:
            os.remove(temp_filename)
            if use_slurm:
                os.remove(batch_script)  # Clean up the batch script
        except OSError:
            pass

    # Clean up output directory if not keeping
    if not opts.keep:
        shutil.rmtree(modeldir)

    logger.debug('Particles %r evaluated to %r', particles, fx)
    count += 1
    return fx

# this is the 'func' that is passed into the pso routine in pso.py
# this doesn't require the new pso.py file, can just run on the original pyswarm function
def run_sage(particle, *args):
    
    # space is the thing containing the parameter values for the model
    opts, space, subvols, statTest = args
    
    # create/clear directory for temporary Dark Sage output
    spid = str(multiprocessing.current_process().pid)
    # modeldir = 'output/DS_output_'+spid+'/'  # Example local output path
    modeldir = os.path.join(opts.outdir, 'DS_output_' + spid + '/')
    if not os.path.exists(modeldir): os.makedirs(modeldir)
    if os.path.isfile(modeldir+'model_z0.000_0'): subprocess.call(['rm', modeldir+'model*'])

    # copy template parameter file and edit accordingly
    slash = len(opts.config) - opts.config[::-1].find('/') - 1
    # temp_filename = 'output/' + opts.config[slash+1:-4] + '_' + spid + '_temp.par'  # Example local temp file
    temp_filename = os.path.join(opts.outdir, opts.config[slash+1:-4] + '_' + spid + '_temp.par')
    #
    write_particle_par(opts.config, temp_filename, modeldir, space, particle)

#    cmdline = ['mpirun', '-np', '8', opts.sage_binary, temp_filename]
    cmdline = [opts.sage_binary, temp_filename]
    _exec_sage('Running SAGE instance', cmdline, cwd=os.path.dirname(opts.sage_binary))

    # Weighted mean of per-constraint reduced scores.  _evaluate divides each
    # constraint's score by its number of data points, so a constraint with 32
    # points and one with 7 get equal say once rel_weight is applied; the total
    # therefore sits between the per-constraint reduced values and is nowhere
    # near a sum of raw chi2 values.
    scores = {c: _evaluate(c, statTest, modeldir, subvols) for c in opts.constraints}
    total = sum(scores[c] * c.rel_weight for c in opts.constraints)
    # Same breakdown the SLURM path logs, so the contribution of each
    # constraint is visible in whichever path is running.
    score_parts = '  '.join(
        f'{c.__class__.__name__}={scores[c]:.3f}*{c.rel_weight:.2f}'
        for c in opts.constraints)
    logger.debug('Particle scores: %s  total=%.4f', score_parts, total)
    logger.debug('Particle %r evaluated to %f', particle, total)

    shutil.rmtree(modeldir)
    return total
