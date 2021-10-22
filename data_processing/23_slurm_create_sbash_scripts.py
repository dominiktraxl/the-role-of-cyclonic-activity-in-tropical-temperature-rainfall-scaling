import os

import argparse
import subprocess
import time
import itertools

import numpy as np

# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-cg', '--coarse-grained',
    type=int)
parser.add_argument(
    "-tcs", '--only-tropical-cyclones', action='store_true',
    help='consider only rainfall events tagged as part of a tropical cyclones')
args = parser.parse_args()

# tropical cyclone strings
if args.only_tropical_cyclones:
    tcfstr = '_tc'
    tcestr = '-tcs '
else:
    tcfstr = ''
    tcestr = ''

# parameters
cg_N = args.coarse_grained
n_locs_per_job = 100
if cg_N == 4 or cg_N == 8:
    n_locs_per_job = 10
r = .1
p = 0

# filesystem folders
storefolder = os.getcwd() + '/'
spartsfolder = 'glcp_burst_sat_grad_parts/'
# spartsfolder = 'glcp_burst_sat_grad_parts_24_12/'
if cg_N is not None:
    glpartsfolder = f'gl_cg_{cg_N}_grad_vs_r{tcfstr}_parts/'
else:
    glpartsfolder = f'gl_grad_vs_r{tcfstr}_parts/'
os.makedirs(storefolder + glpartsfolder, exist_ok=True)

# slurm folders
if cg_N is not None:
    slurmSBFiles = f'gl_cg_{cg_N}_grad_vs_r{tcfstr}_parts_SBFiles/'
    slurmOUT = f'gl_cg_{cg_N}_grad_vs_r{tcfstr}_parts_OUTPUT/'
    slurmSTDERR = f'gl_cg_{cg_N}_grad_vs_r{tcfstr}_parts_STDERR/'
else:
    slurmSBFiles = f'gl_grad_vs_r{tcfstr}_parts_SBFiles/'
    slurmOUT = f'gl_grad_vs_r{tcfstr}_parts_OUTPUT/'
    slurmSTDERR = f'gl_grad_vs_r{tcfstr}_parts_STDERR/'
os.makedirs(storefolder + slurmSBFiles, exist_ok=True)
os.makedirs(storefolder + slurmOUT, exist_ok=True)
os.makedirs(storefolder + slurmSTDERR, exist_ok=True)


# bursts to compute sat grads for
if cg_N is not None:
    locs = ['{:06d}.pickle'.format(loc) for loc in range(400*1440//cg_N**2)]
else:
    locs = os.listdir(storefolder + spartsfolder)
    locs.sort()
n_total = len(locs)

# already computed gl parts
clocs = os.listdir(storefolder + glpartsfolder)
clocs.sort()

# filter already computed locations
locs = np.setdiff1d(locs, clocs, assume_unique=True)

# random shuffle
np.random.seed(0)
np.random.shuffle(locs)

print('number of locations not yet computed: {:06d}/{:06d}'.format(
    len(locs), n_total))

# ints
locs = [int(loc[:loc.find('.')]) for loc in locs]

# number of jobs
n_jobs = (len(locs) + n_locs_per_job - 1) // n_locs_per_job
print('n_jobs: {}'.format(n_jobs))


def _generate_sbatch_commandfile(locs, sbfile):

    # node = np.random.choice(node_list)
    # print(node)

    # slurm files
    output = storefolder + slurmOUT + '{:06d}.out'.format(locs[0])
    stderr = storefolder + slurmSTDERR + '{:06d}.err'.format(locs[0])

    f = open(sbfile, 'w')

    f.write('#!/bin/bash\n')

    # assign a short name to your job
    f.write('#SBATCH --job-name=g{:06d}\n'.format(locs[0]))

    # partition
    f.write('#SBATCH --partition=pcpool,geopool,base\n')

    # nodes/cpus/mpthreads
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks-per-node=1\n')
    f.write('#SBATCH --cpus-per-task=1\n')
    f.write('#SBATCH --ntasks-per-core=1\n')

    # specify node randomly
    # f.write('#SBATCH --nodelist={}\n'.format(node))

    # memory per cpu-core\n')
    if cg_N is not None:
        f.write('#SBATCH --mem-per-cpu=5G\n')
    else:
        f.write('#SBATCH --mem-per-cpu=2M\n')

    # total run time limit (HH:MM:SS)
    f.write('#SBATCH --time=20:00:00\n')

    # STDOUT
    f.write('#SBATCH --output={}\n'.format(output))

    # STDERR
    f.write('#SBATCH --error={}\n'.format(stderr))

    # temporal partition
    # f.write('#SBATCH --partition=long\n')

    # exclude nodes
    f.write("#SBATCH --exclude="
            "manaslu,cachi,pcpool12,pcpool18\n")

    # set array
    # f.write('#SBATCH --array=1-8\n')

    # export paths
    # CONDA_PATH = '/home/traxl/anaconda3/bin'
    # PY3_PATH = '/home/traxl/anaconda3/envs/py3/bin/'
    # f.write('export PATH={}:{}\n'.format(CONDA_PATH, PY3_PATH))

    # activate conda
    f.write('\nsource {}/../../.bashrc\n'.format(os.getcwd()))
    # f.write('/home/traxl/anaconda3/bin/conda init bash\n')
    # f.write('/home/traxl/anaconda3/bin/conda activate py3\n')

    # execute python script
    # $SLURM_ARRAY_TASK_ID
    f.write('cd {}\n'.format(os.getcwd()))
    if cg_N is not None:
        f.write(
            f"srun 13b_slurm_create_gl_grad_vs_r_linregress.py {tcestr}"
            "-cg {} -l {}\n".format(cg_N, ' '.join([str(loc) for loc in locs]))
        )
    else:
        f.write(
            f"srun 13b_slurm_create_gl_grad_vs_r_linregress.py {tcestr}"
            "-l {}\n".format(' '.join([str(loc) for loc in locs]))
        )
    f.flush()
    f.close()


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# write and execute
# locs = locs[:20]
locsblocks = grouper(n_locs_per_job, locs)
i = 1
for locs in locsblocks:

    # create sbatch file
    sbfile = storefolder + slurmSBFiles + '{:06d}.sh'.format(locs[0])
    _generate_sbatch_commandfile(locs, sbfile)

    # queue sbatch file
    print('{:06d}/{:06d} | {:06d}\n'.format(
        i, n_jobs, locs[0]), end='', flush=True)
    # print("submitting sbatch: {}\n".format(sbfile), end='', flush=True)
    cmd = ['sbatch', sbfile]
    subprocess_p = subprocess.Popen(cmd)
    subprocess_p.wait()

    time.sleep(.1)

    i += 1
