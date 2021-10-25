import os
import subprocess
import time
import itertools

import numpy as np

# parameters
# n_jobs = 2
n_locs_per_job = 100

r = .1
p = 0
node_list = ['kailash', 'manaslu', 'everest', 'k2'] + \
    ['pcpool{}'.format(i) for i in range(1, 18)]

# filesystem folders
storefolder = os.getcwd() + '/'
partsfolder = 'glcp_burst_parts/'
spartsfolder = 'glcp_burst_sat_grad_parts/'
os.makedirs(storefolder + spartsfolder, exist_ok=True)

# slurm folders
slurmSBFiles = 'glcp_burst_parts_sat_grad_SBFiles/'
slurmOUT = 'glcp_burst_parts_sat_grad_OUTPUT/'
slurmSTDERR = 'glcp_burst_parts_sat_grad_STDERR/'
os.makedirs(storefolder + slurmSBFiles, exist_ok=True)
os.makedirs(storefolder + slurmOUT, exist_ok=True)
os.makedirs(storefolder + slurmSTDERR, exist_ok=True)


# bursts to compute sat grads for
locs = os.listdir(storefolder + partsfolder)
# locs.sort()
n_total = len(locs)

# already computed sat grads
clocs = os.listdir(storefolder + spartsfolder)
# clocs.sort()

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
    f.write('#SBATCH --job-name=s{:06d}\n'.format(locs[0]))

    # partition
    f.write('#SBATCH --partition=pcpool,base\n')

    # nodes/cpus/mpthreads
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks-per-node=1\n')
    f.write('#SBATCH --cpus-per-task=1\n')
    f.write('#SBATCH --ntasks-per-core=1\n')

    # specify node randomly
    # f.write('#SBATCH --nodelist={}\n'.format(node))

    # memory per cpu-core\n')
    f.write('#SBATCH --mem-per-cpu=800M\n')

    # total run time limit (HH:MM:SS)
    f.write('#SBATCH --time=05:00:00\n')

    # STDOUT
    f.write('#SBATCH --output={}\n'.format(output))

    # STDERR
    f.write('#SBATCH --error={}\n'.format(stderr))

    # temporal partition
    # f.write('#SBATCH --partition=long\n')

    # exclude nodes
    f.write("#SBATCH --exclude="
            "cachi,pcpool4,pcpool5,pcpool12,pcpool20\n")

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
    f.write(
        "srun 20_slurm_create_glcp_burst_per_location_sat_grad.py "
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
locs = locs[:n_jobs*n_locs_per_job]
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

    time.sleep(.5)

    i += 1
