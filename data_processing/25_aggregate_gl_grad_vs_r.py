import os
import argparse

import numpy as np
import pandas as pd

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

# tropical cyclone folder string
if args.only_tropical_cyclones:
    tcfstr = '_tc'
else:
    tcfstr = ''

# parameters
r = .1
p = 0
cg_N = args.coarse_grained

# filesystem folders
storefolder = os.getcwd() + '/'
if cg_N is not None:
    glpartsfolder = f'gl_cg_{cg_N}_grad_vs_r{tcfstr}_parts/'.format(cg_N)
    gl_file_name = f'gl_r{r}_p{p}_cg_{cg_N}_grad_vs_r{tcfstr}_fits.pickle'
    # gl_file_name = f'gl_r{r}_p{p}_cg_{cg_N}_grad_vs_r{tcfstr}_fits_24_12.pickle'
else:
    glpartsfolder = f'gl_grad_vs_r{tcfstr}_parts/'
    gl_file_name = f'gl_r{r}_p{p}_grad_vs_r{tcfstr}_fits.pickle'

# load data
fs = os.listdir(storefolder + glpartsfolder)
fs.sort()


def find_scrambled_files():

    # gls = []
    bgs = []
    for i, f in enumerate(fs):
        try:
            print('{:06d}/{:06d} | {}'.format(i, len(fs), f))
            pd.read_pickle(storefolder + glpartsfolder + f)
            # print('{:06d}/{:06d} | {}'.format(i, len(fs), f))
            # gls.append(pd.read_pickle(storefolder + glpartsfolder + f))
        except EOFError:
            print('{:06d}/{:06d} | {}   BAD GUY!'.format(i, len(fs), f))
            bgs.append(int(f.split('.')[0]))

    bgs = np.asarray(bgs)
    np.save(storefolder + 'tmp_grad_vs_r_repair_locations.npy', bgs)


def store_agg():

    gls = []
    for i, f in enumerate(fs):
        try:
            gls.append(pd.read_pickle(storefolder + glpartsfolder + f))
            print('{:06d}/{:06d} | {}'.format(i, len(fs), f))
        except EOFError:
            print('{:06d}/{:06d} | {}   BAD GUY!'.format(i, len(fs), f))
            raise

    gl = pd.concat(gls, axis=0, sort=True)

    # store
    fname = storefolder + gl_file_name
    print('store {}'.format(fname))
    gl.to_pickle(fname)


# find_scrambled_files()
store_agg()
