import os

import argparse

import numpy as np
import pandas as pd

# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('which',
                    help='which temperature type to use',
                    choices=['SAT', 'SDT'],
                    type=str)
args = parser.parse_args()

# parameters
r = .1
p = 0
which = args.which
# tds = [
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
#     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
#     27, 30, 33, 36, 39, 42, 45, 48,
#     72, 96,
# ]
tds = [
    25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 47,
]
# load data
storefolder = os.getcwd() + '/'

v_sxt_parts_folder = 'v_r{}_p{}_SXT_td_X_parts/'.format(r, p)
subfolder = 'v_r{}_p{}_SXT_td_X/'.format(r, p)
os.makedirs(storefolder + subfolder, exist_ok=True)

# select temp type
allfiles = os.listdir(storefolder + v_sxt_parts_folder)
sxtfiles = []
for f in allfiles:
    if f.startswith(which.lower()):
        sxtfiles.append(f)

# go through tds and append to store
for td in tds:
    sxt_td_files = []
    for f in sxtfiles:
        if int(f.split('_')[2]) == td:
            sxt_td_files.append(f)
    sxt_td_files.sort()

    sxt_tds = []
    for sxt_td_f in sxt_td_files:
        print(sxt_td_f)
        sxt_td = np.load(storefolder + v_sxt_parts_folder + sxt_td_f)
        sxt_tds.append(sxt_td)

    sxt_td = np.concatenate(sxt_tds)
    col = '{}_td_{}'.format(which.lower(), td)
    v = pd.DataFrame(data={col: sxt_td})
    print('store as feather ..')
    v.to_feather(
        storefolder + subfolder + '{}_td_{}.feather'.format(which.lower(), td)
    )
