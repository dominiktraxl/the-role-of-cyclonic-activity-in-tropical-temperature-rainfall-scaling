import os

import argparse
import subprocess

import pandas as pd
# import dask.dataframe as dd
import numpy as np

# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('which',
                    help="which temperature type to use "
                         "[sat, sst, sdt, RH]",
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

# filesystem folders
storefolder = os.getcwd() + '/'
subfolder = 'v_r{}_p{}_SXT_td_X/'.format(r, p)

# load data
print('loading v_r{}_p{}.feather ..'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')[['l']]

print('loading v_r{}_p{}_l_sort_index.npy'.format(r, p))
index = np.load(storefolder + 'v_r{}_p{}_l_sort_index.npy'.format(r, p))

for td in tds:

    print('create vt by copying v ..')
    vt = v.copy()

    sxtcol = '{}_td_{}'.format(which, td)

    print('loading {}'.format(sxtcol))
    sxt = pd.read_feather(
        storefolder + subfolder + '{}.feather'.format(sxtcol))
    vt[sxtcol] = sxt[sxtcol].values

    # delete sxt
    del sxt

    # convert to dask dataframe
    # print('converting to dask dataframe ..')
    # vt = dd.from_pandas(vt, npartitions=100)

    print('sort {}_td_{} by index ..'.format(which, td))
    vt = vt.iloc[index]
    print(vt.columns.values)

    # store sorted dataframe
    print('storing {}_td_{}_l_sorted.h5 ..'.format(which, td))
    store = pd.HDFStore(
        storefolder + subfolder + '{}_td_{}_l_sorted.h5'.format(which, td),
        mode='w')
    store.append('v', vt, format='t', data_columns=True, index=False)
    store.close()

    # compress
    print('compressing v...')
    c = 7
    cmd = ['ptrepack',
           '--overwrite',
           '--chunkshape=auto',
           '--complib=blosc',
           '--complevel={}'.format(c),
           storefolder + subfolder + '{}_td_{}_l_sorted.h5'.format(which, td),
           storefolder + subfolder + '{}_td_{}_l_sortedc{}.h5'.format(
               which, td, c)]
    subprocess.Popen(cmd).wait()

    # create index
    print('creating table index...')
    store = pd.HDFStore(
        storefolder + subfolder + '{}_td_{}_l_sortedc{}.h5'.format(
            which, td, c))
    store.create_table_index('v', columns=['l'], kind='full')
    store.close()

    # remove uncompressed file
    os.remove(
        storefolder + subfolder + '{}_td_{}_l_sorted.h5'.format(which, td))
    print('deleted {}!'.format(
        storefolder + subfolder + '{}_td_{}_l_sorted.h5'.format(which, td)))
