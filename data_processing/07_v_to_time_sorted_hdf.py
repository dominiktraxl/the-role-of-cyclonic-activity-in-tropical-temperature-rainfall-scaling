import os
import subprocess

import pandas as pd

# parameters
r = .1
p = 0

# filesystem folders
storefolder = os.getcwd() + '/'

# load data
print('loading v_r{}_p{}.feather ..'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')

# store sorted dataframe
print('storing v_r{}_p{}_dtime_sorted.h5 ..'.format(r, p))
store = pd.HDFStore(
    storefolder + 'v_r{}_p{}_dtime_sorted.h5'.format(r, p),
    mode='w')
store.append('v', v, format='t', data_columns=True, index=False)
store.close()

# compress
print('compressing v...')
c = 7
cmd = ['ptrepack',
       '--overwrite',
       '--chunkshape=auto',
       '--complib=blosc',
       '--complevel={}'.format(c),
       storefolder + 'v_r{}_p{}_dtime_sorted.h5'.format(r, p),
       storefolder + 'v_r{}_p{}_dtime_sortedc{}.h5'.format(r, p, c)]
subprocess.Popen(cmd).wait()

# create index
print('creating table index...')
store = pd.HDFStore(storefolder + 'v_r{}_p{}_dtime_sortedc{}.h5'.format(
    r, p, c))
store.create_table_index('v', columns=['dtime'], kind='full')
store.close()

# remove uncompressed file
os.remove(storefolder + 'v_r{}_p{}_dtime_sorted.h5'.format(r, p))
print('deleted {}!'.format(storefolder + 'v_r{}_p{}_dtime_sorted.h5'.format(
    r, p)))
