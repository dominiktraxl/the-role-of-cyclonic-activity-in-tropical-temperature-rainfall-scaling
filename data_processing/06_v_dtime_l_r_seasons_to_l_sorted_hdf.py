import os
import subprocess

import pandas as pd
import numpy as np

# parameters
r = .1
p = 0

# filesystem folders
storefolder = os.getcwd() + '/'

# load data
print('loading v_r{}_p{}.feather ..'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')

print('loading v_r{}_p{}_seasons.feather'.format(r, p))
v_season = pd.read_feather(
    storefolder + 'v_r{}_p{}_seasons.feather'.format(r, p))
for col in v_season.columns.values:
    print('appending {} ..'.format(col))
    v[col] = v_season[col].values
del v_season

print('loading v_r{}_p{}_l_sort_index.npy'.format(r, p))
index = np.load(storefolder + 'v_r{}_p{}_l_sort_index.npy'.format(r, p))

# delete unused columns
print('sort by index, select ..')
locs = np.where(v.columns.values == 'l')[0][0]
rs = np.where(v.columns.values == 'r')[0][0]
dts = np.where(v.columns.values == 'dtime')[0][0]
v = v.iloc[index, [dts, locs, rs, -4, -3, -2, -1]]
print(v.columns.values)

# store sorted dataframe
print('storing v_r{}_p{}_dtime_l_r_seasons_l_sorted.h5 ..'.format(r, p))
store = pd.HDFStore(
    storefolder + 'v_r{}_p{}_dtime_l_r_seasons_l_sorted.h5'.format(r, p),
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
       storefolder + 'v_r{}_p{}_dtime_l_r_seasons_l_sorted.h5'.format(r, p),
       storefolder + 'v_r{}_p{}_dtime_l_r_seasons_l_sortedc{}.h5'.format(
           r, p, c)]
subprocess.Popen(cmd).wait()

# create index
print('creating table index...')
store = pd.HDFStore(
    storefolder + 'v_r{}_p{}_dtime_l_r_seasons_l_sortedc{}.h5'.format(
        r, p, c))
store.create_table_index('v', columns=['l'], kind='full')
store.close()
