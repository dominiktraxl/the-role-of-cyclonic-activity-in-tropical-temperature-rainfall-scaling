import os

import numpy as np
import pandas as pd

# parameters
r = .1
p = 90

# filesystem folders
storefolder = os.getcwd() + '/'

# load v, cp
print('loading data...')
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')

# v
print('create month index ..')
month = v['dtime'].dt.month

print('create new df ..')
v = pd.DataFrame(index=v.index)

print('set columns ..')
v['JJA'] = np.zeros(len(v), dtype=bool)
v['MJJAS'] = np.zeros(len(v), dtype=bool)
v['DJF'] = np.zeros(len(v), dtype=bool)
v['NDJFM'] = np.zeros(len(v), dtype=bool)

print('JJA')
v.loc[month.isin([6, 7, 8]), 'JJA'] = True
print('DJF')
v.loc[month.isin([12, 1, 2]), 'DJF'] = True
print('MJJAS')
v.loc[month.isin([5, 6, 7, 8, 9]), 'MJJAS'] = True
print('NDJFM')
v.loc[month.isin([11, 12, 1, 2, 3]), 'NDJFM'] = True

# store v, cp
print('storing v_r{}_p{}_seasons.feather ..'.format(r, p))
v.to_feather(storefolder + 'v_r{}_p{}_seasons.feather'.format(r, p))
