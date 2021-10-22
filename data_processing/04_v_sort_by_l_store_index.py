import os

import numpy as np
import pandas as pd

# parameters
r = .1
p = 0

# filesystem folders
storefolder = os.getcwd() + '/'

# load data
print('loading v_r{}_p{}.feather'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')

# delete unused columns
print('select l values from v ..')
v = v['l'].values

# sort
print('sorting v by geographical locations...')
index = np.argsort(v, kind='mergesort')

# store
print('storing v_r{}_p{}_l_sort_index.npy'.format(r, p))
np.save(storefolder + 'v_r{}_p{}_l_sort_index.npy'.format(r, p), index)
