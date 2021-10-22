import os

import pandas as pd

# parameters
r = .1
p = 0
cg_N = 8

# filesystem folders
storefolder = os.getcwd() + '/'

# gl_regions
gl_regions = pd.read_pickle(storefolder + 'gl_regions.pickle')

# to select from corse grained information
gl_cg = pd.read_feather(
    storefolder + 'gl_r{}_p{}_cg_{}.feather'.format(r, p, cg_N)
)

gl_cg_locs = gl_cg.copy()
del gl_cg_locs['l']
del gl_cg_locs['x']
del gl_cg_locs['y']
del gl_cg_locs['lon']
del gl_cg_locs['lat']

# majority vote
data = {'hs': [], 'climate': [], 'lw': [], 'ocean': []}
for i, locs in gl_cg_locs.iterrows():
    print(i)
    for key in data.keys():
        winner = gl_regions.loc[locs.values, key].value_counts().index[0]
        data[key].append(winner)
gl_cg_regions = pd.DataFrame(index=gl_cg.index, data=data)

# store
gl_cg_regions.to_pickle(storefolder + 'gl_cg_{}_regions.pickle'.format(cg_N))
