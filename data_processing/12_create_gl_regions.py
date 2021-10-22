import os

import numpy as np
import pandas as pd

# parameters
r = .1
p = 0

# filesystem folders
storefolder = os.getcwd() + '/'

# load masks
print('loading data ..')
water = np.load(storefolder + 'sea.npy')
ocean = np.load(storefolder + 'ocean.npy')

# load gl
gl = pd.read_feather(storefolder + f'gl_r{r}_p{p}_JJA.feather')
gl = gl[['x', 'y', 'lat', 'lon']]

# climate strings
climates = []
for hs in ['N', 'S']:
    for clim in ['', 'sub', 'exo']:
        climate_str = '{}_{}tropical'.format(hs, clim)
        climates.append(climate_str)

# north/south
print('N/S ..')
gl['hs'] = 'N'
gl.loc[gl['lat'] < 0, 'hs'] = 'S'
gl['hs'] = gl['hs'].astype('category')

# climate zone
print('tropics ..')
gl['N_tropical'] = False
gl['S_tropical'] = False
gl.loc[(gl['lat'] > 0) & (gl['lat'] <= 23.5), 'N_tropical'] = True
gl.loc[(gl['lat'] < 0) & (gl['lat'] >= -23.5), 'S_tropical'] = True

print('subtropics ..')
gl['N_subtropical'] = False
gl['S_subtropical'] = False
gl.loc[(gl['lat'] > 23.5) & (gl['lat'] <= 35), 'N_subtropical'] = True
gl.loc[(gl['lat'] < -23.5) & (gl['lat'] >= -35), 'S_subtropical'] = True

print('exotropics ..')
gl['N_exotropical'] = False
gl['S_exotropical'] = False
gl.loc[gl['lat'] > 35, 'N_exotropical'] = True
gl.loc[gl['lat'] < -35, 'S_exotropical'] = True

gl['climate'] = gl[climates].idxmax(axis=1)
gl['climate'] = gl['climate'].astype('category')

for clim in climates:
    del gl[clim]

# land/water
print('land/water...')
gl['land'] = ~ water.astype(bool)[gl.y.values, gl.x.values]
gl['lw'] = 'W'
gl.loc[gl['land'], 'lw'] = 'L'
del gl['land']
gl['lw'] = gl['lw'].astype('category')

# ocean types
print('ocean types...')
gl['ocean_type'] = ocean[gl['y'].values, gl['x'].values]
gl[['other', 'pacific', 'atlantic', 'indian']] = pd.get_dummies(
    gl['ocean_type'])
gl['ocean'] = gl[['other', 'pacific', 'atlantic', 'indian']].idxmax(axis=1)
gl['ocean'] = gl['ocean'].astype('category')

for col in ['ocean_type', 'other', 'pacific', 'atlantic', 'indian']:
    del gl[col]

# oceanic region
gl['region'] = gl['climate'].astype(str) + '_' + gl['ocean'].astype(str)
gl['region'] = gl['region'].astype('category')

# delete x, y, lat, lon
for col in ['x', 'y', 'lat', 'lon']:
    del gl[col]

# store
print('storing gl_regions.feather ..'.format(r, p))
gl.to_pickle(storefolder + 'gl_regions.pickle')
