import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction import image

# parameters
r = .1
p = 0
cg_N = 8

# filesystem folders
storefolder = os.getcwd() + '/'

# load original gl
gl = pd.read_feather(storefolder + 'gl_r{}_p{}_JJA.feather'.format(r, p))

# create grid
if cg_N == 2:

    xx, yy = np.meshgrid(np.arange(720), np.arange(199, -1, -1))
    lons, lats = np.meshgrid(np.arange(-179.75, 180., .5),
                             np.arange(49.75, -50, -.5))

    # labels for geographical locations
    locs = np.arange(xx.shape[0] * xx.shape[1]).reshape(xx.shape)

    # image
    img = gl['l'].values.reshape(400, 1440)

    # extract subtiles
    patches = image.extract_patches_2d(img, (2, 2)).reshape([-1] + [2, 2])

    # put together
    gl_cg = pd.DataFrame(data={
        'l': locs.flatten(),
        'x': xx.flatten(),
        'y': yy.flatten(),
        'lon': lons.flatten(),
        'lat': lats.flatten()})

    for i in range(2):
        for j in range(2):
            gl_cg['l{}{}'.format(i, j)] = patches[:, i, j]

if cg_N == 4:

    xx, yy = np.meshgrid(np.arange(360), np.arange(99, -1, -1))
    lons, lats = np.meshgrid(np.arange(-179.5, 180.5, 1),
                             np.arange(49.5, -50.5, -1))

    # labels for geographical locations
    locs = np.arange(xx.shape[0] * xx.shape[1]).reshape(xx.shape)

    # image
    img = gl['l'].values.reshape(400, 1440)

    # extract subtiles
    patches = image.extract_patches_2d(img, (4, 4)).reshape([-1] + [4, 4])

    # put together
    gl_cg = pd.DataFrame(data={
        'l': locs.flatten(),
        'x': xx.flatten(),
        'y': yy.flatten(),
        'lon': lons.flatten(),
        'lat': lats.flatten()})

    for i in range(4):
        for j in range(4):
            gl_cg['l{}{}'.format(i, j)] = patches[:, i, j]

if cg_N == 8:

    xx, yy = np.meshgrid(np.arange(180), np.arange(49, -1, -1))
    lons, lats = np.meshgrid(np.arange(-179., 180., 2),
                             np.arange(49., -50., -2))

    # labels for geographical locations
    locs = np.arange(xx.shape[0] * xx.shape[1]).reshape(xx.shape)

    # image
    img = gl['l'].values.reshape(400, 1440)

    # extract subtiles
    patches = image.extract_patches_2d(img, (8, 8)).reshape([-1] + [8, 8])

    # put together
    gl_cg = pd.DataFrame(data={
        'l': locs.flatten(),
        'x': xx.flatten(),
        'y': yy.flatten(),
        'lon': lons.flatten(),
        'lat': lats.flatten()})

    for i in range(8):
        for j in range(8):
            gl_cg['l{}{}'.format(i, j)] = patches[:, i, j]

# store
gl_cg.to_feather(storefolder + 'gl_r{}_p{}_cg_{}.feather'.format(r, p, cg_N))
print('stored gl_r{}_p{}_cg_{}.feather ..'.format(r, p, cg_N))
