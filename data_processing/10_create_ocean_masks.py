import os

from netCDF4 import Dataset
from scipy import ndimage
import numpy as np

# filesystem folders
storefolder = os.getcwd() + '/'

# load mask (cut off extra data)
lsm = Dataset(storefolder + 'TMPA_mask.nc').variables['landseamask'][:]
lsm = lsm[160:-160, :]

# 100% = all water and 0% = all land
sea = lsm.copy()
sea[lsm < 100] = 0
sea[lsm == 100] = 1

# enforce seperate oceans
ocean = sea.copy()
ocean[:200, 800] = 0
ocean[:170, 1200] = 0

# labeling
ocean, num = ndimage.label(ocean, structure=np.ones((3, 3)))

# merge pacific
ocean[ocean == 5] = 1

# get rid of the rest
ocean[ocean > 3] = 0

# remove borders between oceans
ocean[:200, 800] = 3
ocean[:170, 1200] = 1
ocean[sea == 0] = 0

# store masks
print('storing masks..')
np.save(storefolder + 'sea.npy', sea)
np.save(storefolder + 'ocean.npy', ocean)
