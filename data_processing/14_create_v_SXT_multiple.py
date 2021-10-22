import os

import argparse

import numpy as np
import pandas as pd
import xarray

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
td_max = max(tds)

# load data
storefolder = os.getcwd() + '/'
data_sat = storefolder + 'ERA5_2mT/'
data_sdt = storefolder + 'ERA5_2m_dewpoint_T/'

v_sxt_parts_folder = 'v_r{}_p{}_SXT_td_X_parts/'.format(r, p)
os.makedirs(storefolder + v_sxt_parts_folder, exist_ok=True)

print('loading v_r{}_p{}.feather ..'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')
print('loading v_r{}_p{}_year_hoy_doy.feather ..'.format(r, p))
v_dt_props = pd.read_feather(
    storefolder + 'v_r{}_p{}_year_doy_hoy.feather'.format(r, p)
)
for col in ['year', 'hoy']:
    print('setting {} ..'.format(col))
    v[col] = v_dt_props[col]
del v_dt_props

# generalize paramters to sxt
if which == 'SAT':
    allsxtfiles = os.listdir(data_sat)
    sxtfiles = []
    for sxtfile in allsxtfiles:
        if sxtfile.startswith('2mT_') and 'NOAA' in sxtfile:
            sxtfiles.append(sxtfile)
    data_sxt = data_sat
    nc_var = 't2m'

elif which == 'SDT':
    allsxtfiles = os.listdir(data_sdt)
    sxtfiles = []
    for sxtfile in allsxtfiles:
        if 'NOAA' in sxtfile:
            sxtfiles.append(sxtfile)
    data_sxt = data_sdt
    nc_var = 'd2m'

# sort files by years
sxtfiles.sort()


def ERA5_to_cube(data_sxt, filen, td_max=None):

    # load dataset
    sxt = xarray.open_dataset(data_sxt + filen)

    # select by time delay
    if td_max is not None:
        sxt = sxt.isel(time=slice(-td_max, None))

    # reshape
    sxt = sxt.isel(lat=slice(160, -160))

    # transform sats
    print('getting {} values ..'.format(nc_var))
    sxt = sxt.variables[nc_var].values
    print('reshaping sxt ..')
    sxt = np.roll(sxt, sxt.shape[2]//2, axis=2)

    return sxt


# filter files
def preprocess(filen, which):

    # year
    if which == 'SAT':
        year = int(filen.split('_')[1])
    elif which == 'SDT':
        year = int(filen.split('_')[3])

    # load dataset
    sxt = ERA5_to_cube(data_sxt, filen)

    # convert to °C
    sxt -= 273.15

    assert sxt.shape[0] >= 365 * 24

    # add time-delayed sxts from year before
    if year > 1998:

        # load last years file
        idx = sxtfiles.index(filen)
        filen_ly = sxtfiles[idx-1]
        print(filen_ly)
        sxt_ly = ERA5_to_cube(data_sxt, filen_ly, td_max=td_max)

        # convert to °C
        sxt_ly -= 273.15

        # combine cubes
        print('concatenate with last years sdts.. ')
        sxt = np.concatenate((sxt, sxt_ly), axis=0)

    return year, sxt


# iterate through years
sxtcols = [which.lower() + '_td_{}'.format(td) for td in tds]

for filen in sxtfiles:

    print('-------------------------------------------------------')
    print('parameters: r={}\t p={}\t which={}'.format(r, p, which))
    print('processing {} ..'.format(filen))
    year, sxt = preprocess(filen, which)
    vyear = v['year'] == year
    vt = v.loc[vyear]

    # add sxt column to v
    for sxtcol, td in zip(sxtcols, tds):
        print('look up temps for {} ..'.format(sxtcol))
        vt['hoy_td'] = vt['hoy'] - td
        vyearsxtcol = sxt[vt['hoy_td'], vt['y'], vt['x']]

        if year == 1998:
            vyearsxtcol[vt['hoy_td'] < 0] = np.nan

        # store
        np.save(storefolder + v_sxt_parts_folder + '{}_td_{}_{}.npy'.format(
            which.lower(), td, year), vyearsxtcol)
