import os
import argparse

import numpy as np
from netCDF4 import Dataset  # @UnresolvedImport
import xarray
from dask.diagnostics import ProgressBar

# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('which',
                    help='which temperature type to use',
                    choices=['SAT', 'SST', 'SDT', 'RH'],
                    type=str)
parser.add_argument('statistic',
                    help='which statistic to compute',
                    choices=['mean', 'var'],
                    type=str)
parser.add_argument('-s', '--season',
                    help='subset to season',
                    choices=['JJA', 'DJF', 'MJJAS', 'NDJFM', 'JASO', 'DJFMA'],
                    type=str)
parser.add_argument('-ts', '--timesplit',
                    help='split data into invervals (1998-2008), (2008, 2019)',
                    choices=['fh', 'sh'],
                    type=str)
args = parser.parse_args()

# parameters
which = args.which
season = args.season
time_split = args.timesplit
statistic = args.statistic

if season is not None:
    sstr = '_{}'.format(season)
else:
    sstr = ''
if time_split is not None:
    tsstr = '_{}'.format(time_split)
else:
    tsstr = ''

# filesystem folders
storefolder = os.getcwd() + '/'
data_sst = storefolder + 'NOAA_OI_SST_V2/'
data_sat = storefolder + 'ERA5_2mT/'
data_sdt = storefolder + 'ERA5_2m_dewpoint_T/'

# generalize paramters to sxt
if which == 'SST':
    allsxtfiles = os.listdir(data_sst)
    sxtfiles = []
    for allsxtfile in allsxtfiles:
        specs = allsxtfile.split('.')
        if specs[0] == 'sst' and specs[2] == 'mean' and int(specs[3]) >= 1998:
            sxtfiles.append(allsxtfile)
    sxtfiles.sort()
    data_sxt = data_sst

elif which == 'SAT':
    allsxtfiles = os.listdir(data_sat)
    sxtfiles = []
    for sxtfile in allsxtfiles:
        if sxtfile.startswith('daily_2mT_') and 'NOAA' in sxtfile:
            sxtfiles.append(sxtfile)
    sxtfiles.sort()
    data_sxt = data_sat
    nc_var = 't2m'

elif which == 'SDT':
    allsxtfiles = os.listdir(data_sdt)
    sxtfiles = []
    for sxtfile in allsxtfiles:
        if 'NOAA' in sxtfile:
            sxtfiles.append(sxtfile)
    sxtfiles.sort()
    data_sxt = data_sdt
    nc_var = 'd2m'


def season_JJA(month):
    return month.isin([6, 7, 8])


def season_DJF(month):
    return month.isin([12, 1, 2])


def season_JASO(month):
    return month.isin([7, 8, 9, 10])


def season_DJFMA(month):
    return month.isin([12, 1, 2, 3, 4])


season_funcs = {
    'JJA': season_JJA,
    'DJF': season_DJF,
    'JASO': season_JASO,
    'DJFMA': season_DJFMA}


def NOAA_to_cube(data_sxt, filen):

    # load dataset
    rootgrp = Dataset(data_sxt + filen, "r", format="NETCDF4")

    # convert to xarray dataset
    sxt = xarray.open_dataset(xarray.backends.NetCDF4DataStore(rootgrp))

    if season is not None:
        # sxt = sxt.sel(time=sxt['time.season'] == season)
        sxt = sxt.sel(time=season_funcs[season](sxt['time.month']))

    # reshape
    sxt = sxt.variables['sst'].values
    sxt = np.roll(sxt, sxt.shape[2]//2, axis=2)

    return sxt


# filter files
def preprocess(filen, which):

    if which == 'SST':

        # year
        year = int(filen.split('.')[3])

        # load dataset
        sxt = NOAA_to_cube(data_sxt, filen)

        return year, sxt


if which == 'SST':

    sxts = []
    for sxtfile in sxtfiles:
        print(sxtfile)
        year, sxt = preprocess(sxtfile, which)
        sxts.append(sxt)

    sxts = np.concatenate(sxts, axis=0)
    if statistic == 'mean':
        sxt_statistic = sxts.mean(axis=0)
    elif statistic == 'var':
        sxt_statistic = sxts.var(axis=0)

elif which in ['SAT', 'SDT']:

    sxt = xarray.open_mfdataset([data_sxt + f for f in sxtfiles],
                                combine='nested', concat_dim='time')

    if season is not None:
        # sxt = sxt.sel(time=sxt['time.season'] == season)
        sxt = sxt.sel(time=season_funcs[season](sxt['time.month']))

    pbar = ProgressBar()
    pbar.register()
    if statistic == 'mean':
        sxt_statistic = sxt.mean('time').compute()
    elif statistic == 'var':
        sxt_statistic = sxt.var('time').compute()

    sxt_statistic = sxt_statistic.variables[nc_var].values
    sxt_statistic = np.roll(sxt_statistic, sxt_statistic.shape[1]//2, axis=1)

    # convert to °C
    if statistic == 'mean':
        sxt_statistic -= 273.15

elif which == 'RH':

    sat_mean = np.load(
        storefolder + 'SAT{}{}_mean.pkl'.format(sstr, tsstr),
        allow_pickle=True)
    sdt_mean = np.load(
        storefolder + 'SDT{}{}_mean.pkl'.format(sstr, tsstr),
        allow_pickle=True)

    # rh = 100 - 5 * (T - T_d)
    # alternative computation (more accurate)
    # https://www.theweatherprediction.com/habyhints/186/
    # https://iridl.ldeo.columbia.edu/dochelp/QA/Basic/dewpoint.html
    T = sat_mean
    T_d = sdt_mean

    T += 273.15
    T_d += 273.15
    E_0 = 0.611
    T_0 = 273.15
    E = E_0 * np.exp(5423 * (1/T_0 - 1/T_d))
    E_s = E_0 * np.exp(5423 * (1/T_0 - 1/T))
    rh = 100 * E/E_s

    sxt_statistic = rh

# store
file = storefolder + f'{which}{sstr}{tsstr}_{statistic}.pkl'
print('storing {} ..'.format(file))
sxt_statistic.dump(file)
