# parts of this script are copied/modified from the climada project
# github: https://github.com/CLIMADA-project/climada_python
# doi: https://doi.org/10.5281/zenodo.4598943

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# filesystem
storefolder = os.getcwd() + '/'
# the following file can be downloaded here
# https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc
ibtracs_file = storefolder + 'IBTrACS.ALL.v04r00.nc'

# parameters
year_range = (1998, 2018)

# constants
SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 137, 1000]
"""Saffir-Simpson Hurricane Wind Scale in kn based on NOAA"""

CAT_NAMES = {
    -1: 'Tropical Depression',
    0: 'Tropical Storm',
    1: 'Hurricane Cat. 1',
    2: 'Hurricane Cat. 2',
    3: 'Hurricane Cat. 3',
    4: 'Hurricane Cat. 4',
    5: 'Hurricane Cat. 5',
}
"""Saffir-Simpson category names."""

IBTRACS_AGENCIES = [
    'wmo', 'usa', 'tokyo', 'newdelhi', 'reunion', 'bom', 'nadi', 'wellington',
    'cma', 'hko', 'ds824', 'td9636', 'td9635', 'neumann', 'mlc',
]
"""Names/IDs of agencies in IBTrACS v4.0"""

IBTRACS_USA_AGENCIES = [
    'atcf', 'cphc', 'hurdat_atl', 'hurdat_epa', 'jtwc_cp', 'jtwc_ep',
    'jtwc_io', 'jtwc_sh', 'jtwc_wp', 'nhc_working_bt', 'tcvightals', 'tcvitals'
]
"""Names/IDs of agencies in IBTrACS that correspond to 'usa_*' variables"""

DEF_ENV_PRESSURE = 1010
"""Default environmental pressure"""

BASIN_ENV_PRESSURE = {
    '': DEF_ENV_PRESSURE,
    'EP': 1010, 'NA': 1010, 'SA': 1010,
    'NI': 1005, 'SI': 1005, 'WP': 1005,
    'SP': 1004,
}
"""Basin-specific default environmental pressure"""


def estimate_roci(roci, cen_pres):
    """Replace missing radius (ROCI) values with statistical estimate.

    In addition to NaNs, negative values and zeros in `roci` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('roci', ['pres'],
    ...                   order=[(872, 950, 985, 1005, 1021)],
    ...                   year_range=(1980, 2019))
    >>> r^2: 0.9148320406675339

    Parameters
    ----------
    roci : array-like
        ROCI values along track in km.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    roci_estimated : np.array
        Estimated ROCI values in km.
    """
    roci = np.where(np.isnan(roci), -1, roci)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    msk = (roci <= 0) & (cen_pres > 0)
    pres_l = [872, 950, 985, 1005, 1021]
    roci_l = [210.711487, 215.897110, 198.261520, 159.589508, 90.900116]
    roci[msk] = 0
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        roci[msk] += roci_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - cen_pres[msk])
                                             - slope_1 * np.fmax(0, cen_pres[msk] - pres_l_i)))
    return np.where(roci <= 0, np.nan, roci)


def estimate_rmw(rmw, cen_pres):
    """Replace missing radius (RMW) values with statistical estimate.

    In addition to NaNs, negative values and zeros in `rmw` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('rmw', ['pres'], order=[(872, 940, 980, 1021)], year_range=(1980, 2019))
    >>> r^2: 0.7905970811843872

    Parameters
    ----------
    rmw : array-like
        RMW values along track in km.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    rmw : np.array
        Estimated RMW values in km.
    """
    rmw = np.where(np.isnan(rmw), -1, rmw)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    msk = (rmw <= 0) & (cen_pres > 0)
    pres_l = [872, 940, 980, 1021]
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]
    rmw[msk] = 0
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        rmw[msk] += rmw_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - cen_pres[msk])
                                           - slope_1 * np.fmax(0, cen_pres[msk] - pres_l_i)))
    return np.where(rmw <= 0, np.nan, rmw)


def _estimate_pressure(cen_pres, lat, lon, v_max):
    """Replace missing pressure values with statistical estimate.

    In addition to NaNs, negative values and zeros in `cen_pres` are
    interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical
    estimation:

    >>> ibtracs_fit_param('pres', ['lat', 'lon', 'wind'], year_range=(1980, 2019))
    >>> r^2: 0.8746154487335112

    Parameters
    ----------
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).
    lat : array-like
        Latitudinal coordinates of eye location.
    lon : array-like
        Longitudinal coordinates of eye location.
    v_max : array-like
        Maximum wind speed along track in knots.

    Returns
    -------
    cen_pres_estimated : np.array
        Estimated central pressure values in hPa (mbar).
    """
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    v_max = np.where(np.isnan(v_max), -1, v_max)
    lat, lon = [np.where(np.isnan(ar), -999, ar) for ar in [lat, lon]]
    msk = (cen_pres <= 0) & (v_max > 0) & (lat > -999) & (lon > -999)
    c_const, c_lat, c_lon, c_vmax = 1024.392, 0.0620, -0.0335, -0.737
    cen_pres[msk] = c_const + c_lat * lat[msk] \
                            + c_lon * lon[msk] \
                            + c_vmax * v_max[msk]
    return np.where(cen_pres <= 0, np.nan, cen_pres)


def _estimate_vmax(v_max, lat, lon, cen_pres):
    """Replace missing wind speed values with a statistical estimate.

    In addition to NaNs, negative values and zeros in `v_max` are interpreted
    as missing values.

    See function `ibtracs_fit_param` for more details about the statistical
    estimation:

    >>> ibtracs_fit_param('wind', ['lat', 'lon', 'pres'], year_range=(1980, 2019))
    >>> r^2: 0.8717153945288457

    Parameters
    ----------
    v_max : array-like
        Maximum wind speed along track in knots.
    lat : array-like
        Latitudinal coordinates of eye location.
    lon : array-like
        Longitudinal coordinates of eye location.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    v_max_estimated : np.array
        Estimated maximum wind speed values in knots.
    """
    v_max = np.where(np.isnan(v_max), -1, v_max)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    lat, lon = [np.where(np.isnan(ar), -999, ar) for ar in [lat, lon]]
    msk = (v_max <= 0) & (cen_pres > 0) & (lat > -999) & (lon > -999)
    c_const, c_lat, c_lon, c_pres = 1216.823, 0.0852, -0.0398, -1.182
    v_max[msk] = c_const + c_lat * lat[msk] \
                         + c_lon * lon[msk] \
                         + c_pres * cen_pres[msk]
    return np.where(v_max <= 0, np.nan, v_max)


def ibtracs_track_agency(ds_sel):
    """Get preferred IBTrACS agency for each entry in the dataset

    Parameters
    ----------
    ds_sel : xarray.Dataset
        Subselection of original IBTrACS NetCDF dataset.

    Returns
    -------
    agency_pref : list of str
        Names of IBTrACS agencies in order of preference.
    track_agency_ix : xarray.DataArray of ints
        For each entry in `ds_sel`, the agency to use, given as an index into
        `agency_pref`.
    """
    agency_pref = IBTRACS_AGENCIES.copy()
    agency_map = {a.encode('utf-8'): i for i, a in enumerate(agency_pref)}
    agency_map.update({
        a.encode('utf-8'): agency_map[b'usa'] for a in IBTRACS_USA_AGENCIES
    })
    agency_map[b''] = agency_map[b'wmo']
    agency_fun = lambda x: agency_map[x]
    track_agency = ds_sel.wmo_agency.where(
        ds_sel.wmo_agency != '', ds_sel.usa_agency)
    track_agency_ix = xr.apply_ufunc(agency_fun, track_agency, vectorize=True)
    return agency_pref, track_agency_ix


def read_ibtracs_netcdf(provider=None, storm_id=None,
                        year_range=None, basin=None, estimate_missing=True,
                        file_name=ibtracs_file):
    """Fill from raw ibtracs v04. Removes nans in coordinates, central
    pressure and removes repeated times data. Fills nans of environmental_pressure
    and radius_max_wind. Checks environmental_pressure > central_pressure.

    Parameters:
        provider (str, optional): If specified, enforce use of specific
            agency, such as "usa", "newdelhi", "bom", "cma", "tokyo".
            Default: None (and automatic choice).
        storm_id (str or list(str), optional): IBTrACS ID of the storm,
            e.g. 1988234N13299, [1988234N13299, 1989260N11316]
        year_range(tuple, optional): (min_year, max_year). Default: (1980, 2018)
        basin (str, optional): e.g. US, SA, NI, SI, SP, WP, EP, NA. if not
            provided, consider all basins.
        estimate_missing (bool, optional): estimate missing central pressure
            wind speed and radius values using other available values.
            Default: False
        file_name (str, optional): name of netcdf file to be dowloaded or located
            at climada/data/system. Default: 'IBTrACS.ALL.v04r00.nc'.
    """

    # create list of tracks
    data = list()

    # open file
    ibtracs_ds = xr.open_dataset(file_name)

    # default: all tracks
    match = np.ones(ibtracs_ds.sid.shape[0], dtype=bool)

    # storm ids?
    if storm_id:
        if not isinstance(storm_id, list):
            storm_id = [storm_id]
        match &= ibtracs_ds.sid.isin([i.encode() for i in storm_id])
        if np.count_nonzero(match) == 0:
            print(f'No tracks with given IDs {storm_id}.')

    # if not storm ids, year range
    else:
        year_range = year_range if year_range else (1980, 2018)

    # select by year range
    if year_range:
        years = ibtracs_ds.sid.str.slice(0, 4).astype(int)
        match &= (years >= year_range[0]) & (years <= year_range[1])
        if np.count_nonzero(match) == 0:
            print('No tracks in time range ({}, {}).'.format(*year_range))

    # select by basin
    if basin:
        match &= (ibtracs_ds.basin == basin.encode()).any(dim='date_time')
        if np.count_nonzero(match) == 0:
            print(f'No tracks in basin {basin}.')

    # no tracks left?
    if np.count_nonzero(match) == 0:
        print('There are no tracks matching the specified requirements.')
        return

    # subset data
    ibtracs_ds = ibtracs_ds.sel(storm=match)

    # select only tracks with valid timestamps
    ibtracs_ds['valid_t'] = ibtracs_ds.time.notnull()
    valid_st = ibtracs_ds.valid_t.any(dim="date_time")
    invalid_st = np.nonzero(~valid_st.data)[0]
    if invalid_st.size > 0:
        st_ids = ', '.join(ibtracs_ds.sid.sel(storm=invalid_st).astype(str).data)
        print(f'No valid timestamps found for {st_ids}.')
        ibtracs_ds = ibtracs_ds.sel(storm=valid_st)

    # find preferred provider
    if not provider:
        agency_pref, track_agency_ix = ibtracs_track_agency(ibtracs_ds)

    # select variables
    for var in ['wind', 'pres', 'rmw', 'poci', 'roci']:
        if provider:
            # enforce use of specified provider's data points
            ibtracs_ds[var] = ibtracs_ds[f'{provider}_{var}']
        else:
            # array of values in order of preference
            cols = [f'{a}_{var}' for a in agency_pref]
            cols = [col for col in cols if col in ibtracs_ds.data_vars.keys()]
            all_vals = ibtracs_ds[cols].to_array(dim='agency')
            preferred_ix = all_vals.notnull().argmax(dim='agency')

            if var in ['wind', 'pres']:
                # choice: wmo -> wmo_agency/usa_agency -> preferred
                ibtracs_ds[var] = ibtracs_ds['wmo_' + var] \
                    .fillna(all_vals.isel(agency=track_agency_ix)) \
                    .fillna(all_vals.isel(agency=preferred_ix))
            else:
                ibtracs_ds[var] = all_vals.isel(agency=preferred_ix)

    ibtracs_ds = ibtracs_ds[[
        'sid', 'name', 'basin', 'lat', 'lon', 'time', 'valid_t', 'wind',
        'pres', 'rmw', 'roci', 'poci'
    ]]

    if estimate_missing:
        ibtracs_ds['pres'][:] = _estimate_pressure(
            ibtracs_ds.pres,
            ibtracs_ds.lat, ibtracs_ds.lon,
            ibtracs_ds.wind)
        ibtracs_ds['wind'][:] = _estimate_vmax(
            ibtracs_ds.wind,
            ibtracs_ds.lat, ibtracs_ds.lon,
            ibtracs_ds.pres)

    # select only tracks with valid wind/pres
    ibtracs_ds['valid_t'] &= ibtracs_ds.wind.notnull() & ibtracs_ds.pres.notnull()
    valid_st = ibtracs_ds.valid_t.any(dim="date_time")
    invalid_st = np.nonzero(~valid_st.data)[0]
    if invalid_st.size > 0:
        st_ids = ', '.join(ibtracs_ds.sid.sel(storm=invalid_st).astype(str).data)
        print(f'No valid wind/pressure values found for {st_ids}.')
        ibtracs_ds = ibtracs_ds.sel(storm=valid_st)

    # max wind
    max_wind = ibtracs_ds.wind.max(dim="date_time").data.ravel()
    category_test = (max_wind[:, None] < np.array(SAFFIR_SIM_CAT)[None])
    category = np.argmax(category_test, axis=1) - 1
    basin_map = {b.encode("utf-8"): v for b, v in BASIN_ENV_PRESSURE.items()}
    basin_fun = lambda b: basin_map[b]

    ibtracs_ds['id_no'] = (ibtracs_ds.sid.str.replace(b'N', b'0')
                           .str.replace(b'S', b'1')
                           .astype(float))
    ibtracs_ds['time_step'] = xr.zeros_like(ibtracs_ds.time, dtype=float)
    ibtracs_ds['time_step'][:, 1:] = (ibtracs_ds.time.diff(dim="date_time")
                                      / np.timedelta64(1, 's'))
    ibtracs_ds['time_step'][:, 0] = ibtracs_ds.time_step[:, 1]
    provider = provider if provider else 'ibtracs'

    last_perc = 0
    all_tracks = []
    for i_track, t_msk in enumerate(ibtracs_ds.valid_t.data):
        perc = 100 * len(all_tracks) / ibtracs_ds.sid.size
        if perc - last_perc >= 10:
            print(f"Progress: {perc}%")
            last_perc = perc
        track_ds = ibtracs_ds.sel(storm=i_track, date_time=t_msk)
        st_penv = xr.apply_ufunc(basin_fun, track_ds.basin, vectorize=True)
        track_ds['time'][:1] = track_ds.time[:1].dt.floor('H')
        if track_ds.time.size > 1:
            track_ds['time_step'][0] = (track_ds.time[1] - track_ds.time[0]) \
                                  / np.timedelta64(1, 's')

        with warnings.catch_warnings():
            # See https://github.com/pydata/xarray/issues/4167
            warnings.simplefilter(action="ignore", category=FutureWarning)

            track_ds['rmw'] = track_ds.rmw \
                .ffill(dim='date_time', limit=1) \
                .bfill(dim='date_time', limit=1) \
                .fillna(0)
            track_ds['roci'] = track_ds.roci \
                .ffill(dim='date_time', limit=1) \
                .bfill(dim='date_time', limit=1) \
                .fillna(0)
            track_ds['poci'] = track_ds.poci \
                .ffill(dim='date_time', limit=4) \
                .bfill(dim='date_time', limit=4)
            # this is the most time consuming line in the processing:
            track_ds['poci'] = track_ds.poci.fillna(st_penv)

        if estimate_missing:
            track_ds['rmw'][:] = estimate_rmw(track_ds.rmw.values, track_ds.pres.values)
            track_ds['roci'][:] = estimate_roci(track_ds.roci.values, track_ds.rmw.values)
            track_ds['roci'][:] = np.fmax(track_ds.rmw.values, track_ds.roci.values)

        # ensure environmental pressure >= central pressure
        # this is the second most time consuming line in the processing:
        track_ds['poci'][:] = np.fmax(track_ds.poci, track_ds.pres)

        all_tracks.append(xr.Dataset({
            'time_step': ('time', track_ds.time_step),
            'radius_max_wind': ('time', track_ds.rmw.data),
            'radius_oci': ('time', track_ds.roci.data),
            'max_sustained_wind': ('time', track_ds.wind.data),
            'central_pressure': ('time', track_ds.pres.data),
            'environmental_pressure': ('time', track_ds.poci.data),
        }, coords={
            'time': track_ds.time.dt.round('s').data,
            'lat': ('time', track_ds.lat.data),
            'lon': ('time', track_ds.lon.data),
        }, attrs={
            'max_sustained_wind_unit': 'kn',
            'central_pressure_unit': 'mb',
            'name': track_ds.name.astype(str).item(),
            'sid': track_ds.sid.astype(str).item(),
            'orig_event_flag': True,
            'data_provider': provider,
            'basin': track_ds.basin.values[0].astype(str).item(),
            'id_no': track_ds.id_no.item(),
            'category': category[i_track],
        }))

    data = all_tracks

    return data


# parameters
ystart = 1998
yend = 2018

# extract storm tracks using climada
data = read_ibtracs_netcdf(year_range=(ystart, yend))

vs = np.zeros(len(data), dtype=object)
cats = np.zeros(len(data), dtype=int)
maxsws = np.zeros(len(data))
for i, track in enumerate(data):

    vt = track.to_dataframe()
    vt.reset_index(inplace=True)
    vt = vt[['time', 'lat', 'lon', 'radius_max_wind', 'radius_oci',
             'max_sustained_wind', 'central_pressure',
             'environmental_pressure']]
    vt['id'] = i
    vt['category'] = track.attrs['category']

    vs[i] = vt
    cats[i] = track.attrs['category']
    maxsws[i] = vt['max_sustained_wind'].max()

# id by category (0 weakest -> len(tctracks.data) strongest)
index = maxsws.argsort()
for i, ind in enumerate(index):
    vs[ind]['id'] = i
vs = vs[index]

# concat
v = pd.concat(vs, axis=0)

# create seasons
month = v['time'].dt.month
v['JJA'] = np.zeros(len(v), dtype=bool)
v['DJF'] = np.zeros(len(v), dtype=bool)
v.loc[month.isin([6, 7, 8]), 'JJA'] = True
v.loc[month.isin([12, 1, 2]), 'DJF'] = True

# correct longitudes >= 180
v.loc[v['lon'] > 180, 'lon'] -= 360

# store as dataframe
filename = storefolder + 'tctracks_{}_{}.pickle'.format(ystart, yend)
print(filename)
v.to_pickle(filename)
