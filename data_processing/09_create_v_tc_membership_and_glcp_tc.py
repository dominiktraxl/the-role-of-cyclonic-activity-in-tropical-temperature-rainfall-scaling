import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

# parameters
buffer = 10  # in degrees lon/lat
r = 0.1
p = 0

# filesystem
storefolder = os.getcwd() + '/'

# load TCs
tcs = pd.read_pickle(storefolder + 'tctracks_1998_2018.pickle')
tcs['n_rainfall_events'] = -1

# load rainfall events
s = pd.HDFStore(storefolder + f'v_r{r}_p{p}_dtime_sortedc7.h5', mode='r')

# load cp_burst membership labels
print('loading v_cp_burst ..')
v_cp_burst = pd.read_hdf(storefolder + f'v_r{r}_p{p}_bursts_l_sortedc7.h5',
                         columns=['cp_burst'])

# select ids to track
tc_ids = tcs['id'].unique()

# run
vts = []
time_mismatches = []
c = 0
for tc_id in tc_ids:

    # select track
    tc = tcs.loc[tcs['id'] == tc_id]

    # check if there's all 3-hourly data points
    time_min = tc['time'].min()
    tmod = time_min.hour % 3
    if tmod != 0:
        time_min += timedelta(hours=3-tmod)
    drange = pd.date_range(time_min, tc['time'].max(), freq='3H')
    if tc['time'].isin(drange).sum() != len(drange):
        time_mismatches.append(tc['id'].values[0])
        assert tc['time'].diff().max() >= timedelta(hours=4)
    # assert tc['time'].isin(drange).sum() == len(drange)

    # consider only times we have trmm data
    tc = tc.loc[tc['time'].isin(drange)]

    # select v time slice
    v = s.select(
        'v', where='(dtime >= {}) & (dtime <= {})'.format(
            tc['time'].min().__repr__(), tc['time'].max().__repr__()),
        # columns=['lat', 'lon', 'dtime'],
    )

    # select v in box
    for i, (_, tct) in enumerate(tc.iterrows()):

        print('processing {:06d}/{:06d} | tc_id {:04d}, {:03d}/{:03d}'.format(
            c, len(tcs), tc_id, i, len(tc)))

        # box coordinates
        lon_left = tct['lon'] - buffer
        lon_right = tct['lon'] + buffer
        lat_bottom = tct['lat'] - buffer
        lat_top = tct['lat'] + buffer

        # wrapping
        if lon_right >= 180:
            lons = (v['lon'] >= lon_left) | (v['lon'] <= lon_right - 360)
        elif lon_left <= -180:
            lons = (v['lon'] <= lon_right) | (v['lon'] >= lon_left + 360)
        else:
            lons = (v['lon'] >= lon_left) & (v['lon'] <= lon_right)

        # subset
        vt = v.loc[lons & (v['dtime'] == tct['time']) &
                   (v['lat'] >= lat_bottom) & (v['lat'] <= lat_top)]

        # keep track of n_events per frame
        # tcs.loc[
        #     (tcs['time'] == tct['time']) &
        #     (tcs['id'] == tct['id']), 'n_rainfall_events'] = len(vt)

        # id and distance to eye
        if len(vt) == 0:
            print('EMPTY DATAFRAME')
        else:
            # store tc variables
            vt.loc[:, 'id'] = tct['id']
            vt.loc[:, 'category'] = tct['category']
            vt.loc[:, 'max_sustained_wind'] = tct['max_sustained_wind']
            vt.loc[:, 'radius_max_wind'] = tct['radius_max_wind']
            vt.loc[:, 'radius_oci'] = tct['radius_oci']

            # compute great circle distances
            X = np.radians(vt[['lat', 'lon']].values)
            Y = np.radians(np.array([[tct['lat'], tct['lon']]]))
            d = haversine_distances(X, Y) * 6371
            vt.loc[:, 'dist_to_eye'] = d

            # append
            vts.append(vt)

        c += 1

# close store
s.close()

# concat
v = pd.concat(vts)
tms = np.asarray(time_mismatches)

# add cp_burst column
# sort v by index (faster)
v.sort_index(inplace=True)

# sorty v_cp_burst by index (faster)
v_cp_burst.sort_index(inplace=True)

# set as range index (faster?)
v_cp_burst.index = range(len(v_cp_burst))

# set cp_burst column
v['cp_burst'] = v_cp_burst.loc[v.index, 'cp_burst']

# dtypes
v['id'] = v['id'].astype(int)
v['category'] = v['category'].astype(int)

# create glcp table
# things that can happen:
#  - events are related to multiple tc time slices (different tracks as well!)
#  - multiple events in a burst have the same minimum distance
glcp = v.sort_values('dist_to_eye').drop_duplicates(
    subset=['l', 'cp_burst'], keep='first')
glcp.sort_values(['l', 'cp_burst'], inplace=True)
glcp = glcp[['l', 'cp_burst', 'id', 'category', 'max_sustained_wind',
             'radius_max_wind', 'radius_oci',
             'dist_to_eye', 'r']]
glcp['l'] = glcp['l'].astype('int')

# store glcp as hdf
fname = storefolder + f'glcp_r{r}_p{p}_tcs_{buffer}_degrees.h5'
store = pd.HDFStore(fname, mode='w')
store.append('glcp', glcp, format='t', data_columns=True, index=False)
store.create_table_index('glcp', columns=['l'], kind='full')
store.close()
print(f'stored {fname}')

# store v
fname = storefolder + f'v_r{r}_p{p}_tcs_{buffer}_degrees.pickle'
print(f'storing {fname}')
v.to_pickle(fname)

# store array of temporal mismatches
print('storing {}_time_mismatches.npy'.format(fname[:-7]))
np.save(fname[:-7] + '_time_mismatches', tms)
