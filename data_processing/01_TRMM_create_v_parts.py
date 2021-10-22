import os

from multiprocessing import Pool
import datetime

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

# filesystem folders
storefolder = os.getcwd() + '/'
v_parts_folder = 'v_parts/'
os.makedirs(storefolder + v_parts_folder, exist_ok=True)

# original trmm files
trmm_files = os.listdir(storefolder + 'TRMM/')
trmm_files.sort()

# check if all files are there
years = np.arange(1998, 2019, 1)
for year in years:
    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year + 1, 1, 1)
    nods = (d2 - d1).days
    year = str(year)
    files = []
    for f in trmm_files:
        if f.startswith('3B42.{}'.format(year)):
            files.append(f)
    print(year, len(files))
    assert(len(files) == nods*8)

# create time<->datetime dic
dates = pd.date_range('1998-1-1', '2019-1-1', freq='3H')
dtdic = {date: i for i, date in enumerate(dates)}

# create grid
xx, yy = np.meshgrid(np.arange(1440), np.arange(399, -1, -1))
lons, lats = np.meshgrid(np.arange(-179.875, 180.125, .25),
                         np.arange(49.875, -50.125, -.25))

# labels for geographical locations
locs = np.arange(xx.shape[0] * xx.shape[1]).reshape(xx.shape)
# compute spatial coverage for each geographical location
areas = 111.2**2 * .25**2 * np.cos(2*np.pi*lats / 360.)


def create_dataframe(year):

    # get files of specified year
    files = []
    for f in trmm_files:
        if f.startswith('3B42.{}'.format(year)):
            files.append(f)

    # iterate through 3-hourly frames
    vts = []
    for i, file in enumerate(files):

        date, hour = file.split('.')[1:3]
        year, month, day = date[:4], date[4:6], date[6:]
        dtime = pd.datetime(int(year), int(month), int(day), int(hour))

        print(storefolder + 'TRMM/' + '{}'.format(file))
        # f = xarray.open_dataset(pio.DATA + '{}'.format(file))
        # r = f.precipitation.values.T[::-1]
        f = SD(storefolder + 'TRMM/' + file, SDC.READ)
        r = f.select('precipitation')[:].T[::-1]

        # store only values r>=0
        x, y = xx[r > 0], yy[r > 0]
        lon, lat = lons[r > 0], lats[r > 0]
        loc = locs[r > 0]
        area = areas[r > 0]
        r = r[r > 0]
        vol = r * 3 * area * 1e-6

        # create dataframe
        vt = pd.DataFrame({'x': x, 'y': y, 'lon': lon, 'lat': lat, 'l': loc,
                           'r': r, 'area': area, 'vol': vol})
        time = dtdic[dtime]
        vt['time'] = time
        vt['dtime'] = dtime

        # dtypes
        vt['area'] = vt['area'].astype(np.uint16)
        vt['l'] = vt['l'].astype(np.uint32)
        vt['lat'] = vt['lat'].astype(np.float32)
        vt['lon'] = vt['lon'].astype(np.float32)
        vt['x'] = vt['x'].astype(np.uint16)
        vt['y'] = vt['y'].astype(np.uint16)
        vt['time'] = vt['time'].astype(np.uint16)

        # append to list
        vts.append(vt)

    # combine
    v = pd.concat(vts)

    # set range index
    v.index = range(len(v))

    # store data table as hdf5
    v.to_feather(storefolder + v_parts_folder + '{}.feather'.format(year))
    print('stored {}.feather ...'.format(year))


if __name__ == '__main__':
    years = np.arange(1998, 2019)
    p = Pool()
    for _ in p.imap_unordered(create_dataframe, years):
        pass
