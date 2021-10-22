#!/raid-manaslu/k2-raid/traxl/anaconda3/envs/py3/bin/python

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import linregress

# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-l', '--locations',
    nargs='+',
    help='list of locations to compute bursts for',
    required=True)
args = parser.parse_args()

# parameters
tfs = np.arange(-24, -2, 1, dtype=int)
tls = np.ones(tfs.shape, dtype=int) * -2

# filesystem folders
storefolder = os.getcwd() + '/'
partsfolder = 'glcp_burst_parts/'
spartsfolder = 'glcp_burst_sat_grad_parts/'


def compute_sat_grad(loc):

    print('running {} {}'.format(loc[:loc.find('.')], datetime.now()))

    # computed gradients
#     clocs = os.listdir(storefolder + spartsfolder)
#
#     if loc in clocs:
#         print('{} already computed!'.format(loc))
#         return

    # load burst data
    v = pd.read_pickle(storefolder + partsfolder + loc)

    if ('sat', -47) not in v.columns:
        print('{} missing temps!'.format(loc))
        raise ValueError('burst {} missing temp!'.format(loc))

    timer_1 = datetime.now()

    # select sats
    vsat = v['sat']
    vsat.columns = vsat.columns.astype(int)

    # collect data in dict
    data = {}
    for tf, tl in zip(tfs, tls):
        col = 's_{}_{}'.format(tf, tl)
        data[col] = []

    n_bursts = len(vsat)
    print('n_bursts {:05d}'.format(n_bursts))
    c = 0
    for _, row in vsat.iterrows():

        # reshape
        tds = list(range(-48, -1))
        vp = pd.DataFrame(index=tds)
        vp.loc[row.loc[:max(tls)].index, 'sat'] = row.loc[-48:max(tls)].values

        # rolling 24h window
        vp['sat_24_mean'] = vp['sat'].rolling(24).mean()

        for tf, tl in zip(tfs, tls):
            col = 's_{}_{}'.format(tf, tl)

            # linregress
            x = vp.loc[tf:tl].index.values
            y = vp.loc[tf:tl, 'sat_24_mean'].values
            m, _, _, _, _ = linregress(x, y)

            data[col].append(m)

        c += 1

    grad = pd.DataFrame(index=v.index, data=data)

    # store
    fname = storefolder + spartsfolder + loc
    grad.to_pickle(fname)

    dt_1 = datetime.now() - timer_1
    ptime = 'comp. time: s={}\t ms={}'.format(
        int(dt_1.total_seconds()),
        str(dt_1.microseconds / 1000.)[:6])
    print(ptime)


for location in args.locations:
    location = int(location)
    location = '{:06d}.pickle'.format(location)
    compute_sat_grad(location)
