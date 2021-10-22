#!/raid-manaslu/k2-raid/traxl/anaconda3/envs/py3/bin/python

import os
import argparse
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

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
r = .1
p = 0

seasons = ['JJA', 'DJF', 'MJJAS', 'NDJFM']
tds = list(range(0, 49))
tds = tds[::-1]

# filesystem folders
storefolder = os.getcwd() + '/'
subfolder = 'v_r{}_p{}_SXT_td_X/'.format(r, p)
v_table = 'v_r{}_p{}_dtime_l_r_seasons_l_sortedc7.h5'.format(r, p)
v_burst_table = 'v_r{}_p{}_bursts_l_sortedc7.h5'.format(r, p)
partsfolder = 'glcp_burst_parts/'
os.makedirs(storefolder + partsfolder, exist_ok=True)


def compute_burst(location):

    print('running {} {}'.format(location, datetime.now()))

    timer_1 = datetime.now()

    # load dtime, l, r, seasons
    v = pd.read_hdf(
        storefolder + v_table, 'v',
        where='l == {}'.format(location),
        # columns=['dtime', 'l', 'r'],
    )

    # load burst
    v['cp_burst'] = pd.read_hdf(
        storefolder + v_burst_table, 'v',
        where='l == {}'.format(location),
        columns=['cp_burst'])

    # load sxt_td_x
    for which in ['sat', 'sdt']:
        for td in tds:
            v_sxt_table = '{}_td_{}_l_sortedc7.h5'.format(which, td)
            v['{}_td_{}'.format(which, td)] = pd.read_hdf(
                storefolder + subfolder + v_sxt_table, 'v',
                where='l == {}'.format(location),
                columns=['{}_td_{}'.format(which, td)])

    gv = v.groupby(['l', 'cp_burst'])
    burst_parts = []

    # compute bursts dataframe
    n_groups = len(gv)
    print('n_bursts {:05d}'.format(n_groups))
    c = 1
    for name, vt in gv:

        vt0 = vt.iloc[0]

        # measure performance
        # time_start = datetime.now()

        # column index tuples, values
        cindex = []
        values = []

        # l, cp_burst
        cindex.append(['l', ''])
        values.append(name[0])
        cindex.append(['cp_burst', ''])
        values.append(name[1])

        # dtime
        cindex.append(['dtime', ''])
        values.append(vt0['dtime'])

        # season
        for season in seasons:
            cindex.append(['season', season])
            values.append(vt0[season])

        # k-burst
        cindex.append(['k_burst', ''])
        values.append(len(vt))

        # mean r
        cindex.append(['r_mean', ''])
        values.append(vt['r'].mean())

        # max r
        cindex.append(['r_max', ''])
        values.append(vt['r'].max())

        # rainfall before the burst being analyzed
        vlb = v.loc[
            (v['dtime'] >= vt0['dtime'] - pd.Timedelta(max(tds), 'h')) &
            (v['dtime'] < vt0['dtime']), ['dtime', 'r']]
        vlb['td'] = (vlb['dtime'] - vt0['dtime'])
        vlb['td'] = vlb['td'].astype('timedelta64[h]')

        # rainfall rates
        vt['td'] = np.arange(0, len(vt)*3, 3)
        rr = pd.concat((vlb[['td', 'r']], vt[['td', 'r']]))

        cindex += (list(itertools.product(['r'], rr['td'].astype(np.int64))))
        values += rr['r'].values.tolist()

        # time reference for sat/sdt/RH
        t_ref = [-td for td in tds] + list(range(1, (len(vt) - 1) * 3 + 1))

        # sat
        sat_t0 = vt[['sat_td_{}'.format(td) for td in tds]].iloc[0].values
        sat_tplus = vt[
            ['sat_td_{}'.format(td) for td in [2, 1, 0]]
        ].iloc[1:].values.flatten()
        sat = np.concatenate((sat_t0, sat_tplus))

        cindex += (list(itertools.product(['sat'], t_ref)))
        values += sat.tolist()

        # sdt
        sdt_t0 = vt[['sdt_td_{}'.format(td) for td in tds]].iloc[0].values
        sdt_tplus = vt[
            ['sdt_td_{}'.format(td) for td in [2, 1, 0]]
        ].iloc[1:].values.flatten()
        sdt = np.concatenate((sdt_t0, sdt_tplus))

        cindex += (list(itertools.product(['sdt'], t_ref)))
        values += sdt.tolist()

        # RH
        # https://www.theweatherprediction.com/habyhints/186/
        # https://iridl.ldeo.columbia.edu/dochelp/QA/Basic/dewpoint.html
        T = sat
        T_d = sdt
        T += 273.15
        T_d += 273.15
        E_0 = 0.611
        T_0 = 273.15
        E = E_0 * np.exp(5423 * (1/T_0 - 1/T_d))
        E_s = E_0 * np.exp(5423 * (1/T_0 - 1/T))
        rh = 100 * E/E_s

        cindex += (list(itertools.product(['RH'], t_ref)))
        values += rh.tolist()

        # create dataframe
        burst = pd.DataFrame(values).T
        burst.columns = pd.MultiIndex.from_tuples(cindex)

        # set dtypes
        exclusion_list = ['l', 'cp_burst', 'k_burst', 'dtime', 'season']

        burst['l'] = burst['l'].astype(np.uint32)
        burst['cp_burst'] = burst['cp_burst'].astype(np.uint16)
        burst['k_burst'] = burst['k_burst'].astype(np.uint8)
        for season in seasons:
            burst['season', season] = burst['season', season].astype(bool)

        for col in burst:
            if not col[0] in exclusion_list:
                burst[col] = burst[col].astype(np.float32)

        # append
        burst_parts.append(burst)

        # print progress
#         timediff = datetime.now() - time_start
#         pprogress = '{:05d} / {:05d}'.format(c, n_groups)
#         ptime = ' | comp. time: s={}\t ms={}'.format(
#             int(timediff.total_seconds()),
#             str(timediff.microseconds / 1000.)[:6])
#         print(pprogress + ptime)

        c += 1

    v_burst = pd.concat(burst_parts, axis=0, sort=False)

    # set index
    v_burst.index = range(len(v_burst))

    # store as pickle
    fname = '{:06d}.pickle'.format(location)
    v_burst.to_pickle(storefolder + partsfolder + fname)

    # timer
    dt_1 = datetime.now() - timer_1
    ptime = 'comp. time s={}\t ms={}'.format(
        int(dt_1.total_seconds()),
        str(dt_1.microseconds / 1000.)[:6])
    print(ptime)


for location in args.locations:
    location = int(location)
    compute_burst(location)
