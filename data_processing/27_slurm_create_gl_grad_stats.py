#!/raid-manaslu/k2-raid/traxl/anaconda3/envs/py3/bin/python

import os
import argparse
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
    '-fl', '--file-locations')
parser.add_argument(
    '-l', '--locations',
    nargs='+',
    help='list of locations to compute bursts for',
    required=False)
parser.add_argument(
    '-cg', '--coarse-grained',
    type=int)
parser.add_argument(
    "-tcs", '--only-tropical-cyclones', action='store_true',
    help='consider only rainfall events tagged as part of a tropical cyclones')
args = parser.parse_args()

# tropical cyclone folder string
if args.only_tropical_cyclones:
    tcfstr = '_tc'
else:
    tcfstr = ''

# parameters
r = .1
p = 0
buffer = 10  # tc track degree buffer
cg_N = args.coarse_grained

if cg_N is not None:
    if cg_N == 2:
        n_bins = 20
    if cg_N == 4:
        n_bins = 30
    if cg_N == 8:
        n_bins = 40
else:
    n_bins = 15

# variables
twcols = ['s_{}_-2'.format(tf) for tf in range(-24, -2, 1)]
# twcols = ['s_-24_-12']
hod = 'all'
varis = ['r_max', 'r_mean']  # ['r_mean', 'r_max']
seasons = ['all', 'JASO', 'DJFMA']  # ['all', 'DJF', 'JJA']
wo_r_befores = [False, True]

# filesystem folders
storefolder = os.getcwd() + '/'
partsfolder = 'glcp_burst_parts/'
spartsfolder = 'glcp_burst_sat_grad_parts/'
# spartsfolder = 'glcp_burst_sat_grad_parts_24_12/'
tc_glcp_store = storefolder + f'glcp_r{r}_p{p}_tcs_{buffer}_degrees.h5'
if cg_N is not None:
    glpartsfolder = f'gl_cg_{cg_N}_grad_stats{tcfstr}_parts/'
else:
    glpartsfolder = f'gl_grad_stats{tcfstr}_parts/'
os.makedirs(storefolder + glpartsfolder, exist_ok=True)

# load data
if cg_N is not None:
    gl_cg = pd.read_feather(
        storefolder + 'gl_r{}_p{}_cg_{}.feather'.format(r, p, cg_N))


def _load_bursts(locs, folder):

    # load burst data
    vs = []
    for loc in locs:
        try:
            v = pd.read_pickle(
                storefolder + folder + '{:06d}.pickle'.format(loc))
            vs.append(v)
        except FileNotFoundError:
            pass
    try:
        v = pd.concat(vs, axis=0)
    except ValueError:
        v = []

    return v


def _grad_vs_r_subset_v(v, col, season, hod, wo_r_before, var):

    # col, get rid of nans
    vt = v[~v[col].isnull()]
    colstr = col

    # season
    if season == 'all':
        sstr = 'all'
    else:
        vt = vt.loc[vt['season', season]]
        sstr = season

    # hod
    if hod == 'all':
        hodstr = 'all'
    else:
        vt = vt.loc[vt['dtime'].dt.hour == hod]
        hodstr = '{:02d}'.format(hod)

    # rain before?
    if wo_r_before:
        for tdr in range(-6, -51, -3):
            try:
                vt = vt.loc[vt['r', tdr].isnull()]
            except KeyError:
                pass
        rbstr = 'worb'
    elif not wo_r_before:
        rbstr = 'wrb'

    # r_max or r_mean?
    varstr = var

    # cut off left and right ends?
    # grad_first_q = .03
    # grad_last_q = .97
    # xfirst = vt[col].quantile(grad_first_q)
    # xlast = vt[col].quantile(grad_last_q)
    # vt = vt[(vt[col] >= xfirst) & (vt[col] <= xlast)]

    return vt, colstr, sstr, hodstr, rbstr, varstr


def compute_grad_stats(v, col, season, hod, wo_r_before, var):

    # subset
    vt, colstr, sstr, hodstr, rbstr, varstr = _grad_vs_r_subset_v(
        v, col, season, hod, wo_r_before, var)

    # return empty series if there are no sxt values (think sst)
    if len(vt) <= n_bins:
        return {}

    # compute stats
    data = {}

    data['mean'] = vt[col].mean()
    data['median'] = vt[col].median()
    data['left'] = len(vt.loc[vt[col] <= 0]) / len(vt) * 100

    # pts per location
    data['n_bursts'] = len(vt)

    # add subset strings
    data = {
        '{}_{}_{}_{}_{}_'.format(colstr, sstr, hodstr, rbstr, varstr) + key:
        value for key, value in data.items()
    }

    return data


def create_gl_part(loc):

    tstart = datetime.now()

    fname = storefolder + glpartsfolder + '{:06d}.pickle'.format(loc)

    print('running {} {}'.format(loc, datetime.now()))

    # load data
    if cg_N is not None:
        lcols = []
        for i in range(cg_N):
            for j in range(cg_N):
                lcols.append('l{}{}'.format(i, j))
        locs = gl_cg.loc[loc, lcols].astype(int).values
    else:
        locs = [loc]

    # load bursts
    vb = _load_bursts(locs, partsfolder)

    # store empty dataframe
    if len(vb) <= n_bins:
        pd.DataFrame(index=[loc]).to_pickle(fname)
        return

    # load sat grads
    vbg = _load_bursts(locs, spartsfolder)
    cols = vbg.columns.values
    vbg.columns = pd.MultiIndex.from_tuples([[col, ''] for col in cols])
    vb = pd.concat((vb, vbg), axis=1)

    # add tropical cyclone flag - keep only tc bursts
    if args.only_tropical_cyclones:
        glcp_tc = pd.read_hdf(tc_glcp_store, where='l in locs')
        glcp_tc.set_index(['l', 'cp_burst'], inplace=True)
        glcp_tc.columns = pd.MultiIndex.from_tuples([['tc', '']])
        vb = pd.merge(vb, glcp_tc, how='left',
                      left_on=['l', 'cp_burst'], right_index=True)
        vb = vb.loc[~vb['tc'].isnull()]
        del glcp_tc
        # add 'all' season
        vb[('season', 'all')] = True

    # add additional seasons (JASO/DJFMA)
    month = vb['dtime'].dt.month
    vb[('season', 'JASO')] = False
    vb[('season', 'DJFMA')] = False
    vb.loc[month.isin([7, 8, 9, 10]), ('season', 'JASO')] = True
    vb.loc[month.isin([12, 1, 2, 3, 4]), ('season', 'DJFMA')] = True
    del month

    # choose subsets, create data dictionary
    data = {}
    for col in twcols:
        for wo_r_before in wo_r_befores:
            for season in seasons:
                for vari in varis:
                    datai = compute_grad_stats(
                        vb, col=col, season=season, hod=hod,
                        wo_r_before=wo_r_before, var=vari
                    )
                    data.update(datai)

    # create dataframe
    glp = pd.DataFrame(index=[loc], data=data)

    # store
    glp.to_pickle(fname)

    # performance
    dt = datetime.now() - tstart
    ptime = 'comp. time: s={}\t ms={}'.format(
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)


if args.file_locations is not None:
    locs = np.load(storefolder + 'tmp_grad_vs_r_repair_locations.npy')
elif args.locations is not None:
    locs = args.locations

for loc in locs:
    loc = int(loc)
    create_gl_part(loc)
