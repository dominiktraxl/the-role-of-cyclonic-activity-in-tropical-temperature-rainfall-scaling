#!/raid-manaslu/k2-raid/traxl/anaconda3/envs/py3/bin/python

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import statsmodels.api as sm
from scipy.interpolate import interp1d

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
sxt_first_q = .03
sxt_last_q = .97
interp1d_bins = 50

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

# variables all bursts
if not args.only_tropical_cyclones:
    sxt_cols = ['sat']  # ['sat', 'sdt']
    tds = [2, 24]
    twcol = 's_-6_-2'
    hod = 'all'
    varis = ['r_mean', 'r_max']
    seasons = ['JASO', 'DJFMA']  # ['DJF', 'JJA']
    wo_r_befores = [False, True]
    grads = ['agrad']  # ['ngrad', 'pgrad', 'agrad']
    rh_ranges = ['worhs']  # ['worhs', 'wrhmean', 'wrh80', 'wrrhmean', 'wrrh80']
    tcs = [False]  # [False, True]
    qs = [90, 95, 99]

# variables only tc bursts
elif args.only_tropical_cyclones:
    sxt_cols = ['sat']
    tds = [2, 24]
    twcol = 's_-6_-2'
    hod = 'all'
    varis = ['r_max', 'r_mean']
    seasons = ['all', 'JASO', 'DJFMA']
    wo_r_befores = [False, True]
    grads = ['agrad']  # ['ngrad', 'pgrad', 'agrad']
    rh_ranges = ['worhs']  # ['worhs', 'wrhmean', 'wrh80', 'wrrhmean', 'wrrh80']
    tcs = [True]  # [False, True]
    qs = [90]

# filesystem folders
storefolder = os.getcwd() + '/'
partsfolder = 'glcp_burst_parts/'
spartsfolder = 'glcp_burst_sat_grad_parts/'
tc_glcp_store = storefolder + f'glcp_r{r}_p{p}_tcs_{buffer}_degrees.h5'
if cg_N is not None:
    glpartsfolder = f'gl_cg_{cg_N}_sxt_vs_r{tcfstr}_parts/'
else:
    glpartsfolder = f'gl_sxt_vs_r{tcfstr}_parts/'
os.makedirs(storefolder + glpartsfolder, exist_ok=True)

# load data
if cg_N is not None:
    gl_cg = pd.read_feather(
        storefolder + f'gl_r{r}_p{p}_cg_{cg_N}.feather')


def exp_fit(x, a, b):
    return a * np.exp(b * x)


def gen_log_fit(x, a, b, c, d):
    return 1 / (c + a*np.exp((-b*x))) + d


def gen_log_jac(x, a, b, c, d):  # @UnusedVariable
    jac = np.array([-(c + a*np.exp(-b*x))**(-2) * np.exp(-b*x),
                    -(c + a*np.exp(-b*x))**(-2) * a*np.exp(-b*x)*(-x),
                    -(c + a*np.exp(-b*x))**(-2),
                    np.ones(x.shape[0])]).T
    return jac


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


def _sxt_vs_r_fits_per_loc(loc, v, td, var, binned):

    xlin = np.linspace(v[-td].min(), v[-td].max(), interp1d_bins)

    # --------------------------------------------------------------------
    # lowess
    lowess = sm.nonparametric.lowess(np.log(v[var]), v[-td], frac=.7)

    # interpolation
    if not binned:
        f = interp1d(
            lowess[:, 0], np.exp(lowess[:, 1]), bounds_error=False)
        xl = xlin
        yl = f(xlin)

    if binned:
        xl = lowess[:, 0]
        yl = np.exp(lowess[:, 1])

    # lowess alpha distribution -> mean, std
    alphas = (yl[:-1] + (yl[1:]-yl[:-1])/(xl[1:]-xl[:-1])) \
        / yl[:-1]
    alphas = (alphas - 1) * 100
    alphas_mean = alphas.mean()
    alphas_std = alphas.std()

    # --------------------------------------------------------------------
    # monotonicity
    dyl = np.diff(yl)
    dyl_mean = dyl.mean()

    if (dyl <= 0).all():
        monotonicity = 'decreasing'
        ppt = np.nan
    elif (dyl >= 0).all():
        monotonicity = 'increasing'
        ppt = np.nan
    elif dyl_mean > 0:
        monotonicity = 'mean increase (ppt)'
        ppt = xl[yl.argmax()]
    elif dyl_mean < 0:
        monotonicity = 'mean decrease'
        ppt = np.nan
    else:
        monotonicity = 'none'
        ppt = np.nan

    # mean increase -> increasing if ppt is last point of lowess
    if ppt == xl[-1]:
        # print('mean increase -> mean increase (ppt last point)')
        monotonicity = 'mean increase (ppt last point)'
        ppt = np.nan

    # --------------------------------------------------------------------
    # exp fit linregress
    try:
        if monotonicity in ['decreasing', 'increasing', 'mean decrease']:

            A, B, rvalue, pvalue, stderr = linregress(
                v[-td], np.log(v[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        elif monotonicity == 'mean increase (ppt last point)':

            global_min_T = xl[yl.argmin()]
            vt_min_T = v[v[-td] >= global_min_T]

            A, B, rvalue, pvalue, stderr = linregress(
                vt_min_T[-td], np.log(vt_min_T[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        elif monotonicity == 'mean increase (ppt)':

            global_min_T = xl[yl.argmin()]
            global_max_T = xl[yl.argmax()]

            # if ppt left of global minimum, fit all
            if global_min_T >= global_max_T:
                vt_min_max_T = v
            else:
                vt_min_max_T = v[
                    (v[-td] >= global_min_T) &
                    (v[-td] <= global_max_T)]

            A, B, rvalue, pvalue, stderr = linregress(
                vt_min_max_T[-td], np.log(vt_min_max_T[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        else:

            A, B, rvalue, pvalue, stderr = linregress(
                v[-td], np.log(v[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

    except Exception as e:
        msg = "loc {} \t linregress \t binned: {} \t m: {} \t {}\n".format(
            loc, binned, monotonicity, str(e))
        print(msg)
        a_lin, b_lin, alpha, rvalue, pvalue, stderr = [np.nan]*6

    # --------------------------------------------------------------------
    # generalzed logistic
    try:
        if monotonicity in ['mean increase (ppt)',
                            'decreasing', 'mean decrease']:
            a, b, c, d, asymp, rmax, saturation = [np.nan] * 7

        elif monotonicity == 'increasing':

            popt, _ = curve_fit(
                gen_log_fit,
                # v[-td], v[var],
                xl, yl,
                p0=[1, .1, .1, 0],
                jac=gen_log_jac,
                bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
                max_nfev=1000*len(xl),
                # verbose=2,
            )
            a, b, c, d = popt
            asymp = 1/c + d
            rmax = gen_log_fit(xlin[-1], *popt)
            saturation = rmax/asymp * 100

        elif monotonicity in ['mean increase (ppt last point)']:

            global_min_T = xl[yl.argmin()]
            vt_min_T = v[v[-td] >= global_min_T]

            popt, _ = curve_fit(
                gen_log_fit,
                # vt_min_T[-td], vt_min_T[var],
                xl[yl.argmin():], yl[yl.argmin():],
                p0=[1, .1, .1, 0],
                jac=gen_log_jac,
                bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
                max_nfev=1000*len(xl),
                # verbose=2,
            )
            a, b, c, d = popt
            asymp = 1/c + d
            rmax = gen_log_fit(xlin[-1], *popt)
            saturation = rmax/asymp * 100

        else:
            a, b, c, d, asymp, rmax, saturation = [np.nan] * 7

    except Exception as e:
        msg = "loc {} \t gen.log. \t binned: {} \t m: {} \t {}\n".format(
            loc, binned, monotonicity, str(e))
        print(msg)
        a, b, c, d, asymp, rmax, saturation = [np.nan] * 7

    # create series
    data = {
        # lowess
        'monotonicity': monotonicity,
        'ppt': ppt,
        'alphas_mean': alphas_mean,
        'alphas_std': alphas_std,
        # linregress (all)
        'exp_a': a_lin,
        'exp_b': b_lin,
        'exp_alpha': alpha,
        'exp_rvalue': rvalue,
        'exp_pvalue': pvalue,
        'exp_stderr': stderr,
        # gen. log. (all)
        'genlog_a': a,
        'genlog_b': b,
        'genlog_c': c,
        'genlog_d': d,
        'asymp': asymp,
        'rmax': rmax,
        'saturation': saturation,
    }

    return data


def _subset_v(
        v, tc, sxtcol, td, twcol, season, hod, wo_r_before, grad, rh_range,
        var):

    vt = v.copy()

    # only tropical cyclones?
    if tc:
        vt = vt.loc[~vt['id'].isnull()]
        tcstr = 'tcs'
    else:
        tcstr = 'ntcs'

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

    # negative/positiv/all gradients?
    if grad == 'ngrad':
        vt = vt.loc[vt[twcol] < 0]
    elif grad == 'pgrad':
        vt = vt.loc[vt[twcol] > 0]
    elif grad == 'agrad':
        pass

    # relative humidity range?
    if not rh_range == 'worhs':
        vt['rrh_td_2'] = vt['RH'].rolling(24, axis=1).mean()[-2]
        if rh_range == 'wrhmean':
            mean_rh = vt['RH', -td].mean()
            std_rh = vt['RH', -td].std()
            vt = vt.loc[
                (vt['RH', -td] >= mean_rh - std_rh/4) &
                (vt['RH', -td] <= mean_rh + std_rh/4)]
        elif rh_range == 'wrh80':
            vt = vt.loc[vt['RH', -td] >= vt['RH', -td].quantile(.8)]
        elif rh_range == 'wrrhmean':
            mean_rh = vt['rrh_td_2'].mean()
            std_rh = vt['rrh_td_2'].std()
            vt = vt.loc[
                (vt['rrh_td_2'] >= mean_rh - std_rh/4) &
                (vt['rrh_td_2'] <= mean_rh + std_rh/4)]
        elif rh_range == 'wrrh80':
            vt = vt.loc[vt['rrh_td_2'] >= vt['rrh_td_2'].quantile(.8)]
        else:
            raise ValueError

    # compute daily avg temps
    vr = vt[sxtcol].rolling(24, axis=1).mean()[-td].to_frame()
    vr[var] = vt[var]

    # get rid of nans
    vr = vr[~vr[-td].isnull()]
    sxtcolstr = sxtcol

    # cut off left and right ends
    xfirst = vr[-td].quantile(sxt_first_q)
    xlast = vr[-td].quantile(sxt_last_q)
    vr = vr[(vr[-td] >= xfirst) & (vr[-td] <= xlast)]

    # strings
    varstr = var
    tdstr = str(td)
    gradstr = grad
    rhrstr = rh_range

    return (vr, tcstr, sxtcolstr, tdstr, sstr, hodstr, rbstr, gradstr, rhrstr,
            varstr)


def compute_sxt_vs_r(
        loc, v, tc, sxtcol, td, twcol, season, hod, wo_r_before, grad,
        rh_range, var, q):

    # subset
    vt, tcstr, sxtcolstr, tdstr, sstr, hodstr, rbstr, gradstr, rhrstr, varstr = \
        _subset_v(
            v, tc, sxtcol, td, twcol, season, hod, wo_r_before, grad, rh_range,
            var,
        )

    # return empty series if there are no sxt values (think sst)
    if len(vt) <= n_bins:
        return {}

    # fits - all
    # data = _sxt_vs_r_fits_per_loc(loc, vt, td, var, binned=False)
    data = {}

    # binning by temperature
    binedges = np.quantile(vt[-td], np.linspace(0, 1, n_bins + 1))
    try:
        vt['{}_d'.format(-td)] = pd.cut(
            vt[-td], binedges, include_lowest=True, duplicates='drop'
        )
    except IndexError:
        print(loc)
        print(vt.shape, binedges)
        print(vt[-td].head())
        raise

    gvt = vt.groupby('{}_d'.format(-td))
    vt_q = gvt[var].quantile(q/100).to_frame()
    vt_q[-td] = gvt[-td].mean()
    # vt_q[-td] = vt_q.index
    # vt_q[-td] = vt_q[-td].apply(lambda x: x.mid)
    # vt_q[-td] = vt_q[-td].astype(float)
    vt_q['size'] = gvt.size()
    pts_per_bin = int(vt_q['size'].mean())

    # fits - binned
    data_q = _sxt_vs_r_fits_per_loc(loc, vt_q, td, var, binned=True)
    data_q = {key + f'_q{q}': value for key, value in data_q.items()}

    # combine
    data.update(data_q)

    # x/y-min/max
    data['xq1'] = vt[-td].quantile(.01)
    data['xq99'] = vt[-td].quantile(.99)
    data['yq1'] = vt[var].quantile(.01)
    data['yq99'] = vt[var].quantile(.99)

    # pts per bin
    data['n_bursts'] = len(vt)
    data['pts_per_bin'] = pts_per_bin

    # add subset strings
    data = {
        '{}_{}_{}_{}_{}_{}_{}_{}_{}_'.format(
            tcstr, sxtcolstr, tdstr, sstr, hodstr,
            rbstr, gradstr, rhrstr, varstr) + key:
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

    if 'ngrad' in grads or 'pgrad' in grads:
        vbg = _load_bursts(locs, spartsfolder)
        vb[twcol, ''] = vbg[twcol].values

    # add tropical cyclone flag
    if args.only_tropical_cyclones:
        glcp_tc = pd.read_hdf(tc_glcp_store, where='l in locs')
        glcp_tc.set_index(['l', 'cp_burst'], inplace=True)
        glcp_tc = glcp_tc[['id']]
        # set column names
        glcp_tc.columns = pd.MultiIndex.from_tuples([['id', '']])
        vb = pd.merge(vb, glcp_tc, how='left',
                      left_on=['l', 'cp_burst'], right_index=True)

    # add additional seasons (JASO/DJFMA)
    month = vb['dtime'].dt.month
    vb[('season', 'JASO')] = False
    vb[('season', 'DJFMA')] = False
    vb.loc[month.isin([7, 8, 9, 10]), ('season', 'JASO')] = True
    vb.loc[month.isin([12, 1, 2, 3, 4]), ('season', 'DJFMA')] = True
    del month

    # store empty dataframe
    if len(vb) <= n_bins:
        pd.DataFrame(index=[loc]).to_pickle(fname)
        return

    # choose subsets, create data dictionary
    data = {}
    for tc in tcs:
        for sxtcol in sxt_cols:
            for td in tds:
                for season in seasons:
                    for wo_r_before in wo_r_befores:
                        for grad in grads:
                            for rh_range in rh_ranges:
                                for vari in varis:
                                    for q in qs:
                                        datai = compute_sxt_vs_r(
                                            loc, vb, tc=tc, sxtcol=sxtcol,
                                            td=td,
                                            twcol=twcol,
                                            season=season, hod=hod,
                                            wo_r_before=wo_r_before, grad=grad,
                                            rh_range=rh_range, var=vari, q=q,
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
