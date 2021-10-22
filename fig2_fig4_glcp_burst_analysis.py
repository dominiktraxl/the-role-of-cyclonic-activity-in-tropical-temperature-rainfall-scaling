import matplotlib as mpl
mpl.use('svg')

import os

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
sns = None

import numpy as np
import pandas as pd
import deepgraph
plot_hist = deepgraph.DeepGraph.plot_hist
from scipy.optimize import curve_fit
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.stats import linregress
import statsmodels.formula.api as smf

# publication plot (default) parameters
# sns.set_context('paper', font_scale=.8)
# width = 3.465  # 6
# height = width / 1.618  # (golden ratio)
plt.rc('text', usetex=True)
# plt.style.use(os.getcwd() + '/../single_column.mplstyle')
width, height = plt.rcParams['figure.figsize']
plt.rc('font', family='serif', serif='Times')


# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-i', '--i',
                    help="do not load burst data "
                         "(implies %run -i this_script.py)",
                    action='store_true')
parser.add_argument('-s', '--selection',
                    help="plot analysis for 'selection' of locations",
                    type=str)
parser.add_argument('-nl', '--nr-locs',
                    help='only take subset of locations in selection',
                    type=int,
                    default=None)
parser.add_argument('-l', '--location',
                    help='manually choose location',
                    type=int)
parser.add_argument('-rs', '--random_seed',
                    help='to choose location',
                    type=int,
                    default=0)
parser.add_argument("-tcs", '--only-tropical-cyclones', action='store_true',
                    help="consider only rainfall events tagged as part of a "
                         "tropical cyclones")
parser.add_argument('-m', '--map-plot',
                    help="plot selection on a basemap",
                    action='store_true')
parser.add_argument('-mhodtd', '--map-plot-by-hod-td',
                    help="plot selection on a basemap, one plot for each "
                         "hod, including all tds",
                    action='store_true')
parser.add_argument('-sb', '--single-burst',
                    help="plot a (random) single burst",
                    action='store_true')
parser.add_argument('-akb', '--agg-k-bursts',
                    help="aggregate k-bursts into on plot",
                    action='store_true')
parser.add_argument('-aab', '--agg-all-bursts',
                    help="aggregate all bursts into on plot",
                    action='store_true')
parser.add_argument('-aabhod', '--agg-all-bursts-by-hod',
                    help="aggregate all bursts split by hour of the day into "
                    "one plot",
                    action='store_true')
parser.add_argument('-tcstats', '--tc-burst-stats',
                    help="analyse hists and relations between "
                    "r/mst/grad/dist_to_eye",
                    action='store_true')
parser.add_argument('-gvcat', '--grad-vs-category',
                    help='sat grad vs tc category',
                    action='store_true')
parser.add_argument('-gvmst', '--grad-vs-max-sustained-wind',
                    help='sat grad vs max sustained wind',
                    action='store_true')
parser.add_argument('-gvr', '--grad-vs-r',
                    help="sat grad vs r independent of hour of the day",
                    action='store_true')
parser.add_argument('-gvrhod', '--grad-vs-r-by-hod',
                    help="sat grad vs r independent by hour of the day",
                    action='store_true')
parser.add_argument('-gvrhodwor', '--grad-vs-r-by-hod-wo-rtd',
                    help="sat grad vs r independent by hour of the day, "
                         "only bursts with no rainfall 48h before",
                    action='store_true')
parser.add_argument('-rvg', '--r-vs-grad',
                    help="r vs sat grad independent of hour of the day",
                    action='store_true')
parser.add_argument('-satvr', '--sat-vs-r',
                    help="sat vs r independent of hour of the day (bursts!)",
                    action='store_true')
parser.add_argument('-sdtvr', '--sdt-vs-r',
                    help="sdt vs r independent of hour of the day (bursts!)",
                    action='store_true')
parser.add_argument('-svrtd', '--sat-vs-r-by-td',
                    help="sat vs r independent of hour of the day",
                    action='store_true')
parser.add_argument('-svrhod', '--sat-vs-r-by-hod',
                    help="sat vs r independent of time delay",
                    action='store_true')
parser.add_argument('-svrtdhod', '--sat-vs-r-by-td-hod',
                    help="sat vs r one plot for each td with all hods",
                    action='store_true')
parser.add_argument('-svrhodtd', '--sat-vs-r-by-hod-td',
                    help="sat vs r one plot for each hod with all tds",
                    action='store_true')
parser.add_argument('-svsvxtd', '--sdt-vs-sat-vs-x-by-td',
                    help="hexbin plot sdt vs sat vs n_nodes/r_q90",
                    action='store_true')
parser.add_argument('-svsvxhodtd', '--sdt-vs-sat-vs-x-by-hod-td',
                    help="sdt vs sat vs n_nodes/r_q90 one plot for each "
                         "hod with all tds",
                    action='store_true')
parser.add_argument('-tvstmp', '--t-vs-tmps',
                    help="split bursts by intensity, plot tmps and RH",
                    action='store_true')
parser.add_argument('-tvstmphod', '--t-vs-tmps-by-hod',
                    help="split bursts by hour of the day and intensity, "
                         "plot tmps and RH",
                    action='store_true')
parser.add_argument('-tvssathod', '--t-vs-sat-by-hod',
                    help="split bursts by hour of the day and intensity, "
                         "one plot",
                    action='store_true')
parser.add_argument('-tvssathodrtd', '--t-vs-sat-by-hod-wo-rtd',
                    help="split bursts by hour of the day and intensity, "
                         "only bursts with no rain at tdX, one plot",
                    action='store_true')
parser.add_argument('-hd', '--hod-distribution',
                    help="plot hour of day distribution by intensity",
                    action='store_true')
parser.add_argument('-tdvsa', '--td-vs-alpha',
                    help="plot td vs alpha mean (+std)",
                    action='store_true')
parser.add_argument('-tdvsahod', '--td-vs-alpha-by-hod',
                    help="plot td vs alpha mean by hod (+std) one plot",
                    action='store_true')
parser.add_argument('-si', '--split_intensity',
                    help="create subplots for each intensity bin. "
                    "only has an effect on -akb and -aab",
                    action='store_true')
parser.add_argument('-a', '--all',
                    help="set all options to true",
                    action='store_true')
args = parser.parse_args()

# if gvmst, consider only tc bursts
if (args.grad_vs_max_sustained_wind or args.tc_burst_stats or
        args.grad_vs_category):
    args.only_tropical_cyclones = True

# tropical cyclone folder string
if args.only_tropical_cyclones:
    tcfstr = '_tc'
else:
    tcfstr = ''

# parameters
r = .1
p = 0
buffer = 10  # tc track degree buffer
min_dist_to_eye = None  # minimum distance from eye (km)
max_dist_to_eye = None  # maximum distance from eye (km)
twcols = ['s_-6_-2']  # ['s_{}_-2'.format(tf) for tf in range(-24, -2, 6)]
n_bootstraps_q90 = 1000
n_bootstraps_qr = 1000
store_nat_comm_data = True

# plot parameters
colors = {
    'r': 'k',
    'sat': 'orangered',
    'sdt': 'royalblue',
    'rh': 'saddlebrown',
    'rrh': 'darkmagenta',
    'grad': 'forestgreen',
    'lat': 'goldenrod',
    'k_burst': 'deepskyblue',
}

# filesystem folders
storefolder = os.getcwd() + '/data_processing/'
glsxtvsrfolder = 'gl_SXT_vs_R_linregress/'
sxtfolder = storefolder + f'v_r{r}_p{p}_SXT_td_X/'
partsfolder = 'glcp_burst_parts/'
spartsfolder = 'glcp_burst_sat_grad_parts/'
# spartsfolder = 'glcp_burst_sat_grad_parts_24_12/'
tc_glcp_store = storefolder + f'glcp_r{r}_p{p}_tcs_{buffer}_degrees.h5'

if args.selection is not None:
    locs = np.load(storefolder + args.selection + '.npy')
    locs = locs.astype(int)
    picfolder = os.getcwd() + f'/glcp_burst{tcfstr}' + \
        '/{}/'.format(args.selection)
    sname = args.selection
    seasons = []
    ns = sname.split('_')
    if 'JJA' in ns:
        seasons.append('JJA')
    if 'DJF' in ns:
        seasons.append('DJF')
    if 'JASO' in ns:
        seasons.append('JASO')
    if 'DJFMA' in ns:
        seasons.append('DJFMA')
    if ('JJA' not in ns and
        'DJF' not in ns and
        'JASO' not in ns and
            'DJFMA' not in ns):
        seasons.append('all')

elif args.location:
    locs = [args.location]
    picfolder = os.getcwd() + f'/glcp_burst{tcfstr}' + \
        '/{:06d}/'.format(locs[0])
    sname = locs[0]
    seasons = ['JASO', 'DJFMA']  # ['JJA', 'DJF']

else:
    clocs = os.listdir(storefolder + partsfolder)
    seed = np.random.seed(args.random_seed)
    locs = [int(np.random.choice(clocs, 1, replace=False)[0].split('.')[0])]
    picfolder = os.getcwd() + f'/glcp_burst{tcfstr}' + \
        '/{:06d}/'.format(locs[0])
    sname = locs[0]
    seasons = ['JASO', 'DJFMA']  # ['JJA', 'DJF']
    print(sname)

# all seasons for tropical cyclones
if args.only_tropical_cyclones:
    seasons = ['all']

os.makedirs(picfolder, exist_ok=True)


def single_burst(v):

    os.makedirs(picfolder + 'single_bursts/', exist_ok=True)

    # select random burst
    vt = v.sample(
        # random_state=0,
    )

    # plot
    fig, _ = _plot_bursts(vt)

    # save figure
    fig.tight_layout()
    picfile = picfolder + 'single_bursts/{}_single_burst_{:06d}.pdf'.format(
        sname, vt.index[0])
    fig.savefig(picfile)
    fig.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig)


def agg_bursts(v, season, k_burst=None, hod=False, si_by='r_mean'):

    if k_burst is None:
        vt = v.loc[v['season', season]]
        kbstr = 'all'
    else:
        vt = v.loc[(v['season', season]) & (v['k_burst'] == k_burst)]
        kbstr = k_burst

    if not hod:

        if args.split_intensity:

            low = vt[si_by].quantile(.1)
            high = vt[si_by].quantile(.9)

            vt_low = vt[vt[si_by] <= low]
            vt_med = vt[(vt[si_by] > low) & (vt[si_by] <= high)]
            vt_high = vt[vt[si_by] > high]

            fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.80))
            axs = axs.flatten()

            ax_xmins = []
            ax_xmaxs = []
            ax_ymins = []
            ax_ymaxs = []
            ax_sxts = []
            sxt_mins = []
            sxt_maxs = []
            for vt_, ax in zip([vt_low, vt_med, vt_high], axs):
                _, ax_sxt = _plot_bursts(vt_, fig=fig, ax=ax)
                # min max xr
                ax_xmin, ax_xmax = ax.get_xlim()
                ax_xmins.append(ax_xmin)
                ax_xmaxs.append(ax_xmax)
                # min max yr
                ax_ymin, ax_ymax = ax.get_ylim()
                ax_ymins.append(ax_ymin)
                ax_ymaxs.append(ax_ymax)
                # min max sxt
                ax_sxts.append(ax_sxt)
                sxt_min, sxt_max = ax_sxt.get_ylim()
                sxt_mins.append(sxt_min)
                sxt_maxs.append(sxt_max)

            for ax, ax_sxt in zip(axs, ax_sxts):
                ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
                ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
                ax_sxt.set_ylim(min(sxt_mins), max(sxt_maxs))

            # save figure
            fig.tight_layout()
            picfile = picfolder + '{}_{}_sep_r_{}_bursts.pdf'.format(
                sname, season, kbstr)
            fig.savefig(picfile)
            fig.savefig(picfile[:-3] + 'png')
            print(picfile)
            plt.close(fig)

        elif not args.split_intensity:

            # plot
            fig, _ = _plot_bursts(vt)

            # save figure
            fig.tight_layout()
            picfile = picfolder + '{}_{}_{}_bursts.pdf'.format(
                sname, season, kbstr)
            fig.savefig(picfile)
            fig.savefig(picfile[:-3] + 'png')
            print(picfile)
            plt.close(fig)

    elif hod:

        fig, axs = plt.subplots(4, 2, figsize=(38.40, 21.60))
        axs = axs.flatten()

        ax_xmins = []
        ax_xmaxs = []
        ax_ymins = []
        ax_ymaxs = []
        ax_sxts = []
        sxt_mins = []
        sxt_maxs = []

        for name, ax in zip([0, 12, 3, 15, 6, 18, 9, 21], axs):

            vtt = vt.loc[vt['dtime'].dt.hour == name]

            _, ax_sxt = _plot_bursts(vtt, n_k_burst=False, fig=fig, ax=ax)
            # min max xr
            ax_xmin, ax_xmax = ax.get_xlim()
            ax_xmins.append(ax_xmin)
            ax_xmaxs.append(ax_xmax)
            # min max yr
            ax_ymin, ax_ymax = ax.get_ylim()
            ax_ymins.append(ax_ymin)
            ax_ymaxs.append(ax_ymax)
            # min max sxt
            ax_sxts.append(ax_sxt)
            sxt_min, sxt_max = ax_sxt.get_ylim()
            sxt_mins.append(sxt_min)
            sxt_maxs.append(sxt_max)

            ax.set_title(
                'hour of the day: {:02d} | n_bursts: {}'.format(
                    name, len(vtt)))

        for ax, ax_sxt in zip(axs, ax_sxts):
            ax.set_xlim(min(ax_xmins), 27)
            ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
            ax_sxt.set_ylim(min(sxt_mins), max(sxt_maxs))

        # save figure
        fig.tight_layout()
        picfile = picfolder + '{}_{}_{}_bursts_by_hod.pdf'.format(
            sname, season, kbstr)
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close(fig)


def t_vs_temps(v):

    os.makedirs(picfolder + 'single_bursts/', exist_ok=True)

    fig, axs = _plot_t_vs_temps(v)

    index = v.index[0]
    loc = v.at[index, ('l', '')]
    dtime = v.at[index, ('dtime', '')]
    lat = v.at[index, ('lat', '')]
    lon = v.at[index, ('lon', '')]
    r_mean = v.at[index, ('r_mean', '')]
    axs[0].set_title(
        "{} | l={} cp={} | dtime={} | lat={} | "
        "lon={} | r_mean={:.2f}".format(
            sname, loc, index, dtime, lat, lon, r_mean))

    # save figure
    fig.tight_layout()
    picfile = picfolder + \
        'single_bursts/{}_single_burst_t_vs_tmps_{:06d}.pdf'.format(
            sname, v.index[0])
    fig.savefig(picfile)
    fig.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig)


def t_vs_temps_by_intensity(v, season):

    # parameters
    td = 2

    vt = v.loc[v['season', season]]

    # rrh
    rrh_td_col = 'rrh_td_{}'.format(td)
    vt[rrh_td_col] = vt['RH'].rolling(24, axis=1).mean()[-td]

    # RH density plots
    n_plots = 6
    fig_RH, axs_RH = plt.subplots(
        n_plots, 1, sharex=True, figsize=(19.20, 10.80)
    )
    axs_RH = axs_RH.flatten()[::-1]

    c = 0
    for wo_r_before in [True, False]:  # [False, True]:
        for rh_range in [False]:  # [False, True]:

            vtt = vt.copy()

            # rain before?
            if not wo_r_before:
                rbstr = 'wrb'
            elif wo_r_before:
                for tdr in range(-6, -51, -3):
                    vtt = vtt.loc[vtt['r', tdr].isnull()]
                rbstr = 'worb'

            if wo_r_before and not rh_range:

                for td_RH, ax_RH in zip(range(-2, -2-n_plots, -1), axs_RH):

                    plot_hist(
                        vtt['RH', td_RH], bins=50, log_bins=False,
                        density=True,
                        floor=False, ax=ax_RH,
                        linestyle='-',
                        lw=1,
                        color='b',
                        # marker='o',
                        # ms=2,
                        # alpha=.3,
                        label='worhr | {}'.format(td_RH))

                    ax_RH.plot(
                        [vtt['RH', td_RH].mean(), vtt['RH', td_RH].mean()],
                        [0, ax_RH.get_ylim()[1]], color='b'
                    )

            # relative humidity range?
            if rh_range:
                # mean_rh = vtt['RH', -td].mean()
                # std_rh = vtt['RH', -td].std()
                # vtt = vtt.loc[
                #     (vtt['RH', -td] >= mean_rh - std_rh/4) &
                #     (vtt['RH', -td] <= mean_rh + std_rh/4)
                # ]
                vtt = vtt.loc[vtt['RH', -td] >= vtt['RH', -td].quantile(.9)]
                rhrstr = 'wrhr'
            elif not rh_range:
                rhrstr = 'worhr'

            if wo_r_before and rh_range:

                for td_RH, ax_RH in zip(range(-2, -2-n_plots, -1), axs_RH):

                    plot_hist(
                        vtt['RH', td_RH], bins=50, log_bins=False,
                        density=True,
                        floor=False, ax=ax_RH,
                        linestyle='-',
                        lw=1,
                        color='r',
                        # marker='o',
                        # ms=2,
                        # alpha=.3,
                        label='wrhr | {}'.format(td_RH))

                    ax_RH.plot(
                        [vtt['RH', td_RH].mean(), vtt['RH', td_RH].mean()],
                        [0, ax_RH.get_ylim()[1]], color='r'
                    )

#             if rh_range:
#                 mean_rh = vtt[rrh_td_col].mean()
#                 std_rh = vtt[rrh_td_col].std()
#                 vtt = vtt.loc[
#                     (vtt[rrh_td_col] >= mean_rh - std_rh/4) &
#                     (vtt[rrh_td_col] <= mean_rh + std_rh/4)
#                 ]
#                 rhrstr = 'wrhr'
#             elif not rh_range:
#                 rhrstr = 'worhr'

            fig = _plot_t_vs_temps_by_intensity(vtt)

            # store
            picfile = picfolder + \
                '{}_{}_{}_{}_td_vs_tmps_by_intensitys'.format(
                    sname, season, rbstr, rhrstr)

            # pdf
            fig.savefig(
                picfile + '.pdf',
                format='pdf',
                bbox_inches='tight', pad_inches=0,
                dpi=600,
            )
            # png
            fig.savefig(
                picfile + '.png',
                format='png',
                bbox_inches='tight', pad_inches=0,
                dpi=600,
            )
            # svg
            fig.savefig(
                picfile + '.svg',
                format='svg',
                bbox_inches='tight', pad_inches=0,
                dpi=600,
            )
            print(picfile + '.xxx')
            plt.close(fig)

            c += 1

    ymins = []
    ymaxs = []
    for ax in axs_RH:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)

    for ax in axs_RH:
        # ax.set_ylim(min(ymins), max(ymaxs))
        ax.grid()
        ax.legend()

    # save figure
    # fig_RH.tight_layout()
    picfile = picfolder + \
        '{}_{}_worb_RH_densities.pdf'.format(
            sname, season, rbstr, rhrstr)
    fig_RH.savefig(picfile)
    fig_RH.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig_RH)


def t_vs_temps_by_intensity_by_hod(v, season):

    os.makedirs(picfolder + 't_vs_tmp_by_hod/', exist_ok=True)

    vt = v.loc[v['season', season]]

    gvt = vt.groupby(vt['dtime'].dt.hour)

    for name, vtt in gvt:

        fig = _plot_t_vs_temps_by_intensity(vtt)

        # save figure
        fig.tight_layout()
        picfile = (
            picfolder + "t_vs_tmp_by_hod/"
            "{}_{}_{:02d}_td_vs_tmps_by_intensitys.pdf".format(
                sname, season, name))
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close(fig)


def t_vs_sat_by_intensity_by_hod(v, season):

    vt = v.loc[v['season', season]]

    fig, axs = plt.subplots(4, 2, figsize=(19.20, 10.80))
    axs = axs.flatten()

    for name, ax in zip([0, 12, 3, 15, 6, 18, 9, 21], axs):
        print(name)
        vtt = vt.loc[vt['dtime'].dt.hour == name]
        _plot_t_vs_sat_by_intensity(vtt, fig=fig, ax=ax)

        ax.set_title('hod {:02d} | n_bursts {}'.format(name, len(vtt)))

    # save figure
    fig.tight_layout()
    picfile = (
        picfolder + "{}_{}_t_vs_sat_by_intensity_by_hod.pdf".format(
            sname, season))
    fig.savefig(picfile)
    fig.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig)


def t_vs_sat_by_intensity_by_hod_wo_r_before(v, season):

    vt = v.loc[v['season', season]]

    allfigs = []
    allaxs = []
    allax_ymins = []
    allax_ymaxs = []

    for tdr in range(-6, -51, -3):

        vt = vt.loc[vt['r', tdr].isnull()]

        fig, axs = plt.subplots(4, 2, figsize=(19.20, 10.80))
        axs = axs.flatten()

        allfigs.append(fig)

        for name, ax in zip([0, 12, 3, 15, 6, 18, 9, 21], axs):
            print(name)
            vtt = vt.loc[vt['dtime'].dt.hour == name]
            _plot_t_vs_sat_by_intensity(vtt, fig=fig, ax=ax)

            # min max yr
            ax_ymin, ax_ymax = ax.get_ylim()
            allax_ymins.append(ax_ymin)
            allax_ymaxs.append(ax_ymax)

            ax.set_title('hod {:02d} | n_bursts {} | wo rtd {}'.format(
                name, len(vtt), tdr))

            allaxs.append(ax)

    for ax in allaxs:
        ax.set_ylim(min(allax_ymins), max(allax_ymaxs))

    for fig, tdr in zip(allfigs, range(-6, -51, -3)):

        # save figure
        fig.tight_layout()
        picfile = (
            picfolder +
            "{}_{}_t_vs_sat_by_intensity_by_hod_wo_rtd_{}.pdf".format(
                sname, season, abs(tdr)))
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close(fig)


def hod_distribution(v, season, si_by='r_mean'):

    vt = v.loc[v['season', season]]
    os.makedirs(picfolder + 't_vs_tmp_by_hod/', exist_ok=True)

    n_bins = 5

    # binning of si_by
    binedges = np.quantile(vt[si_by], np.linspace(.5, 1, n_bins + 1))
    vt['{}_d'.format(si_by)] = pd.cut(
        vt[si_by], binedges,
        labels=np.linspace(.5, 1, n_bins + 1)[1:],
        include_lowest=True, duplicates='drop')
    gvt = vt.groupby('{}_d'.format(si_by))

    fig, axs = plt.subplots(2, 1, figsize=(19.20, 10.80))

    for name, vtt in gvt:

        gvtt = vtt.groupby(vtt['dtime'].dt.hour)

        ihod = gvtt.size().sort_index()

        # plot
        axs[0].plot(
            ihod.index, ihod.values/len(vtt) * 100,
            marker='o',
            lw=2.5,
            ms=12,
            label='{} | n_bursts: {}'.format(name, len(vtt)),
        )

        ihodtmp = gvtt.apply(lambda x: x['sat', 0].mean()).to_frame()
        ihodtmp['std'] = gvtt.apply(lambda x: x['sat', 0].std())

        axs[1].plot(
            ihodtmp.index, ihodtmp[0],
            marker='o',
            lw=2,
            ms=10,
            label='{}'.format(name),
        )

    axs[0].hlines(12.5, xmin=-1, xmax=22, color='k')
    axs[0].legend(loc='upper left')
    axs[0].set_xticks(list(range(0, 24, 3)))
    axs[0].grid()
    # axs[0].set_xlabel('hour of the day')
    axs[0].set_ylabel('proportion of bursts [%]')

    axs[1].legend(loc='upper left')
    axs[1].set_xticks(list(range(0, 24, 3)))
    axs[1].grid()
    axs[1].set_xlabel('hour of the day')
    axs[1].set_ylabel('sat [°C]')

    # save figure
    fig.tight_layout()
    picfile = picfolder + \
        't_vs_tmp_by_hod/{}_{}_hod_distribution_by_intensity.pdf'.format(
            sname, season)
    fig.savefig(picfile)
    fig.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig)


def _plot_t_vs_temps_by_intensity(v, si_by='r_max'):

    n_bins = 5
    colors = [
        # "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71", "faded green",
        "#9b59b6", "#3498db", "#e74c3c",
    ]

    # store data for nat.comm. supplementary material
    if store_nat_comm_data:
        data = v.copy()
        for col in ['RH', 'sdt', 'r', 'r_mean', 'cp_burst', 'l', 'rrh_td_2',
                    'dtime', 'k_burst', 'season']:
            del data[col]
        data = data.iloc[:, :62]

        cols = [
            ' '.join(tuple(map(str, t))).strip() for t in data.columns.values]
        ncols = []
        for col in cols:
            if col == 'r_max':
                ncols.append('maximum intensity of episode [mm/h]')
            else:
                ncols.append(f'temperature at t={col.split(" ")[-1]}h [°C]')
        data.columns = ncols

        if args.selection == 'nta_neg_box_JASO':
            fstr = 'fig2a'
        elif args.selection == 'nta_pos_box_JASO':
            fstr = 'fig2b'
        else:
            fstr = 'other'
        os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
        data.to_csv(storefolder + f'Nat_Comm_Fig_data/{fstr}.csv', index=False)

    # binning of si_by
    binedges = np.quantile(v[si_by], np.linspace(.5, 1, n_bins + 1))
    # binedges = np.quantile(v[si_by], np.linspace(0, 1, n_bins + 1))
    v['{}_d'.format(si_by)] = pd.cut(
        v[si_by], binedges,
        labels=np.linspace(.5, 1, n_bins + 1)[1:],
        # labels=np.linspace(0, 1, n_bins + 1)[1:],
        include_lowest=True, duplicates='drop')
    gv = v.groupby('{}_d'.format(si_by))

    # plot
    fig, ax = plt.subplots(figsize=(width, height))

    lines = {
        'sat': [],
        'sdt': [],
        'RH': [],
    }
    c = 0
    for name, vt in gv:

        if name == .7 or name == .9:
            continue

        # n_bursts = len(vt)

        # select columns
        vt = vt[['sat', 'sdt', 'RH']]

        # mean
        vttmean = vt.mean().to_frame().T

        # reshape
        tds = list(range(-48, vttmean['sat'].columns.max()+1))
        vp = pd.DataFrame(index=tds)
        for col in ['sat', 'sdt', 'RH']:
            vp.loc[vttmean[col].columns, col] = vttmean[col].values

        # cut off bursts
        vp = vp[vp.index <= 12]

        # interpolate
#         for col in ['sat', 'sdt', 'RH']:
#             vp[col] = vp[col].interpolate(
#                 method='index',
#                 limit=1,
#                 limit_direction='both'
#             )

        # rolling 24h window
        rvp = vp.rolling(24)
        vp['sat_24_mean'] = rvp['sat'].mean()

        # plot lines
        line = ax.plot(
            vp.index.values, vp['sat'].values,
            color=colors[c],
            # marker='o',
            lw=.6,
            linestyle=(0, (6, 1)),
            # ms=5,
            # label='{} {:.2f} bursts: {}'.format('sat', name, n_bursts),
            zorder=5,
            label=r"$r_{max}$" + \
                  r" between {:d}th and {:d}th percentile".format(
                      int(name*100-10), int(name*100)),
        )
        lines['sat'].append(line[0])

        # plot moving average
        line = ax.plot(
            vp.index.values, vp['sat_24_mean'].values,
            color=colors[c],
            label="rolling (24h) mean",
            # lw=3,
            zorder=6,
        )
        lines['sat'].append(line[0])

        # plot gradient
        x = vp.loc[-6:-2].index.values
        y = vp.loc[-6:-2, 'sat_24_mean'].values
        m, t, _, _, _ = linregress(x, y)
        ax.plot(
            [-6, -2], m*np.asarray([-6, -2]) + t,
            color='k',  # colors[c],
            linestyle=(0, (1, 1)),
            marker='o',
            ms=2.5,
            label="sat_grad_-6_-2",
            zorder=7,
            # lw=2,
        )

        c += 1

    ax.set_xticks(list(range(-54, 18, 6)))
    ax.grid()

    # set labels
    ax.set_ylabel('$T$ [$^{\circ}$C]')
    ax.set_xlabel('$time$ [hours before/after onset of episodes]')

    # x/y lims
    # ax_rh.set_ylim(70, 100)

    # paper: set same limits for nta_neg_box and nta_pos_box
    ymin = 25.7
    ymax = 27.95
    dy = .05 * (ymax - ymin)
    ax.set_ylim(25.7, 27.95)

    # vertical line at td 0
    # ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin+dy, ymax-dy)

    # legend
    # for col, ax in zip(['sat', 'sdt', 'RH'], axs):
    #     labs = [l.get_label() for l in lines[col]]
    #     ax.legend(lines[col], labs, fontsize='medium', loc='upper left')

    labs = [l.get_label() for l in lines['sat']]
    # labs = labs[::2] + labs[1::2]
    lines['sat'] = lines['sat'] + lines['sat']
    # lines['sat'] = lines['sat'][::2] + lines['sat'][1::2]
    if 'neg' in args.selection:
        loc = 'lower left'
    else:
        loc = 'upper left'
    ax.legend(
        lines['sat'], labs,
        # fontsize='small',
        loc=loc,
        ncol=1,
        facecolor='white', framealpha=1,
    )

    # title
    # ax.set_title('n_bursts: {}'.format(n_bursts))

    return fig


def _plot_t_vs_temps(v):

    # plot
    fig, axs = plt.subplots(
        3, 1, figsize=(19.20, 10.80), sharex=True)
    axs = axs.flatten()

    lines = {
        'sat': [],
        'sdt': [],
        'RH': [],
    }

    vt = v.copy()
    n_bursts = len(vt)

    # select columns
    vt = vt[['sat', 'sdt', 'RH']]

    # mean
    vttmean = vt.mean().to_frame().T

    # reshape
    tds = list(range(-48, vttmean['sat'].columns.max()+1))
    vp = pd.DataFrame(index=tds)
    for col in ['sat', 'sdt', 'RH']:
        vp.loc[vttmean[col].columns, col] = vttmean[col].values

    # cut off bursts
    vp = vp[vp.index <= 12]

    # interpolate
#     for col in ['sat', 'sdt', 'RH']:
#         vp[col] = vp[col].interpolate(
#             method='index',
#             limit=1,
#             limit_direction='both'
#         )

    # rolling 24h window
    rvp = vp.rolling(24)

    # plot lines
    for col, ax_ in zip(['sat', 'sdt', 'RH'], axs):
        line = ax_.plot(
            vp.index.values, vp[col].values,
            color='k',
            # marker='o',
            lw=2,
            # ms=5,
            label='{} bursts: {}'.format(col, n_bursts),
        )
        lines[col].append(line[0])

        # plot moving average
        ax_.plot(
            vp.index.values, rvp[col].mean().values,
            color='k',
            lw=1)

    for ax in axs:
        ax.set_xticks(list(range(-51, 15, 3)))
        ax.grid()

    # set labels
    # ax.set_xlabel("ref time")
    axs[0].set_ylabel("T [°C]")
    axs[1].set_ylabel("dPt [°C]")
    axs[2].set_ylabel("RH [%]")

    # x/y lims
    # ax_rh.set_ylim(70, 100)

    # vertical line at td 0
    for ax in axs:
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax)

    # legend
    for col, ax in zip(['sat', 'sdt', 'RH'], axs):
        labs = [l.get_label() for l in lines[col]]
        ax.legend(lines[col], labs, fontsize='medium', loc='upper left')

    # title
    # ax.set_title('n_bursts: {}'.format(n_bursts))

    return fig, axs


def _plot_t_vs_sat_by_intensity(v, si_by='r_mean', fig=None, ax=None):

    n_bins = 5
    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    # binning of si_by
    binedges = np.quantile(v[si_by], np.linspace(.5, 1, n_bins + 1))
    v['{}_d'.format(si_by)] = pd.cut(
        v[si_by], binedges,
        labels=np.linspace(.5, 1, n_bins + 1)[1:],
        include_lowest=True, duplicates='drop')
    gv = v.groupby('{}_d'.format(si_by))

    # plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

    c = 0
    for name, vt in gv:

        n_bursts = len(vt)

        # select columns
        vt = vt[['sat']]

        # mean
        vtmean = vt.mean().to_frame().T

        # reshape
        tds = list(range(-48, vtmean['sat'].columns.max()+1))
        vp = pd.DataFrame(index=tds)
        vp.loc[vtmean['sat'].columns, 'sat'] = vtmean['sat'].values

        # cut off bursts
        vp = vp[vp.index <= 12]

        # interpolate
#         vp['sat'] = vp['sat'].interpolate(
#             method='index',
#             limit=1,
#             limit_direction='both'
#         )

        # plot lines
        ax.plot(
            vp.index.values, vp['sat'].values,
            color=colors[c],
            # marker='o',
            lw=2,
            # ms=5,
            label='sat {:.2f} bursts: {}'.format(name, n_bursts),
        )

        # plot moving average
        rvp = vp.rolling(24)
        ax.plot(
            vp.index.values, rvp['sat'].mean().values,
            color=colors[c],
            lw=1)

        c += 1

    ax.set_xticks(list(range(-51, 15, 3)))
    ax.grid()

    # set labels
    # ax.set_xlabel("ref time")
    ax.set_ylabel("T [°C]")

    # x/y lims
    # ax_rh.set_ylim(70, 100)

    # vertical line at td 0
    ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin, ymax)

    # legend
    ax.legend(fontsize='x-small', loc='upper left')

    # title
    # ax.set_title('n_bursts: {}'.format(n_bursts))

    return fig, ax


def _plot_bursts(v, n_k_burst=True, fig=None, ax=None):

    # number of bursts
    n_bursts = len(v)

    # k_burst counts
    vc = v['k_burst'].value_counts().sort_index().to_frame()
    vc['td'] = (vc.index-1)*3

    # select columns
    v = v[['r', 'sat', 'sdt', 'RH']]

    # mean, std
    vtmean = v.mean().to_frame().T
    vtstd = v.std().to_frame().T

    # v['r'] median, q25, q75
    vtmedian = v.median().to_frame().T
    vtq25 = v.quantile(.25).to_frame().T
    vtq75 = v.quantile(.75).to_frame().T

    # reshape
    tds = list(range(-48, vtmean['r'].columns.max()+1))
    vp = pd.DataFrame(index=tds)
    for col in ['r', 'sat', 'sdt', 'RH']:
        vp.loc[vtmean[col].columns, col] = vtmean[col].values
        vp.loc[vtstd[col].columns, col + '_std'] = vtstd[col].values
        vp.loc[vtmedian[col].columns, col + '_median'] = vtmedian[col].values
        vp.loc[vtq25[col].columns, col + '_q25'] = vtq25[col].values
        vp.loc[vtq25[col].columns, col + '_q75'] = vtq75[col].values

    # alphas
    alphas = (~v['r'].isnull()).sum(axis=0) / n_bursts
    vp.loc[alphas.index, 'alpha'] = alphas.values

    # fill r values to hourly
    # vp.fillna(method='bfill', limit=1, inplace=True)
    # vp.fillna(method='ffill', limit=1, inplace=True)

    # interpolate
#     for col in ['sat', 'sdt', 'RH', 'sat_std', 'sdt_std', 'RH_std']:
#         vp[col] = vp[col].interpolate(
#             method='index',
#             limit=1,
#             limit_direction='both'
#         )
    for col in ['r', 'r_std', 'r_median', 'r_q25', 'r_q75']:
        vp.loc[0:, col] = vp.loc[0:, col].interpolate(
            method='index',
            limit=1,
            limit_direction='both',
        )

    # plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

    # r, sat, sdt, RH
    ax_sxt = ax.twinx()
    ax_rh = ax.twinx()
    ax_rh.spines["right"].set_position(("axes", 1.05))
    _make_patch_spines_invisible(ax_rh)
    ax_rh.spines["right"].set_visible(True)
    ax.set_zorder(10)
    ax.patch.set_visible(False)

    # line plots
    line_r = ax.plot(
        vp.loc[0:].index.values, vp.loc[0:, 'r_median'].values,
        color=colors['r'],
        # marker='o',
        lw=3,
        # ms=15,
        label='r',
        # markeredgecolor='r',
        zorder=10,
        alpha=.2,
    )
    ax.fill_between(
        vp.loc[0:].index.values,
        vp.loc[0:, 'r_q25'].values,
        vp.loc[0:, 'r_q75'].values,
        color='k', alpha=.2, lw=1,
    )

    lines = []
    for col, ax_ in zip(['sat', 'sdt', 'RH'], [ax_sxt, ax_sxt, ax_rh]):
        # print(vp)
        line = ax_.plot(
            vp.index.values, vp[col].values,
            color=colors[col.lower()],
            marker='o',
            lw=2,
            ms=5,
            label=col,
        )
        ax_.fill_between(
            vp.index.values,
            vp[col].values - vp[col + '_std'].values,
            vp[col].values + vp[col + '_std'].values,
            color=colors[col.lower()], alpha=.2, lw=1,
        )
        lines.append(line[0])
    lines.append(line_r[0])

    # scatter rainfall events
    rgba_colors = np.zeros((len(vp), 4))
    rgba_colors[:, 3] = vp['alpha']
    pc = ax.scatter(
        vp.index.values, vp['r_median'].values,
        # color=rgba_colors,
        c=vp['alpha']*100,
        cmap='Greys',
        vmin=0,
        s=150,
        zorder=20,
    )

#     rgba_colors = np.zeros((len(vp.loc[:-6, 'alpha']), 4))
#     rgba_colors[:, 3] = vp.loc[:-6, 'alpha']
#     pc = ax.scatter(
#         vp.loc[:-6].index.values, vp.loc[:-6, 'r'].values,
#         # color=rgba_colors,
#         c=vp.loc[:-6, 'alpha']*100,
#         cmap='Greys',
#         vmin=0,
#         s=200,
#         zorder=10,
#     )

    # set xticks, grid
    last_x = int(ax.get_xticks()[-1])
    ax.set_xticks(list(range(-51, last_x, 3)))
    ax_sxt.set_xticks(list(range(-51, last_x, 3)))
    ax_rh.set_xticks(list(range(-51, last_x, 3)))
    ax.grid(axis='x')
    ax_sxt.grid()

    # set labels
    # ax.set_xlabel("ref time")
    ax.set_ylabel("r [mm/h]")
    ax_sxt.set_ylabel("T [°C]")
    ax_rh.set_ylabel("RH [%]")

    # x/y lims
    ax_rh.set_ylim(50, 100)
    try:
        ax.set_xlim(
            vp['RH'].first_valid_index()-3, vp['RH'].last_valid_index()+3)
    except TypeError:
        pass

    # add k_burst counts as text
    if n_k_burst:
        for _, vci in vc.iterrows():
            x_pos = vci['td']
            text = int(vci['k_burst'])
            bottom, top = ax.get_ylim()
            # dy = (top - bottom) * .05
            # y = vp.loc[x_pos, 'r'] + dy
            y = top - (top - bottom) * .1
            ax.text(
                x=x_pos, y=y, s=text,
                ha='center', va='center', rotation=45
            )

    # legend
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, fontsize='medium')

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="3%", pad=0.7)
    # cax = fig.add_axes([0.1, 0.1, 0.03, 0.8])
    fig.colorbar(pc, cax=cax)
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    # fake colorbar axes
    for ax_ in [ax_sxt, ax_rh]:
        dax = make_axes_locatable(ax_)
        cax_ = dax.append_axes("left", size="3%", pad=0.7)
        cax_.axis('off')

    # title
    ax.set_title('n_bursts: {}'.format(n_bursts))

    return fig, ax_sxt


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


def _sxt_vs_r_fits_per_loc(
        v, gv, v_binned, var, binned, xcol, ycol, rhcol, twcol,  # @UnusedVariable
        n_bursts, pts_per_bin, rbstr, gradstr,  # @UnusedVariable
        rhrstr, q):  # @UnusedVariable

    if store_nat_comm_data:
        data = v.copy()
        data = data[['rsat_td_2', 'r_max']]
        data.rename(columns={
            'rsat_td_2': '24-hour mean temperature two hours before episode [°C]',
            'r_max': 'maximum intensity of episode [mm/h]'},
            inplace=True
        )

        if args.selection == 'nta_neg_box_JASO':
            fstr = 'fig2c'
        elif args.selection == 'nta_pos_box_JASO':
            fstr = 'fig2d'
        else:
            fstr = 'other'
        os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
        data.to_csv(storefolder + f'Nat_Comm_Fig_data/{fstr}.csv', index=False)

    fig, ax = plt.subplots(figsize=(width, height))

    # parameters
    interp1d_bins = 50

    # linspace
    xlin = np.linspace(v_binned[xcol].min(), v_binned[xcol].max(), interp1d_bins)
    # dx = (v_binned[xcol].max() - v_binned[xcol].min()) * .01

    # lines
    lns = []

    # bootstrapping q90 and quantile regression
    # ------------------------------------------------------------------------

    def bootstrap(array, q, n_elements=None, n_bootstraps=1000):
        "estimate percentile"
        if not n_elements:
            n_elements = len(array)
        bootstrap_estimates = np.empty(n_bootstraps, dtype=np.float64)
        for c in range(n_bootstraps):
            sample = np.random.choice(array, size=n_elements, replace=True)
            estimate = np.percentile(sample, q)
            bootstrap_estimates[c] = estimate
        return bootstrap_estimates

    def bootstrap_qr(df, x, y, q, n_elements=None, n_bootstraps=1000):
        "quantile regression bootstrap"
        if not n_elements:
            n_elements = len(df)
        bootstrap_estimates = np.empty((n_bootstraps, 2), dtype=np.float64)
        for c in range(n_bootstraps):
            print(f'{c+1}/{n_bootstraps} | {(c+1)/n_bootstraps*100}%')
            dft = df.sample(n=n_elements, replace=True)
            mod = smf.quantreg(f'{y} ~ {x}', dft)
            res = mod.fit(q=q)
            m, t = res.params[x], res.params['Intercept']
            bootstrap_estimates[c, 0] = m
            bootstrap_estimates[c, 1] = t
        return bootstrap_estimates

    # for confidence interval
    x_lin_ci = np.linspace(v_binned[xcol].min(), v_binned[xcol].max(), 50)  # for plotting

    # bootstrap q90
    # file name
    fname = storefolder + 'bootstrapping_sxt_vs_r/'
    os.makedirs(fname, exist_ok=True)
    fname += '{}_{}_{}_{}_{}_{}_vs_{}_q{}'.format(
        sname, season, rbstr, gradstr, rhrstr, xcol, var, q)
    try:
        # load?
        bes = np.load(fname + '.npy')
    except FileNotFoundError:
        print('could not find ', fname)
        bes = np.empty((n_bootstraps_q90, len(gv)), dtype=float)
        for c, (index, vt) in enumerate(gv):
            bes[:, c] = bootstrap(vt[var].values, q, n_bootstraps=n_bootstraps_q90)
        # store
        np.save(fname, bes)
    for c, (index, vt) in enumerate(gv):
        v_binned.at[index, 'bt_mean'] = bes[:, c].mean()
        v_binned.at[index, 'bt_q2.5'] = np.percentile(bes[:, c], 2.5)
        v_binned.at[index, 'bt_q97.5'] = np.percentile(bes[:, c], 97.5)

    # qr
    # file name
    fname += '_qr'
    v['sxt'] = v[xcol]
    v[f'log_{var}'] = np.log(v[var])
    try:
        alpha_qr = np.loadtxt(fname + '_alpha.txt')
    except OSError:
        print('could not find ', fname)
        mod = smf.quantreg(f'log_{var} ~ sxt', v)
        res = mod.fit(q=q/100)
        alpha_qr = (np.e**res.params['sxt'] - 1) * 100
        # store
        with open(fname + '_summary.txt', 'w') as file:
            file.write(res.summary().as_latex())
        with open(fname + '_alpha.txt', 'w') as file:
            file.write(str(alpha_qr))

    # res.summary()
    # y_pred = res.params['sxt'] * x_lin_ci + res.params['Intercept']

    # bootstrap qr
    # file name
    fname += '_bt'
    try:
        bes = np.load(fname + '.npy')
    except FileNotFoundError:
        print('could not find ', fname)
        bes = bootstrap_qr(v, 'sxt', f'log_{var}', q=q/100, n_elements=None,
                           n_bootstraps=n_bootstraps_qr)
        # store
        np.save(fname, bes)

    # bes: [slope, intercept] <-> [A, B]
    # bootstrap qr line
    y_pred_bt = exp_fit(x_lin_ci, np.e**bes[:, 1].mean(), bes[:, 0].mean())

    # # confidence interval
    y_pred_ci = np.atleast_2d(np.exp(bes[:, 1])).T * \
        np.exp(np.dot(np.atleast_2d(bes[:, 0]).T, np.atleast_2d(x_lin_ci)))
    y_pred_ub = np.percentile(y_pred_ci, 97.5, axis=0)
    y_pred_lb = np.percentile(y_pred_ci, 2.5, axis=0)

    # ------------------------------------------------------------------------
    # plot
    # bootstrapping qr lines
    # for c in range(len(bes)):
    #     ax.plot(x_lin_ci, bes[c, 0]*x_lin_ci + bes[c, 1],
    #             color='tab:blue',
    #             lw=.1,
    #             alpha=.2)

    # bootstrapping q90
    lns.append(ax.plot(
        v_binned[xcol], v_binned['bt_mean'],
        color=colors['r'],
        marker='o',
        ms=1.2,
        lw=.3,
        label="$P^{" + str(q) + "}$ [mmh$^{-1}$]"))

    # error bars bootstrapping q90
    for _, row in v_binned.iterrows():
        ax.plot([row[xcol], row[xcol]], [row['bt_q2.5'], row['bt_q97.5']],
                color='k', lw=.4)

    # bootstrapping confidence interval (ci)
    ax.fill_between(x_lin_ci, y_pred_lb, y_pred_ub,
                    facecolor='tab:blue',
                    alpha=.7,
                    edgecolor='none')

    # q90
    # ax.scatter([xcol], [var],
    #            s=50,
    #            c='grey',
    #            marker='+',
    #            linewidths=.2,
    #            label='q90')

    # qr
    # ax.plot(x_lin_ci, y_pred,
    #         linestyle=(0, (5, 1)),
    #         color='lightsteelblue',
    #         label='qr | intensity ~ gradient')

    # bootstrapping qr
    lns.append(ax.plot(
        x_lin_ci, y_pred_bt,
        linestyle=(0, (10, 1)),
        color='darkmagenta',
        lw=.5,
        label='exponential regression between $P^{90}$ and $T^r$'))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # plot r
    # lns.append(ax.plot(
    #     v_binned[xcol], v_binned[var],
    #     color='k',
    #     label="$P^{" + str(q) + "}$ [mmh$^{-1}$]",
    #     marker='o',
    #     ms=2,
    #     lw=.8,
    #     # ms=4
    #     ))

    # --------------------------------------------------------------------
    # lowess
    lowess = sm.nonparametric.lowess(np.log(v_binned[var]), v_binned[xcol], frac=.7)

    # interpolation
    if not binned:
        f = interp1d(
            lowess[:, 0], np.exp(lowess[:, 1]), bounds_error=False)
        xl = xlin
        yl = f(xlin)

    if binned:
        xl = lowess[:, 0]
        yl = np.exp(lowess[:, 1])

    # plot lowess
    # lns.append(ax.plot(
    #     xl, yl, 'r', label='lowess',
    #     alpha=.8, marker='o', ms=2
    # ))

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

    # plot ppt
    # if not np.isnan(ppt):
    #     ax.scatter(ppt, yl.max(), s=120, facecolors='none', edgecolors='k')

    # --------------------------------------------------------------------
    # exp fit linregress
    try:
        if monotonicity in ['decreasing', 'increasing', 'mean decrease']:

            A, B, rvalue, pvalue, stderr = linregress(
                v_binned[xcol], np.log(v_binned[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        elif monotonicity == 'mean increase (ppt last point)':

            global_min_T = xl[yl.argmin()]
            vt_min_T = v_binned[v_binned[xcol] >= global_min_T]

            A, B, rvalue, pvalue, stderr = linregress(
                vt_min_T[xcol], np.log(vt_min_T[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        elif monotonicity == 'mean increase (ppt)':

            global_min_T = xl[yl.argmin()]
            global_max_T = xl[yl.argmax()]

            # if ppt left of global minimum, fit all
            if global_min_T >= global_max_T:
                vt_min_max_T = v_binned
            else:
                vt_min_max_T = v_binned[
                    (v_binned[xcol] >= global_min_T) &
                    (v_binned[xcol] <= global_max_T)]

            A, B, rvalue, pvalue, stderr = linregress(
                vt_min_max_T[xcol], np.log(vt_min_max_T[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        else:

            A, B, rvalue, pvalue, stderr = linregress(
                v_binned[xcol], np.log(v_binned[var])
            )
            a_lin = np.e**B
            b_lin = A
            alpha = (np.e**b_lin - 1) * 100

        # plot linregress
        # lns.append(ax.plot(
        #     xl, exp_fit(xl, a_lin, b_lin),
        #     linestyle=(0, (5, 1)),
        #     color='darkmagenta',
        #     alpha=1,
        #     label='exponential regression between $P^{90}$ and $T^r$'
        # ))

        # add text
        # ax.text(
        #     xl[0] + dx, exp_fit(xl[0], a_lin, b_lin),
        #     "{:.1f}% (linregress)\n"
        #    # "a:            {:.3f}\n"
        #    # "a_lin:       {:.3f}\n"
        #    # "b:            {:.3f}\n"
        #    # "b_lin:       {:.3f}\n"
        #    # "alpha:      {:.3f}\n"
        #     "rvalue:    {:.3f}\n"
        #     "pvalue:    {:.3f}\n"
        #    # "stderr:    {:.3f}\n"
        #     "".format(
        #         alpha,
        #        # a_lin,
        #        # b_lin,
        #         rvalue,
        #         pvalue,
        #        # stderr,
        #     ),
        #     color='k',
        #    # fontsize=?,
        # )

        # add text box
        textstr = '\n'.join((
            'No. of episodes per bin: {}'.format(pts_per_bin),
            r'$\alpha$ = {} [\%$^{{\circ}}$C$^{{-1}}$]'.format(
                np.round(alpha_qr, 1)),
            'PCC = {}'.format(np.round(rvalue, 3)),
            'p-value = {:.1e}'.format(pvalue),
        ))
        props = dict(
            boxstyle='round',
            facecolor='w',
            alpha=1,
            edgecolor='lightgrey',
        )

        # place a text box axes coords
        if 'neg' in args.selection:
            # ax.text(0.65, 0.77, textstr, transform=ax.transAxes,
            #         verticalalignment='top', bbox=props)
            ax.text(0.0275, 0.50, textstr, transform=ax.transAxes,
                    verticalalignment='top', bbox=props)
        elif 'pos' in args.selection:
            ax.text(0.0275, 0.74, textstr, transform=ax.transAxes,  # y shift: .3
                    verticalalignment='top', bbox=props)

    except Exception as e:
        msg = "linregress \t binned: {} \t m: {} \t {}\n".format(
            binned, monotonicity, str(e))
        print(msg)
        a_lin, b_lin, alpha, rvalue, pvalue, stderr = [np.nan]*6

    # --------------------------------------------------------------------
    # generalzed logistic
    try:
        popt = None
        if monotonicity in ['mean increase (ppt)',
                            'decreasing', 'mean decrease']:
            a, b, c, d, asymp, rmax, saturation = [np.nan] * 7

        elif monotonicity == 'increasing':

            popt, _ = curve_fit(
                gen_log_fit,
                # v_binned[xcol], v_binned[var],
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
            vt_min_T = v_binned[v_binned[xcol] >= global_min_T]

            popt, _ = curve_fit(
                gen_log_fit,
                # vt_min_T[xcol], vt_min_T[var],
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

        # plot
        if popt is not None:

            asymp = 1/popt[2] + popt[3]
            rmax = gen_log_fit(xl[-1], *popt)
            saturation = rmax/asymp * 100

            # lns.append(ax.plot(
            #     xl, gen_log_fit(xl, *popt),
            #     color='paleturquoise',
            #     # lw=lw_fit,
            #     alpha=1,
            #     label="gen. log."
            # ))

            # add text
            # ax.text(
            #     xl[-1] + dx, gen_log_fit(xl[-1], *popt),
            #    # "a:       {:.1E}\n"
            #    # "b:       {:.3f}\n"
            #    # "c:       {:.3f}\n"
            #    # "d:       {:.3f}\n"
            #     "saturation: {:.1f}%\n"
            #     "".format(
            #        # *popt,
            #         saturation,
            #     ),
            #     color='k',
            #    # fontsize=fs_slope,
            # )

        else:
            asymp = 'none'
            rmax = 'none'
            saturation = 'none'

    except Exception as e:
        msg = "gen.log. \t binned: {} \t m: {} \t {}\n".format(
            binned, monotonicity, str(e))
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

    # extra axes
    # syt
    # ax_syt = ax.twinx()

    # r
    ax.set_xlabel('$T^r$ [$^{{\circ}}$C]')
    ax.set_ylabel("$P^{" + str(q) + "}$ [mmh$^{-1}$]")

    # rh
    # ax_rh = ax.twinx()
    # ax_rh.spines["right"].set_position(("axes", 1.04))
    # _make_patch_spines_invisible(ax_rh)
    # ax_rh.spines["right"].set_visible(True)
    # ax_rh.set_ylabel('RH [%]')
    # ax_rh.yaxis.label.set_color(colors['rrh'])
    # ax_rh.tick_params(axis='y', colors=colors['rrh'])

    # grad
    # ax_grad = ax.twinx()
    # ax_grad.spines["right"].set_position(("axes", 1.04))
    # _make_patch_spines_invisible(ax_grad)
    # ax_grad.spines["right"].set_visible(True)
    # ax_grad.set_ylabel(twcol + ' [°C/h]')
    # ax_grad.yaxis.label.set_color(colors['grad'])
    # ax_grad.tick_params(axis='y', colors=colors['grad'])

    # lat
    # ax_lat = ax.twinx()
    # ax_lat.spines["right"].set_position(("axes", 1.08))
    # _make_patch_spines_invisible(ax_lat)
    # ax_lat.spines["right"].set_visible(True)
    # ax_lat.set_ylabel('lat')
    # ax_lat.yaxis.label.set_color(colors['lat'])
    # ax_lat.tick_params(axis='y', colors=colors['lat'])

    # k_burst
    # ax_k = ax.twinx()
    # ax_k.spines["right"].set_position(("axes", 1.12))
    # _make_patch_spines_invisible(ax_k)
    # ax_k.spines["right"].set_visible(True)
    # ax_k.set_ylabel('avg. lifetime')
    # ax_k.yaxis.label.set_color(colors['k_burst'])
    # ax_k.tick_params(axis='y', colors=colors['k_burst'])

    # plot syt
    # if 'sat' in xcol:
    #     color = colors['sdt']
    # elif 'sdt' in xcol:
    #     color = colors['sat']
    # lns.append(ax_syt.plot(
    #     v_binned[xcol], v_binned[ycol],
    #     marker='o', color=color, ms=2,
    #     label=ycol
    # ))

    # plot rh
    # lns.append(ax_rh.plot(
    #     v_binned[xcol], v_binned[rhcol],
    #     marker='o', color=colors['rrh'], ms=2,
    #     label=rhcol
    # ))
    # lns.append(ax_rh.plot(
    #     v_binned[xcol], v_binned['rh_td_2'],
    #     marker='o', color=colors['rh'], ms=2,
    #     label='rh_td_2'))
    # ax_rh.set_ylim(75, 90)

    # plot grad
    # lns.append(ax_grad.plot(
    #     v_binned[xcol], v_binned[twcol],
    #     marker='o', color=colors['grad'], ms=2,
    #     label=twcol
    # ))

    # plot lat mean
    # lns.append(ax_lat.plot(
    #     v_binned[xcol], v_binned['lat'], marker='o', ms=2, color=colors['lat'],
    #     label='lat_mean'
    # ))

    # plot k_burst
    # lns.append(ax_k.plot(
    #     v_binned[xcol], v_binned['k_burst'],
    #     marker='o', color=colors['k_burst'], ms=2,
    #     label='avg. lifetime'
    # ))

    # ax_lat.fill_between(
    #     v_binned[xcol],
    #     v_binned['lat'] - v_binned['lat_std'], v_binned['lat'] + v_binned['lat_std'],
    #     color='b', alpha=.2, lw=0)

#     # density
#     ax_d = ax.twinx()
#     _make_patch_spines_invisible(ax_d)
#     xmin, xmax = ax.get_xlim()
#
#     plot_hist(
#         v_unbinned[xcol], bins=100, log_bins=False, density=True,
#         floor=False, ax=ax_d,
#         linestyle='-',
#         # lw=1,
#         color='darkolivegreen',
#         # marker='o',
#         # ms=2,
#         alpha=.0)
#
#     # get data to fill lines
#     x, y = ax_d.lines[0].get_data()
#     fbl = ax_d.fill_between(
#         x, y, alpha=.2, zorder=0, color='darkolivegreen',
#         label='probability density of $T^r$',
#     )
#     ax_d.get_yaxis().set_ticks([])
#     ax_d.set_xlim(xmin, xmax)

    # figure stuff
    ax.grid()

    # paper: same axis limits for nta_neg_box and nta_pos_box
    ax.set_xlim(24.9, 28.63)
    ax.set_ylim(.85, 4.55)

    # legend
    lns = [l[0] for l in lns]
    # lns.append(fbl)
    labs = [l.get_label() for l in lns]
    # labs.append(fbl.get_label())
    if 'neg' in args.selection:
        ax.legend([l for l in lns], labs, loc='lower left',
                  facecolor='white', framealpha=1)
    elif 'pos' in args.selection:
        ax.legend([l for l in lns], labs, loc='upper left',
                  facecolor='white', framealpha=1)

    # ax.legend(fontsize='x-small', loc='upper left')
    # ax_syt.legend(fontsize='x-small', loc='upper right')

    # title
    # ax.set_title(
    #     "{} vs {} (grad tw: {}) | n_bursts: {} ({} per bin)"
    #     " | {} | {} | {}".format(
    #         xcol, var, twcol, n_bursts, pts_per_bin, rbstr, gradstr, rhrstr)
    # )

    # store
    os.makedirs(picfolder + 'sxt_vs_r/', exist_ok=True)
    picfile = picfolder + \
        'sxt_vs_r/{}_{}_{}_{}_{}_{}_vs_{}_q{}_tw_{}'.format(
            sname, season, rbstr, gradstr, rhrstr, xcol, var, q, twcol)

    # pdf
    fig.savefig(
        picfile + '.pdf',
        format='pdf',
        bbox_inches='tight', pad_inches=0,
        dpi=600,
    )
    # png
    fig.savefig(
        picfile + '.png',
        format='png',
        bbox_inches='tight', pad_inches=0,
        dpi=600,
    )
    # svg
    fig.savefig(
        picfile + '.svg',
        format='svg',
        bbox_inches='tight', pad_inches=0,
        dpi=600,
    )
    print(picfile + '.xxx')
    plt.close()

    return data


def _sxt_subset_v(
        v, sxtcol, td, twcol, season, hod, wo_r_before, grad, rh_range,
        var):

    # parameters
    sxt_first_q = .03
    sxt_last_q = .97

    vt = v

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

    # relative humidity range?
    # vt['rrh_td_2'] = vt['RH'].rolling(24, axis=1).mean()[-2]
    if rh_range == 'wrhmean':
        mean_rh = vt['RH', -td].mean()
        std_rh = vt['RH', -td].std()
        vt = vt.loc[
            (vt['RH', -td] >= mean_rh - std_rh/4) &
            (vt['RH', -td] <= mean_rh + std_rh/4)
        ]
    elif rh_range == 'wrh80':
        vt = vt.loc[vt['RH', -td] >= vt['RH', -td].quantile(.8)]
    elif rh_range == 'wrrhmean':
        mean_rh = vt['rrh_td_2'].mean()
        std_rh = vt['rrh_td_2'].std()
        vt = vt.loc[
            (vt['rrh_td_2'] >= mean_rh - std_rh/4) &
            (vt['rrh_td_2'] <= mean_rh + std_rh/4)
        ]
    elif rh_range == 'wrrh80':
        vt = vt.loc[vt['rrh_td_2'] >= vt['rrh_td_2'].quantile(.8)]
    elif rh_range == 'worhs':
        pass
    else:
        raise ValueError

    # compute daily avg temps
    if (not args.selection == 'N_tropical_W_tcs_JASO' and not
            args.selection == 'S_tropical_W_tcs_DJFMA'):

        xcol = 'r{}_td_{}'.format(sxtcol, td)
        # vt['rsdt_td_{}'.format(td)] = vt['sdt'].rolling(24, axis=1).mean()[-td]
        vt['rsat_td_{}'.format(td)] = vt['sat'].rolling(24, axis=1).mean()[-td]
        # vt['rrh_td_{}'.format(td)] = vt['RH'].rolling(24, axis=1).mean()[-td]

    # get rid of nans
    vt = vt[~vt[xcol].isnull()]
    sxtcolstr = sxtcol

    # cut off left and right ends
    xfirst = vt[xcol].quantile(sxt_first_q)
    xlast = vt[xcol].quantile(sxt_last_q)
    vt = vt[(vt[xcol] >= xfirst) & (vt[xcol] <= xlast)]

    # strings
    varstr = var
    tdstr = str(td)
    gradstr = grad
    rhrstr = rh_range

    return vt, sxtcolstr, tdstr, sstr, hodstr, rbstr, gradstr, rhrstr, varstr


def compute_sxt_vs_r(
        v, sxtcol, td, twcol, season, hod, wo_r_before, grad, rh_range,
        var, q):

    n_bins = 40

    # subset
    vt, sxtcolstr, tdstr, sstr, hodstr, rbstr, gradstr, rhrstr, varstr = \
        _sxt_subset_v(
            v, sxtcol, td, twcol, season, hod, wo_r_before, grad, rh_range,
            var
        )

    # return empty series if there are no sxt values (think sst)
    if len(vt) <= n_bins:
        print('not enough datapoints!')
        return {}

    # fits - all
    # data = _sxt_vs_r_fits_per_loc(loc, vt, td, var, binned=False)
    data = {}

    # binning by sxt
    xcol = 'r{}_td_{}'.format(sxtcol, td)
    binedges = np.quantile(vt[xcol], np.linspace(0, 1, n_bins + 1))
    try:
        vt['{}_d'.format(xcol)] = pd.cut(
            vt[xcol], binedges, include_lowest=True, duplicates='drop'
        )
    except IndexError:
        print(vt.shape, binedges)
        print(vt[xcol].head())
        raise

    # find better solution to this
    # vt['rh_td_2'] = vt['RH', -2]

    # groupby x
    gvt = vt.groupby('{}_d'.format(xcol))

    # r_q90
    vt_q = gvt[var].quantile(q/100).to_frame()

    # sxt
    vt_q[xcol] = gvt[xcol].mean()

    # lat
    vt_q['lat'] = gvt['lat'].mean()
    vt_q['lat_std'] = gvt['lat'].std()

    # syt
    if sxtcol == 'sat':
        ycol = 'rsdt_td_{}'.format(td)
    elif sxtcol == 'sdt':
        ycol = 'rsat_td_{}'.format(td)
    # vt_q[ycol] = gvt[ycol].mean()

    # rh
    rhcol = 'rrh_td_{}'.format(td)
    # vt_q[rhcol] = gvt[rhcol].mean()
    # vt_q['rh_td_2'] = gvt['rh_td_2'].mean()

    # grad
    vt_q[twcol] = gvt[twcol].mean()

    # pts_per_bin
    vt_q['size'] = gvt.size()
    pts_per_bin = int(vt_q['size'].mean())

    # burst duration
    # vt_q['k_burst'] = gvt['k_burst'].mean()

    # fits - quantile
    data_q = _sxt_vs_r_fits_per_loc(
        vt, gvt, vt_q, var, True, xcol, ycol, rhcol, twcol, len(vt),
        pts_per_bin, rbstr, gradstr, rhrstr, q,
    )
    data_q = {key + f'_q{q}': value for key, value in data_q.items()}

    # combine
    data.update(data_q)

    # x/y-min/max
    data['xq1'] = vt[xcol].quantile(.01)
    data['xq99'] = vt[xcol].quantile(.99)
    data['yq1'] = vt[var].quantile(.01)
    data['yq99'] = vt[var].quantile(.99)

    # pts per bin
    data['n_bursts'] = len(vt)
    data['pts_per_bin'] = pts_per_bin

    # add subset strings
    data = {
        '{}_{}_{}_{}_{}_{}_{}_{}_'.format(
            sxtcolstr, tdstr, sstr, hodstr,
            rbstr, gradstr, rhrstr, varstr) + key:
        value for key, value in data.items()
    }

    return data


def sxt_vs_r(v, season, which, twcol):

    td = 2
    vari = 'r_max'
    wo_r_befores = [True]  # [True, False]
    grads = ['agrad']  # ['ngrad', 'pgrad', 'agrad']
    rh_ranges = ['worhs']  # ['wrhmean', 'wrh80', 'wrrhmean', 'wrrh80']
    qs = [90]  # [90, 95, 99]

    for wo_r_before in wo_r_befores:
        for grad in grads:
            for rh_range in rh_ranges:
                for q in qs:
                    compute_sxt_vs_r(
                        v, sxtcol=which, td=td, twcol=twcol, season=season,
                        hod='all', wo_r_before=wo_r_before, grad=grad,
                        rh_range=rh_range, var=vari, q=q,
                    )


def sat_vs_r_by_td(v, season, hod=False):

    # parameters
    tds_sat_vs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    vt = v.loc[v[season]]

    if not hod:

        fig, axs = plt.subplots(3, 4, figsize=(19.20, 10.80))
        axs = axs.flatten()

        ax_xmins = []
        ax_xmaxs = []
        ax_ymins = []
        ax_ymaxs = []
        ax_sxts = []
        sxt_ymins = []
        sxt_ymaxs = []
        for td, ax in zip(tds_sat_vs, axs):
            _, _, ax_sxt = _sat_vs_r_ax_level(vt, td, fig=fig, ax=ax)

            # min max xr
            ax_xmin, ax_xmax = ax.get_xlim()
            ax_xmins.append(ax_xmin)
            ax_xmaxs.append(ax_xmax)
            # min max yr
            ax_ymin, ax_ymax = ax.get_ylim()
            ax_ymins.append(ax_ymin)
            ax_ymaxs.append(ax_ymax)
            # min max ysxt
            ax_sxts.append(ax_sxt)
            sxt_ymin, sxt_ymax = ax_sxt.get_ylim()
            sxt_ymins.append(sxt_ymin)
            sxt_ymaxs.append(sxt_ymax)

            ax.set_title('td={} | n_bursts={}'.format(td, len(vt)))

        for ax, ax_sxt in zip(axs, ax_sxts):
            ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
            ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
            ax_sxt.set_ylim(min(sxt_ymins), max(sxt_ymaxs))

        # store
        fig.tight_layout()
        picfile = picfolder + '{}_{}_sat_vs_r_by_td.pdf'.format(sname, season)
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close()

    elif hod:

        os.makedirs(picfolder + 'sat_vs_r_by_td_hod/', exist_ok=True)

        for td in tds_sat_vs:

            fig, axs = plt.subplots(4, 2, figsize=(19.20, 10.80))
            axs = axs.flatten()

            ax_xmins = []
            ax_xmaxs = []
            ax_ymins = []
            ax_ymaxs = []
            ax_sxts = []
            sxt_ymins = []
            sxt_ymaxs = []

            for hod, ax in zip([0, 12, 3, 15, 6, 18, 9, 21], axs):
                vtt = vt.loc[vt['dtime'].dt.hour == hod]
                _, _, ax_sxt = _sat_vs_r_ax_level(vtt, td, fig=fig, ax=ax)

                # min max xr
                ax_xmin, ax_xmax = ax.get_xlim()
                ax_xmins.append(ax_xmin)
                ax_xmaxs.append(ax_xmax)
                # min max yr
                ax_ymin, ax_ymax = ax.get_ylim()
                ax_ymins.append(ax_ymin)
                ax_ymaxs.append(ax_ymax)
                # min max ysxt
                ax_sxts.append(ax_sxt)
                sxt_ymin, sxt_ymax = ax_sxt.get_ylim()
                sxt_ymins.append(sxt_ymin)
                sxt_ymaxs.append(sxt_ymax)

                ax.set_title('hod={}'.format(hod))

            for ax, ax_sxt in zip(axs, ax_sxts):
                ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
                ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
                ax_sxt.set_ylim(min(sxt_ymins), max(sxt_ymaxs))

            # store
            fig.tight_layout()
            picfile = picfolder + \
                'sat_vs_r_by_td_hod/{}_{}_{}_sat_vs_r_by_hod.pdf'.format(
                    sname, season, td)
            fig.savefig(picfile)
            fig.savefig(picfile[:-3] + 'png')
            print(picfile)
            plt.close()


def sat_vs_r_by_hod(v, season, td=False):

    # parameters
    tds_sat_vs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    vt = v.loc[v[season]]

    if not td:

        fig, axs = plt.subplots(4, 2, figsize=(19.20, 10.80))
        axs = axs.flatten()

        ax_xmins = []
        ax_xmaxs = []
        ax_ymins = []
        ax_ymaxs = []
        ax_sxts = []
        sxt_ymins = []
        sxt_ymaxs = []

        for hod, ax in zip([0, 12, 3, 15, 6, 18, 9, 21], axs):
            vtt = vt.loc[vt['dtime'].dt.hour == hod]
            _, _, ax_sxt = _sat_vs_r_ax_level(vtt, td=0, fig=fig, ax=ax)

            # min max xr
            ax_xmin, ax_xmax = ax.get_xlim()
            ax_xmins.append(ax_xmin)
            ax_xmaxs.append(ax_xmax)
            # min max yr
            ax_ymin, ax_ymax = ax.get_ylim()
            ax_ymins.append(ax_ymin)
            ax_ymaxs.append(ax_ymax)
            # min max ysxt
            ax_sxts.append(ax_sxt)
            sxt_ymin, sxt_ymax = ax_sxt.get_ylim()
            sxt_ymins.append(sxt_ymin)
            sxt_ymaxs.append(sxt_ymax)

            ax.set_title('hod={}'.format(hod))

        for ax, ax_sxt in zip(axs, ax_sxts):
            ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
            ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
            ax_sxt.set_ylim(min(sxt_ymins), max(sxt_ymaxs))

        # store
        fig.tight_layout()
        picfile = picfolder + '{}_{}_sat_td_0_vs_r_by_hod.pdf'.format(
            sname, season)
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close()

    elif td:

        os.makedirs(picfolder + 'sat_vs_r_by_hod_td/', exist_ok=True)

        for hod in range(0, 24, 3):

            vtt = vt.loc[vt['dtime'].dt.hour == hod]

            fig, axs = plt.subplots(3, 4, figsize=(19.20, 10.80))
            axs = axs.flatten()

            ax_xmins = []
            ax_xmaxs = []
            ax_ymins = []
            ax_ymaxs = []
            ax_sxts = []
            sxt_ymins = []
            sxt_ymaxs = []

            for td, ax in zip(tds_sat_vs, axs):
                _, _, ax_sxt = _sat_vs_r_ax_level(vtt, td, fig=fig, ax=ax)

                # min max xr
                ax_xmin, ax_xmax = ax.get_xlim()
                ax_xmins.append(ax_xmin)
                ax_xmaxs.append(ax_xmax)
                # min max yr
                ax_ymin, ax_ymax = ax.get_ylim()
                ax_ymins.append(ax_ymin)
                ax_ymaxs.append(ax_ymax)
                # min max ysxt
                ax_sxts.append(ax_sxt)
                sxt_ymin, sxt_ymax = ax_sxt.get_ylim()
                sxt_ymins.append(sxt_ymin)
                sxt_ymaxs.append(sxt_ymax)

                ax.set_title('td={}'.format(td))

            for ax, ax_sxt in zip(axs, ax_sxts):
                ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
                ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
                ax_sxt.set_ylim(min(sxt_ymins), max(sxt_ymaxs))

            # store
            fig.tight_layout()
            picfile = picfolder + \
                'sat_vs_r_by_hod_td/{}_{}_{}_sat_vs_r_by_td.pdf'.format(
                    sname, season, hod)
            fig.savefig(picfile)
            fig.savefig(picfile[:-3] + 'png')
            print(picfile)
            plt.close()


def _sat_vs_r_ax_level(v, td, fig=None, ax=None):

    n_bins = 15
    sxt_first_q = .03
    sxt_last_q = .97

    # plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

    satcol = 'sat_td_{}'.format(td)
    sdtcol = 'sdt_td_{}'.format(td)
    rhcol = 'RH_td_{}'.format(td)

    # there seem to be some nan sxt values, get rid of them
    v = v[~v[satcol].isnull()]

    # cut off left and right ends
    xfirst = v[satcol].quantile(sxt_first_q)
    xlast = v[satcol].quantile(sxt_last_q)
    v = v[(v[satcol] >= xfirst) & (v[satcol] <= xlast)]

    # binning of sxt
    binedges = np.quantile(
        v[satcol], np.linspace(0, 1, n_bins + 1))
    v['{}_d'.format(satcol)] = pd.cut(
        v[satcol], binedges, include_lowest=True, duplicates='drop')
    gvt = v.groupby('{}_d'.format(satcol))
    vt_q = gvt['r'].quantile(.9).to_frame()
    vt_q['size'] = gvt.size()
    vt_q[satcol] = vt_q.index
    vt_q[satcol] = vt_q[satcol].apply(lambda x: x.mid)
    vt_q[satcol] = vt_q[satcol].astype(float)
    # pts_per_bin = int(vt_q['size'].mean())

    # plotting
    ax_sdt = ax.twinx()
    ax_rh = ax.twinx()
    ax_rh.spines["right"].set_position(("axes", 1.06))
    _make_patch_spines_invisible(ax_rh)
    # ax_rh.spines["right"].set_visible(True)

    # plot sat vs r
    color = 'k'
    l1 = ax.plot(
        vt_q[satcol], vt_q['r'], color=color,
        label=satcol,
        marker='o',
        ms=3,
    )

    # plot RH
    color = 'c'
    vt_q[rhcol] = gvt[rhcol].mean()
    l2 = ax_rh.plot(
        vt_q[satcol], vt_q[rhcol], color=color,
        label=rhcol,
    )
    ax_rh.set_ylim(20, 100)
    ax_rh.get_yaxis().set_visible(False)

    # plot sdt
    color = 'g'
    vt_q[sdtcol] = gvt[sdtcol].mean()
    l3 = ax_sdt.plot(
        vt_q[satcol], vt_q[sdtcol], color=color,
        label=sdtcol,
    )

    # X/Y-labels
    ax.grid()

    # legend
    lns = l1+l2+l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize='x-small')

    return fig, ax, ax_sdt


def sdt_vs_sat_vs_x_by_td(v, var, season, hod=False):

    # parameters
    tds_sat_vs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    vt = v.loc[v[season]]

    if not hod:

        fig, axs = plt.subplots(3, 4, figsize=(19.20, 10.80))
        axs = axs.flatten()

        ax_xmins = []
        ax_xmaxs = []
        ax_ymins = []
        ax_ymaxs = []
        for td, ax in zip(tds_sat_vs, axs):
            _sdt_vs_sat_vs_x_ax_level(vt, td, var, fig=fig, ax=ax)

            # min max xr
            ax_xmin, ax_xmax = ax.get_xlim()
            ax_xmins.append(ax_xmin)
            ax_xmaxs.append(ax_xmax)
            # min max yr
            ax_ymin, ax_ymax = ax.get_ylim()
            ax_ymins.append(ax_ymin)
            ax_ymaxs.append(ax_ymax)

            ax.set_title('td={} | n_bursts: {}'.format(td, len(vt)))

        for ax in axs:
            ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
            ax.set_ylim(min(ax_ymins), max(ax_ymaxs))

        # store
        fig.tight_layout()
        picfile = picfolder + '{}_{}_sdt_td_x_vs_sat_td_x_vs_{}.pdf'.format(
            sname, season, var)
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close()

    elif hod:
        pass


def sdt_vs_sat_vs_x_by_hod(v, var, season, td=False):

    # parameters
    tds_sat_vs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    vt = v.loc[v[season]]

    if not td:
        pass

    elif td:

        os.makedirs(picfolder + 'sdt_vs_sat_vs_X_by_hod_td/', exist_ok=True)

        for hod in range(0, 24, 3):

            vtt = vt.loc[vt['dtime'].dt.hour == hod]

            fig, axs = plt.subplots(3, 4, figsize=(19.20, 10.80))
            axs = axs.flatten()

            ax_xmins = []
            ax_xmaxs = []
            ax_ymins = []
            ax_ymaxs = []

            for td, ax in zip(tds_sat_vs, axs):
                _sdt_vs_sat_vs_x_ax_level(vtt, td, var, fig=fig, ax=ax)

                # min max xr
                ax_xmin, ax_xmax = ax.get_xlim()
                ax_xmins.append(ax_xmin)
                ax_xmaxs.append(ax_xmax)
                # min max yr
                ax_ymin, ax_ymax = ax.get_ylim()
                ax_ymins.append(ax_ymin)
                ax_ymaxs.append(ax_ymax)

                ax.set_title('td={} | n_bursts={}'.format(td, len(vtt)))

            for ax in axs:
                ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
                ax.set_ylim(min(ax_ymins), max(ax_ymaxs))

            # store
            fig.tight_layout()
            picfile = (
                picfolder + "sdt_vs_sat_vs_X_by_hod_td/"
                "{}_{}_{}_sdt_vs_sat_vs_{}_by_td.pdf".format(
                    sname, season, hod, var))
            fig.savefig(picfile)
            fig.savefig(picfile[:-3] + 'png')
            print(picfile)
            plt.close()


def _sdt_vs_sat_vs_x_ax_level(v, td, var, fig=None, ax=None):

    mincnt = 20
    gridsize = 30

    # plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

    satcol = 'sat_td_{}'.format(td)
    sdtcol = 'sdt_td_{}'.format(td)

    # there seem to be some nan sxt values, get rid of them
    v = v[~v[satcol].isnull()]
    v = v[~v[sdtcol].isnull()]

    if var == 'r':
        # vmin = v['r'].quantile(.05)
        # vmax = v['r'].quantile(.95)
        vmin = 2
        vmax = 6
    else:
        vmin = None
        vmax = None

    hb = ax.hexbin(
        v[sdtcol],
        v[satcol],
        C=None if var == 'n_nodes' else v[var],
        gridsize=gridsize,
        xscale='linear', yscale='linear',
        marginals=False if var == 'n_nodes' else True,
        bins='log' if var == 'n_nodes' else None,
        mincnt=1 if var == 'n_nodes' else mincnt,
        cmap='viridis_r',
        vmin=vmin, vmax=vmax,
        alpha=1,
        edgecolors='none',
        reduce_C_function=_custom_q90,
        # linewidths=0.01,
    )

    # diagonal
    # left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    ax.plot([bottom, top], [bottom, top], c='k', lw=.2, alpha=.5)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    # cb = fig.colorbar(obj['pc'], cax=cax)
    fig.colorbar(hb, cax=cax, extend='both')

    return fig, ax


def td_vs_alpha(season):

    os.makedirs(picfolder + 'td_vs_alpha/', exist_ok=True)

    which = 'sat'
    mincnt = 20*15
    tds = list(range(0, 24)) + list(range(24, 51, 3)) + [72, 96]

    gl_n_nodes = pd.read_feather(storefolder + 'gl_r{}_p{}_{}.feather'.format(
        r, p, season))

    gls = []
    for td in tds:

        gl = pd.read_pickle(
            storefolder + glsxtvsrfolder +
            'gl_{}_td_{}_vs_R_linregress_{}.pickle'.format(which, td, season))
        gl['R2'] = gl['exp_rvalue']**2

        # mask non-significant values
        # gl.loc[gl['exp_pvalue'] > .05, 'exp_alpha'] = np.nan
        # gl.loc[gl['exp_pvalue_q90'] > .05, 'exp_alpha_q90'] = np.nan

        # mask locations with less than mincnt events
        gl.loc[gl_n_nodes['n_nodes'] <= mincnt, :] = np.nan

        # subset to locations
        gl = gl.loc[locs]

        # append
        gl['td'] = td
        gls.append(gl)

    gl = pd.concat(gls, axis=0)

    # change td to int rather than category
    # gl['td'] = gl['td'].astype('category')
    # gl['td'].cat.set_categories(range(0, 97), inplace=True)

    for var in ['exp_alpha', 'exp_rvalue', 'exp_pvalue', 'exp_stderr', 'R2']:

        picfile = picfolder + 'td_vs_alpha/' + '{}_{}_{}_td_vs_{}.pdf'.format(
            sname, season, which, var)

        print('boxplot {} ..'.format(picfile))
        cp = sns.catplot(
            x='td', y=var,
            kind='point',
            data=gl,
            # dodge=True,
            # whis=0,
            # showfliers=False,
            # estimator=np.mean,
            ci='sd',
        )

        cp.ax.grid()
        # cp.ax.set_ylim(0, 100)
        cp.fig.set_size_inches((19.20, 10.80))

        cp.fig.tight_layout()
        cp.fig.savefig(picfile)
        cp.fig.savefig(picfile[:-3] + 'png')
        plt.close(cp.fig)


def td_vs_alpha_by_hod(season):

    os.makedirs(picfolder + 'td_vs_alpha/', exist_ok=True)

    which = 'sat'
    mincnt = 20*15

    hods = [0, 3, 6, 9, 12, 15, 18, 21]
    tds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    gl_n_nodes = pd.read_feather(storefolder + 'gl_r{}_p{}_{}.feather'.format(
        r, p, season))

    gls = []

    for hod in hods:
        for td in tds:

            hstr = '_hod_{:02d}'.format(hod)

            gl = pd.read_pickle(
                storefolder + glsxtvsrfolder +
                'gl_{}_td_{}_vs_R_linregress{}_{}.pickle'.format(
                    which, td, hstr, season))
            gl['R2'] = gl['exp_rvalue']**2

            # mask non-significant values
            # gl.loc[gl['exp_pvalue'] > .05, 'exp_alpha'] = np.nan
            # gl.loc[gl['exp_pvalue_q90'] > .05, 'exp_alpha_q90'] = np.nan

            # mask locations with less than mincnt events
            gl.loc[gl_n_nodes['n_nodes'] <= mincnt, :] = np.nan

            # subset to locations
            gl = gl.loc[locs]

            # append
            gl['hod'] = hod
            gl['td'] = td
            gls.append(gl)

    gl = pd.concat(gls, axis=0)

    # change td to int rather than category
    # gl['td'] = gl['td'].astype('category')
    # gl['td'].cat.set_categories(range(0, 97), inplace=True)

    fig, axs = plt.subplots(4, 2, figsize=(19.20, 10.80))
    axs = axs.flatten()

    ggl = gl.groupby('hod')

    ax_ymins = []
    ax_ymaxs = []
    for (name, glt), ax in zip(ggl, axs):

        sns.pointplot(
            x='td', y='exp_alpha',
            data=glt,
            # dodge=True,
            # whis=0,
            # showfliers=False,
            # estimator=np.mean,
            ci='sd',
            ax=ax,
        )

        # min max yr
        ax_ymin, ax_ymax = ax.get_ylim()
        ax_ymins.append(ax_ymin)
        ax_ymaxs.append(ax_ymax)

        ax.grid()
        ax.set_title('hod {:02d}'.format(name))
        # cp.ax.set_ylim(0, 100)

    for ax in axs:
        ax.set_ylim(min(ax_ymins), max(ax_ymaxs))

    picfile = picfolder + \
        'td_vs_alpha/' + '{}_{}_{}_td_vs_{}_by_hod.pdf'.format(
            sname, season, which, 'exp_alpha')

    fig.tight_layout()
    fig.savefig(picfile)
    fig.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig)


def selection_on_map():

    which = 'sat'
    sstr_select = '_DJF'
    sstr_plot = '_JJA'
    td1 = 0
    td2 = 6
    td3 = 30

    # gl n_nodes
    gl_n_JJA = pd.read_feather(
        storefolder + 'gl_r{}_p{}_JJA.feather'.format(r, p))
    gl_n_DJF = pd.read_feather(
        storefolder + 'gl_r{}_p{}_DJF.feather'.format(r, p))

    # load gl_maps
    gl1 = pd.read_pickle(
        storefolder + glsxtvsrfolder +
        'gl_{}_td_{}_vs_R_linregress{}.pickle'.format(
            which, td1, sstr_select))
    gl2 = pd.read_pickle(
        storefolder + glsxtvsrfolder +
        'gl_{}_td_{}_vs_R_linregress{}.pickle'.format(
            which, td2, sstr_select))
    gl3 = pd.read_pickle(
        storefolder + glsxtvsrfolder +
        'gl_{}_td_{}_vs_R_linregress{}.pickle'.format(
            which, td3, sstr_select))
    gl4 = pd.read_pickle(
        storefolder + glsxtvsrfolder +
        'gl_{}_td_{}_vs_R_linregress{}.pickle'.format(
            which, td1, sstr_plot))
    gl5 = pd.read_pickle(
        storefolder + glsxtvsrfolder +
        'gl_{}_td_{}_vs_R_linregress{}.pickle'.format(
            which, td2, sstr_plot))
    gl6 = pd.read_pickle(
        storefolder + glsxtvsrfolder +
        'gl_{}_td_{}_vs_R_linregress{}.pickle'.format(
            which, td3, sstr_plot))

    gl_fits = {
        '{}_td_{}{}'.format(which, td1, sstr_select): gl1,
        '{}_td_{}{}'.format(which, td1, sstr_plot): gl4,
        '{}_td_{}{}'.format(which, td2, sstr_select): gl2,
        '{}_td_{}{}'.format(which, td2, sstr_plot): gl5,
        '{}_td_{}{}'.format(which, td3, sstr_select): gl3,
        '{}_td_{}{}'.format(which, td3, sstr_plot): gl6,
    }

    kwds_basemap = {'projection': 'cyl',
                    # 'lon_0': 0,
                    # 'lat_0': 0,
                    'llcrnrlon': 60,
                    'urcrnrlon': 420,
                    'llcrnrlat': -50,
                    'urcrnrlat': 50,
                    'resolution': 'c'}

    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    axs = axs.flatten()

    for (title, glt), ax in zip(gl_fits.items(), axs):

        # basemap
        m = Basemap(ax=ax, **kwds_basemap)

        # exp_alpha
        z = np.roll(
            glt['exp_alpha'].values.reshape(400, 1440)[::-1], 480, axis=1)
        im = m.imshow(
            z, cmap='PiYG', interpolation='none',
            vmin=-28,
            vmax=28,
        )

        # scatter
        m.scatter(gl_n_JJA.loc[locs, 'lon'], gl_n_DJF.loc[locs, 'lat'],
                  latlon=True,
                  ax=ax,
                  s=20, facecolors='none', edgecolors='darkorange',
                  zorder=10)

        # coastlines, parallels/meridians
        m.drawcoastlines(linewidth=1.5, ax=ax, zorder=10)
        parallels = [-23, 0, 23]  # np.arange(-50, 50, 5)
        meridians = np.arange(-180, 180, 60)  # (-180, 180, 5)
        m.drawparallels(parallels,
                        linewidth=.3,
                        labels=[True, False, False, False])
        m.drawmeridians(meridians,
                        linewidth=.3,
                        labels=[False, False, False, False])

        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=.05)
        fig.colorbar(im, cax=cax)

        # title
        ax.set_title('{} alpha'.format(title))

    # save figure
    fig.tight_layout()
    picfile = picfolder + '{}_map.pdf'.format(sname)
    fig.savefig(picfile)
    fig.savefig(picfile[:-3] + 'png')
    print(picfile)
    plt.close(fig)

    return gl_fits


def selection_on_map_by_hod_td(season):

    os.makedirs(picfolder + 'map_plots/', exist_ok=True)

    which = 'sat'

    lons, lats = np.meshgrid(np.arange(-179.875, 180.125, .25),
                             np.arange(49.875, -50.125, -.25))

    # min/max lat/lon
    gl_n_nodes = pd.read_feather(storefolder + 'gl_r{}_p{}.feather'.format(
        r, p))

    lon_min = gl_n_nodes.loc[locs, 'lon'].min()
    lon_max = gl_n_nodes.loc[locs, 'lon'].max()
    lat_min = gl_n_nodes.loc[locs, 'lat'].min()
    lat_max = gl_n_nodes.loc[locs, 'lat'].max()

    dlon = (lon_max - lon_min) * .05
    dlat = (lat_max - lat_min) * .05

    kwds_basemap = {'projection': 'cyl',
                    # 'lon_0': 0,
                    # 'lat_0': 0,
                    'llcrnrlon': lon_min - dlon,
                    'urcrnrlon': lon_max + dlon,
                    'llcrnrlat': lat_min - dlat,
                    'urcrnrlat': lat_max + dlat,
                    'resolution': 'c'}

    hods = [0, 3, 6, 9, 12, 15, 18, 21]
    tds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    for hod in hods:

        fig, axs = plt.subplots(3, 4, figsize=(19.20, 10.80))
        axs = axs.flatten()

        for td, ax in zip(tds, axs):

            print(hod, td)

            hstr = '_hod_{:02d}'.format(hod)

            gl = pd.read_pickle(
                storefolder + glsxtvsrfolder +
                'gl_{}_td_{}_vs_R_linregress{}_{}.pickle'.format(
                    which, td, hstr, season))

            # basemap
            m = Basemap(ax=ax, **kwds_basemap)

            # exp_alpha
            gl.loc[~gl.index.isin(locs), 'exp_alpha'] = np.nan
            z = gl['exp_alpha'].values.reshape(400, 1440)
            im = m.pcolormesh(
                lons, lats, z, cmap='PiYG',
                edgecolors='None',
                # interpolation='none',
                vmin=-28,
                vmax=28,
                latlon=True,
            )

            # coastlines, parallels/meridians
            m.drawcoastlines(linewidth=1.5, ax=ax, zorder=10)
            parallels = [-23, 0, 23]  # np.arange(-50, 50, 5)
            meridians = np.arange(-180, 180, 60)  # (-180, 180, 5)
            m.drawparallels(parallels,
                            linewidth=.3,
                            labels=[False, False, False, False])
            m.drawmeridians(meridians,
                            linewidth=.3,
                            labels=[False, False, False, False])

            # colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=.05)
            fig.colorbar(im, cax=cax)

            # title
            ax.set_title('td={}'.format(td))

        # save figure
        fig.tight_layout()
        picfile = picfolder + \
            'map_plots/{}_{}_map_hod_{:02d}_by_td.pdf'.format(
                sname, season, hod)
        # fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close(fig)


def _load_bursts(locs):

    # load burst data
    vs = []
    c = 1
    for loc in locs:
        print('loading bursts loc {:06d}/{:06d}'.format(c, len(locs)))
        v = pd.read_pickle(
            storefolder + partsfolder + '{:06d}.pickle'.format(loc))

        # get rid of unused columns
        if (args.grad_vs_r or args.tc_burst_stats) and not args.t_vs_tmps:
            del v['sdt']
            del v['RH']
            del v['k_burst']
            # delete r at t>0 columns
            rts = v['r'].columns.values[v['r'].columns.values >0]
            for rt in rts:
                del v[('r', rt)]
            if (args.selection == 'N_tropical_W_tcs_JASO' or
                    args.selection == 'S_tropical_W_DJFMA'):
                td = 2
                v['rsat_td_{}'.format(td)] = v['sat'].rolling(
                    24, axis=1).mean()[-td]
                del v['sat']

        vs.append(v)
        c += 1
    print('concat bursts of locations ..')
    v = pd.concat(vs, axis=0)
    # v['r_sum'] = v['r'].iloc[:, 15:].sum(axis=1)
    print('done concating')

    return v


def _load_burst_grads(locs, twcols):

    # load burst data
    vs = []
    c = 1
    for loc in locs:
        print('loading burst sat grads loc {:06d}/{:06d}'.format(c, len(locs)))
        v = pd.read_pickle(
            storefolder + spartsfolder + '{:06d}.pickle'.format(loc))

        # select only columns in twcols
        v = v[twcols]

        vs.append(v)
        c += 1
    print('concat bursts of locations ..')
    v = pd.concat(vs, axis=0)
    print('done concating')

    return v


def _grad_vs_r_fits_per_loc(v, col, var, binned):

    # store data
    data = {}

    # lowess
    # lowess = sm.nonparametric.lowess(v[var], v[col], frac=.05)

    # interpolation
    # interp1d_bins = 50
    # xintp = np.linspace(v[col].min(), v[col].max(), interp1d_bins)
    # if not binned:
    #     f = interp1d(lowess[:, 0], lowess[:, 1], bounds_error=False)
    #     xl = xintp
    #     yl = f(xintp)
    # if binned:
    #     xl = lowess[:, 0]
    #     yl = lowess[:, 1]

    # lowess alpha distribution -> mean, std
    # alphas = (yl[:-1] + (yl[1:]-yl[:-1])/(xl[1:]-xl[:-1])) / yl[:-1]
    # alphas = (alphas - 1) * 100
    # data['alphas_mean'] = alphas.mean()
    # data['alphas_std'] = alphas.std()

    # monotonicity
    # dyl = np.diff(yl)
    # dyl_mean = dyl.mean()
    # if (dyl <= 0).all():
    #     monotonicity = 'decreasing'
    # elif (dyl >= 0).all():
    #     monotonicity = 'increasing'
    # elif dyl_mean > 0:
    #     monotonicity = 'mean increase'
    # elif dyl_mean < 0:
    #     monotonicity = 'mean decrease'
    # else:
    #     monotonicity = 'none'
    # data['monotonicity'] = monotonicity

    # x of global min/max
    # data['xgmax'] = xl[yl.argmax()]
    # data['xgmin'] = xl[yl.argmin()]
    if binned:
        data['xgmin'] = v.at[v[var].idxmin(), col]
        data['xgmax'] = v.at[v[var].idxmax(), col]
        data['gmin'] = v[var].min()
        data['gmax'] = v[var].max()

    # linregress
    tstart = datetime.now()

    # linregress all
    try:
        m, t, rvalue, pvalue, stderr = linregress(v[col], v[var])
        data['m'] = m
        data['t'] = t
        data['rvalue'] = rvalue
        data['pvalue'] = pvalue
        data['stderr'] = stderr
    except Exception as e:
        print("loc ?\tlinregress\tbinned: {}\t{}".format(
            binned, str(e)))
        data['m'] = np.nan
        data['t'] = np.nan
        data['rvalue'] = np.nan
        data['pvalue'] = np.nan
        data['stderr'] = np.nan

    # linregress left
    try:
        vt = v.loc[v[col] <= 0]
        m, t, rvalue, pvalue, stderr = linregress(vt[col], vt[var])
        data['m_left'] = m
        data['t_left'] = t
        data['rvalue_left'] = rvalue
        data['pvalue_left'] = pvalue
        data['stderr_left'] = stderr
    except Exception as e:
        print("loc ?\tlinregress\tbinned left: {}\t{}".format(
            binned, str(e)))
        data['m_left'] = np.nan
        data['t_left'] = np.nan
        data['rvalue_left'] = np.nan
        data['pvalue_left'] = np.nan
        data['stderr_left'] = np.nan

    # linregress right
    try:
        vt = v.loc[v[col] > 0]
        m, t, rvalue, pvalue, stderr = linregress(vt[col], vt[var])
        data['m_right'] = m
        data['t_right'] = t
        data['rvalue_right'] = rvalue
        data['pvalue_right'] = pvalue
        data['stderr_right'] = stderr
    except Exception as e:
        print("loc ?\tlinregress\tbinned right: {}\t{}".format(
            binned, str(e)))
        data['m_right'] = np.nan
        data['t_right'] = np.nan
        data['rvalue_right'] = np.nan
        data['pvalue_right'] = np.nan
        data['stderr_right'] = np.nan

    dt = datetime.now() - tstart
    ptime = 'linregress (binned: {}):\ts={}\tms={}'.format(
        binned,
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)

    return data


def _grad_subset_v(v, twcol, season, hod, wo_r_before, var):

    # twcol, get rid of nans
    vt = v[~v[twcol].isnull()]
    colstr = twcol

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
            vt = vt.loc[vt['r', tdr].isnull()]
        rbstr = 'worb'
    elif not wo_r_before:
        rbstr = 'wrb'

    # r_max or r_mean?
    varstr = var

    # cut off left and right ends?
    # grad_first_q = .03
    # grad_last_q = .97
    # xfirst = vt[twcol].quantile(grad_first_q)
    # xlast = vt[twcol].quantile(grad_last_q)
    # vt = vt[(vt[twcol] >= xfirst) & (vt[twcol] <= xlast)]

    return vt, colstr, sstr, hodstr, rbstr, varstr


def compute_grad_vs_r(v, twcol, season, hod, wo_r_before, var, q):

    # parameters
    n_bins = 25

    # subset
    vt, colstr, sstr, hodstr, rbstr, varstr = _grad_subset_v(
        v, twcol, season, hod, wo_r_before, var)

    # return empty series if there are no sxt values (think sst)
    # if len(vt) <= n_bins:
    #     return pd.DataFrame(index=[loc])  # pd.Series(index=data.keys())

    # fits - all
    data = {}
    # data = _grad_vs_r_fits_per_loc(vt, twcol, var, binned=False)

    # binning by sat grad
    binedges = np.quantile(vt[twcol], np.linspace(0, 1, n_bins + 1))
    try:
        vt['{}_d'.format(twcol)] = pd.cut(
            vt[twcol], binedges, include_lowest=True, duplicates='drop'
        )
    except IndexError:
        print('-' * 80)
        # print(loc)
        print(vt.shape, binedges)
        print(vt[twcol].head())
        raise

    vt.to_pickle('test.pkl')

    gvt = vt.groupby('{}_d'.format(twcol))
    if var == 'max_sustained_wind' or var == 'category':
        vt_binned = gvt[var].mean().to_frame()
    else:
        vt_binned = gvt[var].quantile(q/100).to_frame()

    vt_binned[twcol] = gvt[twcol].mean()
    # vt_binned[twcol] = vt_binned.index
    # vt_binned[twcol] = vt_binned[twcol].apply(lambda x: x.mid)
    # vt_binned[twcol] = vt_binned[twcol].astype(float)
    vt_binned['size'] = gvt.size()
    pts_per_bin = int(vt_binned['size'].mean())

    # fits - quantile
    data_q = _grad_vs_r_fits_per_loc(vt_binned, twcol, var, binned=True)
    data_q = {key + '_binned': value for key, value in data_q.items()}

    # combine
    data.update(data_q)

    # x/y-min/max
    # data['xq1'] = vt[twcol].quantile(.01)
    # data['xq99'] = vt[twcol].quantile(.99)
    # data['yq1'] = vt[var].quantile(.01)
    # data['yq99'] = vt[var].quantile(.99)

    # pts per bin
    data['n_bursts'] = len(vt)
    data['pts_per_bin'] = pts_per_bin

    # add subset strings
    data = {
        '{}_{}_{}_{}_{}_'.format(colstr, sstr, hodstr, rbstr, varstr) + key:
        value for key, value in data.items()
    }

    return vt, data


def _grad_vs_r_ax_level(v, twcol, var, data, q,
                        wo_r_before=None, fig=None, ax=None):

    n_bins = 25

    if store_nat_comm_data:
        data_ = v.copy()
        data_ = data_[['s_-6_-2', 'r_max']]
        data_.rename(columns={
            's_-6_-2': 'temporal pre-rainfall temperature gradient [°C/h]',
            'r_max': 'maximum intensity of episode [mm/h]'},
            inplace=True
        )

        if args.selection == 'nta_neg_box_JASO':
            fstr = 'fig2e'
        elif args.selection == 'nta_pos_box_JASO':
            fstr = 'fig2f'
        elif args.selection == 'N_tropical_W_tcs_JASO':
            if args.only_tropical_cyclones:
                fstr = 'fig4b'
            else:
                fstr = 'fig4a'
        else:
            fstr = 'other'
        os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
        data_.to_csv(storefolder + f'Nat_Comm_Fig_data/{fstr}.csv', index=False)

    for name, value in data.items():
        if name.endswith('m_left_binned'):
            m_left_q = value
        if name.endswith('t_left_binned'):
            t_left_q = value
        if name.endswith('rvalue_left_binned'):
            rvalue_left_q = value
        if name.endswith('pvalue_left_binned'):
            pvalue_left_q = value
        # if name.endswith('xgmax_binned'):
        #     xgmax_q = value
        # if name.endswith('xgmin_binned'):
        #     xgmin_q = value
        # if name.endswith('gmin_binned'):
        #     gmin_q = value
        # if name.endswith('gmax_binned'):
        #     gmax_q = value
        if name.endswith('pts_per_bin'):
            pts_per_bin = value

    # plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

    # bounding quartile lines
#     left = v[twcol].quantile(.01)
#     right = v[twcol].quantile(.99)
#     bottom = v[var].quantile(.01)
#     top = v[var].quantile(.99)
#     ax.vlines([left, right], bottom, top)
#     ax.hlines([bottom, top], left, right)

    # lowess
    # ax.plot(xl, yl, colors[1], label='lowess', alpha=.8, marker='o')
    # ax.plot(
        # xl_q, yl_q, colors[1], label='lowess (q90)', alpha=.8, marker='o')

    # lowess xgmin/max, xgmin/max_q90
    # ax.scatter(
    #     xl[yl.argmin()], yl.min(), s=120, facecolors='none', edgecolors='k')
    # ax.scatter(
    #     xl[yl.argmax()], yl.max(), s=120, facecolors='none', edgecolors='k')
    # ax.scatter(
    #     xgmin_q, gmin_q, s=120, facecolors='none',
    #     edgecolors='k')
    # ax.scatter(
    #     xgmax_q, gmax_q, s=120, facecolors='none',
    #     edgecolors='k')

    # linregress line
#     ax.plot(
#         [v[twcol].min(), v[twcol].max()],
#         [m*v[twcol].min() + t, m*v[twcol].max() + t],
#         colors[0],
#         alpha=.8,
#         label='linregress'
#     )

    # scatter
    # ax.scatter(v[twcol], v[var], alpha=.5, edgecolors='none', label='All')

    # add daily average temp at tw_first
    td = int(twcol.split('_')[2])
    # rvsat = v['sat'].rolling(24, axis=1).mean()
    # rvsdt = v['sdt'].rolling(24, axis=1).mean()
    # rvrh = v['RH'].rolling(24, axis=1).mean()
    # v['rsat_td_{}'.format(td)] = rvsat.loc[:, td]
    # v['rsdt_td_{}'.format(td)] = rvsdt.loc[:, td]
    # v['rrh_td_{}'.format(td)] = rvrh.loc[:, td]

    # binning
    binedges = np.quantile(v[twcol], np.linspace(0, 1, n_bins + 1))
    v['{}_d'.format(twcol)] = pd.cut(
        v[twcol], binedges, include_lowest=True, duplicates='drop')

    # groupby x
    gv = v.groupby('{}_d'.format(twcol))

    # r/mst
    if var == 'max_sustained_wind' or var == 'category':
        v_binned = gv[var].mean().to_frame()
    else:
        v_binned = gv[var].quantile(q/100).to_frame()

    # xcol
    v_binned[twcol] = gv[twcol].mean()

    # sat
    # satcol = 'rsat_td_{}'.format(td)
    # v_binned[satcol] = gv[satcol].mean()

    # sdt
    # sdtcol = 'rsdt_td_{}'.format(td)
    # v_binned[sdtcol] = gv[sdtcol].mean()

    # rh
    # rhcol = 'rrh_td_{}'.format(td)
    # v_binned[rhcol] = gv[rhcol].mean()

    # lat
    # v_binned['lat'] = gv['lat'].mean()

    # pts_per_bin
    v_binned['size'] = gv.size()

    # extra axes
    # grad/r
    ax.set_xlabel('$T^r_g$ [$^\circ$Ch$^{-1}$]')
    if var == 'max_sustained_wind':
        label = 'mean maximum sustained wind [kn]'
    if var == 'category':
        label = 'mean tc category'
    else:
        label = "$P^{" + str(q) + "}$ [mmh$^{-1}$]"
    ax.set_ylabel(label)

    # sat
    # ax_sat = ax.twinx()
    # ax_sat.set_ylabel('SAT [°C]')
    # ax_sat.yaxis.label.set_color(colors['sat'])
    # ax_sat.tick_params(axis='y', colors=colors['sat'])

    # rh
    # ax_rh = ax.twinx()
    # ax_rh.spines["right"].set_position(("axes", 1.04))
    # _make_patch_spines_invisible(ax_rh)
    # ax_rh.spines["right"].set_visible(True)
    # ax_rh.set_ylabel('RH [%]')
    # ax_rh.yaxis.label.set_color(colors['rrh'])
    # ax_rh.tick_params(axis='y', colors=colors['rrh'])

    # lat
    # ax_lat = ax.twinx()
    # ax_lat.spines["right"].set_position(("axes", 1.08))
    # _make_patch_spines_invisible(ax_lat)
    # ax_lat.spines["right"].set_visible(True)
    # ax_lat.set_ylabel('lat')
    # ax_lat.yaxis.label.set_color(colors['lat'])
    # ax_lat.tick_params(axis='y', colors=colors['lat'])

    # bootstrapping q90 and quantile regression
    # ------------------------------------------------------------------------

    def bootstrap(array, q, n_elements=None, n_bootstraps=1000):
        "estimate percentile"
        if not n_elements:
            n_elements = len(array)
        bootstrap_estimates = np.empty(n_bootstraps, dtype=np.float64)
        for c in range(n_bootstraps):
            sample = np.random.choice(array, size=n_elements, replace=True)
            estimate = np.percentile(sample, q)
            bootstrap_estimates[c] = estimate
        return bootstrap_estimates

    def bootstrap_qr(df, x, y, q, n_elements=None, n_bootstraps=1000):
        "quantile regression bootstrap"
        if not n_elements:
            n_elements = len(df)
        bootstrap_estimates = np.empty((n_bootstraps, 2), dtype=np.float64)
        for c in range(n_bootstraps):
            print(f'{c+1}/{n_bootstraps} | {(c+1)/n_bootstraps*100}%')
            dft = df.sample(n=n_elements, replace=True)
            mod = smf.quantreg(f'{y} ~ {x}', dft)
            res = mod.fit(q=q)
            m, t = res.params[x], res.params['Intercept']
            bootstrap_estimates[c, 0] = m
            bootstrap_estimates[c, 1] = t
        return bootstrap_estimates

    # for confidence interval
    x_lin = np.linspace(v_binned[twcol].min(), 0, 50)  # for plotting

    # bootstrap q90
    # file name
    fname = storefolder + 'bootstrapping_grad_vs_r/'
    os.makedirs(fname, exist_ok=True)
    fname += args.selection
    fname += '_'
    fname += '_'.join(next(iter(data.keys())).split('_')[:8])
    fname += f'_q{q}'
    try:
        # load?
        bes = np.load(fname + '.npy')
    except FileNotFoundError:
        print('could not find ', fname)
        bes = np.empty((n_bootstraps_q90, len(gv)), dtype=float)
        for c, (index, vt) in enumerate(gv):
            bes[:, c] = bootstrap(vt[var].values, q, n_bootstraps=n_bootstraps_q90)
        # store
        np.save(fname, bes)
    for c, (index, vt) in enumerate(gv):
        v_binned.at[index, 'bt_mean'] = bes[:, c].mean()
        v_binned.at[index, 'bt_lb'] = np.percentile(bes[:, c], 2.5)
        v_binned.at[index, 'bt_ub'] = np.percentile(bes[:, c], 97.5)

    # qr
    # file name
    fname += '_qr'
    vt = v.loc[v[twcol] < 0]
    vt['grad'] = vt[twcol]
    try:
        slope = np.loadtxt(fname + '_slope.txt')
    except OSError:
        print('could not find ', fname)
        mod = smf.quantreg(f'{var} ~ grad', vt)
        res = mod.fit(q=q/100)
        # store
        with open(fname + '_summary.txt', 'w') as file:
            file.write(res.summary().as_latex())
        with open(fname + '_slope.txt', 'w') as file:
            file.write(str(res.params['grad']))
        slope = res.params['grad']

    # res.summary()
    # y_pred = res.params['grad'] * x_lin + res.params['Intercept']

    # bootstrap qr
    # file name
    fname += '_bt'
    try:
        bes = np.load(fname + '.npy')
    except FileNotFoundError:
        print('could not find ', fname)
        bes = bootstrap_qr(vt, 'grad', f'{var}', q=q/100, n_elements=None,
                           n_bootstraps=n_bootstraps_qr)
        # store
        np.save(fname, bes)

    # bootstrap qr line
    y_pred_bt = bes[:, 0].mean()*x_lin + bes[:, 1].mean()

    # confidence interval
    y_pred_ci = np.dot(np.atleast_2d(bes[:, 0]).T,
                       np.atleast_2d(x_lin)) + np.atleast_2d(bes[:, 1]).T
    y_pred_ub = np.percentile(y_pred_ci, 97.5, axis=0)
    y_pred_lb = np.percentile(y_pred_ci, 2.5, axis=0)

    # ----------------------------------------------------------------------------
    # plot
    # bootstrapping qr lines
    # for c in range(len(bes)):
    #     ax.plot(x_lin, bes[c, 0]*x_lin + bes[c, 1],
    #             color='tab:blue',
    #             lw=.1,
    #             alpha=.2)

    # bootstrapping q90
    l1 = ax.plot(v_binned[twcol], v_binned['bt_mean'],
                 color=colors['r'],
                 marker='o',
                 ms=2,
                 lw=.3,
                 label=label)

    # error bars bootstrapping q90
    for _, row in v_binned.iterrows():
        ax.plot([row[twcol], row[twcol]], [row['bt_lb'], row['bt_ub']],
                color='k', lw=.5)

    # bootstrapping confidence interval (ci)
    ax.fill_between(x_lin, y_pred_lb, y_pred_ub,
                    facecolor='tab:blue',
                    alpha=.7,
                    edgecolor='none')

    # q90
    # ax.scatter(vt[twcol], vt[var],
    #            s=50,
    #            c='grey',
    #            marker='+',
    #            linewidths=.2,
    #            label='q90')

    # qr
    # ax.plot(x_lin, y_pred,
    #         linestyle=(0, (5, 1)),
    #         color='lightsteelblue',
    #         label='qr | intensity ~ gradient')

    # bootstrapping qr
    if var == 'max_sustained_wind':
        varlabel = 'MST'
    elif var == 'category':
        varlabel = 'CAT'
    else:
        varlabel = "$P^{" + str(q) + "}$"
    l5 = ax.plot(x_lin, y_pred_bt,
                 linestyle=(0, (10, 1)),
                 color='darkmagenta',
                 lw=.5,
                 label=f'linear regression between {varlabel} and $T^r_g$')

    # ------------------------------------------------------------------------

    # q90
    # l1 = ax.plot(
    #     v_binned[twcol], v_binned[var], color=colors['r'],
    #     label=label,
    #     marker='o',
    #     ms=2,
    #     lw=.8,
    # )

    # plot sat
    # l2 = ax_sat.plot(
    #     v_binned[twcol], v_binned[satcol],
    #     marker='o', ms=3, color=colors['sat'],
    #     label=satcol)

    # plot rh
    # l3 = ax_rh.plot(
    #     v_binned[twcol], v_binned[rhcol],
    #     marker='o', ms=3, color=colors['rrh'],
    #     label=rhcol)

    # plot lat
    # l4 = ax_lat.plot(
    #     v_binned[twcol], v_binned['lat'], marker='o', ms=3, color=colors['lat'],
    #     label='lat_mean')

    # v_binned['lat_std'] = gv['lat'].std() / 10.
    # ax_lat.fill_between(
    #     v_binned[twcol],
    #     v_binned['lat'] - v_binned['lat_std'], v_binned['lat'] + v_binned['lat_std'],
    #     color='b', alpha=.2, lw=0)

    # linregress line (left q90)
    # if var == 'max_sustained_wind':
    #     varlabel = 'MST'
    # elif var == 'category':
    #     varlabel = 'CAT'
    # else:
    #     varlabel = "$P^{" + str(q) + "}$"
    # l5 = ax.plot(
    #     [v_binned[twcol].min(), 0],
    #     [m_left_q*v_binned[twcol].min() + t_left_q, t_left_q],
    #     linestyle=(0, (5, 1)),
    #     color='darkmagenta',  # 'lightsteelblue',
    #     label='linear regression between {} and $T^r_g$'.format(varlabel)
    # )

    # add data text
    # print('-'*80)
    # for key, value in data.items():
    #     print(key, value)

    # add text (all)
    # ax.text(
    #     v[twcol].min(), m*v[twcol].min() + t,
    #     "m={:.2f}\n"
    #     "r={:.2f}\n"
    #     "p={:.4f}\n".format(m, rvalue, pvalue),
    #     color='k')
    # add text (q90)
    # ax.text(
    #     vt_q[twcol].min(), m_q*vt_q[twcol].min() + t_q,
    #     "m={:.2f}\n"
    #     "r={:.2f}\n"
    #     "p={:.4f}\n".format(m_q, rvalue_q, pvalue_q),
    #     color='k')

    # add text box
    if var == 'max_sustained_wind':
        unit = 'kn'
    elif var == 'category':
        unit = '1'
    else:
        unit = 'mmh$^{{-1}}$'
    textstr = '\n'.join((
        'No. of episodes per bin: {}'.format(pts_per_bin),
        'slope of regression: {:.1f} [{}/$^{{\circ}}$Ch$^{{-1}}$]'.format(
            # np.round(m_left_q, 1), unit),
            np.round(slope, 1), unit),
        'PCC = {:.3f}'.format(np.round(rvalue_left_q, 3)),
        'p-value = {:.1e}'.format(pvalue_left_q),
    ))
    props = dict(
        boxstyle='round',
        facecolor='w',
        alpha=1,
        edgecolor='lightgrey',
    )

    # place a text box in upper left in axes coords
    # orig: .47, .77
    # twcol, var, data, q, wo_r_before
    if ((args.selection == 'N_tropical_W_tcs_JASO') and
        (twcol == 's_-6_-2') and
        (var == 'r_max') and
        (q == 90) and
            (wo_r_before is False)):
        ax.text(.02, 0.28, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

    elif ((args.selection == 'N_tropical_W_tcs_JASO') and
          (twcol == 's_-6_-2') and
          (var == 'r_max') and
          (q == 90) and
          (args.only_tropical_cyclones is True) and
          (wo_r_before is True)):
        ax.text(.02, 0.28, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

    elif ((args.selection == 'S_tropical_W_DJFMA') and
          (twcol == 's_-6_-2') and
          (var == 'r_max') and
          (q == 90) and
          (args.only_tropical_cyclones is True) and
          (wo_r_before is True)):
        ax.text(.02, 0.28, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

    else:
        ax.text(.36, .73, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

    # add text (left q90)
    # ax.text(
    #     # vt_q[twcol].min(), m_left_q*vt_q[twcol].min() + t_left_q,
    #     0, t_left_q,
    #     "m={:.2f}\n"
    #     "r={:.2f}\n"
    #     "p={:.4f}\n".format(m_left_q, rvalue_left_q, pvalue_left_q),
    #     color='k')

    # X/Y-labels
    ax.grid()

    # legend
    lns = l1 + l5  # l1+l2+l3+l4+l5
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc='upper right')

    # return fig, ax, ax_sdt
    return fig, ax, lns


def tc_burst_stats(v, season, twcol):  # @UnusedVariable

    tstart = datetime.now()

    picfolder = os.getcwd() + f'/glcp_burst{tcfstr}' + \
        '/{}/'.format(args.selection)

    os.makedirs(picfolder + 'tc_burst_stats/', exist_ok=True)

    # dist_to_eye / nr of locs string
    dnstr = (f"nl_{args.nr_locs}_min_dist_{min_dist_to_eye}_"
             f"max_dist_{max_dist_to_eye}_")
    dntstr = (f" | {min_dist_to_eye} < dist_to_eye < {max_dist_to_eye}"
              f" | {args.nr_locs} locations")

    # twcol, get rid of nans
    # v = v[~v[twcol].isnull()]
    colstr = twcol

    for wo_r_before in [True, False]:

        vt = v.copy()

        # rain before?
        if wo_r_before:
            for tdr in range(-6, -51, -3):
                vt = vt.loc[vt['r', tdr].isnull()]
            rbstr = 'worb'
        elif not wo_r_before:
            rbstr = 'wrb'

        # # HISTOGRAMS
        # # ----------
        #
        # # MST
        # fig, ax = plt.subplots(figsize=(width, height))
        # plot_hist(vt['max_sustained_wind'], bins=35, log_bins=False,
        #           density=False,
        #           ls='-', lw=.5, marker='o', ms=5, ax=ax)
        # ax.set_yscale('log')
        # ax.grid()
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(picfolder + f'tc_burst_stats/{dnstr}hist_mst_{rbstr}.png')
        # plt.close(fig)
        #
        # # MST bar plot
        # vt['msti'] = vt['max_sustained_wind'].astype(int)
        # vc = vt['msti'].value_counts()
        # fig, ax = plt.subplots(figsize=(width, height))
        # vc.sort_index().plot(kind='bar', ax=ax)
        # ax.set_yscale('log')
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(picfolder + "tc_burst_stats/"
        #             f"{dnstr}hist_msti_counts_{rbstr}.png")
        # plt.close(fig)
        #
        # # DIST TO EYE
        # fig, ax = plt.subplots(figsize=(width, height))
        # plot_hist(vt['dist_to_eye'], bins=50, log_bins=False, density=False,
        #           ls='-', lw=.5, marker='o', ms=5, ax=ax)
        # ax.grid()
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(picfolder + "tc_burst_stats/"
        #             f"{dnstr}hist_dist_to_eye_{rbstr}.png")
        # plt.close(fig)
        #
        # # R
        # fig, ax = plt.subplots(figsize=(width, height))
        # plot_hist(vt['r_min_dist'], bins=40, log_bins=False, density=False,
        #           ls='-', lw=.5, marker='o', ms=5, ax=ax)
        # ax.set_yscale('log')
        # ax.grid()
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(picfolder + f'tc_burst_stats/{dnstr}hist_r_{rbstr}.png')
        # plt.close(fig)
        #
        # # GRAD
        # fig, ax = plt.subplots(figsize=(width, height))
        # plot_hist(vt[twcol], bins=40, log_bins=False, density=False,
        #           ls='-', lw=.5, marker='o', ms=5, ax=ax)
        # ax.grid()
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(picfolder + "tc_burst_stats/"
        #             f"{dnstr}hist_grad_{colstr}_{rbstr}.png")
        # plt.close(fig)
        #
        # # 2D DENSITIES
        # # ------------
        #
        # # MST vs GRAD
        # fig, ax = plt.subplots(figsize=(width, height))
        # im = ax.hexbin(vt['max_sustained_wind'], vt[twcol], gridsize=(20, 20),
        #                mincnt=1)
        # fig.colorbar(im)
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(
        #     picfolder + f"tc_burst_stats/{dnstr}"
        #     f"2d_hist_mst_vs_grad_{twcol}_{rbstr}.png")
        # plt.close(fig)
        #
        # # MST vs GRAD vs R
        # fig, ax = plt.subplots(figsize=(width, height))
        # im = ax.hexbin(vt['max_sustained_wind'], vt[twcol], C=vt['r_min_dist'],
        #                reduce_C_function=_custom_q90, vmax=25,
        #                gridsize=(12, 12),
        #                mincnt=1)
        # im.set_clim(vmax=np.percentile(im.get_array(), 98))
        #
        # fig.colorbar(im)
        # ax.set_title(f'pts: {len(vt)}' + dntstr)
        # fig.savefig(picfolder + f"tc_burst_stats/{dnstr}"
        #             f"2d_hist_mst_vs_grad_{twcol}_vs_r_{rbstr}.png")
        # plt.close(fig)
        #
        # # mst vs r
        # # fig, ax = plt.subplots(figsize=(width, height))
        # # im = ax.hexbin(vt['max_sustained_wind'], vt['r'], bins=30,
        # #                yscale='log', mincnt=1, )
        # # fig.colorbar(im)
        # # ax.grid()
        # # fig.savefig(picfolder + f'hexbin_mst_r_{colstr}_{rbstr}.png')
        #
        # # 2D CONDITIONAL STATS
        # # -----------------
        #
        def bin_it(v, col, n_bins):
            binedges = np.quantile(v[col], np.linspace(0, 1, n_bins + 1))
            v[f'{col}_d'] = pd.cut(v[col], binedges, include_lowest=True,
                                   duplicates='drop')
            gv = v.groupby(f'{col}_d')
            return gv
        #
        # # MST vs GRAD
        # col = 'max_sustained_wind'
        # gv = bin_it(vt, col, n_bins=25)
        # v_binned = gv[twcol].mean().to_frame()
        # v_binned[col] = gv[col].mean()
        # v_binned['size'] = gv.size()
        # pts_per_bin = int(v_binned['size'].mean())
        #
        # fig, ax = plt.subplots(figsize=(width, height))
        # ax.plot(v_binned[col], v_binned[twcol], marker='o')
        # ax.grid()
        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)
        # fig.savefig(picfolder + "tc_burst_stats/"
        #             f"{dnstr}mst_vs_grad_{colstr}_{rbstr}.png")
        # plt.close(fig)
        #
        # # GRAD vs CAT
        # col = twcol
        # gv = bin_it(vt, col, n_bins=25)
        # v_binned = gv['category'].mean().to_frame()
        # v_binned[col] = gv[col].mean()
        # v_binned['size'] = gv.size()
        # pts_per_bin = int(v_binned['size'].mean())
        #
        # fig, ax = plt.subplots(figsize=(width, height))
        # ax.plot(v_binned[col], v_binned['category'], marker='o')
        # ax.grid()
        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)
        # fig.savefig(picfolder + f"tc_burst_stats/"
        #             f"{dnstr}grad_{twcol}_vs_category_{rbstr}.png")
        # plt.close(fig)
        #
        # # GRAD vs MST
        # col = twcol
        # gv = bin_it(vt, col, n_bins=25)
        # v_binned = gv['max_sustained_wind'].mean().to_frame()
        # v_binned[col] = gv[col].mean()
        # v_binned['size'] = gv.size()
        # pts_per_bin = int(v_binned['size'].mean())
        #
        # fig, ax = plt.subplots(figsize=(width, height))
        # ax.plot(v_binned[col], v_binned['max_sustained_wind'], marker='o')
        # ax.grid()
        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)
        # fig.savefig(picfolder + f"tc_burst_stats/"
        #             f"{dnstr}grad_{twcol}_vs_mst_{rbstr}.png")
        # plt.close(fig)

        # MST vs R
        col = 'max_sustained_wind'
        gv = bin_it(vt, col, n_bins=20)
        # v_binned = gv['r_min_dist'].quantile(.90).to_frame()
        v_binned = gv['r_max'].quantile(.9).to_frame()
        v_binned[col] = gv[col].mean()
        v_binned['size'] = gv.size()
        pts_per_bin = int(v_binned['size'].mean())

        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot(v_binned[col],
                # v_binned['r_min_dist'],
                v_binned['r_max'],
                color='k', marker='o',
                label='No. of episodes per bin: {}'.format(pts_per_bin)
                # ms=4,
                # lw=1
                )
        ax.set_ylabel("$P^{" + str(90) + "}$ [mmh$^{-1}$]")
        ax.set_xlabel("maximum sustained wind [kn]")
        ax.grid()

        leg = ax.legend(
            handlelength=0,
            handletextpad=0,
            markerscale=0,
            framealpha=1,
        )
        for item in leg.legendHandles:
            item.set_visible(False)

        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)

        # store
        # fig.tight_layout()
        picfile = picfolder + f'tc_burst_stats/{dnstr}mst_vs_r_{rbstr}'

        # pdf
        fig.savefig(
            picfile + '.pdf',
            format='pdf',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        # png
        fig.savefig(
            picfile + '.png',
            format='png',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        # svg
        fig.savefig(
            picfile + '.svg',
            format='svg',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        print(picfile + '.xxx')
        plt.close(fig)

        # # R vs MST
        # col = 'r_min_dist'
        # gv = bin_it(vt, col, n_bins=20)
        # v_binned = gv['max_sustained_wind'].mean().to_frame()
        # v_binned[col] = gv[col].mean()
        # v_binned['size'] = gv.size()
        # pts_per_bin = int(v_binned['size'].mean())
        #
        # fig, ax = plt.subplots(figsize=(width, height))
        # ax.plot(v_binned[col], v_binned['max_sustained_wind'], marker='o')
        # ax.grid()
        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)
        # fig.savefig(picfolder + f'tc_burst_stats/{dnstr}r_vs_mst_{rbstr}.png')
        # plt.close(fig)
        #
        # # GRAD vs R
        # col = twcol
        # gv = bin_it(vt, col, n_bins=20)
        # v_binned = gv['r_min_dist'].quantile(.9).to_frame()
        # v_binned[col] = gv[col].mean()
        # v_binned['size'] = gv.size()
        # pts_per_bin = int(v_binned['size'].mean())
        #
        # fig, ax = plt.subplots(figsize=(width, height))
        # ax.plot(v_binned[col], v_binned['r_min_dist'], marker='o')
        # ax.grid()
        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)
        # fig.savefig(
        #     picfolder + f'tc_burst_stats/{dnstr}grad_{twcol}_vs_r_{rbstr}.png')
        # plt.close(fig)
        #
        # # DIST_TO_EYE vs R
        # col = 'dist_to_eye'
        # gv = bin_it(vt, col, n_bins=20)
        # v_binned = gv['r_min_dist'].quantile(.90).to_frame()
        # v_binned[col] = gv[col].mean()
        # v_binned['size'] = gv.size()
        # pts_per_bin = int(v_binned['size'].mean())
        #
        # fig, ax = plt.subplots(figsize=(width, height))
        # ax.plot(v_binned[col], v_binned['r_min_dist'], marker='o')
        # ax.set_yscale('log')
        # ax.grid()
        # ax.set_title(f'pts per bin: {pts_per_bin}' + dntstr)
        # fig.savefig(picfolder + "tc_burst_stats/"
        #             f"{dnstr}dist_to_eye_vs_r_{rbstr}.png")
        # plt.close(fig)

    dt = datetime.now() - tstart
    ptime = 'comp. time:\ts={}\tms={}'.format(
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)


def grad_vs_category(v, season, twcol, var):

    tstart = datetime.now()

    os.makedirs(picfolder + 'grad_vs_category/', exist_ok=True)

    for wo_r_before in [True, False]:

        fig, ax = plt.subplots(figsize=(width, height))

        if wo_r_before:
            rbstr = 'worb'
        else:
            rbstr = 'wrb'

        vt, data = compute_grad_vs_r(
            v, twcol=twcol,
            season=season,
            hod='all',
            wo_r_before=wo_r_before,
            var=var,
            q=90,
        )

        # _, _, ax_sxt = _grad_vs_r_ax_level(vt, twcol, fig=fig, ax=ax)
        _, _, lns = _grad_vs_r_ax_level(
            vt, twcol, var, data, q=90, fig=fig, ax=ax
        )

#             # density
#             ax_d = ax.twinx()
#             _make_patch_spines_invisible(ax_d)
#             xmin, xmax = ax.get_xlim()
#
#             plot_hist(
#                 vt[twcol], bins=100, log_bins=False, density=True,
#                 floor=False, ax=ax_d,
#                 linestyle='-',
#                 # lw=1,
#                 color='darkolivegreen',
#                 # marker='o',
#                 # ms=2,
#                 alpha=.0)
#
#             # get data to fill lines
#             x, y = ax_d.lines[0].get_data()
#             fbl = ax_d.fill_between(
#                 x, y, alpha=.2, zorder=0, color='darkolivegreen',
#                 label='probability density of $T^r_g$',
#             )
#             ax_d.get_yaxis().set_ticks([])
#             ax_d.set_xlim(xmin, xmax)

        # ax.set_title(
        #     'tf={}, tl={} | n_bursts: {} ({} per bin) | {}'.format(
        #         twcol.split('_')[1], twcol.split('_')[2], len(vt),
        #         len(vt)//15, rbstr))

        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, alpha=.6)

        # legend
        # lns.append(fbl)
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', facecolor='white',
                  framealpha=1)

        # store
        # fig.tight_layout()
        picfile = picfolder + \
            'grad_vs_category/{}_{}_grad_{}_vs_{}_{}'.format(
                sname, season, twcol, var, rbstr)

        # pdf
        fig.savefig(
            picfile + '.pdf',
            format='pdf',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        # png
        fig.savefig(
            picfile + '.png',
            format='png',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        # svg
        fig.savefig(
            picfile + '.svg',
            format='svg',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        print(picfile + '.xxx')
        plt.close()

    dt = datetime.now() - tstart
    ptime = 'comp. time:\ts={}\tms={}'.format(
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)


def grad_vs_mst(v, season, twcol, var):

    tstart = datetime.now()

    os.makedirs(picfolder + 'grad_vs_mst/', exist_ok=True)

    for wo_r_before in [True, False]:

        fig, ax = plt.subplots(figsize=(width, height))

        if wo_r_before:
            rbstr = 'worb'
        else:
            rbstr = 'wrb'

        vt, data = compute_grad_vs_r(
            v, twcol=twcol,
            season=season,
            hod='all',
            wo_r_before=wo_r_before,
            var=var,
            q=90,
        )

        # _, _, ax_sxt = _grad_vs_r_ax_level(vt, twcol, fig=fig, ax=ax)
        _, _, lns = _grad_vs_r_ax_level(
            vt, twcol, var, data, q=90, fig=fig, ax=ax
        )

#             # density
#             ax_d = ax.twinx()
#             _make_patch_spines_invisible(ax_d)
#             xmin, xmax = ax.get_xlim()
#
#             plot_hist(
#                 vt[twcol], bins=100, log_bins=False, density=True,
#                 floor=False, ax=ax_d,
#                 linestyle='-',
#                 # lw=1,
#                 color='darkolivegreen',
#                 # marker='o',
#                 # ms=2,
#                 alpha=.0)
#
#             # get data to fill lines
#             x, y = ax_d.lines[0].get_data()
#             fbl = ax_d.fill_between(
#                 x, y, alpha=.2, zorder=0, color='darkolivegreen',
#                 label='probability density of $T^r_g$',
#             )
#             ax_d.get_yaxis().set_ticks([])
#             ax_d.set_xlim(xmin, xmax)

        # ax.set_title(
        #     'tf={}, tl={} | n_bursts: {} ({} per bin) | {}'.format(
        #         twcol.split('_')[1], twcol.split('_')[2], len(vt),
        #         len(vt)//15, rbstr))

        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, alpha=.6)

        # legend
        # lns.append(fbl)
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', facecolor='white',
                  framealpha=1)

        # store
        # fig.tight_layout()
        picfile = picfolder + \
            'grad_vs_mst/{}_{}_grad_{}_vs_mst_{}_{}'.format(
                sname, season, twcol, var, rbstr)

        # pdf
        fig.savefig(
            picfile + '.pdf',
            format='pdf',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        # png
        fig.savefig(
            picfile + '.png',
            format='png',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        # svg
        fig.savefig(
            picfile + '.svg',
            format='svg',
            bbox_inches='tight', pad_inches=0,
            dpi=600,
        )
        print(picfile + '.xxx')
        plt.close()

    dt = datetime.now() - tstart
    ptime = 'comp. time:\ts={}\tms={}'.format(
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)


def grad_vs_r(v, season, twcol, var):

    hod = False
    wo_r_befores = [True]  # [True, False]
    qs = [90]  # [90, 95, 99]
    tstart = datetime.now()

    os.makedirs(picfolder + 'grad_vs_r/', exist_ok=True)

    if not hod:

        for wo_r_before in wo_r_befores:
            for q in qs:

                fig, ax = plt.subplots(figsize=(width, height))

                if wo_r_before:
                    rbstr = 'worb'
                else:
                    rbstr = 'wrb'

                vt, data = compute_grad_vs_r(
                    v, twcol=twcol,
                    season=season,
                    hod='all',
                    wo_r_before=wo_r_before,
                    var=var,
                    q=q,
                )

                # _, _, ax_sxt = _grad_vs_r_ax_level(vt, twcol, fig=fig, ax=ax)
                _, _, lns = _grad_vs_r_ax_level(
                    vt, twcol, var, data, q, wo_r_before, fig=fig, ax=ax
                )

    #             # density
    #             ax_d = ax.twinx()
    #             _make_patch_spines_invisible(ax_d)
    #             xmin, xmax = ax.get_xlim()
    #
    #             plot_hist(
    #                 vt[twcol], bins=100, log_bins=False, density=True,
    #                 floor=False, ax=ax_d,
    #                 linestyle='-',
    #                 # lw=1,
    #                 color='darkolivegreen',
    #                 # marker='o',
    #                 # ms=2,
    #                 alpha=.0)
    #
    #             # get data to fill lines
    #             x, y = ax_d.lines[0].get_data()
    #             fbl = ax_d.fill_between(
    #                 x, y, alpha=.2, zorder=0, color='darkolivegreen',
    #                 label='probability density of $T^r_g$',
    #             )
    #             ax_d.get_yaxis().set_ticks([])
    #             ax_d.set_xlim(xmin, xmax)
                # ax.set_title(
                #     'tf={}, tl={} | n_bursts: {} ({} per bin) | {}'.format(
                #         twcol.split('_')[1], twcol.split('_')[2], len(vt),
                #         len(vt)//15, rbstr))

                # legend
                # lns.append(fbl)
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc='upper right', facecolor='white',
                          framealpha=1)

                def store_figure(add_str=''):
                    # store
                    # fig.tight_layout()
                    picfile = picfolder + \
                        'grad_vs_r/{}_{}_grad_{}_vs_r_{}_q{}_{}'.format(
                            sname, season, twcol, var, q, rbstr)
                    picfile += add_str

                    # pdf
                    fig.savefig(
                        picfile + '.pdf',
                        format='pdf',
                        bbox_inches='tight', pad_inches=0,
                        dpi=600,
                    )
                    # png
                    fig.savefig(
                        picfile + '.png',
                        format='png',
                        bbox_inches='tight', pad_inches=0,
                        dpi=600,
                    )
                    # svg
                    fig.savefig(
                        picfile + '.svg',
                        format='svg',
                        bbox_inches='tight', pad_inches=0,
                        dpi=600,
                    )
                    print(picfile + '.xxx')
                    plt.close()

                # paper: same axis for nta_neg_box and nta_pos_box
                if (args.selection == 'nta_neg_box_JASO' or
                        args.selection == 'nta_pos_box_JASO'):
                    ymin = 1.2
                    ymax = 6.6
                    dy = .05 * (ymax - ymin)
                    ax.set_xlim(-0.082, 0.051)
                    ax.set_ylim(ymin, ymax)
                    ax.vlines(0, ymin+dy, ymax-dy, alpha=.6)
                    store_figure()

                # paper: same axis for ..
                elif args.selection == 'S_tropical_W_DJFMA':
                    # tc / no tc (fig.4)
                    if ((twcol == 's_-6_-2') and
                        (var == 'r_max') and
                        (wo_r_before is True) and
                            (q == 90)):
                        ymin = 1.48
                        ymax = 6.05
                        dy = .05 * (ymax - ymin)
                        ax.set_xlim(-0.082, 0.055)
                        ax.set_ylim(ymin, ymax)
                        ax.vlines(0, ymin+dy, ymax-dy, alpha=.6)
                        store_figure('_tc')

                elif args.selection == 'N_tropical_W_tcs_JASO':

                    # tc / no tc (fig.4)
                    if ((twcol == 's_-6_-2') and
                        (var == 'r_max') and
                        (wo_r_before is True) and
                            (q == 90)):
                        ymin = 2.25
                        ymax = 7.25
                        dy = .05 * (ymax - ymin)
                        ax.set_xlim(-0.086, 0.062)
                        ax.set_ylim(ymin, ymax)
                        ax.vlines(0, ymin+dy, ymax-dy, alpha=.6)
                        store_figure('_tc')

                    # wrb/worb (fig.4s)
                    if ((twcol == 's_-6_-2') and
                        (var == 'r_max') and
                        ((wo_r_before is False) or (wo_r_before is True)) and
                            (q == 90)):
                        ymin = 2.4
                        ymax = 6.1
                        dy = .05 * (ymax - ymin)
                        ax.set_xlim(-0.086, 0.069)
                        ax.set_ylim(ymin, ymax)
                        ax.vlines(0, ymin+dy, ymax-dy, alpha=.6)
                        store_figure('_rb')

                    # tws (fig.6s)
                    if ((q == 90) and
                        (var == 'r_max') and
                        (wo_r_before is True) and
                        ((twcol == 's_-6_-2') or
                         (twcol == 's_-12_-2') or
                         (twcol == 's_-18_-2') or
                         (twcol == 's_-24_-12'))):
                        ymin = 2.4
                        ymax = 5.9
                        dy = .05 * (ymax - ymin)
                        ax.set_xlim(-0.086, 0.058)
                        ax.set_ylim(ymin, ymax)
                        ax.vlines(0, ymin+dy, ymax-dy, alpha=.6)
                        store_figure('_tws')

                    # r_max/mean (fig.13s)
                    if ((q == 90) and
                        ((var == 'r_max') or (var == 'r_mean')) and
                        (wo_r_before is True) and
                            (twcol == 's_-6_-2')):
                        ymin = 1.69
                        ymax = 5.9
                        dy = .05 * (ymax - ymin)
                        ax.set_xlim(-0.086, 0.058)
                        ax.set_ylim(ymin, ymax)
                        ax.vlines(0, ymin+dy, ymax-dy, alpha=.6)
                        store_figure('_rmax_mean')

                    else:
                        ymin, ymax = ax.get_ylim()
                        ax.vlines(0, ymin, ymax, alpha=.6)
                        store_figure()

                else:
                    ymin, ymax = ax.get_ylim()
                    ax.vlines(0, ymin, ymax, alpha=.6)
                    store_figure()

    elif hod:

        if wo_r_before:
            rbstr = 'worb'
        else:
            rbstr = 'wrb'

        fig, axs = plt.subplots(4, 2, figsize=(19.20, 10.80))
        axs = axs.flatten()
        axs_d = []

        ax_xmins = []
        ax_xmaxs = []
        ax_ymins = []
        ax_ymaxs = []
        # ax_sxts = []
        # sxt_ymins = []
        # sxt_ymaxs = []

        for hod, ax in zip([0, 12, 3, 15, 6, 18, 9, 21], axs):

            vt, data = compute_grad_vs_r(
                v, twcol=twcol,
                season=season,
                hod=hod,
                wo_r_before=wo_r_before,
                var=var
            )

            # _, _, ax_sxt = _grad_vs_r_ax_level(vt, twcol, fig=fig, ax=ax)
            _grad_vs_r_ax_level(
                vt, twcol, var, data, fig=fig, ax=ax
            )

            # min max xr
            ax_xmin, ax_xmax = ax.get_xlim()
            ax_xmins.append(ax_xmin)
            ax_xmaxs.append(ax_xmax)
            # min max yr
            ax_ymin, ax_ymax = ax.get_ylim()
            ax_ymins.append(ax_ymin)
            # ax_ymaxs.append(vt[var].quantile(.99) + .1)
            ax_ymaxs.append(ax_ymax)
            # min max ysxt
        #         ax_sxts.append(ax_sxt)
        #         sxt_ymin, sxt_ymax = ax_sxt.get_ylim()
        #         sxt_ymins.append(sxt_ymin)
        #         sxt_ymaxs.append(sxt_ymax)

            # density
            ax_d = ax.twinx()
            _make_patch_spines_invisible(ax_d)
            axs_d.append(ax_d)

            plot_hist(
                vt[twcol], bins=100, log_bins=False, density=True,
                floor=False, ax=ax_d,
                linestyle='-',
                lw=1,
                # color='b',
                # marker='o',
                # ms=2,
                alpha=.3)

            # get data to fill lines
            x, y = ax_d.lines[0].get_data()
            ax_d.fill_between(x, y, alpha=.2, zorder=0)

            ax.set_title("tf={}, tl={} | hod={:02d} | "
                         "n_bursts: {} ({} per bin) ".format(
                             twcol.split('_')[1],
                             twcol.split('_')[2],
                             hod,
                             len(vt), len(vt)//15))

        #     for ax, ax_sxt in zip(axs, ax_sxts):
        #         ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
        #         ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
        #         ax_sxt.set_ylim(min(sxt_ymins), max(sxt_ymaxs))

        for ax, ax_d in zip(axs, axs_d):
            ax.set_xlim(min(ax_xmins), max(ax_xmaxs))
            ax.set_ylim(min(ax_ymins), max(ax_ymaxs))
            ax_d.set_xlim(min(ax_xmins), max(ax_xmaxs))
            ax.vlines(0, min(ax_ymins), max(ax_ymaxs), alpha=.6)
            ax_d.get_yaxis().set_ticks([])

        # store
        fig.tight_layout()
        picfile = picfolder + \
            'grad_vs_r/{}_{}_grad_{}_vs_r_{}_{}_by_hod.pdf'.format(
                sname, season, twcol, rbstr, var)
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close()

    dt = datetime.now() - tstart
    ptime = 'comp. time:\ts={}\tms={}'.format(
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)


def _r_vs_grad_ax_level(v, twcol, var, fig=None, ax=None):

    n_bins = 25

    # plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))

    # add daily average temp at tw_first
    td = int(twcol.split('_')[2])
    rvsat = v['sat'].rolling(24, axis=1).mean()
    rvsdt = v['sdt'].rolling(24, axis=1).mean()
    rvrh = v['RH'].rolling(24, axis=1).mean()
    v['rsat_td_{}'.format(td)] = rvsat.loc[:, td]
    v['rsdt_td_{}'.format(td)] = rvsdt.loc[:, td]
    v['rrh_td_{}'.format(td)] = rvrh.loc[:, td]

    # binning
    binedges = np.quantile(v[var], np.linspace(0, 1, n_bins + 1))
    v['{}_d'.format(var)] = pd.cut(
        v[var], binedges, include_lowest=True, duplicates='drop')

    # groupby x
    gvt = v.groupby('{}_d'.format(var))

    # grad
    vt = gvt[twcol].mean().to_frame()

    # xcol
    vt[var] = gvt[var].mean()

    # sat
    satcol = 'rsat_td_{}'.format(td)
    vt[satcol] = gvt[satcol].mean()

    # sdt
    sdtcol = 'rsdt_td_{}'.format(td)
    vt[sdtcol] = gvt[sdtcol].mean()

    # rh
    rhcol = 'rrh_td_{}'.format(td)
    vt[rhcol] = gvt[rhcol].mean()

    # lat
    vt['lat'] = gvt['lat'].mean()

    # pts_per_bin
    vt['size'] = gvt.size()

    # extra axes

    # grad
    ax.set_xlabel('r_max [mm/h]')
    ax.set_ylabel('grad {} [°C/h]'.format(twcol))
    ax.yaxis.label.set_color(colors['grad'])
    ax.tick_params(axis='y', colors=colors['grad'])

    # sat
    ax_sat = ax.twinx()
    ax_sat.set_ylabel('SAT [°C]')
    ax_sat.yaxis.label.set_color(colors['sat'])
    ax_sat.tick_params(axis='y', colors=colors['sat'])

    # rh
    ax_rh = ax.twinx()
    ax_rh.spines["right"].set_position(("axes", 1.04))
    _make_patch_spines_invisible(ax_rh)
    ax_rh.spines["right"].set_visible(True)
    ax_rh.set_ylabel('RH [%]')
    ax_rh.yaxis.label.set_color(colors['rrh'])
    ax_rh.tick_params(axis='y', colors=colors['rrh'])

    # lat
    ax_lat = ax.twinx()
    ax_lat.spines["right"].set_position(("axes", 1.08))
    _make_patch_spines_invisible(ax_lat)
    ax_lat.spines["right"].set_visible(True)
    ax_lat.set_ylabel('lat')
    ax_lat.yaxis.label.set_color(colors['lat'])
    ax_lat.tick_params(axis='y', colors=colors['lat'])

    # plot grad
    l1 = ax.plot(
        vt[var], vt[twcol], color=colors['grad'],
        label=twcol+'_mean',
        marker='o',
        ms=5)

    # plot sat
    l2 = ax_sat.plot(
        vt[var], vt[satcol],
        marker='o', ms=3, color=colors['sat'],
        label=satcol)

    # plot rh
    l3 = ax_rh.plot(
        vt[var], vt[rhcol],
        marker='o', ms=3, color=colors['rrh'],
        label=rhcol)

    # plot lat
    l4 = ax_lat.plot(
        vt[var], vt['lat'], marker='o', ms=3, color=colors['lat'],
        label='lat_mean')

    # vt['lat_std'] = gvt['lat'].std() / 10.
    # ax_lat.fill_between(
    #     vt[var],
    #     vt['lat'] - vt['lat_std'], vt['lat'] + vt['lat_std'],
    #     color='b', alpha=.2, lw=0)

    # X/Y-labels
    ax.grid()

    # legend
    lns = l1+l2+l3+l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', fontsize='x-small')

    # return fig, ax, ax_sdt
    return fig, ax


def r_vs_grad(v, season, twcol, var):

    tstart = datetime.now()

    os.makedirs(picfolder + 'r_vs_grad/', exist_ok=True)

    for wo_r_before in [True, False]:

        fig, ax = plt.subplots(figsize=(19.20, 10.80))

        # set x-scale to logarithmic
        ax.set_xscale('log')

        # subset
        vt, _colstr, _sstr, _hodstr, rbstr, _varstr = \
            _grad_subset_v(
                v, twcol, season, 'all', wo_r_before, var)

        # _, _, ax_sxt = _grad_vs_r_ax_level(vt, twcol, fig=fig, ax=ax)
        _r_vs_grad_ax_level(vt, twcol, var, fig=fig, ax=ax)

        # density
        ax_d = ax.twinx()
        _make_patch_spines_invisible(ax_d)
        xmin, xmax = ax.get_xlim()

        plot_hist(
            vt[var], bins=15, log_bins=True, density=True,
            floor=False, ax=ax_d,
            linestyle='-',
            lw=1,
            # color='b',
            # marker='o',
            # ms=2,
            alpha=.3)

        # get data to fill lines
        x, y = ax_d.lines[0].get_data()
        ax_d.fill_between(x, y, alpha=.2, zorder=0)
        ax_d.get_yaxis().set_ticks([])
        ax_d.set_xlim(xmin, xmax)

        ax.set_title(
            'tf={}, tl={} | n_bursts: {} ({} per bin) | {}'.format(
                twcol.split('_')[1], twcol.split('_')[2], len(vt),
                len(vt)//15, rbstr))

        # set x-scale to normal again
        # ax.set_xscale('linear')

        # store
        fig.tight_layout()
        picfile = picfolder + \
            'r_vs_grad/{}_{}_r_{}_vs_grad_{}_{}.pdf'.format(
                sname, season, var, twcol, rbstr)
        fig.savefig(picfile)
        fig.savefig(picfile[:-3] + 'png')
        print(picfile)
        plt.close()

    dt = datetime.now() - tstart
    ptime = 'comp. time:\ts={}\tms={}'.format(
        int(dt.total_seconds()),
        str(dt.microseconds / 1000.)[:6])
    print(ptime)


def grad_vs_sat_vs_r():

    print('implement!')


def _load_events(locs):

    # parameters
    tds_sat_vs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39]

    v_table = 'v_r{}_p{}_dtime_l_r_seasons_l_sortedc7.h5'.format(r, p)

    vs = []
    c = 1
    for loc in locs:

        # load EVENTS
        v = pd.read_hdf(
            storefolder + v_table, 'v', where='l=={}'.format(loc))

        for td in tds_sat_vs:

            print('loc {:06d}/{:06d} | td {:02d}'.format(c, len(locs), td))

            # load SAT
            satcol = 'sat_td_{}'.format(td)
            v[satcol] = pd.read_hdf(
                sxtfolder + '{}_l_sortedc7.h5'.format(satcol),
                'v', where='l=={}'.format(loc), columns=[satcol])

            # load SDT
            sdtcol = 'sdt_td_{}'.format(td)
            v[sdtcol] = pd.read_hdf(
                sxtfolder + '{}_l_sortedc7.h5'.format(sdtcol),
                'v', where='l=={}'.format(loc), columns=[sdtcol])

            # compute RH
            # https://www.theweatherprediction.com/habyhints/186/
            # https://iridl.ldeo.columbia.edu/dochelp/QA/Basic/dewpoint.html
            rhcol = 'RH_td_{}'.format(td)
            T = v[satcol]
            T_d = v[sdtcol]
            T = T + 273.15
            T_d = T_d + 273.15
            E_0 = 0.611
            T_0 = 273.15
            E = E_0 * np.exp(5423 * (1/T_0 - 1/T_d))
            E_s = E_0 * np.exp(5423 * (1/T_0 - 1/T))
            rh = 100 * E/E_s
            v[rhcol] = rh

        vs.append(v)

        c += 1

    v = pd.concat(vs, axis=0)

    return v


def _make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _custom_q90(x):
    q90 = np.percentile(x, 90)
    return q90


if args.nr_locs is not None:
    # np.random.seed(0)
    try:
        locs = np.random.choice(locs, args.nr_locs, replace=False)
    except ValueError:
        pass

# picfolder += 'single_loc_grad_vs_r/{}/'.format(locs[0])
# os.makedirs(picfolder, exist_ok=True)

# # load data
# vb = _load_bursts(locs)
# vbg = _load_burst_grads(locs)
# cols = vbg.columns.values
# vbg.columns = pd.MultiIndex.from_tuples([[col, ''] for col in cols])
# vb = pd.concat((vb, vbg), axis=1)
#
# # vt, data, xl, yl, xl_q, yl_q = compute_grad_vs_r(vb)
#
# grad_vs_r(
#     vb, season='all', cols=cols, hod=False, wo_r_before=False, var='r_mean')

# all?
if args.all:

    # args.map_plot = True
    args.map_plot_by_hod_td = True

    args.single_burst = True
    # args.agg_k_bursts = True
    args.agg_all_bursts = True
    args.agg_all_bursts_by_hod = True
    args.t_vs_tmps = True
    args.t_vs_tmps_by_hod = True
    args.t_vs_sat_by_hod = True
    args.t_vs_sat_by_hod_wo_rtd = True
    args.hod_distribution = True

    args.grad_vs_r = True
    args.grad_vs_r_by_hod = True
    args.grad_vs_r_by_hod_wo_rtd = True

    args.r_vs_grad = True

    args.sat_vs_r = True
    args.sdt_vs_r = True
    args.sat_vs_r_by_td = True
    args.sat_vs_r_by_hod = True
    # args.sat_vs_r_by_td_hod = True
    args.sat_vs_r_by_hod_td = True
    args.sdt_vs_sat_vs_x_by_td = True
    args.sdt_vs_sat_vs_x_by_hod_td = True

    args.td_vs_alpha = True
    args.td_vs_alpha_by_hod = True

# map plot
if args.map_plot:
    selection_on_map()

if args.map_plot_by_hod_td:
    for season in seasons:
        selection_on_map_by_hod_td(season)

# burst related
if (args.single_burst or
        args.agg_k_bursts or
        args.agg_all_bursts or
        args.agg_all_bursts_by_hod or
        args.t_vs_tmps or
        args.t_vs_tmps_by_hod or
        args.t_vs_sat_by_hod or
        args.t_vs_sat_by_hod_wo_rtd or
        args.hod_distribution or
        args.tc_burst_stats or
        args.grad_vs_category or
        args.grad_vs_max_sustained_wind or
        args.grad_vs_r or
        args.grad_vs_r_by_hod or
        args.grad_vs_r_by_hod_wo_rtd or
        args.r_vs_grad or
        args.sat_vs_r or
        args.sdt_vs_r):

    if not args.i:

        # load bursts
        vb = _load_bursts(locs)

        # add lat/lon and sat grads
        if (args.tc_burst_stats or
                args.grad_vs_category or
                args.grad_vs_max_sustained_wind or
                args.grad_vs_r or
                args.grad_vs_r_by_hod or
                args.grad_vs_r_by_hod_wo_rtd or
                args.r_vs_grad or
                args.sat_vs_r or
                args.sdt_vs_r):

            glxy = pd.read_feather(storefolder + 'gl_r{}_p{}_JJA.feather'.format(r, p))
            cols = glxy.columns.values
            glxy.columns = pd.MultiIndex.from_tuples([[col, ''] for col in cols])
            vb = pd.merge(vb, glxy[['l', 'lat', 'lon']], how='left',
                          left_on='l', right_on='l', sort=False)

            # load temperature gradients
            vbg = _load_burst_grads(locs, twcols)
            print('adding sat grad to bursts ..')
            # cols = vbg.columns.values
            # vbg.columns = pd.MultiIndex.from_tuples([[col, ''] for col in cols])
            # vb = pd.concat((vb, vbg), axis=1)
            for twcol in twcols:
                vb[twcol, ''] = vbg[twcol].values

            print('done adding ..')

            # add additional seasons (JASO/DJFMA)
            # for some reason, 'dtime' is not recognized as dtime any more
            vb['dtime'] = pd.to_datetime(vb['dtime'])
            month = vb['dtime'].dt.month
            vb[('season', 'all')] = True
            vb[('season', 'JASO')] = False
            vb[('season', 'DJFMA')] = False
            vb.loc[month.isin([7, 8, 9, 10]), ('season', 'JASO')] = True
            vb.loc[month.isin([12, 1, 2, 3, 4]), ('season', 'DJFMA')] = True
            del month

    # add tropical cyclone flag - keep only tc bursts
    if args.only_tropical_cyclones:
        # load tc bursts
        glcp_tc = pd.read_hdf(tc_glcp_store, where='l in locs')
        # threshold by distance to eye
        if min_dist_to_eye is not None:
            glcp_tc = glcp_tc[glcp_tc['dist_to_eye'] > min_dist_to_eye]
        if max_dist_to_eye is not None:
            glcp_tc = glcp_tc[glcp_tc['dist_to_eye'] < max_dist_to_eye]
        # set index
        glcp_tc.set_index(['l', 'cp_burst'], inplace=True)
        # set column names
        glcp_tc.columns = pd.MultiIndex.from_tuples(
            [['id', ''], ['category', ''], ['max_sustained_wind', ''],
             ['radius_max_wind', ''], ['radius_oci', ''],
             ['dist_to_eye', ''], ['r_min_dist', '']])
        # merge
        vb = pd.merge(vb, glcp_tc, how='right',
                      left_on=['l', 'cp_burst'], right_index=True)
        # vb = vb.loc[~vb['id'].isnull()]
        del glcp_tc

    if args.single_burst:
        single_burst(vb)

    if args.agg_k_bursts:
        for season in seasons:
            k_bursts = vb.loc[
                vb['season', season], 'k_burst'].value_counts().index
            for k_burst in k_bursts:
                agg_bursts(vb, season, k_burst)

    if args.agg_all_bursts:
        for season in seasons:
            agg_bursts(vb, season)

    if args.agg_all_bursts_by_hod:
        for season in seasons:
            agg_bursts(vb, season, hod=True)

    if args.t_vs_tmps:
        for season in seasons:
            t_vs_temps_by_intensity(vb, season)

    if args.t_vs_tmps_by_hod:
        for season in seasons:
            t_vs_temps_by_intensity_by_hod(vb, season)

    if args.t_vs_sat_by_hod:
        for season in seasons:
            t_vs_sat_by_intensity_by_hod(vb, season)

    if args.t_vs_sat_by_hod_wo_rtd:
        for season in seasons:
            t_vs_sat_by_intensity_by_hod_wo_r_before(vb, season)

    if args.hod_distribution:
        for season in seasons:
            hod_distribution(vb, season)

    if args.sat_vs_r:
        for season in seasons:
            for twcol in twcols:
                sxt_vs_r(vb, season, which='sat', twcol=twcol)

    if args.sdt_vs_r:
        for season in seasons:
            for twcol in twcols:
                sxt_vs_r(vb, season, which='sdt', twcol=twcol)

    if args.tc_burst_stats:
        for season in seasons:
            for twcol in twcols:
                tc_burst_stats(vb, season, twcol=twcol)

    if args.grad_vs_category:
        for season in seasons:
            for twcol in twcols:
                grad_vs_category(vb, season, twcol=twcol, var='category')

    if args.grad_vs_max_sustained_wind:
        for season in seasons:
            for twcol in twcols:
                grad_vs_mst(vb, season, twcol=twcol, var='max_sustained_wind')

    if args.grad_vs_r:
        for season in seasons:
            for twcol in twcols:
                grad_vs_r(vb, season, twcol=twcol, var='r_max')

    if args.grad_vs_r_by_hod:
        for season in seasons:
            for twcol in twcols:
                grad_vs_r(
                    vb, season, col=twcol, hod=True, wo_r_before=False,
                    var='r_max')

    if args.grad_vs_r_by_hod_wo_rtd:
        for season in seasons:
            for twcol in twcols:
                grad_vs_r(
                    vb, season, col=twcol, hod=True, wo_r_before=True,
                    var='r_max')

    if args.r_vs_grad:
        for season in seasons:
            for twcol in twcols:
                r_vs_grad(vb, season, twcol=twcol, var='r_max')

# event related
if (args.sat_vs_r_by_td or
        args.sat_vs_r_by_hod or
        args.sat_vs_r_by_td_hod or
        args.sat_vs_r_by_hod_td or
        args.sdt_vs_sat_vs_x_by_td or
        args.sdt_vs_sat_vs_x_by_hod_td):

    ve = _load_events(locs)

    if args.sat_vs_r_by_td:
        for season in seasons:
            sat_vs_r_by_td(ve, season)

    if args.sat_vs_r_by_hod:
        for season in seasons:
            sat_vs_r_by_hod(ve, season)

    if args.sat_vs_r_by_td_hod:
        for season in seasons:
            sat_vs_r_by_td(ve, season, hod=True)

    if args.sat_vs_r_by_hod_td:
        for season in seasons:
            sat_vs_r_by_hod(ve, season, td=True)

    if args.sdt_vs_sat_vs_x_by_td:
        for var in ['n_nodes', 'r']:
            for season in seasons:
                sdt_vs_sat_vs_x_by_td(ve, var, season)

    if args.sdt_vs_sat_vs_x_by_hod_td:
        for var in ['n_nodes', 'r']:
            for season in seasons:
                sdt_vs_sat_vs_x_by_hod(ve, var, season, td=True)

# td vs alpha
if args.td_vs_alpha:
    for season in seasons:
        td_vs_alpha(season)
if args.td_vs_alpha_by_hod:
    for season in seasons:
        td_vs_alpha_by_hod(season)
