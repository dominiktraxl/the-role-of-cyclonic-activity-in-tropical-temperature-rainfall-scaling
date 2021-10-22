import matplotlib as mpl
mpl.use('svg')

import os
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

import numpy as np
import pandas as pd

# publication plot (default) parameters
# sns.set_context('paper', font_scale=.8)
# width = 3.465  # 6
# height = width / 1.618  # (golden ratio)
plt.rc('text', usetex=True)
# plt.style.use(os.getcwd() + '/../double_column.mplstyle')
width, height = plt.rcParams['figure.figsize']
plt.rc('font', family='serif', serif='Times')

# set hatching line width
mpl.rc('hatch', color='k', linewidth=.2)

# argument parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
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
cg_N = args.coarse_grained
plot_type = 'contourf'  # 'imshow'
# mincnt = 40*20
mincnt_bins = 20
mincnt_pts_per_bin = 100  # 20

# variables to plot
twcols = ['s_-6_-2']  # ['s_{}_-2'.format(tf) for tf in range(-24, -2, 6)]
# twcols = ['s_-24_-12']
hod = 'all'
varis = ['r_max']  # ['r_max', 'r_mean']  # ['r_mean', 'r_max']
seasons = ['JASO']  # , 'DJFMA']  # ['all', 'DJF', 'JJA']
wo_r_befores = [True]  # [False, True]
qs = [90]

# only information, not used!
keys = [
    'm',
    't',
    'rvalue',
    'pvalue',
    'stderr',
    'xgmin_q90',
    'xgmax_q90',
    'gmin_q90',
    'gmax_q90',
    'm_q90',
    't_q90',
    'rvalue_q90',
    'pvalue_q90',
    'stderr_q90',
    'xq1',
    'xq99',
    'yq1',
    'yq99',
    'n_bursts',
    'pts_per_bin',
]

# filesystem folders
storefolder = os.getcwd() + '/data_processing/'

if cg_N is not None:
    picfolder = os.getcwd() + \
        f'/map_plots/gl_cg_{cg_N}/r{r}_p{p}_grad_vs_r{tcfstr}/'
    glxy = pd.read_feather(storefolder + f'gl_r{r}_p{p}_cg_{cg_N}.feather')
    glfit = pd.read_pickle(
        storefolder + f'gl_r{r}_p{p}_cg_{cg_N}_grad_vs_r{tcfstr}_fits.pickle')
    glfit['lw'] = pd.read_pickle(
        storefolder + f'gl_cg_{cg_N}_regions.pickle')['lw']
    # glfit = pd.read_pickle(
    #     storefolder + f'gl_r{r}_p{p}_cg_{cg_N}_grad_vs_r{tcfstr}_fits_24_12.pickle')
    if cg_N == 2:
        lon = np.arange(-179.75, 180., .5)
        lat = np.arange(49.75, -50, -.5)
        z0 = np.zeros((200, 720), dtype=np.float64)
    elif cg_N == 4:
        lon = np.arange(-179.5, 180.5, 1)
        lat = np.arange(49.5, -50.5, -1)
        z0 = np.zeros((100, 360), dtype=np.float64)
    elif cg_N == 8:
        lon = np.arange(-179., 180., 2)
        lat = np.arange(49., -50., -2)
        z0 = np.zeros((50, 180), dtype=np.float64)

else:
    picfolder = os.getcwd() + \
        f'/map_plots/gl/r{r}_p{p}_grad_vs_r{tcfstr}/'
    glxy = pd.read_feather(storefolder + f'gl_r{r}_p{p}_JJA.feather')
    glfit = pd.read_pickle(
        storefolder + f'gl_r{r}_p{p}_grad_vs_r{tcfstr}_fits.pickle')
    lon = np.arange(-179.875, 180.125, .25)
    lat = np.arange(49.875, -50.125, -.25)
    z0 = np.zeros((400, 1440), dtype=np.float64)

os.makedirs(picfolder, exist_ok=True)
gl = pd.merge(glfit, glxy[['x', 'y', 'lon', 'lat']], how='left',
              left_index=True, right_index=True, sort=False)
lons, lats = np.meshgrid(lon, lat)

# store for nature communications data supplement
bs = 's_-6_-2_JASO_all_worb_r_max_'
# cols: lat, lon, n_bursts, p_value, alpha,
cols = ['m_left_q90', 'rvalue_left_q90', 'pvalue_left_q90',
        'pts_per_bin', 'pts_left_q90']
cols = [bs + col for col in cols]
data = gl[['lat', 'lon'] + cols]
data.rename(columns={
    'lat': 'latitude [°N]',
    'lon': 'longitude [°E]',
    bs+'m_left_q90': 'slope of regression between rainfall intensity (P^90) and temporal temperature gradients [mm/°C]',
    bs+'rvalue_left_q90': 'Pearson correlation coefficient of regression',
    bs+'pvalue_left_q90': 'p-value of regression',
    bs+'pts_per_bin': 'No. of episodes per bin',
    bs+'pts_left_q90': 'No. of bins with negative temporal temperature gradient'},
    inplace=True
)
os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
data.to_csv(storefolder + 'Nat_Comm_Fig_data/fig3bc.csv', index=False)


def cstr(col, season, hod, wo_r_before, var):
    # col, get rid of nans
    colstr = col
    # season
    if season == 'all':
        sstr = 'all'
    else:
        sstr = season
    # hod
    if hod == 'all':
        hodstr = 'all'
    else:
        hodstr = '{:02d}'.format(hod)
    # rain before?
    if wo_r_before:
        rbstr = 'worb'
    elif not wo_r_before:
        rbstr = 'wrb'
    # r_max or r_mean?
    varstr = var
    # add subset strings
    substr = '{}_{}_{}_{}_{}_'.format(colstr, sstr, hodstr, rbstr, varstr)
    return substr


# kwargs
kwds_basemap = {
    'projection': 'cyl',
    # 'lon_0': 0,
    # 'lat_0': 0,
    'llcrnrlon': -300,
    'urcrnrlon': 60,
    'llcrnrlat': -50,
    'urcrnrlat': 50,
    'resolution': 'c',
}

# for contourf
# levels_m = [-40, -30, -20, -10, -5, -1, 1, 5, 10, 20, 30, 40]
levels_m = np.arange(-50, 10, 10)  # [-80, -60, -40, -20, 0]
levels_p = [0, .05]
levels_r = [-.15, -.10, -.05, .05, .10, .15]
levels_r_q90 = [-1, -.8, -.6, -.4, -.2, 0]
levels_xgmin_max = [-.20, -.15, -.05, .05, .15, .20]

# colormaps
piyg = mpl.cm.get_cmap('PiYG')
piyg.set_bad(color='cornflowerblue')
viridis = mpl.cm.get_cmap('viridis')
viridis.set_bad(color='cornflowerblue')
puor = mpl.cm.get_cmap('PuOr')
puor.set_bad(color='cornflowerblue')

# variables to plot
data = {}
for season in seasons:
    for vari in varis:
        for wrb in wo_r_befores:
            for twcol in twcols:
                for q in qs:

                    subset = cstr(
                        col=twcol,
                        season=season,
                        hod='all',
                        wo_r_before=wrb,
                        var=vari,
                    )

                    if plot_type == 'contourf':

                        for which in ['', '_left', '_right']:
                            data.update({
                                subset + 'm' + which: {
                                    'cmap': piyg,
                                    'levels': levels_m,
                                    'latlon': True,
                                    'extend': 'both',
                                },
                                subset + 'rvalue' + which: {
                                    'cmap': piyg,
                                    'levels': levels_r,
                                    'latlon': True,
                                    'extend': 'both',
                                },
                                subset + 'pvalue' + which: {
                                    'cmap': viridis,
                                    'levels': levels_p,
                                    'extend': 'max',
                                    'latlon': True,
                                },
                                subset + 'm' + which + f'_q{q}': {
                                    'cmap': 'RdPu_r',
                                    'levels': levels_m,
                                    'latlon': True,
                                    'extend': 'both',
                                },
                                subset + 'rvalue' + which + f'_q{q}': {
                                    'cmap': 'viridis',
                                    'levels': levels_r_q90,
                                    'latlon': True,
                                    'extend': 'max',
                                },
                                subset + 'pvalue' + which + f'_q{q}': {
                                    'cmap': viridis,
                                    'levels': levels_p,
                                    'extend': 'max',
                                    'latlon': True,
                                }
                            })

                        data.update({
                            subset + 'xgmin_q90': {
                                'cmap': piyg,
                                'levels': levels_xgmin_max,
                                'latlon': True,
                                'extend': 'both',
                            },
                            subset + 'xgmax_q90': {
                                'cmap': piyg,
                                'levels': levels_xgmin_max,
                                'latlon': True,
                                'extend': 'both',
                            },
                            subset + 'n_bursts': {'cmap': viridis,
                                                  'latlon': True,
                                                  'extend': 'both'},
                            subset + 'pts_per_bin': {'cmap': viridis,
                                                     'latlon': True,
                                                     'extend': 'both'},
                        })

                    if plot_type == 'imshow':

                        for which in ['', '_left', '_right']:
                            data.update({
                                subset + 'pts' + which: {'cmap': viridis,},
                                subset + 'm' + which: {
                                    'cmap': piyg,
                                    'vmin': levels_m[0],
                                    'vmax': levels_m[-1],
                                },
                                subset + 'rvalue' + which: {
                                    'cmap': piyg,
                                    'vmin': levels_r[0],
                                    'vmax': levels_r[-1],
                                },
                                subset + 'pvalue' + which: {
                                    'cmap': viridis,
                                    'vmin': levels_p[0],
                                    'vmax': levels_p[-1],
                                },
                                subset + 'pts' + which + f'_q{q}': {'cmap': viridis},
                                subset + 'm' + which + f'_q{q}': {
                                    'cmap': piyg,
                                    'vmin': levels_m[0],
                                    'vmax': levels_m[-1],
                                },
                                subset + 'rvalue' + which + f'_q{q}': {
                                    'cmap': piyg,
                                    'vmin': levels_r_q90[0],
                                    'vmax': levels_r_q90[-1],
                                },
                                subset + 'pvalue' + which + f'_q{q}': {
                                    'cmap': viridis,
                                    'vmin': levels_p[0],
                                    'vmax': levels_p[-1],
                                }
                            })

                        data.update({
                            subset + 'xgmin_q90': {
                                'cmap': piyg,
                                'vmin': levels_xgmin_max[0],
                                'vmax': levels_xgmin_max[-1],
                            },
                            subset + 'xgmax_q90': {
                                'cmap': piyg,
                                'vmin': levels_xgmin_max[0],
                                'vmax': levels_xgmin_max[-1],
                            },
                            subset + 'n_bursts': {'cmap': viridis,},
                            subset + 'pts_per_bin': {'cmap': viridis,},
                        })

for var, kwds_update in data.items():

    if (not var.endswith('m_left_q90') and not
        var.endswith('m_left_q95') and not
        var.endswith('m_left_q99') and not
        var.endswith('rvalue_left_q90') and not
        var.endswith('rvalue_left_q95') and not
            var.endswith('rvalue_left_q99')):
        continue

    # folder
    tw = var.split('_')[1]
    subfolder = '{}/'.format(tw)
    os.makedirs(picfolder + subfolder, exist_ok=True)

    # plot
    fig, ax = plt.subplots(figsize=(width, height))
    m = Basemap(ax=ax, **kwds_basemap)

    # create matrix
    z = z0.copy()
    z[:, :] = np.nan

    # only show locations with at least "mincnt" bursts per bin
    pts_per_bin_col = '_'.join(var.split('_')[:8]) + '_pts_per_bin'
    if (var.endswith('m_left_q90') or
        var.endswith('m_left_q95') or
        var.endswith('m_left_q99') or
        var.endswith('rvalue_left_q90') or
        var.endswith('rvalue_left_q95') or
            var.endswith('rvalue_left_q99')):
        n_bins_col = '_'.join(var.split('_')[:8]) + '_pts_left_q90'
        gl.loc[gl[pts_per_bin_col] < mincnt_pts_per_bin, var] = np.nan
        gl.loc[gl[n_bins_col] < mincnt_bins, var] = np.nan
    elif (var.endswith('m_right_q90') or
          var.endswith('m_right_q95') or
          var.endswith('m_right_q99') or
          var.endswith('rvalue_right_q90') or
          var.endswith('rvalue_right_q95') or
          var.endswith('rvalue_right_q99')):
        n_bins_col = '_'.join(var.split('_')[:8]) + '_pts_right_q90'
        gl.loc[gl[pts_per_bin_col] < mincnt_pts_per_bin, var] = np.nan
        gl.loc[gl[n_bins_col] < mincnt_bins, var] = np.nan

    # only show locations with p < .05
    p_col = ('_'.join(var.split('_')[:8]) +
             '_pvalue_' + var.split('_')[-2] + '_' + var.split('_')[-1])
    # gl.loc[gl[p_col] > .05, var] = np.nan

    # set matrix values
    z[gl['y'], gl['x']] = gl[var].values

    # only water
    gllw = gl[gl['lw'] == 'L']
    z[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan

    # pvalue matrix
    zp = z.copy()
    zp[:, :] = np.nan
    zp[gl['y'], gl['x']] = gl[p_col].values

    # kwds_use = kwds_pcolormesh.copy()
    kwds_use = kwds_update.copy()

    if plot_type == 'contourf':
        z = z[::-1]

        # contourf
        im = m.contourf(lons, lats, z, **kwds_use)

        # hatching p > 0.05 values
        m.contourf(
            lons, lats,
            np.ma.masked_where(zp[::-1] <= 0.05, z),
            # levels_l,
            latlon=True,
            ax=ax,
            # extend='both',
            # vmin=-30, vmax=30,
            colors='none',
            # hatches=[5*'.'],
            hatches=[15*'x'],
        )

    elif plot_type == 'imshow':
        z = np.roll(z, 60, axis=1)
        # imshow
        im = m.imshow(z, **kwds_use)

    # pcolormesh
    # im = m.pcolormesh(lons, lats, z, **kwds_use)

    # scatter
    # pc = m.scatter(
    #     gl['lon'], gl['lat'], s=1, c=gl[var], latlon=True,
    #     vmin=gl[var].quantile(.01), vmax=gl[var].quantile(.99),
    #     marker='s', edgecolors='none')
    # map stuff
    m.drawcoastlines(
        ax=ax, zorder=10,
        # linewidth=1,
    )

    parallels = [-23, 0, 23]  # np.arange(-50, 50, 5)  #
    meridians = np.arange(-180, 180, 60)  # np.arange(-180, 180, 5)  #
    m.drawparallels(
        parallels,
        linewidth=.5,
        labels=[True, False, False, False],
        dashes=(None,None),
    )
    merids = m.drawmeridians(
        meridians,
        linewidth=.5,
        labels=[False, False, True, False],
        dashes=(None,None),
    )
    # for merid in merids:
    #     merids[merid][1][0].set_rotation(90)

    # # boxes
    # # northern atlantic neg alpha
    # lat_min, lat_max, lon_min, lon_max = 12, 18, -82, -76
    # poly = mpl.patches.Polygon(np.array(
    #     [(lon_min, lat_min), (lon_max, lat_min),
    #      (lon_max, lat_max), (lon_min, lat_max)]),
    #     facecolor='none', edgecolor='cyan',
    #     linewidth=1.5,
    # )
    # ax.add_patch(poly)
    # # northern atlantic pos alpha
    # lat_min, lat_max, lon_min, lon_max = 16, 22, -46, -40
    # poly = mpl.patches.Polygon(np.array(
    #     [(lon_min, lat_min), (lon_max, lat_min),
    #      (lon_max, lat_max), (lon_min, lat_max)]),
    #     facecolor='none', edgecolor='cyan',
    #     linewidth=1.5,
    # )
    # ax.add_patch(poly)
    # # northern pacific neg alpha
    # lat_min, lat_max, lon_min, lon_max = 2, 8, 130-360, 136-360
    # poly = mpl.patches.Polygon(np.array(
    #     [(lon_min, lat_min), (lon_max, lat_min),
    #      (lon_max, lat_max), (lon_min, lat_max)]),
    #     facecolor='none', edgecolor='cyan',
    #     linewidth=1.5,
    # )
    # ax.add_patch(poly)
    # # northern pacific pos alpha
    # lat_min, lat_max, lon_min, lon_max = 18, 24, 124-360, 130-360
    # poly = mpl.patches.Polygon(np.array(
    #     [(lon_min, lat_min), (lon_max, lat_min),
    #      (lon_max, lat_max), (lon_min, lat_max)]),
    #     facecolor='none', edgecolor='cyan',
    #     linewidth=1.5,
    # )
    # ax.add_patch(poly)
    # # northern indian neg alpha
    # lat_min, lat_max, lon_min, lon_max = 2, 8, 64-360, 70-360
    # poly = mpl.patches.Polygon(np.array(
    #     [(lon_min, lat_min), (lon_max, lat_min),
    #      (lon_max, lat_max), (lon_min, lat_max)]),
    #     facecolor='none', edgecolor='cyan',
    #     linewidth=1.5,
    # )
    # ax.add_patch(poly)
    # # northern indian pos alpha
    # lat_min, lat_max, lon_min, lon_max = 16, 22, 62-360, 68-360
    # poly = mpl.patches.Polygon(np.array(
    #     [(lon_min, lat_min), (lon_max, lat_min),
    #      (lon_max, lat_max), (lon_min, lat_max)]),
    #     facecolor='none', edgecolor='cyan',
    #     linewidth=1.5,
    # )
    # ax.add_patch(poly)

    # region lines
    # season = var.split('_')[3]
    # if season == 'JJA':
    #     m.drawparallels(
    #         [-16, 36], linewidth=3, labels=[False, False, False, False],
    #         dashes=[1, 0],
    #         zorder=20)
    # if season == 'DJF':
    #     m.drawparallels(
    #         [-24, 21], linewidth=3, labels=[False, False, False, False],
    #         dashes=[1, 0],
    #         zorder=20)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=.08)
    cb = fig.colorbar(im, cax=cax,
                      orientation='horizontal',
                      extend='both',
                      )

    if (var.endswith('m_left_q90') or
        var.endswith('m_left_q95') or
            var.endswith('m_left_q99')):
        q = var.split('_')[-1][1:]
        cax.set_xlabel(
            r"slope of regression between "
            "$P^{" + q + "}$ "
            "and $T^r_g$ [mm$^{\circ}$C$^{-1}$]"
        )
    elif (var.endswith('rvalue_left_q90') or
          var.endswith('rvalue_left_q95') or
            var.endswith('rvalue_left_q99')):
        q = var.split('_')[-1][1:]
        cax.set_xlabel(
            r"PCC between "
            "$P^{" + q + "}$ "
            " and $T^r_g$")

    # fake colorbars
    # fax1 = divider.append_axes("bottom", size="5%", pad=.4)
    # fax2 = divider.append_axes("bottom", size="2%", pad=.4)
    # fax1.axis('off')
    # fax2.axis('off')

    # title
    # ax.set_title(var)

    # save figure
    picfile = picfolder + subfolder + var

    # png
    fig.savefig(
        picfile + '.png',
        format='png',
        bbox_inches='tight', pad_inches=0,
        dpi=600,
    )
    # pdf
    fig.savefig(
        picfile + '.pdf',
        format='pdf',
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


# per_tw()
# tw_argmin_max(var_str='m', var_str_q90='_q90')
# tw_argmin_max(var_str='pvalue', var_str_q90='_q90')
# tw_argmin_max(var_str='rvalue', var_str_q90='_q90')
