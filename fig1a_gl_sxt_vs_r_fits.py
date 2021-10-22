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
# plt.style.use(os.getcwd() + '/../../double_column.mplstyle')
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
if args.only_tropical_cyclones:
    mincnt = 40*20
else:
    mincnt = 40*100  # 40*20

# variables all bursts
if not args.only_tropical_cyclones:
    sxtcols = ['sat']  # ['sat', 'sdt']
    tds = [2]  # [0, 1, 2, 3, 4, 5, 6]
    hod = 'all'
    varis = ['r_max']  # ['r_mean', 'r_max']
    seasons = ['JASO']  # ['JASO', 'DJFMA']
    wo_r_befores = [True]  # [False, True]
    grads = ['agrad']  # ['ngrad', 'pgrad', 'agrad']
    rh_ranges = ['worhs']  # ['wrhmean', 'wrh80', 'wrrhmean', 'wrrh80']
    tcs = [False]  # [False, True]
    qs = [90]  # [90, 95, 99]

# variables only tropical cyclones
elif args.only_tropical_cyclones:
    sxtcols = ['sat']
    tds = [2]  # [0, 1, 2, 3, 4, 5, 6]
    hod = 'all'
    varis = ['r_max']  # ['r_max', 'r_mean']
    seasons = ['JASO', 'DJFMA']  # ['all']
    wo_r_befores = [True, False]
    grads = ['agrad']  # ['ngrad', 'pgrad', 'agrad']
    rh_ranges = ['worhs']  # ['wrhmean', 'wrh80', 'wrrhmean', 'wrrh80']
    tcs = [True]  # [False, True]
    qs = [90]

keys = [
    # lowess
    # 'ppt',
    # linregress (all)
    'exp_alpha',
    # 'exp_rvalue',
    # 'exp_pvalue',
    # gen. log. (all)
    # 'saturation',
]

# filesystem folders
storefolder = os.getcwd() + '/data_processing/'

if cg_N is not None:
    picfolder = os.getcwd() + \
        f'/map_plots/gl_cg_{cg_N}/r{r}_p{p}_sxt_vs_r{tcfstr}/'
    glxy = pd.read_feather(
        storefolder + f'gl_r{r}_p{p}_cg_{cg_N}.feather')
    glfit = pd.read_pickle(
        storefolder + f'gl_r{r}_p{p}_cg_{cg_N}_sxt_vs_r{tcfstr}_fits.pickle')
    glfit['lw'] = pd.read_pickle(
        storefolder + f'gl_cg_{cg_N}_regions.pickle')['lw']
    if cg_N == 2:
        n_bins = 20
        lon = np.arange(-179.75, 180., .5)
        lat = np.arange(49.75, -50, -.5)
        z0 = np.zeros((200, 720), dtype=np.float64)
        roll = 240
    elif cg_N == 4:
        n_bins = 30
        lon = np.arange(-179.5, 180.5, 1)
        lat = np.arange(49.5, -50.5, -1)
        z0 = np.zeros((100, 360), dtype=np.float64)
        roll = 120
    elif cg_N == 8:
        n_bins = 40
        lon = np.arange(-179., 180., 2)
        lat = np.arange(49., -50., -2)
        z0 = np.zeros((50, 180), dtype=np.float64)
        roll = 60

else:
    n_bins = 15
    picfolder = os.getcwd() + \
        f'/map_plots/gl/r{r}_p{p}_sxt_vs_r{tcfstr}/'
    glxy = pd.read_feather(storefolder + f'gl_r{r}_p{p}_JJA.feather')
    glfit = pd.read_pickle(
        storefolder + f'gl_r{r}_p{p}_sxt_vs_r{tcfstr}_fits.pickle')
    lon = np.arange(-179.875, 180.125, .25)
    lat = np.arange(49.875, -50.125, -.25)
    z0 = np.zeros((400, 1440), dtype=np.float64)
    roll = 480

os.makedirs(picfolder, exist_ok=True)
gl = pd.merge(glfit, glxy[['x', 'y', 'lat', 'lon']], how='left', left_index=True,
              right_index=True, sort=False)
lons, lats = np.meshgrid(lon, lat)

# store for nature communications data supplement
bs = 'ntcs_sat_2_JASO_all_worb_agrad_worhs_r_max_'
# cols: lat, lon, n_bursts, p_value, alpha,
data = gl[['lat', 'lon', bs+'n_bursts', bs+'exp_pvalue_q90', bs+'exp_alpha_q90']]
data.rename(columns={
    'lat': 'latitude [°N]',
    'lon': 'longitude [°E]',
    bs+'n_bursts': 'No. of episodes',
    bs+'exp_pvalue_q90': 'p-value of the regression',
    bs+'exp_alpha_q90': 'alpha-scaling factor [%/°C]'},
    inplace=True
)
os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
data.to_csv(storefolder + 'Nat_Comm_Fig_data/fig1a.csv', index=False)

# load SST data for contour line
sst_mean = {}
for season in seasons:
    sst_mean[season] = np.load(storefolder + f'SST_{season}_mean.pkl',
                               allow_pickle=True)

sst_mean['JASO_contourline_temp'] = 28
sst_mean['JASO_clabel_loc'] = [(-150, 15)]
sst_mean['DJFMA_contourline_temp'] = 28.5
sst_mean['DJFMA_clabel_loc'] = [(-170, 0)]

lons_sst, lats_sst = np.meshgrid(np.arange(-179.875, 180.125, .25),
                                 np.arange(89.875, -90.125, -.25))


def cstr(tc, sxtcol, td, season, hod, wo_r_before, grad, rh_range, var):
    if tc is True:
        tcstr = 'tcs'
    elif tc is False:
        tcstr = 'ntcs'
    sxtcolstr = sxtcol
    tdstr = str(td)
    if season == 'all':
        sstr = 'all'
    else:
        sstr = season
    if hod == 'all':
        hodstr = 'all'
    else:
        hodstr = '{:02d}'.format(hod)
    if wo_r_before:
        rbstr = 'worb'
    elif not wo_r_before:
        rbstr = 'wrb'
    varstr = var
    gradstr = grad
    rhrstr = rh_range
    substr = '{}_{}_{}_{}_{}_{}_{}_{}_{}_'.format(
        tcstr, sxtcolstr, tdstr, sstr, hodstr, rbstr, gradstr, rhrstr, varstr)
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


def per_td():

    # variables to plot
    data = {}
    for tc in tcs:
        for sxtcol in sxtcols:
            for td in tds:
                for season in seasons:
                    for vari in varis:
                        for wrb in wo_r_befores:
                            for grad in grads:
                                for rh_range in rh_ranges:
                                    for q in qs:

                                        subset = cstr(
                                            tc=tc,
                                            sxtcol=sxtcol,
                                            td=td,
                                            season=season,
                                            hod=hod,
                                            wo_r_before=wrb,
                                            grad=grad,
                                            rh_range=rh_range,
                                            var=vari,
                                        )

                                        for key in keys:
                                            data.update(
                                                {subset + key + f'_q{q}': {}})

                                        # nr burst, pts per bin
                                        data.update({
                                            # subset + 'n_bursts': {},
                                            # subset + 'pts_per_bin': {},
                                        })

    # colormaps
    # piyg = mpl.cm.get_cmap('PiYG')
    # piyg.set_bad(color='cornflowerblue', alpha=0)
    # viridis = mpl.cm.get_cmap('viridis')
    # viridis.set_bad(color='cornflowerblue', alpha=0)
    # puor = mpl.cm.get_cmap('PuOr')
    # puor.set_bad(color='cornflowerblue', alpha=0)

    for var, kwds_update in data.items():

        # folder
        sxtcol = var.split('_')[1]
        td = var.split('_')[2]
        season = var.split('_')[3]
        subfolder = '{}/{}/'.format(sxtcol, td)
        os.makedirs(picfolder + subfolder, exist_ok=True)

        # plot
        fig_l, ax_l = plt.subplots(figsize=(width, height))
        fig_w, ax_w = plt.subplots(figsize=(width, height))

        m_l = Basemap(ax=ax_l, **kwds_basemap)
        m_w = Basemap(ax=ax_w, **kwds_basemap)

        # only show locations with at least "mincnt" bursts per bin
        n_bursts_col = '_'.join(var.split('_')[:10]) + '_n_bursts'
        if not var.endswith('pts_per_bin') and not var.endswith('n_bursts'):
            gl.loc[gl[n_bursts_col] < mincnt, var] = np.nan

        # split by significance (p=.05)
        p_col = '_'.join(var.split('_')[:10]) + '_exp_pvalue_' + var.split('_')[-1]
        # if not var.endswith('pts_per_bin') and not var.endswith('n_bursts'):
        #     gl.loc[gl[p_col] > .05, var] = np.nan
        #     gls = gl.copy()
        #     glis = gl.copy()
        #     gls.loc[gl[p_col] > .05, var] = np.nan
        #     glis.loc[gl[p_col] <= .05, var] = np.nan

        if 'alpha' in var:

            # contourf
            levels_l = [-21, -14, -7, 0, 7, 14, 21]
            levels_w = [-42, -28, -14, 0, 14, 28, 42]

            # create pvalues matrix
            zp = z0.copy()
            zp[:, :] = np.nan
            zp[gl['y'], gl['x']] = gl[p_col].values

            # create matrix
            z = z0.copy()
            z[:, :] = np.nan
            z[gl['y'], gl['x']] = gl[var].values

            # land
            zl = z.copy()
            gllw = gl[gl['lw'] == 'W']
            zl[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zl = np.ma.masked_where(zl == 999999, zl)

            # land (not significant)
            # zlis = z.copy()
            # zlis[:, :] = np.nan
            # glis = gl.loc[(gl[p_col] > .05) & (gl['lw'] == 'L')]
            # zlis[glis['y'], glis['x']] = glis[var].values

            # water
            zw = z.copy()
            gllw = gl[gl['lw'] == 'L']
            zw[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zw = np.ma.masked_where(zl == 999999, zl)

            # water (not significant)
            # zwis = z.copy()
            # zwis[:, :] = np.nan
            # glis = gl.loc[(gl[p_col] > .05) & (gl['lw'] == 'W')]
            # zwis[glis['y'], glis['x']] = glis[var].values

            # nans
            znan = z0.copy()
            znan[gl.loc[gl[var].isnull(), 'y'], gl.loc[gl[var].isnull(), 'x']] = 1
            znan = np.ma.masked_where(znan == 0, znan)

            # interpolate/smoothen
            # zw = scipy.ndimage.zoom(zw, 3)
            # zl = scipy.ndimage.zoom(zl, 3)

            # zw = gaussian_filter(zw, sigma=.1)
            # zl = gaussian_filter(zl, sigma=.1)

            # contourf (land)
            iml = m_l.contourf(
                lons, lats[::-1], zl,
                levels_l,
                latlon=True,
                ax=ax_l,
                extend='both',
                # vmin=-30, vmax=30,
                cmap='PiYG',
            )

            # contourf (land, not significant))
            imlis = m_l.contourf(
                lons, lats[::-1],
                # zlis,
                np.ma.masked_where(zp <= 0.05, zl),
                # levels_l,
                latlon=True,
                ax=ax_l,
                # extend='both',
                # vmin=-30, vmax=30,
                colors='none',
                # hatches=[5*'.'],
                hatches=[15*'x'],
            )

            # contourf (water)
            imw = m_w.contourf(
                lons, lats[::-1], zw,
                levels_w,
                latlon=True,
                ax=ax_w,
                extend='both',
                # vmin=-30, vmax=30,
                cmap='PuOr',
            )

            # contourf (water, not significant)
            imwis = m_w.contourf(
                lons, lats[::-1],
                # zwis,
                np.ma.masked_where(zp <= 0.05, zw),
                # levels_w,
                latlon=True,
                ax=ax_w,
                # extend='both',
                # vmin=-30, vmax=30,
                colors='none',
                # hatches=[5*'.'],
                hatches=[15*'x'],
                # clip_on=False,  # no effect
            )

            # imshow
    #         z = np.roll(z, roll, axis=1)
    #         zl = np.roll(zl, roll, axis=1)
    #         zw = np.roll(zw, roll, axis=1)
    #         znan = np.roll(znan, roll, axis=1)

            # im = m.imshow(z, vmin=-28, vmax=28, cmap=piyg, interpolation='none')

            # imshow (land)
    #         iml = m_l.imshow(
    #             zl, vmin=-2*7, vmax=2*7, cmap='RdYlGn',
    #             interpolation='nearest',
    #             ax=ax_l,
    #         )

            # imshow (water)
    #         imw = m_w.imshow(
    #             zw, vmin=-6*7, vmax=6*7, cmap='coolwarm',
    #             interpolation='nearest',
    #             ax=ax_w,
    #         )

            # imshow (nan)
            # cmap = mpl.colors.ListedColormap(['cornflowerblue'])
            # imnan = m.imshow(znan, cmap=cmap, interpolation='none', ax=ax)

            # sst mean contour line
            z = sst_mean[season][::-1]
            cs = m_w.contour(lons_sst, lats_sst, z,
                             # cmap='viridis',
                             linewidths=2,
                             colors=['red'],
                             levels=[sst_mean[f'{season}_contourline_temp']],
                             ax=ax_w,
                             latlon=True)
            cl = ax_w.clabel(cs, fmt=lambda x: r'\textbf{{{:.1f} °C}}'.format(x),
                             # fontsize=12,
                             manual=sst_mean[f'{season}_clabel_loc'])

            [txt.set_backgroundcolor('whitesmoke') for txt in cl]
            [txt.set_bbox(
                dict(facecolor='whitesmoke',
                     edgecolor='none', pad=1)) for txt in cl]

            # cl[0].set_weight('extra bold')

        elif 'rvalue' in var:

            # create matrix
            z = z0.copy()
            z[:, :] = np.nan
            z[gl['y'], gl['x']] = gl[var].values
            z = np.roll(z, roll, axis=1)

            # imshow
            # im = m.imshow(z, vmin=-1, vmax=1, cmap=piyg, interpolation='none')

        elif 'pvalue' in var:

            # contourf
            levels_p = [0, .05, 1]

            # create matrix
            z = z0.copy()
            z[:, :] = np.nan
            z[gl['y'], gl['x']] = gl[var].values

            zl = z.copy()
            gllw = gl[gl['lw'] == 'W']
            zl[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zl = np.ma.masked_where(zl == 999999, zl)

            zw = z.copy()
            gllw = gl[gl['lw'] == 'L']
            zw[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zw = np.ma.masked_where(zl == 999999, zl)

            # contourf (land)
            iml = m_l.contourf(
                lons, lats[::-1], zl,
                levels_p,
                latlon=True,
                ax=ax_l,
                # extend='both',
                # vmin=-30, vmax=30,
                # cmap='PiYG',
            )

            # contourf (water)
            imw = m_w.contourf(
                lons, lats[::-1], zw,
                levels_p,
                latlon=True,
                ax=ax_w,
                # extend='both',
                # vmin=-30, vmax=30,
                # cmap='PuOr',
            )

            # # imshow
            # z = np.roll(z, roll, axis=1)
            # zl = np.roll(zl, roll, axis=1)
            # zw = np.roll(zw, roll, axis=1)
            #
            # # imshow (land)
            # iml = m_l.imshow(
            #     zl,
            #     vmin=0,
            #     vmax=.05,
            #     # cmap='RdYlGn',
            #     interpolation='none',
            #     ax=ax_l,
            # )
            #
            # # imshow (water)
            # imw = m_w.imshow(
            #     zw,
            #     vmin=0,
            #     vmax=.05,
            #     # cmap='coolwarm',
            #     interpolation='none',
            #     ax=ax_w,
            # )

            # sst mean contour line
            z = sst_mean[season][::-1]
            cs = m_w.contour(lons_sst, lats_sst, z,
                             # cmap='viridis',
                             linewidths=2,
                             colors=['red'],
                             levels=[sst_mean[f'{season}_contourline_temp']],
                             ax=ax_w,
                             latlon=True)
            cl = ax_w.clabel(cs, fmt=lambda x: r'\textbf{{{:.1f} °C}}'.format(x),
                             # fontsize=12,
                             manual=sst_mean[f'{season}_clabel_loc'])

            [txt.set_backgroundcolor('whitesmoke') for txt in cl]
            [txt.set_bbox(
                dict(facecolor='whitesmoke',
                     edgecolor='none', pad=1)) for txt in cl]

            # cl[0].set_weight('extra bold')

        elif 'pts_per_bin' in var:

            # contourf
            levels = [0, .05, 1]

            # create matrix
            z = z0.copy()
            z[:, :] = np.nan
            z[gl['y'], gl['x']] = gl[var].values

            zl = z.copy()
            gllw = gl[gl['lw'] == 'W']
            zl[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zl = np.ma.masked_where(zl == 999999, zl)

            zw = z.copy()
            gllw = gl[gl['lw'] == 'L']
            zw[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zw = np.ma.masked_where(zl == 999999, zl)

            # contourf (land)
            iml = m_l.contourf(
                lons, lats[::-1], zl,
                # levels_p,
                latlon=True,
                ax=ax_l,
                # extend='both',
                # vmin=-30, vmax=30,
                # cmap='PiYG',
            )

            # contourf (water)
            imw = m_w.contourf(
                lons, lats[::-1], zw,
                # levels_p,
                latlon=True,
                ax=ax_w,
                # extend='both',
                # vmin=-30, vmax=30,
                # cmap='PuOr',
            )

            # # imshow
            # z = np.roll(z, roll, axis=1)
            # zl = np.roll(zl, roll, axis=1)
            # zw = np.roll(zw, roll, axis=1)
            #
            # # imshow (land)
            # iml = m_l.imshow(
            #     zl,
            #     vmin=0,
            #     vmax=.05,
            #     # cmap='RdYlGn',
            #     interpolation='none',
            #     ax=ax_l,
            # )
            #
            # # imshow (water)
            # imw = m_w.imshow(
            #     zw,
            #     vmin=0,
            #     vmax=.05,
            #     # cmap='coolwarm',
            #     interpolation='none',
            #     ax=ax_w,
            # )

            # # sst mean contour line
            # z = sst_mean[season][::-1]
            # cs = m_w.contour(lons_sst, lats_sst, z,
            #                  # cmap='viridis',
            #                  linewidths=2,
            #                  colors=['red'],
            #                  levels=[sst_mean[f'{season}_contourline_temp']],
            #                  ax=ax_w,
            #                  latlon=True)
            # cl = ax_w.clabel(cs, fmt=lambda x: r'\textbf{{{:.1f} °C}}'.format(x),
            #                  fontsize=12,
            #                  manual=sst_mean[f'{season}_clabel_loc'])
            #
            # [txt.set_backgroundcolor('whitesmoke') for txt in cl]
            # [txt.set_bbox(
            #     dict(facecolor='whitesmoke',
            #          edgecolor='none', pad=1)) for txt in cl]
            #

            # # cl[0].set_weight('extra bold')

        elif 'n_bursts' in var:

            # create matrix
            z = z0.copy()
            z[:, :] = np.nan
            z[gl['y'], gl['x']] = gl[var].values

            zl = z.copy()
            gllw = gl[gl['lw'] == 'W']
            zl[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zl = np.ma.masked_where(zl == 999999, zl)

            zw = z.copy()
            gllw = gl[gl['lw'] == 'L']
            zw[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
            # zw = np.ma.masked_where(zl == 999999, zl)

            # contourf (land)
            iml = m_l.contourf(
                lons, lats[::-1], zl,
                levels=np.arange(0, 32000, 4000),
                latlon=True,
                ax=ax_l,
                extend='max',
                # vmin=-30,
                # vmax=28000,
                cmap='plasma_r',
            )

            # contourf (water)
            imw = m_w.contourf(
                lons, lats[::-1], zw,
                levels=np.arange(0, 32000, 4000),
                latlon=True,
                ax=ax_w,
                extend='max',
                # vmin=-30,
                # vmax=28000,
                cmap='plasma_r',
            )

            # # imshow
            # z = np.roll(z, roll, axis=1)
            # zl = np.roll(zl, roll, axis=1)
            # zw = np.roll(zw, roll, axis=1)
            #
            # # imshow (land)
            # iml = m_l.imshow(
            #     zl,
            #     vmin=0,
            #     vmax=.05,
            #     # cmap='RdYlGn',
            #     interpolation='none',
            #     ax=ax_l,
            # )
            #
            # # imshow (water)
            # imw = m_w.imshow(
            #     zw,
            #     vmin=0,
            #     vmax=.05,
            #     # cmap='coolwarm',
            #     interpolation='none',
            #     ax=ax_w,
            # )

            # sst mean contour line
            z = sst_mean[season][::-1]
            cs = m_w.contour(lons_sst, lats_sst, z,
                             # cmap='viridis',
                             linewidths=2,
                             colors=['k'],
                             levels=[sst_mean[f'{season}_contourline_temp']],
                             ax=ax_w,
                             latlon=True)
            cl = ax_w.clabel(cs, fmt=lambda x: r'\textbf{{{:.1f} °C}}'.format(x),
                             # fontsize=12,
                             manual=sst_mean[f'{season}_clabel_loc'])

            [txt.set_backgroundcolor('whitesmoke') for txt in cl]
            [txt.set_bbox(dict(facecolor='whitesmoke',
                               edgecolor='none', pad=1)) for txt in cl]

            # cl[0].set_weight('extra bold')

        else:

            # create matrix
            z = z0.copy()
            z[:, :] = np.nan
            z[gl['y'], gl['x']] = gl[var].values
            z = np.roll(z, roll, axis=1)

            # imshow
            # im = m.imshow(z, interpolation='none')

        # basemap stuf
        def basemap_stuff(m, ax, which):

            m.drawcoastlines(
                ax=ax, zorder=10,
                # linewidth=1,
            )

            parallels = [-23, 0, 23]  # np.arange(-50, 50, 5)  #
            m.drawparallels(
                parallels,
                dashes=(None,None),
                linewidth=.5,
                labels=[True, False, False, False],
            )
            meridians = np.arange(-180, 180, 60)  # np.arange(-180, 180, 5)  #
            m.drawmeridians(
                meridians,
                dashes=(None,None),
                linewidth=.5,
                labels=[False, False, True, False],
            )
            # for merid in merids:
            #     merids[merid][1][0].set_rotation(90)

            # boxes
            if 'pts_per_bin' not in var:
                if which == 'water':
                    # northern atlantic neg alpha
                    lat_min, lat_max, lon_min, lon_max = 12, 18, -82, -76
                    poly = mpl.patches.Polygon(np.array(
                        [(lon_min, lat_min), (lon_max, lat_min),
                         (lon_max, lat_max), (lon_min, lat_max)]),
                        facecolor='none', edgecolor='cyan',
                        # linewidth=1.5,
                    )
                    ax.add_patch(poly)
                    # northern atlantic pos alpha
                    lat_min, lat_max, lon_min, lon_max = 16, 22, -46, -40
                    poly = mpl.patches.Polygon(np.array(
                        [(lon_min, lat_min), (lon_max, lat_min),
                         (lon_max, lat_max), (lon_min, lat_max)]),
                        facecolor='none', edgecolor='snow',
                        # linewidth=1.5,
                    )
                    ax.add_patch(poly)
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

        basemap_stuff(m_w, ax_w, 'water')
        basemap_stuff(m_l, ax_l, 'land')

        # region lines
        # season = var.split('_')[2]
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

        def colorbars(fig, ax, im):

            divider = make_axes_locatable(ax)

            cax = divider.append_axes("bottom", size="5%", pad=.08)
            _ = fig.colorbar(im, cax=cax,
                             orientation='horizontal',
                             # extend='both',
                             )

            if 'pts_per_bin' in var:
                cax.set_xlabel("points per bin")

            elif 'n_bursts' in var:
                cax.set_xlabel('number of rainfall episodes')

            else:
                cax.set_xlabel(
                    r"$\alpha$ [\%$^{\circ}$C$^{{-1}}$]"
                    # "\n"
                    # r"(ocean)",
                )

                # caxl = divider.append_axes("right", size="3%", pad=.45)
                # cbl = fig.colorbar(iml, cax=caxl,
                #                    orientation='vertical',
                #                    extend='both',
                #                    )
                # caxl.set_xlabel(
                #     r"$\alpha$ [\%$^{\circ}$C$^{{-1}}$]"
                #     "\n"
                #     r"(land)",
                # )

        colorbars(fig_w, ax_w, imw)
        colorbars(fig_l, ax_l, iml)

        # title
        # ax.set_title(var)

        # save figure
        # fig.tight_layout()
        def store_figure(fig, which):
            picfile = picfolder + subfolder + var + which

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

        store_figure(fig_l, '_land')
        store_figure(fig_w, '_water')


def t_24_2_difference():

        # parameters
        for season in ['JASO', 'DJFMA']:

            # var
            var = f'ntcs_sat_24_minus_2_{season}_all_worb_agrad_worhs_r_max_exp_alpha_q90'

            # folder
            subfolder = '{}/{}/'.format('sat', 24)
            os.makedirs(picfolder + subfolder, exist_ok=True)

            # plot
            fig_l, ax_l = plt.subplots(figsize=(width, height))
            fig_w, ax_w = plt.subplots(figsize=(width, height))

            m_l = Basemap(ax=ax_l, **kwds_basemap)
            m_w = Basemap(ax=ax_w, **kwds_basemap)

            # compute difference
            gl[var] = gl[f'ntcs_sat_2_{season}_all_worb_agrad_worhs_r_max_exp_alpha_q90'] - \
                gl[f'ntcs_sat_24_{season}_all_worb_agrad_worhs_r_max_exp_alpha_q90']

            # only show locations with at least "mincnt" bursts per bin
            gl['min_n_bursts'] = gl[[f'ntcs_sat_2_{season}_all_worb_agrad_worhs_r_max_n_bursts',
                                     f'ntcs_sat_24_{season}_all_worb_agrad_worhs_r_max_n_bursts']].min(axis=1)
            gl.loc[gl['min_n_bursts'] < mincnt, var] = np.nan

            if 'alpha' in var:

                # contourf
                levels_l = list(range(-35, 36))
                levels_w = list(range(-35, 36))

                # create matrix
                z = z0.copy()
                z[:, :] = np.nan
                z[gl['y'], gl['x']] = gl[var].values

                # land
                zl = z.copy()
                gllw = gl[gl['lw'] == 'W']
                zl[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
                # zl = np.ma.masked_where(zl == 999999, zl)

                # water
                zw = z.copy()
                gllw = gl[gl['lw'] == 'L']
                zw[gllw['y'], gllw['x']] = np.nan  # 999999  # np.nan
                # zw = np.ma.masked_where(zl == 999999, zl)

                # nans
                znan = z0.copy()
                znan[gl.loc[gl[var].isnull(), 'y'], gl.loc[gl[var].isnull(), 'x']] = 1
                znan = np.ma.masked_where(znan == 0, znan)

                # interpolate/smoothen
                # zw = scipy.ndimage.zoom(zw, 3)
                # zl = scipy.ndimage.zoom(zl, 3)

                # zw = gaussian_filter(zw, sigma=.1)
                # zl = gaussian_filter(zl, sigma=.1)

                # contourf (land)
                iml = m_l.contourf(
                    lons, lats[::-1], zl,
                    levels_l,
                    latlon=True,
                    ax=ax_l,
                    extend='both',
                    # vmin=-30, vmax=30,
                    cmap='PiYG',
                )

                # contourf (water)
                imw = m_w.contourf(
                    lons, lats[::-1], zw,
                    levels_w,
                    latlon=True,
                    ax=ax_w,
                    extend='both',
                    # vmin=-30, vmax=30,
                    cmap='RdBu_r',
                )

                # imshow
        #         z = np.roll(z, roll, axis=1)
        #         zl = np.roll(zl, roll, axis=1)
        #         zw = np.roll(zw, roll, axis=1)
        #         znan = np.roll(znan, roll, axis=1)

                # im = m.imshow(z, vmin=-28, vmax=28, cmap=piyg, interpolation='none')

                # imshow (land)
        #         iml = m_l.imshow(
        #             zl, vmin=-2*7, vmax=2*7, cmap='RdYlGn',
        #             interpolation='nearest',
        #             ax=ax_l,
        #         )

                # imshow (water)
        #         imw = m_w.imshow(
        #             zw, vmin=-6*7, vmax=6*7, cmap='coolwarm',
        #             interpolation='nearest',
        #             ax=ax_w,
        #         )

                # imshow (nan)
                # cmap = mpl.colors.ListedColormap(['cornflowerblue'])
                # imnan = m.imshow(znan, cmap=cmap, interpolation='none', ax=ax)

                # sst mean contour line
                z = sst_mean[season][::-1]
                cs = m_w.contour(lons_sst, lats_sst, z,
                                 # cmap='viridis',
                                 linewidths=2,
                                 colors=['red'],
                                 levels=[sst_mean[f'{season}_contourline_temp']],
                                 ax=ax_w,
                                 latlon=True)
                cl = ax_w.clabel(cs, fmt=lambda x: r'\textbf{{{:.1f} °C}}'.format(x),
                                 # fontsize=12,
                                 manual=sst_mean[f'{season}_clabel_loc'])

                [txt.set_backgroundcolor('whitesmoke') for txt in cl]
                [txt.set_bbox(
                    dict(facecolor='whitesmoke',
                         edgecolor='none', pad=1)) for txt in cl]

                # cl[0].set_weight('extra bold')

            # basemap stuf
            def basemap_stuff(m, ax, which):

                m.drawcoastlines(
                    ax=ax, zorder=10,
                    # linewidth=1,
                )

                parallels = [-23, 0, 23]  # np.arange(-50, 50, 5)  #
                m.drawparallels(
                    parallels,
                    dashes=(None,None),
                    linewidth=.5,
                    labels=[True, False, False, False],
                )
                meridians = np.arange(-180, 180, 60)  # np.arange(-180, 180, 5)  #
                m.drawmeridians(
                    meridians,
                    dashes=(None,None),
                    linewidth=.5,
                    labels=[False, False, True, False],
                )
                # for merid in merids:
                #     merids[merid][1][0].set_rotation(90)

                # boxes
                if 'pts_per_bin' not in var:
                    if which == 'water':
                        # northern atlantic neg alpha
                        lat_min, lat_max, lon_min, lon_max = 12, 18, -82, -76
                        poly = mpl.patches.Polygon(np.array(
                            [(lon_min, lat_min), (lon_max, lat_min),
                             (lon_max, lat_max), (lon_min, lat_max)]),
                            facecolor='none', edgecolor='cyan',
                            # linewidth=1.5,
                        )
                        ax.add_patch(poly)
                        # northern atlantic pos alpha
                        lat_min, lat_max, lon_min, lon_max = 16, 22, -46, -40
                        poly = mpl.patches.Polygon(np.array(
                            [(lon_min, lat_min), (lon_max, lat_min),
                             (lon_max, lat_max), (lon_min, lat_max)]),
                            facecolor='none', edgecolor='snow',
                            # linewidth=1.5,
                        )
                        ax.add_patch(poly)
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

            basemap_stuff(m_w, ax_w, 'water')
            basemap_stuff(m_l, ax_l, 'land')

            # region lines
            # season = var.split('_')[2]
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

            def colorbars(fig, ax, im):

                divider = make_axes_locatable(ax)

                cax = divider.append_axes("bottom", size="5%", pad=.08)
                _ = fig.colorbar(im, cax=cax,
                                 orientation='horizontal',
                                 # extend='both',
                                 )

                cax.set_xlabel(
                    r"$\alpha_{t=-2h} - \alpha_{t=-24h}$ [percentage points]"
                    # "\n"
                    # r"(ocean)",
                )

                # caxl = divider.append_axes("right", size="3%", pad=.45)
                # cbl = fig.colorbar(iml, cax=caxl,
                #                    orientation='vertical',
                #                    extend='both',
                #                    )
                # caxl.set_xlabel(
                #     r"$\alpha$ [\%$^{\circ}$C$^{{-1}}$]"
                #     "\n"
                #     r"(land)",
                # )

            colorbars(fig_w, ax_w, imw)
            colorbars(fig_l, ax_l, iml)

            # title
            # ax.set_title(var)

            # save figure
            # fig.tight_layout()
            def store_figure(fig, which):
                picfile = picfolder + subfolder + var + which

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

            store_figure(fig_l, '_land')
            store_figure(fig_w, '_water')


# def td_argmin_max(var_str='m', var_str_q90='_q90'):
#
#     levels = [-23, -19, -15, -11, -7, -3]
#
#     for season in seasons:
#         for wo_r_before in wo_r_befores:
#             for vari in varis:
#                 for which in ['_left', '_right']:
#
#                     # slope left/right q90 cols
#                     cols = []
#                     for twcol in twcols:
#                         col = cstr(twcol, season, hod, wo_r_before, vari) + \
#                             var_str + which + var_str_q90
#                         cols.append(col)
#
#                     # find min/max value (left/right)
#                     t = gl[cols].idxmin(axis=1).astype(str)
#                     t = t.str.split('_', expand=True)[1].astype(float)
#                     if var_str == 'm' or var_str == 'rvalue':
#                         if which == '_left':
#                             t = gl[cols].idxmin(axis=1).astype(str)
#                             t = t.str.split('_', expand=True)[1].astype(float)
#                         elif which == '_right':
#                             t = gl[cols].idxmax(axis=1).astype(str)
#                             t = t.str.split('_', expand=True)[1].astype(float)
#
#                     # kwds_use = kwds_pcolormesh.copy()
#                     kwds_use = kwds_contourf.copy()
#                     kwds_use.update({'levels': levels})
#
#                     # plot
#                     fig, ax = plt.subplots(figsize=(19.20, 10.80))
#                     m = Basemap(ax=ax, **kwds_basemap)
#
#                     # create matrix
#                     z = z0.copy()
#                     z[:, :] = np.nan
#                     z[gl['y'], gl['x']] = t.values
#                     z = z[::-1]
#                     # z = np.roll(z, 60, axis=1)
#
#                     # imshow
#                     # im = m.imshow(z)
#
#                     # pcolormesh
#                     # im = m.pcolormesh(lons, lats, z, **kwds_use)
#
#                     # contourf
#                     im = m.contourf(lons, lats, z, **kwds_use)
#
#                     # map stuff
#                     m.drawcoastlines(linewidth=1.5, ax=ax, zorder=10)
#
#                     parallels = [-23, 0, 23]  # np.arange(-50, 50, 5)
#                     meridians = np.arange(-180, 180, 60)  # (-180, 180, 5)
#                     m.drawparallels(
#                         parallels, linewidth=.6,
#                         labels=[True, False, False, False],
#                         dashes=[1, 0],)
#                     m.drawmeridians(
#                         meridians, linewidth=.6,
#                         labels=[False, False, True, False],
#                         dashes=[1, 0],)
#
#                     # region lines
#                     if season == 'JJA':
#                         m.drawparallels(
#                             [-16, 36], linewidth=3,
#                             labels=[False, False, False, False],
#                             dashes=[1, 0],
#                             zorder=20)
#                     if season == 'DJF':
#                         m.drawparallels(
#                             [-24, 21], linewidth=3,
#                             labels=[False, False, False, False],
#                             dashes=[1, 0],
#                             zorder=20)
#
#                     # colorbar
#                     divider = make_axes_locatable(ax)
#                     cax = divider.append_axes("bottom", size="5%", pad=.05)
#                     fig.colorbar(im, cax=cax, orientation='horizontal')
#
#                     # fake colorbars
#                     # fax1 = divider.append_axes("bottom", size="2%", pad=.3)
#                     # fax2 = divider.append_axes("bottom", size="2%", pad=.4)
#                     # fax1.axis('off')
#                     # fax2.axis('off')
#
#                     # title
#                     name = cstr(
#                         'argmin_max', season, hod, wo_r_before, vari
#                     ) + var_str + which + var_str_q90
#                     ax.set_title(name)
#
#                     # save figure
#                     fig.tight_layout()
#                     fig.savefig(picfolder + name + '.png')
#                     print(picfolder + name + '.png')
#                     plt.close(fig)


per_td()
# t_24_2_difference()
# tw_argmin_max(var_str='m', var_str_q90='_q90')
# tw_argmin_max(var_str='pvalue', var_str_q90='_q90')
# tw_argmin_max(var_str='rvalue', var_str_q90='_q90')
