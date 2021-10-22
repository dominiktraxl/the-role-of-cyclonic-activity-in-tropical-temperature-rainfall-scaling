import matplotlib as mpl
mpl.use('Agg')

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

# variables to plot
twcols = ['s_-6_-2']  # ['s_{}_-2'.format(tf) for tf in range(-24, -2, 6)]
# twcols = ['s_-24_-12']
hod = 'all'
varis = ['r_max']  # ['r_mean', 'r_max']
seasons = ['JASO']  # , 'DJFMA']  # ['all', 'DJF', 'JJA']
wo_r_befores = [True]  # [True, False]

keys = [
    'mean',
    # 'median',
    # 'left',
    # 'n_bursts',
]

# filesystem folders
storefolder = os.getcwd() + '/data_processing/'

if cg_N is not None:
    picfolder = os.getcwd() + \
        f'/map_plots/gl_cg_{cg_N}/r{r}_p{p}_grad_stats{tcfstr}/'
    glxy = pd.read_feather(
        storefolder + f'gl_r{r}_p{p}_cg_{cg_N}.feather')
    glfit = pd.read_pickle(
        storefolder + f'gl_r{r}_p{p}_cg_{cg_N}_grad_stats{tcfstr}.pickle')
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
        f'/map_plots/gl/r{r}_p{p}_grad_stats{tcfstr}/'
    glxy = pd.read_feather(storefolder + f'gl_r{r}_p{p}_JJA.feather')
    glfit = pd.read_pickle(
        storefolder + f'gl_r{r}_p{p}_grad_stats{tcfstr}.pickle')
    lon = np.arange(-179.875, 180.125, .25)
    lat = np.arange(49.875, -50.125, -.25)
    z0 = np.zeros((400, 1440), dtype=np.float64)

os.makedirs(picfolder, exist_ok=True)
gl = pd.merge(glfit, glxy[['x', 'y', 'lat', 'lon']], how='left',
              left_index=True, right_index=True, sort=False)
lons, lats = np.meshgrid(lon, lat)

# store for nature communications data supplement
data = gl[['lat', 'lon', 's_-6_-2_JASO_all_worb_r_max_mean']]
data.rename(columns={
    'lat': 'latitude [°N]',
    'lon': 'longitude [°E]',
    's_-6_-2_JASO_all_worb_r_max_mean': 'average temporal pre-rainfall temperature gradient [°C/h]'},
    inplace=True
)
os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
data.to_csv(storefolder + 'Nat_Comm_Fig_data/fig3a.csv', index=False)

# consider only water pixels
glregions = pd.read_pickle(storefolder + 'gl_regions.pickle')
gl.loc[:, 'lw'] = glregions['lw']
gl = gl.loc[gl['lw'] == 'W']

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

print('loaded all data..')


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


# def per_tw():

# variables to plot
data = {}
for season in seasons:
    for vari in varis:
        for wrb in wo_r_befores:
            for twcol in twcols:

                subset = cstr(
                    col=twcol,
                    season=season,
                    hod='all',
                    wo_r_before=wrb,
                    var=vari,
                )

                for key in keys:
                    data.update({subset + key: {}})

for var, kwds_update in data.items():

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
    z[gl['y'], gl['x']] = gl[var].values
    z = z[::-1]
    # z = np.roll(z, 480, axis=1)

    # imshow
    if var.endswith('mean') or var.endswith('median'):
        cmap = 'PiYG'
        q1 = abs(gl[var].quantile(.01))
        q2 = abs(gl[var].quantile(.99))
        qmax = max(q1, q2)
        vmin = -qmax
        vmax = qmax
        # manual selection
        vmin = -0.03
        vmax = 0.03
    if var.endswith('left'):
        cmap = 'PiYG'
        vmin = 0
        vmax = 100
    if var.endswith('n_bursts'):
        cmap = 'viridis'
        vmin = None
        vmax = gl[var].quantile(.99)

    # im = m.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax)

    # pcolormesh
    # im = m.pcolormesh(lons, lats, z, **kwds_use)

    # contourf
    levels = [-0.025, -0.02, -0.015, -0.01, -0.005, 0]
    im = m.contourf(
        lons, lats, z,
        levels=levels,
        extend='both',
        cmap='Greys_r',
        latlon=True,
        ax=ax)

    # contour line
    if season == 'JASO':
        grad = -0.015
        clabel_loc = [(190, 8)]
    elif season == 'DJFMA':
        grad = -0.015
        clabel_loc = [(190, 8)]

    # cs = m.contour(lons, lats, z,
    #                # cmap='viridis',
    #                linewidths=2,
    #                colors=['black'],
    #                levels=[grad],
    #                ax=ax,
    #                latlon=True)

    # scatter
    # pc = m.scatter(
    #     gl['lon'], gl['lat'], s=1, c=gl[var], latlon=True,
    #     vmin=gl[var].quantile(.01), vmax=gl[var].quantile(.99),
    #     marker='s', edgecolors='none')

    # map stuff
    m.drawcoastlines(ax=ax, zorder=10,
                     # linewidth=1.5,
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

    # region lines
    season = var.split('_')[3]
    if season == 'JJA':
        m.drawparallels(
            [-16, 36], linewidth=3, labels=[False, False, False, False],
            dashes=[1, 0],
            zorder=20)
    if season == 'DJF':
        m.drawparallels(
            [-24, 21], linewidth=3, labels=[False, False, False, False],
            dashes=[1, 0],
            zorder=20)

    # sst mean contour line
    z = sst_mean[season][::-1]
    cs = m.contour(lons_sst, lats_sst, z,
                   # cmap='viridis',
                   linewidths=2,
                   colors=['red'],
                   levels=[sst_mean[f'{season}_contourline_temp']],
                   ax=ax,
                   latlon=True)
    cl = ax.clabel(cs, fmt=lambda x: r'\textbf{{{:.1f} °C}}'.format(x),
                   # fontsize=12,
                   manual=sst_mean[f'{season}_clabel_loc'])

    [txt.set_backgroundcolor('whitesmoke') for txt in cl]
    [txt.set_bbox(
        dict(facecolor='whitesmoke', edgecolor='none', pad=1)) for txt in cl]

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=.08)
    cb = fig.colorbar(im, cax=cax,
                      orientation='horizontal',
                      )
    cax.set_xlabel(
        "average pre-rainfall temperature gradient "
        # "$T^r_g$"
        " [$^\circ$Ch$^{-1}$]"
    )

    # fake colorbars
    # fax1 = divider.append_axes("bottom", size="2%", pad=.3)
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
