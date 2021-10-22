import os

import matplotlib as mpl
mpl.use('Agg')

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
parser.add_argument('which',
                    help='which temperature type to use',
                    choices=['SAT', 'SST', 'SDT', 'RH'],
                    type=str)
parser.add_argument('statistic',
                    help='which statistic to compute',
                    choices=['mean', 'var'],
                    type=str)
parser.add_argument('-s', '--season',
                    help='subset to season',
                    choices=['JJA', 'DJF', 'MJJAS', 'NDJFM', 'JASO', 'DJFMA'],
                    type=str)
parser.add_argument('-ts', '--timesplit',
                    help='split data into invervals (1998-2008), (2008, 2019)',
                    choices=['fh', 'sh'],
                    type=str)
args = parser.parse_args()

# parameters
which = args.which
season = args.season
time_split = args.timesplit
statistic = args.statistic
store_nat_comm_data = True

# plot parameters
if statistic == 'mean':
    if which == 'SST' and args.season == 'JASO':
        # levels = np.arange(18, 30.5, .5)
        levels = np.arange(26, 30.5, .25)
    elif which == 'SST' and args.season == 'DJFMA':
        levels = np.arange(27, 30.5, .25)
    elif which == 'SAT':
        levels = np.arange(18, 28.5, .5)  # [18, 20, 22, 24, 26, 28, 30]
    elif which == 'SDT':
        levels = np.arange(18, 26.5, .5)  # [18, 20, 22, 24, 26, 28, 30]
    elif which == 'RH':
        levels = np.arange(60, 92.5, 2.5)  # [18, 20, 22, 24, 26, 28, 30]
elif statistic == 'var':
    levels = np.arange(0, 1.75, .15)

# filesystem folders
storefolder = os.getcwd() + '/data_processing/'
picfolder = os.getcwd() + '/map_plots/SXT/'
os.makedirs(picfolder, exist_ok=True)

if statistic == 'var':
    stastr = '_std'
elif statistic == 'mean':
    stastr = '_mean'
if season is not None:
    sstr = '_{}'.format(season)
else:
    sstr = ''
if time_split is not None:
    tsstr = '_{}'.format(time_split)
else:
    tsstr = ''

# load data
sxt_statistic = np.load(
    storefolder + f'{which}{sstr}{tsstr}_{statistic}.pkl',
    allow_pickle=True)
if statistic == 'var':
    sxt_statistic = np.sqrt(sxt_statistic)

# load SST data for contour line
sst_mean = {}
sst_mean[args.season] = np.load(storefolder + f'SST_{season}_mean.pkl',
                                allow_pickle=True)

sst_mean['JASO_contourline_temp'] = 28
sst_mean['JASO_clabel_loc'] = [(-150+360, 15)]
sst_mean['DJFMA_contourline_temp'] = 28.5
sst_mean['DJFMA_clabel_loc'] = [(-170+360, 0)]

lons_sst, lats_sst = np.meshgrid(np.arange(-179.875, 180.125, .25),
                                 np.arange(89.875, -90.125, -.25))

# plot params
# lon = np.arange(-179.875, 180.125, .25)
# lat = np.arange(49.875, -50.125, -.25)
lons, lats = np.meshgrid(np.arange(-179.875, 180.125, .25),
                         np.arange(89.875, -90.125, -.25))

# plot regions
kwds_basemap = {'projection': 'cyl',
                # 'lon_0': 0,
                # 'lat_0': 0,
                'llcrnrlon': 60,
                'urcrnrlon': 420,
                'llcrnrlat': -50,
                'urcrnrlat': 50,
                'resolution': 'c'}


def sxt(ax):

    m = Basemap(ax=ax, **kwds_basemap)

    # contourf
    z = sxt_statistic[::-1]

    # store figure data for nat.comm.
    if store_nat_comm_data:
        data = pd.DataFrame(data={
            'latitude [°N]': lats.flatten(),
            'longitude [°E]': lons.flatten(),
            'average sea surface temperature [°C]': z.flatten()})
        data = data.loc[(data['latitude [°N]'] <= 50) &
                        (data['latitude [°N]'] >= -50)]
        os.makedirs(storefolder + 'Nat_Comm_Fig_data/', exist_ok=True)
        data.to_csv(storefolder + 'Nat_Comm_Fig_data/fig1b.csv', index=False)

    im = m.contourf(lons, lats, z,
                    cmap='coolwarm' if statistic == 'mean' else 'cividis',
                    levels=levels,
                    extend='both' if statistic == 'mean' else 'max',
                    # vmin=vmin, vmax=vmax,
                    latlon=True)

    # contour
    if statistic == 'mean':
        if season == 'JASO':
            temp = 28.
            clabel_loc = [(210, 15)]
        elif season == 'DJFMA':
            temp = 28.5
            clabel_loc = [(150, 10)]

        cs = m.contour(lons, lats, z,
                       # cmap='viridis',
                       linewidths=2,
                       colors=['black'],
                       levels=[temp],
                       ax=ax,
                       latlon=True)

    # elif statistic == 'var':
    #     if season == 'JASO':
    #         var = .6
    #         clabel_loc = [(190, -15)]
    #     elif season == 'DJFMA':
    #         var = .7
    #         clabel_loc = [(150, 15)]
    #
    #     cs = m.contour(lons, lats, z,
    #                    # cmap='viridis',
    #                    linewidths=2,
    #                    colors=['orchid'],
    #                    levels=[var],
    #                    # levels=[.6, .7, .8],
    #                    # levels=10,
    #                    ax=ax,
    #                    latlon=True)

    # ax.clabel(cs)
        csl = ax.clabel(cs, fmt=lambda x: r'\textbf{{ {:.1f} °C }}'.format(x),
                        inline_spacing=12,
                        # fontsize=12,
                        manual=clabel_loc)
        # [txt.set_backgroundcolor('white') for txt in csl]
        # [txt.set_bbox(
        #     dict(facecolor='white', edgecolor='none', pad=0)) for txt in csl]

    # sst mean contour line for var plot
    if statistic == 'var':
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
    fig.colorbar(
        im,
        cax=cax,
        extend='both' if statistic == 'mean' else 'max',
        orientation='horizontal'
    )
    if statistic == 'mean':
        cxlabel = (r"average sea surface temperature "
                   "[$^{\circ}$C]")
    elif statistic == 'var':
        cxlabel = (r"standard deviation of daily sea surface temperatures "
                   "[$^{\circ}$C]")
    cax.set_xlabel(cxlabel)

    # basemap stuff
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

    # title
    # ax.set_title(f'{stastr} {which}{sstr}{tsstr}')


# plot
fig, ax = plt.subplots(figsize=(width, height))

# fill axes
lons, lats, z = sxt(ax)

picfile = picfolder + f'{which}{sstr}{tsstr}{stastr}_of_all_time'
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
