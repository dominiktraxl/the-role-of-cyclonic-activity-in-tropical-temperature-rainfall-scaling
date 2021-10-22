import os

import pandas as pd
import deepgraph as dg

# parameters
r = .1
p = 0
seasons = ['JJA', 'DJF']
qs = [.9, .95, .99]
qstrs = ['{}'.format(int(q*100)) for q in qs]

# load data
storefolder = os.getcwd() + '/'

print('loading v_r{}_p{}.feather ..'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')

print('loading v_r{}_p{}_seasons.feather'.format(r, p))
v_season = pd.read_feather(
    storefolder + 'v_r{}_p{}_seasons.feather'.format(r, p))

# compute
for season in seasons:

    print(season)

    sstr = '_{}'.format(season)
    vt = v[v_season[season]]

    # vt = dd.from_pandas(vt, npartitions=100)
    # cp = g.load_np('cp', r=r, p=p)

    # initiate DataGraph
    g = dg.DeepGraph(vt)

    # count / accumulate node-component participation coefficients
    # def vol_sum(group):
    #     return cp.loc[group.unique(), 'vol_sum'].sum()

    # def area_sum(group):
    #     return cp.loc[group.unique(), 'area'].sum()

    feature_funcs = {
        # 'cp': [vol_sum, area_sum],
        'vol': ['sum'],
        'r': ['min', 'sum', 'mean']  # , 'median'],
    }

    # create gl
    print('partitioning vt ..')
    gv = vt.groupby('l')
    # gl, gv = g.partition_nodes('l', feature_funcs, return_gv=True)

    gl = gv.size().to_frame().rename(columns={0: 'n_nodes'})
    gl['vol_sum'] = gv['vol'].sum()
    gl['r_min'] = gv['r'].min()
    gl['r_mean'] = gv['r'].mean()

    # additional columns
    # gl['n_cps'] = gv.cp.nunique()
    # gl['n_nodes_vs_n_cps'] = gl['n_nodes'] / gl['n_cps']

    # quantiles
    for q, qstr in zip(qs, qstrs):
        print(qstr)
        gl['r_q{}'.format(qstr)] = gv['r'].quantile(q)

    # coordinates
    print('adding coordinates ..')
    gl[['x', 'y', 'lat', 'lon']] = gv[['x', 'y', 'lat', 'lon']].first()

    # acutally compute
    # print('start computation ..')
    # pbar = ProgressBar()
    # pbar.register()
    # gl = gl.compute()

    # reindex
    if gl.shape[0] != 1440*400:
        print('gl.shape <= 1440*400 -> reindexing ..')
        gl = gl.reindex(index=range(1440*400))
        gl.loc[gl['n_nodes'].isnull(), 'n_nodes'] = 0
        gl['n_nodes'] = gl['n_nodes'].astype(int)

    # reset index for feather format
    print('reset index ..')
    gl.reset_index(inplace=True)

    # store
    print('storing gl_r{}_p{}{}.feather ..'.format(r, p, sstr))
    gl.to_feather(storefolder + 'gl_r{}_p{}{}.feather'.format(r, p, sstr))
