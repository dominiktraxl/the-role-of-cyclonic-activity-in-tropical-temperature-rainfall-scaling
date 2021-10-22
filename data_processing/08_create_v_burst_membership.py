import os
from multiprocessing import Pool
import subprocess

import numpy as np
import pandas as pd
import deepgraph as dg

# parameters
r = .1
p = 0
n_proc = 500

# filesystem folders
storefolder = os.getcwd() + '/'
v_table = 'v_r{}_p{}_dtime_l_r_seasons_l_sortedc7.h5'.format(r, p)

# index array
pos_array = np.array(np.linspace(0, 1440*400, n_proc), dtype=int)


def get_data(i):

    print('starting {}/{}'.format(i+1, n_proc-1))

    l1 = pos_array[i]
    l2 = pos_array[i+1]

    # store
    print('select from store ({}) ..'.format(i+1))
    vstore = pd.HDFStore(storefolder + v_table, mode='r')
    v = vstore.select('v', where='l >= {} & l < {}'.format(l1, l2),
                      columns=['l', 'dtime'])
    vstore.close()
    print('done selecting ({}) ..'.format(i+1))

    gv = v.groupby('l')
    vt_parts = []
    for name, vt in gv:
        print('-' * 80)
        print(name)
        # vt.sort_values('time', inplace=True)
        g = dg.DeepGraph(vt)
        g.create_edges_ft(ft_feature=('dtime', 3, 'h'))
        g.append_cp(col_name='cp_burst')
        cp = g.partition_nodes('cp_burst')
        vt = pd.merge(
            g.v, cp, left_on='cp_burst', right_index=True, how='left')
        vt.rename(columns={'n_nodes': 'k_burst'}, inplace=True)
        vt['burst_position'] = vt.groupby('cp_burst').cumcount() + 1
        vt = vt[['l', 'cp_burst', 'k_burst', 'burst_position']]
        vt_parts.append(vt)

    v_burst = pd.concat(vt_parts, axis=0, sort=False)

    return v_burst


if __name__ == '__main__':

    indices = np.arange(n_proc - 1)
    pool = Pool()

    v_bursts = pool.map(get_data, indices)
    v_bursts = pd.concat(v_bursts, axis=0, sort=False)

    # reset index
    # v_bursts.reset_index(inplace=True)

    # store as feather
    # print('storing v ..')
    # v_bursts.to_feather(
    #     storefolder + 'v_r{}_p{}_bursts.feather'.format(r, p))

    # store sorted dataframe
    print('storing v_r{}_p{}_bursts_l_sorted.h5 ..'.format(r, p))
    store = pd.HDFStore(
        storefolder + 'v_r{}_p{}_bursts_l_sorted.h5'.format(r, p),
        mode='w')
    store.append('v', v_bursts, format='t', data_columns=True, index=False)
    store.close()

    # compress
    print('compressing v...')
    c = 7
    cmd = ['ptrepack',
           '--overwrite',
           '--chunkshape=auto',
           '--complib=blosc',
           '--complevel={}'.format(c),
           storefolder + 'v_r{}_p{}_bursts_l_sorted.h5'.format(r, p),
           storefolder + 'v_r{}_p{}_bursts_l_sortedc{}.h5'.format(
               r, p, c)]
    subprocess.Popen(cmd).wait()

    # create index
    print('creating table index...')
    store = pd.HDFStore(
        storefolder + 'v_r{}_p{}_bursts_l_sortedc{}.h5'.format(
            r, p, c))
    store.create_table_index('v', columns=['l'], kind='full')
    store.close()
