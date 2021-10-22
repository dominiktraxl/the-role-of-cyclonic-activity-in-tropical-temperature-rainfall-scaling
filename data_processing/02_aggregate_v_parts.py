import os
import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("-a", "--all-events",
                    help="if set, do not filter out wet-times",
                    action='store_true')
args = parser.parse_args()

# parameters
r = .1

# filesystem folders
storefolder = os.getcwd() + '/'
v_parts_folder = 'v_parts/'

# dataset-specific files/folders
r_str = '' if args.all_events else '_r{}'.format(r)

# load v_parts
feathers = os.listdir(storefolder + v_parts_folder)
feathers.sort()

# load yearly files
vs = []
for f in feathers:
    print('loading {} ..'.format(f))
    v = pd.read_feather(storefolder + v_parts_folder + '{}'.format(f))
    if not args.all_events:
        v = v[v['r'] >= r]
    vs.append(v)

print('concat v ..')
v = pd.concat(vs, axis=0)

# set range index
v.index = range(len(v))

# store
print('storing v ..')
v.to_feather(storefolder + 'v{}_p0.feather'.format(r_str))


# store for the merged vts
# store = pd.HDFStore(storefolder + v_table, mode='w')
# for picklefile in picklefiles:
#     print('loading {}...'.format(picklefile))
#     vt = pd.read_pickle(
#         storefolder + v_parts_folder + '{}'.format(picklefile))
#     if not args.all_events:
#         vt = vt[vt.r >= r]
#     print('copy to hdf..')
#     store.append('v', vt, format='t', data_columns=True, index=False)
#
# store.close()
#
# if not args.all_events:
#
#     # sort by geo_label
#     print('loading entire v for sorting by locations...')
#     v = pd.read_hdf(storefolder + v_table)
#     print('sorting v by geographical locations...')
#     v.sort_values('l', inplace=True, kind='heapsort')
#
#     # store sorted dataframe
#     print('storing sorted v...')
#     store = pd.HDFStore(
#         storefolder + v_table[:-3] + '_l_sorted.h5', mode='w')
#     store.append('v', v, format='t', data_columns=True, index=False)
#     store.close()
#
#     # compress
#     print('compressing v...')
#     c = 7
#     cmd = ['ptrepack',
#            '--overwrite',
#            '--chunkshape=auto',
#            '--complib=blosc',
#            '--complevel={}'.format(c),
#            storefolder + v_table[:-3] + '_l_sorted.h5',
#            storefolder + v_table[:-3] + '_l_sortedc{}.h5'.format(c)]
#     subprocess.Popen(cmd).wait()
#
#     # create index
#     print('creating table index...')
#     store = pd.HDFStore(
#         storefolder + v_table[:-3] + '_l_sortedc{}.h5'.format(c))
#     store.create_table_index('v', columns=['l'], kind='full')
#     store.close()
