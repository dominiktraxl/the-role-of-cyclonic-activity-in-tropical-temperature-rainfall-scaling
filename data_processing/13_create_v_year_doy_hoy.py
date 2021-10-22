import os

import pandas as pd

# parameters
r = .1
p = 0

# load data
storefolder = os.getcwd() + '/'

print('loading v_r{}_p{}.feather ..'.format(r, p))
v = pd.read_feather(storefolder + f'v_r{r}_p{p}.feather')

print('appending year ..')
v['year'] = v['dtime'].dt.year

print('append doy ..')
v['doy'] = v['dtime'].dt.dayofyear - 1

print('append hoy ..')
v['hoy'] = 0
for year in range(1998, 2019):
    print('adding hoy for year {} ..'.format(year))
    dts = pd.date_range(
        '01-01-{}'.format(year), '01-01-{}'.format(year + 1), freq='H'
    )
    dtdic = {date: i for i, date in enumerate(dts)}
    vyear = v['year'] == year
    v.loc[vyear, 'hoy'] = v.loc[vyear, 'dtime'].apply(lambda x: dtdic[x])

# select columns
v = v[['year', 'doy', 'hoy']]

# store
print('storing v_r{}_p{}_year_doy_hoy.feather ..'.format(r, p))
v.to_feather(storefolder + 'v_r{}_p{}_year_doy_hoy.feather'.format(r, p))
