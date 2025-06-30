####################################################################################################
#
#  Copyright (C) 2022, 2023
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import numpy as np
import zarr
import xarray as xr
import code

import cartopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.linewidth'] = 0.1
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


model_id = '8e9sltzr' 
field = 'total_precip'

# 4dropnio - OG Vorticity 
# Wandb run: atmorep-wvwb4fxy-17555290 -   

store = zarr.ZipStore( f'results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
ds = zarr.group( store=store)

# create empty canvas where local patches can be filled in
# use i=0 as template; structure is regular
i = 0
ds_o = xr.Dataset( coords={ 'ml' : ds[ f'{field}/sample={i:05d}/ml' ][:],
                            'datetime': ds[ f'{field}/sample={i:05d}/datetime' ][:], 
                            'lat' : np.linspace( -90., 90., num=180*4+1, endpoint=True), 
                            'lon' : np.linspace( 0., 360., num=360*4, endpoint=False) } )
nlevels = ds[ f'{field}/sample={i:05d}/ml' ].shape[0]
ds_o['vo'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( nlevels, 3, 721, 1440)))

# fill in local patches
for i_str in ds[ f'{field}']:
  ds_o['vo'].loc[ dict( datetime=ds[ f'{field}/{i_str}/datetime' ][:],
        lat=ds[ f'{field}/{i_str}/lat' ][:],
        lon=ds[ f'{field}/{i_str}/lon' ][:]) ] = ds[ f'vorticity/{i_str}/data' ]

# plot and save the three time steps that form a token
cmap = matplotlib.colormaps.get_cmap('RdBu_r')
vmin, vmax = ds_o['vo'].values[0].min(), ds_o['vo'].values[0].max()
for k in range( 3) :
  fig = plt.figure( figsize=(10,5), dpi=300)
  ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
  ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
  ax.set_global()
  date = ds_o['datetime'].values[k].astype('datetime64[m]')
  ax.set_title(f'{field} : {date}')
  im = ax.imshow( np.flip(ds_o['vo'].values[0,k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
                  transform=cartopy.crs.PlateCarree( central_longitude=180.))
  axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
  fig.colorbar( im, cax=axins, orientation="horizontal")
  plt.savefig( f'/results/id{model_id}/example_{k:03d}.png')
  plt.close()

  '''
source /work/ab1412/atmorep/pyenv/bin/activate
module load python3
python /work/ab1412/atmorep/plotting/plot_forecast.py

'''
