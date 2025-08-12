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

# 0: Wandb run: atmorep-to9m4vuf-18687042 re-trained t2m model hjbmsjft
# 0: Wandb run: atmorep-e27ydobl-18688645 0: Loaded model id = 58ipo6bs. v3
# 0: Wandb run: atmorep-f1br76ag-18896986 0: Loaded model id = 0rmiio09. v4
# 0: Wandb run: atmorep--18947992 0: Loaded model id = qw047nnt. v5
# 0: Wandb run: atmorep-c36hghai-19060281 0: Loaded model id = ugqn2s9m. v7


'''
source /work/ab1412/atmorep/pyenv/bin/activate
python /work/ab1412/atmorep/plotting/plot_forecast.py

'''


model_id = 'c36hghai' 
model_run = '19060281'
loaded_model = 'ugqn2s9m'
field = 't2m'

# 4dropnio - OG Vorticity 

store = zarr.ZipStore( f'results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
ds = zarr.group( store=store)

print(f"Loaded Zarr store from {store}")

# create empty canvas where local patches can be filled in
# use i=0 as template; structure is regular
i = 0
ds_o = xr.Dataset( coords={ 'ml' : ds[ f'{field}/sample={i:05d}/ml' ][:],
                            'datetime': ds[ f'{field}/sample={i:05d}/datetime' ][:], 
                            'lat' : np.linspace( -90., 90., num=180*4+1, endpoint=True), 
                            'lon' : np.linspace( 0., 360., num=360*4, endpoint=False) } )
nlevels = ds[ f'{field}/sample={i:05d}/ml' ].shape[0]
ds_o['vo'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( nlevels, 6, 721, 1440)))

print(f"Created output dataset with shape: {ds_o['vo'].shape}")

# fill in local patches
for i_str in ds[ f'{field}']:
  ds_o['vo'].loc[ dict( datetime=ds[ f'{field}/{i_str}/datetime' ][:],
        lat=ds[ f'{field}/{i_str}/lat' ][:],
        lon=ds[ f'{field}/{i_str}/lon' ][:]) ] = ds[ f't2m/{i_str}/data' ]

# plot and save the three time steps that form a token
cmap = matplotlib.colormaps.get_cmap('RdBu_r')
vmin, vmax = ds_o['vo'].values[0].min(), ds_o['vo'].values[0].max()
for k in range( 3) :
  fig = plt.figure( figsize=(10,5), dpi=300)
  ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
  ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
  ax.set_global()
  date = ds_o['datetime'].values[k].astype('datetime64[m]')
  ax.set_title(f'{field} : {date}\nModel run : {model_id} {model_run}\nLoaded model : {loaded_model}', fontsize=8)
  im = ax.imshow( np.flip(ds_o['vo'].values[0,k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
                  transform=cartopy.crs.PlateCarree( central_longitude=180.))
  axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
  fig.colorbar( im, cax=axins, orientation="horizontal")
  plt.savefig( f"/work/ab1412/atmorep/plotting/forecast_plot_id_{model_id}_example_{k:03d}.png")
  plt.close()

