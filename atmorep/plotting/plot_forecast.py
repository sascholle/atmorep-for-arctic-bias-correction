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
import re

import cartopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.linewidth'] = 0.1
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 0: Wandb run: atmorep-5k7qf095-19238962 0: Loaded model id = wc5e2i3t. original 6 field model

# 0: Wandb run: atmorep-to9m4vuf-18687042 re-trained t2m model hjbmsjft
# 0: Wandb run: atmorep-e27ydobl-18688645 0: Loaded model id = 58ipo6bs. v3
# 0: Wandb run: atmorep-f1br76ag-18896986 0: Loaded model id = 0rmiio09. v4
# 0: Wandb run: atmorep--18947992 0: Loaded model id = qw047nnt. v5
# 0: Wandb run: atmorep-c36hghai-19060281 0: Loaded model id = ugqn2s9m. v7
# 0: Wandb run: atmorep-apadxke2-19093350  0: Loaded model id = zxipahjj. corrected t2m model

# Round of comparing temp 
# 0: Wandb run: atmorep-hha4bqe6-19513219 0: Loaded model id = iuw3ce3v. Jan
# 0: Wandb run: atmorep-9ah3u47h-19513414 0: Loaded model id = wc5e2i3t. Jan
# 0: Wandb run: atmorep-mh62gs52-19513494 0: Loaded model id = j2l0sz9j. Jan
# 0: Wandb run: atmorep-zwxjwne7-19513570 0: Loaded model id = wc5e2i3t. April
# 0: Wandb run: atmorep-cyv829p3-19513573 0: Loaded model id = wc5e2i3t. April again
# 0: Wandb run: atmorep-cztlfxsg-19513838 0: Loaded model id = wc5e2i3t. May
# 0: Wandb run: atmorep-97fv1tzd-19513857 0: Loaded model id = wc5e2i3t. May again
# 0: Wandb run: atmorep-ru0l87n7-19513986 0: Loaded model id = wc5e2i3t. Sept

# 0: Wandb run: atmorep-jbgafymp-19514003  0: Loaded model id = iuw3ce3v. Sept
# 0: Wandb run: atmorep-vqt0cahy-19514007  0: Loaded model id = iuw3ce3v. May
# 0: Wandb run: atmorep-j5putaeg-19514008 0: Loaded model id = iuw3ce3v. Sept again
# 0: Wandb run: atmorep-yip9b2fw-19514049 0: Loaded model id = iuw3ce3v. Jan
# 0: Wandb run: atmorep-igdw4ghm-19514068 0: Loaded model id = iuw3ce3v. Sept 26


'''
source /work/ab1412/atmorep/pyenv/bin/activate
python /work/ab1412/atmorep/plotting/plot_forecast.py

'''

## CONFIG 

field = 'corrected_t2m'  # 'temperature' 'specific_humidity' 'velocity_u' 'velocity_v' 'velocity_z' 'vorticity' 'divergence' 'total_precipitation'
level = 4 # index * ml int64 40B 96 105 114 123 137

# Extract model info from commented Wandb lines
wandb_lines = [
  # "# 0: Wandb run: atmorep-apadxke2-19093350  0: Loaded model id = zxipahjj. corrected t2m model"
   #"# 0: Wandb run: atmorep-c36hghai-19060281 0: Loaded model id = ugqn2s9m. v7"
   #"# 0: Wandb run: atmorep-igdw4ghm-19514068 0: Loaded model id = iuw3ce3v. Sept 26"
   #'# 0: Wandb run: atmorep-to9m4vuf-18687042 0: Loaded model id = hjbmsjft'
   "0: Wandb run: atmorep-hha4bqe6-19513219 0: Loaded model id = iuw3ce3v. Jan"
   #"0: Wandb run: atmorep-9ah3u47h-19513414 0: Loaded model id = wc5e2i3t. Jan"

]

model_infos = []
for line in wandb_lines:
    match = re.search(r'atmorep-([a-z0-9]+)-(\d+).*Loaded model id = ([a-z0-9]+)', line)
    if match:
        model_id = match.group(1)
        model_run = match.group(2)
        loaded_model = match.group(3)
        model_infos.append({'model_id': model_id, 'model_run': model_run, 'loaded_model': loaded_model})

print("Extracted model info from comments:")
for info in model_infos:
    print(info)


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
print(f"Number of model levels: {nlevels}")
ds_o['vo'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( nlevels, 6, 721, 1440)))

print(f"Created output dataset with shape: {ds_o['vo'].shape}")
#print model level key values?
print(ds_o['vo'].coords)

# fill in local patches
for i_str in ds[ f'{field}']:
  ds_o['vo'].loc[ dict( datetime=ds[ f'{field}/{i_str}/datetime' ][:],
        lat=ds[ f'{field}/{i_str}/lat' ][:],
        lon=ds[ f'{field}/{i_str}/lon' ][:]) ] = ds[ f'temperature/{i_str}/data' ]

# plot and save the three time steps that form a token
vmin, vmax = 220, 300  # Set min and max for colorbar
cmap = matplotlib.colormaps.get_cmap('RdBu_r')
#vmin, vmax = ds_o['vo'].values[0].min(), ds_o['vo'].values[0].max()
for k in range( 1) :
  # fig = plt.figure( figsize=(10,5), dpi=300)
  # ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
  # ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
  # ax.set_global()
  # date = ds_o['datetime'].values[k].astype('datetime64[m]')
  # ax.set_title(f'{field} : {date}\nModel run : {model_id} {model_run}\nLoaded model : {loaded_model}', fontsize=8)
  # im = ax.imshow( np.flip(ds_o['vo'].values[level,k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
  #                 transform=cartopy.crs.PlateCarree( central_longitude=180.))
  # axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
  # fig.colorbar( im, cax=axins, orientation="horizontal")
  #plt.savefig( f"/work/ab1412/atmorep/plotting/forecast_{field}_level_{0}_plot_id_{model_id}_date_{date}_level_{level}.png")
  #plt.close()

  #print(f"plot saved as /work/ab1412/atmorep/plotting/forecast_{field}_plot_id_{model_id}_date_{date}_level_{level}.png")

  # Arctic-focused plot
  date = ds_o['datetime'].values[k].astype('datetime64[m]')
  fig = plt.figure(figsize=(8, 8), dpi=300)
  ax = plt.axes(projection=cartopy.crs.NorthPolarStereo())
  ax.set_extent([-180, 180, 60, 90], cartopy.crs.PlateCarree())
  ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
  ax.set_title(f'{field} : {date}\nModel run : {model_id} {model_run}\nLoaded model : {loaded_model}', fontsize=8)

  arctic_lat_mask = ds_o['lat'] >= 60
  data_arctic = ds_o['vo'].values[level, k][arctic_lat_mask, :]
  lats_arctic = ds_o['lat'].values[arctic_lat_mask]
  lons = ds_o['lon'].values

  im = ax.pcolormesh(lons, lats_arctic, np.flip(data_arctic, 0),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=cartopy.crs.PlateCarree())

  axins = inset_axes(ax, width="80%", height="5%", loc='lower center', borderpad=-2)
  fig.colorbar(im, cax=axins, orientation="horizontal")
  plt.savefig(f"/work/ab1412/atmorep/plotting/arctic_forecast_{field}_plot_id_{model_id}_date_{date}_level_{level}.png")
  plt.close()
  print(f"Arctic-focused plot saved as /work/ab1412/atmorep/plotting/arctic_forecast_{field}_plot_id_{model_id}_date_{date}_level_{level}.png")