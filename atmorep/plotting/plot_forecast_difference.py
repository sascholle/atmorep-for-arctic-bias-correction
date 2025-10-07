import numpy as np
import zarr
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def centers_to_edges(arr):
    edges = np.empty(arr.size + 1)
    edges[1:-1] = (arr[:-1] + arr[1:]) / 2
    edges[0] = arr[0] - (arr[1] - arr[0]) / 2
    edges[-1] = arr[-1] + (arr[-1] - arr[-2]) / 2
    return edges

def plot_forecast_difference(plot_type, field1, field2, model_id="hha4bqe6", model_run="19513219", loaded_model="iuw3ce3v"):
    """
    Plot forecast differences based on the specified plot type.

    Parameters:
    plot_type (str): Type of plot to generate. Options are:
                     'global_diff_same_model', 'arctic_diff_same_model', 'arctic_diff_separate_models'.
    """


    ######
    ### Global difference plot of corrected_t2m - t2m for the same model ####
    ######

    if plot_type == 'global_diff_same_model':

        # Load Zarr store
        store = zarr.ZipStore(f'results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
        ds = zarr.group(store=store)
        print(f"Loaded Zarr store from {store}")

        # Use i=0 as template
        i = 0
        lat = np.linspace(-90., 90., num=180*4+1, endpoint=True)
        lon = np.linspace(0., 360., num=360*4, endpoint=False)
        ml = ds['temperature/sample=00000/ml'][:]
        datetime = ds['temperature/sample=00000/datetime'][:]

        ds_o = xr.Dataset(coords={
            'ml': ml,
            'datetime': datetime,
            'lat': lat,
            'lon': lon
        })

        nlevels = ml.shape[0]
        ntime = datetime.shape[0]
        ds_o['diff'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros((nlevels, ntime, lat.shape[0], lon.shape[0])))

        # Fill in local patches for both fields and compute difference
        for i_str in ds['temperature']:
            t2m_data = np.array(ds[f'temperature/{i_str}/data'])
            corrected_data = np.array(ds[f'corrected_t2m/{i_str}/data'])
            diff_data = corrected_data - t2m_data
            ds_o['diff'].loc[dict(
                datetime=ds[f'temperature/{i_str}/datetime'][:],
                lat=ds[f'temperature/{i_str}/lat'][:],
                lon=ds[f'temperature/{i_str}/lon'][:]
            )] = diff_data

        # Plot the difference for the first time step and specified level
        level = 4 # Change as needed
        k = 0      # First time step
        vmin, vmax = -10, 10  # Adjust as needed for difference
        cmap = matplotlib.colormaps.get_cmap('RdBu_r')

        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax = plt.axes(projection=cartopy.crs.Robinson(central_longitude=0.))
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
        ax.set_global()
        date = ds_o['datetime'].values[k].astype('datetime64[m]')
        ax.set_title(f'Difference: corrected_t2m - t2m\n{date}\nModel run: {model_id} {model_run}\nLoaded model: {loaded_model}', fontsize=8)
        im = ax.imshow(np.flip(ds_o['diff'].values[level, k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=cartopy.crs.PlateCarree(central_longitude=180.))
        axins = inset_axes(ax, width="80%", height="5%", loc='lower center', borderpad=-2)
        fig.colorbar(im, cax=axins, orientation="horizontal")
        plt.savefig(f"/work/ab1412/atmorep/plotting/forecast_diff_t2m_plot_id_{model_id}_date_{date}_level_{level}.png")
        plt.close()

        print(f"Plot saved as /work/ab1412/atmorep/plotting/forecast_diff_t2m_plot_id_{model_id}_date_{date}_level_{level}.png")

        ######
        ### Arctic-focused difference plot of same model ####
        ######

    if plot_type == 'arctic_diff_same_model':

        print(f"Generating Arctic-focused difference plot of {field1} - {field2} for model {model_id}")
        # Load Zarr store
        store = zarr.ZipStore(f'results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
        ds = zarr.group(store=store)
        print(f"Loaded Zarr store from {store}")

        # Use i=0 as template
        i = 0
        lat = np.linspace(-90., 90., num=180*4+1, endpoint=True)
        lon = np.linspace(0., 360., num=360*4, endpoint=False)
        ml = ds[f'{field1}/sample=00000/ml'][:]
        datetime = ds[f'{field1}/sample=00000/datetime'][:]

        ds_o = xr.Dataset(coords={
            'ml': ml,
            'datetime': datetime,
            'lat': lat,
            'lon': lon
        })

        nlevels = ml.shape[0]
        ntime = datetime.shape[0]
        ds_o['diff'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros((nlevels, ntime, lat.shape[0], lon.shape[0])))

        # Fill in local patches for both fields and compute difference
        for i_str in ds[f'{field1}']:
            #print(f"Processing patch {i_str} for fields {field1} and {field2}")
            field1_array = np.array(ds[f'{field1}/{i_str}/data'])
            field2_array = np.array(ds[f'{field2}/{i_str}/data'])
            diff_data = field2_array - field1_array
            ds_o['diff'].loc[dict(
                datetime=ds[f'{field1}/{i_str}/datetime'][:],
                lat=ds[f'{field1}/{i_str}/lat'][:],
                lon=ds[f'{field1}/{i_str}/lon'][:]
            )] = diff_data
            

        # Plot the difference for the first time step and specified level
        level = 4 # Change as needed
        k = 0      # First time step
        vmin, vmax = -10, 10  # Adjust as needed for difference
        cmap = matplotlib.colormaps.get_cmap('RdBu_r')

        lat_vals = ds_o['lat'].values
        lat_mask = (lat_vals >= 60) & (lat_vals <= 90)
        lat_indices = np.where(lat_mask)[0]

        # Get index for the target longitude
        lon_vals = ds_o['lon'].values
        lon_idx = 80

        print(f"Difference values for lon={lon_vals[lon_idx]} from lat 60N to 90N:")
        for i in lat_indices:
            print(f"lat={lat_vals[i]:.2f}, diff={ds_o['diff'].values[level, k, i, lon_idx]}")

        date = ds_o['datetime'].values[k].astype('datetime64[m]')
        fig = plt.figure(figsize=(8, 9), dpi=300)
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo())
        ax.set_extent([-180, 180, 70, 90], cartopy.crs.PlateCarree())
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
        ax.set_title(f'Arctic Difference: {field1} - {field2}\n{date}\nModel run: {model_id} {model_run}\nLoaded model: {loaded_model}', fontsize=14)

        arctic_lat_mask = ds_o['lat'] >= 60
        data_arctic = ds_o['diff'].values[level, k][arctic_lat_mask, :]
        lats_arctic = ds_o['lat'].values[arctic_lat_mask]
        lons = ds_o['lon'].values

        lats_arctic_edges = centers_to_edges(lats_arctic)
        lons_edges = centers_to_edges(lons)

        #can you add latitude lines and longitude lines to the arctic plot with labels?
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        im = ax.imshow(
            np.flip(data_arctic, 0),
            extent=[lons[0], lons[-1], lats_arctic[0], lats_arctic[-1]],
            cmap=cmap, vmin=vmin, vmax=vmax,
            transform=cartopy.crs.PlateCarree()
        )
        # im = ax.pcolormesh(lons_edges, lats_arctic_edges, np.flip(data_arctic, 0),
        #                 cmap=cmap, vmin=vmin, vmax=vmax,
        #                 transform=cartopy.crs.PlateCarree())

        axins = inset_axes(ax, width="80%", height="5%", loc='lower center', borderpad=-2)
        fig.colorbar(im, cax=axins, orientation="horizontal")
        fig_name=f"/work/ab1412/atmorep/plotting/arctic_forecast_diff_{field1}-{field2}_plot_id_{model_id}_date_{date}_level_{level}.png"
        plt.savefig(fig_name)
        plt.close()
        print(f"Arctic-focused difference plot saved as {fig_name}")


    ######
    ### Arctic-focused difference plot of two different models  ####
    ######


    if plot_type == 'arctic_diff_separate_models':
        
        # Load Zarr stores for both models
        store_main = zarr.ZipStore(f'results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
        ds_main = zarr.group(store=store_main)
        print(f"Loaded main Zarr store from {store_main}")

        # Other model info
        other_model_id = "9ah3u47h"
        other_model_run = "19513414"
        store_other = zarr.ZipStore(f'results/id{other_model_id}/results_id{other_model_id}_epoch00000_pred.zarr')
        ds_other = zarr.group(store=store_other)
        print(f"Loaded other Zarr store from {store_other}")

        # Use i=0 as template
        i = 0
        lat = np.linspace(-90., 90., num=180*4+1, endpoint=True)
        lon = np.linspace(0., 360., num=360*4, endpoint=False)
        ml = ds_main['t2m/sample=00000/ml'][:]
        datetime = ds_main['t2m/sample=00000/datetime'][:]

        ds_o = xr.Dataset(coords={
            'ml': ml,
            'datetime': datetime,
            'lat': lat,
            'lon': lon
        })

        nlevels = ml.shape[0]
        ntime = datetime.shape[0]
        ds_o['diff'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros((nlevels, ntime, lat.shape[0], lon.shape[0])))

        # Fill in local patches for both fields and compute difference
        for i_str in ds_main['t2m']:
            corrected_data = np.array(ds_main[f'corrected_t2m/{i_str}/data'])
            # Use t2m from the other model
            t2m_other_data = np.array(ds_other[f't2m/{i_str}/data'])
            diff_data = corrected_data - t2m_other_data
            ds_o['diff'].loc[dict(
                datetime=ds_main[f't2m/{i_str}/datetime'][:],
                lat=ds_main[f't2m/{i_str}/lat'][:],
                lon=ds_main[f't2m/{i_str}/lon'][:]
            )] = diff_data

        # Arctic-focused difference plot
        level = 4 # Change as needed
        k = 0      # First time step
        vmin, vmax = -10, 10  # Adjust as needed for difference
        cmap = matplotlib.colormaps.get_cmap('RdBu_r')
        date = ds_o['datetime'].values[k].astype('datetime64[m]')
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo())
        ax.set_extent([-180, 180, 60, 90], cartopy.crs.PlateCarree())
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
        ax.set_title(f'Arctic Difference: corrected_t2m ({model_id}) - t2m ({other_model_id})\n{date}\nMain model: {model_id} {model_run}\nOther model: {other_model_id} {other_model_run}', fontsize=8)

        arctic_lat_mask = ds_o['lat'] >= 60
        data_arctic = ds_o['diff'].values[level, k][arctic_lat_mask, :]
        lats_arctic = ds_o['lat'].values[arctic_lat_mask]
        lons = ds_o['lon'].values

        im = ax.pcolormesh(lons, lats_arctic, np.flip(data_arctic, 0),
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        transform=cartopy.crs.PlateCarree())

        axins = inset_axes(ax, width="80%", height="5%", loc='lower center', borderpad=-2)
        fig.colorbar(im, cax=axins, orientation="horizontal")
        plt.savefig(f"/work/ab1412/atmorep/plotting/arctic_forecast_diff_t2m_vs_wc_plot_id_{model_id}_vs_{other_model_id}_date_{date}_level_{level}.png")
        plt.close()
        print(f"Arctic-focused difference plot saved as /work/ab1412/atmorep/plotting/arctic_forecast_diff_t2m_vs_wc_plot_id_{model_id}_vs_{other_model_id}_date_{date}_level_{level}.png")

#
if __name__ == "__main__":

#0: Processing line: 0: Wandb run: atmorep-cjtlxcuc-19707366 - 2015-01-21 19:00:00
#0: Processing line: 0: Wandb run: atmorep-ny8qo9me-19707374 - 2015-02-17 17:00:00
#0: Processing line: 0: Wandb run: atmorep-ts9hael0-19707380 - 2015-03-15 05:00:00
#0: Processing line: 0: Wandb run: atmorep-qsubzdar-19707380 - 2015-04-22T21
#0: Processing line: 0: Wandb run: atmorep-8jvqsal4-19707380 - 2015-04-23T15
#0: Processing line: 0: Wandb run: atmorep-vhko6m4j-19707381 - 2015-04-24T15
#0: Processing line: 0: Wandb run: atmorep-5vogn4mz-19707390 - 2015-05-21T20
#0: Processing line: 0: Wandb run: atmorep-8dgi3kal-19707397 - 2015-06-



    # Model/run info from the comment
    model_id = "8dgi3kal" # "hha4bqe6"
    model_run = "19707397" #"19513219"
    loaded_model =  "iuw3ce3v_single_gpu" #"iuw3ce3v"

    # Choose one of the following plot types to generate:
    # 'global_diff_same_model', 'arctic_diff_same_model', 'arctic_diff_separate_models'
    
   # plot_forecast_difference('global_diff_same_model')
    plot_forecast_difference('arctic_diff_same_model', 't2m', 'corrected_t2m', model_id=model_id, model_run=model_run, loaded_model=loaded_model)
   # plot_forecast_difference('arctic_diff_separate_models')


'''
source /work/ab1412/atmorep/pyenv/bin/activate
python /work/ab1412/atmorep/plotting/plot_forecast_difference.py

'''