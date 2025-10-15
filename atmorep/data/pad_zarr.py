import zarr
import numpy as np
import os
import shutil
from scipy.interpolate import interp1d
from joblib import Parallel, delayed



# Define paths
source_path = "/scratch/a/a270277/atmorep/data_t2m_Akil.zarr"
output_path = "/scratch/a/a270277/atmorep/data_t2m_Akil_padded3.zarr"


def interpolate_longitude(data, target_lon_size=1440):
    """
    Interpolates the last axis (longitude) of data from its current size to target_lon_size.
    Assumes longitude is evenly spaced from 0 to 360 (not including 360).
    """
    orig_lon_size = data.shape[-1]
    # Original and target longitude coordinates
    orig_lons = np.linspace(0, 360, orig_lon_size, endpoint=False)
    target_lons = np.linspace(0, 360, target_lon_size, endpoint=False)
    # Interpolate along the last axis
    interp_func = interp1d(orig_lons, data, axis=-1, kind='linear', fill_value='extrapolate', assume_sorted=True)
    return interp_func(target_lons)

# Create a fresh output directory
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

print(f"Opening source dataset: {source_path}")
source = zarr.open(source_path, mode='r')

# Find the main array to pad
if isinstance(source, zarr.hierarchy.Group):
    if 'data' in source:
        arr = source['data']
    else:
        for key in source.keys():
            if hasattr(source[key], 'shape'):
                arr = source[key]
                print(f"Using array '{key}' from source")
                break
else:
    arr = source

print(f"Source data shape: {arr.shape}")

# Assume arr.shape = (time, lat, lon) or (time, 1, lat, lon)
# If 3D, add singleton dim at axis=1
if arr.ndim == 3:
    time_dim, lat_dim, lon_dim = arr.shape
    arr_shape = (time_dim, 1, lat_dim, lon_dim)
else:
    arr_shape = arr.shape

# Target shape
target_shape = (arr_shape[0], 1, 721, 1440)
chunks = (50, 50, 721, 1440)

# Create output Zarr array
z = zarr.open(output_path, mode='w')
out_arr = z.create_dataset('data', shape=target_shape, chunks=chunks, dtype=arr.dtype)

# Padding config
pad_top_lat = 0
pad_bottom_lat = 721 - arr_shape[2]

pad_config = [
    (0, 0),  # time
    (0, 0),  # singleton
    (pad_top_lat, pad_bottom_lat),  # lat
    (0,0),  # lon (no padding, already interpolated)
]

def process_timestep(t):
    if arr.ndim == 3:
        data = arr[t][np.newaxis, :, :]
        data = np.expand_dims(data, axis=0)
    else:
        data = arr[t:t+1, :, :, :]
    padded_interp = interpolate_longitude(data, target_lon_size=1440)
    padded_lat = np.pad(padded_interp, pad_config, mode='constant', constant_values=0)
    return padded_lat

results = Parallel(n_jobs=8)(delayed(process_timestep)(t) for t in range(arr_shape[0]))

for t, padded_lat in enumerate(results):
    out_arr[t:t+1, :, :, :] = padded_lat

# print("Processing and padding/interpolating in chunks...")
# for t in range(arr_shape[0]):
#     if t % 50 == 0:
#         print(f"Processing time step {t+1}/{arr_shape[0]}") 

    # # Read one time step
    # if arr.ndim == 3:
    #     data = arr[t][np.newaxis, :, :]  # shape (1, lat, lon)
    #     data = np.expand_dims(data, axis=0)  # shape (1, 1, lat, lon)
    # else:
    #     data = arr[t:t+1, :, :, :]  # shape (1, 1, lat, lon)
    # # Interpolate longitude
    # padded_interp = interpolate_longitude(data, target_lon_size=1440)
    # # Pad latitude
    # padded_lat = np.pad(padded_interp, pad_config, mode='constant', constant_values=0)

    # # Write
    # out_arr[t:t+1, :, :, :] = padded_lat

print(f"Final data shape: {out_arr.shape}")

# Add metadata
attrs = {
    'description': 'T2M data reshaped to match ERA5 surface variable dimensions',
    'source_shape': str(arr.shape),
    'target_shape': str(out_arr.shape),
    'latitude_padding': f"Added {pad_top_lat} rows at top, {pad_bottom_lat} rows at bottom",
    'longitude_padding': f"Interpolated {arr.shape[-1]} to 1440",
    'variable': 't2m'
}
z.attrs.put(attrs)

print("Reshaping complete!")
print(f"Original shape: {arr.shape}")
print(f"New shape: {out_arr.shape}")