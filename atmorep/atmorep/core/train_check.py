import os
import zarr

file_path = '/work/ab1412/atmorep/data/vorticity/ml137/era5_y2021_res025_chunk8.zarr'

assert os.path.exists(file_path), f"File path {file_path} does not exist"
ds = zarr.open( file_path, mode='r')

# Print available keys
print("Available keys in the Zarr dataset:", list(ds.array_keys()))
        