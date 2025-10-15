import zarr, glob, os
zarr_paths = [
    "/work/ab1412/atmorep/data/t2m/era5_y2010_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2011_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2012_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2013_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2014_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2015_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2016_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2017_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2018_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2019_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2020_res025_chunk8.zarr",
    "/work/ab1412/atmorep/data/t2m/era5_y2021_res025_chunk8.zarr"
    
]

for path in zarr_paths:
    z = zarr.open(path, mode='a')
    # For 4D data_sfc: (time, field, lat, lon)
    if 'data_sfc' in z:
        z['data_sfc'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'field', 'latitude', 'longitude']
    # For 5D data: (time, level, field, lat, lon)
    if 'data' in z:
        z['data'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'field', 'level', 'latitude', 'longitude']
    print(f"Added _ARRAY_DIMENSIONS to {path}")
    # for lats Shape: (721,)
    if 'lats' in z:
        z['lats'].attrs['_ARRAY_DIMENSIONS'] = ['latitude']
        print(f"Added _ARRAY_DIMENSIONS to lats in {path}")
    # for lons Shape: (1440,)
    if 'lons' in z:
        z['lons'].attrs['_ARRAY_DIMENSIONS'] = ['longitude']
        print(f"Added _ARRAY_DIMENSIONS to lons in {path}")
    # for time Shape: (8760,)
    if 'time' in z:
        z['time'].attrs['_ARRAY_DIMENSIONS'] = ['time']
        print(f"Added _ARRAY_DIMENSIONS to time in {path}")



zarr_files = glob.glob("/work/ab1412/atmorep/data/t2m/era5_y20*_res025_chunk8.zarr")
for path in zarr_files:
    z = zarr.open(path, mode='a')
    # Remove 'data' if it exists and is empty or has time=0
    if 'data' in z:
        if z['data'].shape[0] == 0:
            print(f"Deleting empty 'data' array in {path}")
            del z['data']
    # Remove other empty arrays if needed
    if 'norm' in z['normalization']:
        if z['normalization']['norm'].shape[0] == 0:
            print(f"Deleting empty 'norm' array in {path}/normalization")
            del z['normalization']['norm']
    if 'global_norm' in z['normalization']:
        if z['normalization']['global_norm'].shape[0] == 0:
            print(f"Deleting empty 'global_norm' array in {path}/normalization")
            del z['normalization']['global_norm']

    
    zarr_files = glob.glob("/work/ab1412/atmorep/data/t2m/era5_y20*_res025_chunk8.zarr")
for path in zarr_files:
    z = zarr.open(path, mode='r')
    for arr_name in z.array_keys():
        arr = z[arr_name]
        if '_ARRAY_DIMENSIONS' not in arr.attrs:
            print(f"Missing _ARRAY_DIMENSIONS in {os.path.join(path, arr_name)}")