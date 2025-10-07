import xarray as xr
import numpy as np


"""
    compare ERA5 T2M vs model t2m against N-ICE T2M

"""
# Read datasets
ERA5_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc")
NICE_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc")

#print era5 data indice slicing shape
print("ERA5 data shape:", ERA5_data['T2M'].shape)

# Select the same time range from N-ICE
NICE_data = NICE_data.sel(time=ERA5_data.time)

# Now extract variables
ERA5_T2M = ERA5_data['T2M']
NICE_T2M = NICE_data['air_temperature_2m']

# Check for NaNs and print their indices in the original datasets
era5_nan_idx = np.where(np.isnan(ERA5_T2M.values))[0]
nice_nan_idx = np.where(np.isnan(NICE_T2M.values))[0]
print("ERA5 NaN indices:", era5_nan_idx.shape)
print("NICE NaN indices:", nice_nan_idx.shape)

# Drop NaNs together
combined = xr.Dataset({'ERA5_T2M': ERA5_T2M, 'NICE_T2M': NICE_T2M}).dropna(dim='time')

ERA5_T2M_clean = combined['ERA5_T2M'] #.values
NICE_T2M_clean = combined['NICE_T2M'] #.values

print("ERA5 T2M values:", ERA5_T2M_clean.shape, ERA5_T2M_clean.values[:3], " ... ", ERA5_T2M_clean.values[-1:])
print("NICE T2M values:", NICE_T2M_clean.shape, NICE_T2M_clean.values[:3], " ... ", NICE_T2M_clean.values[-1:])
print("Date range:", combined.time.min().values, "to", combined.time.max().values)


"""
ERA5 T2M values: (2099,) [265.40625 265.06296 264.92712]  ...  [273.5279]
NICE T2M values: (2099,) [268.00564583 267.92406667 266.87071667]  ...  [272.45]
Date range: 2015-01-21T21:00:00.000000000 to 2015-06-12T23:00:00.000000000

"""