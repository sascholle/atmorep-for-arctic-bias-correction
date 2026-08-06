import numpy as np
import xarray as xr
import zarr
import netCDF4

def examine_nc(nc_file):
    """Examine NetCDF file structure and data using netCDF4"""
    ds = netCDF4.Dataset(nc_file)
    
    print("\nNetCDF Dataset Info:")
    print("-------------------")
    print(f"Dimensions:")
    for dim in ds.dimensions.items():
        print(f"  {dim[0]}: {dim[1].size}")
    
    print(f"\nVariables:")
    for var_name, var in ds.variables.items():
        print(f"\n  {var_name}:")
        print(f"    Shape: {var.shape}")
        print(f"    Dtype: {var.dtype}")
        print(f"    Attributes: {var.ncattrs()}")
        
        # Show a small slice of data
        if len(var.shape) == 3:  # For 3D variables (time, lat, lon)
            print(f"    First timestep data slice (3x3x3):\n{var[:3, :3, :3]}")
        elif len(var.shape) == 1:  # For coordinate variables
            print(f"    First 5 values: {var[:5]}")
            print(f"    Last 5 values: {var[-5:]}")
    
    print(f"\nGlobal Attributes:")
    for attr in ds.ncattrs():
        print(f"  {attr}: {ds.getncattr(attr)}")
    
    ds.close()

if __name__ == "__main__":
    #nc_file = "/work/ab1385/a270082/era_corrected/T2M_2011_grid.nc"
    #nc_file = '/work/ab1385/a270164/2024-sebai/data/arctic_era5/E5_T2M_sf121H_199410_202409_Arc.nc'
    #nc_file = '/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc'
    nc_file = '/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc'

    # Examine NC file first
    print("Examining NetCDF file structure and data...")
    #examine_nc(nc_file)
    
    # Alternative using xarray for more detailed inspection
    print("\nExamining with xarray...")
    ds = xr.open_dataset(nc_file)
    #print(ds)
    
    # Show actual data for main variable
    print("\nFirst 100 timesteps of time, air_temperature_2m, lat and lon:")
    print(ds['time'][59:100].values)  # Show first 100 time values
    print(ds['air_temperature_2m'][59:100].values)  # Show first 100 values
    print(ds['latitude'][59:100].values)
    print(ds['longitude'][59:3000].values)

    ds.close()