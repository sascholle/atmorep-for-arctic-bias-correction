import numpy as np
import xarray as xr
import zarr
import netCDF4
import glob
import os


def examine_nc(nc_file):
    """Examine NetCDF file structure using netCDF4"""
    ds = netCDF4.Dataset(nc_file)
    
    print("\nNetCDF Dataset Info:")
    print("-------------------")
    print(f"Dimensions:")
    for dim in ds.dimensions.items():
        print(f"  {dim[0]}: {dim[1].size}")
    
    print(f"\nVariables:")
    for var in ds.variables.items():
        print(f"  {var[0]}: {var[1].shape} {var[1].dtype}")
    
    print(f"\nGlobal Attributes:")
    for attr in ds.ncattrs():
        print(f"  {attr}: {ds.getncattr(attr)}")
    
    ds.close()
    return

# NetCDF variable mapping
nc_index = {
    't2m': 'T2M',  # Update these to match your NetCDF variable names
}

def convert_nc_to_zarr(nc_file, zarr_file, chunks=None):
    """Convert NetCDF to Zarr with optional chunking."""
    print(f"Reading NetCDF file: {nc_file}")
    
    # Open NetCDF dataset
    ds_nc = xr.open_dataset(nc_file)
    
    # Print dataset info
    print("\nDataset Info:")
    print(f"Dimensions: {ds_nc.dims}")
    print(f"Variables: {list(ds_nc.variables)}")
    print(f"Coordinates: {list(ds_nc.coords)}")
    
    # Set default chunks if not provided
    if chunks is None:
        chunks = {
            'time': 8,
            'latitude': 54,
            'longitude': 108
        }
    
    # Convert to zarr
    print(f"\nConverting to Zarr with chunks: {chunks}")
    ds_nc.chunk(chunks).to_zarr(zarr_file, mode='w')
    
    print(f"\nConversion complete. Zarr dataset saved to {zarr_file}")
    
    # Verify zarr structure
    zg = zarr.open(zarr_file)
    print("\nZarr structure:")
    print(zg.tree())

if __name__ == "__main__":
    ########################################## old code
    # # Input/Output paths
    # nc_file = "/work/ab1385/a270082/era_corrected/T2M_2021_xr1.nc"
    # zarr_file = "/work/ab1412/atmorep/data/era_corrected/T2M_2021.zarr"
    
    # # Define chunks based on your model's token structure
    # chunks = {
    #     'dim_0': 8, #time
    #     'dim_1': -1, #lat # -1 means no chunking in this dimension
    #     'dim_2': -1 #long
    # }
    
    # # Convert file
    # convert_nc_to_zarr(nc_file, zarr_file, chunks)

    ########################################

    # Input/Output paths
    nc_path = "/scratch/a/a270277/atmorep/data_nc"
    nc_pattern = "prediction_T2M_arctic_*.nc"
    #nc_pattern = "ERA5lt80_merged_pred_T2M_*.nc"
    zarr_file = "/scratch/a/a270277/atmorep/data_t2m_Akil.zarr"

    # Create directory if it doesn't exist
    zarr_dir = os.path.dirname(zarr_file)
    if not os.path.exists(zarr_dir):
        os.makedirs(zarr_dir)
    
    # Find all matching files and sort them
    nc_files = sorted(glob.glob(os.path.join(nc_path, nc_pattern)))
    print(f"Found {len(nc_files)} files to process:")
    print(nc_files)

     # Process in batches
    batch_size = 2  # Adjust based on available memory
    mode = 'w'  # First batch creates new zarr
    
    for i in range(0, len(nc_files), batch_size):
        batch_files = nc_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} of {(len(nc_files)-1)//batch_size + 1}")
        
        # Read and combine batch
        datasets = []
        for nc_file in batch_files:
            print(f"Reading: {os.path.basename(nc_file)}")
            ds = xr.open_dataset(nc_file)
            datasets.append(ds)
        
        # Combine batch
        batch_ds = xr.concat(datasets, dim='time')
        
        # Define chunks
        chunks = {
            'time': 8,
            'lat': 54,  # Specify chunk size based on your data
            'lon': 108  # Specify chunk size based on your data
        }
        
        print(f"Converting batch to Zarr ({mode} mode)")
        if mode == 'w':
            # First batch - create new zarr
            batch_ds.chunk(chunks).to_zarr(zarr_file, mode=mode)
            mode = 'a'  # Switch to append mode for subsequent batches
        else:
            # Append to existing zarr
            batch_ds.chunk(chunks).to_zarr(zarr_file, append_dim='time')
        
        # Clean up batch
        for ds in datasets:
            ds.close()
        del datasets
        del batch_ds
    
    # Verify final zarr structure
    print("\nVerifying final Zarr dataset structure:")
    zg = zarr.open(zarr_file)
    print(zg.tree())
    
    print(f"\nConversion complete. Combined Zarr dataset saved to {zarr_file}")