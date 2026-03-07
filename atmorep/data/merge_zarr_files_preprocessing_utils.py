import zarr
import numpy as np
import os
import shutil
import xarray as xr
import glob
import pandas as pd
import dask.array as da
import examine_zarr
from pathlib import Path

import subprocess

def merge_data(source_path, target_path, output_path):
    """
    Merge data from source zarr file into target zarr file along dimension 2.
    Result will have shape (8760, 1, 2, 721, 1440)
    
    Parameters:
    - source_path: Path to the source zarr with t2m data.
    - target_path: Path to the target zarr with temperature data.
    - output_path: Path to save the merged output zarr file.
    """
    # Define file paths
    source_path = "/work/ab1412/atmorep/data/era_corrected/T2M_y2021_regridded_reshaped.zarr"
    target_path = "/work/ab1412/atmorep/data/temperature/ml137/era5_y2021_res025_chunk8.zarr"
    output_path = "/work/ab1412/atmorep/data/temperature/ml137/era5_y2021_res025_chunk8_with_t2m.zarr"

    print(f"Opening source: {source_path}")
    print(f"Opening target: {target_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Open source and target datasets
    source_zarr = zarr.open(source_path, mode='r')
    target_zarr = zarr.open(target_path, mode='r')

    # Get data arrays but don't load content yet
    if 'data' in source_zarr:
        source_array = source_zarr['data']
    else:
        # Find array with the right shape
        for key in source_zarr.array_keys():
            if hasattr(source_zarr[key], 'shape') and len(source_zarr[key].shape) == 5:
                source_array = source_zarr[key]
                print(f"Using source array '{key}'")
                break
    
    if 'data' in target_zarr:
        target_array = target_zarr['data']
    else:
        # Find array with the right shape
        for key in target_zarr.array_keys():
            if hasattr(target_zarr[key], 'shape') and len(target_zarr[key].shape) == 5:
                target_array = target_zarr[key]
                print(f"Using target array '{key}'")
                break
    
    print(f"Source array shape: {source_array.shape}")
    print(f"Target array shape: {target_array.shape}")
    
    # Create output zarr store
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    output_zarr = zarr.open(output_path, mode='w')
    
    # Create the output dataset with the correct shape but empty
    output_shape = list(source_array.shape)
    output_shape[2] = 2  # Two variables in dimension 2
    chunks = (10, 1, 1, 721, 1440)  # Process in small chunks of 10 timesteps
    
    print(f"Creating output dataset with shape {output_shape}")
    output_array = output_zarr.create_dataset('data', shape=tuple(output_shape), 
                                             chunks=chunks, dtype=source_array.dtype)
    
    # Process in chunks to avoid loading everything into memory at once
    chunk_size = 100  # Process 100 timesteps at a time
    total_steps = source_array.shape[0]
    
    print(f"Processing data in chunks of {chunk_size} timesteps")
    for i in range(0, total_steps, chunk_size):
        end = min(i + chunk_size, total_steps)
        print(f"Processing chunk {i} to {end} (of {total_steps})")
        
        # Load only this chunk of data
        source_chunk = source_array[i:end]
        target_chunk = target_array[i:end]
        
        # Concatenate just this chunk along dimension 2
        merged_chunk = np.concatenate([source_chunk, target_chunk], axis=2)
        
        # Write just this chunk to the output
        output_array[i:end] = merged_chunk
        
        print(f"Chunk {i}-{end} completed")
    
    # Copy metadata and update it
    if hasattr(target_zarr, 'attrs') and target_zarr.attrs:
        # Copy target attributes
        for key, value in target_zarr.attrs.items():
            output_zarr.attrs[key] = value
    
    # Update attributes to reflect the merged dataset
    output_zarr.attrs['description'] = 'Merged dataset containing temperature and t2m'
    output_zarr.attrs['source_datasets'] = [source_path, target_path]
    output_zarr.attrs['variables'] = ['temperature', 't2m']
    
    # Set array dimensions for xarray compatibility
    output_array.attrs['_ARRAY_DIMENSIONS'] = ['time', 'level', 'variable', 'latitude', 'longitude']
    
    print(f"Merged dataset created successfully at {output_path}")
    print(f"Final shape: {tuple(output_shape)}")

def replace_sfc_data(source_path, target_path):
    """
    Replace data_sfc in the target zarr file with data from source zarr file.
    
    Parameters:
    -----------
    source_path : str
        Path to source zarr containing T2M data
    target_path : str
        Path to target zarr where data_sfc will be replaced
    """
    print(f"Replacing data_sfc in target file with data from source")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Open source and target datasets
    source_zarr = zarr.open(source_path, mode='r')
    target_zarr = zarr.open(target_path, mode='a')  # Open in append mode
    
    # Get source data
    if 'data' in source_zarr:
        source_data = source_zarr['data']
    else:
        # Find array with the right shape
        for key in source_zarr.array_keys():
            if hasattr(source_zarr[key], 'shape'):
                source_data = source_zarr[key]
                print(f"Using source array '{key}'")
                break
    
    print(f"Source data shape: {source_data.shape}")
    
    # Check if data_sfc exists in target
    has_data_sfc = 'data_sfc' in target_zarr
    if has_data_sfc:
        print(f"Target data_sfc shape: {target_zarr['data_sfc'].shape}")
    else:
        print("data_sfc does not exist in target, will create it")
    
    # Process in chunks to avoid memory issues
    chunk_size = 100
    total_steps = source_data.shape[0]
    
    # Delete existing data_sfc if it exists
    if has_data_sfc:
        print("Deleting existing data_sfc")
        del target_zarr['data_sfc']
    
    # Create new data_sfc with source data shape and chunks
    print("Creating new data_sfc with source data")
    chunks = (10, 1, 721, 1440) if len(source_data.shape) == 4 else source_data.chunks
    data_sfc = target_zarr.create_dataset('data_sfc', shape=source_data.shape, 
                                       chunks=chunks, dtype=source_data.dtype)
    
    # Copy data in chunks
    print(f"Copying data in chunks of {chunk_size} timesteps")
    for i in range(0, total_steps, chunk_size):
        end = min(i + chunk_size, total_steps)
        print(f"Processing chunk {i} to {end} (of {total_steps})")
        
        # Load this chunk of data
        source_chunk = source_data[i:end]
        
        # Copy to target
        data_sfc[i:end] = source_chunk
        
        print(f"Chunk {i}-{end} completed")
    
    # Update array dimensions for xarray compatibility
    if len(source_data.shape) == 4:  # 4D array (time, level, lat, lon)
        data_sfc.attrs['_ARRAY_DIMENSIONS'] = ['time', 'level', 'latitude', 'longitude']
    elif len(source_data.shape) == 5:  # 5D array (time, level, variable, lat, lon)
        data_sfc.attrs['_ARRAY_DIMENSIONS'] = ['time', 'level', 'variable', 'latitude', 'longitude']
    
    # Update attributes in metadata
    if '.zattrs' in target_zarr:
        if 'fields_sfc' in target_zarr['.zattrs']:
            # Make sure fields_sfc contains t2m
            if 't2m' not in target_zarr['.zattrs']['fields_sfc']:
                target_zarr['.zattrs']['fields_sfc'].append('t2m')
        else:
            # Create fields_sfc if it doesn't exist
            target_zarr['.zattrs']['fields_sfc'] = ['t2m']
    
    print(f"Successfully replaced data_sfc in {target_path}")
    print(f"Final data_sfc shape: {data_sfc.shape}")

def copy_with_float32_conversion(source_path, target_path):
    """
    Copy data from source zarr to target zarr, ensuring the data is stored as float32.
    
    Parameters:
    -----------
    source_path : str
        Path to source zarr containing the data to copy
    target_path : str
        Path to target zarr where data will be replaced
    """
    print(f"Copying data with float32 conversion")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Open source and target datasets
    source_zarr = zarr.open(source_path, mode='r')
    target_zarr = zarr.open(target_path, mode='a')  # Open in append mode
    
    # Get source data array
    if 'data' in source_zarr:
        source_data = source_zarr['data']
        print(f"Source data shape: {source_data.shape}")
        print(f"Source data dtype: {source_data.dtype}")
    else:
        raise ValueError("Source zarr does not contain 'data' array")
    
    # Create new dataset in target with float32 dtype
    if 'data_sfc' in target_zarr:
        print(f"Target data exists with shape: {target_zarr['data_sfc'].shape}")
        print(f"Target data dtype: {target_zarr['data_sfc'].dtype}")
        print("Deleting existing data")
        del target_zarr['data_sfc']
    
    # Process in chunks to avoid memory issues
    chunk_size = 100  # Process 100 timesteps at a time
    total_steps = source_data.shape[0]
    
    # Create new dataset with float32 dtype and same shape
    print(f"Creating new data array with float32 dtype")
    chunks = source_data.chunks
    target_data = target_zarr.create_dataset('data_sfc', 
                                           shape=source_data.shape,
                                           chunks=chunks,
                                           dtype=np.float32)  # Explicitly use float32
    
    print(f"Processing data in chunks of {chunk_size} timesteps")
    for i in range(0, total_steps, chunk_size):
        end = min(i + chunk_size, total_steps)
        print(f"Processing chunk {i} to {end} (of {total_steps})")
        
        # Load this chunk and convert to float32
        source_chunk = source_data[i:end]
        if source_chunk.dtype != np.float32:
            print(f"Converting chunk from {source_chunk.dtype} to float32")
            source_chunk = source_chunk.astype(np.float32)
        
        # Write to target
        target_data[i:end] = source_chunk
        
        print(f"Chunk {i}-{end} completed")
    
    # Copy array dimension attributes
    if hasattr(source_data, 'attrs') and '_ARRAY_DIMENSIONS' in source_data.attrs:
        target_data.attrs['_ARRAY_DIMENSIONS'] = source_data.attrs['_ARRAY_DIMENSIONS']
    
    print(f"Successfully copied data with float32 conversion")
    print(f"Final data shape: {target_data.shape}")
    print(f"Final data dtype: {target_data.dtype}")

def reshape_zarr_array(source_path, target_path):
    """
    Reshape a zarr array by removing a dimension of size 1.
    
    Parameters:
    -----------
    source_path : str
        Path to the source zarr directory
    target_path : str
        Path to save the reshaped zarr
    """
    print(f"Reshaping zarr array from {source_path} to {target_path}")
    
    # Open the source zarr array
    source = zarr.open(source_path, mode='r')
    
    # Get the original shape and chunks
    original_shape = source.shape
    original_chunks = source.chunks
    
    print(f"Original shape: {original_shape}")
    print(f"Original chunks: {original_chunks}")
    
    # Define new shape and chunks (removing the fourth dimension which has size 1)
    new_shape = original_shape[:3] + original_shape[4:]
    new_chunks = original_chunks[:3] + original_chunks[4:]
    
    print(f"New shape: {new_shape}")
    print(f"New chunks: {new_chunks}")
    
    # Create the target directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Remove the target directory if it already exists
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    
    # Create the target zarr array
    target = zarr.create(
        shape=new_shape,
        chunks=new_chunks,
        dtype=source.dtype,
        store=target_path,
        compressor=source.compressor,
        fill_value=source.fill_value,
        order=source.order
    )
    
    # Copy data in chunks to avoid memory issues
    chunk_size = 4  # Process 4 months at a time
    for i in range(0, original_shape[0], chunk_size):
        end = min(i + chunk_size, original_shape[0])
        print(f"Processing chunk {i} to {end} (of {original_shape[0]})")
        
        # Load source chunk
        source_chunk = source[i:end]
        
        # Reshape the chunk by removing the fourth dimension
        target_chunk = source_chunk.reshape(
            (end-i,) + new_shape[1:]
        )
        
        # Write to target
        target[i:end] = target_chunk
    
    print(f"Successfully reshaped zarr array")
    print(f"Final shape: {target.shape}")
    
    # Copy any attributes
    if hasattr(source, 'attrs'):
        for key, value in source.attrs.items():
            target.attrs[key] = value

def merge_several_years():
    """
    Merge several years of zarr data into a single zarr file, including norm_sfc.
    """
    # Find all yearly Zarr stores and sort them
    zarr_files = sorted(glob.glob("/work/ab1412/atmorep/data/t2m/era5_y20*_res025_chunk8.zarr"))

    print("Combining these files:")
    for f in zarr_files:
        print(f)

    # Open both data_sfc and norm_sfc from each file
    datasets = [xr.open_zarr(f, consolidated=False)[['data_sfc', 'norm_sfc']] for f in zarr_files]

    # Concatenate along the time dimension for data_sfc and month (first dim) for norm_sfc
    combined = xr.concat(datasets, dim="time")

    # Rechunk for efficient storage (optional, adjust as needed)
    combined = combined.chunk({"time": 1000})

    # Set the _ARRAY_DIMENSIONS attribute for xarray/zarr compatibility
    combined['data_sfc'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'field', 'latitude', 'longitude']
    combined['norm_sfc'].attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'field', 'latitude', 'longitude']

    # Save to a new Zarr store
    combined_path = "/work/ab1412/atmorep/data/t2m/era5_y2010_2021_res025_chunk8.zarr"
    combined.to_zarr(combined_path, mode="w")

    print(f"Combined Zarr written to: {combined_path}")
    print(f"Final data_sfc shape: {combined['data_sfc'].shape}")
    print(f"Final norm_sfc shape: {combined['norm_sfc'].shape}")

def add_norm_sfc_to_t2m():
    # t2m_path = "/work/ab1412/atmorep/data/t2m/era5_y2010_2021_res025_chunk8.zarr"
    t2m_path = "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr"
    t2m_ds = xr.open_zarr(t2m_path, consolidated=False)

    # Assume data_sfc shape: (time, lat, lon) or (time, 1, lat, lon)
    data = t2m_ds['data_sfc']
    if data.ndim == 3:
        # Add singleton dimension at axis=1
        data = data.expand_dims(dim={'field': [0]})
        data = data.transpose('time', 'field', 'lat', 'lon')
    data = data.chunk({'time': 1000})
    data = data.assign_coords(time=t2m_ds['time'])

   # Calculate year and month indices for each time step
    time = t2m_ds['time'].values  # shape: (time,)
    time_pd = pd.to_datetime(time)
    years = time_pd.year
    months = time_pd.month

    # Build (year, month) pairs
    year_month_pairs = [(y, m) for y, m in zip(years, months)]
    unique_pairs = sorted(set(year_month_pairs))
    print(f"Unique year-month pairs: {unique_pairs}")

    n_months = len(unique_pairs)
    lat_dim = data.shape[2]
    lon_dim = data.shape[3]

    # norm_sfc: (n_months, 2, 1, lat, lon)
    norm_sfc = np.zeros((n_months, 2, 1, lat_dim, lon_dim), dtype=np.float32)
    global_norm_sfc = np.zeros((n_months, 2, 1), dtype=np.float32)

    for idx, (year, month) in enumerate(unique_pairs):
        mask = (years == year) & (months == month)
        month_data = data[mask, 0, :, :]
        print(f"Year {year} Month {month}: month_data shape: {month_data.shape}")
        norm_sfc[idx, 0, 0] = month_data.mean(axis=0).compute()
        norm_sfc[idx, 1, 0] = month_data.std(axis=0).compute()
        global_norm_sfc[idx, 0, 0] = month_data.mean().compute()
        global_norm_sfc[idx, 1, 0] = month_data.std().compute()
        print(f"norm_sfc[{idx}] mean shape: {norm_sfc[idx, 0, 0].shape}, std shape: {norm_sfc[idx, 1, 0].shape}")
        print(f"global_norm_sfc[{idx}] mean: {global_norm_sfc[idx, 0, 0]}, std: {global_norm_sfc[idx, 1, 0]}")

    print("Final norm_sfc shape:", norm_sfc.shape)
    print("Final global_norm_sfc shape:", global_norm_sfc.shape)

    # Save these as new variables in the t2m Zarr store
    t2m_ds['norm_sfc'] = (('month', 'stat', 'field', 'latitude', 'longitude'), norm_sfc)
    t2m_ds['global_norm_sfc'] = (('month', 'stat', 'field'), global_norm_sfc)
    t2m_ds.to_zarr(t2m_path, mode="a")
    print("Added norm_sfc and global_norm_sfc to t2m Zarr.")

def add_surface_norms_to_Akil_t2m():
    # t2m_path = "/work/ab1412/atmorep/data/t2m/era5_y2010_2021_res025_chunk8.zarr"
    t2m_path = "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr"
    t2m_ds = xr.open_zarr(t2m_path, consolidated=False)

    # Assume data_sfc shape: (time, lat, lon) or (time, 1, lat, lon)
    data = t2m_ds['data_sfc']
    if data.ndim == 3:
        # Add singleton dimension at axis=1
        data = data.expand_dims(dim={'field': [0]})
        data = data.transpose('time', 'field', 'lat', 'lon')
    data = data.chunk({'time': 100})
    data = data.assign_coords(time=t2m_ds['time'])

    # Get latitude values and indices for index 0-70
    lat_name = 'lat' if 'lats' in data.dims else 'latitude'
    lat_vals = data[lat_name].values
    #idx_71N = np.where(lat_vals >= 71)[0] 

    print(f"Latitude coordinate name: {lat_name}")
    print(f"Latitude values shape: {lat_vals.shape}")
    print(f"First few lat values: {lat_vals[:10]}")
    print(f"Last few lat values: {lat_vals[-10:]}")
    print(f"Data shape: {data.shape}")

    # **CRITICAL FIX: Use latitude INDICES 0-70, not lat values ≥71**
    # Akil's valid data is only in the first 71 latitude indices (0-70)
    valid_lat_indices = slice(0, 71)  # Indices 0 through 70 (71 total)
    
    print(f"Using latitude indices 0-70 for Arctic region")
    print(f"This corresponds to lat values: {lat_vals[0]:.2f} to {lat_vals[70]:.2f}")

    sample_valid = data[0, 0, 0:71, 0:100].values  # First 71 lats
    sample_padded = data[0, 0, 71:81, 0].values  # Next 10 lats (should be zeros)

    print(f"\nData verification:")
    print(f"Valid region (lat idx 0-70) sample mean: {np.nanmean(sample_valid):.2f}")
    print(f"Valid region (lat idx 0-70) sample range: {np.nanmin(sample_valid):.2f} to {np.nanmax(sample_valid):.2f}")
    print(f"Padded region: {sample_padded} at 0 longitude")

    # Calculate year and month indices for each time step
    time = t2m_ds['time'].values  # shape: (time,)
    time_pd = pd.to_datetime(time)
    years = time_pd.year
    months = time_pd.month

    # Build (year, month) pairs
    year_month_pairs = [(y, m) for y, m in zip(years, months)]
    unique_pairs = sorted(set(year_month_pairs))
    print(f"Unique year-month pairs: {unique_pairs}")

    n_months = len(unique_pairs)
    lat_dim = data.shape[2]
    lon_dim = data.shape[3]

    # norm_sfc: (n_months, 2, 1, lat, lon)
    norm_sfc = np.zeros((n_months, 2, 1, lat_dim, lon_dim), dtype=np.float32)
    global_norm_sfc = np.zeros((n_months, 2, 1), dtype=np.float32)

    print(f"Calculating norms for {n_months} months...")
    print(f"Norm arrays shape: {norm_sfc.shape}")

    for idx, (year, month) in enumerate(unique_pairs):
        mask = (years == year) & (months == month)
        month_data = data[mask, 0, :, :]  # (n_times, lat, lon)
        
        print(f"Year {year} Month {month:2d}: {mask.sum()} timesteps, month_data shape: {month_data.shape}")

        # **CRITICAL: Only calculate norms for VALID latitude indices (0-70)**
        # Extract only the valid Arctic region
        valid_month_data = month_data[:, valid_lat_indices, :]  # (n_times, 71, lon)
        
        if valid_month_data.size == 0:
            print(f"  ⚠️  No valid data for {year}-{month}")
            continue
            
        # Calculate stats only for valid region
        mean_valid = valid_month_data.mean(axis=0).compute()  # (71, lon)
        std_valid = valid_month_data.std(axis=0).compute()    # (71, lon)

        # Place results only in valid latitude indices
        norm_sfc[idx, 0, 0, valid_lat_indices, :] = mean_valid
        norm_sfc[idx, 1, 0, valid_lat_indices, :] = std_valid
        
        # All other latitudes (71+) remain zero - DO NOT include padded zeros in global norms
        global_mean = valid_month_data.mean().compute()
        global_std = valid_month_data.std().compute()
        global_norm_sfc[idx, 0, 0] = global_mean
        global_norm_sfc[idx, 1, 0] = global_std

        print(f"  Valid region mean: {global_mean:.2f}K, std: {global_std:.2f}K")
        print(f"  Norm shape for valid region: {mean_valid.shape}")
        
        # **VERIFICATION: Check that we're not including zeros**
        if idx == 0:  # First month verification
            sample_norms = norm_sfc[idx, 0, 0, 0:5, 0:5]  # Sample 5x5
            print(f"  Sample norm means (first month): {sample_norms}")
            if np.any(sample_norms < 200):  # Temperatures should be >200K in Arctic
                print(f"  ⚠️  WARNING: Suspiciously low norm values detected!")

    # **FINAL VERIFICATION**
    print(f"\n=== Final Verification ===")
    print(f"Norm_sfc shape: {norm_sfc.shape}")
    print(f"Global_norm_sfc shape: {global_norm_sfc.shape}")

    # Check valid vs padded regions in final norms
    valid_norms = norm_sfc[:, 0, 0, 0:71, :]  # Valid region norms
    if lat_dim > 71:
        padded_norms = norm_sfc[:, 0, 0, 71:, :]  # Padded region norms

        print(f"Valid region norm stats:")
        print(f"  Mean range: {np.nanmin(valid_norms):.2f} to {np.nanmax(valid_norms):.2f}K")
        print(f"  Non-zero values: {np.count_nonzero(valid_norms)}")
        
        print(f"Padded region norm stats:")
        print(f"  Mean range: {np.nanmin(padded_norms):.2f} to {np.nanmax(padded_norms):.2f}K")
        print(f"  Non-zero values: {np.count_nonzero(padded_norms)} (should be 0)")
        
        if np.count_nonzero(padded_norms) > 0:
            print("⚠️  WARNING: Padded region has non-zero norms!")
        else:
            print("✓ Padded region correctly has zero norms")

    # Save results
    t2m_ds['norm_sfc'] = (('month', 'stat', 'field', lat_name, 'longitude'), norm_sfc)
    t2m_ds['global_norm_sfc'] = (('month', 'stat', 'field'), global_norm_sfc)
    t2m_ds.to_zarr(t2m_path, mode="a")
    print(f"\n✓ Added norm_sfc and global_norm_sfc to {t2m_path}")
    print(f"✓ Normalization calculated ONLY for valid Arctic region (lat indices 0-70)")

def add_t2m_into_era5(): 
# Paths
    main_path = "/work/ab1412/atmorep/data/era5_y2010_2020_res25.zarr"
    t2m_path = "/work/ab1412/atmorep/data/t2m/era5_t2m_y2010_2021_res025_chunk8.zarr"
    output_path = "/work/ab1412/atmorep/data/era5_y2010_2020_with_t2m_res25.zarr"

    # Open datasets
    ds_main = xr.open_zarr(main_path, consolidated=False)
    ds_t2m = xr.open_zarr(t2m_path, consolidated=False)

    # Combine data_sfc along the field dimension
    # ds_main['data_sfc']: (time, field, lat, lon)
    # ds_t2m['data_sfc']: (time, 1, lat, lon)
    data_sfc_combined = xr.concat([ds_main['data_sfc'], ds_t2m['data_sfc']], dim='field')
    print("Combined data_sfc shape:", data_sfc_combined.shape)

    # Combine norm_sfc and global_norm_sfc along the field dimension (axis=2)
    # Open normalization groups as separate datasets
    norm_main = xr.open_zarr(main_path + "/normalization", consolidated=False)
    norm_t2m = xr.open_zarr(t2m_path + "/normalization", consolidated=False)

    # Now you can access norm_sfc and global_norm_sfc
    norm_sfc_combined = xr.concat([norm_main['norm_sfc'], norm_t2m['norm_sfc']], dim=2)
    global_norm_sfc_combined = xr.concat([norm_main['global_norm_sfc'], norm_t2m['global_norm_sfc']], dim=2)
    
    print("Combined norm_sfc shape:", norm_sfc_combined.shape)
    print("Combined global_norm_sfc shape:", global_norm_sfc_combined.shape)

    # Create a new dataset for output
    ds_out = ds_main.copy(deep=True)
    ds_out['data_sfc'] = data_sfc_combined
    ds_out['normalization/norm_sfc'] = norm_sfc_combined
    ds_out['normalization/global_norm_sfc'] = global_norm_sfc_combined

    # Update fields_sfc attribute if present
    fields_sfc = ds_out.attrs.get('fields_sfc', [])
    if isinstance(fields_sfc, list):
        if 't2m' not in fields_sfc:
            fields_sfc.append('t2m')
        ds_out.attrs['fields_sfc'] = fields_sfc

    # Write to new Zarr store, chunked for efficiency
    ds_out = ds_out.chunk({'time': 1000})
    ds_out.to_zarr(output_path, mode="w")
    print(f"Combined Zarr written to: {output_path}")

def add_array_dimensions_attrs(zarr_path):

    # Open the Zarr store in append mode
    z = zarr.open(zarr_path, mode='a')

    # Top-level arrays
    if 'data' in z:
        z['data'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'field', 'level','latitude', 'longitude']
    if 'data_sfc' in z:
        z['data_sfc'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'field_sfc', 'latitude', 'longitude']
    if 'lats' in z:
        z['lats'].attrs['_ARRAY_DIMENSIONS'] = ['latitude']
    if 'lons' in z:
        z['lons'].attrs['_ARRAY_DIMENSIONS'] = ['longitude']
    if 'time' in z:
        z['time'].attrs['_ARRAY_DIMENSIONS'] = ['time']

    # Normalization group
    if 'normalization' in z:
        norm = z['normalization']
        if 'norm' in norm:
            norm['norm'].attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'field', 'level', 'latitude', 'longitude']
        if 'norm_sfc' in norm:
            norm['norm_sfc'].attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'field_sfc', 'latitude', 'longitude']
        if 'global_norm' in norm:
            norm['global_norm'].attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'variable', 'level']
        if 'global_norm_sfc' in norm:
            norm['global_norm_sfc'].attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'field_sfc']

    print(f"Added _ARRAY_DIMENSIONS to arrays in {zarr_path}")

def concat_data_sfc_and_norms_inplace(main_path, t2m_path, output_path, chunk_size=1000, replace_field_idx=2):
    """
    Concatenate data_sfc and norm_sfc from t2m_path into main_path.
    This modifies the main_path Zarr in place, adding T2M data to data_sfc and norm_sfc.
    Parameters:
    - main_path: Path to the main Zarr store (ERA5 data).
    - t2m_path: Path to the T2M Zarr store.
    - output_path: Path to save the modified main Zarr store.
    - chunk_size: Size of chunks to process at a time.
    """
    # # Step 1: Copy the entire ERA5 folder to output_path (except sfc/norm arrays)
    # print(f"Copying {main_path} to {output_path}...")
    # print('                     This will take a while, please be patient.')
    # shutil.copytree(main_path, output_path, dirs_exist_ok=True)
    # print(f"Copied {main_path} to {output_path}")

    # Step 2: Open Zarr groups
    z_out = zarr.open(output_path, mode='a')
    z_main = zarr.open(main_path, mode='r')
    z_t2m = zarr.open(t2m_path, mode='r')

    # --- data_sfc ---
    arr_main = z_main['data_sfc']
    arr_t2m = z_t2m['data_sfc']
    assert arr_main.shape[0] == arr_t2m.shape[0]
    assert arr_main.shape[2:] == arr_t2m.shape[2:]

    # Create new data_sfc with extra field dimension
    out_shape = (arr_main.shape[0], arr_main.shape[1] + arr_t2m.shape[1], arr_main.shape[2], arr_main.shape[3])
    out_chunks = (chunk_size, arr_main.shape[1] + arr_t2m.shape[1], arr_main.shape[2], arr_main.shape[3])

    # Remove and recreate data_sfc
    del z_out['data_sfc']
    out_arr = z_out.create_dataset('data_sfc', shape=out_shape, chunks=out_chunks, dtype=arr_main.dtype)

    # Chunked copy/concat
    for i in range(0, arr_main.shape[0], chunk_size):
        i_end = min(i + chunk_size, arr_main.shape[0])
        chunk_main = arr_main[i:i_end]
        chunk_t2m = arr_t2m[i:i_end]
        chunk_out = np.concatenate([chunk_main, chunk_t2m], axis=1)
        out_arr[i:i_end] = chunk_out
        print(f"Written data_sfc chunk {i}:{i_end}")

    out_arr.attrs.update(arr_main.attrs)
    out_arr.attrs['_ARRAY_DIMENSIONS'] = ['time', 'field_sfc', 'latitude', 'longitude']

 # --- norm_sfc (REPLACE instead of concatenate) ---
    norm_main = z_main['normalization']['norm_sfc']
    norm_t2m = z_t2m['normalization']['norm_sfc']
    
    print(f"norm_main shape: {norm_main.shape}")
    print(f"norm_t2m shape: {norm_t2m.shape}")
    print(f"Replacing field index {replace_field_idx} in norm_sfc")
    
    # Check that the field index exists in main norm array
    if replace_field_idx >= norm_main.shape[2]:
        raise ValueError(f"Field index {replace_field_idx} is out of bounds for norm_main shape {norm_main.shape}")
    
    # **REPLACE: Copy main norm array, then overwrite specific field**
    norm_out = z_out['normalization']['norm_sfc']  # Already exists from copy
    
    # Replace field by field to avoid memory issues
    for m in range(norm_main.shape[0]):  # For each month
        print(f"Replacing norm_sfc month {m}, field {replace_field_idx}")
        
        # Load T2M norm for this month (shape: (2, 1, lat, lon))
        t2m_month_norm = np.array(norm_t2m[m])  # (stat, field=1, lat, lon)
        
        # Replace the specific field in the output array
        # norm_out[month, stat, field, lat, lon]
        norm_out[m, :, replace_field_idx, :, :] = t2m_month_norm[:, 0, :, :]  # Take field 0 from T2M

    norm_out.attrs.update(norm_main.attrs)
    norm_out.attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'field', 'latitude', 'longitude']

    # --- global_norm_sfc (REPLACE instead of concatenate) ---
    gnorm_main = z_main['normalization']['global_norm_sfc']
    gnorm_t2m = z_t2m['normalization']['global_norm_sfc']
    
    print(f"gnorm_main shape: {gnorm_main.shape}")
    print(f"gnorm_t2m shape: {gnorm_t2m.shape}")
    print(f"Replacing field index {replace_field_idx} in global_norm_sfc")
    
    # Check that the field index exists in main global norm array
    if replace_field_idx >= gnorm_main.shape[2]:
        raise ValueError(f"Field index {replace_field_idx} is out of bounds for gnorm_main shape {gnorm_main.shape}")
    
    # **REPLACE: Copy main global norm array, then overwrite specific field**
    gnorm_out = z_out['normalization']['global_norm_sfc']  # Already exists from copy
    
    # Replace field by field
    for m in range(gnorm_main.shape[0]):  # For each month
        print(f"Replacing global_norm_sfc month {m}, field {replace_field_idx}")
        
        # Load T2M global norm for this month (shape: (2, 1))
        t2m_month_gnorm = np.array(gnorm_t2m[m])  # (stat, field=1)
        
        # Replace the specific field in the output array
        # gnorm_out[month, stat, field]
        gnorm_out[m, :, replace_field_idx] = t2m_month_gnorm[:, 0]  # Take field 0 from T2M

    gnorm_out.attrs.update(gnorm_main.attrs)
    gnorm_out.attrs['_ARRAY_DIMENSIONS'] = ['month', 'stat', 'field']

    print("Finished updating data_sfc (concatenated) and replacing norm_sfc/global_norm_sfc at specified field index.")

def fast_rsync_copy(src, dst):
    """
    Fast, resumable copy of large directories using rsync.
    Args:
        src (str): Source directory (can be a Zarr store).
        dst (str): Destination directory.
    """
    cmd = [
        "rsync",
        "-av",
        "--info=progress2",
        "--partial",
        "--append-verify",
        src.rstrip("/") + "/",  # Ensure trailing slash for directory contents
        dst.rstrip("/") + "/"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
   
def create_zarr_subset(src_path, dst_path, n_samples=100, keys=None):
    """
    Create a small Zarr subset with the first n_samples along the time axis.
    Only copies arrays/groups that exist and are non-empty.
    If keys is provided, only those arrays/groups are copied.
    """
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    zsrc = zarr.open(src_path, mode='r')
    zdst = zarr.open(dst_path, mode='w')

    # Helper to copy an array if it exists and is non-empty
    def copy_array(src_group, dst_group, key, n_samples):
        if key in src_group:
            print(f"Copying array '{key}' from source to destination")
            arr = src_group[key]
            if hasattr(arr, 'shape') and arr.shape and arr.shape[0] > 0 and arr.chunks and arr.chunks[0] > 0:
                # Only subset arrays with time as the first dimension
                data = arr[:n_samples] if arr.shape[0] >= n_samples else arr[:]
                chunks = tuple(min(c, n_samples) if i == 0 else c for i, c in enumerate(arr.chunks))
                dst_group.create_dataset(
                    name=key,
                    data=data,
                    chunks=chunks,
                    dtype=arr.dtype
                )
                dst_group[key].attrs.update(arr.attrs)

    # If keys is None, use all array_keys
    if keys is None:
        keys = zsrc.array_keys()

    for key in keys:
        # Handle nested keys like normalization/norm_sfc
        if "/" in key:
            group_name, arr_name = key.split("/", 1)
            if group_name in zsrc:
                if group_name not in zdst:
                    zdst.create_group(group_name)
                copy_array(zsrc[group_name], zdst[group_name], arr_name, n_samples)
        else:
            copy_array(zsrc, zdst, key, n_samples)

    # Copy group attributes
    zdst.attrs.update(zsrc.attrs)
    print(f"Created subset: {dst_path}")

def test_concat_data_sfc_and_norms_inplace(era5_path, t2m_path, test_dir, n_samples=100):
    """
    Create small test Zarrs and run the concat function.
    """
    t2m_keys = [
    "data_sfc",
    "normalization/norm_sfc",
    "normalization/global_norm_sfc"
    ]

    era5_keys = [
    "data",
    "data_sfc",
    "lats",
    "lons",
    "time",
    "normalization/norm",
    "normalization/norm_sfc",
    "normalization/global_norm",
    "normalization/global_norm_sfc"
    ]

    era5_test = os.path.join(test_dir, "era5_test.zarr")
    t2m_test = os.path.join(test_dir, "t2m_test.zarr")
    out_test = os.path.join(test_dir, "era5_with_t2m_test.zarr")
    #create_zarr_subset(era5_path, era5_test, n_samples, keys=era5_keys)
    #create_zarr_subset(t2m_path, t2m_test, n_samples, keys=t2m_keys)
    concat_data_sfc_and_norms_inplace(era5_test, t2m_test, out_test, chunk_size=100)
    print(f"Test pipeline completed. Output: {out_test}")

def check_for_nans(
    zarr_path,
    field_idx=2,
    lat_slice=slice(0, 71),
    lon_slice=slice(None),
    time_slice=slice(0, 10),
    print_samples=True,
    norm_field='norm_sfc'
    ):
    """
    Check for NaNs and zeros in data_sfc and normalization arrays for a specific field and region.
    Parameters:
        zarr_path (str): Path to the Zarr store.
        field_idx (int): Index of the field to check (default: 2).
        lat_slice, lon_slice, time_slice: slices for latitude, longitude, and time.
        print_samples (bool): Whether to print a small sample of the data.
        norm_field (str): Which normalization field to check ('norm_sfc' or 'global_norm_sfc').
    """

    z = zarr.open(zarr_path, mode='r')

    # --- data_sfc ---
    data_sfc = z['data_sfc']
    data_subset = data_sfc[time_slice, field_idx, lat_slice, lon_slice]
    print(f"data_sfc subset shape: {data_subset.shape}")
    print("NaNs in data_sfc:", np.isnan(data_subset).any())
    print("Zeros in data_sfc:", (data_subset == 0).any())
    if print_samples:
        print("Sample data_sfc values:", data_subset.flatten()[:10])

    # --- norm_sfc ---
    norm_sfc = z['normalization'][norm_field]
    # norm_sfc shape: (month, stat, field, lat, lon)
    norm_subset = norm_sfc[0:3, :, field_idx, lat_slice, lon_slice]
    print(f"{norm_field} subset shape: {norm_subset.shape}")
    print(f"NaNs in {norm_field}:", np.isnan(norm_subset).any())
    print(f"Zeros in {norm_field}:", (norm_subset == 0).any())
    if print_samples:
        print(f"Sample {norm_field} values:", norm_subset.flatten()[:10])

    # --- global_norm_sfc ---
    if 'global_norm_sfc' in z['normalization']:
        global_norm_sfc = z['normalization']['global_norm_sfc']
        global_norm_subset = global_norm_sfc[0:3, :, field_idx]
        print("global_norm_sfc subset shape:", global_norm_subset.shape)
        print("NaNs in global_norm_sfc:", np.isnan(global_norm_subset).any())
        print("Zeros in global_norm_sfc:", (global_norm_subset == 0).any())
        if print_samples:
            print("Sample global_norm_sfc values:", global_norm_subset.flatten()[:10])

def check_longitude_boundary(zarr_path):
    z = zarr.open(zarr_path, mode='r')
    # --- data_sfc --
    data_sfc = z['data_sfc']
    print(f"data_sfc shape: {data_sfc.shape}")
    # Before interpolation, check if original data is periodic
    print("=== Original Arctic Data Boundary Check ===")
    for i in list([0, 100, 1000, 50000, 100000]):
        sample_data = data_sfc[i, :, :, :]  # First timestep, shape (71, 1280)
        sample_data = np.squeeze(sample_data)  # Remove singleton field dimension if present

        # Check boundary difference across all latitudes
        boundary_diffs = np.abs(sample_data[:, 0] - sample_data[:, -1])  # Shape (71,)
        max_boundary_diff = np.max(boundary_diffs)
        mean_boundary_diff = np.mean(boundary_diffs)

        print(f"Original longitude boundary differences:")
        print(f"  Max difference: {max_boundary_diff:.3f}K")
        print(f"  Mean difference: {mean_boundary_diff:.3f}K")
        #print(f"  Per-latitude diffs: {boundary_diffs}")

        # Show actual values at boundaries
        print(f"First longitude (0°) sample: {sample_data[35, 0]:.2f}K")  # Middle latitude
        print(f"Last longitude (359.72°) sample: {sample_data[35, -1]:.2f}K")

        if max_boundary_diff > 2.0:
            print("⚠️  Original Arctic data has large longitude boundary discontinuities!")
            print("   This suggests the data isn't truly global/periodic")

def update_zarr_metadata():

    output_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr"
    z = zarr.open(output_path, mode='a')

    # Check current metadata
    print("Current fields_sfc:", z.attrs.get('fields_sfc', 'MISSING'))
    print("Current data_sfc shape:", z['data_sfc'].shape)

    # Update metadata to match your 3-field data_sfc
    expected_fields_sfc = ['total_precip', 't2m', 'corrected_t2m']
    z.attrs['fields_sfc'] = expected_fields_sfc

    print("Updated fields_sfc:", z.attrs['fields_sfc'])

def diagnose_missing_norms(zarr_path):
    '''
    Diagnose missing normalization values in a Zarr store. Currently only for first month and 0,0 lat and lon. 
    To Do: check over wider range of lat/lon, and over all months.
    '''
    store = zarr.open(zarr_path, mode='r')
    norm_array = store['normalization/norm']
    
    print(f"Norm array shape: {norm_array.shape}")
    print(f"Fields: {store.attrs['fields']}")
    print(f"Levels: {store.attrs['levels']}")
    
    # Check all field-level combinations
    for field_idx, field_name in enumerate(store.attrs['fields']):
        print(f"\n--- Field {field_idx}: {field_name} ---")
        
        for level_idx, level in enumerate(store.attrs['levels']):
            # Check mean and std for this field-level combo
            mean_val = norm_array[0, 0, field_idx, level_idx, 0, 0]  # month 0, mean
            std_val = norm_array[0, 1, field_idx, level_idx, 0, 0]   # month 0, std
            
            if mean_val == 0 and std_val == 0:
                print(f"  Level {level} (idx {level_idx}): ❌ MISSING (0.0, 0.0)")
            else:
                print(f"  Level {level} (idx {level_idx}): ✓ (mean={mean_val:.6f}, std={std_val:.6f})")
    
    # Check if any field has ALL zeros
    for field_idx, field_name in enumerate(store.attrs['fields']):
        field_norms = norm_array[:, :, field_idx, :, :, :]  # All months, stats, levels, lat, lon
        nonzero_count = np.count_nonzero(field_norms)
        total_count = field_norms.size
        
        if nonzero_count == 0:
            print(f"\n⚠️  Field {field_idx} ({field_name}) has NO normalization data!")
        elif nonzero_count < total_count * 0.1:  # Less than 10% non-zero
            print(f"\n⚠️  Field {field_idx} ({field_name}) has sparse normalization data: {nonzero_count}/{total_count}")
        else:
            print(f"\n✓ Field {field_idx} ({field_name}) has good normalization data: {nonzero_count}/{total_count}")

def replace_zeros_with_nans_in_norm_sfc(zarr_path, field_idx=2, lat_start=71, block_size=16, dry_run=True):
    """
    Replace exact 0 -> NaN in normalization/norm_sfc[:, :, field_idx, lat_start:, :] in-place.
    
    norm_sfc shape: (144, 2, 3, 721, 1440) float32
      - axis 0: months (144)
      - axis 1: mean/std (2)
      - axis 2: fields (3)
      - axis 3: lat (721)
      - axis 4: lon (1440)
    
    Parameters:
    - zarr_path: path to zarr group
    - field_idx: index of field to process (default 2 = corrected_t2m)
    - lat_start: start latitude index (inclusive) to begin replacement (default 71)
    - block_size: number of month slices to process per iteration
    - dry_run: if True, only report counts and do not write
    """
    zp = Path(zarr_path)
    if not zp.exists():
        raise FileNotFoundError(f"{zarr_path} not found")

    mode = 'r' if dry_run else 'r+'
    z = zarr.open_group(str(zp), mode=mode)

    arr_path = 'normalization/norm_sfc'
    if arr_path not in z:
        raise KeyError(f"{arr_path} not found in store")

    arr = z[arr_path]
    shape = arr.shape
    print(f"norm_sfc shape: {shape} dtype: {arr.dtype}")

    if len(shape) != 5:
        raise ValueError(f"expected norm_sfc rank 5, got shape={shape}")

    # validate field_idx
    if field_idx < 0:
        field_idx = shape[2] + field_idx
    if not (0 <= field_idx < shape[2]):
        raise IndexError(f"field_idx {field_idx} out of range (0..{shape[2]-1})")

    # validate lat_start
    if not (0 <= lat_start < shape[3]):
        raise IndexError(f"lat_start {lat_start} out of range (0..{shape[3]-1})")

    # dtype must be floating to store NaNs
    if not np.issubdtype(np.dtype(arr.dtype), np.floating):
        raise TypeError(f"norm_sfc dtype is {arr.dtype}; must be floating to hold NaN.")

    total_checked = 0
    total_zeros = 0
    total_replaced = 0

    print(f"Processing {zarr_path}  norm_sfc  field_idx={field_idx} lat_start={lat_start}  dry_run={dry_run}")

    # iterate over axis 0 (months) in blocks
    n_months = shape[0]
    for m0 in range(0, n_months, block_size):
        m1 = min(n_months, m0 + block_size)
        # slice: [m0:m1, :, field_idx, lat_start:, :]
        sl = (slice(m0, m1), slice(None), field_idx, slice(lat_start, None), slice(None))
        try:
            block = np.asarray(arr[sl])
        except Exception as e:
            print(f"ERROR reading slice {sl}: {e}")
            raise

        total_checked += block.size
        mask = (block == 0)
        nzeros = int(np.count_nonzero(mask))
        total_zeros += nzeros

        if nzeros:
            print(f"months {m0}:{m1}  zeros={nzeros}")
            if not dry_run:
                block = block.copy()
                block[mask] = np.nan
                arr[sl] = block
                total_replaced += nzeros

    print("Finished scanning norm_sfc.")
    print(f"Elements checked: {total_checked:,}")
    print(f"Exact-zero entries found: {total_zeros:,}")
    if not dry_run:
        print(f"Exact-zero entries replaced: {total_replaced:,}")
        try:
            z.store.flush()
        except Exception:
            pass
    else:
        print("Dry-run: no changes written. Rerun with dry_run=False to apply changes.")



if __name__ == "__main__":

    replace_zeros_with_nans_in_norm_sfc("/work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr", 
                                       dry_run=False 
    )

    # /work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr

    #diagnose_missing_norms("/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr")

    #add_array_dimensions_attrs("/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr")
    #check_longitude_boundary("/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr")
    #add_surface_norms_to_Akil_t2m()
    #add_norm_sfc_to_t2m()

    # concat_data_sfc_and_norms_inplace(
    #     "/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr",
    #     "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr",  
    #     "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_t2m.zarr",
    #      chunk_size=100
    #  )

#     test_concat_data_sfc_and_norms_inplace(
#         "/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr", 
#         "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr",  
#         "/work/ab1412/atmorep/data/test_concat_zarrs", 
#     n_samples=100
# )
    # test_concat_data_sfc_and_norms_inplace(
    #     "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr",
    #     "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr",
    #     "/scratch/a/a270277/atmorep/test_concat_zarrs",
    #     n_samples=100
    # )

    #fast_rsync_copy("/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr/", "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr/")
    #fast_rsync_copy("/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr/", "/work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr/")

    # concat_data_sfc_and_norms_inplace(
    #     "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr",
    #     "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr",
    #     "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr",
    #      chunk_size=100
    #  )
    # update_zarr_metadata()

    # examine_zarr.explore_zarr_structure("/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr")

    # fast_rsync_copy("/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr/", "/work/ab1385/a270277/era5_y2010_2020_res25_with_t2m.zarr/"
    # )
        

    # examine_zarr.explore_zarr_structure("/scratch/a/a270277/atmorep/era5_y2010_2020_res25_t2m.zarr")

    #check_for_nans('/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m.zarr')

    
#     test_concat_data_sfc_and_norms_inplace(
#     "/work/ab1412/atmorep/data/era5_y2010_2020_res25_copy.zarr",
#     "/work/ab1412/atmorep/data/t2m/era5_t2m_y2010_2021_res025_chunk8.zarr",
#     "/work/ab1412/atmorep/data/test_concat_zarrs",
#     n_samples=1000
# )

    # concat_data_sfc_and_norms_inplace(
    #     "/work/ab1412/atmorep/data/era5_y2010_2020_res25_copy.zarr",
    #     "/work/ab1412/atmorep/data/t2m/era5_t2m_y2010_2021_res025_chunk8.zarr",
    #     "/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr"
    # )

