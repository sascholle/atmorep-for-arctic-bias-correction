import numpy as np
import zarr
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import gc  # For garbage collection

def calculate_normalization_for_t2m():
    """
    Calculate proper normalization for T2M data and update normalization arrays,
    processing data in chunks to avoid memory issues.
    """
    # Path to the zarr file
    source_zarr_path = '/work/ab1412/atmorep/data/temperature/ml137/era5_y2021_res025_chunk8_with_t2m.zarr'
    
    print(f"Opening zarr dataset from {source_zarr_path}")
    source_zarr = zarr.open(source_zarr_path, mode='r+')
    
    # Verify T2M is in the dataset
    fields_sfc = source_zarr.attrs['fields_sfc']
    if 't2m' not in fields_sfc:
        raise ValueError(f"T2M field not found in fields_sfc: {fields_sfc}")
    
    # Find the index of t2m
    t2m_idx = fields_sfc.index('t2m')
    print(f"T2M found at index {t2m_idx} in fields_sfc")
    
    # Create backup
    backup_dir = Path("/work/ab1412/atmorep/data/temperature/ml137/normalization_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Get current norm arrays and verify shapes
    norm_sfc = source_zarr['normalization/norm_sfc']
    global_norm_sfc = source_zarr['normalization/global_norm_sfc']
    
    print(f"Current norm_sfc shape: {norm_sfc.shape}")
    print(f"Current global_norm_sfc shape: {global_norm_sfc.shape}")
    
    # Save backups
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(backup_dir / f"norm_sfc_backup_{backup_time}.npy", norm_sfc[:])
    np.save(backup_dir / f"global_norm_sfc_backup_{backup_time}.npy", global_norm_sfc[:])
    
    # Get time values and convert to datetime
    time_values = source_zarr['time'][:]
    time_dates = pd.to_datetime(time_values)
    
    # Get the data shape and verify
    data_sfc = source_zarr['data_sfc']
    print(f"Data_sfc shape: {data_sfc.shape}")
    
    # Spatial dimensions
    nlat = source_zarr['lats'].shape[0]
    nlon = source_zarr['lons'].shape[0]
    print(f"Spatial dimensions: {nlat} x {nlon}")
    
    # For storing updated norm values
    norm_sfc_data = norm_sfc[:]
    global_norm_sfc_data = global_norm_sfc[:]
    
    year = 2021  # Hardcoded for this dataset
    
    for month_idx, month in enumerate(range(1, 13)):  # 12 months
        print(f"\nProcessing year {year}, month {month}")
        
        # Get indices for this month
        month_mask = (time_dates.month == month) & (time_dates.year == year)
        month_indices = np.where(month_mask)[0]
        
        if len(month_indices) == 0:
            print(f"No data found for month {month}, skipping")
            continue
            
        print(f"Found {len(month_indices)} timesteps")
        
        # Initialize arrays for accumulating statistics
        sum_values = np.zeros((nlat, nlon), dtype=np.float64)
        sum_squares = np.zeros((nlat, nlon), dtype=np.float64)
        count = 0
        
        # Process data in chunks to avoid memory issues
        chunk_size = 24  # Process one day at a time
        num_chunks = len(month_indices) // chunk_size + (1 if len(month_indices) % chunk_size > 0 else 0)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(month_indices))
            
            chunk_indices = month_indices[start_idx:end_idx]
            print(f"Processing chunk {chunk_idx+1}/{num_chunks}, timesteps {start_idx} to {end_idx-1}")
            
            # Process each timestep individually to minimize memory usage
            for i, idx in enumerate(chunk_indices):
                # Extract one timestep of data
                t2m_data = data_sfc[idx, t2m_idx]
                
                # Update running sums
                sum_values += t2m_data
                sum_squares += t2m_data ** 2
                count += 1
                
                # Clean up to free memory
                del t2m_data
                if i % 10 == 0:
                    gc.collect()
        
        # Calculate mean and standard deviation
        local_mean = sum_values / count
        local_std = np.sqrt(np.maximum(0, sum_squares / count - local_mean ** 2))  # Ensure non-negative

        # Use a more reasonable approach for near-zero std devs
        zero_std_mask = local_std < 1e-6
        if np.any(zero_std_mask):
            num_zeros = np.sum(zero_std_mask)
            print(f"Warning: Found {num_zeros} grid points with near-zero std")
            
            # Calculate a typical std from non-zero areas
            typical_std = np.median(local_std[~zero_std_mask])
            print(f"Using median std value of {typical_std:.6f} for these points")
            
            # Use a small fraction of the typical std instead of 1.0
            local_std[zero_std_mask] = typical_std * 0.1
        
        # Calculate global statistics
        global_mean = np.mean(local_mean)
        # For global std, we need to use the original data or a good approximation
        # This is an approximation of the global std:
        global_std = np.sqrt(np.mean(local_std ** 2 + local_mean ** 2) - global_mean ** 2)
        
        # Print stats
        print(f"Global - mean: {global_mean:.4f}, std: {global_std:.4f}")
        print(f"Local - mean range: {np.min(local_mean):.4f} to {np.max(local_mean):.4f}")
        print(f"Local - std range: {np.min(local_std):.4f} to {np.max(local_std):.4f}")
        
        # Update the normalization arrays for this month
        # For global norm: [month, mean/std, field_idx]
        global_norm_sfc_data[month_idx, 0, t2m_idx] = global_mean  # Mean
        global_norm_sfc_data[month_idx, 1, t2m_idx] = global_std   # Std
        
        # For local norm: [month, mean/std, field_idx, lat, lon]
        norm_sfc_data[month_idx, 0, t2m_idx] = local_mean  # Mean per grid point
        norm_sfc_data[month_idx, 1, t2m_idx] = local_std   # Std per grid point
        
        # Clean up
        gc.collect()
    
    # Verify the updated shapes
    print(f"\nUpdated norm_sfc shape: {norm_sfc_data.shape}")
    print(f"Updated global_norm_sfc shape: {global_norm_sfc_data.shape}")
    
    # Write back the updated normalization values
    print("\nWriting updated normalization values back to zarr...")
    norm_sfc[:] = norm_sfc_data
    global_norm_sfc[:] = global_norm_sfc_data
    
    print("Normalization update completed successfully!")

if __name__ == "__main__":
    calculate_normalization_for_t2m()