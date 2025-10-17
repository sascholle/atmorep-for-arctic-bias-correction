import zarr 
import numpy as np
import os
import json
import pandas as pd

# source /work/ab1412/atmorep/pyenv/bin/activate

output_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr"
#output_path = "/work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr"
# output_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr/era5_y2010_2020_res25_corrected_t2m_new.zarr"
#output_path = "/scratch/a/a270277/atmorep/test_concat_zarrs/era5_with_t2m_test.zarr"
#output_path = "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr"

# output_path = "/scratch/a/a270277/atmorep/data_t2m_Akil.zarr"
#output_path = "/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr"
#output_path = "/scratch/a/a270277/atmorep/backup_norm_sfcs_of_3_fields_for_corrected_t2m_data.zarr"
#output_path = "/scratch/a/a270277/atmorep/test_concat_zarrs/era5_with_t2m_test.zarr"
#output_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr"

def explore_zarr_structure(zarr_path):
    print(f"Opening target dataset: {zarr_path}")
    output_zarr = zarr.open(zarr_path, mode='r')
    #print("Target arrays:", list(output_zarr.array_keys()))
    print(output_zarr.tree())
    
    # Check what fields are stored
    print("Atmospheric fields (data):", output_zarr.attrs['fields'])
    print("Surface fields (data_sfc):", output_zarr.attrs['fields_sfc'])
    print("Vertical levels:", output_zarr.attrs['levels'])

    # This will show you exactly which field names map to which indices
    for i, field in enumerate(output_zarr.attrs['fields']):
        print(f"data[:, {i}, :, :, :] = {field}")

    for i, field in enumerate(output_zarr.attrs['fields_sfc']):
        print(f"data_sfc[:, {i}, :, :] = {field}")

    #print(f"data_sfc: {output_zarr['data_sfc'][:20, :3, 0, 0]}")  # Adjust axes as needed
    print(f"normalization/norm: {output_zarr['normalization/norm'][0, 1, :, :, 0, 0]}")  # Adjust axes as needed

    if 'data' in output_zarr.array_keys():
        main_data = output_zarr['data']
        expected_shape = (main_data.shape[0], main_data.shape[1], main_data.shape[2])
        print(f"Main data shape: {main_data.shape}")
        """Explore any Zarr structure regardless of its organization"""
        print(f"\nExploring Zarr at: {zarr_path}")
        print("=" * 50)
    
    try:
        # Open the zarr store
        store = zarr.open(zarr_path, mode='r')
        data = store['normalization/norm_sfc'][0, :2, 0, :5, :5]  # Larger slice
    
        # for var_idx in range(data.shape[0]):
        #     print(f"\n--- Variable {var_idx} ---")
        #     df = pd.DataFrame(
        #         data[var_idx, :, :], 
        #         index=[f"lat_{i}" for i in range(data.shape[1])],
        #         columns=[f"lon_{i}" for i in range(data.shape[2])]
        #     )
        #     print(df)
        
        # Check what type of object we have
        if isinstance(store, zarr.hierarchy.Group):
            print("This is a Zarr Group (directory/folder structure)")
            print(f"Keys at root level: {list(store.keys())}")
            #explore_group(store)
        else:
            print("This is a direct Zarr Array")
            print(f"Shape: {store.shape}")
            print(f"Dtype: {store.dtype}")
            print(f"Chunks: {store.chunks}")
            print(f"Sample data: {store[-3:]}") # Sample last 3 entries

    except Exception as e:
        print(f"Error exploring Zarr: {str(e)}")

def explore_group(group, prefix=""):
    """Recursively explore a Zarr group"""
    for key in group.keys():
        item = group[key]
        if isinstance(item, zarr.hierarchy.Group):
            print(f"{prefix}Group: {key}/")
            #explore_group(item, prefix + "  ")
        else:
            try:
                print(f"{prefix}Array: {key}")
                print(f"{prefix}  Shape: {item.shape}")
                print(f"{prefix}  Dtype: {item.dtype}")
                print(f"{prefix}  Chunks: {item.chunks}")
                # Try to sample a bit of data
                if len(item.shape) > 0 and item.shape[0] > 0:
                    sample_indices = tuple(slice(None, 2) for s in item.shape) # sample last 3 entries in each dimension
                    print(f"{prefix}  Sample data: {item[sample_indices]}")
                else:
                    print(f"{prefix}  Sample data: [] (empty array)")
            except Exception as e:
                print(f"{prefix}  Error reading array details: {str(e)}")

def check_nonzero(array_name):
    if array_name in store:
        arr = store[array_name]
        # Check in chunks to avoid memory issues
        for idx in range(arr.shape[0]):
            chunk = arr[idx]
            if (chunk == 0).any():
                print(f"Zero data found in '{array_name}' at index {idx}: {chunk}")
                return True
        print(f"All data in '{array_name}' is non-zero.")
        return False
    else:
        print(f"Array '{array_name}' not found in Zarr store.")
        return False

if __name__ == "__main__":
    explore_zarr_structure(output_path)

    #store = zarr.open(output_path, mode='r')
    #check_nonzero('data')
    #check_nonzero('data_sfc')