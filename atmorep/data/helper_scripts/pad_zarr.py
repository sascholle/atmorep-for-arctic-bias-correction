import zarr
import numpy as np
import os
import shutil
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

# Define paths
source_path = "/scratch/a/a270277/atmorep/data_t2m_Akil.zarr"
output_path = "/scratch/a/a270277/atmorep/data_t2m_Akil_padded_with_recalculated_interpolation.zarr"

def interpolate_longitude_periodic(data, target_lon_size=1440):
    """
    Interpolates longitude with proper periodic boundary conditions for global data.
    Ensures that longitude 360° = 0° (periodic wrapping).
    """
    orig_lon_size = data.shape[-1]
    
    # Original coordinates: 0 to 360 (exclusive)
    orig_lons = np.linspace(0, 360, orig_lon_size, endpoint=False)
    target_lons = np.linspace(0, 360, target_lon_size, endpoint=False)
    
    # Add periodic boundary: duplicate first longitude point at 360°
    # This ensures smooth interpolation across the 0°/360° boundary
    extended_lons = np.concatenate([orig_lons, [360]])
    extended_data = np.concatenate([data, data[..., :1]], axis=-1)  # Wrap first column to end
    
    # Interpolate with periodic boundaries
    interp_func = interp1d(extended_lons, extended_data, axis=-1, kind='linear', 
                          bounds_error=False, fill_value='extrapolate', assume_sorted=True)
    
    interpolated = interp_func(target_lons)
    
    return interpolated

def validate_interpolation(original_data, interpolated_data):
    """
    Validate that the interpolation preserves important properties
    """
    print("=== Interpolation Validation ===")
    
    # Check overall statistics
    orig_mean = np.nanmean(original_data)
    interp_mean = np.nanmean(interpolated_data)
    print(f"Original data mean: {orig_mean:.3f}K")
    print(f"Interpolated mean:  {interp_mean:.3f}K") 
    print(f"Mean difference:    {abs(orig_mean - interp_mean):.6f}K")
    
    # Check temperature range (should be reasonable for T2M)
    orig_min, orig_max = np.nanmin(original_data), np.nanmax(original_data)
    interp_min, interp_max = np.nanmin(interpolated_data), np.nanmax(interpolated_data)
    print(f"Original range: {orig_min:.1f}K to {orig_max:.1f}K")
    print(f"Interpolated range: {interp_min:.1f}K to {interp_max:.1f}K")
    
    # Check for unrealistic temperatures
    if interp_min < 180 or interp_max > 350:
        print("⚠️  WARNING: Interpolated temperatures outside realistic range!")
    
    # Check periodic boundary consistency
    boundary_diff = abs(interpolated_data[..., 0] - interpolated_data[..., -1])
    max_boundary_diff = np.nanmax(boundary_diff)
    print(f"Max boundary difference (0° vs 359.75°): {max_boundary_diff:.3f}K")
    
    if max_boundary_diff > 1.0:  # Should be very small for smooth interpolation
        print("⚠️  WARNING: Large discontinuity at longitude boundary!")
    else:
        print("✓ Longitude boundary is smooth")
    
    return True

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

print(f"Processing shape: {arr_shape}")
print(f"Longitude interpolation: {arr_shape[-1]} → 1440 points")
print(f"Longitude resolution: {360/arr_shape[-1]:.4f}° → {360/1440:.4f}°")

# Target shape
target_shape = (arr_shape[0], 1, 721, 1440)
chunks = (50, 1, 721, 1440)  # Better chunking for this data structure

# Create output Zarr array
z = zarr.open(output_path, mode='w')
out_arr = z.create_dataset('data_sfc', shape=target_shape, chunks=chunks, dtype=arr.dtype)

# Padding config for latitude
pad_top_lat = 0
pad_bottom_lat = 721 - arr_shape[2]

print(f"Latitude padding: +{pad_top_lat} rows at top, +{pad_bottom_lat} rows at bottom")
print(f"Arctic data spans latitude indices 0 to {arr_shape[2]-1} (out of 0-720)")

pad_config = [
    (0, 0),  # time
    (0, 0),  # field dimension
    (pad_top_lat, pad_bottom_lat),  # lat
    (0, 0),  # lon (no padding, handled by interpolation)
]

def process_timestep(t):
    """Process one timestep with periodic longitude interpolation"""
    if arr.ndim == 3:
        data = arr[t][np.newaxis, :, :]  # shape (1, lat, lon)
        data = np.expand_dims(data, axis=0)  # shape (1, 1, lat, lon)
    else:
        data = arr[t:t+1, :, :, :]  # shape (1, 1, lat, lon)
    
    # Interpolate longitude with periodic boundaries
    interp_data = interpolate_longitude_periodic(data, target_lon_size=1440)
    
    # Pad latitude (add zeros for non-Arctic regions)
    padded_data = np.pad(interp_data, pad_config, mode='constant', constant_values=0)
    
    return padded_data

# Validate interpolation on first timestep
print("\n=== Validating interpolation method ===")
if arr.ndim == 3:
    sample_data = arr[0][np.newaxis, :, :]
    sample_data = np.expand_dims(sample_data, axis=0)
else:
    sample_data = arr[0:1, :, :, :]

sample_interp = interpolate_longitude_periodic(sample_data, target_lon_size=1440)
validate_interpolation(sample_data, sample_interp)

# Process all timesteps in parallel
print(f"\nProcessing {arr_shape[0]} timesteps with periodic longitude interpolation...")
results = Parallel(n_jobs=8, verbose=10)(
    delayed(process_timestep)(t) for t in range(arr_shape[0])
)

# Write results to output array
print("Writing interpolated data to output array...")
for t, padded_data in enumerate(results):
    out_arr[t:t+1, :, :, :] = padded_data
    if (t + 1) % 1000 == 0:
        print(f"Written {t+1}/{len(results)} timesteps")

print(f"Final data shape: {out_arr.shape}")

# Copy additional arrays if they exist
print("Copying additional arrays...")
for key in source.keys():
    if key != 'data' and not key.startswith('.'):  # Skip the main data array and hidden files
        try:
            if hasattr(source[key], 'shape'):
                # Copy array data
                if key == 'lons' or key == 'longitude':
                    # Update longitude coordinates for new grid
                    new_lons = np.linspace(0, 360, 1440, endpoint=False)
                    z.create_dataset(key, data=new_lons, chunks=None)
                    print(f"Updated longitude coordinates: {key}")
                elif key == 'lats' or key == 'latitude':
                    # Update latitude coordinates for new grid (if needed)
                    if source[key].shape[0] != 721:
                        # Create new latitude grid
                        new_lats = np.linspace(90, -90, 721, endpoint=True)
                        z.create_dataset(key, data=new_lats, chunks=None)
                        print(f"Updated latitude coordinates: {key}")
                    else:
                        z.copy_source = source[key][...]
                        z.create_dataset(key, data=z.copy_source, chunks=source[key].chunks)
                        print(f"Copied array: {key}")
                else:
                    # Copy other arrays as-is
                    z.copy_source = source[key][...]
                    z.create_dataset(key, data=z.copy_source, chunks=source[key].chunks)
                    print(f"Copied array: {key}")
            else:
                # Copy attributes/metadata
                z.attrs[key] = source.attrs.get(key, '')
        except Exception as e:
            print(f"Could not copy {key}: {e}")

# Add comprehensive metadata
attrs = {
    'description': 'T2M data interpolated and padded to match ERA5 surface variable dimensions',
    'source_shape': str(arr.shape),
    'target_shape': str(out_arr.shape),
    'longitude_interpolation': f'Periodic interpolation from {arr.shape[-1]} to 1440 points',
    'longitude_resolution': f'{360/1440:.6f} degrees',
    'latitude_padding': f'Added {pad_top_lat} rows at top, {pad_bottom_lat} rows at bottom with zeros',
    'arctic_region': f'Valid data in latitude indices 0-{arr_shape[2]-1}',
    'variable': 't2m_corrected',
    'interpolation_method': 'linear with periodic boundary conditions',
    'boundary_condition': '360° longitude = 0° longitude (periodic)',
    'processing_date': str(np.datetime64('now'))
}
z.attrs.update(attrs)

# Final validation
print("\n=== Final Validation ===")
final_sample = out_arr[0, 0, :72, :]  # Valid Arctic region
padded_sample = out_arr[0, 0, 72:, :] if out_arr.shape[2] > 72 else None

print(f"Valid region stats: mean={np.nanmean(final_sample):.2f}K, std={np.nanstd(final_sample):.2f}K")
if padded_sample is not None:
    padded_nonzero = np.count_nonzero(padded_sample)
    print(f"Padded region: {padded_nonzero} non-zero values (should be 0)")

# Check longitude boundary consistency
boundary_consistency = np.nanmean(np.abs(out_arr[:100, 0, :72, 0] - out_arr[:100, 0, :72, -1]))
print(f"Longitude boundary consistency: {boundary_consistency:.6f}K (should be ~0)")

print("\n✅ Periodic longitude interpolation and padding complete!")
print(f"Original shape: {arr.shape}")
print(f"New shape: {out_arr.shape}")
print(f"Output saved to: {output_path}")