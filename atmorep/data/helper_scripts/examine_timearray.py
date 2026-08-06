import zarr
import numpy as np
from datetime import datetime, timedelta

# Open the zarr root group
root = zarr.open('/work/ab1412/atmorep/data/era5_y2010_2020_res25.zarr', mode='r')

# Print available arrays/groups at the root level
print("Available keys:", list(root.keys()))

# Try to access and examine the time array if it exists
if 'time' in root:
    time_array = root['time']
    print("\nTime array info:")
    print("Shape:", time_array.shape)
    print("Data type:", time_array.dtype)
    
    # Print a few values
    print("First 5 values:", time_array[:5])
    print("Last 5 values:", time_array[-5:])
    
    # If there are attributes, print them
    if hasattr(time_array, 'attrs') and time_array.attrs:
        print("\nTime array attributes:")
        for key, value in time_array.attrs.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo attributes found in time array")
    
    # Try to interpret time values - need to know the reference date
    # Common references: hours since 1900-01-01, seconds since 1970-01-01, etc.
    # Let's try a few common ones
    
    print("\nPossible interpretations:")
    
    # If they look like Unix timestamps (seconds since 1970-01-01)
    if time_array.dtype in [np.int32, np.int64, np.float32, np.float64] and time_array[0] > 10000000:  # Likely unix timestamp
        try:
            times = [datetime.fromtimestamp(t) for t in time_array[:5]]
            print("As Unix timestamps (seconds since 1970-01-01):")
            print(times)
        except:
            print("Not Unix timestamps")
    
    # If they look like hours since 1900-01-01 (common for ERA5)
    if time_array.dtype in [np.float32, np.float64]:
        ref_date = datetime(1900, 1, 1)
        try:
            times = [ref_date + timedelta(hours=float(t)) for t in time_array[:5]]
            print("As hours since 1900-01-01:")
            print(times)
        except:
            print("Not hours since 1900-01-01")
else:
    print("No 'time' array found")

# Check other potential date-related arrays
for key in root.keys():
    if any(time_related in key.lower() for time_related in ['date', 'time', 'year', 'month', 'day']):
        print(f"\nExamining {key}:")
        array = root[key]
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print("First 5 values:", array[:5])