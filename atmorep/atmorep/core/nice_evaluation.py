import xarray as xr
import numpy as np
import os
import time
import subprocess
import sys 

NICE_nc_path = "/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc"
ERA5_path = "/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc"

# Arguments: start_idx, end_idx
start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

NICE_data = xr.open_dataset(NICE_nc_path)
ERA5_data = xr.open_dataset(ERA5_path)
NICE_data = NICE_data.sel(time=ERA5_data.time)
ERA5_T2M = ERA5_data['T2M']
NICE_T2M = NICE_data['air_temperature_2m']
combined = xr.Dataset({'ERA5_T2M': ERA5_T2M, 'NICE_T2M': NICE_T2M}).dropna(dim='time')
NICE_T2M_clean = combined['NICE_T2M']
nice_times = NICE_T2M_clean['time'].values
print("Evaluating the following times:", nice_times.shape) # full ds is 2099 samples long 

for idx in range(start_idx, end_idx):
    t = nice_times[idx]
    dt = np.datetime_as_string(t, unit='h')
    dt_obj = np.datetime64(dt).astype('datetime64[h]').astype(object)
    year, month, day, hour = dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour
    # Call your single-evaluation script with arguments
    subprocess.run([
        "python", "/work/ab1412/atmorep/atmorep/core/nice_evaluation_single.py",
        "--year", str(year),
        "--month", str(month),
        "--day", str(day),
        "--hour", str(hour),
        "--idx", str(idx)
    ])

# for idx, t in enumerate(nice_times):
#     dt = np.datetime_as_string(t, unit='h')
#     dt_obj = np.datetime64(dt).astype('datetime64[h]').astype(object)
#     year, month, day, hour = dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour

#     # Call your single-evaluation script with arguments
#     subprocess.run([
#         "python", "/work/ab1412/atmorep/atmorep/core/nice_evaluation_single.py",
#         "--year", str(year),
#         "--month", str(month),
#         "--day", str(day),
#         "--hour", str(hour),
#         "--idx", str(idx)
#     ])

'''
source /work/ab1412/atmorep/pyenv/bin/activate
python /work/ab1412/atmorep/atmorep/core/nice_evaluation.py
'''
