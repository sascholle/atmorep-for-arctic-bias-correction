from py_compile import main
import xarray as xr  
import zarr
import numpy as np
import re
import sys
from pathlib import Path
import pandas as pd



def extract_wandb_ids(output_ids, search_dir="/work/ab1412/atmorep/output", out_path=None):
    """
    Scan .txt files under search_dir for lines like:
      0: Wandb run: atmorep-<modelid>-<outputid>
    where <outputid> is one of output_ids (list of str/int). Return unique list of matches.
    If out_path provided, save one id per line.
    """
    output_ids = [str(x) for x in output_ids]
    ids_re = "|".join(re.escape(x) for x in output_ids)
    # match prefix exactly as shown, capture full atmorep-...-<outputid>
    pattern = re.compile(r'0:\s*Wandb run:\s*atmorep-([A-Za-z0-9_-]+)-(' + ids_re + r')', re.IGNORECASE)

    search_dir = Path(search_dir)
    matches = []
    txt_files = list(search_dir.rglob("*.txt"))

    for p in txt_files:
        name = str(p.name)
        if not any(oid in name for oid in output_ids):
            # still read if you want full search; skip for speed
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = pattern.search(line)
                    if m:
                        matches.append(m.group(1))  # the full model id
        except Exception:
            continue

    result = set(matches)
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(result)  + ("\n" if result else ""))

    print(f"Found {len(result)} ids. Saved to {out_path}" if out_path else f"Found {len(result)} ids.")

    return result

def match_nice_era5_atmorep(model_ids):
    # Read Datasets 
    ERA5_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc")
    NICE_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc")
    # Select the same time range from N-ICE
    NICE_data = NICE_data.sel(time=ERA5_data.time)
    # Now extract variables
    ERA5_T2M = ERA5_data['T2M']
    NICE_T2M = NICE_data['air_temperature_2m']
    # Drop NaNs together
    combined = xr.Dataset({'ERA5_T2M': ERA5_T2M, 'NICE_T2M': NICE_T2M}).dropna(dim='time')
    ERA5_T2M_clean = combined['ERA5_T2M'] #.values
    NICE_T2M_clean = combined['NICE_T2M'] #.values

    # create a dataset of matched values
    all_atmorep_times = set()
    times_outside_tolerance = set()
    matched_values = []
    for model_id in model_ids: 
        store = zarr.ZipStore( f'id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
        if not Path(store.path).exists():   
            print(f"Zarr store not found for model {model_id}, skipping.")
            continue
        ds = zarr.open(store=store, mode ='r')

        # for every sample in the zarr, match to NICE times / locations: 
        for i in range(3):
            if f'corrected_t2m/sample={i:05d}/data' not in ds:
                print(f"Sample {i} not found in model {model_id}, skipping.")
                continue
            atmorep_corrected_T2M = ds[ f'corrected_t2m/sample={i:05d}/data' ][:]
            #print(f"Processing model {model_id}, sample {i}, corrected_T2M shape: {atmorep_corrected_T2M.shape}")
            atmorep_time = ds[ f'corrected_t2m/sample={i:05d}/datetime' ][:]  # check name
            all_atmorep_times.add(atmorep_time[0])
            # for every time in NICE time, match to Atmorep time
            for t in ERA5_T2M_clean.time.values:
                if t == atmorep_time[0]: # first timestep in a window
                    # find nearest lat/lon
                    NICE_lat = NICE_T2M_clean['lat'].sel(time=t).values
                    NICE_lon = NICE_T2M_clean['lon'].sel(time=t).values
                    # find nearest grid point in Atmorep
                    atmorep_lats = ds[ f'corrected_t2m/sample={i:05d}/lat' ][:]
                    atmorep_lons = ds[ f'corrected_t2m/sample={i:05d}/lon' ][:]  # check name
                    # find a lat and lon that match within some tolerance of 0.125 degrees (Atmorep is 0.25 deg resolution)
                    lat_idx = int(np.argmin(np.abs(atmorep_lats - NICE_lat)))
                    lon_idx = int(np.argmin(np.abs(atmorep_lons - NICE_lon)))
                    if abs(atmorep_lats[lat_idx] - NICE_lat) < 0.125 and abs(atmorep_lons[lon_idx] - NICE_lon) < 0.125:
                        atmorep_value = float(atmorep_corrected_T2M[0, 2, lat_idx, lon_idx])
                        nice_value = float(NICE_T2M_clean.sel(time=t).values)
                        era5_value = float(ERA5_T2M_clean.sel(time=t).values)
                        print(f"    Atmorep value: {atmorep_value}, NICE value: {nice_value}, ERA5 value: {era5_value}")
                        matched_values.append({
                            'model_id': model_id,
                            'sample': i,
                            'time': t,
                            'lat': float(atmorep_lats[lat_idx]),
                            'lon': float(atmorep_lons[lon_idx]),
                            'atmorep_value': atmorep_value,
                            'nice_value': nice_value,
                            'era5_value': era5_value
                        })
                    else:
                        times_outside_tolerance.add(t)
                    
    # put all values into a dataset for comparison
    print(f"Total matched values: {len(matched_values)}")
    print(matched_values[:3])  # print first 3 matched values
    
    df = pd.DataFrame(matched_values)
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])
        ds_out = df.set_index('time').to_xarray()
        nc_out = Path('/work/ab1412/atmorep/results/atmorep_nice_matched_valuesROUND7first_timestep.nc')
        ds_out.to_netcdf(nc_out)
        print(f"Saved matched values to {nc_out}")

    # After the loop:
    times_not_in_atmorep = set(ERA5_T2M_clean.time.values) - all_atmorep_times
    print(f"All times in atmorep: {len(all_atmorep_times)}")
    print(f"Times missing from atmorep output: {len(times_not_in_atmorep)}")
    print(f"Times rejected due to lat/lon tolerance: {len(times_outside_tolerance)}")

    matched_times = set(m['time'] for m in matched_values)
    truly_lost = times_outside_tolerance - matched_times
    print(f"Timestamps that NEVER passed tolerance: {len(truly_lost)}")

def calculate_rmse(matched_nc_path):
    ds = xr.open_dataset(matched_nc_path)
    atmorep_values = ds['atmorep_value'].values
    nice_values = ds['nice_value'].values
    era5_values = ds['era5_value'].values

    print(f"VALUE CHECKING:")
    print(f"Atmorep values range: {atmorep_values.min():.2f} - {atmorep_values.max():.2f} K")
    print(f"NICE values range: {nice_values.min():.2f} - {nice_values.max():.2f} K")
    print(f"ERA5 values range: {era5_values.min():.2f} - {era5_values.max():.2f} K")
    print(f"Number of matched values: {len(era5_values)}")
    time_values = ds['time'].values
    print(f"Time range: {pd.to_datetime(time_values.min())} - {pd.to_datetime(time_values.max())}")

    rmse_atmorep_nice = np.sqrt(np.mean((atmorep_values - nice_values) ** 2))
    rmse_era5_nice = np.sqrt(np.mean((era5_values - nice_values) ** 2))

    print(f"RMSE between Atmorep and NICE: {rmse_atmorep_nice:.4f}")
    print(f"RMSE between ERA5 and NICE: {rmse_era5_nice:.4f}")

if __name__ == "__main__":

    #output_ids = [20466838, 20466837, 20466836, 20466835, 20466834, 20466833, 20466832, 20466831, 20466830, 20466829, 20466828, 20466827, 20466826, 20466825, 20466824, 20466823, 20466822, 20466821, 20466820, 20466819, 20466818, 20466817, 20466816, 20466815, 20466814, 20466813, 20466808, 20466810, 20466811, 20466807]
    #first eval straight from Bert 
    #output_ids = [21658205, 21658204, 21658203, 21658202, 21658201, 21658200, 21658199, 21658198, 21658197, 21658196, 21658195, 21658194, 21658193, 21658192, 21658191, 21658190, 21658189, 21658188, 21658187, 21658186, 21658185, 21658184, 21658183, 21658182, 21658181, 21658180, 21658176, 21658177, 21658178, 21658179]
    #second eval with forecasting fine-tuning 
    #output_ids = [23395374,23395373,23395372,23395371,23395370,23395369,23395368,23395367,23395366,23395364,23395363,23395362,23395361,23395360,23395359,23395358,23395357,23395356,23395355,23395354,23395353,23395352,23395351,23395350,23395349,23395348,23395344,23395345,23395346,23395347]
    #ids = extract_wandb_ids(output_ids)
    #match_nice_era5_atmorep(ids)
    matched_nc_path = '/work/ab1412/atmorep/results/atmorep_nice_matched_valuesROUND7.nc'
    calculate_rmse(matched_nc_path)

    print(f"CHECK ORGINAL DATASETS:")
    # Read Datasets 
    ERA5_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc")
    NICE_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc")
    # Select the same time range from N-ICE
    NICE_data = NICE_data.sel(time=ERA5_data.time)
    # Now extract variables
    ERA5_T2M = ERA5_data['T2M']
    NICE_T2M = NICE_data['air_temperature_2m']
    # Drop NaNs together
    combined = xr.Dataset({'ERA5_T2M': ERA5_T2M, 'NICE_T2M': NICE_T2M}).dropna(dim='time')
    ERA5_T2M_clean = combined['ERA5_T2M'] #.values
    NICE_T2M_clean = combined['NICE_T2M'] #.values

    print(f"ERA5 T2M range: {ERA5_T2M_clean.values.min():.2f} - {ERA5_T2M_clean.values.max():.2f} K")
    print(f"NICE T2M range: {NICE_T2M_clean.values.min():.2f} - {NICE_T2M_clean.values.max():.2f} K")
    print(f"Number of matched points: {len(NICE_T2M_clean)}")