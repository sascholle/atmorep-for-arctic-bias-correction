from py_compile import main
import xarray as xr  
import zarr
import numpy as np
import re
import sys
from pathlib import Path



def extract_wandb_ids(output_ids, search_dir="/work/ab1412/atmorep/output", out_path="/work/ab1412/atmorep/results/important_result_ids2.txt"):
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

    # Option: only open files that contain one of the output_ids in filename to speed up
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

    result = matches
    print(result)
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(result)

    print(f"Found {len(result)} ids. Saved to {out_path}" if out_path else f"Found {len(result)} ids.")

    return result

def match_nice_era5_atmorep():
    # match NICE, ERA5, ATMOREP model, and AKILS data 

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

    # print("NICE T2M values:", NICE_T2M_clean.shape, NICE_T2M_clean.values[:3], " ... ", NICE_T2M_clean.values[-1:])
    # print("ERA5 lat/lon:", ERA5_T2M_clean['lat'].values, ERA5_T2M_clean['lon'].values)
    # print("Date range:", combined.time.min().values, "to", combined.time.max().values)


    # ERA5 T2M values: (2099,) [265.40625 265.06296 264.92712]  ...  [273.5279]
    # NICE T2M values: (2099,) [268.00564583 267.92406667 266.87071667]  ...  [272.45]
    # ERA5 lat/lon: [83.0443534  83.0443534  83.0443534  ... 80.79614519 80.79614519
    #  80.79614519] [20.25    20.25    20.25    ... 11.53125 11.53125 11.25   ]
    # Date range: 2015-01-21T21:00:00.000000000 to 2015-06-12T23:00:00.000000000

    # Read Atmorep Predictions 
    for model_id in ['cjtlxcuc', 'us1hzz9o', '7xpjmvbk']: #['apadxke2', 'zxm8jvgr', 't3y2w4u8']:
        store = zarr.ZipStore( f'id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
        ds = zarr.open(store=store, mode ='r')

        # for every sample in the zarr, match to NICE times / locations: 
        for i in range(3):
            #print(f"Processing sample={i}")
            atmorep_corrected_T2M = ds[ f'corrected_t2m/sample={i:05d}/data' ][:]
            #print(f"Atmorep T2M shape: {atmorep_corrected_T2M.shape}")
            #print(atmorep_corrected_T2M[0, :, :5, :5])  # print a small slice for inspection
            atmorep_time = ds[ f'corrected_t2m/sample={i:05d}/datetime' ][:]  # check name
            #print(f"Atmorep time shape: {atmorep_time.shape}, values: {atmorep_time}")        
            # for every time in NICE time, match to Atmorep time
            for t in ERA5_T2M_clean.time.values:
                #print(f"Checking NICE time: {t}")
                #print(f"Atmorep time: {atmorep_time}")
                if t == atmorep_time[2]:
                    print(f"Matched time: {t}")
                    # find nearest lat/lon
                    NICE_lat = NICE_T2M_clean['lat'].sel(time=t).values
                    NICE_lon = NICE_T2M_clean['lon'].sel(time=t).values
                    print(f"NICE lat/lon at time {t}: {NICE_lat}, {NICE_lon}")
                    # find nearest grid point in Atmorep
                    atmorep_lats = ds[ f'corrected_t2m/sample={i:05d}/lat' ][:]
                    #print(f"Atmorep lat shape: {atmorep_lats.shape}, values: {atmorep_lats}")
                    atmorep_lons = ds[ f'corrected_t2m/sample={i:05d}/lon' ][:]  # check name
                    #print(f"Atmorep lon shape: {atmorep_lons.shape}, values: {atmorep_lons}")
                    #find a lat and lon that match within some tolerance of 0.125 degrees (Atmorep is 0.25 deg resolution)
                    lat_idx = int(np.argmin(np.abs(atmorep_lats - NICE_lat)))
                    lon_idx = int(np.argmin(np.abs(atmorep_lons - NICE_lon)))
                    if abs(atmorep_lats[lat_idx] - NICE_lat) < 0.125 and abs(atmorep_lons[lon_idx] - NICE_lon) < 0.125:
                        print(f"    Nearest Atmorep grid point lat: {atmorep_lats[lat_idx]}, lon: {atmorep_lons[lon_idx]} which matches NICE lat/lon {NICE_lat}, {NICE_lon}")
                        atmorep_value = atmorep_corrected_T2M[:, 2, lat_idx, lon_idx]  # take the last timestep of 3 timesteps for that window
                    print(f"Atmorep predicted T2M at nearest grid point: {atmorep_value}")
                
        #check if lat lon match era5 / nice


        #put all values into a dataset for comparison


        #calculate MSE

if __name__ == "__main__":
    output_ids = [20466838, 20466837, 20466836, 20466835, 20466834, 20466833, 20466832, 20466831, 20466830, 20466829, 20466828, 20466827, 20466826, 20466825, 20466824, 20466823, 20466822, 20466821, 20466820, 20466819, 20466818, 20466817, 20466816, 20466815, 20466814, 20466813, 20466808, 20466810, 20466811, 20466807]

    ids = extract_wandb_ids(output_ids)
    print("Extracted Wandb IDs:", ids)



# import sys
# from pathlib import Path
# import numpy as np
# import xarray as xr
# import zarr
# from zarr import ZipStore
# import pandas as pd
# import math

# """
# Match NICE / ERA5 T2M observation times & locations to model predictions (corrected_t2m)
# Usage:
#   python match_nice_predictions.py <model_id> [output_dir]
# Example:
#   python match_nice_predictions.py zxm8jvgr /work/ab1412/atmorep/results
# Output:
#   CSV and NetCDF saved to output_dir/matched_predictions_<model_id>.{csv,nc}
# Notes:
#  - The script looks for predictions in: results/id<model_id>/results_id<model_id>_epoch00000_pred.zarr
#  - It searches each sample in the zarr group for sample datetime arrays and lat/lon arrays,
#    finds samples that contain the NICE observation time and selects the nearest grid point
#    in that sample (by lat/lon). If multiple samples contain the same time, the nearest
#    spatial match is used.
# """

# FIELD = "corrected_t2m"
# DEFAULT_ERA5 = "/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc"
# DEFAULT_NICE = "/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc"


# def to_360(lon):
#     """Convert lon to 0..360"""
#     lon = float(lon)
#     if lon < 0:
#         return lon + 360.0
#     return lon


# def angular_distance_deg(lat1, lon1, lat2, lon2):
#     """Approx great-circle angular distance in degrees (approx)."""
#     # Convert to radians and use haversine then convert to degrees
#     r = 6371.0  # km, not used to convert; we'll return degrees via central angle*180/pi
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dl = math.radians(lon2 - lon1)
#     dphi = phi2 - phi1
#     a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
#     central = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
#     return math.degrees(central)


# def find_samples_for_field(group, field):
#     """Return sorted unique sample names under group/field by inspecting array_keys"""
#     sample_names = set()
#     for parts in group.array_keys():
#         if len(parts) >= 2 and parts[0] == field:
#             # parts example: ('corrected_t2m', 'sample=00000', 'datetime')
#             if len(parts) >= 2:
#                 sample_names.add(parts[1])
#     return sorted(sample_names)


# def load_sample_metadata(group, field, sample):
#     """
#     Try to load datetime, lat, lon arrays for a sample.
#     Returns (datetimes, lats, lons, data_exists_flag)
#     datetimes as np.datetime64[ns], lat/lon as numpy floats.
#     """
#     base = f"{field}/{sample}"
#     # possible array names
#     dt_keys = ("datetime", "time", "dates")
#     lat_keys = ("lat", "latitude")
#     lon_keys = ("lon", "longitude", "longitudes")

#     def _try_get(name):
#         try:
#             arr = group[f"{base}/{name}"][:]
#             return arr
#         except Exception:
#             return None

#     dt = None
#     for k in dt_keys:
#         dt = _try_get(k)
#         if dt is not None:
#             break
#     if dt is None:
#         return None, None, None

#     # normalize datetime to numpy datetime64[ns]
#     dt = np.asarray(dt)
#     if not np.issubdtype(dt.dtype, np.datetime64):
#         try:
#             dt = pd.to_datetime(dt).values.astype("datetime64[ns]")
#         except Exception:
#             # try interpreting as numeric epoch (seconds)
#             dt = dt.astype(np.int64)
#             dt = dt.astype("datetime64[s]").astype("datetime64[ns]")

#     lat = None
#     for k in lat_keys:
#         lat = _try_get(k)
#         if lat is not None:
#             break
#     lon = None
#     for k in lon_keys:
#         lon = _try_get(k)
#         if lon is not None:
#             break
#     # data array candidate names
#     data_candidates = ("data", "pred", "temperature", FIELD)
#     data_exists = False
#     for dc in data_candidates:
#         try:
#             _ = group[f"{base}/{dc}"]
#             data_exists = True
#             data_name = dc
#             break
#         except Exception:
#             data_name = None
#     return dt, lat, lon, data_exists, data_name


# def get_value_from_sample(group, field, sample, data_name, target_time, target_lat, target_lon, preferred_ml_index=0):
#     """
#     Given a sample and target coordinates, return the nearest predicted value and distance.
#     Returns (value, dist_deg, info_dict) or (None, None, None) if not found.
#     """
#     base = f"{field}/{sample}"
#     # load arrays
#     try:
#         dt = np.asarray(group[f"{base}/datetime"][:])
#     except Exception:
#         # try alternative name
#         try:
#             dt = np.asarray(group[f"{base}/time"][:])
#         except Exception:
#             return None, None, None
#     # normalize dt to ns
#     if not np.issubdtype(dt.dtype, np.datetime64):
#         dt = pd.to_datetime(dt).values.astype("datetime64[ns]")

#     # find time index within sample
#     matches = np.where(dt == np.datetime64(target_time))[0]
#     if matches.size == 0:
#         return None, None, None
#     t_idx = int(matches[0])

#     # load lat/lon arrays
#     try:
#         lats = np.asarray(group[f"{base}/lat"][:]).astype(float)
#     except Exception:
#         try:
#             lats = np.asarray(group[f"{base}/latitude"][:]).astype(float)
#         except Exception:
#             return None, None, None
#     try:
#         lons = np.asarray(group[f"{base}/lon"][:]).astype(float)
#     except Exception:
#         try:
#             lons = np.asarray(group[f"{base}/longitude"][:]).astype(float)
#         except Exception:
#             return None, None, None

#     # normalize lon ranges to 0..360 for both arrays
#     lons360 = np.array([to_360(float(x)) for x in lons])
#     target_lon360 = to_360(target_lon)

#     # find nearest lat/lon indices
#     lat_idx = int(np.argmin(np.abs(lats - float(target_lat))))
#     # handle wrap-around for lon by comparing circular distance
#     lon_diffs = np.minimum(np.abs(lons360 - target_lon360), 360.0 - np.abs(lons360 - target_lon360))
#     lon_idx = int(np.argmin(lon_diffs))

#     # compute angular distance
#     dist_deg = angular_distance_deg(float(lats[lat_idx]), lons360[lon_idx], float(target_lat), target_lon360)

#     # load data array and index appropriately
#     try:
#         arr = np.asarray(group[f"{base}/{data_name}"][:])
#     except Exception:
#         return None, None, None

#     # arr might have shape: (ml, datetime, lat, lon) or (datetime, lat, lon)
#     try:
#         if arr.ndim == 4:
#             # ml, time, lat, lon
#             val = float(arr[preferred_ml_index, t_idx, lat_idx, lon_idx])
#         elif arr.ndim == 3:
#             # time, lat, lon
#             val = float(arr[t_idx, lat_idx, lon_idx])
#         else:
#             # unknown layout
#             return None, None, None
#     except Exception:
#         return None, None, None

#     info = {"sample": sample, "data_name": data_name, "sample_time_index": t_idx,
#             "sample_lat_index": lat_idx, "sample_lon_index": lon_idx}
#     return val, dist_deg, info


# def main(model_id, output_dir):
#     outp = Path(output_dir)
#     outp.mkdir(parents=True, exist_ok=True)

#     # load reference datasets
#     era5 = xr.open_dataset(DEFAULT_ERA5)
#     nice = xr.open_dataset(DEFAULT_NICE)

#     # determine NICE time, lat, lon arrays (name variations)
#     if "time" in nice.coords:
#         nice_times = np.asarray(nice["time"].values).astype("datetime64[ns]")
#     else:
#         nice_times = np.asarray(nice["Time"].values).astype("datetime64[ns]")

#     # try lat/lon names
#     if "latitude" in nice.coords:
#         nice_lats = np.asarray(nice["latitude"].values)
#     elif "lat" in nice.coords:
#         nice_lats = np.asarray(nice["lat"].values)
#     else:
#         # fallback: try variable dims
#         nice_lats = np.asarray(nice.coords[list(nice.coords.keys())[-2]].values)

#     if "longitude" in nice.coords:
#         nice_lons = np.asarray(nice["longitude"].values)
#     elif "lon" in nice.coords:
#         nice_lons = np.asarray(nice["lon"].values)
#     else:
#         nice_lons = np.asarray(nice.coords[list(nice.coords.keys())[-1]].values)

#     # If NICE variables are 1D time series, align indexes -> create list of observations
#     # Many N-ICE station datasets are time series; here we assume 1D arrays with same len
#     if nice_times.shape[0] != nice_lats.shape[0] and nice_lats.size == 1:
#         # single station lat/lon repeated
#         lat_series = np.repeat(float(nice_lats), nice_times.shape[0])
#         lon_series = np.repeat(float(nice_lons), nice_times.shape[0])
#     else:
#         lat_series = nice_lats
#         lon_series = nice_lons

#     # If NICE has 1D time series variable e.g. air_temperature_2m, use its time length
#     n_obs = nice_times.shape[0]

#     # open zarr predictions
#     zarr_path = Path(f"results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr")
#     if not zarr_path.exists():
#         print(f"Zarr store not found: {zarr_path}")
#         return

#     store = ZipStore(str(zarr_path))
#     group = zarr.open(store=store, mode="r")

#     # find sample names under FIELD
#     sample_names = find_samples_for_field(group, FIELD)
#     print(f"Found {len(sample_names)} samples for field {FIELD}")

#     # preload metadata per sample: datetimes -> which times they contain, and data_name
#     time_to_samples = {}  # map datetime64 -> list of (sample, data_name)
#     sample_meta = {}
#     for sample in sample_names:
#         dt, lat_arr, lon_arr, data_exists, data_name = load_sample_metadata(group, FIELD, sample)
#         if dt is None or not data_exists:
#             continue
#         sample_meta[sample] = {"datetime": dt, "lat": lat_arr, "lon": lon_arr, "data_name": data_name}
#         for t in np.unique(dt):
#             time_to_samples.setdefault(np.datetime64(t), []).append(sample)

#     print(f"Indexed times across samples, unique times in samples: {len(time_to_samples)}")

#     # Prepare results containers
#     records = []
#     # iterate observations from NICE; for each, try to find prediction
#     for i in range(n_obs):
#         t = nice_times[i]
#         lat = float(lat_series[i]) if np.ndim(lat_series) > 0 else float(lat_series)
#         lon = float(lon_series[i]) if np.ndim(lon_series) > 0 else float(lon_series)
#         # ensure datetime64[ns]
#         t64 = np.datetime64(t)

#         # try matching ERA5 at same time and location (optional)
#         # find nearest ERA5 grid value
#         try:
#             era5_val = None
#             if "T2M" in era5:
#                 # find nearest indices in era5 grid
#                 era_lats = np.asarray(era5["lat"].values) if "lat" in era5.coords else np.asarray(era5["latitude"].values)
#                 era_lons = np.asarray(era5["lon"].values) if "lon" in era5.coords else np.asarray(era5["longitude"].values)
#                 ilat = int(np.argmin(np.abs(era_lats - lat)))
#                 ilon = int(np.argmin(np.abs( (era_lons + 360) % 360 - to_360(lon) )))
#                 # find time index
#                 era_times = np.asarray(era5["time"].values).astype("datetime64[ns]")
#                 tmatch = np.where(era_times == t64)[0]
#                 if tmatch.size:
#                     era5_val = float(era5["T2M"].values[tmatch[0]])
#         except Exception:
#             era5_val = None

#         nice_val = None
#         try:
#             # variable name in NICE might be 'air_temperature_2m' or similar
#             var_candidates = ["air_temperature_2m", "T2M", "t2m"]
#             for v in var_candidates:
#                 if v in nice:
#                     nice_val = float(nice[v].isel(time=i).values)
#                     break
#         except Exception:
#             nice_val = None

#         # find candidate samples for this time
#         candidates = time_to_samples.get(t64, [])
#         best = {"dist": 1e9, "value": None, "sample": None, "info": None}
#         for sample in candidates:
#             meta = sample_meta.get(sample)
#             if meta is None:
#                 continue
#             # choose lat/lon arrays from meta to compute approximate distance
#             lats = np.asarray(meta["lat"], dtype=float)
#             lons = np.asarray(meta["lon"], dtype=float)
#             # compute minimal possible distance to sample grid (approx)
#             # use central or nearest gridpoint
#             # quick heuristic: check min distance to any lat/lon in sample (cheap-ish)
#             # vectorized dist on coarse representation: take every 5th point if large
#             idxs = np.s_[::max(1, max(1, len(lats)//20))]
#             sample_lats_coarse = lats[idxs]
#             sample_lons_coarse = lons[idxs] if lons.ndim == 1 else lons[idxs]
#             # compute approximate lon conversion
#             sample_lons_coarse360 = np.array([to_360(x) for x in sample_lons_coarse])
#             lon360 = to_360(lon)
#             dlat = np.abs(sample_lats_coarse - lat)
#             dlon = np.minimum(np.abs(sample_lons_coarse360 - lon360), 360.0 - np.abs(sample_lons_coarse360 - lon360))
#             approx_deg = np.hypot(dlat, dlon)
#             approx_min = float(np.min(approx_deg))
#             # short-circuit if too far relative to best
#             if approx_min >= best["dist"]:
#                 continue
#             # attempt retrieval
#             val, dist_deg, info = get_value_from_sample(group, FIELD, sample, meta["data_name"], t64, lat, lon)
#             if val is None:
#                 continue
#             if dist_deg < best["dist"]:
#                 best.update({"dist": dist_deg, "value": val, "sample": sample, "info": info})

#         record = {
#             "time": np.datetime_as_string(t64, unit="s"),
#             "nice_value": nice_val,
#             "era5_value": era5_val,
#             "pred_value": best["value"],
#             "pred_distance_deg": best["dist"] if best["value"] is not None else np.nan,
#             "pred_sample": best["sample"],
#             "model_id": model_id,
#         }
#         records.append(record)

#     df = pd.DataFrame(records)
#     csv_out = outp / f"matched_predictions_{model_id}.csv"
#     nc_out = outp / f"matched_predictions_{model_id}.nc"

#     df.to_csv(csv_out, index=False)
#     # also save as xarray dataset
#     ds_out = xr.Dataset(
#         {
#             "nice_value": ("time", df["nice_value"].values),
#             "era5_value": ("time", df["era5_value"].values),
#             "pred_value": ("time", df["pred_value"].values),
#             "pred_distance_deg": ("time", df["pred_distance_deg"].values),
#         },
#         coords={"time": pd.to_datetime(df["time"].values)}
#     )
#     ds_out.to_netcdf(nc_out)

#     print(f"Saved CSV -> {csv_out}")
#     print(f"Saved NetCDF -> {nc_out}")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: match_nice_predictions.py <model_id> [output_dir]")
#         sys.exit(1)
#     model_id = sys.argv[1]
#     outdir = sys.argv[2] if len(sys.argv) > 2 else "/work/ab1412/atmorep/results"
#     main(model_id, outdir)
