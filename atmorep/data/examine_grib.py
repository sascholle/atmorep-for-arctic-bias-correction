import xarray as xr
# source /work/ab1412/atmorep/pyenv/bin/activate

grib_path = "/work/ab1412/atmorep/data/grib_files/era5_2m_temperature_2010_2021.grib"

print(f"Opening GRIB file: {grib_path}")
try:
    ds = xr.open_dataset(grib_path, engine="cfgrib")
except Exception as e:
    print(f"Error opening GRIB file: {e}")
    exit(1)

print("\nVariables in the GRIB file:")
for var in ds.data_vars:
    arr = ds[var]
    print(f"  - {var}: shape={arr.shape}, dtype={arr.dtype}")
    for key in arr.attrs:
        print(f"      {key}: {arr.attrs[key]}")
    print()

print("Coordinates:")
for coord in ds.coords:
    arr = ds[coord]
    print(f"  - {coord}: shape={arr.shape}, dtype={arr.dtype}")

print("\nGlobal attributes:")
for key in ds.attrs:
    print(f"  {key}: {ds.attrs[key]}")