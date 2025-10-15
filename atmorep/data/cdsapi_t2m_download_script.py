import cdsapi
import sys

year = sys.argv[1]

request = {
    "product_type": "reanalysis",
    "variable": "2m_temperature",
    "year": year,
    "month": [f"{m:02d}" for m in range(1, 13)],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": [f"{h:02d}:00" for h in range(24)],
    "format": "grib"
}
filename = f"era5_2m_temperature_{year}.grib"
c = cdsapi.Client()
c.retrieve("reanalysis-era5-single-levels", request, filename)