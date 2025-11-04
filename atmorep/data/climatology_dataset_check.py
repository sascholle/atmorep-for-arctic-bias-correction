#!/usr/bin/env python3
import os
import numpy as np
import zarr
from datetime import datetime

# ==================================================
# CONFIGURATION
# ==================================================
ZARR_PATH = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr"
REPORT_PATH = "validation_copy_report.txt"

PHYSICAL_LIMITS = {
    "temperature": (180, 330),          # Kelvin
    "t2m": (180, 330),
    "corrected_t2m": (180, 330),
    "specific_humidity": (0, 0.05),     # dimensionless (kg/kg)
    "vorticity": (-0.002, 0.002),
    "divergence": (-0.002, 0.002),
    "velocity_u": (-120, 120),          # m/s
    "velocity_v": (-120, 120),
    "velocity_z": (-10, 10),
    "total_precip": (0, 0.1),
}

PRECIP_KEYS = {"precip", "precipitation", "tp", "total_precip"}

# Sampling settings
SPATIAL_SAMPLE_STEP = 100
TIME_SAMPLE_STEP = 200
LEVEL_SAMPLE_STEP = 1  # only 5 levels
MONTH_SAMPLE_STEP = 6  # for normalization datasets (monthly data)

# ==================================================
# SAMPLING HELPERS
# ==================================================
def sample_array(arr, name=""):
    """Sample array depending on its dimensionality and meaning."""
    ndim = arr.ndim

    # --- data: (time, field, levels, lat, lon)
    if ndim == 5:
        sl = (
            slice(None, None, TIME_SAMPLE_STEP),
            slice(None),
            slice(None, None, LEVEL_SAMPLE_STEP),
            slice(None, None, SPATIAL_SAMPLE_STEP),
            slice(None, None, SPATIAL_SAMPLE_STEP),
        )

    # --- data_sfc: (time, field, lat, lon)
    elif ndim == 4 and "global" not in name and "norm" not in name:
        sl = (
            slice(None, None, TIME_SAMPLE_STEP),
            slice(None),
            slice(None, None, SPATIAL_SAMPLE_STEP),
            slice(None, None, SPATIAL_SAMPLE_STEP),
        )

    # --- global_norm: (months, mean/std, field, level)
    elif ndim == 4 and "global" in name:
        sl = (
            slice(None, None, MONTH_SAMPLE_STEP),
            slice(None),
            slice(None),
            slice(None),
        )

    # --- norm_sfc: (months, stat, field, lat, lon)
    elif ndim == 5 and "sfc" in name:
        sl = (
            slice(None, None, MONTH_SAMPLE_STEP),
            slice(None),
            slice(None),
            slice(None, None, SPATIAL_SAMPLE_STEP),
            slice(None, None, SPATIAL_SAMPLE_STEP),
        )

    # --- norm: (months, stat, field, levels, lat, lon)
    elif ndim == 6:
        sl = (
            slice(None, None, MONTH_SAMPLE_STEP),
            slice(None),
            slice(None),
            slice(None, None, LEVEL_SAMPLE_STEP),
            slice(None, None, SPATIAL_SAMPLE_STEP),
            slice(None, None, SPATIAL_SAMPLE_STEP),
        )
    else:
        sl = (slice(None),)

    return arr[sl]

# ==================================================
# CHECK FUNCTIONS
# ==================================================
def check_values(name, arr):
    """Check for NaNs, zeros, and climatology limits."""
    data = np.array(sample_array(arr, name))
    flat = data.ravel()
    n_total = flat.size
    n_nan = np.isnan(flat).sum()
    n_zero = np.sum(flat == 0)
    mean_val = float(np.nanmean(flat))
    min_val = float(np.nanmin(flat))
    max_val = float(np.nanmax(flat))

    limits = PHYSICAL_LIMITS.get(name.lower(), None)
    out_of_bounds = 0
    if limits:
        out_of_bounds = int(np.sum((flat < limits[0]) | (flat > limits[1])))

    # --- precipitation zeros are expected ---
    if any(k in name.lower() for k in PRECIP_KEYS):
        zero_flag = "valid"
    # --- corrected_t2m zeros expected beyond lat idx 71 (not spatially sampled here) ---
    elif "corrected_t2m" in name.lower():
        zero_flag = "expected_partial"
    else:
        zero_flag = "suspicious" if n_zero > 0.5 * n_total else "ok"

    return {
        "name": name,
        "shape": arr.shape,
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "n_nan": int(n_nan),
        "n_zero": int(n_zero),
        "zero_flag": zero_flag,
        "out_of_bounds": out_of_bounds,
        "n_total": int(n_total),
    }

def check_norm_dataset(name, arr):
    """
    Check normalization datasets (mean/std) against climatology.
    Uses PHYSICAL_LIMITS for the 'mean' layer, and relative sanity for 'std'.
    """
    data = np.array(sample_array(arr, name))
    ndim = data.ndim

    # (months, stat, field, ...) → second axis = 0=mean, 1=std
    if ndim >= 2 and data.shape[1] == 2:
        means = data[:, 0, ...]
        stds  = data[:, 1, ...]
    else:
        means, stds = data, None

    flat_means = means.ravel()
    mean_val = float(np.nanmean(flat_means))
    min_val = float(np.nanmin(flat_means))
    max_val = float(np.nanmax(flat_means))
    n_nan = np.isnan(flat_means).sum()
    n_zero = np.sum(flat_means == 0)

    # Apply climatological sanity per physical limits
    out_of_bounds = 0
    for key, (low, high) in PHYSICAL_LIMITS.items():
        if key in name.lower():
            out_of_bounds = int(np.sum((flat_means < low) | (flat_means > high)))
            break

    range_flag = "reasonable" if out_of_bounds == 0 else "out_of_range"

    # --- check stds if available ---
    std_flag = "ok"
    if stds is not None:
        flat_stds = stds.ravel()
        std_mean = float(np.nanmean(flat_stds))
        std_max = float(np.nanmax(flat_stds))
        std_n_nan = np.isnan(flat_stds).sum()

        # std should be positive, not NaN, and not unphysically large
        valid_std = np.all(flat_stds >= 0)
        # sanity bound: std less than half the climatological span
        clim_span = 999  # fallback
        for key, (low, high) in PHYSICAL_LIMITS.items():
            if key in name.lower():
                clim_span = high - low
                break
        if std_max > clim_span / 2:
            std_flag = "suspect"
        elif not valid_std or std_n_nan > 0:
            std_flag = "invalid"
    else:
        std_mean, std_max, std_flag = np.nan, np.nan, "missing"

    return {
        "name": name,
        "shape": arr.shape,
        "mean_mean": mean_val,
        "mean_min": min_val,
        "mean_max": max_val,
        "n_nan": int(n_nan),
        "n_zero": int(n_zero),
        "out_of_bounds": out_of_bounds,
        "range_flag": range_flag,
        "std_mean": std_mean,
        "std_max": std_max,
        "std_flag": std_flag,
    }

# ==================================================
# MAIN SCAN LOGIC
# ==================================================
def scan_zarr(zarr_path):
    store = zarr.open(zarr_path, mode="r")
    report = []

    # --- Check data and data_sfc ---
    for grp_name in ["data", "data_sfc"]:
        if grp_name not in store:
            continue
        grp = store[grp_name]
        fields = grp.attrs.get("fields", [])
        for i, var_name in enumerate(fields):
            try:
                arr = grp[:, i, ...]
                report.append(check_values(var_name, arr))
            except Exception as e:
                report.append({"name": var_name, "error": str(e)})

    # --- Check normalization datasets ---
    if "normalization" in store:
        norm_grp = store["normalization"]
        for name, arr in norm_grp.items():
            try:
                report.append(check_norm_dataset(name, arr))
            except Exception as e:
                report.append({"name": name, "error": str(e)})

    return report


# ==================================================
# EXECUTION
# ==================================================
if __name__ == "__main__":
    report = scan_zarr(ZARR_PATH)
    suspicious = [r for r in report if "error" in r or r.get("zero_flag") in ["suspicious", "expected_partial"]
                  or r.get("out_of_bounds", 0) > 0 or r.get("std_flag", "") == "suspect"]

    with open(REPORT_PATH, "w") as f:
        f.write(f"Validation Report for {ZARR_PATH}\nGenerated: {datetime.now()}\n\n")
        for r in report:
            f.write(f"{r}\n")

        f.write("\n=== SUMMARY OF POTENTIAL ISSUES ===\n")
        for r in suspicious:
            f.write(f"{r['name']}: {r}\n")

    print(f"✅ Validation complete. Report written to {REPORT_PATH}")
    print(f"⚠️ Found {len(suspicious)} potential issues (see summary in report).")

