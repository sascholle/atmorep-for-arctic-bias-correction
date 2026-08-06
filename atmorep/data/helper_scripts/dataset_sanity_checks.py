import zarr 
import numpy as np
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors

# Inserted: physical plausibility limits (used to flag suspicious climatology)
PHYSICAL_LIMITS = {
    "temperature": (180, 330),          # Kelvin
    "t2m": (180, 330),
    "corrected_t2m": (180, 330),
    "specific_humidity": (0, 0.05),     # kg/kg
    "vorticity": (-0.002, 0.002),
    "divergence": (-0.002, 0.002),
    "velocity_u": (-120, 120),          # m/s
    "velocity_v": (-120, 120),
    "velocity_z": (-10, 10),
    "total_precip": (0, 0.1),
}

def comprehensive_zarr_analysis(zarr_path):
    """
    Comprehensive analysis of the entire Zarr dataset with broad sweeping checks
    """
    print(f"Opening target dataset: {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    print(store.tree())
    
    # Get metadata
    fields = store.attrs['fields']
    fields_sfc = store.attrs['fields_sfc']
    levels = store.attrs['levels']
    
    print("Atmospheric fields (data):", fields)
    print("Surface fields (data_sfc):", fields_sfc)
    print("Vertical levels:", levels)
    
    # Get array references
    data = store['data']
    data_sfc = store['data_sfc']
    norm = store['normalization/norm']
    norm_sfc = store['normalization/norm_sfc']
    global_norm = store['normalization/global_norm']
    global_norm_sfc = store['normalization/global_norm_sfc']
    time_coords = store['time']
    
    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Time range: {len(time_coords)} timesteps")
    print(f"Data shape: {data.shape}")
    print(f"Data_sfc shape: {data_sfc.shape}")
    print(f"Norm shape: {norm.shape}")
    print(f"Norm_sfc shape: {norm_sfc.shape}")

    print(f"\n=== BASIC ZERO/NAN ANALYSIS ===")
    check_zeros_and_nans_comprehensive(store, fields, fields_sfc)

    
    # # **1. CHECK FOR ZEROS AND NaNs ACROSS ALL ARRAYS**
    # # Denser checks:
    # print(f"\n=== ZERO/NaN/CLIMATOLOGY ANALYSIS (denser sampling) ===")
    # check_zeros_nans_and_climatology(store, fields, fields_sfc, phys_limits=PHYSICAL_LIMITS)

    # print(f"\n=== SPATIAL AND TEMPORAL SAMPLING (denser) ===")
    # spatial_temporal_sampling(store, fields, fields_sfc, denser=True)

    # print(f"\n=== NORMALIZATION ANALYSIS (sanity/climatology) ===")
    # check_normalization_consistency(store, fields, fields_sfc, phys_limits=PHYSICAL_LIMITS)
    
    # # **4. CORRECTED T2M SPECIFIC CHECKS**
    # print(f"\n=== CORRECTED T2M SPECIFIC ANALYSIS ===")
    # check_corrected_t2m_boundaries(store)

    # #**4. Climatology plots**
    # print(f"\n=== CLIMATOLOGY PLOTS ===")
    # try:
    #     sample_and_scatter_seasonality(store,
    #                                    fields,
    #                                    fields_sfc,
    #                                    level_count_samples=5, 
    #                                    lat_centers=(10,360,710))
    # except Exception as e:
    #     print("Failed to generate weekly seasonality plots:", e

def sample_and_scatter_seasonality(store,
                                             fields,
                                             fields_sfc,
                                             level_count_samples=5,
                                             lat_centers=(10, 360, 710),
                                             lat_window=5,
                                             lon_stride=100,
                                             months=12,
                                             fields_to_plot=7,
                                             fields_to_plot_sfc=3,
                                             time_block=1000,
                                             outdir="/work/ab1412/atmorep/plotting/era5_seasonality/scatter_monthly_stream"):
    """
    Memory-light monthly-climatology scatter plots.

    Strategy:
    - For each field (one figure at a time) build a list of sample points (level,lat,lon).
    - For each sample point keep monthly_sum and monthly_count (shape (n_points,12)).
    - Iterate time in small blocks (time_block steps) and read small 1D slices
      store['data'][t0:t1, f_idx, lvl, lat, lon] per point; accumulate sums/counts per month.
    - After streaming all time, compute monthly means and coverage fractions and plot.
    """

    import gc
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)

    # build months_idx for all time positions (uint8)
    time_raw = store['time'][:]
    try:
        import pandas as pd
        if np.issubdtype(time_raw.dtype, np.number):
            times_all = pd.to_datetime(time_raw, unit='s', origin='unix')
        else:
            times_all = pd.to_datetime(time_raw.astype('U'))
        months_idx = np.array([t.month for t in times_all], dtype=np.uint8)
    except Exception:
        ntime = store['data'].shape[0]
        months_idx = np.tile(np.arange(1, months+1, dtype=np.uint8), int(np.ceil(ntime / months)))[:ntime]

    # precompute total samples per month for coverage fraction
    total_per_month = np.array([int(np.count_nonzero(months_idx == m)) for m in range(1, 13)], dtype=int)
    # safe guard
    total_per_month[total_per_month == 0] = 1

    ntime = store['data'].shape[0]
    nlevels = store['data'].shape[2]
    nlat = store['data'].shape[3]
    nlon = store['data'].shape[4] if store['data'].ndim == 5 else store['data'].shape[-1]

    # contiguous first N levels (model has 5)
    level_count = min(nlevels, level_count_samples)
    level_idxs = list(range(level_count))

    sampled_lons = list(range(0, nlon, lon_stride))[:level_count_samples]

    # helper: list of lat sample indices for a lat_center
    def lat_samples_for_center(center):
        start = max(0, center - lat_window)
        end = min(nlat - 1, center + lat_window)
        return list(range(start, end + 1))

    # --- ATMOSPHERIC FIELDS (data with levels) ---
    for f_idx, field_name in enumerate(fields[:fields_to_plot]):
        print(f"Processing field {f_idx+1}/{min(len(fields), fields_to_plot)}: {field_name}")
        # build list of sample points: tuples (level, lat, lon)
        points = []
        point_to_subplot = []  # (row_idx, col_idx) mapping to assign points to subplot
        for r, lat_center in enumerate(lat_centers):
            lat_samples = lat_samples_for_center(lat_center)
            for c, lvl in enumerate(level_idxs):
                for lat in lat_samples:
                    for lon in sampled_lons:
                        points.append((int(lvl), int(lat), int(lon)))
                        point_to_subplot.append((r, c))
        n_points = len(points)
        if n_points == 0:
            print("No sample points selected for atmospheric plotting, skipping.")
            continue

        # monthly accumulators per point
        monthly_sum = np.zeros((n_points, 12), dtype=np.float64)
        monthly_count = np.zeros((n_points, 12), dtype=np.int32)

        # stream through time in blocks
        for t0 in range(0, ntime, time_block):
            t1 = min(ntime, t0 + time_block)
            months_chunk = months_idx[t0:t1]
            # for each point load the small chunk and accumulate
            for p_idx, (lvl, lat, lon) in enumerate(points):
                try:
                    ts = store['data'][t0:t1, f_idx, lvl, lat, lon]
                except Exception:
                    continue
                ts = np.asarray(ts, dtype=np.float32)
                # iterate unique months in this chunk (small)
                unique_months = np.unique(months_chunk)
                for m in unique_months:
                    mask = (months_chunk == m)
                    if not np.any(mask):
                        continue
                    vals = ts[mask]
                    if vals.size == 0:
                        continue
                    valid = ~np.isnan(vals)
                    if np.count_nonzero(valid) == 0:
                        continue
                    s = float(np.nansum(vals))
                    c = int(np.count_nonzero(valid))
                    monthly_sum[p_idx, m - 1] += s
                    monthly_count[p_idx, m - 1] += c
                # free ts quickly
                del ts
            # end points loop
        # end time streaming

        # compute monthly mean and coverage per point
        monthly_mean = np.full_like(monthly_sum, np.nan, dtype=np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            mask_pos = monthly_count > 0
            monthly_mean[mask_pos] = (monthly_sum[mask_pos] / monthly_count[mask_pos]).astype(np.float32)
        coverage = monthly_count.astype(np.float32) / total_per_month.reshape(1, 12)

        # plotting: 3 x len(level_idxs) grid
        fig, axes = plt.subplots(nrows=len(lat_centers), ncols=len(level_idxs),
                                 figsize=(4 * len(level_idxs), 3 * len(lat_centers)),
                                 squeeze=False)
        fig.suptitle(f"Atmospheric field: {field_name}")

        # for each subplot (row r, col c) collect points assigned there
        for p_idx, (r, c) in enumerate(point_to_subplot):
            # for each month where count>0 for this point, add scatter point
            for m in range(12):
                if monthly_count[p_idx, m] == 0:
                    continue
                ax = axes[r][c]
                ax.scatter(m + 1, monthly_mean[p_idx, m], c=coverage[p_idx, m:m+1],
                           cmap=plt.cm.viridis, norm=mcolors.Normalize(0, 1),
                           s=10, alpha=0.9, edgecolors='none')
        # finalize axes
        for r in range(len(lat_centers)):
            for c in range(len(level_idxs)):
                ax = axes[r][c]
                ax.set_xlim(0.5, 12.5)
                ax.set_xticks(range(1, 13))
                if c == 0:
                    ax.set_ylabel("monthly mean")
                ax.set_title(f"level={level_idxs[c]} lat~{lat_centers[r]}")
                ax.grid(True, lw=0.3)

        # colorbar (single mappable): create a proxy scatter for colorbar
        proxy = axes[0][0].scatter([], [], c=[], cmap=plt.cm.viridis, norm=mcolors.Normalize(0, 1))
        fig.colorbar(proxy, ax=axes.ravel().tolist(), label="coverage fraction")

        outpath = outp / f"{field_name.replace(' ', '_')}_monthly_scatter_atmos_stream.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(str(outpath), dpi=150); plt.close(fig)

        # free memory
        del monthly_sum, monthly_count, monthly_mean, coverage, points, point_to_subplot
        gc.collect()

    # --- SURFACE FIELDS (data_sfc, no levels) ---
    for f_idx, field_name in enumerate(fields_sfc[:fields_to_plot_sfc]):
        print(f"Processing surface field {f_idx+1}/{min(len(fields_sfc), fields_to_plot_sfc)}: {field_name}")
        # points: (lat, lon) grouped by lat_center row and sampled_lons columns
        points = []
        point_to_subplot = []
        for r, lat_center in enumerate(lat_centers):
            lat_samples = lat_samples_for_center(lat_center)
            for c, lon in enumerate(sampled_lons):
                for lat in lat_samples:
                    points.append((int(lat), int(lon)))
                    point_to_subplot.append((r, c))
        n_points = len(points)
        if n_points == 0:
            print("No points for sfc plotting, skipping.")
            continue

        monthly_sum = np.zeros((n_points, 12), dtype=np.float64)
        monthly_count = np.zeros((n_points, 12), dtype=np.int32)

        for t0 in range(0, ntime, time_block):
            t1 = min(ntime, t0 + time_block)
            months_chunk = months_idx[t0:t1]
            for p_idx, (lat, lon) in enumerate(points):
                try:
                    ts = store['data_sfc'][t0:t1, f_idx, lat, lon]
                except Exception:
                    continue
                ts = np.asarray(ts, dtype=np.float32)
                unique_months = np.unique(months_chunk)
                for m in unique_months:
                    mask = (months_chunk == m)
                    if not np.any(mask):
                        continue
                    vals = ts[mask]
                    if vals.size == 0:
                        continue
                    valid = ~np.isnan(vals)
                    if np.count_nonzero(valid) == 0:
                        continue
                    monthly_sum[p_idx, m - 1] += float(np.nansum(vals))
                    monthly_count[p_idx, m - 1] += int(np.count_nonzero(valid))
                del ts
        # compute means/coverage
        monthly_mean = np.full_like(monthly_sum, np.nan, dtype=np.float32)
        mask_pos = monthly_count > 0
        monthly_mean[mask_pos] = (monthly_sum[mask_pos] / monthly_count[mask_pos]).astype(np.float32)
        coverage = monthly_count.astype(np.float32) / total_per_month.reshape(1, 12)

        # plotting: grid rows=len(lat_centers), cols=len(sampled_lons)
        fig, axes = plt.subplots(nrows=len(lat_centers), ncols=len(sampled_lons),
                                 figsize=(4 * len(sampled_lons), 3 * len(lat_centers)),
                                 squeeze=False)
        fig.suptitle(f"Surface field: {field_name}")

        for p_idx, (r, c) in enumerate(point_to_subplot):
            for m in range(12):
                if monthly_count[p_idx, m] == 0:
                    continue
                ax = axes[r][c]
                ax.scatter(m + 1, monthly_mean[p_idx, m], c=coverage[p_idx, m:m+1],
                           cmap=plt.cm.viridis, norm=mcolors.Normalize(0, 1),
                           s=10, alpha=0.9, edgecolors='none')

        for r in range(len(lat_centers)):
            for c in range(len(sampled_lons)):
                ax = axes[r][c]
                ax.set_xlim(0.5, 12.5); ax.set_xticks(range(1, 13))
                if c == 0:
                    ax.set_ylabel("monthly mean")
                ax.set_title(f"lon={sampled_lons[c]} lat~{lat_centers[r]}")
                ax.grid(True, lw=0.3)

        proxy = axes[0][0].scatter([], [], c=[], cmap=plt.cm.viridis, norm=mcolors.Normalize(0, 1))
        fig.colorbar(proxy, ax=axes.ravel().tolist(), label="coverage fraction")
        outpath = outp / f"{field_name.replace(' ', '_')}_monthly_scatter_sfc_stream.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(str(outpath), dpi=150); plt.close(fig)

        del monthly_sum, monthly_count, monthly_mean, coverage, points, point_to_subplot
        gc.collect()

    print(f"Saved streaming monthly scatter plots to {outdir}")

def check_zeros_and_nans_comprehensive(store, fields, fields_sfc):
    """Check for problematic zeros and NaNs across all arrays"""
    
    # Sample time indices (every ~8760 timesteps ≈ yearly)
    time_samples = list(range(0, store['data'].shape[0], 1000))[:]  # ~12 samples
    
    # Spatial samples: Arctic, mid-lat, Antarctic
    lat_samples = list(range(0, store['lats'].shape[0], 100))[:]  # Arctic, mid, Antarctic
    lon_samples = list(range(0, store['lons'].shape[0], 1000))[:]  # Various longitudes
    
    print("Sampling strategy:")
    print(f"  Time: {len(time_samples)} samples across {store['data'].shape[0]} timesteps")
    print(f"  Space: {len(lat_samples)} x {len(lon_samples)} grid points")
    
    # **1. Data array checks**
    print(f"\n--- Atmospheric Data (data) Analysis ---")
    for field_idx, field_name in enumerate(fields):
        print(f"\nField {field_idx}: {field_name}")
        
        field_stats = {'total_zeros': 0, 'total_nans': 0, 'total_samples': 0}
        
        for time_idx in time_samples[:3]:  # Check first 3 time samples
            for level_idx in range(len(store.attrs['levels'])):
                for lat_idx in lat_samples:
                    for lon_idx in lon_samples[:3]:  # Subset of longitudes
                        try:
                            sample = store['data'][time_idx, field_idx, level_idx, lat_idx, lon_idx]
                            field_stats['total_samples'] += 1
                            
                            if sample == 0:
                                field_stats['total_zeros'] += 1
                            if np.isnan(sample):
                                field_stats['total_nans'] += 1
                                
                        except Exception as e:
                            print(f"    Error accessing data[{time_idx}, {field_idx}, {level_idx}, {lat_idx}, {lon_idx}]: {e}")
        
        zero_pct = (field_stats['total_zeros'] / field_stats['total_samples']) * 100
        nan_pct = (field_stats['total_nans'] / field_stats['total_samples']) * 100
        
        status = "❌" if zero_pct > 20 or nan_pct > 5 else "⚠️" if zero_pct > 5 or nan_pct > 1 else "✓"
        print(f"  {status} Zeros: {zero_pct:5.1f}% | NaNs: {nan_pct:5.1f}% | Samples: {field_stats['total_samples']}")
    
    # **2. Data_sfc array checks**
    print(f"\n--- Surface Data (data_sfc) Analysis ---")
    for field_idx, field_name in enumerate(fields_sfc):
        print(f"\nField {field_idx}: {field_name}")
        
        field_stats = {'total_zeros': 0, 'total_nans': 0, 'total_samples': 0}
        
        # Special handling for corrected_t2m (only valid in Arctic region :71)
        if field_name == 'corrected_t2m':
            lat_check = [10, 35, 70]  # Only Arctic region
            print("  (Checking only Arctic region lat indices 0-70 for corrected_t2m)")
        else:
            lat_check = lat_samples
            
        for time_idx in time_samples[:3]:
            for lat_idx in lat_check:
                for lon_idx in lon_samples[:3]:
                    try:
                        sample = store['data_sfc'][time_idx, field_idx, lat_idx, lon_idx]
                        field_stats['total_samples'] += 1
                        
                        if sample == 0:
                            field_stats['total_zeros'] += 1
                        if np.isnan(sample):
                            field_stats['total_nans'] += 1
                            
                    except Exception as e:
                        print(f"    Error accessing data_sfc[{time_idx}, {field_idx}, {lat_idx}, {lon_idx}]: {e}")
        
        zero_pct = (field_stats['total_zeros'] / field_stats['total_samples']) * 100
        nan_pct = (field_stats['total_nans'] / field_stats['total_samples']) * 100
        
        # Different thresholds for precipitation (expects some zeros)
        if 'precip' in field_name.lower():
            status = "❌" if zero_pct > 90 or nan_pct > 5 else "⚠️" if zero_pct > 70 or nan_pct > 1 else "✓"
        else:
            status = "❌" if zero_pct > 20 or nan_pct > 5 else "⚠️" if zero_pct > 5 or nan_pct > 1 else "✓"
            
        print(f"  {status} Zeros: {zero_pct:5.1f}% | NaNs: {nan_pct:5.1f}% | Samples: {field_stats['total_samples']}")

def check_zeros_nans_and_climatology(store, fields, fields_sfc, phys_limits=None):
    """
    Denser sampling across time/space and simple climatology checks against PHYSICAL_LIMITS.
    - time: ~monthly samples (≈120 samples across dataset if possible)
    - lat: 9 evenly spaced samples
    - lon: 12 evenly spaced samples
    """
    ntime = store['data'].shape[0]
    nlat = store['data'].shape[3]
    nlon = store['data'].shape[4]

    # choose ~120 time samples (denser than yearly)
    n_samples_time = min(120, max(12, ntime // max(1, int(ntime / 120))))
    time_step = max(1, ntime // n_samples_time)
    time_samples = list(range(0, ntime, time_step))
    # lat/lon samples
    lat_samples = np.linspace(0, nlat - 1, 9, dtype=int).tolist()
    lon_samples = np.linspace(0, nlon - 1, 12, dtype=int).tolist()

    print(f"Sampling: time {len(time_samples)} pts, lat {len(lat_samples)} pts, lon {len(lon_samples)} pts")

    # Atmospheric data checks + climatology
    print("\n--- Atmospheric (data) detailed checks ---")
    for fidx, fname in enumerate(fields):
        vals = []
        nzeros = nans = 0
        nchecked = 0
        for t in time_samples:
            for lvl in range(store['data'].shape[2]):
                for la in lat_samples:
                    for lo in lon_samples:
                        try:
                            v = store['data'][t, fidx, lvl, la, lo]
                        except Exception:
                            continue
                        nchecked += 1
                        if v == 0:
                            nzeros += 1
                        if np.isnan(v):
                            nans += 1
                        if np.isfinite(v):
                            vals.append(float(v))
        if nchecked == 0:
            print(f"{fname}: no samples")
            continue
        vals = np.asarray(vals, dtype=np.float64) if vals else np.array([], dtype=float)
        zpct = 100.0 * nzeros / nchecked
        npct = 100.0 * nans / nchecked
        print(f"{fname}: checked={nchecked} zeros={zpct:.1f}% nans={npct:.1f}%  vals_count={vals.size}")
        if vals.size:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            vmean, vstd = float(np.nanmean(vals)), float(np.nanstd(vals))
            print(f"   climatology: mean={vmean:.6g} std={vstd:.6g} min={vmin:.6g} max={vmax:.6g}")
            # physical limits check (if mapping available)
            if phys_limits:
                key = None
                lname = fname.lower()
                for k in phys_limits:
                    if k in lname:
                        key = k; break
                if key is not None:
                    lo, hi = phys_limits[key]
                    if vmin < lo or vmax > hi:
                        print(f"   ⚠️  {fname} outside physical limits [{lo},{hi}] (min={vmin}, max={vmax})")
                    else:
                        print(f"   ✓ {fname} within physical limits [{lo},{hi}]")
        else:
            print(f"   no finite values sampled for {fname}")

    # Surface data_sfc checks
    print("\n--- Surface (data_sfc) detailed checks ---")
    for fidx, fname in enumerate(fields_sfc):
        vals = []
        nzeros = nans = 0
        nchecked = 0
        for t in time_samples:
            for la in lat_samples:
                for lo in lon_samples:
                    try:
                        v = store['data_sfc'][t, fidx, la, lo]
                    except Exception:
                        continue
                    nchecked += 1
                    if v == 0:
                        nzeros += 1
                    if np.isnan(v):
                        nans += 1
                    if np.isfinite(v):
                        vals.append(float(v))
        if nchecked == 0:
            print(f"{fname}: no samples")
            continue
        vals = np.asarray(vals, dtype=np.float64) if vals else np.array([], dtype=float)
        zpct = 100.0 * nzeros / nchecked
        npct = 100.0 * nans / nchecked
        print(f"{fname}: checked={nchecked} zeros={zpct:.1f}% nans={npct:.1f}%  vals_count={vals.size}")
        if vals.size:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            vmean, vstd = float(np.nanmean(vals)), float(np.nanstd(vals))
            print(f"   climatology: mean={vmean:.6g} std={vstd:.6g} min={vmin:.6g} max={vmax:.6g}")
            if phys_limits:
                key = None
                lname = fname.lower()
                for k in phys_limits:
                    if k in lname:
                        key = k; break
                if key is not None:
                    lo, hi = phys_limits[key]
                    if vmin < lo or vmax > hi:
                        print(f"   ⚠️  {fname} outside physical limits [{lo},{hi}] (min={vmin}, max={vmax})")
                    else:
                        print(f"   ✓ {fname} within physical limits [{lo},{hi}]")
        else:
            print(f"   no finite values sampled for {fname}")

def spatial_temporal_sampling(store, fields, fields_sfc, denser=False):
    """Sample data across different regions and seasons with optional denser sampling"""
    ntime = store['data'].shape[0]
    if denser:
        # monthly-ish sampling (≈120 points)
        n_samples = min(120, max(24, ntime // max(1, int(ntime / 120))))
        step = max(1, ntime // n_samples)
        time_samples = list(range(0, ntime, step))[:120]
    else:
        time_samples = list(range(0, ntime, 8760))[:12]

    nlat = store['data'].shape[3]
    # choose 9 lat bands (dense)
    lat_samples = np.linspace(0, nlat - 1, 9, dtype=int).tolist()

    print(f"Spatial-Temporal Sampling using {len(time_samples)} time samples and {len(lat_samples)} latitude samples")
    regions = { 'bands': lat_samples }

    print("\n--- Atmospheric fields seasonal/spatial sampling (denser) ---")
    for field_idx, field_name in enumerate(fields):
        print(f"\n{field_name} (field {field_idx}):")
        for lat_idx in lat_samples:
            lon_idx = store['data'].shape[-1] // 2
            level_idx = store['data'].shape[2] // 2
            values = []
            for t in time_samples[:36]:  # limit to ~3 years worth for summary
                try:
                    val = store['data'][t, field_idx, level_idx, lat_idx, lon_idx]
                    values.append(val)
                except:
                    values.append(np.nan)
            print(f"  lat~{lat_idx}: mean={np.nanmean(values):.6f} std={np.nanstd(values):.6f} (n={len(values)})")

    print("\n--- Surface fields seasonal/spatial sampling (denser) ---")
    for field_idx, field_name in enumerate(fields_sfc):
        print(f"\n{field_name} (field {field_idx}):")
        for lat_idx in lat_samples:
            lon_idx = store['data_sfc'].shape[-1] // 2
            values = []
            for t in time_samples[:36]:
                try:
                    val = store['data_sfc'][t, field_idx, lat_idx, lon_idx]
                    values.append(val)
                except:
                    values.append(np.nan)
            print(f"  lat~{lat_idx}: mean={np.nanmean(values):.6f} std={np.nanstd(values):.6f} (n={len(values)})")

def check_normalization_consistency(store, fields, fields_sfc, phys_limits=None):
    """Check normalization arrays for consistency and reasonable climatology"""
    norm = store['normalization/norm']
    norm_sfc = store['normalization/norm_sfc']
    print(f"norm shape: {norm.shape}  norm_sfc shape: {norm_sfc.shape}")

    # summarize overall stats for norm (min/max/mean/std)
    def summarize(arr, name, n_display=8):
        a = np.asarray(arr)
        print(f"\n{name} stats: min={np.nanmin(a):.6g} max={np.nanmax(a):.6g} mean={np.nanmean(a):.6g} std={np.nanstd(a):.6g}")
        # check if large fraction are zeros
        frac_zero = float(np.count_nonzero(a == 0)) / a.size
        if frac_zero > 0.5:
            print(f"  ⚠️  More than {frac_zero*100:.1f}% of {name} entries are exactly zero")
        return a

    a_norm = summarize(norm, "norm")
    a_norm_sfc = summarize(norm_sfc, "norm_sfc")

    # per-field quick checks (atmospheric)
    print("\nPer-field norm sanity (atmospheric):")
    for fidx, fname in enumerate(fields):
        try:
            # sample month 0, mean/std for levels and mid spatial point
            sample_vals = []
            for lvl in range(norm.shape[3]):
                v = norm[0, 1, fidx, lvl, norm.shape[-2]//2, norm.shape[-1]//2]
                sample_vals.append(float(v))
            print(f" {fname}: sample mean/std across levels = {np.nanmean(sample_vals):.6g}/{np.nanstd(sample_vals):.6g}")
        except Exception:
            print(f" {fname}: could not sample norm (shape mismatch?)")

    print("\nPer-field norm_sfc sanity (surface):")
    for fidx, fname in enumerate(fields_sfc):
        try:
            sample_vals = []
            for month in range(min(4, norm_sfc.shape[0])):
                meanv = float(norm_sfc[month, 0, fidx, norm_sfc.shape[-2]//2, norm_sfc.shape[-1]//2])
                stdv = float(norm_sfc[month, 1, fidx, norm_sfc.shape[-2]//2, norm_sfc.shape[-1]//2])
                sample_vals.append((meanv, stdv))
            print(f" {fname}: example (mean,std) = {sample_vals[0]} ... {sample_vals[-1]}")
        except Exception:
            print(f" {fname}: could not sample norm_sfc (shape mismatch?)")

    # Optionally compare norm ranges to physical limits if possible (simple heuristic)
    if phys_limits:
        print("\nComparing normalization magnitudes to physical limits (heuristic):")
        for fidx, fname in enumerate(fields):
            lname = fname.lower()
            key = next((k for k in phys_limits if k in lname), None)
            if key is None:
                continue
            lo, hi = phys_limits[key]
            # check whether norms produce reasonable std magnitude (use global_norm if exists)
            try:
                g = store['normalization/global_norm']
                mean_std = float(np.nanmean(g[1, fidx, ...]))
                if mean_std == 0:
                    print(f" {fname}: global std seems zero (suspicious)")
                elif mean_std > (hi - lo) * 0.5:
                    print(f" {fname}: global std ({mean_std:.3g}) is large relative to physical range [{lo},{hi}]")
            except Exception:
                pass
            
def check_corrected_t2m_boundaries(store):
    """Specific checks for corrected_t2m field boundaries"""
    
    print("Corrected T2M Boundary Analysis:")
    
    # Get corrected_t2m field index
    fields_sfc = store.attrs['fields_sfc']
    if 'corrected_t2m' not in fields_sfc:
        print("  corrected_t2m not found in fields_sfc")
        return
        
    t2m_idx = fields_sfc.index('corrected_t2m')
    
    # **1. Check valid region (lat indices 0-70)**
    print(f"\n--- Valid Region (lat indices 0-70) ---")
    valid_samples = []
    for lat_idx in [10, 35, 60]:  # Arctic region
        for lon_idx in [0, 720, 1439]:  # Various longitudes
            try:
                val = store['data_sfc'][100, t2m_idx, lat_idx, lon_idx]  # Sample timestep
                valid_samples.append(val)
            except:
                valid_samples.append(np.nan)
    
    valid_zeros = sum(1 for x in valid_samples if x == 0)
    valid_nans = sum(1 for x in valid_samples if np.isnan(x))
    print(f"  Valid region samples: {len(valid_samples)} points")
    print(f"  Zeros: {valid_zeros} | NaNs: {valid_nans}")
    print(f"  Sample values: {valid_samples[:3]} ...")
    
    # **2. Check invalid region (lat indices 71+)**
    print(f"\n--- Invalid Region (lat indices 71+) - Should be zeros/NaNs ---")
    invalid_samples = []
    for lat_idx in [350, 500, 700]:  # Mid/Antarctic regions
        for lon_idx in [0, 720, 1439]:
            try:
                val = store['data_sfc'][100, t2m_idx, lat_idx, lon_idx]
                invalid_samples.append(val)
            except:
                invalid_samples.append(np.nan)
    
    invalid_zeros = sum(1 for x in invalid_samples if x == 0)
    invalid_nans = sum(1 for x in invalid_samples if np.isnan(x))
    nonzero_unexpected = sum(1 for x in invalid_samples if x != 0 and not np.isnan(x))
    
    print(f"  Invalid region samples: {len(invalid_samples)} points")
    print(f"  Zeros: {invalid_zeros} | NaNs: {invalid_nans} | Unexpected non-zeros: {nonzero_unexpected}")
    
    if nonzero_unexpected > 0:
        print(f"  ⚠️  WARNING: Found {nonzero_unexpected} unexpected non-zero values in invalid region!")
        print(f"  Sample values: {invalid_samples}")
    else:
        print(f"  ✓ Invalid region correctly contains only zeros/NaNs")
    
    # **3. Check corresponding norms**
    print(f"\n--- Corrected T2M Normalization Check ---")
    norm_sfc = store['normalization/norm_sfc']
    
    # Valid region norms
    valid_norm_mean = norm_sfc[0, 0, t2m_idx, 35, 720]  # Month 0, mean, field t2m_idx
    valid_norm_std = norm_sfc[0, 1, t2m_idx, 35, 720]   # Month 0, std
    
    # Invalid region norms  
    invalid_norm_mean = norm_sfc[0, 0, t2m_idx, 500, 720]
    invalid_norm_std = norm_sfc[0, 1, t2m_idx, 500, 720]
    
    print(f"  Valid region norms (lat 35): mean={valid_norm_mean:.3f}, std={valid_norm_std:.3f}")
    print(f"  Invalid region norms (lat 500): mean={invalid_norm_mean:.3f}, std={invalid_norm_std:.3f}")
    
    if invalid_norm_mean != 0 or invalid_norm_std != 0:
        print(f"  ⚠️  WARNING: Invalid region has non-zero norms!")
    else:
        print(f"  ✓ Invalid region norms are correctly zero")

if __name__ == "__main__":
    #zarr_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr"
    #zarr_path = "/work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr"
    zarr_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr"
    #zarr_path = "/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr"
    comprehensive_zarr_analysis(zarr_path)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
