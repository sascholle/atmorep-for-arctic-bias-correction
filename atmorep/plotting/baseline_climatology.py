#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def parse_args():
    p = argparse.ArgumentParser(
        description="Empirical Arctic climatology baseline from corrected_t2m Zarr."
    )
    p.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr"),
    )
    p.add_argument(
        "--year-base",
        type=int,
        default=2010,
        help="Base year used for normalization month index",
    )
    p.add_argument(
        "--start-time",
        type=str,
        default="2021-01-01",
        help="Optional inclusive start time, e.g. 2019-01-01",
    )
    p.add_argument(
        "--end-time",
        type=str,
        default="2021-12-31",
        help="Optional inclusive end time, e.g. 2020-12-31",
    )
    p.add_argument("--t-chunk", type=int, default=24, help="Time chunk size")
    p.add_argument(
        "--use-area-weights",
        action="store_true",
        help="If set, compute cosine(latitude)-weighted metrics",
    )

    # Option A: threshold-based Arctic selection
    p.add_argument(
        "--lat-min",
        type=float,
        default=None,
        help="Latitude threshold selection, e.g. 66.5. Ignored if --lat-start/--lat-end are set.",
    )

    # Option B: exact index windows (end-exclusive, like Python range)
    p.add_argument(
        "--lat-start",
        type=int,
        default=0,
        help="Latitude start index (inclusive).",
    )
    p.add_argument(
        "--lat-end",
        type=int,
        default=71,
        help="Latitude end index (exclusive).",
    )
    p.add_argument(
        "--lon-start",
        type=int,
        default=0,
        help="Longitude start index (inclusive).",
    )
    p.add_argument(
        "--lon-end",
        type=int,
        default=1440,
        help="Longitude end index (exclusive).",
    )
    return p.parse_args()


def month_index_from_time(times_s, year_base):
    # times_s: datetime64[s] array, shape [T]
    months = times_s.astype("datetime64[M]")
    years = months.astype("datetime64[Y]").astype(np.int64) + 1970
    month_in_year = months.astype(np.int64) % 12  # 0..11
    m_idx = (years - year_base) * 12 + month_in_year
    return m_idx.astype(np.int64)


def resolve_spatial_indices(args, lats, nlon):
    use_index_window = (
        args.lat_start is not None
        or args.lat_end is not None
        or args.lon_start is not None
        or args.lon_end is not None
    )

    if use_index_window:
        if args.lat_start is None or args.lat_end is None:
            raise RuntimeError("For index-window mode, provide both --lat-start and --lat-end.")
        if args.lon_start is None or args.lon_end is None:
            raise RuntimeError("For index-window mode, provide both --lon-start and --lon-end.")

        if not (0 <= args.lat_start < args.lat_end <= lats.shape[0]):
            raise RuntimeError(
                f"Invalid latitude index range [{args.lat_start}, {args.lat_end}) for nlat={lats.shape[0]}"
            )
        if not (0 <= args.lon_start < args.lon_end <= nlon):
            raise RuntimeError(
                f"Invalid longitude index range [{args.lon_start}, {args.lon_end}) for nlon={nlon}"
            )

        lat_idx = np.arange(args.lat_start, args.lat_end, dtype=np.int64)
        lon_idx = np.arange(args.lon_start, args.lon_end, dtype=np.int64)

        selection_desc = {
            "mode": "index_window",
            "lat_start": int(args.lat_start),
            "lat_end_exclusive": int(args.lat_end),
            "lon_start": int(args.lon_start),
            "lon_end_exclusive": int(args.lon_end),
            "n_lat": int(lat_idx.size),
            "n_lon": int(lon_idx.size),
        }
        return lat_idx, lon_idx, selection_desc

    # Fallback: lat-min threshold
    lat_min = 66.5 if args.lat_min is None else args.lat_min
    lat_idx = np.where(lats >= lat_min)[0].astype(np.int64)
    if lat_idx.size == 0:
        raise RuntimeError(f"No latitudes >= {lat_min}")
    lon_idx = np.arange(nlon, dtype=np.int64)

    selection_desc = {
        "mode": "lat_min",
        "lat_min": float(lat_min),
        "n_lat": int(lat_idx.size),
        "n_lon": int(lon_idx.size),
    }
    return lat_idx, lon_idx, selection_desc


def main():
    args = parse_args()
    root = zarr.open_group(str(args.zarr_path), mode="r")

    fields_sfc = list(root.attrs["fields_sfc"])
    if "corrected_t2m" not in fields_sfc:
        raise RuntimeError(f"corrected_t2m not in fields_sfc: {fields_sfc}")
    field_idx = fields_sfc.index("corrected_t2m")

    data_sfc = root["data_sfc"]  # [time, field, lat, lon]
    norm_sfc = root["normalization"]["norm_sfc"]  # [month, stat, field, lat, lon]
    lats = root["lats"][:].astype(np.float64)  # [lat]
    times = root["time"][:]  # datetime64[s], [time]

    nlon = data_sfc.shape[3]
    lat_idx, lon_idx, selection_desc = resolve_spatial_indices(args, lats, nlon)

    tmask = np.ones(times.shape[0], dtype=bool)
    if args.start_time is not None:
        tmask &= times >= np.datetime64(args.start_time)
    if args.end_time is not None:
        tmask &= times <= np.datetime64(args.end_time)

    time_idx = np.where(tmask)[0]
    if time_idx.size == 0:
        raise RuntimeError("No timesteps selected by time filter.")

    # Latitude weights only vary by latitude; broadcast to [1, lat, 1]
    if args.use_area_weights:
        w_lat = np.cos(np.deg2rad(lats[lat_idx])).astype(np.float64)
        w_lat_3d = w_lat[None, :, None]
    else:
        w_lat_3d = None

    sum_sq_norm = 0.0
    sum_sq_phys = 0.0
    sum_w = 0.0

    for i0 in range(0, time_idx.size, args.t_chunk):
        sel = time_idx[i0 : i0 + args.t_chunk]
        t_chunk = times[sel]  # [Tc]
        m_idx = month_index_from_time(t_chunk, args.year_base)  # [Tc]

        if np.any(m_idx < 0) or np.any(m_idx >= norm_sfc.shape[0]):
            bad = m_idx[(m_idx < 0) | (m_idx >= norm_sfc.shape[0])]
            raise RuntimeError(
                f"Month index out of range. Got {bad[:10]} "
                f"(year_base={args.year_base}, norm months={norm_sfc.shape[0]})."
            )

        # Data/climatology slices: [Tc, lat, lon]
        y = data_sfc.oindex[sel, field_idx, lat_idx, lon_idx].astype(np.float64)
        mu = norm_sfc.oindex[m_idx, 0, field_idx, lat_idx, lon_idx].astype(np.float64)
        sd = norm_sfc.oindex[m_idx, 1, field_idx, lat_idx, lon_idx].astype(np.float64)

        # Residual in physical space
        r_phys = y - mu

        # Residual in normalized space (climatology predicts 0)
        valid = np.isfinite(r_phys) & np.isfinite(sd) & (sd > 0.0)
        z = np.zeros_like(r_phys, dtype=np.float64)
        z[valid] = r_phys[valid] / sd[valid]

        if w_lat_3d is None:
            w = valid.astype(np.float64)
        else:
            w = valid.astype(np.float64) * w_lat_3d

        sum_sq_norm += np.sum((z ** 2) * w)
        sum_sq_phys += np.sum((r_phys ** 2) * w)
        sum_w += np.sum(w)

    if sum_w == 0:
        raise RuntimeError("No valid points after masking.")

    mse_norm = sum_sq_norm / sum_w
    mse_phys = sum_sq_phys / sum_w
    rmse_phys = np.sqrt(mse_phys)

    out = {
        "zarr_path": str(args.zarr_path),
        "field": "corrected_t2m",
        "time_start": str(times[time_idx[0]]),
        "time_end": str(times[time_idx[-1]]),
        "n_timesteps": int(time_idx.size),
        "area_weighted": bool(args.use_area_weights),
        "selection": selection_desc,
        "empirical_climatology_mse_normalized": float(mse_norm),
        "empirical_climatology_mse_K2": float(mse_phys),
        "empirical_climatology_rmse_K": float(rmse_phys),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()