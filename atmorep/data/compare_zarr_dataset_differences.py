import argparse
import gzip
import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import zarr


def human(n):
    for u in ('B','KB','MB','GB','TB','PB'):
        if abs(n) < 1024.0:
            return f"{n:.2f}{u}"
        n /= 1024.0
    return f"{n:.2f}EB"


def chunk_counts(shape, chunks):
    return tuple(math.ceil(s / c) for s, c in zip(shape, chunks))


def chunk_slice_from_index(idx_tuple, chunks, shape):
    slices = []
    for idx, c, s in zip(idx_tuple, chunks, shape):
        start = idx * c
        end = min((idx + 1) * c, s)
        slices.append(slice(start, end))
    return tuple(slices)


def compare_blocks(src_block, dst_block, rtol, atol):
    # return boolean mask of differences (True where different)
    if src_block.shape != dst_block.shape:
        raise ValueError("Block shapes differ")
    if np.issubdtype(src_block.dtype, np.floating) or np.issubdtype(dst_block.dtype, np.floating):
        # treat NaN==NaN as equal
        with np.errstate(invalid='ignore'):
            close = np.isclose(src_block, dst_block, rtol=rtol, atol=atol, equal_nan=True)
        diff_mask = ~close
    else:
        diff_mask = ~(src_block == dst_block)
    return diff_mask


def load_coord(arr_group, name):
    try:
        if name in arr_group:
            return np.asarray(arr_group[name][:])
    except Exception:
        pass
    return None


def map_index_to_coords(store, abs_idx, array_name):
    # naive mapping heuristics
    coords = {}
    try:
        if 'time' in store:
            time = np.asarray(store['time'][:])
            if abs_idx[0] < len(time):
                coords['time'] = str(time[abs_idx[0]])
    except Exception:
        pass
    # data_sfc: (time, field_sfc, lat, lon) -> lat index -2, lon -1
    # data: (time, field, level, lat, lon)
    try:
        lats = np.asarray(store['lats'][:]) if 'lats' in store else None
        lons = np.asarray(store['lons'][:]) if 'lons' in store else None
        if lats is not None and lons is not None:
            if len(abs_idx) >= 2:
                lat_idx = abs_idx[-2]
                lon_idx = abs_idx[-1]
                if 0 <= lat_idx < len(lats):
                    coords['lat'] = float(lats[lat_idx])
                if 0 <= lon_idx < len(lons):
                    coords['lon'] = float(lons[lon_idx])
    except Exception:
        pass
    # fields & levels from attrs if present
    try:
        fields = store.attrs.get('fields') or store.attrs.get('fields_sfc')
        levels = store.attrs.get('levels')
        if fields is not None and len(abs_idx) >= 4:
            # best-effort: for data_sfc fields at axis 1, for data axis 1 or 2...
            # attempt mapping by length
            if len(abs_idx) == 4:
                field_idx = abs_idx[1]
                if 0 <= field_idx < len(fields):
                    coords['field'] = fields[field_idx]
            elif len(abs_idx) == 5:
                field_idx = abs_idx[1]
                level_idx = abs_idx[2]
                if 0 <= field_idx < len(fields):
                    coords['field'] = fields[field_idx]
                if levels and 0 <= level_idx < len(levels):
                    coords['level'] = levels[level_idx]
    except Exception:
        pass
    return coords


def main():
    p = argparse.ArgumentParser(description="Pointwise compare two Zarr stores (chunked). Outputs gzipped CSV of differences.")
    p.add_argument("src", help="source zarr (will be compared TO dst)")
    p.add_argument("dst", help="destination zarr")
    p.add_argument("--arrays", nargs="*", default=None,
                   help="arrays to compare (default: common top-level arrays: data, data_sfc, normalization/norm, normalization/norm_sfc)")
    p.add_argument("--out", default="/scratch/a/a270277/atmorep/compare_zarr_diffs.csv.gz", help="output gz CSV")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-diffs-per-array", type=int, default=0, help="stop after this many diffs per array (0=unlimited)")
    p.add_argument("--progress-every", type=int, default=1000, help="print progress every N chunks")
    args = p.parse_args()

    src_root = zarr.open_group(str(Path(args.src)), mode='r')
    dst_root = zarr.open_group(str(Path(args.dst)), mode='r')

    # determine arrays to compare
    if args.arrays:
        arr_names = args.arrays
    else:
        common = sorted(set(src_root.array_keys()) & set(dst_root.array_keys()))
        # prefer common arrays of interest
        prefer = ['data', 'data_sfc', 'normalization/norm', 'normalization/norm_sfc', 'normalization/global_norm', 'normalization/global_norm_sfc']
        arr_names = [a for a in prefer if a in common] + [a for a in common if a not in prefer]

    print(f"Comparing arrays: {arr_names}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ['array', 'abs_index', 'src_value', 'dst_value', 'coords_json']
    f = gzip.open(str(out_path), 'wt', newline='')
    writer = csv.writer(f)
    writer.writerow(header)

    for arr_name in arr_names:
        if arr_name not in src_root or arr_name not in dst_root:
            print(f"Skipping {arr_name} (missing in one store)")
            continue
        print(f"\n--- array: {arr_name} ---")
        src_arr = src_root[arr_name]
        dst_arr = dst_root[arr_name]
        shape = src_arr.shape
        if shape != dst_arr.shape:
            print(f"  shape mismatch src{shape} dst{dst_arr.shape} -> skipping")
            continue
        chunks = src_arr.chunks
        if chunks is None:
            # fallback: treat as single chunk
            chunks = shape
        counts = chunk_counts(shape, chunks)
        total_chunks = math.prod(counts)
        print(f"  shape={shape} chunks={chunks} chunk_counts={counts} total_chunks={total_chunks}")

        diffs_found = 0
        chunk_idx = 0
        # iterate over chunk indices
        for idx_tuple in product(*(range(c) for c in counts)):
            chunk_idx += 1
            if chunk_idx % args.progress_every == 0:
                print(f"  processed {chunk_idx}/{total_chunks} chunks; diffs so far {diffs_found}")
            sl = chunk_slice_from_index(idx_tuple, chunks, shape)
            try:
                src_block = np.asarray(src_arr[sl])
            except Exception as e:
                print(f"ERROR reading src {arr_name} slice {sl}: {e}")
                continue
            try:
                dst_block = np.asarray(dst_arr[sl])
            except Exception as e:
                print(f"ERROR reading dst {arr_name} slice {sl}: {e}")
                continue
            # quick shortcut: if blocks identical as bytes then skip
            try:
                if src_block.dtype == dst_block.dtype and src_block.size == dst_block.size and np.array_equal(src_block, dst_block):
                    continue
            except Exception:
                pass
            # compute diff mask
            diff_mask = compare_blocks(src_block, dst_block, args.rtol, args.atol)
            if not np.any(diff_mask):
                continue
            # get coordinates of differences
            rel_indices = np.nonzero(diff_mask)
            # rel_indices are tuples of arrays
            for k in range(len(rel_indices[0])):
                rel_idx = tuple(int(arr[k]) for arr in rel_indices)
                abs_idx = tuple(int(s.start + r) for s, r in zip(sl, rel_idx))
                sval = src_block[rel_idx].item()
                dval = dst_block[rel_idx].item()
                coords = map_index_to_coords(src_root, abs_idx, arr_name)
                writer.writerow([arr_name, json.dumps(abs_idx), repr(sval), repr(dval), json.dumps(coords)])
                diffs_found += 1
                if 0 < args.max_diffs_per_array <= diffs_found:
                    break
            if 0 < args.max_diffs_per_array <= diffs_found:
                print(f"  reached max diffs for {arr_name} ({diffs_found})")
                break
        print(f"Finished array {arr_name}: diffs found {diffs_found}")
    f.close()
    print(f"\nAll done. Differences written to: {out_path}")


if __name__ == "__main__":
    main()