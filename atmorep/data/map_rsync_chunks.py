import re
import csv
import argparse
from pathlib import Path
import zarr
import numpy as np
import sys
import os
import ast

line_re = re.compile(r'^(?P<flag>\*deleting|>).*?\s+(?P<path>.+)$')
chunk_re = re.compile(r'^(?P<array>[^/]+)/(?P<chunk>[-\d\.]+)$')

def parse_lines(source):
    if source == '-':
        for L in sys.stdin:
            yield L.rstrip("\n")
    else:
        yield from Path(source).read_text().splitlines()

def parse_chunk(rel):
    m = chunk_re.match(rel)
    if not m:
        return None, None
    arr = m.group('array')
    try:
        tup = tuple(int(x) for x in m.group('chunk').split('.'))
    except Exception:
        tup = None
    return arr, tup

def chunk_to_slice(tup, chunks, shape):
    rng = []
    for idx, c, s in zip(tup, chunks, shape):
        start = idx * c
        end = min((idx+1)*c, s)  # exclusive end
        rng.append((start, end-1))
    return tuple(rng)

def main():
    p = argparse.ArgumentParser(description="Map rsync itemize output chunk paths -> index ranges + coords (also support CSV input)")
    p.add_argument("rsync_output", nargs="?", default="/tmp/rsync_diff.txt",
                   help="rsync itemize output file path (use '-' to read stdin). Default /tmp/rsync_diff.txt")
    p.add_argument("--src", required=True, help="SRC zarr root (rsync source)")
    p.add_argument("--dst", required=True, help="DST zarr root (rsync dest)")
    p.add_argument("--out", default="/scratch/a/a270277/atmorep/rsync_chunk_map.csv",
                   help="output CSV path (default in scratch)")
    p.add_argument("--dump-first-outdir", default="/scratch/a/a270277/atmorep/first_chunk_dump",
                   help="directory to write the first src/dst chunk .npy and summary (default in scratch)")
    args = p.parse_args()

    raw_lines = list(parse_lines(args.rsync_output))

    for line in raw_lines[:10]:
        print(f"  {line.rstrip()}")
    if not raw_lines:
        print("No input lines read.", file=sys.stderr)
        return

    src_store = zarr.open_group(args.src, mode='r')
    dst_store = zarr.open_group(args.dst, mode='r')

    # load coords if present (prefer SRC)
    time = np.asarray(src_store['time'][:]) if 'time' in src_store else (np.asarray(dst_store['time'][:]) if 'time' in dst_store else None)
    lats = np.asarray(src_store['lats'][:]) if 'lats' in src_store else (np.asarray(dst_store['lats'][:]) if 'lats' in dst_store else None)
    lons = np.asarray(src_store['lons'][:]) if 'lons' in src_store else (np.asarray(dst_store['lons'][:]) if 'lons' in dst_store else None)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_first_outdir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    first_dump_done = False

    # If input looks like CSV (header starting with "flag,") handle with csv.DictReader
    is_csv_input = False
    first_line = raw_lines[0].lstrip()
    if first_line.startswith("flag,") or first_line.startswith('"flag",') or first_line.count(',') and 'relpath' in first_line:
        is_csv_input = True
    if is_csv_input:
        print("Detected CSV input format; parsing with csv.DictReader")
        # iterate rows from CSV; still write an output CSV (copy rows) but also trigger first-deleting dump
        reader = csv.DictReader(raw_lines)
        with outp.open('w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(reader.fieldnames)
            for row in reader:
                writer.writerow([row.get(h, "") for h in reader.fieldnames])
                if (not first_dump_done) and row.get('flag', '') == '*deleting':
                    first_dump_done = True
                    # parse fields
                    rel = row.get('relpath', '')
                    arr_name = row.get('array', '')
                    try:
                        tup = ast.literal_eval(row.get('chunk_tuple', 'None'))
                        ranges = ast.literal_eval(row.get('index_ranges', 'None'))
                    except Exception:
                        tup = None
                        ranges = None
                    if tup is None or ranges is None:
                        print("CSV row lacks parsable chunk_tuple/index_ranges; cannot dump", rel, file=sys.stderr)
                        continue
                    # build slice objects
                    sl = tuple(slice(r[0], r[1]+1) for r in ranges)
                    safe_name = rel.replace('/', '_').replace('"','').replace(' ','_')
                    src_path = dump_dir / f"{safe_name}_src.npy"
                    dst_path = dump_dir / f"{safe_name}_dst.npy"
                    summary_path = dump_dir / f"{safe_name}_summary.txt"
                    src_present = arr_name in src_store
                    dst_present = arr_name in dst_store
                    summary_lines = [f"CSV-mode input", f"relpath: {rel}", f"array: {arr_name}", f"chunk_tuple: {tup}", f"index_ranges: {ranges}"]
                    try:
                        if src_present:
                            src_block = np.asarray(src_store[arr_name][sl])
                            np.save(str(src_path), src_block)
                            summary_lines.append(f"wrote src chunk to {src_path}  shape={src_block.shape} dtype={src_block.dtype}")
                        else:
                            summary_lines.append("src chunk not present")
                    except Exception as e:
                        summary_lines.append(f"src read error: {e}")
                    try:
                        if dst_present:
                            dst_block = np.asarray(dst_store[arr_name][sl])
                            np.save(str(dst_path), dst_block)
                            summary_lines.append(f"wrote dst chunk to {dst_path}  shape={dst_block.shape} dtype={dst_block.dtype}")
                        else:
                            summary_lines.append("dst chunk not present")
                    except Exception as e:
                        summary_lines.append(f"dst read error: {e}")
                    # diff summary if both present
                    if src_present and dst_present:
                        try:
                            sa = src_block.astype(np.float64)
                            da = dst_block.astype(np.float64)
                            neq = ~np.isclose(sa, da, rtol=0, atol=0, equal_nan=True)
                            n_diff = int(np.count_nonzero(neq))
                            summary_lines.append(f"n_diff_elements (exact): {n_diff} of {sa.size}")
                            if n_diff:
                                diffs = np.argwhere(neq)
                                for idx in diffs[:10]:
                                    idxt = tuple(int(i) for i in idx)
                                    vs = sa[idxt].item(); vd = da[idxt].item()
                                    summary_lines.append(f"diff idx(rel):{idxt} src={vs} dst={vd} delta={vs-vd}")
                        except Exception as e:
                            summary_lines.append(f"diff calc error: {e}")
                    try:
                        summary_path.write_text("\n".join(summary_lines))
                    except Exception:
                        pass
        if first_dump_done:
            print(f"First-deleting chunk saved to {dump_dir}. Check *_src.npy *_dst.npy and *_summary.txt")
        else:
            print("No *deleting row found in CSV input; no dump created.")
        return

    # Non-CSV mode: parse raw rsync lines with regex (existing behavior)
    with outp.open('w', newline='') as fh:
        print(f"Writing output CSV to: {outp}")
        w = csv.writer(fh)
        w.writerow(["flag","relpath","array","chunk_tuple","index_ranges","time_start","time_end","lat_start","lat_end","lon_start","lon_end","note"])
        for L in raw_lines:
            m = line_re.match(L.strip())
            if not m:
                continue
            flag = m.group('flag')
            rel = m.group('path').strip()
            if not (flag == '*deleting' or flag.startswith('>')):
                continue
            arr_name, tup = parse_chunk(rel)
            note = ""
            if tup is None:
                w.writerow([flag, rel, None, None, None, None, None, None, None, None, "not_chunk_or_unparsable"])
                continue
            store = src_store if arr_name in src_store else (dst_store if arr_name in dst_store else None)
            if store is None:
                w.writerow([flag, rel, arr_name, repr(tup), None, None, None, None, None, None, "array_missing_in_both"])
                continue
            try:
                arr = store[arr_name]
                chunks = arr.chunks or arr.shape
                shape = arr.shape
                ranges = chunk_to_slice(tup, chunks, shape)
                time_start = time_end = ""
                try:
                    t0, t1 = ranges[0]
                    if time is not None and 0 <= t0 < len(time):
                        time_start = str(time[t0])
                    if time is not None and 0 <= t1 < len(time):
                        time_end = str(time[t1])
                except Exception:
                    pass
                lat_start = lat_end = lon_start = lon_end = ""
                try:
                    lat_rng = ranges[-2]; lon_rng = ranges[-1]
                    if lats is not None and 0 <= lat_rng[0] < len(lats):
                        lat_start = float(lats[lat_rng[0]])
                    if lats is not None and 0 <= lat_rng[1] < len(lats):
                        lat_end = float(lats[lat_rng[1]])
                    if lons is not None and 0 <= lon_rng[0] < len(lons):
                        lon_start = float(lons[lon_rng[0]])
                    if lons is not None and 0 <= lon_rng[1] < len(lons):
                        lon_end = float(lons[lon_rng[1]])
                except Exception:
                    pass
                w.writerow([flag, rel, arr_name, repr(tup), repr(ranges), time_start, time_end, lat_start, lat_end, lon_start, lon_end, note])

                if (flag == '*deleting') and (not first_dump_done):
                    first_dump_done = True
                    sl = tuple(slice(r[0], r[1]+1) for r in ranges)
                    safe_name = rel.replace('/', '_').replace('"','').replace(' ','_')
                    src_path = dump_dir / f"{safe_name}_src.npy"
                    dst_path = dump_dir / f"{safe_name}_dst.npy"
                    summary_path = dump_dir / f"{safe_name}_summary.txt"
                    src_present = arr_name in src_store
                    dst_present = arr_name in dst_store
                    summary_lines = [f"relpath: {rel}", f"array: {arr_name}", f"chunk_tuple: {tup}", f"index_ranges: {ranges}"]
                    try:
                        if src_present:
                            src_block = np.asarray(src_store[arr_name][sl])
                            np.save(str(src_path), src_block)
                            summary_lines.append(f"wrote src chunk to {src_path}  shape={src_block.shape} dtype={src_block.dtype}")
                        else:
                            summary_lines.append("src chunk not present")
                    except Exception as e:
                        summary_lines.append(f"src read error: {e}")
                    try:
                        if dst_present:
                            dst_block = np.asarray(dst_store[arr_name][sl])
                            np.save(str(dst_path), dst_block)
                            summary_lines.append(f"wrote dst chunk to {dst_path}  shape={dst_block.shape} dtype={dst_block.dtype}")
                        else:
                            summary_lines.append("dst chunk not present")
                    except Exception as e:
                        summary_lines.append(f"dst read error: {e}")
                    if src_present and dst_present:
                        try:
                            sa = src_block.astype(np.float64); da = dst_block.astype(np.float64)
                            neq = ~np.isclose(sa, da, rtol=0, atol=0, equal_nan=True)
                            n_diff = int(np.count_nonzero(neq))
                            summary_lines.append(f"n_diff_elements (exact): {n_diff} of {sa.size}")
                            if n_diff:
                                diffs = np.argwhere(neq)
                                for idx in diffs[:1000]:
                                    idxt = tuple(int(i) for i in idx); vs = sa[idxt].item(); vd = da[idxt].item()
                                    summary_lines.append(f"diff idx(rel):{idxt} src={vs} dst={vd} delta={vs-vd}")
                        except Exception as e:
                            summary_lines.append(f"diff calc error: {e}")
                    try:
                        summary_path.write_text("\n".join(summary_lines))
                    except Exception:
                        pass

            except Exception as e:
                w.writerow([flag, rel, arr_name, repr(tup), None, None, None, None, None, None, f"error:{e}"])
    print(f"Wrote mapping to {outp}")
    if first_dump_done:
        print(f"First-deleting chunk saved to {dump_dir}. Check *_src.npy *_dst.npy and *_summary.txt")

if __name__ == "__main__":
    main()

'''    
rsync -avhn --itemize-changes --delete \
  /scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr/ \
  /scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr/ \
| python /work/ab1412/atmorep/data/map_rsync_chunks.py - \
    --src /scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr \
    --dst /scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr \
    --out /scratch/a/a270277/atmorep/rsync_chunk_map.csv
    '''