#!/usr/bin/env python3
"""
Plot sparsity vs performance from AtmoRep SLURM output logs.

What it does:
- Parses output_<jobid>.txt files from /work/ab1412/atmorep/output
- Extracts all validation loss lines for corrected_t2m
- Supports three comparison modes:
  - last: last validation point in the log
  - best: minimum validation loss seen in the log
  - common: uses a requested epoch if every run reached it
- Writes a CSV summary and a PNG scatter plot
- Writes an epoch-wise filled-variance plot grouped by sparsity


Recommended metric:
- Use validation MSE on corrected_t2m as the primary score.
- RMSE is computed as sqrt(MSE) for interpretability.
- If you want the most scientifically defensible single number, compare runs at
  the same checkpoint epoch or use best validation on a fixed validation set with
  identical stopping rules. Rolling averages are useful for smoothing curves, but
  not ideal as the final reported metric.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib#
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("/work/ab1412/atmorep/output")
DEFAULT_PLOT_DIR = Path("/work/ab1412/atmorep/plotting")
DEFAULT_CSV_PATH = DEFAULT_PLOT_DIR / "sparsity_vs_performance_summary.csv"
DEFAULT_PNG_PATH = DEFAULT_PLOT_DIR / "sparsity_vs_performance_common.png"
DEFAULT_EPOCH_PNG_PATH = DEFAULT_PLOT_DIR / "sparsity_over_epochs_filled.png"
DEFAULT_COMMON_AREA_PNG_PATH = DEFAULT_PLOT_DIR / "sparsity_common_epoch_area.png"


# Map your run ids to sparsity values.
RUN_SPECS = [
    {"job_id": 23978827, "sparsity": 0.00},
    {"job_id": 23994917, "sparsity": 0.25},
    {"job_id": 23647012, "sparsity": 0.50},
    {"job_id": 23995579, "sparsity": 0.75},
    {"job_id": 24012088, "sparsity": 0.85},
    {"job_id": 24025102, "sparsity": 0.95},

    {"job_id": 24075112, "sparsity": 0.25},
    {"job_id": 24106052, "sparsity": 0.50},
    {"job_id": 24075176, "sparsity": 0.75},
    {"job_id": 24106063, "sparsity": 0.85},

    {"job_id": 24106664, "sparsity": 0.25},
    #{"job_id": 24106669, "sparsity": 0.50},
    {"job_id": 24106671, "sparsity": 0.95},
    #{"job_id": 24106672, "sparsity": 0.95},

]



@dataclass
class EpochResult:
    epoch: int
    strategy_loss: Optional[float]
    corrected_t2m_loss: Optional[float]
    valid_points: Optional[float]


@dataclass
class RunSummary:
    job_id: int
    sparsity: float
    log_path: Path
    last_epoch: Optional[int]
    last_mse: Optional[float]
    last_rmse: Optional[float]
    best_epoch: Optional[int]
    best_mse: Optional[float]
    best_rmse: Optional[float]
    common_epoch_mse: Optional[float]
    common_epoch_rmse: Optional[float]
    common_epoch: Optional[int]


VAL_STRATEGY_RE = re.compile(
    r"validation loss for strategy=BERT at epoch\s+(-?\d+)\s*:\s*([0-9eE+\-.]+)"
)
VAL_CORRECTED_RE = re.compile(
    r"validation loss for corrected_t2m \((?:Arctic|.*?),\s*([0-9,\.]+)\s*valid points\)\s*:\s*([0-9eE+\-.]+)"
)
VAL_CORRECTED_SIMPLE_RE = re.compile(
    r"validation loss for corrected_t2m\s*:\s*([0-9eE+\-.]+)"
)
EPOCH_LINE_RE = re.compile(r"epoch:\s+(-?\d+)\s+\[(\d+)/(\d+)\s+\((\d+)%\)\]")

SPARSITY_RE = re.compile(r"sparse_target_sparsity\s*:\s*([0-9eE+\-.]+)")

def extract_sparsity_from_log(log_path: Path) -> Optional[float]:
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            m = SPARSITY_RE.search(line)
            if m:
                return float(m.group(1))
    return None

def parse_output_log(log_path: Path) -> List[EpochResult]:
    epoch_results: List[EpochResult] = []
    current_epoch: Optional[int] = None
    current_strategy_loss: Optional[float] = None
    current_corrected_loss: Optional[float] = None
    current_valid_points: Optional[float] = None

    # Rebase resumed logs so epochs remain monotonic across restarts.
    epoch_offset = 0
    last_physical_epoch: Optional[int] = None
    max_logical_epoch = -1

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            epoch_match = VAL_STRATEGY_RE.search(line)
            if epoch_match:
                physical_epoch = int(epoch_match.group(1))

                # If the log restarted, epoch usually drops back to -1.
                # Detect that and shift the new segment forward.
                if last_physical_epoch is not None and physical_epoch < last_physical_epoch:
                    epoch_offset = max_logical_epoch + 1

                current_epoch = physical_epoch + epoch_offset
                last_physical_epoch = physical_epoch
                max_logical_epoch = max(max_logical_epoch, current_epoch)

                current_strategy_loss = float(epoch_match.group(2))
                current_corrected_loss = None
                current_valid_points = None
                continue

            corrected_match = VAL_CORRECTED_RE.search(line)
            if corrected_match and current_epoch is not None:
                current_valid_points = float(corrected_match.group(1).replace(",", ""))
                current_corrected_loss = float(corrected_match.group(2))
                epoch_results.append(
                    EpochResult(
                        epoch=current_epoch,
                        strategy_loss=current_strategy_loss,
                        corrected_t2m_loss=current_corrected_loss,
                        valid_points=current_valid_points,
                    )
                )
                continue

            simple_match = VAL_CORRECTED_SIMPLE_RE.search(line)
            if simple_match and current_epoch is not None:
                current_corrected_loss = float(simple_match.group(1))
                if not epoch_results or epoch_results[-1].epoch != current_epoch:
                    epoch_results.append(
                        EpochResult(
                            epoch=current_epoch,
                            strategy_loss=current_strategy_loss,
                            corrected_t2m_loss=current_corrected_loss,
                            valid_points=current_valid_points,
                        )
                    )
                else:
                    epoch_results[-1] = EpochResult(
                        epoch=current_epoch,
                        strategy_loss=current_strategy_loss,
                        corrected_t2m_loss=current_corrected_loss,
                        valid_points=current_valid_points,
                    )

    return epoch_results


def summarize_run(job_id: int, sparsity: float, log_path: Path, common_epoch: Optional[int]) -> RunSummary:
    epoch_results = parse_output_log(log_path)

    last = None
    if epoch_results:
        last = epoch_results[-1]

    best = None
    if epoch_results:
        candidates = [item for item in epoch_results if item.corrected_t2m_loss is not None]
        if candidates:
            best = min(candidates, key=lambda item: item.corrected_t2m_loss)

    common = None
    if common_epoch is not None:
        for item in epoch_results:
            if item.epoch == common_epoch and item.corrected_t2m_loss is not None:
                common = item
                break

    def to_rmse(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return math.sqrt(value)

    return RunSummary(
        job_id=job_id,
        sparsity=sparsity,
        log_path=log_path,
        last_epoch=last.epoch if last else None,
        last_mse=last.corrected_t2m_loss if last else None,
        last_rmse=to_rmse(last.corrected_t2m_loss) if last else None,
        best_epoch=best.epoch if best else None,
        best_mse=best.corrected_t2m_loss if best else None,
        best_rmse=to_rmse(best.corrected_t2m_loss) if best else None,
        common_epoch_mse=common.corrected_t2m_loss if common else None,
        common_epoch_rmse=to_rmse(common.corrected_t2m_loss) if common else None,
        common_epoch=common_epoch,
    )


def write_csv(summaries: List[RunSummary], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "job_id",
                "sparsity",
                "log_path",
                "last_epoch",
                "last_mse",
                "last_rmse",
                "best_epoch",
                "best_mse",
                "best_rmse",
                "common_epoch",
                "common_epoch_mse",
                "common_epoch_rmse",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item.job_id,
                    item.sparsity,
                    str(item.log_path),
                    item.last_epoch,
                    item.last_mse,
                    item.last_rmse,
                    item.best_epoch,
                    item.best_mse,
                    item.best_rmse,
                    item.common_epoch,
                    item.common_epoch_mse,
                    item.common_epoch_rmse,
                ]
            )


def create_plot(summaries: List[RunSummary], metric: str, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    x = []
    y = []
    labels = []

    for item in summaries:
        if metric == "last":
            value = item.last_mse
            epoch = item.last_epoch
        elif metric == "best":
            value = item.best_mse
            epoch = item.best_epoch
        elif metric == "common":
            value = item.common_epoch_mse
            epoch = item.common_epoch
        else:
            raise ValueError(f"Unknown metric mode: {metric}")

        if value is None:
            continue

        x.append(item.sparsity)
        y.append(value)
        labels.append(f"{item.job_id}\nepoch {epoch}")

    if not x:
        raise RuntimeError("No valid points found for plotting.")

    x_arr = list(x)
    y_arr = list(y)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        x_arr,
        y_arr,
        s=110,
        c=["#1f77b4" if s < 0.5 else "#d62728" for s in x_arr],
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )

    for xi, yi, label in zip(x_arr, y_arr, labels):
        ax.annotate(
            label,
            (xi, yi),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Validation MSE on corrected_t2m")
    title_map = {
        "last": "Last available checkpoint",
        "best": "Best validation checkpoint",
        "common": "Common epoch checkpoint",
    }
    ax.set_title(f"Sparsity vs performance ({title_map[metric]})")

    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    if y_arr:
        ymin = min(y_arr)
        ymax = max(y_arr)
        pad = 0.05 * (ymax - ymin if ymax > ymin else max(ymin, 1.0))
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_epoch_variance_plot(summaries: List[RunSummary], epoch_png: Path) -> None:
    """
    Creates epoch-wise plot:
    - x: epoch
    - y: corrected_t2m validation MSE
    - one line per sparsity (mean across runs at each epoch)
    - filled area = mean +/- std dev (where at least 2 runs are available)
    - thin transparent lines = individual runs
    """
    epoch_png.parent.mkdir(parents=True, exist_ok=True)

    # sparsity -> epoch -> list of losses from replicated runs
    grouped: Dict[float, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    # sparsity -> list of individual run curves for thin background lines
    run_curves: Dict[float, List[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for item in summaries:
        epoch_results = parse_output_log(item.log_path)
        # keep one value per epoch per run (latest if repeated)
        epoch_to_loss: Dict[int, float] = {}
        for er in epoch_results:
            if er.corrected_t2m_loss is not None:
                epoch_to_loss[er.epoch] = er.corrected_t2m_loss

        if not epoch_to_loss:
            continue

        epochs_sorted = np.array(sorted(epoch_to_loss.keys()), dtype=int)
        losses_sorted = np.array([epoch_to_loss[e] for e in epochs_sorted], dtype=float)
        run_curves[item.sparsity].append((epochs_sorted, losses_sorted))

        for e, v in epoch_to_loss.items():
            grouped[item.sparsity][e].append(v)

    if not grouped:
        raise RuntimeError("No epoch-wise data found to build filled variance plot.")

    fig, ax = plt.subplots(figsize=(11, 7))
    sparsity_levels = sorted(grouped.keys())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(sparsity_levels)))

    for color, sparsity in zip(cmap, sparsity_levels):
        epoch_map = grouped[sparsity]
        epochs = np.array(sorted(epoch_map.keys()), dtype=int)

        means = np.array([np.mean(epoch_map[e]) for e in epochs], dtype=float)
        stds = np.array([np.std(epoch_map[e]) for e in epochs], dtype=float)
        counts = np.array([len(epoch_map[e]) for e in epochs], dtype=int)

        # thin individual run traces
        for e_curve, v_curve in run_curves[sparsity]:
            ax.plot(e_curve, v_curve, color=color, alpha=0.18, linewidth=1.0, zorder=1)

        # main mean line
        ax.plot(
            epochs,
            means,
            color=color,
            linewidth=2.4,
            zorder=3,
            label=f"s={sparsity:.2f} (n={len(run_curves[sparsity])})",
        )

        # fill only where at least 2 values contribute (true variance band)
        lower = means - stds
        upper = means + stds
        valid_band = counts >= 2
        if np.any(valid_band):
            ax.fill_between(
                epochs[valid_band],
                lower[valid_band],
                upper[valid_band],
                color=color,
                alpha=0.22,
                linewidth=0.0,
                zorder=2,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE on Corrected t2m")
    ax.set_title("Validation Performance Over Epochs by Sparsity (mean ± 1σ)")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Sparsity groups", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(epoch_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
def create_common_epoch_area_plot(summaries: List[RunSummary], png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    by_sparsity = defaultdict(list)
    for s in summaries:
        if s.common_epoch_mse is not None:
            by_sparsity[float(s.sparsity)].append(float(s.common_epoch_mse))

    if not by_sparsity:
        raise RuntimeError("No common-epoch values found for area plot.")

    x = np.array(sorted(by_sparsity.keys()), dtype=float)
    y_mean = np.array([np.mean(by_sparsity[k]) for k in x], dtype=float)
    y_min = np.array([np.min(by_sparsity[k]) for k in x], dtype=float)
    y_max = np.array([np.max(by_sparsity[k]) for k in x], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Shaded band only between min and max runs for each sparsity
    ax.fill_between(
        x,
        y_min,
        y_max,
        alpha=0.25,
        color="#ff7f0e",
        label="Replicate range (min-max)",
    )

    # Mean line + points
    ax.plot(
        x,
        y_mean,
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Mean MSE",
    )
    ax.scatter(x, y_mean, color="#1f77b4", edgecolors="black", linewidths=0.7, zorder=3)

    for i, xi in enumerate(x):
        n = len(by_sparsity[float(xi)])
        ax.annotate(
            f"n={n}",
            (xi, y_mean[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Validation MSE on corrected_t2m (common epoch)")
    ax.set_title("Common-Epoch Sparsity vs Performance")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0.1)
    ax.invert_xaxis()
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG_PATH)
    parser.add_argument("--epoch-png", type=Path, default=DEFAULT_EPOCH_PNG_PATH)
    parser.add_argument(
        "--metric",
        choices=["last", "best", "common"],
        default="common",
        help="Which checkpoint summary to plot.",
    )
    parser.add_argument(
        "--common-epoch",
        type=int,
        default=None,
        help="Epoch to use for common-epoch comparison. Only used when --metric common.",
    )
    parser.add_argument(
        "--skip-epoch-plot",
        action="store_true",
        help="If set, skip the epoch-wise filled variance plot.",
    )

    parser.add_argument(
        "--area-png",
        type=Path,
        default=DEFAULT_COMMON_AREA_PNG_PATH,
        help="Output path for common-epoch area plot.",
    )
    args = parser.parse_args()

    summaries: List[RunSummary] = []
    for spec in RUN_SPECS:
        job_id = spec["job_id"]
        spec_sparsity = spec.get("sparsity")
        log_path = args.output_dir / f"output_{job_id}.txt"
        if not log_path.exists():
            print(f"Skipping {job_id}: missing log {log_path}")
            continue

        log_sparsity = extract_sparsity_from_log(log_path)
        if log_sparsity is None and spec_sparsity is None:
            print(
                f"Skipping {job_id}: could not find sparse_target_sparsity in log and no fallback in RUN_SPECS."
            )
            continue

        if log_sparsity is None:
            sparsity = float(spec_sparsity)
            print(f"Warning: {job_id} missing sparse_target_sparsity in log; using RUN_SPECS value {sparsity}.")
        else:
            sparsity = float(log_sparsity)
            if spec_sparsity is not None and abs(float(spec_sparsity) - sparsity) > 1e-12:
                print(
                    f"Warning: {job_id} RUN_SPECS sparsity={spec_sparsity} differs from log sparsity={sparsity}; using log value."
                )

        summaries.append(summarize_run(job_id, sparsity, log_path, args.common_epoch))

    if not summaries:
        raise RuntimeError("No logs found for the requested run IDs.")

    if args.common_epoch is None:
        common_candidates = [s.last_epoch for s in summaries if s.last_epoch is not None]
        if not common_candidates:
            raise RuntimeError("Could not infer a common epoch from run logs.")
        args.common_epoch = min(common_candidates)
        print(f"Auto-selected common epoch: {args.common_epoch}")
    else:
        print(f"Using user-provided common epoch: {args.common_epoch}")

    summaries = [summarize_run(s.job_id, s.sparsity, s.log_path, args.common_epoch) for s in summaries]

    write_csv(summaries, args.csv)
    create_plot(summaries, args.metric, args.png)

    if not args.skip_epoch_plot:
        create_epoch_variance_plot(summaries, args.epoch_png)
    
    create_common_epoch_area_plot(summaries, args.area_png)

    print(f"Wrote CSV summary to {args.csv}")
    print(f"Wrote scatter plot to {args.png}")
    if not args.skip_epoch_plot:
        print(f"Wrote epoch variance plot to {args.epoch_png}")
    print(f"Wrote common-epoch area plot to {args.area_png}")



    print("Summary:")
    for item in summaries:
        if args.metric == "best":
            value = item.best_mse
            epoch = item.best_epoch
        elif args.metric == "common":
            value = item.common_epoch_mse
            epoch = item.common_epoch
        else:
            value = item.last_mse
            epoch = item.last_epoch
        rmse = math.sqrt(value) if value is not None else None
        print(
            f"  job {item.job_id}: sparsity={item.sparsity:.2f}, "
            f"epoch={epoch}, MSE={value}, RMSE={rmse}"
        )

    if args.metric == "common" and any(item.common_epoch_mse is None for item in summaries):
        print(
            "Warning: not all runs reached the requested common epoch, so the common-epoch comparison is incomplete."
        )


if __name__ == "__main__":
    main()