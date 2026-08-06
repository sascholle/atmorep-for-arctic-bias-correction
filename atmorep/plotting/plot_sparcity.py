#!/usr/bin/env python3
"""
Plot sparsity vs performance from AtmoRep SLURM output logs.

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

import matplotlib
from matplotlib.colors import hex2color
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("/work/ab1412/atmorep/output")
DEFAULT_PLOT_DIR = Path("/work/ab1412/atmorep/plotting/sparsity_vs_performance/reruns_07_2026")
DEFAULT_CSV_PATH = DEFAULT_PLOT_DIR / "sparsity_vs_performance_summary.csv"
DEFAULT_PNG_PATH = DEFAULT_PLOT_DIR / "sparsity_vs_performance.png"
DEFAULT_EPOCH_PNG_PATH = DEFAULT_PLOT_DIR / "sparsity_over_epochs.png"
DEFAULT_COMMON_AREA_PNG_PATH = DEFAULT_PLOT_DIR / "sparsity_common_epoch_area.png"
DEFAULT_ROLLING_WINDOW = 3


# Map your run ids to sparsity values.
RUN_SPECS = [
    {"job_id": 23978827, "sparsity": 0.00},
    {"job_id": 24518395, "sparsity": 0.00}, #uff

    {"job_id": 24075112, "sparsity": 0.25},

    {"job_id": 23647012, "sparsity": 0.50},
    {"job_id": 24106052, "sparsity": 0.50},
    {"job_id": 24518397, "sparsity": 0.50},

    {"job_id": 23995579, "sparsity": 0.75},
    {"job_id": 24075176, "sparsity": 0.75},
    {"job_id": 24518398, "sparsity": 0.75},

    {"job_id": 24012088, "sparsity": 0.85},
    {"job_id": 24106063, "sparsity": 0.85},

    {"job_id": 24106671, "sparsity": 0.95},
    {"job_id": 24549676, "sparsity": 0.95},

    {"job_id": 24589450, "sparsity": 0.999, "description": "baseline"},
    {"job_id": 24589447, "sparsity": 0.999, "description": "baseline"},
    {"job_id": 24589754, "sparsity": 0.999, "description": "baseline"},


    {"job_id": 24358859, "sparsity": 0.00, "description": "attentionbaseline"}, #uff
    {"job_id": 24518419, "sparsity": 0.00, "description": "baseline"},


    {"job_id": 23994917, "sparsity": 0.25, "description": "combined"},
    {"job_id": 24025102, "sparsity": 0.95, "description": "combined"},
    {"job_id": 24168063, "sparsity": 0.00, "description": "combined"},
    {"job_id": 24106664, "sparsity": 0.25, "description": "combined"},
    {"job_id": 24169650, "sparsity": 0.75, "description": "combined"} #uff
]

DEFAULT_BASELINE_JOB_ID = 24518419
CLIMATOLOGY_2021_MSE_NORMALIZED = 1.0006451907686735
CLIMATOLOGY_2021_MSE_K2 = 24.034043796572636
CLIMATOLOGY_2021_RMSE_K = 4.902452834711685

# Conversion factor derived from your 2021 climatology stats
# normalized_mse * K2_PER_NORMALIZED = mse_K2
K2_PER_NORMALIZED = CLIMATOLOGY_2021_MSE_K2 / CLIMATOLOGY_2021_MSE_NORMALIZED

def norm_mse_to_rmse_k(mse_norm: float) -> float:
    arr = np.asarray(mse_norm, dtype=float)
    out = np.sqrt(np.maximum(arr, 0.0) * K2_PER_NORMALIZED)
    return float(out) if np.ndim(arr) == 0 else out

def rmse_k_to_norm_mse(rmse_k: float) -> float:
    arr = np.asarray(rmse_k, dtype=float)
    out = (arr ** 2) / K2_PER_NORMALIZED
    return float(out) if np.ndim(arr) == 0 else out

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
    run_group: str
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
    rolling_epoch: Optional[int] = None
    rolling_mse: Optional[float] = None
    rolling_rmse: Optional[float] = None


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

# add near other small helpers
def add_metric_to_path(path: Path, metric: str) -> Path:
    return path.with_name(f"{path.stem}_{metric}{path.suffix}")

def rolling_mean(values: List[float], window_size: int) -> Optional[float]:
    """Compute mean of last N values."""
    if not values:
        return None
    window_size = max(1, min(window_size, len(values)))
    return float(np.mean(values[-window_size:]))

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

def summarize_run(job_id: int, sparsity: float, run_group: str, log_path: Path, common_epoch: Optional[int]) -> RunSummary:
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
        run_group=run_group,
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

def summarize_run_rolling(
    job_id: int,
    sparsity: float,
    run_group: str,
    log_path: Path,
    common_epoch: Optional[int],
    window_size: int,
) -> RunSummary:
    """Summarize run using a rolling average ending at the common epoch."""
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

    rolling_epoch = common_epoch
    rolling_mse = None
    if common_epoch is not None:
        window_start = common_epoch - window_size + 1
        window_losses = [
            float(item.corrected_t2m_loss)
            for item in epoch_results
            if item.corrected_t2m_loss is not None
            and window_start <= item.epoch <= common_epoch
        ]
        if window_losses:
            rolling_mse = float(np.mean(window_losses))

    def to_rmse(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return math.sqrt(value)

    return RunSummary(
        job_id=job_id,
        sparsity=sparsity,
        run_group=run_group,
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
        rolling_epoch=rolling_epoch,
        rolling_mse=rolling_mse,
        rolling_rmse=to_rmse(rolling_mse) if rolling_mse is not None else None,
    )

def write_csv(summaries: List[RunSummary], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "job_id",
                "sparsity",
                "run_group",
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
                "rolling_epoch",
                "rolling_mse",
                "rolling_rmse",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item.job_id,
                    item.sparsity,
                    item.run_group,
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
                    item.rolling_epoch,
                    item.rolling_mse,
                    item.rolling_rmse,
                ]
            )

# def create_plot(summaries: List[RunSummary], metric: str, png_path: Path, baseline_job_id: int = DEFAULT_BASELINE_JOB_ID,) -> None:
#     png_path.parent.mkdir(parents=True, exist_ok=True)

#     group_colors = {
#         "regular": "#d62728",   # red
#         "combined": "#ff7f0e",  # orange
#         "baseline": "#b41fb2",  
#     }

#     baseline_value = None
#     for b in summaries:
#         if b.job_id != baseline_job_id:
#             continue
#         if metric == "last":
#             v = b.last_mse
#         elif metric == "best":
#             v = b.best_mse
#         elif metric == "rolling":
#             v = b.rolling_mse
#         else:  # common
#             v = b.common_epoch_mse
#         if v is not None:
#             baseline_value = float(v)
#             break

#     points = []
#     for item in summaries:
#         if item.job_id == baseline_job_id:
#             continue    
#         if metric == "last":
#             value = item.last_mse
#             epoch = item.last_epoch
#         elif metric == "best":
#             value = item.best_mse
#             epoch = item.best_epoch
#         elif metric == "common":
#             value = item.common_epoch_mse
#             epoch = item.common_epoch
#         elif metric == "rolling":
#             value = item.rolling_mse
#             epoch = item.rolling_epoch
#         else:
#             raise ValueError(f"Unknown metric mode: {metric}")

#         if value is None:
#             continue

#         group = item.run_group if item.run_group in group_colors else "regular"
#         points.append(
#             {
#                 "x": float(item.sparsity),
#                 "y": float(value),
#                 "label": f"{item.job_id}",
#                 "group": group,
#             }
#         )

#     if not points:
#         raise RuntimeError("No valid points found for plotting.")

#     fig, ax = plt.subplots(figsize=(9, 6))
#     fig.set_size_inches(11.69, 8.27)

#     # # Scatter by run group
#     # for group in ["regular", "combined", "baseline"]:
#     #     pts = [p for p in points if p["group"] == group]
#     #     if not pts:
#     #         continue
#     ax.scatter(
#         [p["x"] for p in points],
#         [p["y"] for p in points],
#         s=110,
#         edgecolors="black",
#         linewidths=0.8,
#         alpha=0.9,
#         zorder=3,
#         color="#ff7f0e",
#         #c=group_colors[group],
#         #label=f"{group} runs",
#     )

#     x_arr = [p["x"] for p in points]
#     y_arr = [p["y"] for p in points]

#     # Mean trend line across sparsity values (all runs)
#     grouped = defaultdict(list)
#     for p in points:
#         grouped[p["x"]].append(p["y"])

#     x_mean = np.array(sorted(grouped.keys()), dtype=float)
#     y_mean = np.array([np.mean(grouped[xi]) for xi in x_mean], dtype=float)

#     ax.plot(
#         x_mean,
#         y_mean,
#         color="#2ca02c",
#         linewidth=2.4,
#         marker="o",
#         markersize=5,
#         alpha=0.95,
#         zorder=4,
#         label="Mean across runs",
#     )

#     if baseline_value is not None:
#         ax.axhline(
#             y=baseline_value,
#             color="#b41fb2",
#             linestyle="-.",
#             linewidth=2.2,
#             alpha=0.95,
#             zorder=2,
#             label=f"Baseline MSE with no cross-attention to other fields",
#     )
#         ax.text(
#             0.02,
#             baseline_value + 0.002,
#             "no cross-attention baseline",
#             color="#b41fb2",
#             fontsize=12,
#             va="bottom",
#             ha="left",
#         )

#     # Observed value range band
#     y_min_obs = float(np.min(y_arr))
#     y_max_obs = float(np.max(y_arr))
#     rmse_k_min = norm_mse_to_rmse_k(y_min_obs)
#     rmse_k_max = norm_mse_to_rmse_k(y_max_obs)

#     ax.axhspan(
#         y_min_obs,
#         y_max_obs,
#         color="#8ecae6",
#         alpha=0.18,
#         zorder=1,
#         label=f"Observed range: {rmse_k_min:.2f}-{rmse_k_max:.2f} K RMSE",
#     )

#     # Right axis in Kelvin RMSE
#     secax = ax.secondary_yaxis(
#         "right",
#         functions=(norm_mse_to_rmse_k, rmse_k_to_norm_mse),
#     )
#     secax.set_ylabel("Validation RMSE on Corrected T2M [K]", fontsize=20)

#     # Keep focus on sparsity points
#     pad = 0.35 * (y_max_obs - y_min_obs if y_max_obs > y_min_obs else max(y_min_obs, 0.2))
#     y_low = max(0.0, y_min_obs - pad)
#     y_high = y_max_obs + pad
#     ax.set_ylim(y_low, y_high)

#     # Climatology baseline marker
#     clim = CLIMATOLOGY_2021_MSE_NORMALIZED
#     if clim <= y_high:
#         ax.axhline(
#             y=clim,
#             color="#9467bd",
#             linestyle="--",
#             linewidth=2.0,
#             alpha=0.95,
#             zorder=2,
#             label=(
#                 f"2021 climatology "
#                 f"(MSE={clim:.3f}, RMSE={CLIMATOLOGY_2021_RMSE_K:.2f} K)"
#             ),
#         )
#     # else:
#     #     ax.annotate(
#     #         (
#     #             f"2021 climatology off-scale:\n"
#     #             f"MSE={clim:.3f}, RMSE={CLIMATOLOGY_2021_RMSE_K:.2f} K"
#     #         ),
#     #         xy=(0.02, 0.98),
#     #         xycoords="axes fraction",
#     #         ha="left",
#     #         va="top",
#     #         fontsize=14,
#     #         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9467bd", alpha=0.9),
#     #     )

#     # for p in points:
#     #     ax.annotate(
#     #         p["label"],
#     #         (p["x"], p["y"]),
#     #         textcoords="offset points",
#     #         xytext=(8, 8),
#     #         fontsize=14,
#     #         alpha=0.9,
#     #     )

#     ax.set_xlabel("Target sparsity", fontsize=20)
#     ax.set_ylabel("Validation MSE on Corrected T2M", fontsize=20)
#     title_map = {
#         "last": "Last available checkpoint",
#         "rolling": "Rolling average checkpoint",
#         "best": "Best validation checkpoint",
#         "common": "Common epoch checkpoint",
#     }
#     ax.set_title(f"Sparsity vs Performance\n({title_map[metric]})", fontsize=25, pad=15)
#     ax.grid(True, alpha=0.25)
#     ax.set_axisbelow(True)
#     ax.legend(frameon=True, fontsize=13)

#     if y_arr:
#         ymin = min(y_arr)
#         ymax = max(y_arr)
#         pad = 0.5 * (ymax - ymin if ymax > ymin else max(ymin, 1.0))
#         ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

#     ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=1)

#     plt.tight_layout()
#     plt.savefig(png_path, dpi=600, bbox_inches="tight")
#     plt.close(fig)

def create_plot(summaries: List[RunSummary], metric: str, png_path: Path, baseline_job_id: int = DEFAULT_BASELINE_JOB_ID,) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    group_colors = {
        "regular": "#d62728",   # red
        "combined": "#ff7f0e",  # orange
        "baseline": "#b41fb2",  
    }

    baseline_value = None
    for b in summaries:
        if b.job_id != baseline_job_id:
            continue
        if metric == "last":
            v = b.last_mse
        elif metric == "best":
            v = b.best_mse
        elif metric == "rolling":
            v = b.rolling_mse
        else:  # common
            v = b.common_epoch_mse
        if v is not None:
            baseline_value = float(v)
            break

    points = []
    for item in summaries:
        if item.job_id == baseline_job_id:
            continue    
        if metric == "last":
            value = item.last_mse
            epoch = item.last_epoch
        elif metric == "best":
            value = item.best_mse
            epoch = item.best_epoch
        elif metric == "common":
            value = item.common_epoch_mse
            epoch = item.common_epoch
        elif metric == "rolling":
            value = item.rolling_mse
            epoch = item.rolling_epoch
        else:
            raise ValueError(f"Unknown metric mode: {metric}")

        if value is None:
            continue

        group = item.run_group if item.run_group in group_colors else "regular"
        points.append(
            {
                "x": float(item.sparsity),
                "y": float(value),
                "label": f"{item.job_id}",
                "group": group,
            }
        )

    if not points:
        raise RuntimeError("No valid points found for plotting.")

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.set_size_inches(12.69, 8.27)

    normal_points = [p for p in points if p["x"] < 0.999 - 0.01]
    baseline_999_points = [p for p in points if abs(p["x"] - 0.999) < 0.01]

    if normal_points:
        ax.scatter(
            [p["x"] for p in normal_points],
            [p["y"] for p in normal_points],
            s=110,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
            color="#ff7f0e",
            label="AtmoRep runs",
        )

    if baseline_999_points:
        ax.scatter(
            [p["x"] for p in baseline_999_points],
            [p["y"] for p in baseline_999_points],
            s=110,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
            color="#7f7f7f",
        )

    x_arr = [p["x"] for p in points]
    y_arr = [p["y"] for p in points]

    # Mean trend line across sparsity values (all runs)
    grouped = defaultdict(list)
    for p in points:
        grouped[p["x"]].append(p["y"])

    x_mean = np.array(sorted(grouped.keys()), dtype=float)
    y_mean = np.array([np.mean(grouped[xi]) for xi in x_mean], dtype=float)

    ax.plot(
        x_mean,
        y_mean,
        color="#2ca02c",
        linewidth=2.4,
        marker="o",
        markersize=5,
        alpha=0.95,
        zorder=4,
        label="Mean across runs",
    )

    # Plot no cross-attention baseline as purple cross at x=0
    if baseline_value is not None:
        ax.scatter(
            [0.0],
            [baseline_value],
            s=150,
            marker="x",
            color="#b41fb2",
            linewidths=2,
            alpha=0.95,
            zorder=5,
            label=f"Baseline: No T2M cross-attention to other fields,\nfull data coverage",
        )
        # ax.text(
        #     -0.03,
        #     baseline_value + 0.01,
        #     "no cross-attention\nbaseline",
        #     color="#b41fb2",
        #     fontsize=16,
        #     va="bottom",
        #     ha="left",
        # )

    # Observed value range band - only for sparsity 0 to 0.95 (exclude 0.999)
    points_in_range = [p for p in points if 0.0 <= p["x"] <= 0.95]
    if points_in_range:
        y_range = [p["y"] for p in points_in_range]
        y_min_obs = float(np.min(y_range))
        y_max_obs = float(np.max(y_range))
        rmse_k_min = norm_mse_to_rmse_k(y_min_obs)
        rmse_k_max = norm_mse_to_rmse_k(y_max_obs)

        ax.axhspan(
            y_min_obs,
            y_max_obs,
            color="#8ecae6",
            alpha=0.18,
            zorder=1,
            label=f"Observed range (s=0–0.95): {rmse_k_min:.2f}-{rmse_k_max:.2f} K RMSE",
        )

            # Calculate axis limits using ALL points (including 0.999)
            
    if y_arr:
        y_min_all = float(np.min(y_arr))
        y_max_all = float(np.max(y_arr))
        pad = 0.3 * (y_max_all - y_min_all if y_max_all > y_min_all else max(y_min_all, 0.2))
        y_low = max(0.0, y_min_all - pad)
        y_high = y_max_all + pad*1.5
    else:
        y_low = 0.0
        y_high = 0.1

    ax.set_ylim(y_low, y_high)

    # Right axis in Kelvin RMSE — create after setting y-limits so ticks map correctly
    secax = ax.secondary_yaxis(
        "right",
        functions=(norm_mse_to_rmse_k, rmse_k_to_norm_mse),
    )
    secax.set_ylabel("Validation RMSE on Corrected T2M [K]", fontsize=25, labelpad=15)
    secax.tick_params(axis="y", which="major", right=True, labelright=True, length=12, width=1.8, labelsize=18)
    secax.spines["right"].set_visible(True)

    # Make left ticks match the right-side styling
    ax.tick_params(axis="both", which="major", labelsize=18, length=12, width=1.8)

    # Climatology baseline marker
    clim = CLIMATOLOGY_2021_MSE_NORMALIZED
    if clim <= y_high:
        ax.axhline(
            y=clim,
            color="#9467bd",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
            label=(
                f"2021 climatology "
                f"(MSE={clim:.3f}, RMSE={CLIMATOLOGY_2021_RMSE_K:.2f} K)"
            ),
        )

    ax.set_xlabel("Target sparsity", fontsize=25)
    ax.set_ylabel("Validation MSE on Corrected T2M", fontsize=25, labelpad=15)
    title_map = {
        "last": "Last available checkpoint",
        "rolling": "Rolling average checkpoint",
        "best": "Best validation checkpoint",
        "common": "Common epoch checkpoint",
    }
    ax.set_title(f"Sparsity vs Performance\n({title_map[metric]})", fontsize=30, pad=15)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    xticks = np.arange(0, 1.01, 0.25)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}".rstrip("0").rstrip(".") for x in xticks])
    ax.set_xlim(-0.05, 1.05)
    ax.legend(frameon=True, fontsize=17)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

def create_sparsity_vs_performance_at_epoch(
    summaries: List[RunSummary], target_epoch: int, png_path: Path, 
    baseline_job_id: int = DEFAULT_BASELINE_JOB_ID,
) -> None:
    """
    Build a sparsity vs performance scatter plot at a fixed epoch.
    Mirrors create_plot() but evaluates performance at target_epoch instead of using common/best/last metrics.
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect points: separate baselines from regular/combined
    points = []
    baseline_0_points = []
    baseline_999_points = []
    by_sparsity: Dict[float, List[float]] = defaultdict(list)
    
    for item in summaries:
        epoch_results = parse_output_log(item.log_path)
        val = next((er.corrected_t2m_loss for er in epoch_results 
                   if er.epoch == target_epoch and er.corrected_t2m_loss is not None), None)
        if val is not None:
            x = float(item.sparsity)
            y = float(val)
            
            if item.run_group == "baseline":
                # Separate baselines by sparsity
                if abs(x - 0.00) < 0.01:
                    baseline_0_points.append({"x": x, "y": y, "job_id": item.job_id})
                elif abs(x - 0.999) < 0.01:
                    baseline_999_points.append({"x": x, "y": y, "job_id": item.job_id})
            else:
                # Regular/combined runs
                by_sparsity[x].append(y)
                points.append({
                    "x": x,
                    "y": y,
                    "label": f"{item.job_id}",
                })

    if not points and not baseline_0_points and not baseline_999_points:
        raise RuntimeError(f"No runs reached epoch {target_epoch}.")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.set_size_inches(11.69, 8.27)

    # Scatter regular/combined run points (orange)
    if points:
        ax.scatter(
            [p["x"] for p in points],
            [p["y"] for p in points],
            s=110,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
            color="#ff7f0e",
        )

    x_arr = [p["x"] for p in points]
    y_arr = [p["y"] for p in points]

    # Mean trend line across sparsity values (green) - only for non-baselines
    if by_sparsity:
        x_mean = np.array(sorted(by_sparsity.keys()), dtype=float)
        y_mean = np.array([np.mean(by_sparsity[xi]) for xi in x_mean], dtype=float)

        ax.plot(
            x_mean,
            y_mean,
            color="#2ca02c",
            linewidth=2.4,
            marker="o",
            markersize=5,
            alpha=0.95,
            zorder=4,
            label="Mean across runs",
        )

    # Plot 0.00 baseline scatter (purple)
    if baseline_0_points:
        ax.scatter(
            [p["x"] for p in baseline_0_points],
            [p["y"] for p in baseline_0_points],
            s=110,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
            color="#b41fb2",
            marker="s",
            label=f"no cross attention baseline (n={len(baseline_0_points)})",
        )

    # Plot 0.999 baseline scatter (dark gray)
    if baseline_999_points:
        ax.scatter(
            [p["x"] for p in baseline_999_points],
            [p["y"] for p in baseline_999_points],
            s=110,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
            color="#1a1a1a",
            marker="^",
            label=f"no target data baseline (n={len(baseline_999_points)})",
        )

    # Observed value range band
    if y_arr:
        y_min_obs = float(np.min(y_arr))
        y_max_obs = float(np.max(y_arr))
        rmse_k_min = norm_mse_to_rmse_k(y_min_obs)
        rmse_k_max = norm_mse_to_rmse_k(y_max_obs)

        ax.axhspan(
            y_min_obs,
            y_max_obs,
            color="#8ecae6",
            alpha=0.18,
            zorder=1,
            label=f"Observed range: {rmse_k_min:.2f}-{rmse_k_max:.2f} K RMSE",
        )

        pad = 0.35 * (y_max_obs - y_min_obs if y_max_obs > y_min_obs else max(y_min_obs, 0.2))
        y_low = max(0.0, y_min_obs - pad)
        y_high = y_max_obs + pad
    else:
        # No regular runs, use baseline range
        all_baseline_y = [p["y"] for p in baseline_0_points + baseline_999_points]
        if all_baseline_y:
            y_min = min(all_baseline_y)
            y_max = max(all_baseline_y)
            pad = 0.35 * (y_max - y_min if y_max > y_min else max(y_min, 0.2))
            y_low = max(0.0, y_min - pad)
            y_high = y_max + pad
        else:
            y_low = 0.0
            y_high = 0.1

    # Right axis in Kelvin RMSE
    secax = ax.secondary_yaxis(
        "right",
        functions=(norm_mse_to_rmse_k, rmse_k_to_norm_mse),
    )
    secax.set_ylabel("Validation RMSE on Corrected T2M [K]", fontsize=20)

    # Climatology baseline marker
    clim = CLIMATOLOGY_2021_MSE_NORMALIZED
    if clim <= y_high:
        ax.axhline(
            y=clim,
            color="#9467bd",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
            label=(
                f"2021 climatology "
                f"(MSE={clim:.3f}, RMSE={CLIMATOLOGY_2021_RMSE_K:.2f} K)"
            ),
        )

    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("Target sparsity", fontsize=20)
    ax.set_ylabel("Validation MSE on Corrected T2M", fontsize=20)
    ax.set_title(f"Sparsity vs Performance at Epoch {target_epoch}", fontsize=25, pad=15)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=1)

    plt.tight_layout()
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
def create_rolling_average_epoch_plot(
    summaries: List[RunSummary], num_epochs: int, window_size: int, png_path: Path
) -> None:
    """
    Scatter plot of MSE at each epoch for first N epochs, colored by sparsity.
    Shows rolling average overlay.
    
    Args:
        summaries: List of run summaries
        num_epochs: Number of epochs to plot (e.g., 20)
        window_size: Rolling window size for averaging (e.g., 3)
        png_path: Output PNG path
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)

    all_epochs_to_losses: Dict[int, List[float]] = defaultdict(list)
    all_x = []
    all_y = []
    all_sparsity = []
    
    for item in summaries:
        epoch_results = parse_output_log(item.log_path)
        for er in epoch_results:
            if er.corrected_t2m_loss is not None and er.epoch < num_epochs:
                all_epochs_to_losses[er.epoch].append(er.corrected_t2m_loss)
                all_x.append(er.epoch)
                all_y.append(er.corrected_t2m_loss)
                all_sparsity.append(item.sparsity)

    if not all_x:
        raise RuntimeError(f"No epoch-wise data found for first {num_epochs} epochs.")

    epochs = np.array(sorted(all_epochs_to_losses.keys()), dtype=int)
    means = np.array([np.mean(all_epochs_to_losses[e]) for e in epochs], dtype=float)
    
    # Apply rolling average
    rolling_means = np.convolve(means, np.ones(window_size) / window_size, mode='valid')
    rolling_epochs = epochs[:(len(rolling_means))]

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color map: red (high sparsity/bad) to blue (low sparsity/good)
    sparsity_values = np.array(all_sparsity)
    cmap = plt.cm.RdYlGn_r  # Red=high sparsity (bad), Green=low sparsity (good)
    norm = plt.Normalize(vmin=0, vmax=1.0)
    colors = cmap(norm(sparsity_values))
    
    # Scatter all individual points with jitter, colored by sparsity
    np.random.seed(42)
    x_jitter = np.array(all_x) + np.random.normal(0, 0.05, len(all_x))
    scatter = ax.scatter(x_jitter, all_y, c=sparsity_values, cmap=cmap, norm=norm, 
                         alpha=0.5, s=50, edgecolors="black", linewidths=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Target sparsity", fontsize=10)
    
    # # Rolling average line (black overlay)
    # ax.plot(rolling_epochs, rolling_means, color="black", linewidth=3.0, marker="D", 
    #         markersize=7, label=f"Rolling avg (window={window_size})", zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE on corrected_t2m")
    ax.set_title(f"Rolling Average Training Trajectory by Sparsity Level (First {num_epochs} Epochs)")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.set_xlim(left=-0.5, right=num_epochs-0.5)

    plt.tight_layout()
    plt.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
def create_epoch_variance_plot(
    summaries: List[RunSummary],
    epoch_png: Path,
    output_dir: Path,
    baseline_job_id: int = DEFAULT_BASELINE_JOB_ID,) -> None:
    """
    Epoch-wise plot grouped by sparsity (combining regular and combined runs):
    - non-baseline runs: solid line + fill (mean ± std), colored by sparsity
    - s=0.00 baseline: purple dotted line, "no cross attention baseline"
    - s=0.999 baseline: dark gray dashed line, "no target data baseline"
    
    Plots only epochs up to 150.
    """
    _ = output_dir, baseline_job_id  # kept for interface compatibility
    epoch_png.parent.mkdir(parents=True, exist_ok=True)

    # Separate baselines from regular/combined
    baseline_runs = [item for item in summaries if item.run_group == "baseline"]
    non_baseline_runs = [item for item in summaries if item.run_group != "baseline"]

    # Group non-baselines by sparsity (combining regular+combined)
    grouped: Dict[str | float, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    run_curves: Dict[str | float, List[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for item in non_baseline_runs:
        key = float(item.sparsity)
        epoch_results = parse_output_log(item.log_path)
        epoch_to_loss: Dict[int, float] = {}
        for er in epoch_results:
            if er.corrected_t2m_loss is not None:
                epoch_to_loss[er.epoch] = er.corrected_t2m_loss
        if not epoch_to_loss:
            continue
        epochs_sorted = np.array(sorted(epoch_to_loss.keys()), dtype=int)
        losses_sorted = np.array([epoch_to_loss[e] for e in epochs_sorted], dtype=float)
        run_curves[key].append((epochs_sorted, losses_sorted))
        for e, v in epoch_to_loss.items():
            grouped[key][e].append(v)

    # Add baselines separately with string keys
    for item in baseline_runs:
        key = f"baseline_{item.sparsity:.2f}"
        epoch_results = parse_output_log(item.log_path)
        epoch_to_loss: Dict[int, float] = {}
        for er in epoch_results:
            if er.corrected_t2m_loss is not None:
                epoch_to_loss[er.epoch] = er.corrected_t2m_loss
        if not epoch_to_loss:
            continue
        epochs_sorted = np.array(sorted(epoch_to_loss.keys()), dtype=int)
        losses_sorted = np.array([epoch_to_loss[e] for e in epochs_sorted], dtype=float)
        run_curves[key].append((epochs_sorted, losses_sorted))
        for e, v in epoch_to_loss.items():
            grouped[key][e].append(v)

    if not grouped:
        raise RuntimeError("No epoch-wise data found to build filled variance plot.")

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.set_size_inches(11.69, 8.27)

    # Separate numeric sparsity keys from baseline string keys
    numeric_keys = sorted([k for k in grouped.keys() if isinstance(k, float)], key=lambda x: float(x))
    baseline_keys = sorted([k for k in grouped.keys() if isinstance(k, str)], key=lambda x: x)

    # Color map for sparsity values (exclude baselines)
    cmap_sparsity = plt.cm.viridis(np.linspace(0.10, 0.95, max(1, len(numeric_keys))))

    color_map: Dict[str | float, tuple] = {}
    for color, key in zip(cmap_sparsity, numeric_keys):
        color_map[key] = color

    # Purple for baseline_0.00, dark gray for baseline_0.999
    purple_rgb = hex2color("#b41fb2")
    dark_gray = np.array([0.07, 0.07, 0.07, 1.0])

    for key in baseline_keys:
        if "baseline_0.00" in key:
            color_map[key] = np.array([*purple_rgb, 1.0])
        else:
            color_map[key] = dark_gray

    ordered_keys = numeric_keys + baseline_keys

    for key in ordered_keys:
        epoch_map = grouped[key]
        epochs = np.array(sorted(epoch_map.keys()), dtype=int)
        
        # Filter to epochs <= 150
        mask = epochs <= 150
        epochs = epochs[mask]
        
        if len(epochs) == 0:
            continue

        means = np.array([np.mean(epoch_map[e]) for e in epochs], dtype=float)
        stds = np.array([np.std(epoch_map[e]) for e in epochs], dtype=float)
        counts = np.array([len(epoch_map[e]) for e in epochs], dtype=int)
        color = color_map[key]

        # thin individual run traces
        for e_curve, v_curve in run_curves[key]:
            # Filter individual curves to <= 150 epochs
            mask_curve = e_curve <= 150
            e_curve_filtered = e_curve[mask_curve]
            v_curve_filtered = v_curve[mask_curve]
            if len(e_curve_filtered) > 0:
                ax.plot(e_curve_filtered, v_curve_filtered, color=color, alpha=0.16, linewidth=1.0, zorder=1)

        # Set label and style based on key type
        if isinstance(key, str):
            # It's a baseline
            if "baseline_0.00" in key:
                line_style = ":"
                line_width = 2.6
                label = f"no cross attention baseline (n={len(run_curves[key])})"
            else:  # baseline_0.999
                line_style = "--"
                line_width = 2.6
                label = f"no target data baseline (n={len(run_curves[key])})"
        else:
            # Regular/combined runs: show sparsity
            line_style = "-"
            line_width = 2.4
            label = f"s={key:.2f} (n={len(run_curves[key])})"

        ax.plot(
            epochs,
            means,
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            zorder=3,
            label=label,
        )

        # fill only for non-baseline groups, and only where true variance exists
        if isinstance(key, float):
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

    ax.set_xlabel("Epoch", fontsize=25)
    ax.set_ylabel("Validation MSE on Corrected T2M", fontsize=25)
    ax.set_title("Validation Performance Over Epochs\nby Sparsity Level", fontsize=33, pad=15)
    ax.set_xlim(left=0, right=110)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    
    ax.legend(title="Sparsity vs Baselines", title_fontsize=18, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)

    plt.tight_layout()
    plt.savefig(epoch_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
def create_averaged_epoch_plot(summaries: List[RunSummary], png_path: Path) -> None:
    """
    Plot mean epoch performance with std fill across ALL runs (ignoring sparsity).
    Single averaged curve showing overall training trajectory.
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all epoch results from all runs
    all_epochs_to_losses: Dict[int, List[float]] = defaultdict(list)

    for item in summaries:
        epoch_results = parse_output_log(item.log_path)
        for er in epoch_results:
            if er.corrected_t2m_loss is not None:
                all_epochs_to_losses[er.epoch].append(er.corrected_t2m_loss)

    if not all_epochs_to_losses:
        raise RuntimeError("No epoch-wise data found for averaged area plot.")

    epochs = np.array(sorted(all_epochs_to_losses.keys()), dtype=int)
    means = np.array([np.mean(all_epochs_to_losses[e]) for e in epochs], dtype=float)
    stds = np.array([np.std(all_epochs_to_losses[e]) for e in epochs], dtype=float)
    counts = np.array([len(all_epochs_to_losses[e]) for e in epochs], dtype=int)

    lower = means - stds
    upper = means + stds

    fig, ax = plt.subplots(figsize=(10, 6))

    # Fill between band (mean ± std)
    ax.fill_between(
        epochs,
        lower,
        upper,
        alpha=0.30,
        color="#1f77b4",
        label="Mean ± 1σ",
    )

    # Mean line
    ax.plot(
        epochs,
        means,
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Mean MSE (all runs)",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE on corrected_t2m")
    ax.set_title("Averaged Training Trajectory Across All Runs")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=10)

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
        choices=["last", "rolling", "best", "common"],
        default="common",
        help="Which checkpoint summary to plot.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help="Window size for the rolling-mean checkpoint metric.",
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
    "--baseline-job-id",
    type=int,
    default=DEFAULT_BASELINE_JOB_ID,
    help="Baseline run id to overlay on epoch plot.",
    )

    parser.add_argument(
        "--averaged-epoch-png",
        type=Path,
        default=DEFAULT_PLOT_DIR / "sparsity_over_epochs_averaged.png",
        help="Output path for averaged-epoch plots (all runs averaged).",
    )

    parser.add_argument(
        "--fixed-epoch",
        type=int,
        default=80,
        help="If set, produce a sparsity vs performance plot at this fixed epoch (e.g., 20).",
    )
    parser.add_argument(
        "--fixed-epoch-png",
        type=Path,
        default=DEFAULT_PLOT_DIR / "sparsity_vs_performance_fixed_epoch.png",
        help="Output path for fixed-epoch sparsity vs performance plot.",
    )

    parser.add_argument(
        "--rolling-avg-png",
        type=Path,
        default=DEFAULT_PLOT_DIR / "sparsity_rolling_avg_epoch.png",
        help="Output path for rolling average epoch plot.",
    )
    
    parser.add_argument(
        "--rolling-epochs",
        type=int,
        default=5,
        help="Number of epochs for rolling average plot.",
    )
    
    args = parser.parse_args()
    metric_png = add_metric_to_path(args.png, args.metric)

    summaries: List[RunSummary] = []
    for spec in RUN_SPECS:
        job_id = spec["job_id"]
        spec_sparsity = spec.get("sparsity")
        run_group = spec.get("description", "regular").strip().lower()
        if "baseline" in run_group:
            run_group = "baseline"
        elif run_group not in ("regular", "combined"):
            run_group = "regular"

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

        summaries.append(summarize_run(job_id, sparsity, run_group, log_path, None))

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

    if args.metric == "rolling":
        summaries = [
            summarize_run_rolling(
                s.job_id,
                s.sparsity,
                s.run_group,
                s.log_path,
                args.common_epoch,
                args.rolling_window,
            )
            for s in summaries
        ]
    else:
        summaries = [
            summarize_run(s.job_id, s.sparsity, s.run_group, s.log_path, args.common_epoch)
            for s in summaries
        ]

    write_csv(summaries, args.csv)
    create_plot(summaries, args.metric, metric_png, args.baseline_job_id)
    #create_epoch_variance_plot(summaries, args.epoch_png, args.output_dir, args.baseline_job_id)
    #create_common_epoch_area_plot(summaries, args.area_png)
    #create_averaged_epoch_plot(summaries, args.averaged_epoch_png)
    #create_sparsity_vs_performance_at_epoch(summaries, args.fixed_epoch, args.fixed_epoch_png)
    #create_rolling_average_epoch_plot(summaries, args.rolling_epochs, args.rolling_window, args.rolling_avg_png)

    print(f"Wrote CSV summary to {args.csv}")
    #print(f"Wrote scatter plot to {metric_png}")

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


