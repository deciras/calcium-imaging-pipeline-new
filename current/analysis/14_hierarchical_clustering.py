#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run hierarchical clustering on ROI response-pattern matrices."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trial_context_utils import load_excluded_trial_ids


LOGGER = logging.getLogger("hierarchical_clustering")
STEP_NAME = "14_hierarchical_clustering"
STEP_OUTPUT_PATTERNS = (
    "*_hierarchical_cluster_labels.csv",
    "*_hierarchical_cluster_summary.csv",
    "*_dendrogram.png",
    "*_dendrogram.pdf",
    "*_clustered_heatmap.png",
    "*_clustered_heatmap.pdf",
    "*_clustered_trace_heatmap.png",
    "*_clustered_trace_heatmap.pdf",
    "*_clustered_trace_heatmap_raw_dff.png",
    "*_clustered_trace_heatmap_raw_dff.pdf",
    "*_clustered_trace_heatmap_normalized_dff.png",
    "*_clustered_trace_heatmap_normalized_dff.pdf",
    "*_raw_cluster_hierarchical_cluster_labels.csv",
    "*_raw_cluster_hierarchical_cluster_summary.csv",
    "*_raw_cluster_dendrogram.png",
    "*_raw_cluster_dendrogram.pdf",
    "*_raw_cluster_clustered_heatmap.png",
    "*_raw_cluster_clustered_heatmap.pdf",
    "*_raw_cluster_clustered_trace_heatmap_raw_dff.png",
    "*_raw_cluster_clustered_trace_heatmap_raw_dff.pdf",
    "*_raw_cluster_clustered_trace_heatmap_normalized_dff.png",
    "*_raw_cluster_clustered_trace_heatmap_normalized_dff.pdf",
    "*_raw_cluster_cluster_mean_traces.png",
    "*_raw_cluster_cluster_mean_traces.pdf",
    "*_raw_cluster_cluster_mean_angle_tuning.png",
    "*_raw_cluster_cluster_mean_angle_tuning.pdf",
    "*_raw_cluster_hierarchical_clustering_summary.json",
    "*_cluster_mean_traces.png",
    "*_cluster_mean_traces.pdf",
    "*_cluster_mean_angle_tuning.png",
    "*_cluster_mean_angle_tuning.pdf",
    "*_hierarchical_clustering_summary.json",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    feature_dir: Path
    feature_matrix_path: Path
    raw_feature_matrix_path: Path | None
    trace_matrix_path: Path | None
    response_matrix_path: Path | None
    angle_response_matrix_path: Path | None
    slice_matrix_path: Path | None
    slice_matrix_raw_path: Path | None
    slice_columns_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


class TrialSkipError(RuntimeError):
    """Raised when a trial has no valid matrix to cluster."""


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_input_root(output_root: Path) -> Path:
    return output_root / "11_population_features"


def default_slice_root(output_root: Path) -> Path:
    return output_root / "12_stimulus_slice_features"


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def trial_output_dir(step_root: Path, trial: TrialInput) -> Path:
    return step_root / trial.rel_parent / trial.trial_id if str(trial.rel_parent) not in ("", ".") else step_root / trial.trial_id


def find_first_existing(folder: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_trials(input_root: Path, slice_root: Path) -> list[TrialInput]:
    trials = []
    for feature_path in sorted(input_root.rglob("*_roi_feature_matrix_zscored.csv")):
        feature_dir = feature_path.parent
        trial_id = feature_dir.name
        rel_parent = feature_dir.parent.relative_to(input_root)
        slice_dir = slice_root / rel_parent / trial_id
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                feature_dir=feature_dir,
                feature_matrix_path=feature_path,
                raw_feature_matrix_path=feature_dir / f"{trial_id}_roi_feature_matrix.csv",
                trace_matrix_path=find_first_existing(feature_dir, ("*_trace_matrix.npy",)),
                response_matrix_path=find_first_existing(feature_dir, ("*_response_matrix.npy",)),
                angle_response_matrix_path=find_first_existing(feature_dir, ("*_angle_response_matrix.npy",)),
                slice_matrix_path=find_first_existing(slice_dir, ("*_stim_slice_feature_matrix_zscored.npy",)),
                slice_matrix_raw_path=find_first_existing(slice_dir, ("*_stim_slice_feature_matrix.npy",)),
                slice_columns_path=find_first_existing(slice_dir, ("*_stim_slice_feature_columns.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_hierarchical_cluster_labels.csv",
        out_dir / f"{trial_id}_hierarchical_cluster_summary.csv",
        out_dir / f"{trial_id}_hierarchical_clustering_summary.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def clean_step_outputs(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    removed = 0
    for pattern in STEP_OUTPUT_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
    return removed


def numeric_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    skip = {
        "trial_id",
        "roi_id",
        "source_roi_id",
        "roi_source",
        "roi_type",
        "manual_roi_id",
        "suite2p_original_id",
        "previous_suite2p_original_id",
        "stat_index",
        "response_type",
    }
    cols = [c for c in df.columns if c not in skip]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.to_numpy(dtype=np.float32), list(numeric.columns)


def slice_feature_names(path: Path | None, n_features: int) -> list[str]:
    if path is not None and path.exists():
        columns = read_csv_if_exists(path)
        if "feature_name" in columns.columns and len(columns) == n_features:
            return columns["feature_name"].astype(str).tolist()
    return [f"slice_feature_{i + 1}" for i in range(n_features)]


def row_zscore(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float32)
    center = np.nanmean(matrix, axis=1, keepdims=True)
    scale = np.nanstd(matrix, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return np.nan_to_num((matrix - center) / scale, nan=0.0).astype(np.float32)


def has_usable_feature_columns(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix)
    return bool(matrix.ndim == 2 and matrix.shape[1] > 0)


def safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def read_csv_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Could not read CSV %s: %s", path, exc)
        return pd.DataFrame()


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        LOGGER.warning("Could not read JSON %s: %s", path, exc)
        return {}


def dff_trial_dir(output_root: Path, trial: TrialInput) -> Path:
    return output_root / "06_dff" / trial.rel_parent / trial.trial_id


def first_finite_time(row: pd.Series, columns: tuple[str, ...]) -> tuple[float | None, str | None]:
    for column in columns:
        if column not in row.index:
            continue
        value = safe_float(row.get(column), None)
        if value is not None:
            return float(value), column
    return None, None


def stim_rows(stim_events: pd.DataFrame) -> list[dict]:
    if stim_events.empty:
        return []
    rows: list[dict] = []
    for index, stim in stim_events.reset_index(drop=True).iterrows():
        start, start_column = first_finite_time(
            stim,
            (
                "analog_detected_start_time_sec",
                "analog_anchored_pulse_onset_sec",
                "start_time_sec",
                "mcu_onset_sec",
                "protocol_packet_onset_sec",
            ),
        )
        if start is None:
            continue
        end, end_column = first_finite_time(
            stim,
            (
                "analog_detected_end_time_sec",
                "analog_anchored_pulse_offset_sec",
                "end_time_sec",
                "mcu_offset_sec",
                "protocol_packet_offset_sec",
            ),
        )
        duration = safe_float(stim.get("duration_sec"), None)
        if end is None:
            end = start + (duration if duration is not None else 1.0)
        rows.append(
            {
                "stim_index": int(stim.get("stim_index", index + 1)),
                "start": float(start),
                "end": float(end),
                "angle": stim.get("pol_angle", np.nan),
                "stim_type": stim.get("stim_type", ""),
                "timing_source_column": start_column,
                "timing_end_column": end_column,
            }
        )
    return rows


def load_trial_stim_events(output_root: Path, trial: TrialInput) -> pd.DataFrame:
    dff_dir = dff_trial_dir(output_root, trial)
    stim_events = read_csv_if_exists(find_first_existing(dff_dir, ("*_stim_events.csv",)))
    if not stim_events.empty:
        return stim_events
    global_events = read_csv_if_exists(output_root / "02_stim_map" / "stim_events.csv")
    if global_events.empty or "trialID" not in global_events.columns:
        return pd.DataFrame()
    return global_events[global_events["trialID"].astype(str) == trial.trial_id].copy()


def fps_for_trial(output_root: Path, trial: TrialInput, default_fps: float) -> float:
    summary = load_json(find_first_existing(dff_trial_dir(output_root, trial), ("*_dff_summary.json",)))
    fps = safe_float(summary.get("fps"), default_fps)
    return float(fps) if fps is not None and fps > 0 else float(default_fps)


def stimulus_label(stim: dict) -> str:
    angle = stim.get("angle")
    if pd.notna(angle):
        try:
            return f"AoLP {float(angle):g} deg"
        except Exception:
            return f"AoLP {angle}"
    stim_type = str(stim.get("stim_type", "")).strip()
    if stim_type:
        return stim_type
    return f"stim {stim.get('stim_index', '')}".strip()


def compact_stimulus_label(stim: dict) -> str:
    stim_type = stimulus_kind(stim)
    prefix = {"pulse": "P", "sustain": "S", "nostim": "N"}.get(stim_type, stim_type[:1].upper() if stim_type else "")
    angle = stim.get("angle")
    if pd.notna(angle):
        try:
            return f"{prefix}{float(angle):g}" if prefix else f"{float(angle):g}"
        except Exception:
            return f"{prefix}{angle}" if prefix else str(angle)
    if stim_type:
        return stim_type[:8]
    return str(stim.get("stim_index", "")).strip()


def stimulus_kind(stim: dict) -> str:
    return str(stim.get("stim_type", "") or "").strip().lower()


def stimulus_color(stim: dict) -> str:
    kind = stimulus_kind(stim)
    if "sustain" in kind or kind in {"steady", "continuous"}:
        return "#7b61ff"
    if "pulse" in kind or "flash" in kind:
        return "#d99a00"
    return "#6b7280"


def row_robust_scale(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    baseline = np.nanmedian(matrix, axis=1, keepdims=True)
    scale = np.nanpercentile(np.abs(matrix - baseline), 95, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return np.nan_to_num((matrix - baseline) / scale, nan=0.0).astype(np.float32)


def global_color_limits(matrix: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(matrix, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    high = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    return -high, high


def trace_heatmap_canvas(n_roi: int, n_frames: int, stimuli: list[dict]) -> tuple[float, float]:
    width_from_time = 10.0 + min(12.0, max(0.0, n_frames / 165.0))
    width_from_stim = 11.0 + min(13.0, max(0.0, len(stimuli) * 0.45))
    width = max(14.0, width_from_time, width_from_stim)
    height = max(8.5, min(28.0, 4.2 + n_roi * 0.082))
    return width, height


def choose_cluster_matrix(
    features: pd.DataFrame,
    trial: TrialInput,
    args: argparse.Namespace,
    normalization: str,
) -> tuple[np.ndarray, list[str], str]:
    source = args.cluster_source
    if source == "slices":
        path = trial.slice_matrix_raw_path if normalization == "raw" else trial.slice_matrix_path
        if path and path.exists():
            matrix = np.load(path, allow_pickle=True).astype(np.float32, copy=False)
            if has_usable_feature_columns(matrix):
                names = slice_feature_names(trial.slice_columns_path, matrix.shape[1])
                if normalization == "raw":
                    return np.nan_to_num(matrix, nan=0.0).astype(np.float32), names, "stimulus_slices_baseline_z"
                return row_zscore(matrix), names, "stimulus_slices_row_zscore"
            LOGGER.warning("Stimulus-slice matrix has no usable feature columns; falling back to z-scored feature matrix.")
        LOGGER.warning("Stimulus-slice matrix is unavailable; falling back to z-scored feature matrix.")
    if source == "traces" and trial.trace_matrix_path and trial.trace_matrix_path.exists():
        matrix = np.load(trial.trace_matrix_path, allow_pickle=True).astype(np.float32, copy=False)
        names = [f"frame_{i + 1}" for i in range(matrix.shape[1])]
        if normalization == "raw":
            return np.nan_to_num(matrix, nan=0.0).astype(np.float32), names, "traces_raw_dff"
        return row_zscore(matrix), names, "traces_row_zscore"
    if source == "responses" and trial.response_matrix_path and trial.response_matrix_path.exists():
        matrix = np.load(trial.response_matrix_path, allow_pickle=True).astype(np.float32, copy=False)
        names = [f"response_{i + 1}" for i in range(matrix.shape[1])]
        if normalization == "raw":
            return np.nan_to_num(matrix, nan=0.0).astype(np.float32), names, "responses_raw"
        return row_zscore(matrix), names, "responses_row_zscore"
    if source == "angle" and trial.angle_response_matrix_path and trial.angle_response_matrix_path.exists():
        matrix = np.load(trial.angle_response_matrix_path, allow_pickle=True).astype(np.float32, copy=False)
        names = [f"angle_{i + 1}" for i in range(matrix.shape[1])]
        if normalization == "raw":
            return np.nan_to_num(matrix, nan=0.0).astype(np.float32), names, "angle_raw_response"
        return row_zscore(matrix), names, "angle_row_zscore"
    if source == "traces" and trial.angle_response_matrix_path and trial.angle_response_matrix_path.exists():
        LOGGER.warning("Trace matrix is unavailable; falling back to row-zscored angle response matrix.")
        matrix = np.load(trial.angle_response_matrix_path, allow_pickle=True).astype(np.float32, copy=False)
        names = [f"angle_{i + 1}" for i in range(matrix.shape[1])]
        if normalization == "raw":
            return np.nan_to_num(matrix, nan=0.0).astype(np.float32), names, "angle_raw_response"
        return row_zscore(matrix), names, "angle_row_zscore"
    if source != "features":
        LOGGER.warning("Requested cluster source %s is unavailable; falling back to z-scored feature matrix.", source)
    if normalization == "raw" and trial.raw_feature_matrix_path and trial.raw_feature_matrix_path.exists():
        matrix, names = numeric_feature_matrix(pd.read_csv(trial.raw_feature_matrix_path))
        return matrix, names, "features_raw"
    matrix, names = numeric_feature_matrix(features)
    return matrix, names, "features_zscored"


def save_dendrogram(linkage_matrix: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(f"Hierarchical clustering - {trial_id}")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_clustered_heatmap(matrix: np.ndarray, order: np.ndarray, labels: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if matrix.size == 0:
        return
    import matplotlib.pyplot as plt

    ordered = matrix[order]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(ordered, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(f"Clustered feature heatmap - {trial_id}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("ROI ordered by cluster")
    fig.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_clustered_trace_heatmap(
    trace_matrix: np.ndarray,
    order: np.ndarray,
    labels: np.ndarray,
    fps: float,
    stimuli: list[dict],
    out_path: Path,
    trial_id: str,
    dpi: int,
    mode: str,
    cluster_source: str,
) -> bool:
    if trace_matrix.size == 0 or trace_matrix.shape[0] != len(labels):
        return False
    import matplotlib.pyplot as plt

    if mode == "normalized":
        ordered = row_robust_scale(trace_matrix)[order]
        cmap = "coolwarm"
        vmin, vmax = -2.0, 2.0
        color_label = "normalized dF/F per ROI"
        title_mode = "normalized dF/F"
    elif mode == "raw":
        ordered = np.nan_to_num(np.asarray(trace_matrix, dtype=np.float32), nan=0.0)[order]
        cmap = "viridis"
        vmin, vmax = global_color_limits(ordered, percentile=99.0)
        color_label = "raw dF/F"
        title_mode = "raw dF/F"
    else:
        raise ValueError(f"Unsupported trace heatmap mode: {mode}")
    ordered_labels = labels[order]
    n_roi, n_frames = ordered.shape
    x0 = 0.0
    x1 = float(n_frames / fps) if fps > 0 else float(n_frames)

    width, height = trace_heatmap_canvas(n_roi=n_roi, n_frames=n_frames, stimuli=stimuli)
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=5,
        height_ratios=[0.9, 1.15, 8.2],
        width_ratios=[1.55, 0.32, 0.20, 14.0, 0.34],
        hspace=0.06,
        wspace=0.05,
    )
    ax_title = fig.add_subplot(grid[0, 3])
    ax_title.set_axis_off()
    ax_stim = fig.add_subplot(grid[1, 3])
    ax_label = fig.add_subplot(grid[2, 0])
    ax_cluster = fig.add_subplot(grid[2, 1])
    ax_gap = fig.add_subplot(grid[2, 2])
    ax = fig.add_subplot(grid[2, 3], sharex=ax_stim)
    cax = fig.add_subplot(grid[2, 4])
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(n_roi, 0)
    ax_label.set_axis_off()
    ax_gap.set_axis_off()

    im = ax.imshow(
        ordered,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[x0, x1, n_roi, 0],
    )
    ax_title.text(
        0.0,
        0.82,
        f"Cluster-ordered ROI time-response heatmap ({title_mode})",
        transform=ax_title.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        color="#111827",
        weight="bold",
    )
    ax_title.text(1.0, 0.82, trial_id, transform=ax_title.transAxes, ha="right", va="center", fontsize=10, color="#4b5563")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")

    cluster_codes = np.asarray(ordered_labels, dtype=float).reshape(-1, 1)
    ax_cluster.imshow(cluster_codes, aspect="auto", interpolation="nearest", cmap="tab20", extent=[0, 1, n_roi, 0])
    ax_cluster.set_xticks([])
    ax_cluster.set_yticks([])
    ax_label.text(0.56, n_roi / 2, "Cluster", ha="center", va="center", rotation=90, fontsize=9, color="#374151")

    boundaries = np.flatnonzero(np.diff(ordered_labels) != 0) + 1
    for boundary in boundaries:
        ax.axhline(boundary, color="white", lw=0.7, alpha=0.85)
        ax_cluster.axhline(boundary, color="white", lw=0.7, alpha=0.85)
    for cluster in sorted(set(ordered_labels)):
        positions = np.flatnonzero(ordered_labels == cluster)
        if len(positions):
            y_mid = float((positions[0] + positions[-1] + 1) / 2)
            ax_label.text(0.92, y_mid, f"C{int(cluster)}", ha="right", va="center", fontsize=9, color="#1f2933", weight="bold")

    ax_stim.set_xlim(x0, x1)
    ax_stim.set_ylim(0, 1.08)
    ax_stim.set_yticks([])
    ax_stim.tick_params(axis="x", labelbottom=False)
    for spine in ax_stim.spines.values():
        spine.set_visible(False)
    seen_kinds: set[str] = set()
    plotted_stimuli: list[dict] = []
    for stim in stimuli:
        start = stim.get("start")
        end = stim.get("end")
        if start is None or end is None or end <= start:
            continue
        start = max(float(start), x0)
        end = min(float(end), x1)
        if end <= start:
            continue
        stim = dict(stim)
        stim["start"] = start
        stim["end"] = end
        plotted_stimuli.append(stim)
    for stim in plotted_stimuli:
        color = stimulus_color(stim)
        start = float(stim["start"])
        end = float(stim["end"])
        ax_stim.axvspan(start, end, ymin=0.05, ymax=0.24, color=color, alpha=0.70, lw=0)
        ax.axvspan(start, end, color=color, alpha=0.08, lw=0)
        ax_stim.text(
            (start + end) / 2,
            0.80,
            compact_stimulus_label(stim),
            va="top",
            ha="center",
            fontsize=6.2,
            color=color,
        )
        seen_kinds.add(stimulus_kind(stim))
    legend_parts = []
    if any("pulse" in kind or "flash" in kind for kind in seen_kinds):
        legend_parts.append("P=pulse")
    if any("sustain" in kind or kind in {"steady", "continuous"} for kind in seen_kinds):
        legend_parts.append("S=sustain")
    uses_analog = any(str(stim.get("timing_source_column", "")).startswith("analog_") for stim in stimuli)
    legend_text = "analog stimulus / AoLP (deg)" if uses_analog else "stimulus / AoLP (deg)"
    if legend_parts:
        legend_text += "   " + "   ".join(legend_parts)
    ax_title.text(0.0, 0.26, legend_text, transform=ax_title.transAxes, ha="left", va="center", fontsize=8.5, color="#4b5563")
    ax_title.text(
        1.0,
        0.26,
        f"cluster source: {cluster_source}",
        transform=ax_title.transAxes,
        ha="right",
        va="center",
        fontsize=8.5,
        color="#6b7280",
    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(color_label)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.96, bottom=0.08)
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)
    return True


def save_cluster_mean_traces(trace_matrix: np.ndarray, labels: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if trace_matrix.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster in sorted(set(labels)):
        mean_trace = np.nanmean(trace_matrix[labels == cluster], axis=0)
        ax.plot(mean_trace, lw=1, label=f"C{cluster}")
    ax.set_title(f"Cluster mean traces - {trial_id}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean dF/F")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_cluster_mean_angle(angle_matrix: np.ndarray, labels: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if angle_matrix.size == 0 or angle_matrix.shape[0] != len(labels):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for cluster in sorted(set(labels)):
        ax.plot(np.nanmean(angle_matrix[labels == cluster], axis=0), marker="o", lw=1, label=f"C{cluster}")
    ax.set_title(f"Cluster mean angle response - {trial_id}")
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Mean response")
    ax.legend(fontsize=8)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def output_name(trial_id: str, tag: str, suffix: str) -> str:
    return f"{trial_id}_{tag}_{suffix}" if tag else f"{trial_id}_{suffix}"


def prepare_matrix_for_linkage(matrix: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.distance import pdist

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise TrialSkipError("Cluster matrix is not two-dimensional.")
    if matrix.shape[0] < 2:
        raise TrialSkipError("Need at least 2 ROI for hierarchical clustering.")
    if matrix.shape[1] == 0:
        raise TrialSkipError("No feature columns available for hierarchical clustering.")

    keep_rows = np.all(np.isfinite(matrix), axis=1)
    if metric == "correlation":
        row_scale = np.nanstd(matrix, axis=1)
        keep_rows &= np.isfinite(row_scale) & (row_scale > 1e-8)
    elif metric == "cosine":
        row_norm = np.linalg.norm(np.nan_to_num(matrix, nan=0.0), axis=1)
        keep_rows &= np.isfinite(row_norm) & (row_norm > 1e-8)

    if int(np.count_nonzero(keep_rows)) < 2:
        raise TrialSkipError(f"Not enough informative ROI rows for {metric} clustering.")

    filtered = matrix[keep_rows]
    distances = pdist(filtered, metric=metric)
    if distances.size == 0:
        raise TrialSkipError("Distance matrix is empty after filtering.")
    if not np.all(np.isfinite(distances)):
        raise TrialSkipError(f"Distance matrix contains non-finite values for {metric} clustering.")
    return filtered, keep_rows


def process_cluster_variant(
    trial: TrialInput,
    out_dir: Path,
    args: argparse.Namespace,
    features: pd.DataFrame,
    normalization: str,
    tag: str = "",
) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage, leaves_list

    matrix, feature_names, cluster_source = choose_cluster_matrix(features, trial, args, normalization=normalization)
    method = args.linkage_method
    metric = args.distance_metric
    if metric is None:
        metric = "euclidean" if normalization == "raw" else "correlation"
    if method == "ward" and metric != "euclidean":
        LOGGER.warning("Ward linkage requires euclidean distance; using euclidean.")
        metric = "euclidean"
    matrix, keep_rows = prepare_matrix_for_linkage(matrix, metric=metric)
    if int(np.count_nonzero(keep_rows)) != len(keep_rows):
        LOGGER.info(
            "[skip-roi] %s%s dropped %d non-informative ROI row(s) before clustering.",
            trial.trial_id,
            f" ({tag})" if tag else "",
            int(len(keep_rows) - np.count_nonzero(keep_rows)),
        )
    linkage_matrix = linkage(matrix, method=method, metric=metric)
    labels = fcluster(linkage_matrix, t=args.n_clusters, criterion="maxclust").astype(int)
    order = leaves_list(linkage_matrix)

    label_cols = [
        col
        for col in (
            "trial_id",
            "roi_id",
            "source_roi_id",
            "roi_source",
            "roi_type",
            "manual_roi_id",
            "suite2p_original_id",
            "previous_suite2p_original_id",
            "stat_index",
        )
        if col in features.columns
    ]
    labels_df = features.loc[keep_rows, label_cols].copy()
    labels_df["hierarchical_cluster"] = labels
    labels_df.to_csv(out_dir / output_name(trial.trial_id, tag, "hierarchical_cluster_labels.csv"), index=False)
    cluster_summary = labels_df.groupby("hierarchical_cluster", as_index=False).agg(n_roi=("roi_id", "count"))
    cluster_summary.to_csv(out_dir / output_name(trial.trial_id, tag, "hierarchical_cluster_summary.csv"), index=False)

    title_trial_id = f"{trial.trial_id} ({cluster_source})" if tag else trial.trial_id
    save_dendrogram(linkage_matrix, out_dir / output_name(trial.trial_id, tag, "dendrogram.png"), title_trial_id, args.dpi)
    save_clustered_heatmap(matrix, order, labels, out_dir / output_name(trial.trial_id, tag, "clustered_heatmap.png"), title_trial_id, args.dpi)
    wrote_trace_heatmap_raw = False
    wrote_trace_heatmap_normalized = False
    if trial.trace_matrix_path and trial.trace_matrix_path.exists():
        trace_matrix = np.load(trial.trace_matrix_path, allow_pickle=True)
        trace_matrix = np.asarray(trace_matrix)
        if trace_matrix.ndim == 2 and trace_matrix.shape[0] == len(keep_rows):
            trace_matrix = trace_matrix[keep_rows]
            stimuli = stim_rows(load_trial_stim_events(args.output_root, trial))
            fps = fps_for_trial(args.output_root, trial, args.default_fps)
            save_cluster_mean_traces(trace_matrix, labels, out_dir / output_name(trial.trial_id, tag, "cluster_mean_traces.png"), title_trial_id, args.dpi)
            wrote_trace_heatmap_normalized = save_clustered_trace_heatmap(
                trace_matrix=trace_matrix,
                order=order,
                labels=labels,
                fps=fps,
                stimuli=stimuli,
                out_path=out_dir / output_name(trial.trial_id, tag, "clustered_trace_heatmap_normalized_dff.png"),
                trial_id=title_trial_id,
                dpi=args.dpi,
                mode="normalized",
                cluster_source=cluster_source,
            )
            wrote_trace_heatmap_raw = save_clustered_trace_heatmap(
                trace_matrix=trace_matrix,
                order=order,
                labels=labels,
                fps=fps,
                stimuli=stimuli,
                out_path=out_dir / output_name(trial.trial_id, tag, "clustered_trace_heatmap_raw_dff.png"),
                trial_id=title_trial_id,
                dpi=args.dpi,
                mode="raw",
                cluster_source=cluster_source,
            )
        else:
            LOGGER.warning("%s%s trace matrix shape does not match ROI mask; skipping trace summary plots.", trial.trial_id, f" ({tag})" if tag else "")
    if trial.angle_response_matrix_path and trial.angle_response_matrix_path.exists():
        angle_matrix = np.load(trial.angle_response_matrix_path, allow_pickle=True)
        angle_matrix = np.asarray(angle_matrix)
        if angle_matrix.ndim == 2 and angle_matrix.shape[0] == len(keep_rows):
            save_cluster_mean_angle(angle_matrix[keep_rows], labels, out_dir / output_name(trial.trial_id, tag, "cluster_mean_angle_tuning.png"), title_trial_id, args.dpi)
        else:
            LOGGER.warning("%s%s angle-response matrix shape does not match ROI mask; skipping angle summary plot.", trial.trial_id, f" ({tag})" if tag else "")

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "output_tag": tag or "primary",
        "n_roi_input": int(len(keep_rows)),
        "n_roi": int(matrix.shape[0]),
        "n_roi_dropped_before_clustering": int(len(keep_rows) - np.count_nonzero(keep_rows)),
        "n_features": int(matrix.shape[1]),
        "n_clusters": int(args.n_clusters),
        "cluster_normalization": normalization,
        "cluster_source": cluster_source,
        "linkage_method": method,
        "distance_metric": metric,
        "feature_names": feature_names,
        "wrote_clustered_trace_heatmap": bool(wrote_trace_heatmap_normalized),
        "wrote_clustered_trace_heatmap_normalized_dff": bool(wrote_trace_heatmap_normalized),
        "wrote_clustered_trace_heatmap_raw_dff": bool(wrote_trace_heatmap_raw),
    }
    (out_dir / output_name(trial.trial_id, tag, "hierarchical_clustering_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(trial.feature_matrix_path)
    primary_normalization = "normalized" if args.cluster_normalization == "both" else args.cluster_normalization
    try:
        summary = process_cluster_variant(
            trial=trial,
            out_dir=out_dir,
            args=args,
            features=features,
            normalization=primary_normalization,
            tag="",
        )
    except TrialSkipError as exc:
        return "skipped", {"trial_id": trial.trial_id, "status": "skipped", "message": str(exc)}
    if args.cluster_normalization == "both":
        try:
            raw_summary = process_cluster_variant(
                trial=trial,
                out_dir=out_dir,
                args=args,
                features=features,
                normalization="raw",
                tag="raw_cluster",
            )
            summary["also_wrote_raw_cluster_outputs"] = True
            summary["raw_cluster_source"] = raw_summary.get("cluster_source")
            summary["raw_cluster_summary_path"] = output_name(trial.trial_id, "raw_cluster", "hierarchical_clustering_summary.json")
        except TrialSkipError as exc:
            summary["also_wrote_raw_cluster_outputs"] = False
            summary["raw_cluster_status"] = "skipped"
            summary["raw_cluster_message"] = str(exc)
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hierarchical clustering on ROI response patterns.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-11 root. Default: OUTPUT_ROOT/11_population_features.")
    parser.add_argument("--slice-root", type=Path, help="Step-12 root. Default: OUTPUT_ROOT/12_stimulus_slice_features.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--trial-id", help="Only process one trial ID, or a comma-separated list of trial IDs.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cluster-source",
        choices=("slices", "angle", "features", "responses", "traces"),
        default="slices",
        help="Matrix used for clustering. Default slices uses normalized peri-stimulus response slices.",
    )
    parser.add_argument(
        "--cluster-normalization",
        choices=("normalized", "raw", "both"),
        default="normalized",
        help="Step 12 cluster matrix scaling. normalized uses row-wise z-scored traces/angle responses; raw clusters unscaled values; both writes normalized primary outputs plus raw-cluster comparison outputs.",
    )
    parser.add_argument("--linkage-method", choices=("ward", "average", "complete"), default="average")
    parser.add_argument(
        "--distance-metric",
        choices=("euclidean", "correlation", "cosine"),
        default=None,
        help="Distance metric. Default is correlation for normalized clustering and euclidean for raw clustering.",
    )
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--default-fps", type=float, default=2.0, help="Fallback fps for trace heatmaps when step-06 summary is unavailable.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root).resolve()
    args.output_root = output_root
    input_root = args.input_root.expanduser().resolve() if args.input_root else default_input_root(output_root).resolve()
    slice_root = args.slice_root.expanduser().resolve() if args.slice_root else default_slice_root(output_root).resolve()
    out_root = step_output_root(output_root)
    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1
    trials = discover_trials(input_root, slice_root)
    excluded_trial_ids = load_excluded_trial_ids(output_root)
    if excluded_trial_ids:
        trials = [trial for trial in trials if trial.trial_id not in excluded_trial_ids]
    if args.trial_id:
        wanted = {item.strip() for item in args.trial_id.split(",") if item.strip()}
        trials = [trial for trial in trials if trial.trial_id in wanted]
    summary = RunSummary(found=len(trials))
    rows = []
    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Slice root : %s", slice_root)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d trial(s).", len(trials))
    for trial in trials:
        out_dir = trial_output_dir(out_root, trial)
        if required_outputs_done(out_dir, trial.trial_id) and args.action == "skip":
            summary.skipped += 1
            LOGGER.info("[skip] %s", trial.trial_id)
            continue
        if args.dry_run:
            summary.processed += 1
            LOGGER.info("[dry-run] Would cluster %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, args)
            rows.append(row)
            if status == "skipped":
                summary.skipped += 1
                LOGGER.info("[skip] %s: %s", trial.trial_id, row.get("message", "no valid clustering matrix"))
            else:
                summary.processed += 1
                LOGGER.info("[ok] %s: n_clusters=%s", trial.trial_id, row.get("n_clusters"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "hierarchical_clustering_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
