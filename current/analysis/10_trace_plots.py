#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot stimulus-aligned dF/F traces for manually curated ROIs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("trace_plots")
STEP_NAME = "10_trace_plots"
STEP_OUTPUT_PATTERNS = (
    "roi_traces",
    "roi_traces_normalized",
    "*_trace_overview.png",
    "*_trace_overview.pdf",
    "*_trace_plot_summary.csv",
    "*_trace_plot_summary.json",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    dff_dir: Path
    dff_path: Path
    roi_table_path: Path
    dff_summary_path: Path | None
    stim_events_path: Path | None
    global_stim_events_path: Path | None
    angle_summary_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_dff_root(output_root: Path) -> Path:
    return output_root / "06_dff"


def default_angle_root(output_root: Path) -> Path:
    return output_root / "09_angle_tuning"


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


def discover_trials(output_root: Path, dff_root: Path, angle_root: Path) -> list[TrialInput]:
    trials: list[TrialInput] = []
    for dff_path in sorted(dff_root.rglob("*_dff.npy")):
        dff_dir = dff_path.parent
        trial_id = dff_dir.name
        rel_parent = dff_dir.parent.relative_to(dff_root)
        angle_dir = angle_root / rel_parent / trial_id
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                dff_dir=dff_dir,
                dff_path=dff_path,
                roi_table_path=dff_dir / f"{trial_id}_roi_table.csv",
                dff_summary_path=find_first_existing(dff_dir, ("*_dff_summary.json",)),
                stim_events_path=find_first_existing(dff_dir, ("*_stim_events.csv",)),
                global_stim_events_path=output_root / "02_stim_map" / "stim_events.csv",
                angle_summary_path=find_first_existing(angle_dir, ("*_preferred_angle_by_roi.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_trace_plot_summary.csv",
        out_dir / f"{trial_id}_trace_plot_summary.json",
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


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Could not read CSV %s: %s", path, exc)
        return pd.DataFrame()


def safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def fps_from_summary(path: Path | None, default_fps: float) -> float:
    fps = safe_float(load_json(path).get("fps"), None)
    return float(fps) if fps is not None and fps > 0 else float(default_fps)


def is_manual_final_dff(path: Path | None) -> bool:
    summary = load_json(path)
    return summary.get("roi_source_used") == "manual_curation"


def smooth_trace(trace: np.ndarray, fps: float, window_sec: float, method: str) -> np.ndarray:
    if method == "none" or window_sec <= 0:
        return trace.astype(np.float32, copy=True)
    window_frames = max(1, int(round(window_sec * fps)))
    if window_frames % 2 == 0:
        window_frames += 1
    series = pd.Series(trace.astype(float))
    if method == "rolling-mean":
        out = series.rolling(window_frames, center=True, min_periods=1).mean()
    else:
        out = series.rolling(window_frames, center=True, min_periods=1).median()
    return out.to_numpy(dtype=np.float32)


def normalize_trace(trace: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return trace.astype(np.float32, copy=True), np.nan, np.nan
    center = float(np.nanmedian(finite))
    scale = float(np.nanpercentile(np.abs(finite - center), 95))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return ((trace - center) / scale).astype(np.float32), center, scale


def robust_display_limits(trace: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> tuple[float, float]:
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.nanpercentile(finite, low_pct))
    high = float(np.nanpercentile(finite, high_pct))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite))
        if not np.isfinite(spread) or spread <= 0:
            spread = 1.0
        low, high = center - spread, center + spread
    return low, high


def full_display_limits(trace: np.ndarray) -> tuple[float, float]:
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite))
        if not np.isfinite(spread) or spread <= 0:
            spread = 1.0
        low, high = center - spread, center + spread
    return low, high


def clipped_for_display(trace: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(trace, low, high).astype(np.float32)


def robust_sigma(trace: np.ndarray) -> tuple[float, float]:
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return 0.0, 1.0
    baseline = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - baseline)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-6
    return baseline, sigma


def detect_global_peak_indices(trace: np.ndarray, fps: float, threshold_sigma: float, min_distance_sec: float) -> np.ndarray:
    if trace.size < 3:
        return np.array([], dtype=int)
    baseline, sigma = robust_sigma(trace)
    threshold = baseline + threshold_sigma * sigma
    finite = np.isfinite(trace)
    candidates = np.where(
        finite[1:-1]
        & (trace[1:-1] >= threshold)
        & (trace[1:-1] >= trace[:-2])
        & (trace[1:-1] >= trace[2:])
    )[0] + 1
    if candidates.size <= 1:
        return candidates.astype(int)

    min_distance = max(1, int(round(min_distance_sec * fps)))
    selected: list[int] = []
    for idx in candidates[np.argsort(trace[candidates])[::-1]]:
        if all(abs(int(idx) - kept) >= min_distance for kept in selected):
            selected.append(int(idx))
    return np.array(sorted(selected), dtype=int)


def select_stim_peak_frames(peak_trace: np.ndarray, global_peak_indices: np.ndarray, fps: float, stimuli: list[dict], response_sec: float) -> list[tuple[int, int]]:
    frames: list[tuple[int, int]] = []
    if global_peak_indices.size == 0:
        return frames
    for stim in stimuli:
        start_frame = max(0, int(np.floor(stim["start"] * fps)))
        end_frame = min(len(peak_trace), int(np.ceil((stim["start"] + response_sec) * fps)))
        if end_frame <= start_frame:
            continue
        in_window = global_peak_indices[(global_peak_indices >= start_frame) & (global_peak_indices < end_frame)]
        if in_window.size == 0:
            continue
        best = int(in_window[np.nanargmax(peak_trace[in_window])])
        frames.append((start_frame, best))
    return frames


def roi_label(roi_table: pd.DataFrame, roi_idx: int) -> str:
    if roi_idx >= len(roi_table):
        return str(roi_idx + 1)
    row = roi_table.iloc[roi_idx]
    if "source_roi_id" in row and pd.notna(row["source_roi_id"]):
        return str(row["source_roi_id"])
    if "roi_id" in row and pd.notna(row["roi_id"]):
        return str(int(row["roi_id"]))
    return str(roi_idx + 1)


def stim_rows(stim_events: pd.DataFrame) -> list[dict]:
    if stim_events.empty or "start_time_sec" not in stim_events:
        return []
    rows: list[dict] = []
    for index, stim in stim_events.reset_index(drop=True).iterrows():
        start = safe_float(stim.get("start_time_sec"), None)
        if start is None:
            continue
        end = safe_float(stim.get("end_time_sec"), None)
        duration = safe_float(stim.get("duration_sec"), None)
        if end is None:
            end = start + (duration if duration is not None else 1.0)
        rows.append(
            {
                "stim_index": int(stim.get("stim_index", index + 1)),
                "start": start,
                "end": end,
                "angle": stim.get("pol_angle", np.nan),
                "stim_type": stim.get("stim_type", ""),
            }
        )
    return rows


def load_trial_stim_events(trial: TrialInput) -> pd.DataFrame:
    stim_events = read_csv(trial.stim_events_path)
    if not stim_events.empty:
        return stim_events
    global_events = read_csv(trial.global_stim_events_path)
    if global_events.empty or "trialID" not in global_events.columns:
        return pd.DataFrame()
    return global_events[global_events["trialID"].astype(str) == trial.trial_id].copy()


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


def response_summary_for_roi(raw_trace: np.ndarray, peak_trace: np.ndarray, peak_source: str, fps: float, stimuli: list[dict], response_sec: float, peak_frames: list[tuple[int, int]]) -> dict:
    raw_peaks = []
    analysis_peaks = []
    latencies = []
    for start_frame, peak_frame in peak_frames:
        analysis_peaks.append(float(peak_trace[peak_frame]))
        raw_peaks.append(float(raw_trace[peak_frame]))
        latencies.append(float((peak_frame - start_frame) / fps))
    prefix = "raw" if peak_source == "raw_dff" else "smoothed"
    out = {
        "max_peak_dff": float(np.nanmax(analysis_peaks)) if analysis_peaks else np.nan,
        "mean_peak_dff": float(np.nanmean(analysis_peaks)) if analysis_peaks else np.nan,
        "max_raw_dff_at_peak": float(np.nanmax(raw_peaks)) if raw_peaks else np.nan,
        "mean_raw_dff_at_peak": float(np.nanmean(raw_peaks)) if raw_peaks else np.nan,
        "mean_latency_sec": float(np.nanmean(latencies)) if latencies else np.nan,
        "n_detected_stim_peaks": int(len(peak_frames)),
    }
    out[f"max_{prefix}_peak_dff"] = out["max_peak_dff"]
    out[f"mean_{prefix}_peak_dff"] = out["mean_peak_dff"]
    return out


def mark_stimuli(ax, stimuli: list[dict], *, label_y: float = 0.98, color: str = "tab:red") -> None:
    has_region_label = False
    for stim in stimuli:
        start = stim.get("start")
        end = stim.get("end")
        if start is None or end is None or end <= start:
            continue
        ax.axvspan(
            start,
            end,
            color=color,
            alpha=0.14,
            lw=0,
            label="stimulus" if not has_region_label else None,
        )
        has_region_label = True
        ax.text(
            start,
            label_y,
            stimulus_label(stim),
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            rotation=90,
            va="top",
            ha="left",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
        )


def save_roi_trace(
    raw_trace: np.ndarray,
    smoothed_trace: np.ndarray,
    time_axis: np.ndarray,
    roi_name: str,
    preferred: pd.Series | None,
    stimuli: list[dict],
    fps: float,
    response_sec: float,
    out_path: Path,
    dpi: int,
    scale_mode: str,
    y_axis_mode: str,
    peak_frames: list[tuple[int, int]],
    peak_raw_value: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if scale_mode == "normalized":
        raw_plot, center, scale = normalize_trace(raw_trace)
        smooth_plot = ((smoothed_trace - center) / scale).astype(np.float32)
        ylabel = "Normalized dF/F"
        title_suffix = f" | peak dF/F {peak_raw_value:.3g}" if peak_raw_value is not None and np.isfinite(peak_raw_value) else ""
    else:
        raw_plot = raw_trace
        smooth_plot = smoothed_trace
        ylabel = "dF/F"
        title_suffix = ""

    if y_axis_mode == "robust":
        clip_low, clip_high = robust_display_limits(raw_plot)
        raw_display = clipped_for_display(raw_plot, clip_low, clip_high)
        smooth_display = clipped_for_display(smooth_plot, clip_low, clip_high)
        clipped_high = np.isfinite(raw_plot) & (raw_plot > clip_high)
        clipped_low = np.isfinite(raw_plot) & (raw_plot < clip_low)
    else:
        clip_low, clip_high = full_display_limits(raw_plot)
        raw_display = raw_plot
        smooth_display = smooth_plot
        clipped_high = np.zeros(raw_plot.shape, dtype=bool)
        clipped_low = np.zeros(raw_plot.shape, dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 4))
    if np.array_equal(raw_plot, smooth_plot):
        ax.plot(time_axis, raw_display, color="black", lw=0.8, label="raw dF/F")
    else:
        ax.plot(time_axis, raw_display, color="0.70", lw=0.6, label="raw dF/F")
        ax.plot(time_axis, smooth_display, color="black", lw=1.0, label="smoothed dF/F")
    if clipped_high.any():
        ax.scatter(time_axis[clipped_high], np.full(int(clipped_high.sum()), clip_high), marker="^", s=10, color="0.35", alpha=0.7, zorder=4, label="clipped high")
    if clipped_low.any():
        ax.scatter(time_axis[clipped_low], np.full(int(clipped_low.sum()), clip_low), marker="v", s=10, color="0.35", alpha=0.7, zorder=4, label="clipped low")
    mark_stimuli(ax, stimuli, color="tab:red")

    for _, peak_frame in peak_frames:
        peak_y = float(np.clip(raw_plot[peak_frame], clip_low, clip_high))
        ax.scatter(time_axis[peak_frame], peak_y, s=20, color="tab:red", zorder=5)

    title = f"ROI {roi_name}"
    if preferred is not None and not preferred.empty:
        parts = []
        if "preferred_angle" in preferred and pd.notna(preferred["preferred_angle"]):
            parts.append(f"pref {float(preferred['preferred_angle']):g}")
        if "OSI" in preferred and pd.notna(preferred["OSI"]):
            parts.append(f"OSI {float(preferred['OSI']):.2f}")
        if parts:
            title += " | " + ", ".join(parts)
    title += title_suffix
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    pad = 0.12 * (clip_high - clip_low)
    ax.set_ylim(clip_low - pad, clip_high + pad)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_overview(dff: np.ndarray, fps: float, stimuli: list[dict], out_path: Path, trial_id: str, max_rois: int, dpi: int) -> None:
    if dff.size == 0:
        return
    import matplotlib.pyplot as plt

    n_rois = dff.shape[0] if max_rois <= 0 else min(max_rois, dff.shape[0])
    matrix = dff[:n_rois].astype(float, copy=True)
    baseline = np.nanmedian(matrix, axis=1, keepdims=True)
    scale = np.nanpercentile(np.abs(matrix - baseline), 95, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    matrix = (matrix - baseline) / scale
    time_axis = np.arange(matrix.shape[1]) / fps

    fig, ax = plt.subplots(figsize=(10, max(4, min(10, n_rois * 0.035))))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-2, vmax=2, extent=[time_axis[0], time_axis[-1], n_rois, 0])
    mark_stimuli(ax, stimuli, color="goldenrod")
    ax.set_title(f"dF/F trace overview - {trial_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, label="robust-scaled dF/F")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    if args.require_manual_final and not is_manual_final_dff(trial.dff_summary_path):
        return (
            "skipped",
            {
                "trial_id": trial.trial_id,
                "status": "skipped",
                "message": "Step-06 output is not based on final manual ROI.",
                "dff_summary_path": str(trial.dff_summary_path) if trial.dff_summary_path else None,
            },
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "roi_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    norm_trace_dir = out_dir / "roi_traces_normalized"
    if args.trace_scale in ("normalized", "both"):
        norm_trace_dir.mkdir(parents=True, exist_ok=True)

    dff = np.load(trial.dff_path, allow_pickle=True).astype(np.float32, copy=False)
    roi_table = read_csv(trial.roi_table_path)
    angle_summary = read_csv(trial.angle_summary_path)
    stim_events = load_trial_stim_events(trial)
    stimuli = stim_rows(stim_events)
    fps = fps_from_summary(trial.dff_summary_path, args.default_fps)
    time_axis = np.arange(dff.shape[1]) / fps

    angle_by_roi = {}
    if not angle_summary.empty and "roi_id" in angle_summary:
        for _, row in angle_summary.iterrows():
            angle_by_roi[int(row["roi_id"])] = row

    rows = []
    max_rois = dff.shape[0] if args.max_rois <= 0 else min(args.max_rois, dff.shape[0])
    for roi_idx in range(max_rois):
        label = roi_label(roi_table, roi_idx)
        roi_id = int(roi_table.iloc[roi_idx]["roi_id"]) if "roi_id" in roi_table and roi_idx < len(roi_table) else roi_idx + 1
        preferred = angle_by_roi.get(roi_id)
        trace = dff[roi_idx]
        smoothed = smooth_trace(trace, fps, args.smooth_window_sec, args.smooth_method)
        peak_source = "raw_dff" if args.smooth_method == "none" else "smoothed_dff"
        peak_trace = trace if peak_source == "raw_dff" else smoothed
        global_peak_indices = detect_global_peak_indices(peak_trace, fps, args.peak_threshold_sigma, args.peak_min_distance_sec)
        peak_frames = select_stim_peak_frames(peak_trace, global_peak_indices, fps, stimuli, args.response_sec)
        stats = response_summary_for_roi(trace, peak_trace, peak_source, fps, stimuli, args.response_sec, peak_frames)
        peak_raw_value = stats.get("max_raw_dff_at_peak")
        rows.append(
            {
                "trial_id": trial.trial_id,
                "roi_id": roi_id,
                "roi_label": label,
                "mean_dff": float(np.nanmean(trace)),
                "std_dff": float(np.nanstd(trace)),
                **stats,
            }
        )
        if args.trace_scale in ("dff", "both"):
            save_roi_trace(
                trace,
                smoothed,
                time_axis,
                label,
                preferred,
                stimuli,
                fps,
                args.response_sec,
                trace_dir / f"ROI_{roi_id:04d}_trace.png",
                args.dpi,
                scale_mode="dff",
                y_axis_mode=args.y_axis_mode,
                peak_frames=peak_frames,
            )
        if args.trace_scale in ("normalized", "both"):
            save_roi_trace(
                trace,
                smoothed,
                time_axis,
                label,
                preferred,
                stimuli,
                fps,
                args.response_sec,
                norm_trace_dir / f"ROI_{roi_id:04d}_trace_norm.png",
                args.dpi,
                scale_mode="normalized",
                y_axis_mode=args.y_axis_mode,
                peak_frames=peak_frames,
                peak_raw_value=peak_raw_value,
            )

    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(out_dir / f"{trial.trial_id}_trace_plot_summary.csv", index=False)
    save_overview(dff, fps, stimuli, out_dir / f"{trial.trial_id}_trace_overview.png", trial.trial_id, args.overview_max_rois, args.dpi)

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(dff.shape[0]),
        "n_roi_plotted": int(max_rois),
        "n_stim_events": int(len(stimuli)),
        "n_stim_events_with_angle": int(sum(pd.notna(stim.get("angle")) for stim in stimuli)),
        "stimulus_annotations": "regions_and_aolp_when_available",
        "fps": float(fps),
        "response_sec": float(args.response_sec),
        "smooth_method": args.smooth_method,
        "smooth_window_sec": float(args.smooth_window_sec),
        "peak_source": "raw_dff" if args.smooth_method == "none" else "smoothed_dff",
        "peak_threshold_sigma": float(args.peak_threshold_sigma),
        "peak_min_distance_sec": float(args.peak_min_distance_sec),
        "trace_scale": args.trace_scale,
        "y_axis_mode": args.y_axis_mode,
        "deconvolution_used": False,
    }
    (out_dir / f"{trial.trial_id}_trace_plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot dF/F traces with stimulus annotations.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-06 root. Default: OUTPUT_ROOT/06_dff.")
    parser.add_argument("--angle-root", type=Path, help="Step-09 root. Default: OUTPUT_ROOT/09_angle_tuning.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--trial-id", help="Only process one trial ID, or a comma-separated list of trial IDs.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--response-sec", type=float, default=6.0)
    parser.add_argument("--smooth-method", choices=("rolling-median", "rolling-mean", "none"), default="none")
    parser.add_argument("--smooth-window-sec", type=float, default=0.0)
    parser.add_argument("--peak-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--peak-min-distance-sec", type=float, default=1.0)
    parser.add_argument("--trace-scale", choices=("dff", "normalized", "both"), default="both")
    parser.add_argument("--y-axis-mode", choices=("full", "robust"), default="full")
    parser.add_argument("--max-rois", type=int, default=0, help="Maximum individual ROI traces to plot; 0 means all.")
    parser.add_argument("--overview-max-rois", type=int, default=500)
    parser.add_argument("--require-manual-final", action=argparse.BooleanOptionalAction, default=True, help="Only plot traces from step-06 outputs that used final manual ROI.")
    parser.add_argument("--default-fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root).resolve()
    dff_root = args.input_root.expanduser().resolve() if args.input_root else default_dff_root(output_root).resolve()
    angle_root = args.angle_root.expanduser().resolve() if args.angle_root else default_angle_root(output_root).resolve()
    out_root = step_output_root(output_root)
    if not dff_root.exists():
        LOGGER.error("Input root does not exist: %s", dff_root)
        return 1
    trials = discover_trials(output_root, dff_root, angle_root)
    if args.trial_id:
        wanted = {item.strip() for item in args.trial_id.split(",") if item.strip()}
        trials = [trial for trial in trials if trial.trial_id in wanted]

    summary = RunSummary(found=len(trials))
    rows = []
    LOGGER.info("DFF root  : %s", dff_root)
    LOGGER.info("Angle root: %s", angle_root)
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
            LOGGER.info("[dry-run] Would plot traces for %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, args)
            rows.append(row)
            if status == "skipped":
                summary.skipped += 1
                LOGGER.info("[skip] %s: %s", trial.trial_id, row.get("message"))
            else:
                summary.processed += 1
                LOGGER.info("[ok] %s: n_roi_plotted=%s", trial.trial_id, row.get("n_roi_plotted"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "trace_plot_summary.csv", index=False)
    LOGGER.info("Trace plot summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
