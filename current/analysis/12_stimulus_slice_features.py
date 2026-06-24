#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build stimulus-slice response calls and clustering matrices from step 08."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("stimulus_slice_features")
STEP_NAME = "12_stimulus_slice_features"
ROI_METADATA_COLUMNS = [
    "source_roi_id",
    "roi_source",
    "roi_type",
    "manual_roi_id",
    "suite2p_original_id",
    "previous_suite2p_original_id",
    "stat_index",
]
STEP_OUTPUT_PATTERNS = (
    "*_stim_slice_response_table.csv",
    "*_stim_slice_roi_summary.csv",
    "*_stim_slice_feature_matrix.npy",
    "*_stim_slice_feature_matrix_zscored.npy",
    "*_stim_slice_feature_columns.csv",
    "*_peri_stimulus_tensor_normalized.npy",
    "*_stim_slice_response_heatmap.png",
    "*_stim_slice_response_heatmap.pdf",
    "*_stim_slice_feature_summary.json",
    "*_stimulus_evoked_slice_panels",
    "stimulus_slice_feature_summary.csv",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    response_dir: Path
    tensor_path: Path
    response_table_path: Path | None
    roi_summary_path: Path | None
    roi_table_path: Path | None
    stim_events_path: Path | None
    summary_path: Path | None


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


def default_input_root(output_root: Path) -> Path:
    return output_root / "08_stim_response"


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def trial_output_dir(step_root: Path, trial: TrialInput) -> Path:
    if str(trial.rel_parent) in ("", "."):
        return step_root / trial.trial_id
    return step_root / trial.rel_parent / trial.trial_id


def find_first_existing(folder: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_trials(input_root: Path) -> list[TrialInput]:
    trials: list[TrialInput] = []
    for tensor_path in sorted(input_root.rglob("*_peri_stimulus_tensor.npy")):
        response_dir = tensor_path.parent
        trial_id = response_dir.name
        rel_parent = response_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                response_dir=response_dir,
                tensor_path=tensor_path,
                response_table_path=find_first_existing(response_dir, ("*_stim_response_table.csv",)),
                roi_summary_path=find_first_existing(response_dir, ("*_roi_response_summary.csv",)),
                roi_table_path=find_first_existing(response_dir, ("*_roi_table.csv",)),
                stim_events_path=find_first_existing(response_dir, ("*_stim_events.csv",)),
                summary_path=find_first_existing(response_dir, ("*_stim_response_summary.json",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_stim_slice_response_table.csv",
        out_dir / f"{trial_id}_stim_slice_roi_summary.csv",
        out_dir / f"{trial_id}_stim_slice_feature_matrix.npy",
        out_dir / f"{trial_id}_stim_slice_feature_columns.csv",
        out_dir / f"{trial_id}_stim_slice_feature_summary.json",
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


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        LOGGER.warning("Could not read CSV %s: %s", path, exc)
        return pd.DataFrame()


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not read JSON %s: %s", path, exc)
        return {}


def finite_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def roi_metadata_table(trial: TrialInput, response_table: pd.DataFrame, roi_table: pd.DataFrame) -> pd.DataFrame:
    if not roi_table.empty and "roi_id" in roi_table.columns:
        keep = ["trial_id", "roi_id"] + [col for col in ROI_METADATA_COLUMNS if col in roi_table.columns]
        return roi_table[[col for col in keep if col in roi_table.columns]].copy()
    if not response_table.empty and "roi_id" in response_table.columns:
        keep = ["trial_id", "roi_id"] + [col for col in ROI_METADATA_COLUMNS if col in response_table.columns]
        return response_table[[col for col in keep if col in response_table.columns]].drop_duplicates("roi_id").copy()
    return pd.DataFrame({"trial_id": [trial.trial_id], "roi_id": [1]})


def stim_metadata(response_table: pd.DataFrame, stim_events: pd.DataFrame, n_stim: int) -> pd.DataFrame:
    columns = [
        "stim_index",
        "stim_type",
        "pol_angle",
        "start_time_sec",
        "end_time_sec",
        "stim_duration_sec",
        "timing_source_column",
        "timing_end_column",
    ]
    if not response_table.empty and "stim_index" in response_table.columns:
        meta = response_table[[col for col in columns if col in response_table.columns]].drop_duplicates("stim_index").copy()
    elif not stim_events.empty:
        meta = stim_events[[col for col in columns if col in stim_events.columns]].copy()
    else:
        meta = pd.DataFrame()
    if "stim_index" not in meta.columns:
        meta["stim_index"] = np.arange(1, len(meta) + 1)
    rows = []
    by_index = {int(row["stim_index"]): row.to_dict() for _, row in meta.iterrows() if pd.notna(row.get("stim_index"))}
    for stim_pos in range(n_stim):
        stim_index = stim_pos + 1
        row = by_index.get(stim_index, {"stim_index": stim_index})
        rows.append(row)
    out = pd.DataFrame(rows)
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan if column not in {"stim_type", "timing_source_column", "timing_end_column"} else ""
    return out[columns]


def robust_normalize_tensor(tensor: np.ndarray, baseline_frames: int, min_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = np.asarray(tensor, dtype=np.float32)
    baseline_frames = max(1, min(int(baseline_frames), tensor.shape[2]))
    baseline = tensor[:, :, :baseline_frames]
    center = np.nanmedian(baseline, axis=2, keepdims=True)
    mad = np.nanmedian(np.abs(baseline - center), axis=2, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.nanstd(baseline, axis=2, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > min_scale), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > min_scale), scale, min_scale)
    normalized = (tensor - center) / scale
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32), center.squeeze(2), scale.squeeze(2)


def first_sustained_crossing(values: np.ndarray, threshold: float, min_frames: int) -> int | None:
    if values.size == 0:
        return None
    mask = np.asarray(values >= threshold, dtype=bool)
    if min_frames <= 1:
        hits = np.flatnonzero(mask)
        return int(hits[0]) if hits.size else None
    for start in range(0, max(0, len(mask) - min_frames + 1)):
        if bool(mask[start : start + min_frames].all()):
            return start
    return None


def first_recovery(values: np.ndarray, threshold: float, min_frames: int) -> int | None:
    if values.size == 0:
        return None
    mask = np.asarray(np.abs(values) <= threshold, dtype=bool)
    if min_frames <= 1:
        hits = np.flatnonzero(mask)
        return int(hits[0]) if hits.size else None
    for start in range(0, max(0, len(mask) - min_frames + 1)):
        if bool(mask[start : start + min_frames].all()):
            return start
    return None


def response_score(peak_z: float, mean_z: float, auc_z: float, onset_latency: float, recovery_latency: float) -> float:
    amp = max(0.0, finite_float(peak_z, 0.0))
    mean_part = max(0.0, finite_float(mean_z, 0.0))
    auc_part = np.sqrt(max(0.0, finite_float(auc_z, 0.0)))
    latency_bonus = 0.0 if not np.isfinite(onset_latency) else 1.0 / (1.0 + max(0.0, onset_latency))
    recovery_bonus = 0.0 if not np.isfinite(recovery_latency) else 1.0 / (1.0 + max(0.0, recovery_latency))
    return float(amp + 0.5 * mean_part + 0.25 * auc_part + latency_bonus + 0.5 * recovery_bonus)


def build_slice_response_table(
    trial: TrialInput,
    normalized: np.ndarray,
    raw_tensor: np.ndarray,
    roi_meta: pd.DataFrame,
    stim_meta: pd.DataFrame,
    fps: float,
    baseline_sec: float,
    response_sec: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_frames = max(1, int(round(baseline_sec * fps)))
    response_start = baseline_frames
    default_response_end = min(normalized.shape[2], response_start + max(1, int(round(response_sec * fps))))
    min_consecutive = max(1, int(round(args.min_consecutive_sec * fps)))
    for roi_pos in range(normalized.shape[0]):
        roi_row = roi_meta.iloc[roi_pos].to_dict() if roi_pos < len(roi_meta) else {"trial_id": trial.trial_id, "roi_id": roi_pos + 1}
        roi_id = int(finite_float(roi_row.get("roi_id", roi_pos + 1), roi_pos + 1))
        metadata = {col: roi_row.get(col, "") for col in ROI_METADATA_COLUMNS if col in roi_row}
        for stim_pos in range(normalized.shape[1]):
            stim = stim_meta.iloc[stim_pos].to_dict() if stim_pos < len(stim_meta) else {"stim_index": stim_pos + 1}
            duration_sec = finite_float(stim.get("stim_duration_sec"), np.nan)
            if not np.isfinite(duration_sec) or duration_sec <= 0:
                duration_sec = response_sec
            response_end = min(default_response_end, response_start + max(1, int(round(duration_sec * fps))))
            if response_end <= response_start:
                response_end = default_response_end
            onset_end = min(response_end, response_start + max(1, int(round(args.onset_window_sec * fps))))
            offset_start = min(normalized.shape[2], response_start + max(1, int(round(duration_sec * fps))))
            offset_end = min(normalized.shape[2], offset_start + max(1, int(round(args.recovery_window_sec * fps))))

            z_trace = normalized[roi_pos, stim_pos]
            raw_trace = raw_tensor[roi_pos, stim_pos]
            response = z_trace[response_start:response_end]
            onset = z_trace[response_start:onset_end]
            offset = z_trace[offset_start:offset_end]
            raw_response = raw_trace[response_start:response_end]

            peak_z = float(np.nanmax(response)) if response.size else np.nan
            mean_z = float(np.nanmean(response)) if response.size else np.nan
            min_z = float(np.nanmin(response)) if response.size else np.nan
            auc_z = float(np.nansum(np.maximum(response, 0.0)) / max(fps, 1e-6)) if response.size else np.nan
            onset_peak_z = float(np.nanmax(onset)) if onset.size else np.nan
            raw_peak = float(np.nanmax(raw_response)) if raw_response.size else np.nan
            raw_mean = float(np.nanmean(raw_response)) if raw_response.size else np.nan
            peak_frame = int(np.nanargmax(response)) if response.size and np.isfinite(peak_z) else -1
            peak_latency = peak_frame / fps if peak_frame >= 0 else np.nan
            crossing = first_sustained_crossing(response, args.response_z_threshold, min_consecutive)
            onset_latency = crossing / fps if crossing is not None else np.nan
            recovery_frame = first_recovery(offset, args.recovery_z_threshold, min_consecutive)
            recovery_latency = recovery_frame / fps if recovery_frame is not None else np.nan
            offset_mean_abs_z = float(np.nanmean(np.abs(offset))) if offset.size else np.nan

            amplitude_pass = bool(
                np.isfinite(peak_z)
                and (
                    peak_z >= args.response_z_threshold
                    or (mean_z >= args.mean_z_threshold and auc_z >= args.auc_threshold)
                )
            )
            latency_pass = bool(np.isfinite(onset_latency) and onset_latency <= args.max_onset_latency_sec)
            if offset.size == 0:
                recovery_pass = True
            else:
                recovery_pass = bool(
                    (np.isfinite(recovery_latency) and recovery_latency <= args.max_recovery_latency_sec)
                    or (np.isfinite(offset_mean_abs_z) and offset_mean_abs_z <= args.recovery_z_threshold)
                )
            suppressed = bool(np.isfinite(min_z) and min_z <= -args.response_z_threshold and not amplitude_pass)
            responsive = bool(amplitude_pass and latency_pass and recovery_pass)
            score = response_score(peak_z, mean_z, auc_z, onset_latency, recovery_latency)

            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "roi_id": roi_id,
                    **metadata,
                    "stim_index": int(finite_float(stim.get("stim_index", stim_pos + 1), stim_pos + 1)),
                    "stim_type": stim.get("stim_type", ""),
                    "pol_angle": stim.get("pol_angle", np.nan),
                    "start_time_sec": stim.get("start_time_sec", np.nan),
                    "end_time_sec": stim.get("end_time_sec", np.nan),
                    "stim_duration_sec": duration_sec,
                    "timing_source_column": stim.get("timing_source_column", ""),
                    "timing_end_column": stim.get("timing_end_column", ""),
                    "normalized_peak_z": peak_z,
                    "normalized_mean_z": mean_z,
                    "normalized_min_z": min_z,
                    "normalized_auc_z": auc_z,
                    "onset_peak_z": onset_peak_z,
                    "onset_latency_sec": onset_latency,
                    "peak_latency_sec": peak_latency,
                    "recovery_latency_sec": recovery_latency,
                    "offset_mean_abs_z": offset_mean_abs_z,
                    "raw_response_peak": raw_peak,
                    "raw_response_mean": raw_mean,
                    "amplitude_pass": amplitude_pass,
                    "latency_pass": latency_pass,
                    "recovery_pass": recovery_pass,
                    "suppressed_flag": suppressed,
                    "stimulus_evoked_flag": responsive,
                    "stimulus_response_score": score,
                    "decision_rule": "amplitude_pass AND latency_pass AND recovery_pass",
                }
            )
    return pd.DataFrame(rows)


def summarize_by_roi(slice_table: pd.DataFrame, roi_meta: pd.DataFrame) -> pd.DataFrame:
    if slice_table.empty:
        out = roi_meta.copy()
        out["n_stimulus_evoked_slices"] = 0
        return out
    group_cols = ["trial_id", "roi_id"] + [col for col in ROI_METADATA_COLUMNS if col in slice_table.columns]
    summary = (
        slice_table.groupby(group_cols, as_index=False)
        .agg(
            n_stimulus_slices=("stim_index", "count"),
            n_stimulus_evoked_slices=("stimulus_evoked_flag", "sum"),
            fraction_stimulus_evoked=("stimulus_evoked_flag", "mean"),
            max_slice_peak_z=("normalized_peak_z", "max"),
            mean_slice_peak_z=("normalized_peak_z", "mean"),
            max_slice_score=("stimulus_response_score", "max"),
            mean_onset_latency_sec=("onset_latency_sec", "mean"),
            mean_recovery_latency_sec=("recovery_latency_sec", "mean"),
            n_suppressed_slices=("suppressed_flag", "sum"),
        )
    )
    summary["slice_responsive_roi"] = summary["n_stimulus_evoked_slices"] > 0
    return summary


def build_feature_matrix(
    normalized: np.ndarray,
    stim_meta: pd.DataFrame,
    fps: float,
    baseline_sec: float,
    bin_sec: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_roi, n_stim, n_time = normalized.shape
    bin_frames = max(1, int(round(bin_sec * fps)))
    columns: list[dict] = []
    features: list[np.ndarray] = []
    for stim_pos in range(n_stim):
        stim = stim_meta.iloc[stim_pos].to_dict() if stim_pos < len(stim_meta) else {"stim_index": stim_pos + 1}
        for start in range(0, n_time, bin_frames):
            end = min(n_time, start + bin_frames)
            values = np.nanmean(normalized[:, stim_pos, start:end], axis=1)
            features.append(values.astype(np.float32))
            rel_start = start / fps - baseline_sec
            rel_end = end / fps - baseline_sec
            stim_index = int(finite_float(stim.get("stim_index", stim_pos + 1), stim_pos + 1))
            stim_type = str(stim.get("stim_type", "") or "")
            angle = stim.get("pol_angle", np.nan)
            angle_label = "nan" if pd.isna(angle) else f"{finite_float(angle):g}"
            columns.append(
                {
                    "feature_index": len(columns),
                    "feature_name": f"stim{stim_index:03d}_{stim_type or 'stim'}_angle{angle_label}_t{rel_start:.3f}_{rel_end:.3f}",
                    "stim_index": stim_index,
                    "stim_type": stim_type,
                    "pol_angle": angle,
                    "relative_time_start_sec": rel_start,
                    "relative_time_end_sec": rel_end,
                    "source": "baseline_zscored_peri_stimulus_tensor",
                }
            )
    matrix = np.vstack(features).T if features else np.empty((n_roi, 0), dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0).astype(np.float32), pd.DataFrame(columns)


def row_zscore(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    center = np.nanmean(matrix, axis=1, keepdims=True)
    scale = np.nanstd(matrix, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return np.nan_to_num((matrix - center) / scale, nan=0.0).astype(np.float32)


def save_response_heatmap(slice_table: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if slice_table.empty:
        return
    pivot = slice_table.pivot_table(index="roi_id", columns="stim_index", values="stimulus_response_score", aggfunc="mean")
    if pivot.empty:
        return
    import matplotlib.pyplot as plt

    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, values.shape[1] * 0.35), max(4, min(14, values.shape[0] * 0.06))))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(f"Stimulus-slice response score - {trial_id}")
    ax.set_xlabel("Stimulus index")
    ax.set_ylabel("ROI")
    ax.set_xticks(np.arange(values.shape[1]), [str(col) for col in pivot.columns], rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, label="response score")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_slice_call_panels(
    trial_id: str,
    out_dir: Path,
    slice_table: pd.DataFrame,
    normalized: np.ndarray,
    raw_tensor: np.ndarray,
    fps: float,
    baseline_sec: float,
    dpi: int,
) -> None:
    if slice_table.empty:
        return

    import matplotlib.pyplot as plt

    panel_dir = out_dir / f"{trial_id}_stimulus_evoked_slice_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(normalized.shape[2], dtype=float) / max(fps, 1e-6) - baseline_sec

    for _, row in slice_table.iterrows():
        roi_id = int(finite_float(row.get("roi_id"), 0))
        stim_index = int(finite_float(row.get("stim_index"), 0))
        roi_pos = roi_id - 1
        stim_pos = stim_index - 1
        if roi_pos < 0 or stim_pos < 0 or roi_pos >= normalized.shape[0] or stim_pos >= normalized.shape[1]:
            continue

        z_trace = normalized[roi_pos, stim_pos]
        raw_trace = raw_tensor[roi_pos, stim_pos]
        flag = int(bool(row.get("stimulus_evoked_flag", False)))
        amp_pass = int(bool(row.get("amplitude_pass", False)))
        lat_pass = int(bool(row.get("latency_pass", False)))
        rec_pass = int(bool(row.get("recovery_pass", False)))
        stim_duration_sec = finite_float(row.get("stim_duration_sec"), 0.0)
        stim_duration_sec = max(0.0, stim_duration_sec)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0]},
        )
        top_ax, bottom_ax = axes

        stim_color = "#16a34a" if flag == 1 else "#dc2626"
        for ax in axes:
            ax.axvspan(0.0, stim_duration_sec, color=stim_color, alpha=0.16, linewidth=0)
            ax.axvline(0.0, color="0.45", linewidth=0.8)
            ax.axvline(stim_duration_sec, color="0.45", linewidth=0.8, linestyle="--")
            ax.grid(True, alpha=0.22)

        top_ax.plot(time_axis, z_trace, color="#2563eb", linewidth=1.0)
        top_ax.axhline(0.0, color="0.55", linewidth=0.8)
        top_ax.axhline(3.0, color="#f59e0b", linewidth=0.8, linestyle="--")
        top_ax.axhline(1.5, color="#f59e0b", linewidth=0.8, linestyle=":")
        top_ax.axhline(-3.0, color="#a855f7", linewidth=0.8, linestyle="--")
        top_ax.set_ylabel("normalized dF/F\n(z)", fontsize=9)

        bottom_ax.plot(time_axis, raw_trace, color="#0f766e", linewidth=1.0)
        bottom_ax.axhline(0.0, color="0.55", linewidth=0.8)
        bottom_ax.set_ylabel("raw dF/F", fontsize=9)
        bottom_ax.set_xlabel("Time from stimulus onset (sec)", fontsize=9)

        top_ax.set_title(
            f"{trial_id} | ROI {roi_id} | stim {stim_index} | evoked={flag}",
            fontsize=11,
            fontweight="bold",
        )
        decision_text = (
            f"amp={amp_pass}  lat={lat_pass}  rec={rec_pass} | "
            f"peak_z={finite_float(row.get('normalized_peak_z'), np.nan):.2f}  "
            f"mean_z={finite_float(row.get('normalized_mean_z'), np.nan):.2f}  "
            f"auc_z={finite_float(row.get('normalized_auc_z'), np.nan):.2f}\n"
            f"onset={finite_float(row.get('onset_latency_sec'), np.nan):.2f}s  "
            f"recovery={finite_float(row.get('recovery_latency_sec'), np.nan):.2f}s  "
            f"offset_abs_z={finite_float(row.get('offset_mean_abs_z'), np.nan):.2f}  "
            f"score={finite_float(row.get('stimulus_response_score'), np.nan):.2f}"
        )
        top_ax.text(
            0.01,
            0.98,
            decision_text,
            transform=top_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.92,
                "linewidth": 0.7,
            },
        )

        panel_stem = panel_dir / f"{trial_id}_roi{roi_id:04d}_stim{stim_index:03d}_evoked{flag}"
        try:
            fig.tight_layout()
            fig.savefig(panel_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            fig.savefig(panel_stem.with_suffix(".pdf"), bbox_inches="tight")
        finally:
            plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tensor = np.load(trial.tensor_path, allow_pickle=True).astype(np.float32, copy=False)
    if raw_tensor.ndim != 3:
        raise ValueError(f"Expected 3D peri-stimulus tensor, got shape {raw_tensor.shape}")

    response_summary = load_json(trial.summary_path)
    fps = finite_float(response_summary.get("fps"), args.default_fps)
    baseline_sec = finite_float(response_summary.get("baseline_sec"), args.baseline_sec)
    response_sec = finite_float(response_summary.get("response_sec"), args.response_sec)
    if fps <= 0:
        fps = args.default_fps

    response_table = read_csv(trial.response_table_path)
    roi_table = read_csv(trial.roi_table_path)
    stim_events = read_csv(trial.stim_events_path)
    roi_meta = roi_metadata_table(trial, response_table, roi_table)
    if len(roi_meta) != raw_tensor.shape[0]:
        roi_meta = roi_meta.head(raw_tensor.shape[0]).copy()
        if len(roi_meta) < raw_tensor.shape[0]:
            roi_meta = pd.DataFrame({"trial_id": trial.trial_id, "roi_id": np.arange(1, raw_tensor.shape[0] + 1)})
    stim_meta = stim_metadata(response_table, stim_events, raw_tensor.shape[1])

    baseline_frames = max(1, int(round(baseline_sec * fps)))
    normalized, baseline_center, baseline_scale = robust_normalize_tensor(raw_tensor, baseline_frames, args.min_baseline_scale)
    slice_table = build_slice_response_table(
        trial=trial,
        normalized=normalized,
        raw_tensor=raw_tensor,
        roi_meta=roi_meta,
        stim_meta=stim_meta,
        fps=fps,
        baseline_sec=baseline_sec,
        response_sec=response_sec,
        args=args,
    )
    roi_summary = summarize_by_roi(slice_table, roi_meta)
    feature_matrix, feature_columns = build_feature_matrix(normalized, stim_meta, fps, baseline_sec, args.feature_bin_sec)
    feature_matrix_z = row_zscore(feature_matrix)

    slice_table.to_csv(out_dir / f"{trial.trial_id}_stim_slice_response_table.csv", index=False)
    roi_summary.to_csv(out_dir / f"{trial.trial_id}_stim_slice_roi_summary.csv", index=False)
    feature_columns.to_csv(out_dir / f"{trial.trial_id}_stim_slice_feature_columns.csv", index=False)
    np.save(out_dir / f"{trial.trial_id}_stim_slice_feature_matrix.npy", feature_matrix)
    np.save(out_dir / f"{trial.trial_id}_stim_slice_feature_matrix_zscored.npy", feature_matrix_z)
    np.save(out_dir / f"{trial.trial_id}_peri_stimulus_tensor_normalized.npy", normalized)
    save_response_heatmap(slice_table, out_dir / f"{trial.trial_id}_stim_slice_response_heatmap.png", trial.trial_id, args.dpi)
    save_slice_call_panels(
        trial_id=trial.trial_id,
        out_dir=out_dir,
        slice_table=slice_table,
        normalized=normalized,
        raw_tensor=raw_tensor,
        fps=fps,
        baseline_sec=baseline_sec,
        dpi=args.dpi,
    )

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(raw_tensor.shape[0]),
        "n_stimulus_slices": int(raw_tensor.shape[1]),
        "n_timepoints_per_slice": int(raw_tensor.shape[2]),
        "n_slice_features": int(feature_matrix.shape[1]),
        "fps": float(fps),
        "baseline_sec": float(baseline_sec),
        "response_sec": float(response_sec),
        "feature_bin_sec": float(args.feature_bin_sec),
        "normalization": "per ROI x stimulus baseline median divided by robust baseline MAD/std",
        "min_baseline_scale": float(args.min_baseline_scale),
        "response_z_threshold": float(args.response_z_threshold),
        "mean_z_threshold": float(args.mean_z_threshold),
        "auc_threshold": float(args.auc_threshold),
        "max_onset_latency_sec": float(args.max_onset_latency_sec),
        "recovery_z_threshold": float(args.recovery_z_threshold),
        "max_recovery_latency_sec": float(args.max_recovery_latency_sec),
        "n_responsive_roi": int(roi_summary["slice_responsive_roi"].sum()) if "slice_responsive_roi" in roi_summary else 0,
        "n_evoked_slices": int(slice_table["stimulus_evoked_flag"].sum()) if "stimulus_evoked_flag" in slice_table else 0,
        "uses_analog_timing": bool(
            "timing_source_column" in slice_table
            and slice_table["timing_source_column"].astype(str).str.startswith("analog_").any()
        ),
        "decision_rule": (
            "stimulus_evoked_flag is true when amplitude_pass, latency_pass, and recovery_pass are all true; "
            "score remains available for softer ranking."
        ),
        "slice_call_panels_dir": f"{trial.trial_id}_stimulus_evoked_slice_panels",
        "feature_matrix_rows": "ROI in roi_id order from step 08 ROI table",
        "feature_matrix_columns": "stimulus-slice time bins from normalized peri-stimulus tensor",
    }
    (out_dir / f"{trial.trial_id}_stim_slice_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stimulus-slice feature matrices and response calls.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-08 root. Default: OUTPUT_ROOT/08_stim_response.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--trial-id", help="Only process one trial ID, or a comma-separated list of trial IDs.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-sec", type=float, default=10.0)
    parser.add_argument("--response-sec", type=float, default=6.0)
    parser.add_argument("--default-fps", type=float, default=2.0)
    parser.add_argument("--feature-bin-sec", type=float, default=0.5)
    parser.add_argument("--min-baseline-scale", type=float, default=0.02)
    parser.add_argument("--response-z-threshold", type=float, default=3.0)
    parser.add_argument("--mean-z-threshold", type=float, default=1.5)
    parser.add_argument("--auc-threshold", type=float, default=1.0)
    parser.add_argument("--onset-window-sec", type=float, default=2.0)
    parser.add_argument("--min-consecutive-sec", type=float, default=0.5)
    parser.add_argument("--max-onset-latency-sec", type=float, default=3.0)
    parser.add_argument("--recovery-window-sec", type=float, default=6.0)
    parser.add_argument("--recovery-z-threshold", type=float, default=1.0)
    parser.add_argument("--max-recovery-latency-sec", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root).resolve()
    input_root = args.input_root.expanduser().resolve() if args.input_root else default_input_root(output_root).resolve()
    out_root = step_output_root(output_root)
    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1
    trials = discover_trials(input_root)
    if args.trial_id:
        wanted = {item.strip() for item in args.trial_id.split(",") if item.strip()}
        trials = [trial for trial in trials if trial.trial_id in wanted]
    summary = RunSummary(found=len(trials))
    rows = []
    LOGGER.info("Input root : %s", input_root)
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
            LOGGER.info("[dry-run] Would build stimulus-slice features for %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            _, row = process_trial(trial, out_dir, args)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_slice_features=%s n_evoked_slices=%s", trial.trial_id, row.get("n_slice_features"), row.get("n_evoked_slices"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "stimulus_slice_feature_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
