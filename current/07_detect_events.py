#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect calcium events from dF/F traces.

Default layout:
  input : DATA_ROOT/06_dff/
  output: DATA_ROOT/07_events/

Events here are calcium events detected from dF/F traces. They should not be
interpreted as electrophysiological spikes without additional validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("detect_events")

STEP_NAME = "07_events"
STEP_OUTPUT_PATTERNS = (
    "*_event_table.csv",
    "*_event_binary.npy",
    "*_event_mask.npy",
    "*_event_rate_by_roi.csv",
    "*_event_summary.csv",
    "*_event_summary.json",
    "*_event_raster.png",
    "*_event_raster.pdf",
    "*_event_examples.png",
    "*_event_examples.pdf",
    "*_metadata.json",
    "*_stim_events.csv",
    "*_stim_map.csv",
    "*_stim_pulse_events.csv",
    "*_roi_table.csv",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    dff_path: Path
    roi_table_path: Path
    metadata_path: Path | None
    stim_events_path: Path | None
    stim_map_path: Path | None
    stim_pulse_events_path: Path | None
    dff_summary_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_input_root(output_root: Path) -> Path:
    return output_root / "06_dff"


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
    for dff_path in sorted(input_root.rglob("*_dff.npy")):
        input_dir = dff_path.parent
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                dff_path=dff_path,
                roi_table_path=input_dir / f"{trial_id}_roi_table.csv",
                metadata_path=find_first_existing(input_dir, ("*_metadata.json",)),
                stim_events_path=find_first_existing(input_dir, ("*_stim_events.csv",)),
                stim_map_path=find_first_existing(input_dir, ("*_stim_map.csv",)),
                stim_pulse_events_path=find_first_existing(input_dir, ("*_stim_pulse_events.csv",)),
                dff_summary_path=find_first_existing(input_dir, ("*_dff_summary.json",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_event_table.csv",
        out_dir / f"{trial_id}_event_binary.npy",
        out_dir / f"{trial_id}_event_rate_by_roi.csv",
        out_dir / f"{trial_id}_event_summary.csv",
        out_dir / f"{trial_id}_event_summary.json",
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
                removed += 1
                LOGGER.info("Removed old step-07 folder: %s", path)
            elif path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                LOGGER.info("Removed old step-07 file: %s", path)
    return removed


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    LOGGER.info("Copied %s -> %s", src, dst)
    return True


def copy_sidecar_outputs(trial: TrialInput, out_dir: Path) -> None:
    copy_if_exists(trial.metadata_path, out_dir / f"{trial.trial_id}_metadata.json")
    copy_if_exists(trial.stim_events_path, out_dir / f"{trial.trial_id}_stim_events.csv")
    copy_if_exists(trial.stim_map_path, out_dir / f"{trial.trial_id}_stim_map.csv")
    copy_if_exists(trial.stim_pulse_events_path, out_dir / f"{trial.trial_id}_stim_pulse_events.csv")
    copy_if_exists(trial.roi_table_path, out_dir / f"{trial.trial_id}_roi_table.csv")


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


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


def fps_from_summary(summary: dict, default_fps: float) -> float:
    fps = safe_float(summary.get("fps"), None)
    return float(fps) if fps is not None and fps > 0 else float(default_fps)


def robust_sigma(trace: np.ndarray) -> tuple[float, float, float]:
    median = float(np.nanmedian(trace))
    mad = float(np.nanmedian(np.abs(trace - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(trace))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-6
    return median, mad, sigma


def detect_events_robust_threshold(
    trace: np.ndarray,
    fps: float,
    threshold_sigma: float,
    max_duration_sec: float,
    min_distance_sec: float,
) -> list[dict]:
    baseline, mad, sigma = robust_sigma(trace)
    threshold = baseline + threshold_sigma * sigma
    half_threshold = baseline + 0.5 * (threshold - baseline)
    max_duration = max(1, int(round(max_duration_sec * fps)))
    min_distance = max(1, int(round(min_distance_sec * fps)))
    above = trace >= threshold
    onset_candidates = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    events = []
    last_onset = -min_distance
    for onset in onset_candidates:
        if onset - last_onset < min_distance:
            continue
        search_end = min(len(trace), onset + max_duration)
        below_half = np.flatnonzero(trace[onset:search_end] < half_threshold)
        offset = int(onset + below_half[0]) if len(below_half) else search_end - 1
        if offset <= onset:
            offset = min(len(trace) - 1, onset + 1)
        segment = trace[onset : offset + 1]
        peak = int(onset + np.nanargmax(segment))
        amplitude = float(trace[peak] - baseline)
        auc = float(np.trapz(np.maximum(segment - baseline, 0), dx=1.0 / fps))
        events.append(
            {
                "onset_frame": int(onset),
                "peak_frame": peak,
                "offset_frame": int(offset),
                "amplitude": amplitude,
                "prominence": float(trace[peak] - threshold),
                "duration_sec": float((offset - onset + 1) / fps),
                "auc": auc,
                "baseline": baseline,
                "mad": mad,
                "robust_sigma": sigma,
                "threshold": threshold,
            }
        )
        last_onset = onset
    return events


def detect_events_find_peaks(
    trace: np.ndarray,
    fps: float,
    min_prominence: float | None,
    min_distance_sec: float,
    min_height: float | None,
    min_width_sec: float,
    max_duration_sec: float,
) -> list[dict]:
    from scipy.signal import find_peaks, peak_widths

    baseline, mad, sigma = robust_sigma(trace)
    distance = max(1, int(round(min_distance_sec * fps)))
    width = max(1, int(round(min_width_sec * fps)))
    peaks, props = find_peaks(
        trace,
        prominence=min_prominence,
        distance=distance,
        height=min_height,
        width=width,
    )
    widths = peak_widths(trace, peaks, rel_height=0.5)[0] if len(peaks) else []
    max_duration = max(1, int(round(max_duration_sec * fps)))
    events = []
    for i, peak in enumerate(peaks):
        half_width = int(max(1, round(widths[i] / 2))) if len(widths) else width
        onset = max(0, int(peak) - half_width)
        offset = min(len(trace) - 1, int(peak) + half_width)
        if offset - onset + 1 > max_duration:
            offset = min(len(trace) - 1, onset + max_duration - 1)
        segment = trace[onset : offset + 1]
        prominence = float(props.get("prominences", [np.nan] * len(peaks))[i])
        amplitude = float(trace[peak] - baseline)
        events.append(
            {
                "onset_frame": int(onset),
                "peak_frame": int(peak),
                "offset_frame": int(offset),
                "amplitude": amplitude,
                "prominence": prominence,
                "duration_sec": float((offset - onset + 1) / fps),
                "auc": float(np.trapz(np.maximum(segment - baseline, 0), dx=1.0 / fps)),
                "baseline": baseline,
                "mad": mad,
                "robust_sigma": sigma,
                "threshold": min_height if min_height is not None else np.nan,
            }
        )
    return events


def detect_events_for_trace(trace: np.ndarray, fps: float, args: argparse.Namespace) -> list[dict]:
    if args.event_method == "find-peaks":
        return detect_events_find_peaks(
            trace=trace,
            fps=fps,
            min_prominence=args.min_prominence,
            min_distance_sec=args.min_distance_sec,
            min_height=args.min_height,
            min_width_sec=args.min_width_sec,
            max_duration_sec=args.max_event_duration_sec,
        )
    return detect_events_robust_threshold(
        trace=trace,
        fps=fps,
        threshold_sigma=args.event_threshold_sigma,
        max_duration_sec=args.max_event_duration_sec,
        min_distance_sec=args.min_distance_sec,
    )


def build_event_outputs(
    dff: np.ndarray,
    roi_table: pd.DataFrame,
    fps: float,
    trial_id: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    n_roi, n_frames = dff.shape
    binary = np.zeros((n_roi, n_frames), dtype=np.uint8)
    mask = np.zeros((n_roi, n_frames), dtype=np.uint8)
    event_rows = []
    rate_rows = []

    for roi_idx in range(n_roi):
        trace = np.asarray(dff[roi_idx], dtype=np.float32)
        events = detect_events_for_trace(trace, fps=fps, args=args)
        roi_id = int(roi_table.iloc[roi_idx]["roi_id"]) if roi_idx < len(roi_table) and "roi_id" in roi_table else roi_idx + 1
        suite2p_id = (
            int(roi_table.iloc[roi_idx]["suite2p_original_id"])
            if roi_idx < len(roi_table) and "suite2p_original_id" in roi_table
            else roi_idx
        )
        for event_id, event in enumerate(events, start=1):
            onset = event["onset_frame"]
            offset = event["offset_frame"]
            binary[roi_idx, onset] = 1
            mask[roi_idx, onset : offset + 1] = 1
            event_rows.append(
                {
                    "trial_id": trial_id,
                    "roi_id": roi_id,
                    "suite2p_original_id": suite2p_id,
                    "event_id": event_id,
                    "onset_frame": onset,
                    "peak_frame": event["peak_frame"],
                    "offset_frame": offset,
                    "onset_sec": onset / fps,
                    "peak_sec": event["peak_frame"] / fps,
                    "offset_sec": offset / fps,
                    "amplitude": event["amplitude"],
                    "prominence": event["prominence"],
                    "duration_sec": event["duration_sec"],
                    "auc": event["auc"],
                    "baseline": event["baseline"],
                    "robust_sigma": event["robust_sigma"],
                    "threshold": event["threshold"],
                    "method": args.event_method,
                }
            )
        duration_min = n_frames / fps / 60.0
        rate_rows.append(
            {
                "trial_id": trial_id,
                "roi_id": roi_id,
                "suite2p_original_id": suite2p_id,
                "n_events": len(events),
                "event_rate_hz": len(events) / (n_frames / fps),
                "event_rate_per_min": len(events) / duration_min if duration_min > 0 else np.nan,
                "mean_amplitude": float(np.nanmean([e["amplitude"] for e in events])) if events else 0.0,
                "mean_duration_sec": float(np.nanmean([e["duration_sec"] for e in events])) if events else 0.0,
                "mean_auc": float(np.nanmean([e["auc"] for e in events])) if events else 0.0,
            }
        )

    return pd.DataFrame(event_rows), binary, mask, pd.DataFrame(rate_rows)


def stim_spans(stim_events: pd.DataFrame) -> list[tuple[float, float]]:
    if stim_events.empty or "start_time_sec" not in stim_events.columns:
        return []
    spans = []
    for _, row in stim_events.iterrows():
        start = safe_float(row.get("start_time_sec"), None)
        if start is None:
            continue
        end = safe_float(row.get("end_time_sec"), None)
        duration = safe_float(row.get("duration_sec"), None)
        if end is None and duration is not None:
            end = start + duration
        if end is None or end < start:
            end = start
        spans.append((float(start), float(end)))
    return spans


def mark_stimuli(ax, stim_events: pd.DataFrame) -> None:
    for start, end in stim_spans(stim_events):
        if end > start:
            ax.axvspan(start, end, color="tab:orange", alpha=0.14, lw=0)
        else:
            ax.axvline(start, color="tab:orange", alpha=0.35, lw=0.8)


def save_event_raster(
    binary: np.ndarray,
    fps: float,
    stim_events: pd.DataFrame,
    output_path: Path,
    trial_id: str,
    max_rois: int,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    raster = binary
    if max_rois > 0 and binary.shape[0] > max_rois:
        event_counts = np.sum(binary, axis=1)
        keep = np.argsort(event_counts)[::-1][:max_rois]
        raster = binary[keep]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(
        raster,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        extent=[0, raster.shape[1] / fps, raster.shape[0], 0],
    )
    mark_stimuli(ax, stim_events)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI")
    ax.set_title(f"Calcium event raster - {trial_id}")
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_event_examples(
    dff: np.ndarray,
    binary: np.ndarray,
    roi_table: pd.DataFrame,
    fps: float,
    stim_events: pd.DataFrame,
    output_path: Path,
    trial_id: str,
    seed: int,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    event_counts = np.sum(binary, axis=1)
    choices = []
    if len(event_counts):
        choices.append(int(np.argmax(event_counts)))
        choices.append(int(np.random.default_rng(seed).integers(0, len(event_counts))))
    choices = [idx for i, idx in enumerate(choices) if idx not in choices[:i]]
    if not choices:
        return

    time_axis = np.arange(dff.shape[1]) / fps
    fig, axes = plt.subplots(len(choices), 1, figsize=(12, max(3, 2.1 * len(choices))), sharex=True)
    if len(choices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, choices):
        mark_stimuli(ax, stim_events)
        ax.plot(time_axis, dff[idx], lw=0.9, color="tab:blue")
        event_frames = np.flatnonzero(binary[idx] > 0)
        if len(event_frames):
            ax.scatter(event_frames / fps, dff[idx, event_frames], s=18, color="tab:red", zorder=3)
        roi_id = int(roi_table.iloc[idx]["roi_id"]) if idx < len(roi_table) and "roi_id" in roi_table else idx + 1
        ax.set_ylabel(f"ROI {roi_id}", rotation=0, labelpad=28, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Calcium event examples - {trial_id}", y=0.995)
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    if not trial.dff_path.exists() or not trial.roi_table_path.exists():
        return "failed", {"trial_id": trial.trial_id, "status": "failed", "message": "Missing dff.npy or roi_table.csv"}

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_outputs(trial, out_dir)

    dff = np.load(trial.dff_path, allow_pickle=True).astype(np.float32, copy=False)
    if dff.ndim != 2:
        raise ValueError(f"dff.npy must be ROI x frame, got shape={dff.shape}")
    roi_table = pd.read_csv(trial.roi_table_path)
    if len(roi_table) != dff.shape[0]:
        LOGGER.warning("ROI table length (%d) differs from dF/F ROI count (%d)", len(roi_table), dff.shape[0])

    dff_summary = load_json(trial.dff_summary_path)
    fps = fps_from_summary(dff_summary, default_fps=args.default_fps)
    stim_events = read_csv_if_exists(trial.stim_events_path)

    event_table, event_binary, event_mask, rate_table = build_event_outputs(
        dff=dff,
        roi_table=roi_table,
        fps=fps,
        trial_id=trial.trial_id,
        args=args,
    )

    event_table.to_csv(out_dir / f"{trial.trial_id}_event_table.csv", index=False)
    np.save(out_dir / f"{trial.trial_id}_event_binary.npy", event_binary)
    np.save(out_dir / f"{trial.trial_id}_event_mask.npy", event_mask)
    rate_table.to_csv(out_dir / f"{trial.trial_id}_event_rate_by_roi.csv", index=False)

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "event_method": args.event_method,
        "event_threshold_sigma": float(args.event_threshold_sigma),
        "n_roi": int(dff.shape[0]),
        "n_frames": int(dff.shape[1]),
        "fps": float(fps),
        "n_events": int(len(event_table)),
        "mean_events_per_roi": float(rate_table["n_events"].mean()) if len(rate_table) else 0.0,
        "mean_event_rate_hz": float(rate_table["event_rate_hz"].mean()) if len(rate_table) else 0.0,
        "dff_path": str(trial.dff_path),
        "roi_table_path": str(trial.roi_table_path),
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{trial.trial_id}_event_summary.csv", index=False)
    with (out_dir / f"{trial.trial_id}_event_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_event_raster(
        binary=event_binary,
        fps=fps,
        stim_events=stim_events,
        output_path=out_dir / f"{trial.trial_id}_event_raster.png",
        trial_id=trial.trial_id,
        max_rois=args.raster_max_rois,
        dpi=args.dpi,
    )
    save_event_examples(
        dff=dff,
        binary=event_binary,
        roi_table=roi_table,
        fps=fps,
        stim_events=stim_events,
        output_path=out_dir / f"{trial.trial_id}_event_examples.png",
        trial_id=trial.trial_id,
        seed=args.random_seed,
        dpi=args.dpi,
    )
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect calcium events from dF/F traces.")
    parser.add_argument("--data-root", type=Path, required=True, help="Original data root.")
    parser.add_argument("--input-root", type=Path, help="Step-06 dF/F root. Default: OUTPUT_ROOT/06_dff.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip", help="Existing-output behavior.")
    parser.add_argument("--dry-run", action="store_true", help="Print work plan without reading dF/F arrays or writing files.")
    parser.add_argument("--event-method", choices=("robust-threshold", "find-peaks"), default="robust-threshold")
    parser.add_argument("--event-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--max-event-duration-sec", type=float, default=20.0)
    parser.add_argument("--min-distance-sec", type=float, default=1.0)
    parser.add_argument("--min-prominence", type=float, default=None)
    parser.add_argument("--min-height", type=float, default=None)
    parser.add_argument("--min-width-sec", type=float, default=0.5)
    parser.add_argument("--default-fps", type=float, default=2.0)
    parser.add_argument("--raster-max-rois", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--random-seed", type=int, default=0)
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
        if args.dry_run:
            LOGGER.warning("Input root does not exist yet: %s", input_root)
            return 0
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1

    trials = discover_trials(input_root)
    summary = RunSummary(found=len(trials))
    rows: list[dict] = []

    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d trial(s) with dF/F arrays.", len(trials))

    for trial in trials:
        out_dir = trial_output_dir(out_root, trial)
        if required_outputs_done(out_dir, trial.trial_id) and args.action == "skip":
            summary.skipped += 1
            LOGGER.info("[skip] %s", trial.trial_id)
            continue
        if args.dry_run:
            summary.processed += 1
            LOGGER.info("[dry-run] Would detect calcium events from %s -> %s", trial.dff_path, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, args)
            rows.append(row)
            if status == "processed":
                summary.processed += 1
                LOGGER.info("[ok] %s: n_events=%s", trial.trial_id, row.get("n_events"))
            elif status == "skipped":
                summary.skipped += 1
                LOGGER.info("[skip] %s: %s", trial.trial_id, row.get("message"))
            else:
                summary.failed += 1
                LOGGER.error("[failed] %s: %s", trial.trial_id, row.get("message"))
        except Exception as exc:
            summary.failed += 1
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "event_summary.csv", index=False)

    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
