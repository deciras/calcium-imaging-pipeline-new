#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze stimulus-locked dF/F and calcium event responses.

Default layout:
  dF/F input : DATA_ROOT/06_dff/
  event input: DATA_ROOT/07_events/
  output     : DATA_ROOT/08_stim_response/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("stim_response")

STEP_NAME = "08_stim_response"
STEP_OUTPUT_PATTERNS = (
    "*_stim_response_table.csv",
    "*_roi_response_summary.csv",
    "*_peri_stimulus_tensor.npy",
    "*_psth_by_roi.csv",
    "*_stim_response_summary.json",
    "*_stim_response_heatmap.png",
    "*_stim_response_heatmap.pdf",
    "*_psth_examples.png",
    "*_psth_examples.pdf",
    "*_response_type_map.csv",
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
    dff_dir: Path
    event_dir: Path
    dff_path: Path
    roi_table_path: Path
    event_table_path: Path | None
    event_binary_path: Path | None
    metadata_path: Path | None
    stim_events_path: Path | None
    stim_map_path: Path | None
    stim_pulse_events_path: Path | None


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


def default_event_root(output_root: Path) -> Path:
    return output_root / "07_events"


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


def discover_trials(dff_root: Path, event_root: Path) -> list[TrialInput]:
    trials: list[TrialInput] = []
    for dff_path in sorted(dff_root.rglob("*_dff.npy")):
        dff_dir = dff_path.parent
        trial_id = dff_dir.name
        rel_parent = dff_dir.parent.relative_to(dff_root)
        event_dir = event_root / rel_parent / trial_id
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                dff_dir=dff_dir,
                event_dir=event_dir,
                dff_path=dff_path,
                roi_table_path=dff_dir / f"{trial_id}_roi_table.csv",
                event_table_path=find_first_existing(event_dir, ("*_event_table.csv",)),
                event_binary_path=find_first_existing(event_dir, ("*_event_binary.npy",)),
                metadata_path=find_first_existing(dff_dir, ("*_metadata.json",)),
                stim_events_path=find_first_existing(dff_dir, ("*_stim_events.csv",)),
                stim_map_path=find_first_existing(dff_dir, ("*_stim_map.csv",)),
                stim_pulse_events_path=find_first_existing(dff_dir, ("*_stim_pulse_events.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_stim_response_table.csv",
        out_dir / f"{trial_id}_roi_response_summary.csv",
        out_dir / f"{trial_id}_stim_response_summary.json",
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


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_sidecar_outputs(trial: TrialInput, out_dir: Path) -> None:
    copy_if_exists(trial.metadata_path, out_dir / f"{trial.trial_id}_metadata.json")
    copy_if_exists(trial.stim_events_path, out_dir / f"{trial.trial_id}_stim_events.csv")
    copy_if_exists(trial.stim_map_path, out_dir / f"{trial.trial_id}_stim_map.csv")
    copy_if_exists(trial.stim_pulse_events_path, out_dir / f"{trial.trial_id}_stim_pulse_events.csv")
    copy_if_exists(trial.roi_table_path, out_dir / f"{trial.trial_id}_roi_table.csv")


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
        return json.loads(path.read_text(encoding="utf-8"))
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


def fps_from_dff_summary(trial: TrialInput, default_fps: float) -> float:
    summary_path = trial.dff_dir / f"{trial.trial_id}_dff_summary.json"
    fps = safe_float(load_json(summary_path).get("fps"), None)
    return float(fps) if fps is not None and fps > 0 else float(default_fps)


def frame_window(start_sec: float, end_sec: float, fps: float, n_frames: int) -> tuple[int, int]:
    start = max(0, min(n_frames, int(np.floor(start_sec * fps))))
    end = max(start, min(n_frames, int(np.ceil(end_sec * fps))))
    return start, end


def event_count(event_binary: np.ndarray | None, roi_idx: int, start: int, end: int) -> int:
    if event_binary is None or event_binary.size == 0 or end <= start:
        return 0
    return int(np.nansum(event_binary[roi_idx, start:end]))


def classify_response(row: pd.Series, z_threshold: float) -> str:
    strong_on = row["onset_zscore"] >= z_threshold
    strong_sustained = row["sustained_zscore"] >= z_threshold
    strong_offset = row["offset_zscore"] >= z_threshold
    suppressed = row["response_mean"] < row["baseline_mean"] - z_threshold * max(row["baseline_std"], 1e-6)
    n_strong = int(strong_on) + int(strong_sustained) + int(strong_offset)
    if suppressed and not n_strong:
        return "suppressed"
    if n_strong >= 2:
        return "mixed"
    if strong_sustained:
        return "sustained"
    if strong_on:
        return "onset"
    if strong_offset:
        return "offset"
    return "nonresponsive"


def build_response_tables(
    trial: TrialInput,
    dff: np.ndarray,
    event_binary: np.ndarray | None,
    roi_table: pd.DataFrame,
    stim_events: pd.DataFrame,
    fps: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows = []
    psth_rows = []
    tensor = np.full(
        (dff.shape[0], max(len(stim_events), 0), int(round((args.baseline_sec + args.response_sec) * fps))),
        np.nan,
        dtype=np.float32,
    )

    if stim_events.empty or "start_time_sec" not in stim_events.columns:
        empty_summary = roi_table[["trial_id", "roi_id", "suite2p_original_id"]].copy()
        empty_summary["response_type"] = "no_stim"
        return pd.DataFrame(), empty_summary, pd.DataFrame(), tensor, empty_summary

    for stim_idx, stim in stim_events.reset_index(drop=True).iterrows():
        start_sec = safe_float(stim.get("start_time_sec"), None)
        if start_sec is None:
            continue
        end_sec = safe_float(stim.get("end_time_sec"), None)
        duration_sec = safe_float(stim.get("duration_sec"), None)
        if end_sec is None:
            end_sec = start_sec + (duration_sec if duration_sec is not None else args.response_sec)
        baseline_start, baseline_end = frame_window(start_sec - args.baseline_sec, start_sec, fps, dff.shape[1])
        response_start, response_end = frame_window(start_sec, min(end_sec, start_sec + args.response_sec), fps, dff.shape[1])
        offset_start, offset_end = frame_window(end_sec, end_sec + args.offset_response_sec, fps, dff.shape[1])
        onset_end = min(response_end, response_start + max(1, int(round(args.onset_sec * fps))))

        peri_start = max(0, int(np.floor((start_sec - args.baseline_sec) * fps)))
        peri_end = min(dff.shape[1], peri_start + tensor.shape[2])
        target_start = 0
        if start_sec - args.baseline_sec < 0:
            target_start = int(round(abs(start_sec - args.baseline_sec) * fps))

        for roi_idx in range(dff.shape[0]):
            trace = dff[roi_idx]
            baseline = trace[baseline_start:baseline_end]
            response = trace[response_start:response_end]
            onset = trace[response_start:onset_end]
            offset = trace[offset_start:offset_end]
            baseline_mean = float(np.nanmean(baseline)) if baseline.size else np.nan
            baseline_std = float(np.nanstd(baseline)) if baseline.size else np.nan
            response_mean = float(np.nanmean(response)) if response.size else np.nan
            response_peak = float(np.nanmax(response)) if response.size else np.nan
            response_auc = float(np.nansum(response - baseline_mean) / fps) if response.size else np.nan
            delta_mean = response_mean - baseline_mean
            zscore = delta_mean / max(baseline_std, args.min_baseline_std) if np.isfinite(delta_mean) else np.nan
            if response.size and np.isfinite(response_peak):
                peak_rel = int(np.nanargmax(response))
                latency = peak_rel / fps
            else:
                latency = np.nan

            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "roi_id": int(roi_table.iloc[roi_idx]["roi_id"]),
                    "suite2p_original_id": int(roi_table.iloc[roi_idx]["suite2p_original_id"]),
                    "stim_index": int(stim.get("stim_index", stim_idx + 1)),
                    "stim_type": stim.get("stim_type", ""),
                    "pol_angle": stim.get("pol_angle", np.nan),
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "response_mean": response_mean,
                    "response_peak": response_peak,
                    "response_auc": response_auc,
                    "delta_mean": delta_mean,
                    "zscore_response": zscore,
                    "onset_zscore": (float(np.nanmean(onset)) - baseline_mean) / max(baseline_std, args.min_baseline_std) if onset.size else np.nan,
                    "sustained_zscore": zscore,
                    "offset_zscore": (float(np.nanmean(offset)) - baseline_mean) / max(baseline_std, args.min_baseline_std) if offset.size else np.nan,
                    "event_count_baseline": event_count(event_binary, roi_idx, baseline_start, baseline_end),
                    "event_count_response": event_count(event_binary, roi_idx, response_start, response_end),
                    "event_rate_baseline": event_count(event_binary, roi_idx, baseline_start, baseline_end) / max(args.baseline_sec, 1e-6),
                    "event_rate_response": event_count(event_binary, roi_idx, response_start, response_end) / max((response_end - response_start) / fps, 1e-6),
                    "latency_to_peak_sec": latency,
                }
            )

            slice_len = max(0, peri_end - peri_start)
            if slice_len:
                tensor[roi_idx, stim_idx, target_start : target_start + slice_len] = trace[peri_start:peri_end]

    response_table = pd.DataFrame(rows)
    if response_table.empty:
        roi_summary = roi_table[["trial_id", "roi_id", "suite2p_original_id"]].copy()
        roi_summary["response_type"] = "no_valid_stim"
        return response_table, roi_summary, pd.DataFrame(), tensor, roi_summary

    roi_summary = (
        response_table.groupby(["trial_id", "roi_id", "suite2p_original_id"], as_index=False)
        .agg(
            mean_response=("response_mean", "mean"),
            max_response=("response_peak", "max"),
            mean_delta=("delta_mean", "mean"),
            max_zscore=("zscore_response", "max"),
            mean_event_rate_response=("event_rate_response", "mean"),
            mean_latency_sec=("latency_to_peak_sec", "mean"),
        )
    )
    response_types = []
    for _, group in response_table.groupby("roi_id"):
        response_types.append({"roi_id": int(group["roi_id"].iloc[0]), "response_type": classify_response(group.loc[group["zscore_response"].idxmax()], args.response_z_threshold)})
    response_type_map = pd.DataFrame(response_types)
    roi_summary = roi_summary.merge(response_type_map, on="roi_id", how="left")

    mean_psth = np.nanmean(tensor, axis=1) if tensor.size else np.empty((0, 0))
    time_axis = (np.arange(mean_psth.shape[1]) / fps) - args.baseline_sec if mean_psth.size else []
    for roi_idx in range(mean_psth.shape[0]):
        for frame, value in enumerate(mean_psth[roi_idx]):
            psth_rows.append({"trial_id": trial.trial_id, "roi_id": int(roi_table.iloc[roi_idx]["roi_id"]), "time_sec": float(time_axis[frame]), "mean_dff": float(value)})
    return response_table, roi_summary, pd.DataFrame(psth_rows), tensor, response_type_map


def save_heatmap(roi_summary: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if roi_summary.empty:
        return
    required = ["mean_response", "max_response", "max_zscore", "mean_event_rate_response"]
    if any(column not in roi_summary.columns for column in required):
        return
    import matplotlib.pyplot as plt

    values = roi_summary[required].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(4, min(10, values.shape[0] * 0.035))))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(range(4), ["mean", "peak", "z", "event Hz"], rotation=30, ha="right")
    ax.set_ylabel("ROI")
    ax.set_title(f"Stim response summary - {trial_id}")
    fig.colorbar(im, ax=ax, label="value")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_psth_examples(psth: pd.DataFrame, roi_summary: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if psth.empty or roi_summary.empty:
        return
    import matplotlib.pyplot as plt

    keep = roi_summary.sort_values("max_zscore", ascending=False).head(5)["roi_id"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for roi_id in keep:
        sub = psth[psth["roi_id"] == roi_id]
        ax.plot(sub["time_sec"], sub["mean_dff"], lw=1, label=f"ROI {roi_id}")
    ax.axvline(0, color="tab:orange", lw=1)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Mean dF/F")
    ax.set_title(f"Example PSTH - {trial_id}")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_outputs(trial, out_dir)

    if not trial.roi_table_path.exists():
        raise FileNotFoundError(f"Missing ROI table: {trial.roi_table_path}")
    dff = np.load(trial.dff_path, allow_pickle=True).astype(np.float32, copy=False)
    roi_table = pd.read_csv(trial.roi_table_path)
    stim_events = read_csv_if_exists(trial.stim_events_path)
    event_binary = np.load(trial.event_binary_path, allow_pickle=True) if trial.event_binary_path and trial.event_binary_path.exists() else None
    fps = fps_from_dff_summary(trial, args.default_fps)

    if event_binary is not None and event_binary.shape != dff.shape:
        LOGGER.warning("Event binary shape mismatch for %s: %s vs dff %s; ignoring events.", trial.trial_id, event_binary.shape, dff.shape)
        event_binary = None

    response_table, roi_summary, psth, tensor, response_type_map = build_response_tables(
        trial=trial,
        dff=dff,
        event_binary=event_binary,
        roi_table=roi_table,
        stim_events=stim_events,
        fps=fps,
        args=args,
    )

    response_table.to_csv(out_dir / f"{trial.trial_id}_stim_response_table.csv", index=False)
    roi_summary.to_csv(out_dir / f"{trial.trial_id}_roi_response_summary.csv", index=False)
    psth.to_csv(out_dir / f"{trial.trial_id}_psth_by_roi.csv", index=False)
    response_type_map.to_csv(out_dir / f"{trial.trial_id}_response_type_map.csv", index=False)
    np.save(out_dir / f"{trial.trial_id}_peri_stimulus_tensor.npy", tensor)

    save_heatmap(roi_summary, out_dir / f"{trial.trial_id}_stim_response_heatmap.png", trial.trial_id, args.dpi)
    save_psth_examples(psth, roi_summary, out_dir / f"{trial.trial_id}_psth_examples.png", trial.trial_id, args.dpi)

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(dff.shape[0]),
        "n_frames": int(dff.shape[1]),
        "fps": float(fps),
        "n_stim_events": int(len(stim_events)),
        "n_response_rows": int(len(response_table)),
        "n_responsive_roi": int((pd.to_numeric(roi_summary.get("max_zscore", pd.Series(dtype=float)), errors="coerce") >= args.response_z_threshold).sum()) if not roi_summary.empty else 0,
        "baseline_sec": float(args.baseline_sec),
        "response_sec": float(args.response_sec),
        "offset_response_sec": float(args.offset_response_sec),
        "response_z_threshold": float(args.response_z_threshold),
    }
    (out_dir / f"{trial.trial_id}_stim_response_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze stimulus-locked dF/F and calcium event responses.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-06 dF/F root. Default: OUTPUT_ROOT/06_dff.")
    parser.add_argument("--event-root", type=Path, help="Step-07 event root. Default: OUTPUT_ROOT/07_events.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-sec", type=float, default=5.0)
    parser.add_argument("--response-sec", type=float, default=10.0)
    parser.add_argument("--offset-response-sec", type=float, default=10.0)
    parser.add_argument("--onset-sec", type=float, default=2.0)
    parser.add_argument("--response-z-threshold", type=float, default=2.0)
    parser.add_argument("--min-baseline-std", type=float, default=1e-6)
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
    event_root = args.event_root.expanduser().resolve() if args.event_root else default_event_root(output_root).resolve()
    out_root = step_output_root(output_root)

    if not dff_root.exists():
        LOGGER.error("Input root does not exist: %s", dff_root)
        return 1
    trials = discover_trials(dff_root, event_root)
    summary = RunSummary(found=len(trials))
    rows: list[dict] = []

    LOGGER.info("DFF root  : %s", dff_root)
    LOGGER.info("Event root: %s", event_root)
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
            LOGGER.info("[dry-run] Would analyze %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, args)
            rows.append(row)
            if status == "processed":
                summary.processed += 1
                LOGGER.info("[ok] %s: n_response_rows=%s", trial.trial_id, row.get("n_response_rows"))
            else:
                summary.failed += 1
                LOGGER.error("[failed] %s: %s", trial.trial_id, row.get("message"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "stim_response_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
