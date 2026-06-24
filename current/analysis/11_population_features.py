#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build per-ROI population feature matrices from steps 06-09."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("population_features")
STEP_NAME = "11_population_features"
ROI_METADATA_COLUMNS = [
    "source_roi_id",
    "roi_source",
    "roi_type",
    "manual_roi_id",
    "suite2p_original_id",
    "previous_suite2p_original_id",
    "stat_index",
]
NON_FEATURE_COLUMNS = {"trial_id", "roi_id", "response_type", *ROI_METADATA_COLUMNS}
STEP_OUTPUT_PATTERNS = (
    "*_roi_feature_matrix.csv",
    "*_roi_feature_matrix_zscored.csv",
    "*_trace_matrix.npy",
    "*_response_matrix.npy",
    "*_angle_response_matrix.npy",
    "*_feature_description.json",
    "*_population_feature_summary.json",
    "*_feature_correlation_heatmap.png",
    "*_feature_correlation_heatmap.pdf",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    dff_dir: Path
    roi_table_path: Path
    dff_path: Path
    event_rate_path: Path | None
    response_summary_path: Path | None
    angle_summary_path: Path | None
    angle_response_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_dff_root(output_root: Path) -> Path:
    return output_root / "06_dff"


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


def discover_trials(output_root: Path, dff_root: Path) -> list[TrialInput]:
    trials = []
    event_root = output_root / "07_events"
    response_root = output_root / "08_stim_response"
    angle_root = output_root / "09_angle_tuning"
    for roi_path in sorted(dff_root.rglob("*_roi_table.csv")):
        dff_dir = roi_path.parent
        trial_id = dff_dir.name
        rel_parent = dff_dir.parent.relative_to(dff_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                dff_dir=dff_dir,
                roi_table_path=roi_path,
                dff_path=dff_dir / f"{trial_id}_dff.npy",
                event_rate_path=find_first_existing(event_root / rel_parent / trial_id, ("*_event_rate_by_roi.csv",)),
                response_summary_path=find_first_existing(response_root / rel_parent / trial_id, ("*_roi_response_summary.csv",)),
                angle_summary_path=find_first_existing(angle_root / rel_parent / trial_id, ("*_preferred_angle_by_roi.csv",)),
                angle_response_path=find_first_existing(angle_root / rel_parent / trial_id, ("*_angle_response_table.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_roi_feature_matrix.csv",
        out_dir / f"{trial_id}_roi_feature_matrix_zscored.csv",
        out_dir / f"{trial_id}_population_feature_summary.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def clean_step_outputs(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    n = 0
    for pattern in STEP_OUTPUT_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            n += 1
    return n


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def add_trace_stats(features: pd.DataFrame, dff: np.ndarray) -> pd.DataFrame:
    features = features.copy()
    features["dff_skewness"] = pd.Series([pd.Series(row).skew() for row in dff])
    features["dff_kurtosis"] = pd.Series([pd.Series(row).kurt() for row in dff])
    features["baseline_noise"] = np.nanstd(dff, axis=1)
    features["fraction_nan"] = np.mean(~np.isfinite(dff), axis=1)
    features["fraction_zero"] = np.mean(dff == 0, axis=1)
    return features


def merge_optional(features: pd.DataFrame, table: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    if table.empty or "roi_id" not in table.columns:
        return features
    columns = ["roi_id"] + [c for c in keep if c in table.columns]
    return features.merge(table[columns], on="roi_id", how="left")


def zscore_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    skip = NON_FEATURE_COLUMNS
    for col in out.columns:
        if col in skip:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        std = float(values.std(ddof=0))
        out[col] = (values - float(values.mean())) / std if std > 0 else 0.0
    return out


def save_feature_corr(features: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    numeric = features.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    import matplotlib.pyplot as plt

    corr = numeric.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    ax.set_title(f"Feature correlation - {trial_id}")
    fig.colorbar(im, ax=ax, label="correlation")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def angle_response_matrix(angle_table: pd.DataFrame) -> np.ndarray:
    if angle_table.empty:
        return np.empty((0, 0), dtype=np.float32)
    pivot = angle_table.pivot_table(index="roi_id", columns="pol_angle", values="mean_response", aggfunc="mean")
    return pivot.to_numpy(dtype=np.float32)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not trial.dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F array: {trial.dff_path}")
    roi = pd.read_csv(trial.roi_table_path)
    dff = np.load(trial.dff_path, allow_pickle=True).astype(np.float32, copy=False)

    base_columns = [
        "trial_id",
        "roi_id",
        *ROI_METADATA_COLUMNS,
        "x_mean",
        "y_mean",
        "npix",
        "mean_dff",
        "std_dff",
        "max_dff",
        "snr_like",
    ]
    features = roi[[col for col in base_columns if col in roi.columns]].copy()
    features = add_trace_stats(features, dff)
    features = merge_optional(features, read_csv(trial.event_rate_path), ["event_rate_hz", "n_events", "mean_amplitude", "mean_duration_sec", "mean_auc"])
    features = merge_optional(features, read_csv(trial.response_summary_path), ["mean_response", "max_response", "mean_delta", "max_zscore", "mean_event_rate_response", "mean_latency_sec", "response_type"])
    features = merge_optional(features, read_csv(trial.angle_summary_path), ["preferred_angle", "preferred_response", "preferred_reliability", "orthogonal_response", "OSI", "vector_strength", "circular_variance", "angle_selective"])
    features_z = zscore_features(features)

    angle_table = read_csv(trial.angle_response_path)
    np.save(out_dir / f"{trial.trial_id}_trace_matrix.npy", dff)
    response_cols = [c for c in ["mean_response", "max_response", "mean_delta", "max_zscore", "mean_event_rate_response"] if c in features]
    np.save(out_dir / f"{trial.trial_id}_response_matrix.npy", features[response_cols].to_numpy(dtype=np.float32) if response_cols else np.empty((len(features), 0), dtype=np.float32))
    np.save(out_dir / f"{trial.trial_id}_angle_response_matrix.npy", angle_response_matrix(angle_table))
    features.to_csv(out_dir / f"{trial.trial_id}_roi_feature_matrix.csv", index=False)
    features_z.to_csv(out_dir / f"{trial.trial_id}_roi_feature_matrix_zscored.csv", index=False)
    save_feature_corr(features, out_dir / f"{trial.trial_id}_feature_correlation_heatmap.png", trial.trial_id, args.dpi)

    description = {
        "trace_features": ["mean_dff", "std_dff", "max_dff", "dff_skewness", "dff_kurtosis"],
        "event_features": ["event_rate_hz", "n_events", "mean_amplitude", "mean_duration_sec", "mean_auc"],
        "stim_response_features": response_cols,
        "angle_features": ["preferred_angle", "preferred_reliability", "OSI", "vector_strength", "circular_variance"],
    }
    (out_dir / f"{trial.trial_id}_feature_description.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(len(features)),
        "n_features": int(features.shape[1]),
        "n_trace_frames": int(dff.shape[1]),
        "has_event_features": bool(trial.event_rate_path),
        "has_stim_response_features": bool(trial.response_summary_path),
        "has_angle_features": bool(trial.angle_summary_path),
    }
    (out_dir / f"{trial.trial_id}_population_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ROI-level population feature matrices.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-06 root. Default: OUTPUT_ROOT/06_dff.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--trial-id", help="Only process one trial ID, or a comma-separated list of trial IDs.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
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
    out_root = step_output_root(output_root)
    if not dff_root.exists():
        LOGGER.error("Input root does not exist: %s", dff_root)
        return 1
    trials = discover_trials(output_root, dff_root)
    if args.trial_id:
        wanted = {item.strip() for item in args.trial_id.split(",") if item.strip()}
        trials = [trial for trial in trials if trial.trial_id in wanted]
    summary = RunSummary(found=len(trials))
    rows = []
    LOGGER.info("Input root : %s", dff_root)
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
            LOGGER.info("[dry-run] Would build features for %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, args)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_roi=%s n_features=%s", trial.trial_id, row.get("n_roi"), row.get("n_features"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "population_feature_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
