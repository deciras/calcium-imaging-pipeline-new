#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze ROI response tuning across stimulus angles."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("angle_tuning")
STEP_NAME = "09_angle_tuning"
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
    "*_angle_response_table.csv",
    "*_angle_tuning_summary.csv",
    "*_preferred_angle_by_roi.csv",
    "*_angle_heatmap.png",
    "*_angle_heatmap.pdf",
    "*_polar_plots.png",
    "*_polar_plots.pdf",
    "*_osi_distribution.png",
    "*_osi_distribution.pdf",
    "*_angle_tuning_summary.json",
    "*_stim_events.csv",
    "*_stim_map.csv",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    response_table_path: Path
    roi_summary_path: Path | None
    stim_events_path: Path | None
    stim_map_path: Path | None


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
    trials = []
    for response_path in sorted(input_root.rglob("*_stim_response_table.csv")):
        input_dir = response_path.parent
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                response_table_path=response_path,
                roi_summary_path=find_first_existing(input_dir, ("*_roi_response_summary.csv",)),
                stim_events_path=find_first_existing(input_dir, ("*_stim_events.csv",)),
                stim_map_path=find_first_existing(input_dir, ("*_stim_map.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_angle_response_table.csv",
        out_dir / f"{trial_id}_angle_tuning_summary.csv",
        out_dir / f"{trial_id}_preferred_angle_by_roi.csv",
        out_dir / f"{trial_id}_angle_tuning_summary.json",
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


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_sidecar_outputs(trial: TrialInput, out_dir: Path) -> None:
    copy_if_exists(trial.stim_events_path, out_dir / f"{trial.trial_id}_stim_events.csv")
    copy_if_exists(trial.stim_map_path, out_dir / f"{trial.trial_id}_stim_map.csv")


def read_response_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def circular_stats(angles_deg: np.ndarray, responses: np.ndarray, period: float) -> tuple[float, float]:
    valid = np.isfinite(angles_deg) & np.isfinite(responses)
    if not valid.any() or np.nansum(np.abs(responses[valid])) <= 0:
        return np.nan, np.nan
    theta = angles_deg[valid] / period * 2 * np.pi
    weights = np.maximum(responses[valid], 0)
    if weights.sum() <= 0:
        weights = np.abs(responses[valid])
    vector = np.sum(weights * np.exp(1j * theta)) / max(weights.sum(), 1e-12)
    preferred = (np.angle(vector) % (2 * np.pi)) / (2 * np.pi) * period
    strength = np.abs(vector)
    return float(preferred), float(strength)


def compute_angle_tables(response_table: pd.DataFrame, angle_period: float, z_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if response_table.empty or "pol_angle" not in response_table.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    response_table = response_table.copy()
    response_table["pol_angle"] = pd.to_numeric(response_table["pol_angle"], errors="coerce")
    response_table = response_table.dropna(subset=["pol_angle"])
    if response_table.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    group_cols = ["trial_id", "roi_id"] + [col for col in ROI_METADATA_COLUMNS if col in response_table.columns]
    grouped = response_table.groupby(group_cols + ["pol_angle"], as_index=False).agg(
        mean_response=("delta_mean", "mean"),
        peak_response=("response_peak", "max"),
        mean_event_rate=("event_rate_response", "mean"),
        sem_response=("delta_mean", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(np.isfinite(x).sum(), 1)))),
        n_trials=("delta_mean", "count"),
        reliability=("zscore_response", lambda x: float(np.nanmean(np.asarray(x, dtype=float) >= z_threshold))),
    )

    preferred_rows = []
    for group_key, group in grouped.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = dict(zip(group_cols, group_key))
        group = group.sort_values("pol_angle")
        best_idx = group["mean_response"].idxmax()
        preferred_angle = float(group.loc[best_idx, "pol_angle"])
        preferred_response = float(group.loc[best_idx, "mean_response"])
        target_orth = (preferred_angle + angle_period / 2.0) % angle_period
        angle_distance = np.abs(((group["pol_angle"] - target_orth + angle_period / 2.0) % angle_period) - angle_period / 2.0)
        orthogonal_response = float(group.loc[angle_distance.idxmin(), "mean_response"])
        denom = abs(preferred_response) + abs(orthogonal_response)
        osi = (preferred_response - orthogonal_response) / denom if denom > 0 else np.nan
        vector_pref, vector_strength = circular_stats(group["pol_angle"].to_numpy(float), group["mean_response"].to_numpy(float), angle_period)
        preferred_rows.append(
            {
                **metadata,
                "preferred_angle": preferred_angle,
                "vector_preferred_angle": vector_pref,
                "preferred_response": preferred_response,
                "orthogonal_response": orthogonal_response,
                "OSI": float(osi) if np.isfinite(osi) else np.nan,
                "vector_strength": vector_strength,
                "circular_variance": 1.0 - vector_strength if np.isfinite(vector_strength) else np.nan,
                "angle_selective": bool(np.isfinite(osi) and osi >= 0.3 and preferred_response > 0),
            }
        )
    preferred = pd.DataFrame(preferred_rows)
    summary = preferred.copy()
    return grouped, summary, preferred


def save_angle_heatmap(angle_table: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if angle_table.empty:
        return
    import matplotlib.pyplot as plt

    pivot = angle_table.pivot_table(index="roi_id", columns="pol_angle", values="mean_response", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, min(10, pivot.shape[0] * 0.035))))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(int(c)) if float(c).is_integer() else f"{c:g}" for c in pivot.columns], rotation=45)
    ax.set_xlabel("Polarization angle")
    ax.set_ylabel("ROI")
    ax.set_title(f"Angle tuning heatmap - {trial_id}")
    fig.colorbar(im, ax=ax, label="mean response")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_osi_distribution(preferred: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if preferred.empty or "OSI" not in preferred:
        return
    import matplotlib.pyplot as plt

    values = pd.to_numeric(preferred["OSI"], errors="coerce").dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color="tab:blue", alpha=0.8)
    ax.set_xlabel("OSI-like index")
    ax.set_ylabel("ROI count")
    ax.set_title(f"OSI distribution - {trial_id}")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_polar_examples(angle_table: pd.DataFrame, preferred: pd.DataFrame, out_path: Path, trial_id: str, period: float, dpi: int) -> None:
    if angle_table.empty or preferred.empty:
        return
    import matplotlib.pyplot as plt

    top = preferred.sort_values("OSI", ascending=False).head(6)["roi_id"].tolist()
    if not top:
        return
    fig, axes = plt.subplots(2, 3, subplot_kw={"projection": "polar"}, figsize=(9, 6))
    axes = axes.ravel()
    for ax, roi_id in zip(axes, top):
        sub = angle_table[angle_table["roi_id"] == roi_id].sort_values("pol_angle")
        theta = sub["pol_angle"].to_numpy(float) / period * 2 * np.pi
        r = sub["mean_response"].to_numpy(float)
        if len(theta):
            theta = np.r_[theta, theta[0]]
            r = np.r_[r, r[0]]
            ax.plot(theta, r, marker="o", lw=1)
        ax.set_title(f"ROI {roi_id}", fontsize=9)
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.suptitle(f"Angle tuning examples - {trial_id}")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_outputs(trial, out_dir)
    response_table = read_response_table(trial.response_table_path)
    angle_table, summary, preferred = compute_angle_tables(response_table, args.angle_period, args.z_threshold)

    angle_table.to_csv(out_dir / f"{trial.trial_id}_angle_response_table.csv", index=False)
    summary.to_csv(out_dir / f"{trial.trial_id}_angle_tuning_summary.csv", index=False)
    preferred.to_csv(out_dir / f"{trial.trial_id}_preferred_angle_by_roi.csv", index=False)
    save_angle_heatmap(angle_table, out_dir / f"{trial.trial_id}_angle_heatmap.png", trial.trial_id, args.dpi)
    save_osi_distribution(preferred, out_dir / f"{trial.trial_id}_osi_distribution.png", trial.trial_id, args.dpi)
    save_polar_examples(angle_table, preferred, out_dir / f"{trial.trial_id}_polar_plots.png", trial.trial_id, args.angle_period, args.dpi)

    out = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_response_rows": int(len(response_table)),
        "n_angle_rows": int(len(angle_table)),
        "n_roi": int(len(preferred)),
        "n_angle_selective_roi": int(preferred["angle_selective"].sum()) if "angle_selective" in preferred else 0,
        "angle_period": float(args.angle_period),
    }
    (out_dir / f"{trial.trial_id}_angle_tuning_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return "processed", out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze response tuning across stimulus angles.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-08 root. Default: OUTPUT_ROOT/08_stim_response.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--trial-id", help="Only process one trial ID, or a comma-separated list of trial IDs.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--angle-period", type=float, choices=(180.0, 360.0), default=180.0)
    parser.add_argument("--z-threshold", type=float, default=2.0)
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
            LOGGER.info("[dry-run] Would analyze %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, args)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_roi=%s", trial.trial_id, row.get("n_roi"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "angle_tuning_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
