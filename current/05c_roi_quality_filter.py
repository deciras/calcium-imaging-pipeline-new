#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter suite2p ROIs using a manual ROI shape prior.

Default layout:
  input : DATA_ROOT/05_suite2p_roi_detection/
  prior : DATA_ROOT/05b_manual_roi_prior/manual_roi_prior.json
  output: DATA_ROOT/05c_roi_quality_filter/

This step does not modify the original suite2p folder. It writes curated ROI
tables and an iscell_curated.npy file that can be used by later steps after the
filtering rules have been inspected.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("roi_quality_filter")

STEP_NAME = "05c_roi_quality_filter"
STEP_OUTPUT_PATTERNS = (
    "*_roi_quality_table.csv",
    "*_accepted_suite2p_indices.csv",
    "*_rejected_suite2p_indices.csv",
    "*_iscell_curated.npy",
    "*_roi_quality_summary.json",
    "*_roi_quality_overlay.png",
    "*_roi_quality_overlay.pdf",
    "*_metadata.json",
    "*_stim_events.csv",
    "*_stim_map.csv",
    "*_stim_pulse_events.csv",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    plane0_dir: Path
    stat_path: Path
    ops_path: Path
    iscell_path: Path
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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_input_root(output_root: Path) -> Path:
    return output_root / "05_suite2p_roi_detection"


def default_prior_path(output_root: Path) -> Path:
    return output_root / "05b_manual_roi_prior" / "manual_roi_prior.json"


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
    for stat_path in sorted(input_root.rglob("suite2p/plane0/stat.npy")):
        plane0_dir = stat_path.parent
        input_dir = plane0_dir.parent.parent
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                plane0_dir=plane0_dir,
                stat_path=stat_path,
                ops_path=plane0_dir / "ops.npy",
                iscell_path=plane0_dir / "iscell.npy",
                metadata_path=find_first_existing(input_dir, ("*_metadata.json",)),
                stim_events_path=find_first_existing(input_dir, ("*_stim_events.csv",)),
                stim_map_path=find_first_existing(input_dir, ("*_stim_map.csv",)),
                stim_pulse_events_path=find_first_existing(input_dir, ("*_stim_pulse_events.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_roi_quality_table.csv",
        out_dir / f"{trial_id}_accepted_suite2p_indices.csv",
        out_dir / f"{trial_id}_rejected_suite2p_indices.csv",
        out_dir / f"{trial_id}_iscell_curated.npy",
        out_dir / f"{trial_id}_roi_quality_summary.json",
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
                LOGGER.info("Removed old step-05c folder: %s", path)
            elif path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                LOGGER.info("Removed old step-05c file: %s", path)
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ops(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = np.load(path, allow_pickle=True)
        item = value.item()
        return item if isinstance(item, dict) else {}
    except Exception:
        return {}


def load_iscell(path: Path, n_roi: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros(n_roi, dtype=bool), np.full(n_roi, np.nan, dtype=float)
    iscell = np.load(path, allow_pickle=True)
    if iscell.ndim != 2 or iscell.shape[0] != n_roi:
        raise ValueError(f"iscell.npy shape invalid: {iscell.shape}, expected rows={n_roi}")
    flags = iscell[:, 0].astype(bool)
    prob = iscell[:, 1].astype(float) if iscell.shape[1] >= 2 else np.full(n_roi, np.nan, dtype=float)
    return flags, prob


def scalar_prior(prior: dict, metric: str, key: str, fallback: float) -> float:
    try:
        value = float(prior["metrics"][metric][key])
        return value if np.isfinite(value) else fallback
    except Exception:
        return fallback


def build_rules(prior: dict, padding_fraction: float) -> dict:
    pad = max(0.0, float(padding_fraction))
    area_min = scalar_prior(prior, "area_px", "p05", 0.0)
    area_max = scalar_prior(prior, "area_px", "p95", np.inf)
    diam_min = scalar_prior(prior, "equiv_diameter_px", "p05", 0.0)
    diam_max = scalar_prior(prior, "equiv_diameter_px", "p95", np.inf)
    aspect_max = scalar_prior(prior, "aspect_ratio", "p95", np.inf)
    circularity_min = scalar_prior(prior, "circularity", "p05", 0.0)
    return {
        "area_px_min": area_min * (1.0 - pad),
        "area_px_max": area_max * (1.0 + pad),
        "equiv_diameter_px_min": diam_min * (1.0 - pad),
        "equiv_diameter_px_max": diam_max * (1.0 + pad),
        "aspect_ratio_max": aspect_max * (1.0 + pad),
        "circularity_min": max(0.0, circularity_min * (1.0 - pad)),
    }


def roi_geometry(roi: dict) -> dict:
    xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
    ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
    valid = np.isfinite(xpix) & np.isfinite(ypix)
    xpix = xpix[valid]
    ypix = ypix[valid]
    area = int(len(xpix))
    if area == 0:
        return {
            "area_px": 0,
            "equiv_diameter_px": np.nan,
            "width_px": np.nan,
            "height_px": np.nan,
            "aspect_ratio": np.nan,
            "perimeter_px": np.nan,
            "circularity": np.nan,
            "x_mean": np.nan,
            "y_mean": np.nan,
        }

    xmin, xmax = int(np.min(xpix)), int(np.max(xpix))
    ymin, ymax = int(np.min(ypix)), int(np.max(ypix))
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    perimeter = pixel_perimeter(xpix, ypix)
    circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else np.nan
    equiv_diameter = 2.0 * math.sqrt(area / math.pi)
    aspect_ratio = max(width, height) / max(min(width, height), 1)
    return {
        "area_px": area,
        "equiv_diameter_px": float(equiv_diameter),
        "width_px": int(width),
        "height_px": int(height),
        "aspect_ratio": float(aspect_ratio),
        "perimeter_px": float(perimeter),
        "circularity": float(circularity) if np.isfinite(circularity) else np.nan,
        "x_mean": float(np.mean(xpix)),
        "y_mean": float(np.mean(ypix)),
    }


def pixel_perimeter(xpix: np.ndarray, ypix: np.ndarray) -> float:
    pixels = set(zip(xpix.tolist(), ypix.tolist()))
    edges = 0
    for x, y in pixels:
        if (x - 1, y) not in pixels:
            edges += 1
        if (x + 1, y) not in pixels:
            edges += 1
        if (x, y - 1) not in pixels:
            edges += 1
        if (x, y + 1) not in pixels:
            edges += 1
    return float(edges)


def score_roi(geom: dict, rules: dict) -> tuple[float, dict]:
    checks = {
        "area_ok": rules["area_px_min"] <= geom["area_px"] <= rules["area_px_max"],
        "diameter_ok": rules["equiv_diameter_px_min"] <= geom["equiv_diameter_px"] <= rules["equiv_diameter_px_max"],
        "aspect_ok": geom["aspect_ratio"] <= rules["aspect_ratio_max"],
        "circularity_ok": geom["circularity"] >= rules["circularity_min"],
    }
    valid_checks = [bool(v) for v in checks.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    score = float(np.mean(valid_checks)) if valid_checks else 0.0
    return score, checks


def quality_label(score: float, accepted: bool) -> str:
    if accepted:
        return "accepted"
    if score >= 0.5:
        return "borderline"
    return "rejected"


def build_quality_table(
    stat: np.ndarray,
    iscell_flag: np.ndarray,
    iscell_prob: np.ndarray,
    rules: dict,
    min_score: float,
    require_suite2p_iscell: bool,
    trial_id: str,
) -> tuple[list[dict], np.ndarray]:
    rows = []
    accepted = np.zeros(len(stat), dtype=bool)
    for idx, roi in enumerate(stat):
        geom = roi_geometry(roi if isinstance(roi, dict) else {})
        score, checks = score_roi(geom, rules)
        pass_shape = score >= min_score
        pass_suite2p = bool(iscell_flag[idx]) if require_suite2p_iscell else True
        is_accepted = bool(pass_shape and pass_suite2p)
        accepted[idx] = is_accepted
        rows.append(
            {
                "trial_id": trial_id,
                "suite2p_original_id": idx,
                "suite2p_iscell": int(bool(iscell_flag[idx])),
                "suite2p_iscell_prob": float(iscell_prob[idx]) if np.isfinite(iscell_prob[idx]) else np.nan,
                "manual_prior_score": score,
                "accepted": int(is_accepted),
                "quality_label": quality_label(score, is_accepted),
                **geom,
                **{key: int(bool(value)) for key, value in checks.items()},
            }
        )
    return rows, accepted


def curated_iscell_array(accepted: np.ndarray, quality_rows: list[dict]) -> np.ndarray:
    scores = np.asarray([row["manual_prior_score"] for row in quality_rows], dtype=np.float32)
    out = np.zeros((len(accepted), 2), dtype=np.float32)
    out[:, 0] = accepted.astype(np.float32)
    out[:, 1] = scores
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_indices(path: Path, trial_id: str, indices: np.ndarray) -> None:
    rows = [{"trial_id": trial_id, "suite2p_original_id": int(idx)} for idx in indices]
    write_csv(path, rows)


def save_overlay(
    ops: dict,
    stat: np.ndarray,
    accepted: np.ndarray,
    output_path: Path,
    trial_id: str,
    dpi: int,
    max_rois: int,
) -> None:
    import matplotlib.pyplot as plt

    mean_img = np.asarray(ops.get("meanImg", []), dtype=np.float32)
    if mean_img.ndim != 2:
        ly = int(ops.get("Ly", 0) or 0)
        lx = int(ops.get("Lx", 0) or 0)
        if ly <= 0 or lx <= 0:
            return
        mean_img = np.zeros((ly, lx), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_img, cmap="gray")
    order = np.arange(len(stat))
    if max_rois > 0 and len(order) > max_rois:
        order = order[:max_rois]
    for idx in order:
        roi = stat[idx]
        if not isinstance(roi, dict):
            continue
        xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
        if len(xpix) == 0:
            continue
        color = "lime" if accepted[idx] else "red"
        alpha = 0.7 if accepted[idx] else 0.28
        ax.scatter(xpix, ypix, s=0.45, c=color, alpha=alpha)
    ax.set_title(f"ROI quality filter - {trial_id}")
    ax.axis("off")
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(
    trial: TrialInput,
    out_dir: Path,
    prior: dict,
    args: argparse.Namespace,
) -> tuple[str, dict]:
    missing = [path.name for path in (trial.stat_path, trial.ops_path, trial.iscell_path) if not path.exists()]
    if missing:
        return "failed", {"trial_id": trial.trial_id, "status": "failed", "message": f"Missing inputs: {', '.join(missing)}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_outputs(trial, out_dir)
    stat = np.load(trial.stat_path, allow_pickle=True)
    ops = load_ops(trial.ops_path)
    iscell_flag, iscell_prob = load_iscell(trial.iscell_path, len(stat))
    rules = build_rules(prior, padding_fraction=args.rule_padding_fraction)
    rows, accepted = build_quality_table(
        stat=stat,
        iscell_flag=iscell_flag,
        iscell_prob=iscell_prob,
        rules=rules,
        min_score=args.min_quality_score,
        require_suite2p_iscell=args.require_suite2p_iscell,
        trial_id=trial.trial_id,
    )

    accepted_indices = np.flatnonzero(accepted)
    rejected_indices = np.flatnonzero(~accepted)
    write_csv(out_dir / f"{trial.trial_id}_roi_quality_table.csv", rows)
    write_indices(out_dir / f"{trial.trial_id}_accepted_suite2p_indices.csv", trial.trial_id, accepted_indices)
    write_indices(out_dir / f"{trial.trial_id}_rejected_suite2p_indices.csv", trial.trial_id, rejected_indices)
    np.save(out_dir / f"{trial.trial_id}_iscell_curated.npy", curated_iscell_array(accepted, rows))
    save_overlay(
        ops=ops,
        stat=stat,
        accepted=accepted,
        output_path=out_dir / f"{trial.trial_id}_roi_quality_overlay.png",
        trial_id=trial.trial_id,
        dpi=args.dpi,
        max_rois=args.overlay_max_rois,
    )

    rescued = int(np.sum(accepted & ~iscell_flag))
    kept_suite2p = int(np.sum(accepted & iscell_flag))
    rejected_suite2p_cell = int(np.sum((~accepted) & iscell_flag))
    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_suite2p_roi": int(len(stat)),
        "n_suite2p_iscell": int(np.sum(iscell_flag)),
        "n_accepted": int(len(accepted_indices)),
        "n_rejected": int(len(rejected_indices)),
        "n_rescued_noncell_by_shape": rescued,
        "n_kept_suite2p_iscell": kept_suite2p,
        "n_rejected_suite2p_iscell": rejected_suite2p_cell,
        "accepted_fraction": float(len(accepted_indices) / max(len(stat), 1)),
        "min_quality_score": float(args.min_quality_score),
        "require_suite2p_iscell": bool(args.require_suite2p_iscell),
        "rule_padding_fraction": float(args.rule_padding_fraction),
        "manual_prior_path": str(args.prior_path),
        "rules": rules,
    }
    with (out_dir / f"{trial.trial_id}_roi_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter suite2p ROIs using a manual ROI shape prior.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-05 suite2p root. Default: OUTPUT_ROOT/05_suite2p_roi_detection.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--prior-path", type=Path, help="manual_roi_prior.json. Default: OUTPUT_ROOT/05b_manual_roi_prior/manual_roi_prior.json.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-quality-score", type=float, default=0.75, help="Minimum fraction of shape checks that must pass.")
    parser.add_argument("--rule-padding-fraction", type=float, default=0.0, help="Expand manual prior p05-p95 ranges by this fraction.")
    parser.add_argument("--require-suite2p-iscell", action="store_true", help="Require original suite2p iscell==1 in addition to shape prior.")
    parser.add_argument("--overlay-max-rois", type=int, default=2500)
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
    args.prior_path = args.prior_path.expanduser().resolve() if args.prior_path else default_prior_path(output_root).resolve()

    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1
    if not args.prior_path.exists():
        LOGGER.error("Manual ROI prior does not exist: %s", args.prior_path)
        return 1

    prior = load_json(args.prior_path)
    trials = discover_trials(input_root)
    summary = RunSummary(found=len(trials))
    rows: list[dict] = []

    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Prior path : %s", args.prior_path)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d suite2p trial(s).", len(trials))

    for trial in trials:
        out_dir = trial_output_dir(out_root, trial)
        if required_outputs_done(out_dir, trial.trial_id) and args.action == "skip":
            summary.skipped += 1
            LOGGER.info("[skip] %s", trial.trial_id)
            continue
        if args.dry_run:
            summary.processed += 1
            LOGGER.info("[dry-run] Would filter ROI quality for %s -> %s", trial.plane0_dir, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)

        try:
            status, row = process_trial(trial, out_dir, prior, args)
            rows.append(row)
            if status == "processed":
                summary.processed += 1
                LOGGER.info("[ok] %s: accepted=%s/%s", trial.trial_id, row.get("n_accepted"), row.get("n_suite2p_roi"))
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
        write_csv(out_root / "roi_quality_summary.csv", rows)

    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
