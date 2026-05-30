#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find independent ROI candidates that suite2p may have missed.

This step does not use suite2p's cell/non-cell classification. It uses suite2p
outputs only as a convenient source of registered images, existing ROI masks,
and optionally the registered movie data.bin for candidate trace summaries.
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


LOGGER = logging.getLogger("independent_roi_candidates")

STEP_NAME = "05d_independent_roi_candidates"
STEP_OUTPUT_PATTERNS = (
    "*_independent_roi_candidates.csv",
    "*_independent_roi_candidates_summary.json",
    "*_independent_roi_candidate_label_map.tif",
    "*_independent_roi_candidate_overlay.png",
    "*_independent_roi_candidate_overlay.pdf",
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
    data_bin_path: Path
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
                data_bin_path=plane0_dir / "data.bin",
                metadata_path=find_first_existing(input_dir, ("*_metadata.json",)),
                stim_events_path=find_first_existing(input_dir, ("*_stim_events.csv",)),
                stim_map_path=find_first_existing(input_dir, ("*_stim_map.csv",)),
                stim_pulse_events_path=find_first_existing(input_dir, ("*_stim_pulse_events.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_independent_roi_candidates.csv",
        out_dir / f"{trial_id}_independent_roi_candidates_summary.json",
        out_dir / f"{trial_id}_independent_roi_candidate_overlay.png",
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
                LOGGER.info("Removed old step-05d folder: %s", path)
            elif path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                LOGGER.info("Removed old step-05d file: %s", path)
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


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ops(path: Path) -> dict:
    value = np.load(path, allow_pickle=True)
    item = value.item()
    return item if isinstance(item, dict) else {}


def choose_image(ops: dict, source: str) -> np.ndarray:
    for key in (source, "max_proj", "Vcorr", "sdmov", "meanImg"):
        image = np.asarray(ops.get(key, []), dtype=np.float32)
        if image.ndim == 2 and image.size:
            return image
    raise ValueError("No usable 2D image found in ops.npy")


def robust_normalize(image: np.ndarray, background_sigma: float) -> np.ndarray:
    from scipy import ndimage as ndi

    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)
    fill = float(np.nanmedian(image[finite]))
    clean = np.where(finite, image, fill)
    if background_sigma > 0:
        background = ndi.gaussian_filter(clean, sigma=float(background_sigma))
        clean = clean - background
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = max(1.4826 * mad, 1e-6)
    return (clean - median) / scale


def prior_radius_px(prior: dict, fallback_radius: float) -> float:
    try:
        diameter = float(prior["metrics"]["equiv_diameter_px"]["median"])
        if np.isfinite(diameter) and diameter > 0:
            return max(2.0, diameter / 2.0)
    except Exception:
        pass
    return float(fallback_radius)


def circular_mask(center_y: float, center_x: float, radius: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    y0 = max(0, int(math.floor(center_y - radius)))
    y1 = min(height, int(math.ceil(center_y + radius + 1)))
    x0 = max(0, int(math.floor(center_x - radius)))
    x1 = min(width, int(math.ceil(center_x + radius + 1)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    keep = ((yy + 0.5 - center_y) ** 2 + (xx + 0.5 - center_x) ** 2) <= radius**2
    return yy[keep], xx[keep]


def suite2p_existing_mask(stat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for roi in stat:
        if not isinstance(roi, dict):
            continue
        ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
        xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
        valid = (ypix >= 0) & (ypix < shape[0]) & (xpix >= 0) & (xpix < shape[1])
        mask[ypix[valid], xpix[valid]] = True
    return mask


def trace_summary(trace: np.ndarray) -> dict:
    finite = np.asarray(trace[np.isfinite(trace)], dtype=float)
    if finite.size == 0:
        return {
            "trace_n_frames": 0,
            "trace_mean": np.nan,
            "trace_cv": np.nan,
            "trace_robust_snr": np.nan,
            "trace_dff_p95": np.nan,
        }
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    p10 = float(np.percentile(finite, 10))
    p95 = float(np.percentile(finite, 95))
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "trace_n_frames": int(finite.size),
        "trace_mean": mean,
        "trace_cv": float(std / max(abs(mean), 1e-6)),
        "trace_robust_snr": float((p95 - median) / max(sigma, 1e-6)),
        "trace_dff_p95": float((p95 - p10) / max(abs(p10), 1e-6)),
    }


def extract_candidate_trace(data_bin: Path, ops: dict, ypix: np.ndarray, xpix: np.ndarray) -> dict:
    if not data_bin.exists() or ypix.size == 0:
        return trace_summary(np.asarray([], dtype=float))
    ly = int(ops.get("Ly", 0) or 0)
    lx = int(ops.get("Lx", 0) or 0)
    nframes = int(ops.get("nframes", 0) or 0)
    if ly <= 0 or lx <= 0 or nframes <= 0:
        return trace_summary(np.asarray([], dtype=float))
    try:
        movie = np.memmap(data_bin, mode="r", dtype=np.int16, shape=(nframes, ly, lx))
        trace = np.asarray(movie[:, ypix, xpix].mean(axis=1), dtype=float)
        return trace_summary(trace)
    except Exception as exc:
        LOGGER.warning("Could not extract candidate trace from %s: %s", data_bin, exc)
        return trace_summary(np.asarray([], dtype=float))


def detect_candidates(
    image_z: np.ndarray,
    raw_image: np.ndarray,
    existing_mask: np.ndarray,
    radius_px: float,
    args: argparse.Namespace,
    trial: TrialInput,
    ops: dict,
) -> tuple[list[dict], np.ndarray]:
    from skimage.feature import peak_local_max

    min_distance = max(2, int(round(radius_px * args.min_distance_radius_factor)))
    coordinates = peak_local_max(
        image_z,
        min_distance=min_distance,
        threshold_abs=float(args.peak_z_threshold),
        exclude_border=max(1, int(math.ceil(radius_px))),
        num_peaks=int(args.max_candidates),
    )

    rows: list[dict] = []
    label_map = np.zeros(raw_image.shape, dtype=np.uint16)
    for candidate_id, (y, x) in enumerate(coordinates, start=1):
        ypix, xpix = circular_mask(float(y), float(x), radius_px, raw_image.shape)
        if ypix.size < args.min_pixels:
            continue
        overlap_fraction = float(np.mean(existing_mask[ypix, xpix])) if ypix.size else 0.0
        novel = overlap_fraction <= args.max_suite2p_overlap
        local_values = raw_image[ypix, xpix]
        trace_features = extract_candidate_trace(trial.data_bin_path, ops, ypix, xpix) if args.extract_traces else {}
        row = {
            "trial_id": trial.trial_id,
            "candidate_id": candidate_id,
            "center_x": float(x),
            "center_y": float(y),
            "radius_px": float(radius_px),
            "area_px": int(ypix.size),
            "image_z": float(image_z[y, x]),
            "mean_image_intensity": float(np.mean(local_values)),
            "max_image_intensity": float(np.max(local_values)),
            "suite2p_overlap_fraction": overlap_fraction,
            "novel_candidate": int(bool(novel)),
            **trace_features,
        }
        rows.append(row)
        if novel:
            label_map[ypix, xpix] = min(candidate_id, np.iinfo(np.uint16).max)

    rows.sort(
        key=lambda row: (
            int(row["novel_candidate"]),
            float(row.get("trace_robust_snr", 0) or 0),
            float(row["image_z"]),
        ),
        reverse=True,
    )
    return rows, label_map


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_label_map(path: Path, label_map: np.ndarray) -> None:
    try:
        import tifffile as tf

        tf.imwrite(path, label_map.astype(np.uint16), imagej=True)
    except Exception as exc:
        LOGGER.warning("Could not write candidate label map %s: %s", path, exc)


def save_overlay(
    image: np.ndarray,
    existing_mask: np.ndarray,
    candidates: list[dict],
    output_path: Path,
    trial_id: str,
    dpi: int,
    max_plot_candidates: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(image[np.isfinite(image)], [1, 99])
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    yy, xx = np.nonzero(existing_mask)
    if yy.size:
        ax.scatter(xx, yy, s=0.05, c="tab:blue", alpha=0.12, label="suite2p ROI pixels")

    plotted = 0
    for row in candidates:
        if plotted >= max_plot_candidates:
            break
        if not int(row.get("novel_candidate", 0)):
            continue
        circle = Circle(
            (float(row["center_x"]), float(row["center_y"])),
            radius=float(row["radius_px"]),
            edgecolor="yellow",
            facecolor="none",
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(circle)
        plotted += 1
    ax.set_title(f"Independent ROI candidates - {trial_id}")
    ax.axis("off")
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, prior: dict, args: argparse.Namespace) -> tuple[str, dict]:
    missing = [path.name for path in (trial.stat_path, trial.ops_path) if not path.exists()]
    if missing:
        return "failed", {"trial_id": trial.trial_id, "status": "failed", "message": f"Missing inputs: {', '.join(missing)}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_outputs(trial, out_dir)
    ops = load_ops(trial.ops_path)
    stat = np.load(trial.stat_path, allow_pickle=True)
    image = choose_image(ops, args.image_source)
    image_z = robust_normalize(image, background_sigma=args.background_sigma)
    existing = suite2p_existing_mask(stat, image.shape)
    radius_px = prior_radius_px(prior, fallback_radius=args.radius_px)

    candidates, label_map = detect_candidates(
        image_z=image_z,
        raw_image=image,
        existing_mask=existing,
        radius_px=radius_px,
        args=args,
        trial=trial,
        ops=ops,
    )

    write_csv(out_dir / f"{trial.trial_id}_independent_roi_candidates.csv", candidates)
    save_label_map(out_dir / f"{trial.trial_id}_independent_roi_candidate_label_map.tif", label_map)
    save_overlay(
        image=image,
        existing_mask=existing,
        candidates=candidates,
        output_path=out_dir / f"{trial.trial_id}_independent_roi_candidate_overlay.png",
        trial_id=trial.trial_id,
        dpi=args.dpi,
        max_plot_candidates=args.overlay_max_candidates,
    )

    n_novel = int(sum(int(row.get("novel_candidate", 0)) for row in candidates))
    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "image_source": args.image_source,
        "n_suite2p_roi": int(len(stat)),
        "n_candidates": int(len(candidates)),
        "n_novel_candidates": n_novel,
        "radius_px": float(radius_px),
        "peak_z_threshold": float(args.peak_z_threshold),
        "max_suite2p_overlap": float(args.max_suite2p_overlap),
        "extract_traces": bool(args.extract_traces),
    }
    with (out_dir / f"{trial.trial_id}_independent_roi_candidates_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find image-based ROI candidates independent of suite2p iscell labels.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-05 suite2p root. Default: OUTPUT_ROOT/05_suite2p_roi_detection.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--prior-path", type=Path, help="manual_roi_prior.json. Default: OUTPUT_ROOT/05b_manual_roi_prior/manual_roi_prior.json.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--image-source", choices=("max_proj", "meanImg", "Vcorr", "sdmov"), default="max_proj")
    parser.add_argument("--radius-px", type=float, default=5.0)
    parser.add_argument("--background-sigma", type=float, default=18.0)
    parser.add_argument("--peak-z-threshold", type=float, default=4.0)
    parser.add_argument("--min-distance-radius-factor", type=float, default=1.2)
    parser.add_argument("--max-suite2p-overlap", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--min-pixels", type=int, default=12)
    parser.add_argument("--extract-traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-max-candidates", type=int, default=300)
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
    prior_path = args.prior_path.expanduser().resolve() if args.prior_path else default_prior_path(output_root).resolve()
    prior = load_json(prior_path)

    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1

    trials = discover_trials(input_root)
    summary = RunSummary(found=len(trials))
    rows: list[dict] = []

    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Prior path : %s", prior_path if prior_path.exists() else "not found; using fallback radius")
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
            LOGGER.info("[dry-run] Would find independent ROI candidates for %s -> %s", trial.plane0_dir, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            status, row = process_trial(trial, out_dir, prior, args)
            rows.append(row)
            if status == "processed":
                summary.processed += 1
                LOGGER.info("[ok] %s: novel_candidates=%s", trial.trial_id, row.get("n_novel_candidates"))
            else:
                summary.failed += 1
                LOGGER.error("[failed] %s: %s", trial.trial_id, row.get("message"))
        except Exception as exc:
            summary.failed += 1
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        write_csv(out_root / "independent_roi_candidates_summary.csv", rows)

    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
