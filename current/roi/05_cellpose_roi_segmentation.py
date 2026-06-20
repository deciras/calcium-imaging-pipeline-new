#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellpose ROI candidate segmentation before manual curation.

Input defaults to DATA_ROOT/04_spatial_highpass when available, falling back to
DATA_ROOT/03_motion_correct. Outputs are separate from suite2p and are only a
candidate layer for the manual GUI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tf


LOGGER = logging.getLogger("cellpose_roi_segmentation")
STEP_NAME = "05_cellpose_roi_segmentation"


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    movie_path: Path
    movie_source: str


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


def candidate_input_roots(output_root: Path) -> tuple[Path, ...]:
    return (output_root / "04_spatial_highpass", output_root / "03_motion_correct")


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def trial_output_dir(step_root: Path, trial: TrialInput) -> Path:
    if str(trial.rel_parent) in ("", "."):
        return step_root / trial.trial_id
    return step_root / trial.rel_parent / trial.trial_id


def choose_movie(input_dir: Path) -> tuple[Path, str] | None:
    choices = (
        ("*_spatial_highpass_movie.tif", "spatial_highpass"),
        ("*_spatial_highpass_movie.tiff", "spatial_highpass"),
        ("*_corrected_movie.tif", "corrected"),
        ("*_corrected_movie.tiff", "corrected"),
    )
    for pattern, source in choices:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0], source
    return None


def discover_trials(output_root: Path) -> list[TrialInput]:
    seen: set[tuple[Path, str]] = set()
    trials: list[TrialInput] = []
    for input_root in candidate_input_roots(output_root):
        if not input_root.exists():
            continue
        trial_dirs: set[Path] = set()
        for pattern in ("*_spatial_highpass_movie.tif", "*_spatial_highpass_movie.tiff", "*_corrected_movie.tif", "*_corrected_movie.tiff"):
            for path in input_root.rglob(pattern):
                trial_dirs.add(path.parent)
        for input_dir in sorted(trial_dirs):
            chosen = choose_movie(input_dir)
            if chosen is None:
                continue
            movie_path, source = chosen
            trial_id = input_dir.name
            rel_parent = input_dir.parent.relative_to(input_root)
            key = (rel_parent, trial_id)
            if key in seen:
                continue
            seen.add(key)
            trials.append(TrialInput(trial_id, rel_parent, input_dir, movie_path, source))
    return trials


def required_outputs_done(out_dir: Path) -> bool:
    return all(
        (out_dir / name).exists() and (out_dir / name).stat().st_size > 0
        for name in ("cellpose_masks.npy", "cellpose_stat.npy", "cellpose_summary.json")
    )


def projection_image(movie_path: Path, max_frames: int) -> np.ndarray:
    arr = tf.memmap(movie_path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        image = arr.astype(np.float32, copy=False)
    else:
        movie = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        if max_frames > 0 and movie.shape[0] > max_frames:
            indices = np.linspace(0, movie.shape[0] - 1, max_frames).astype(int)
            movie = movie[indices]
        image = np.mean(movie.astype(np.float32, copy=False), axis=0)
    finite = image[np.isfinite(image)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        image = (image - lo) / max(float(hi - lo), 1e-6)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def boundary_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy import ndimage as ndi

        inner = ndi.binary_erosion(mask, iterations=1, border_value=0)
        boundary = mask & ~inner
    except Exception:
        boundary = mask
    return np.nonzero(boundary)


def stat_from_masks(masks: np.ndarray) -> np.ndarray:
    stats: list[dict] = []
    for label in sorted(int(v) for v in np.unique(masks) if int(v) > 0):
        roi_mask = masks == label
        ypix, xpix = np.nonzero(roi_mask)
        if xpix.size == 0:
            continue
        by, bx = boundary_pixels(roi_mask)
        stats.append(
            {
                "ypix": ypix.astype(np.int32),
                "xpix": xpix.astype(np.int32),
                "boundary_ypix": by.astype(np.int32),
                "boundary_xpix": bx.astype(np.int32),
                "npix": int(xpix.size),
                "med": np.array([float(np.mean(ypix)), float(np.mean(xpix))], dtype=np.float32),
                "cellpose_label": int(label),
            }
        )
    return np.array(stats, dtype=object)


def run_cellpose(image: np.ndarray, diameter: float | None, model_type: str, gpu: bool) -> np.ndarray:
    try:
        from cellpose import models
    except Exception as exc:
        raise RuntimeError("Cellpose is not installed in this environment. Install cellpose or run this step in a cellpose env.") from exc

    model = models.CellposeModel(gpu=gpu, model_type=model_type)
    result = model.eval(image, channels=[0, 0], diameter=diameter)
    masks = result[0] if isinstance(result, tuple) else result
    return np.asarray(masks, dtype=np.int32)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> str:
    if required_outputs_done(out_dir) and args.action == "skip":
        return "skipped"
    if out_dir.exists() and args.action == "overwrite":
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = projection_image(trial.movie_path, max_frames=args.max_projection_frames)
    masks = run_cellpose(
        image=image,
        diameter=args.diameter if args.diameter > 0 else None,
        model_type=args.model_type,
        gpu=args.gpu,
    )
    stat = stat_from_masks(masks)
    np.save(out_dir / "cellpose_masks.npy", masks)
    np.save(out_dir / "cellpose_stat.npy", stat)
    summary = {
        "trial_id": trial.trial_id,
        "rel_parent": str(trial.rel_parent),
        "movie_path": str(trial.movie_path),
        "movie_source": trial.movie_source,
        "model_type": args.model_type,
        "diameter": args.diameter,
        "gpu": bool(args.gpu),
        "n_roi": int(len(stat)),
    }
    (out_dir / "cellpose_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cellpose ROI candidate segmentation.")
    parser.add_argument("--data-root", type=Path, help="Pipeline data root.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trial-id", help="Only process one trial ID.")
    parser.add_argument("--model-type", default="cyto3")
    parser.add_argument("--diameter", type=float, default=0.0, help="Cellpose diameter in pixels; <=0 lets Cellpose estimate.")
    parser.add_argument("--max-projection-frames", type=int, default=200)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--require-cellpose", action="store_true", help="Fail instead of soft-skipping when cellpose is not installed.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    data_root = args.data_root.expanduser().resolve() if args.data_root else Path.cwd()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root)
    if importlib.util.find_spec("cellpose") is None:
        message = "Cellpose is not installed in this environment; skipping Cellpose ROI candidate segmentation."
        if args.require_cellpose:
            LOGGER.error(message)
            return 1
        LOGGER.warning(message)
        return 0
    step_root = step_output_root(output_root)
    trials = discover_trials(output_root)
    if args.trial_id:
        wanted = {item.strip() for item in args.trial_id.split(",") if item.strip()}
        trials = [trial for trial in trials if trial.trial_id in wanted]
    summary = RunSummary(found=len(trials))
    LOGGER.info("Found %d trial(s) for Cellpose.", len(trials))
    for trial in trials:
        out_dir = trial_output_dir(step_root, trial)
        if args.dry_run:
            LOGGER.info("[dry-run] Would segment %s -> %s", trial.movie_path, out_dir)
            continue
        try:
            status = process_trial(trial, out_dir, args)
            if status == "skipped":
                summary.skipped += 1
                LOGGER.info("[skip] %s", trial.trial_id)
            else:
                summary.processed += 1
                LOGGER.info("[ok] %s", trial.trial_id)
        except Exception as exc:
            summary.failed += 1
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    LOGGER.info(
        "Summary: found=%d processed=%d skipped=%d failed=%d",
        summary.found,
        summary.processed,
        summary.skipped,
        summary.failed,
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
