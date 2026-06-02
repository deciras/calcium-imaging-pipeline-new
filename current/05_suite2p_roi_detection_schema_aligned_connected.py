#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative suite2p ROI detection.

Default layout:
  input : DATA_ROOT/03_motion_correct/
  output         : DATA_ROOT/05_suite2p_roi_detection/

The input movies are not modified. suite2p outputs are written into this step's
own output folder, with one folder per trial.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from tqdm import tqdm


LOGGER = logging.getLogger("suite2p_roi_detection")

RECOMMENDED_S2P_VERSION = "0.14.4"
STEP_NAME = "05_suite2p_roi_detection"

STEP_OUTPUT_PATTERNS = (
    "suite2p",
    "benchmark",
    "suite2p_run.log",
    "suite2p_trial_summary.json",
    "*_metadata.json",
    "*_brightness_trace.csv",
    "*_stim_events.csv",
    "*_stim_map.csv",
    "*_stim_pulse_events.csv",
    "*_stim_trace.png",
    "*_stim_trace.pdf",
    "*_stim_schematic.png",
    "*_stim_schematic.pdf",
    "*_stim_pulse_trace.png",
    "*_stim_pulse_trace.pdf",
    "roi_overlay.pdf",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    movie_path: Path
    movie_source: str
    metadata_path: Path | None


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
    return output_root / "03_motion_correct"


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


def choose_movie(input_dir: Path) -> tuple[Path, str] | None:
    choices = (
        ("*_spatial_highpass_movie.tif", "spatial_highpass"),
        ("*_spatial_highpass_movie.tiff", "spatial_highpass"),
        ("*_brightness_corrected_movie.tif", "brightness_corrected"),
        ("*_brightness_corrected_movie.tiff", "brightness_corrected"),
        ("*_corrected_movie.tif", "corrected"),
        ("*_corrected_movie.tiff", "corrected"),
    )
    for pattern, source in choices:
        matches = sorted(input_dir.glob(pattern))
        matches = [path for path in matches if "_brightness_" not in path.name or source != "corrected"]
        if matches:
            return matches[0], source
    return None


def discover_trials(input_root: Path) -> list[TrialInput]:
    trial_dirs: set[Path] = set()
    for pattern in (
        "*_spatial_highpass_movie.tif",
        "*_spatial_highpass_movie.tiff",
        "*_brightness_corrected_movie.tif",
        "*_brightness_corrected_movie.tiff",
        "*_corrected_movie.tif",
        "*_corrected_movie.tiff",
    ):
        for path in input_root.rglob(pattern):
            trial_dirs.add(path.parent)

    trials: list[TrialInput] = []
    for input_dir in sorted(trial_dirs):
        chosen = choose_movie(input_dir)
        if chosen is None:
            continue
        movie_path, source = chosen
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        metadata_path = find_first_existing(input_dir, ("*_metadata.json",))
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                movie_path=movie_path,
                movie_source=source,
                metadata_path=metadata_path,
            )
        )
    return trials


def required_outputs_done(out_dir: Path) -> bool:
    plane0 = out_dir / "suite2p" / "plane0"
    required = (plane0 / "stat.npy", plane0 / "ops.npy", plane0 / "F.npy", plane0 / "Fneu.npy", plane0 / "iscell.npy")
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
                LOGGER.info("Removed old step-05 folder: %s", path)
            elif path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                LOGGER.info("Removed old step-05 file: %s", path)
    return removed


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_sidecar_outputs(trial: TrialInput, out_dir: Path) -> None:
    suffixes = (
        "_metadata.json",
        "_stim_events.csv",
        "_stim_map.csv",
        "_stim_pulse_events.csv",
    )
    for suffix in suffixes:
        copy_if_exists(trial.input_dir / f"{trial.trial_id}{suffix}", out_dir / f"{trial.trial_id}{suffix}")


def set_low_level_thread_env(num_threads: int) -> None:
    value = str(max(1, int(num_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", value)
    os.environ.setdefault("MKL_NUM_THREADS", value)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", value)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", value)


def check_suite2p_version(strict: bool) -> str:
    try:
        version = importlib.metadata.version("suite2p")
    except Exception:
        version = "unknown"

    if version != RECOMMENDED_S2P_VERSION:
        message = (
            f"suite2p version mismatch: installed={version}, "
            f"recommended={RECOMMENDED_S2P_VERSION}. Version 0.14.5 produced abnormal ROI behavior on this dataset."
        )
        if strict:
            raise RuntimeError(message)
        LOGGER.warning(message)
    return version


def sanity_check_tiff_is_time_series_2d(tif_path: Path, min_frames: int) -> tuple[bool, str]:
    try:
        with tf.TiffFile(tif_path) as tif:
            n_pages = len(tif.pages)
            series = tif.series[0]
            shape = getattr(series, "shape", None)
            ndim = len(shape) if shape is not None else None
            if shape is not None and len(shape) == 3 and int(shape[0]) >= min_frames:
                return True, f"OK: shape={shape}"
            if n_pages >= min_frames and shape is not None and len(shape) == 2:
                return True, f"OK: multipage 2D pages={n_pages}, frame={shape}"
            return False, f"Too few frames or unsupported shape: pages={n_pages}, shape={shape}, ndim={ndim}"
    except Exception as exc:
        return False, f"Failed to read TIFF: {exc}"


def get_trial_metadata(metadata_path: Path | None, cell_diameter_um: float, min_diameter_px: int, diameter_scale: float) -> dict:
    meta = {
        "nplanes": 1,
        "fs": 1.0,
        "nchannels": 1,
        "pixel_size_um": None,
        "diameter_px": max(min_diameter_px, int(round(cell_diameter_um))),
        "json_found": False,
    }
    if metadata_path is None or not metadata_path.exists():
        return meta

    meta["json_found"] = True
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        fps = data.get("temporal_calibration", {}).get("fps", 1.0)
        meta["fs"] = float(fps) if fps is not None and float(fps) > 0 else 1.0

        physical = data.get("physical_size", {})
        dims = data.get("dimensions", {})
        pixel_size_um = physical.get("pixel_size_um")
        if pixel_size_um is None:
            fov_width_um = physical.get("fov_width_um", physical.get("width"))
            width_px = dims.get("width_pixel")
            if fov_width_um is not None and width_px not in (None, 0):
                pixel_size_um = float(fov_width_um) / float(width_px)

        if pixel_size_um is not None and float(pixel_size_um) > 0:
            pixel_size_um = float(pixel_size_um)
            meta["pixel_size_um"] = pixel_size_um
            diameter_px = int(round((cell_diameter_um / pixel_size_um) * diameter_scale))
            meta["diameter_px"] = max(min_diameter_px, diameter_px)
    except Exception:
        pass

    return meta


def build_ops_template(args: dict, diameter_px: int) -> dict:
    return {
        "do_registration": False,
        "nonrigid": True,
        "block_size": [128, 128],
        "snr_thresh": args["snr_thresh"],
        "maxregshiftNR": 5,
        "nimg_init": 300,
        "maxregshift": 0.1,
        "smooth_sigma": 1.15,
        "smooth_sigma_time": 0,
        "roidetect": True,
        "sparse_mode": False,
        "diameter": [int(diameter_px), int(diameter_px)],
        "connected": True,
        "threshold_scaling": args["threshold_scaling"],
        "max_overlap": args["max_overlap"],
        "max_iterations": args["max_iterations"],
        "high_pass": args["high_pass"],
        "spatial_hp_detect": args["spatial_hp_detect"],
        "allow_overlap": False,
        "neuropil_extract": True,
        "inner_neuropil_radius": args["inner_neuropil_radius"],
        "min_neuropil_pixels": args["min_neuropil_pixels"],
        "tau": args["tau"],
        "batch_size": int(args["batch_size"]),
        "nthreads": int(args["suite2p_threads"]),
        "combined": True,
        "reg_tif": False,
        "delete_bin": 0,
        "move_bin": False,
        "keep_movie_raw": False,
    }


def load_iscell(plane0_dir: Path, n_stat: int) -> tuple[np.ndarray, np.ndarray]:
    path = plane0_dir / "iscell.npy"
    if not path.exists():
        return np.ones(n_stat, dtype=bool), np.full(n_stat, np.nan, dtype=float)
    iscell = np.load(path, allow_pickle=True)
    if iscell.ndim != 2 or iscell.shape[0] != n_stat:
        return np.ones(n_stat, dtype=bool), np.full(n_stat, np.nan, dtype=float)
    flags = iscell[:, 0].astype(bool)
    prob = iscell[:, 1].astype(float) if iscell.shape[1] >= 2 else np.full(n_stat, np.nan, dtype=float)
    return flags, prob


def build_roi_maps(stat, ly: int, lx: int) -> tuple[np.ndarray, np.ndarray]:
    roi_mask = np.zeros((ly, lx), dtype=np.uint8)
    roi_label_map = np.zeros((ly, lx), dtype=np.int32)
    for index, roi in enumerate(stat, start=1):
        xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
        valid = (xpix >= 0) & (xpix < lx) & (ypix >= 0) & (ypix < ly)
        xpix = xpix[valid]
        ypix = ypix[valid]
        if len(xpix) == 0:
            continue
        roi_mask[ypix, xpix] = 255
        roi_label_map[ypix, xpix] = index
    return roi_mask, roi_label_map


def summarize_rois(stat, iscell_flag, prob, pixel_size_um, trial_id: str) -> list[dict]:
    rows = []
    for index, roi in enumerate(stat):
        xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
        area_px = int(len(xpix))
        row = {
            "trial": trial_id,
            "roi_id": index + 1,
            "suite2p_index": index,
            "iscell": int(bool(iscell_flag[index])) if index < len(iscell_flag) else 1,
            "iscell_prob": float(prob[index]) if index < len(prob) and np.isfinite(prob[index]) else np.nan,
            "area_px": area_px,
            "centroid_x": float(np.mean(xpix)) if area_px else np.nan,
            "centroid_y": float(np.mean(ypix)) if area_px else np.nan,
            "area_um2": float(area_px * (pixel_size_um**2)) if pixel_size_um is not None else np.nan,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_roi_overlay(mean_img, stat, iscell_flag, output_path: Path, trial_id: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_img, cmap="gray")
    for index, roi in enumerate(stat):
        xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
        if len(xpix) == 0:
            continue
        color = "lime" if iscell_flag[index] else "red"
        ax.scatter(xpix, ypix, s=0.5, c=color, alpha=0.7)
    ax.set_title(f"Suite2p ROI overlay - {trial_id}")
    ax.axis("off")
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def export_suite2p_summary(out_dir: Path, trial_id: str, pixel_size_um: float | None, dpi: int) -> tuple[int, int]:
    plane0 = out_dir / "suite2p" / "plane0"
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
    ly = int(ops.get("Ly"))
    lx = int(ops.get("Lx"))
    mean_img = np.asarray(ops.get("meanImg", np.zeros((ly, lx))), dtype=np.float32)

    iscell_flag, prob = load_iscell(plane0, len(stat))
    roi_mask, roi_label_map = build_roi_maps(stat, ly=ly, lx=lx)
    roi_summary = summarize_rois(stat, iscell_flag, prob, pixel_size_um, trial_id)

    bench_dir = out_dir / "benchmark" / "suite2p"
    bench_dir.mkdir(parents=True, exist_ok=True)
    tf.imwrite(bench_dir / "roi_mask.tif", roi_mask, imagej=True)
    tf.imwrite(bench_dir / "roi_label_map.tif", roi_label_map.astype(np.uint16), imagej=True)
    write_csv(bench_dir / "roi_summary.csv", roi_summary)
    plot_roi_overlay(mean_img, stat, iscell_flag, bench_dir / "roi_overlay.png", trial_id, dpi=dpi)
    return int(len(stat)), int(np.sum(iscell_flag))


def run_one_trial(payload: dict) -> dict:
    set_low_level_thread_env(payload["num_threads"])

    import suite2p
    from suite2p.run_s2p import default_ops, run_s2p

    trial_id = payload["trial_id"]
    movie_path = Path(payload["movie_path"])
    input_dir = Path(payload["input_dir"])
    out_dir = Path(payload["out_dir"])
    metadata_path = Path(payload["metadata_path"]) if payload["metadata_path"] else None
    log_path = out_dir / "suite2p_run.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    try:
        ok, info = sanity_check_tiff_is_time_series_2d(movie_path, payload["min_frames"])
        log(f"movie_path: {movie_path}")
        log(f"input_dir: {input_dir}")
        log(f"out_dir: {out_dir}")
        log(f"sanity_check: {info}")
        if not ok:
            return {"trial": trial_id, "status": "skipped", "message": info, "n_rois": 0, "n_iscell": 0}

        meta = get_trial_metadata(
            metadata_path=metadata_path,
            cell_diameter_um=payload["cell_diameter_um"],
            min_diameter_px=payload["min_diameter_px"],
            diameter_scale=payload["diameter_scale"],
        )
        ops = default_ops()
        ops.update(build_ops_template(payload, diameter_px=meta["diameter_px"]))
        ops.update({"nplanes": meta["nplanes"], "fs": meta["fs"], "nchannels": meta["nchannels"]})

        db = {
            "data_path": [str(input_dir)],
            "tiff_list": [movie_path.name],
            "save_path0": str(out_dir),
            "nchannels": meta["nchannels"],
            "nplanes": meta["nplanes"],
        }

        log(f"suite2p_version: {getattr(suite2p, '__version__', 'unknown')}")
        log(f"movie_source: {payload['movie_source']}")
        log(f"diameter_px: {meta['diameter_px']}")
        log(f"threshold_scaling: {payload['threshold_scaling']}")
        log(f"snr_thresh: {payload['snr_thresh']}")
        log("run_s2p starting")
        run_s2p(ops=ops, db=db)
        log("run_s2p finished")

        n_rois, n_iscell = export_suite2p_summary(
            out_dir=out_dir,
            trial_id=trial_id,
            pixel_size_um=meta["pixel_size_um"],
            dpi=payload["overlay_dpi"],
        )
        with (out_dir / "suite2p_trial_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "trial": trial_id,
                    "movie_path": str(movie_path),
                    "movie_source": payload["movie_source"],
                    "metadata_path": str(metadata_path) if metadata_path else None,
                    "diameter_px": meta["diameter_px"],
                    "n_rois": n_rois,
                    "n_iscell": n_iscell,
                },
                handle,
                indent=2,
            )
        return {"trial": trial_id, "status": "ok", "message": "success", "n_rois": n_rois, "n_iscell": n_iscell}
    except Exception as exc:
        log("ERROR:")
        log(str(exc))
        log(traceback.format_exc())
        return {"trial": trial_id, "status": "failed", "message": str(exc), "n_rois": 0, "n_iscell": 0}


def payload_for_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> dict:
    return {
        "trial_id": trial.trial_id,
        "input_dir": str(trial.input_dir),
        "movie_path": str(trial.movie_path),
        "movie_source": trial.movie_source,
        "metadata_path": str(trial.metadata_path) if trial.metadata_path else None,
        "out_dir": str(out_dir),
        "min_frames": args.min_frames,
        "num_threads": args.num_threads,
        "cell_diameter_um": args.cell_diameter_um,
        "min_diameter_px": args.min_diameter_px,
        "diameter_scale": args.diameter_scale,
        "threshold_scaling": args.threshold_scaling,
        "max_overlap": args.max_overlap,
        "snr_thresh": args.snr_thresh,
        "high_pass": args.high_pass,
        "spatial_hp_detect": args.spatial_hp_detect,
        "max_iterations": args.max_iterations,
        "inner_neuropil_radius": args.inner_neuropil_radius,
        "min_neuropil_pixels": args.min_neuropil_pixels,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "suite2p_threads": args.suite2p_threads,
        "overlay_dpi": args.overlay_dpi,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run conservative suite2p ROI detection.")
    parser.add_argument("--data-root", type=Path, required=True, help="Original data root.")
    parser.add_argument("--input-root", type=Path, help="Movie root. Default: 03_motion_correct.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip", help="Existing-output behavior.")
    parser.add_argument("--dry-run", action="store_true", help="Print work plan without running suite2p.")
    parser.add_argument("--strict-suite2p-version", action="store_true", help="Fail if suite2p is not version 0.14.4.")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of trials to run in parallel.")
    parser.add_argument("--num-threads", type=int, default=1, help="Low-level BLAS/OpenMP thread limit per worker.")
    parser.add_argument("--suite2p-threads", type=int, default=8, help="suite2p nthreads setting.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--cell-diameter-um", type=float, default=5.0)
    parser.add_argument("--min-diameter-px", type=int, default=4)
    parser.add_argument("--diameter-scale", type=float, default=1.2)
    parser.add_argument("--threshold-scaling", type=float, default=1.2)
    parser.add_argument("--max-overlap", type=float, default=0.5)
    parser.add_argument("--snr-thresh", type=float, default=1.5)
    parser.add_argument("--high-pass", type=int, default=40)
    parser.add_argument("--spatial-hp-detect", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--inner-neuropil-radius", type=int, default=2)
    parser.add_argument("--min-neuropil-pixels", type=int, default=350)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--overlay-dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    check_suite2p_version(strict=args.strict_suite2p_version)

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
    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d trial(s) for suite2p.", len(trials))

    payloads = []
    for trial in trials:
        out_dir = trial_output_dir(out_root, trial)
        if required_outputs_done(out_dir) and args.action == "skip":
            summary.skipped += 1
            LOGGER.info("[skip] %s", trial.trial_id)
            continue
        if args.dry_run:
            summary.processed += 1
            LOGGER.info("[dry-run] Would run suite2p on %s -> %s", trial.movie_path, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        copy_sidecar_outputs(trial, out_dir)
        payloads.append(payload_for_trial(trial, out_dir, args))

    results: list[dict] = []
    if payloads:
        if args.n_workers <= 1:
            for payload in tqdm(payloads, desc="suite2p"):
                results.append(run_one_trial(payload))
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=context) as executor:
                futures = [executor.submit(run_one_trial, payload) for payload in payloads]
                for future in tqdm(as_completed(futures), total=len(futures), desc="suite2p"):
                    results.append(future.result())

        for result in results:
            if result["status"] == "ok":
                summary.processed += 1
                LOGGER.info("[ok] %s: n_rois=%s n_iscell=%s", result["trial"], result["n_rois"], result["n_iscell"])
            elif result["status"] == "skipped":
                summary.skipped += 1
                LOGGER.info("[skip] %s: %s", result["trial"], result["message"])
            else:
                summary.failed += 1
                LOGGER.error("[failed] %s: %s", result["trial"], result["message"])

        out_root.mkdir(parents=True, exist_ok=True)
        write_csv(out_root / "suite2p_summary.csv", results)

    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
