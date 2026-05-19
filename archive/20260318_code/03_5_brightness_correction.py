
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_5_brightness_correction.py

Global illumination correction for calcium imaging movies.

Default behaviour
-----------------
- Use divisive normalization by frame mean (divide_mean)
- Recursively discover all trial folders that contain *_corrected_movie.tif
- Export a canonical movie:
      *_brightness_corrected_movie.tif

Optional
--------
- ENABLE_DEBUG_MODES = True will also export:
      *_brightness_corrected_subtract_mean_movie.tif
      *_brightness_corrected_subtract_median_movie.tif
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "overwrite"   # "skip" / "overwrite"

ENABLE_DEBUG_MODES = False
EPS = 1e-6
REFERENCE_MODE = "prestim_mean"   # "prestim_mean" / "global_mean"


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_outputs_for_trial(trial_dir: Path, prefix: str):
    targets = [
        trial_dir / f"{prefix}_brightness_corrected_movie.tif",
        trial_dir / f"{prefix}_global_brightness_trace.csv",
        trial_dir / f"{prefix}_global_brightness_trace.png",
        trial_dir / f"{prefix}_brightness_corrected_subtract_mean_movie.tif",
        trial_dir / f"{prefix}_brightness_corrected_subtract_median_movie.tif",
    ]
    bc_dir = trial_dir / "brightness_correction"
    for p in targets:
        if p.exists():
            p.unlink()
    if bc_dir.exists():
        for p in bc_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()


def discover_trials(root_dir: Path):
    trials = []
    for movie_path in sorted(root_dir.rglob("*_corrected_movie.tif"), key=lambda p: natural_key(str(p))):
        trial_dir = movie_path.parent
        prefix = movie_path.name.replace("_corrected_movie.tif", "")
        meta_path = trial_dir / f"{prefix}_metadata.json"
        stim_path = trial_dir / f"{prefix}_stim_events.csv"
        trials.append({
            "trial_dir": trial_dir,
            "prefix": prefix,
            "movie_path": movie_path,
            "meta_path": meta_path if meta_path.exists() else None,
            "stim_path": stim_path if stim_path.exists() else None,
        })
    return trials


def load_fps(meta_path):
    if meta_path is None or not meta_path.exists():
        return 1.0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fps = meta.get("temporal_calibration", {}).get("fps", 1.0)
        return float(fps) if fps is not None else 1.0
    except Exception:
        return 1.0


def load_first_stim_start_sec(stim_path):
    if stim_path is None or not stim_path.exists():
        return None
    try:
        df = pd.read_csv(stim_path)
        if "start_time_sec" not in df.columns:
            return None
        vals = pd.to_numeric(df["start_time_sec"], errors="coerce").dropna()
        if len(vals) == 0:
            return None
        return float(vals.min())
    except Exception:
        return None


def compute_frame_mean(movie):
    return movie.reshape(movie.shape[0], -1).mean(axis=1)


def compute_frame_median(movie):
    return np.median(movie.reshape(movie.shape[0], -1), axis=1)


def divide_mean_correction(movie, frame_mean, ref):
    scale = (frame_mean / ref)[:, None, None]
    return movie / (scale + EPS)


def subtract_mean_correction(movie, frame_mean, ref):
    offset = (frame_mean - ref)[:, None, None]
    return movie - offset


def subtract_median_correction(movie, frame_median, ref):
    offset = (frame_median - ref)[:, None, None]
    return movie - offset


def choose_reference(frame_mean, fps, first_stim_start_sec):
    if REFERENCE_MODE == "prestim_mean" and first_stim_start_sec is not None and fps > 0:
        n_base = int(np.floor(first_stim_start_sec * fps))
        if n_base >= 3:
            return float(np.mean(frame_mean[:n_base]))
    return float(np.mean(frame_mean))


def save_preview(trial_dir, movie, corrected_divide, corrected_sub_mean, corrected_sub_median):
    out_dir = trial_dir / "brightness_correction"
    safe_mkdir(out_dir)

    mean_raw = np.mean(movie, axis=0)
    mean_div = np.mean(corrected_divide, axis=0)
    mean_sub_mean = np.mean(corrected_sub_mean, axis=0)
    mean_sub_med = np.mean(corrected_sub_median, axis=0)

    imgs = [mean_raw, mean_div, mean_sub_mean, mean_sub_med]
    titles = ["raw mean", "divide_mean", "subtract_mean", "subtract_median"]

    vmin = np.percentile(mean_div, 1)
    vmax = np.percentile(mean_div, 99.5)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, img, title in zip(axes.flat, imgs, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "preview_before_after.png", dpi=150)
    plt.close(fig)

    summary = {
        "reference_mode": REFERENCE_MODE,
        "default_output_mode": "divide_mean",
    }
    with open(out_dir / "correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def process_trial(info):
    trial_dir = info["trial_dir"]
    prefix = info["prefix"]
    movie_path = info["movie_path"]
    meta_path = info["meta_path"]
    stim_path = info["stim_path"]

    if EXISTING_MODE == "skip":
        out_movie = trial_dir / f"{prefix}_brightness_corrected_movie.tif"
        if out_movie.exists():
            print(f"[SKIP] {trial_dir.name}: brightness corrected movie already exists")
            return True
    elif EXISTING_MODE == "overwrite":
        clear_outputs_for_trial(trial_dir, prefix)

    print(f"\\n=== Trial: {trial_dir.name} ===")
    print(f"    movie: {movie_path.name}")

    movie = tiff.imread(movie_path).astype(np.float32)
    T = movie.shape[0]

    fps = load_fps(meta_path)
    first_stim_start_sec = load_first_stim_start_sec(stim_path)

    frame_mean = compute_frame_mean(movie)
    frame_median = compute_frame_median(movie)
    ref_mean = choose_reference(frame_mean, fps, first_stim_start_sec)
    ref_median = float(np.median(frame_median))

    corrected_divide = divide_mean_correction(movie, frame_mean, ref_mean)
    corrected_sub_mean = subtract_mean_correction(movie, frame_mean, ref_mean)
    corrected_sub_median = subtract_median_correction(movie, frame_median, ref_median)

    out_movie = trial_dir / f"{prefix}_brightness_corrected_movie.tif"
    tiff.imwrite(out_movie, corrected_divide.astype(np.float32))
    print(f"    saved: {out_movie.name}")

    trace_df = pd.DataFrame({
        "frame": np.arange(T),
        "frame_mean": frame_mean,
        "frame_median": frame_median,
    })
    trace_df.to_csv(trial_dir / f"{prefix}_global_brightness_trace.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(frame_mean, label="mean")
    plt.plot(frame_median, label="median", alpha=0.7)
    if first_stim_start_sec is not None and fps > 0:
        stim_frame = int(round(first_stim_start_sec * fps))
        plt.axvline(stim_frame, linestyle="--", linewidth=0.8, label="first stim")
    plt.title(f"{trial_dir.name} global brightness")
    plt.xlabel("frame")
    plt.ylabel("brightness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(trial_dir / f"{prefix}_global_brightness_trace.png", dpi=150)
    plt.close()

    if ENABLE_DEBUG_MODES:
        tiff.imwrite(trial_dir / f"{prefix}_brightness_corrected_subtract_mean_movie.tif",
                     corrected_sub_mean.astype(np.float32))
        tiff.imwrite(trial_dir / f"{prefix}_brightness_corrected_subtract_median_movie.tif",
                     corrected_sub_median.astype(np.float32))

    save_preview(trial_dir, movie, corrected_divide, corrected_sub_mean, corrected_sub_median)
    return True


def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("03.5 Brightness correction")
    print("=" * 70)
    print(f"ROOT_DIR        : {root_dir}")
    print(f"EXISTING_MODE   : {EXISTING_MODE}")
    print(f"REFERENCE_MODE  : {REFERENCE_MODE}")
    print(f"DEBUG_MODES     : {ENABLE_DEBUG_MODES}")

    trials = discover_trials(root_dir)
    print(f"Found {len(trials)} trial(s) with *_corrected_movie.tif")
    if len(trials) == 0:
        return

    ok_n = 0
    fail_n = 0
    for info in trials:
        try:
            if process_trial(info):
                ok_n += 1
            else:
                fail_n += 1
        except Exception as e:
            fail_n += 1
            print(f"[FAILED] {info['trial_dir']}: {e}")

    print("\\n" + "=" * 70)
    print(f"Done. success={ok_n}, failed={fail_n}")
    print("=" * 70)


if __name__ == "__main__":
    main()
