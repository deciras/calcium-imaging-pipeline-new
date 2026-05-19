#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_6_spatial_highpass.py

目的
----
对每一帧做空间高通：
    highpass = frame - gaussian_blur(frame, sigma)

适合：
- 小尺度 ROI / 细胞
- 大尺度背景发亮、illumination 不均匀
- 给 04a pixelwise patch detection 提供更干净的输入

输入优先级
---------
1. *_brightness_adjusted_movie.tif
2. *_brightness_corrected_movie.tif
3. *_corrected_movie.tif

输出
----
trial_dir/
├── *_spatial_highpass_movie.tif
└── spatial_highpass/
    ├── preview_before_after.png
    └── highpass_summary.json
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.ndimage import gaussian_filter

ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "skip"  # "skip" / "overwrite"

# 必须明显大于细胞尺度；可试 8 / 12 / 16
GAUSSIAN_SIGMA_PX = 12.0

SAVE_FLOAT32 = True
CLIP_NEGATIVE = False
PREVIEW_FIGSIZE = (10, 8)
FIG_DPI = 150


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def safe_makedirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir_contents(path: Path):
    if not path.exists():
        return
    for p in path.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            print(f"    [Warning] Failed to remove {p}: {e}")


def find_trial_inputs(trial_dir: Path):
    adjusted = sorted(trial_dir.glob("*_brightness_adjusted_movie.tif"), key=lambda p: natural_key(p.name))
    corrected_brightness = sorted(trial_dir.glob("*_brightness_corrected_movie.tif"), key=lambda p: natural_key(p.name))
    corrected = sorted(trial_dir.glob("*_corrected_movie.tif"), key=lambda p: natural_key(p.name))

    movie_source = None
    if adjusted:
        movie_path = adjusted[0]
        prefix = movie_path.name.replace("_brightness_adjusted_movie.tif", "")
        movie_source = "brightness_adjusted"
    elif corrected_brightness:
        movie_path = corrected_brightness[0]
        prefix = movie_path.name.replace("_brightness_corrected_movie.tif", "")
        movie_source = "brightness_corrected"
    elif corrected:
        # 排除误生成的 brightness 系列
        filtered = [p for p in corrected if "_brightness_" not in p.name]
        if not filtered:
            return None
        movie_path = filtered[0]
        prefix = movie_path.name.replace("_corrected_movie.tif", "")
        movie_source = "corrected"
    else:
        return None

    json_files = sorted(trial_dir.glob("*_metadata.json"), key=lambda p: natural_key(p.name))
    return {
        "trial_dir": trial_dir,
        "prefix": prefix,
        "movie_path": movie_path,
        "movie_source": movie_source,
        "json_path": json_files[0] if json_files else None,
    }


def discover_trials(root_dir: Path):
    trial_dirs = set()
    for pattern in ["*_brightness_adjusted_movie.tif", "*_brightness_corrected_movie.tif", "*_corrected_movie.tif"]:
        for p in root_dir.rglob(pattern):
            if pattern == "*_corrected_movie.tif" and "_brightness_" in p.name:
                continue
            trial_dirs.add(p.parent)

    trials = []
    for trial_dir in sorted(trial_dirs, key=lambda p: natural_key(str(p))):
        inputs = find_trial_inputs(trial_dir)
        if inputs is not None:
            trials.append(inputs)
    return trials


def load_movie(movie_path: Path) -> np.ndarray:
    with tf.TiffFile(movie_path) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Movie must be 3D (T,Y,X), got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def save_movie(path: Path, movie: np.ndarray):
    arr = np.asarray(movie, dtype=np.float32 if SAVE_FLOAT32 else np.uint16)
    tf.imwrite(str(path), arr, imagej=False, bigtiff=True)


def percentile_stretch(img: np.ndarray, pmin=1.0, pmax=99.5):
    x = np.asarray(img, dtype=np.float32)
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def spatial_highpass_movie(movie: np.ndarray, sigma_px: float) -> np.ndarray:
    # 只做空间高通，不做时间平滑
    lowfreq = gaussian_filter(movie, sigma=(0, sigma_px, sigma_px))
    hp = movie - lowfreq
    if CLIP_NEGATIVE:
        hp = np.maximum(hp, 0)
    return hp.astype(np.float32, copy=False)


def save_preview(raw_movie: np.ndarray, hp_movie: np.ndarray, out_png: Path):
    idx = raw_movie.shape[0] // 2
    raw = raw_movie[idx]
    hp = hp_movie[idx]

    fig, axes = plt.subplots(2, 2, figsize=PREVIEW_FIGSIZE)

    axes[0, 0].imshow(percentile_stretch(raw), cmap="gray")
    axes[0, 0].set_title("Raw frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(percentile_stretch(hp), cmap="gray")
    axes[0, 1].set_title("Spatial high-pass frame")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(percentile_stretch(np.mean(raw_movie, axis=0)), cmap="gray")
    axes[1, 0].set_title("Raw mean")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(percentile_stretch(np.mean(hp_movie, axis=0)), cmap="gray")
    axes[1, 1].set_title("High-pass mean")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    prefix = inputs["prefix"]
    movie_path = Path(inputs["movie_path"])
    movie_source = inputs["movie_source"]

    out_movie = trial_dir / f"{prefix}_spatial_highpass_movie.tif"
    out_dir = trial_dir / "spatial_highpass"

    if out_movie.exists() and EXISTING_MODE == "skip":
        print(f"[SKIP] {trial_dir.name} -> spatial high-pass already exists")
        return True

    if EXISTING_MODE == "overwrite":
        if out_movie.exists():
            out_movie.unlink()
        if out_dir.exists():
            clear_dir_contents(out_dir)

    safe_makedirs(out_dir)

    print(f"\n=== Trial: {trial_dir.name} ===")
    print(f"    movie: {movie_path.name}")
    print(f"    source: {movie_source}")

    movie = load_movie(movie_path)
    hp_movie = spatial_highpass_movie(movie, GAUSSIAN_SIGMA_PX)

    save_movie(out_movie, hp_movie)
    save_preview(movie, hp_movie, out_dir / "preview_before_after.png")

    summary = {
        "trial": trial_dir.name,
        "prefix": prefix,
        "movie_source": movie_source,
        "gaussian_sigma_px": GAUSSIAN_SIGMA_PX,
        "clip_negative": CLIP_NEGATIVE,
        "input_movie": str(movie_path),
        "output_movie": str(out_movie),
    }
    with open(out_dir / "highpass_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"    saved: {out_movie.name}")
    return True


def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("03.6 Spatial high-pass for 04a")
    print("=" * 70)
    print(f"ROOT_DIR            : {root_dir}")
    print(f"GAUSSIAN_SIGMA_PX   : {GAUSSIAN_SIGMA_PX}")
    print(f"EXISTING_MODE       : {EXISTING_MODE}")

    trials = discover_trials(root_dir)
    print(f"Found {len(trials)} trial(s)")
    if len(trials) == 0:
        return

    ok_n = 0
    fail_n = 0
    for inputs in trials:
        try:
            if process_single_trial(inputs):
                ok_n += 1
            else:
                fail_n += 1
        except Exception as e:
            fail_n += 1
            print(f"[FAILED] {inputs['trial_dir']}: {e}")

    print("\n" + "=" * 70)
    print(f"Done. success={ok_n}, failed={fail_n}")
    print("=" * 70)


if __name__ == "__main__":
    main()
