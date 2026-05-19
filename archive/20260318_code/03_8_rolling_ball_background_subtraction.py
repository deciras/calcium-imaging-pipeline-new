#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_8_rolling_ball_background_subtraction.py

目的
----
对 movie 的每一帧做 rolling-ball / white-top-hat 风格的背景扣除，
尽量去掉大尺度、缓慢变化的空间背景，保留小尺度结构（例如细胞）。

这里采用的实现方式是：
    background = gray_opening(frame, footprint=disk(radius))
    corrected  = frame - background

这和 ImageJ 的 Subtract Background（rolling ball）思想接近：
- 大于 ball radius 的平滑背景会被估计出来
- 小于 ball radius 的亮结构会被保留

适用场景
--------
- retina / slice / epifluorescence 数据
- illumination gradient
- 背景不均匀
- 给 04a pixelwise ROI detection 提供更干净的输入

输入优先级
---------
1. *_brightness_adjusted_movie.tif
2. *_brightness_corrected_movie.tif
3. *_corrected_movie.tif

输出
----
trial_dir/
├── *_rolling_ball_movie.tif
└── rolling_ball/
    ├── preview_before_after.png
    └── rolling_ball_summary.json
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.ndimage import grey_opening

ROOT_DIR = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected'
EXISTING_MODE = "overwrite"  # "skip" / "overwrite"

# 核心参数：应明显大于细胞大小
BALL_RADIUS_PX = 30

SAVE_FLOAT32 = True
CLIP_NEGATIVE = True
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


def make_disk_footprint(radius_px: int) -> np.ndarray:
    r = int(radius_px)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x*x + y*y) <= (r*r)
    return mask.astype(bool)


def rolling_ball_subtraction_movie(movie: np.ndarray, radius_px: int) -> np.ndarray:
    """
    逐帧做灰度开运算近似 rolling-ball background subtraction:
        background = grey_opening(frame, footprint=disk(radius))
        corrected  = frame - background
    """
    footprint = make_disk_footprint(radius_px)
    T = movie.shape[0]
    out = np.empty_like(movie, dtype=np.float32)

    for i in range(T):
        frame = movie[i]
        background = grey_opening(frame, footprint=footprint)
        corrected = frame - background
        if CLIP_NEGATIVE:
            corrected = np.maximum(corrected, 0)
        out[i] = corrected.astype(np.float32, copy=False)

    return out


def save_preview(raw_movie: np.ndarray, rb_movie: np.ndarray, out_png: Path):
    idx = raw_movie.shape[0] // 2
    raw = raw_movie[idx]
    rb = rb_movie[idx]

    fig, axes = plt.subplots(2, 2, figsize=PREVIEW_FIGSIZE)

    axes[0, 0].imshow(percentile_stretch(raw), cmap="gray")
    axes[0, 0].set_title("Raw frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(percentile_stretch(rb), cmap="gray")
    axes[0, 1].set_title("Rolling-ball frame")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(percentile_stretch(np.mean(raw_movie, axis=0)), cmap="gray")
    axes[1, 0].set_title("Raw mean")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(percentile_stretch(np.mean(rb_movie, axis=0)), cmap="gray")
    axes[1, 1].set_title("Rolling-ball mean")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    prefix = inputs["prefix"]
    movie_path = Path(inputs["movie_path"])
    movie_source = inputs["movie_source"]

    out_movie = trial_dir / f"{prefix}_rolling_ball_movie.tif"
    out_dir = trial_dir / "rolling_ball"

    if out_movie.exists() and EXISTING_MODE == "skip":
        print(f"[SKIP] {trial_dir.name} -> rolling ball already exists")
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
    rb_movie = rolling_ball_subtraction_movie(movie, BALL_RADIUS_PX)

    save_movie(out_movie, rb_movie)
    save_preview(movie, rb_movie, out_dir / "preview_before_after.png")

    summary = {
        "trial": trial_dir.name,
        "prefix": prefix,
        "movie_source": movie_source,
        "ball_radius_px": BALL_RADIUS_PX,
        "clip_negative": CLIP_NEGATIVE,
        "input_movie": str(movie_path),
        "output_movie": str(out_movie),
    }
    with open(out_dir / "rolling_ball_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"    saved: {out_movie.name}")
    return True


def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("03.8 Rolling-ball background subtraction")
    print("=" * 70)
    print(f"ROOT_DIR          : {root_dir}")
    print(f"BALL_RADIUS_PX    : {BALL_RADIUS_PX}")
    print(f"EXISTING_MODE     : {EXISTING_MODE}")

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
