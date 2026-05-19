#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_7_global_signal_regression.py

目的
----
对每个像素做全局信号回归（Global Signal Regression, GSR）：

    pixel_trace(t) ~ beta0 + beta1 * global_trace(t)

取残差作为新的 movie。

这比简单 subtract/divide 更接近“方案 B”：
- 允许不同像素对全局伪差的敏感度不同
- 每个像素有各自的 beta1
- 本质上是把 stimulus-coupled global component 回归掉

输入优先级
---------
1. *_brightness_adjusted_movie.tif
2. *_brightness_corrected_movie.tif
3. *_corrected_movie.tif

输出
----
trial_dir/
├── *_gsr_movie.tif
└── global_signal_regression/
    ├── preview_before_after.png
    └── gsr_summary.json
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "overwrite"  # "skip" / "overwrite"

GLOBAL_TRACE_MODE = "mean"   # "mean" / "median"
SAVE_FLOAT32 = True
KEEP_BASELINE_LEVEL = True
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

    return {
        "trial_dir": trial_dir,
        "prefix": prefix,
        "movie_path": movie_path,
        "movie_source": movie_source,
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


def global_trace_from_movie(movie: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return movie.mean(axis=(1, 2), dtype=np.float64)
    if mode == "median":
        return np.median(movie, axis=(1, 2)).astype(np.float64)
    raise ValueError(f"Unknown GLOBAL_TRACE_MODE: {mode}")


def regress_out_global_signal(movie: np.ndarray, global_trace: np.ndarray) -> np.ndarray:
    """
    movie shape: [T, Y, X]
    global_trace shape: [T]

    对每个像素做：
        y = beta0 + beta1 * g + e
    输出 residual + mean(y)（可选）
    """
    T, Y, X = movie.shape
    M = movie.reshape(T, -1).astype(np.float64)  # [T, N]
    g = np.asarray(global_trace, dtype=np.float64)

    g_centered = g - g.mean()
    denom = np.sum(g_centered ** 2)
    if denom <= 1e-12:
        raise ValueError("Global trace variance is too small for regression.")

    M_mean = M.mean(axis=0, keepdims=True)       # [1, N]
    M_centered = M - M_mean

    beta1 = (g_centered[:, None] * M_centered).sum(axis=0) / denom
    fitted = np.outer(g_centered, beta1)
    residual = M_centered - fitted

    if KEEP_BASELINE_LEVEL:
        residual = residual + M_mean

    out = residual.reshape(T, Y, X).astype(np.float32)
    return out


def save_preview(raw_movie: np.ndarray, gsr_movie: np.ndarray, out_png: Path):
    idx = raw_movie.shape[0] // 2
    raw = raw_movie[idx]
    gsr = gsr_movie[idx]

    fig, axes = plt.subplots(2, 2, figsize=PREVIEW_FIGSIZE)

    axes[0, 0].imshow(percentile_stretch(raw), cmap="gray")
    axes[0, 0].set_title("Raw frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(percentile_stretch(gsr), cmap="gray")
    axes[0, 1].set_title("GSR frame")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(percentile_stretch(np.mean(raw_movie, axis=0)), cmap="gray")
    axes[1, 0].set_title("Raw mean")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(percentile_stretch(np.mean(gsr_movie, axis=0)), cmap="gray")
    axes[1, 1].set_title("GSR mean")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    prefix = inputs["prefix"]
    movie_path = Path(inputs["movie_path"])
    movie_source = inputs["movie_source"]

    out_movie = trial_dir / f"{prefix}_gsr_movie.tif"
    out_dir = trial_dir / "global_signal_regression"

    if out_movie.exists() and EXISTING_MODE == "skip":
        print(f"[SKIP] {trial_dir.name} -> GSR movie already exists")
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
    gtrace = global_trace_from_movie(movie, GLOBAL_TRACE_MODE)
    gsr_movie = regress_out_global_signal(movie, gtrace)

    save_movie(out_movie, gsr_movie)
    save_preview(movie, gsr_movie, out_dir / "preview_before_after.png")

    summary = {
        "trial": trial_dir.name,
        "prefix": prefix,
        "movie_source": movie_source,
        "global_trace_mode": GLOBAL_TRACE_MODE,
        "keep_baseline_level": KEEP_BASELINE_LEVEL,
        "input_movie": str(movie_path),
        "output_movie": str(out_movie),
    }
    with open(out_dir / "gsr_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"    saved: {out_movie.name}")
    return True


def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("03.7 Global signal regression")
    print("=" * 70)
    print(f"ROOT_DIR            : {root_dir}")
    print(f"GLOBAL_TRACE_MODE   : {GLOBAL_TRACE_MODE}")
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
