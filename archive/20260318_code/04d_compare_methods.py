#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04d_compare_methods.py

第一版 benchmark comparison:
- 读取 trial_dir/benchmark/pixelwise/averaged/patch_mask.tif
- 读取 trial_dir/benchmark/suite2p/roi_mask.tif
- 可选读取 trial_dir/benchmark/aqua/event_mask.tif
- 计算像素级 overlap mask
- 输出静态 panel 图和 overlap 视频

输出目录:
trial_dir/benchmark/
    comparison_summary.csv
    all_methods_overlap.png
    all_methods_overlap_movie.mp4
    overlap_masks/
        A_only_mask.tif
        B_only_mask.tif
        C_only_mask.tif        (如果 C 存在)
        AB_mask.tif
        AC_mask.tif            (如果 C 存在)
        BC_mask.tif            (如果 C 存在)
        ABC_mask.tif           (如果 C 存在)

约定:
A = pixelwise
B = suite2p
C = aqua
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf

# ============================================================
# 配置区域
# ============================================================
ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "overwrite"   # "skip" / "overwrite"

# 输出
SAVE_VIDEO = True
VIDEO_FPS = 5
MAX_VIDEO_FRAMES = None   # None = 全部帧；可设整数限制输出长度

# panel 可视化
PANEL_FIGSIZE = (12, 8)
PANEL_DPI = 160
MASK_ALPHA = 0.55

# 颜色（RGB 0-1）
COLOR_A = (0.20, 0.90, 0.20)   # green
COLOR_B = (1.00, 0.20, 0.20)   # red
COLOR_C = (0.20, 0.60, 1.00)   # blue
COLOR_AB = (1.00, 0.85, 0.10)  # yellow
COLOR_AC = (0.75, 0.30, 1.00)  # magenta-ish
COLOR_BC = (1.00, 0.50, 0.10)  # orange
COLOR_ABC = (1.00, 1.00, 1.00) # white

# ============================================================
# 工具函数
# ============================================================

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


def discover_trials(root_dir: Path):
    movie_files = sorted(root_dir.rglob("*_corrected_movie.tif"), key=lambda p: natural_key(str(p)))
    trials = []
    for movie_path in movie_files:
        trials.append({
            "trial_dir": movie_path.parent,
            "movie_path": movie_path,
            "prefix": movie_path.name.replace("_corrected_movie.tif", "").replace("_corrected_movie.tiff", ""),
        })
    return trials


def load_movie(movie_path: Path) -> np.ndarray:
    with tf.TiffFile(movie_path) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Movie must be 3D (T,Y,X), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return np.zeros(frame.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.float32)
    out = (frame - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def load_binary_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = tf.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}")
    return arr > 0


def save_mask(path: Path, mask: np.ndarray):
    tf.imwrite(str(path), (mask.astype(np.uint8) * 255), imagej=True)


def count_components(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask.astype(bool))
    return int(n)


def make_color_overlay(gray01: np.ndarray, mask: np.ndarray, color, alpha=MASK_ALPHA) -> np.ndarray:
    base = np.dstack([gray01, gray01, gray01]).astype(np.float32)
    if mask is None:
        return base
    out = base.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color_arr
    return np.clip(out, 0, 1)


def panel_layout(has_c: bool):
    if has_c:
        return [
            ("A only", "A_only"),
            ("B only", "B_only"),
            ("C only", "C_only"),
            ("A∩B", "AB"),
            ("A∩C", "AC"),
            ("B∩C", "BC"),
            ("A∩B∩C", "ABC"),
        ], (2, 4)
    return [
        ("A only", "A_only"),
        ("B only", "B_only"),
        ("A∩B", "AB"),
    ], (1, 4)


def colors_for_key(key: str):
    return {
        "A_only": COLOR_A,
        "B_only": COLOR_B,
        "C_only": COLOR_C,
        "AB": COLOR_AB,
        "AC": COLOR_AC,
        "BC": COLOR_BC,
        "ABC": COLOR_ABC,
    }[key]


def compute_overlap_masks(A: np.ndarray, B: np.ndarray, C: np.ndarray | None):
    if C is None:
        return {
            "A_only": A & ~B,
            "B_only": B & ~A,
            "AB": A & B,
        }
    return {
        "A_only": A & ~B & ~C,
        "B_only": B & ~A & ~C,
        "C_only": C & ~A & ~B,
        "AB": A & B & ~C,
        "AC": A & C & ~B,
        "BC": B & C & ~A,
        "ABC": A & B & C,
    }


def build_summary_rows(trial_name: str, masks: dict[str, np.ndarray], has_c: bool) -> pd.DataFrame:
    rows = []
    method_name = {
        "A_only": "pixelwise_only",
        "B_only": "suite2p_only",
        "C_only": "aqua_only",
        "AB": "pixelwise_suite2p_overlap",
        "AC": "pixelwise_aqua_overlap",
        "BC": "suite2p_aqua_overlap",
        "ABC": "triple_overlap",
    }
    for key, mask in masks.items():
        rows.append({
            "trial": trial_name,
            "group": key,
            "label": method_name[key],
            "n_pixels": int(mask.sum()),
            "n_components": count_components(mask),
        })
    # method-level totals for convenience
    rows.extend([
        {"trial": trial_name, "group": "A_total", "label": "pixelwise_total", "n_pixels": int(np.sum(masks.get("A_only", 0)) + np.sum(masks.get("AB", 0)) + (np.sum(masks.get("AC", 0)) if has_c else 0) + (np.sum(masks.get("ABC", 0)) if has_c else 0)), "n_components": np.nan},
        {"trial": trial_name, "group": "B_total", "label": "suite2p_total", "n_pixels": int(np.sum(masks.get("B_only", 0)) + np.sum(masks.get("AB", 0)) + (np.sum(masks.get("BC", 0)) if has_c else 0) + (np.sum(masks.get("ABC", 0)) if has_c else 0)), "n_components": np.nan},
    ])
    if has_c:
        rows.append({"trial": trial_name, "group": "C_total", "label": "aqua_total", "n_pixels": int(np.sum(masks.get("C_only", 0)) + np.sum(masks.get("AC", 0)) + np.sum(masks.get("BC", 0)) + np.sum(masks.get("ABC", 0))), "n_components": np.nan})
    return pd.DataFrame(rows)


def save_static_panel(out_png: Path, mean_img: np.ndarray, masks: dict[str, np.ndarray], has_c: bool, trial_name: str):
    gray = normalize_frame(mean_img)
    layout, grid = panel_layout(has_c)
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows, ncols, figsize=PANEL_FIGSIZE)
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx, (title, key) in enumerate(layout):
        overlay = make_color_overlay(gray, masks[key], colors_for_key(key))
        axes[idx].imshow(overlay)
        axes[idx].set_title(f"{title}\npx={int(masks[key].sum())}", fontsize=10)
        axes[idx].axis("off")

    if has_c and len(axes) > len(layout):
        # last empty panel as legend/original
        axes[len(layout)].imshow(np.dstack([gray, gray, gray]))
        axes[len(layout)].set_title("Mean image", fontsize=10)
        axes[len(layout)].axis("off")
    elif (not has_c) and len(axes) > len(layout):
        axes[len(layout)].imshow(np.dstack([gray, gray, gray]))
        axes[len(layout)].set_title("Mean image", fontsize=10)
        axes[len(layout)].axis("off")

    fig.suptitle(f"Method overlap - {trial_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)


def save_video(out_mp4: Path, movie: np.ndarray, masks: dict[str, np.ndarray], has_c: bool):
    layout, grid = panel_layout(has_c)
    nrows, ncols = grid
    T = movie.shape[0]
    if MAX_VIDEO_FRAMES is not None:
        T = min(T, int(MAX_VIDEO_FRAMES))

    writer = iio.imopen(str(out_mp4), "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=VIDEO_FPS)

    try:
        for t in range(T):
            gray = normalize_frame(movie[t])
            panels = []
            for _, key in layout:
                overlay = make_color_overlay(gray, masks[key], colors_for_key(key))
                panels.append((overlay * 255).astype(np.uint8))
            # add mean/raw reference panel to fill grid
            panels.append((np.dstack([gray, gray, gray]) * 255).astype(np.uint8))

            # ensure full grid size
            target = nrows * ncols
            while len(panels) < target:
                panels.append(np.zeros_like(panels[0], dtype=np.uint8))

            rows = []
            k = 0
            for _ in range(nrows):
                row = np.concatenate(panels[k:k+ncols], axis=1)
                rows.append(row)
                k += ncols
            frame = np.concatenate(rows, axis=0)
            writer.write_frame(frame)
    finally:
        writer.close()


# ============================================================
# 单个 trial
# ============================================================

def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    movie_path = Path(inputs["movie_path"])

    bench_root = trial_dir / "benchmark"
    out_dir = bench_root
    overlap_dir = out_dir / "overlap_masks"

    png_out = out_dir / "all_methods_overlap.png"
    mp4_out = out_dir / "all_methods_overlap_movie.mp4"
    csv_out = out_dir / "comparison_summary.csv"

    # inputs
    A_path = bench_root / "pixelwise" / "averaged" / "patch_mask.tif"
    B_path = bench_root / "suite2p" / "roi_mask.tif"
    C_path = bench_root / "aqua" / "event_mask.tif"

    if not A_path.exists() or not B_path.exists():
        print(f"[SKIP] {trial_dir.name}: missing A or B mask")
        return False

    if EXISTING_MODE == "skip" and png_out.exists() and csv_out.exists() and (not SAVE_VIDEO or mp4_out.exists()):
        print(f"[SKIP] {trial_dir.name}: comparison outputs already exist")
        return True

    safe_makedirs(overlap_dir)
    if EXISTING_MODE == "overwrite":
        for p in [png_out, mp4_out, csv_out]:
            if p.exists():
                p.unlink()
        clear_dir_contents(overlap_dir)
        safe_makedirs(overlap_dir)

    A = load_binary_mask(A_path)
    B = load_binary_mask(B_path)
    C = load_binary_mask(C_path) if C_path.exists() else None

    if A is None or B is None:
        print(f"[SKIP] {trial_dir.name}: failed to load A/B masks")
        return False

    if A.shape != B.shape:
        raise ValueError(f"A/B mask shape mismatch: {A.shape} vs {B.shape}")
    if C is not None and C.shape != A.shape:
        raise ValueError(f"C mask shape mismatch: {C.shape} vs {A.shape}")

    masks = compute_overlap_masks(A, B, C)
    has_c = C is not None

    # save overlap masks
    for key, mask in masks.items():
        save_mask(overlap_dir / f"{key}_mask.tif", mask)

    # summary csv
    summary_df = build_summary_rows(trial_dir.name, masks, has_c)
    summary_df.to_csv(csv_out, index=False)

    # static image
    movie = load_movie(movie_path)
    mean_img = np.mean(movie, axis=0, dtype=np.float32)
    save_static_panel(png_out, mean_img, masks, has_c, trial_dir.name)

    # video
    if SAVE_VIDEO:
        save_video(mp4_out, movie, masks, has_c)

    print(f"[OK] {trial_dir.name}: saved comparison outputs")
    return True


# ============================================================
# main
# ============================================================

def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("04d Compare methods")
    print("=" * 70)
    print(f"ROOT_DIR      : {root_dir}")
    print(f"EXISTING_MODE : {EXISTING_MODE}")
    print(f"SAVE_VIDEO    : {SAVE_VIDEO}")
    print()

    trials = discover_trials(root_dir)
    print(f"Found {len(trials)} trial(s) with *_corrected_movie.tif")
    if len(trials) == 0:
        print("No trials found. Nothing to do.")
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
