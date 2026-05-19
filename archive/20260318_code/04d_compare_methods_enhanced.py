#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04d_compare_methods.py

Benchmark comparison visualization for ROI methods.

Reads per-trial outputs:
- benchmark/pixelwise/averaged/patch_mask.tif
- benchmark/suite2p/roi_mask.tif + roi_label_map.tif + roi_summary.csv
- optional benchmark/aqua/event_mask.tif

Computes pixel-level overlaps and saves:
trial_dir/benchmark/
    comparison_summary.csv
    all_methods_overlap.png
    all_methods_overlap_movie.mp4
    overlap_masks/
        A_only_mask.tif
        B_only_mask.tif
        ...
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

# ============================================================
# 配置区域
# ============================================================
ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "overwrite"   # "skip" / "overwrite"

# 视频输出
SAVE_VIDEO = True
PLAYBACK_SPEED_MULTIPLIER = 30.0   # 输出视频播放速度 = 实际成像速度 × 此倍数
MAX_OUTPUT_VIDEO_FPS = 60.0        # 编码输出 fps 上限；超过时自动抽帧
MAX_VIDEO_FRAMES = None            # None = 全部帧；可设整数限制输出长度（针对抽帧后）
VIDEO_CODEC = "libx264"
VIDEO_PLUGIN = "pyav"

# 叠加绘制
ROI_DRAW_STYLE = "outline"        # 目前实现 outline
OUTLINE_WIDTH = 1

# suite2p 显示模式
# "iscell_only" = 只把 iscell=1 纳入 B 集合（默认）
# "all"         = iscell 0/1 全部纳入 B 集合，不区分
# "split"       = 集合运算仍用全部 B；绘图时 B-only 面板区分 iscell=1/0 颜色
SUITE2P_DISPLAY_MODE = "all"

# 对比度增强
CONTRAST_MODE = "percentile"      # 当前实现 percentile
CONTRAST_PMIN = 1.0
CONTRAST_PMAX = 99.5
CONTRAST_SAMPLE_FRAMES = 200       # 估计全局显示范围时最多采样多少帧

# panel 可视化
PANEL_FIGSIZE = (12, 8)
PANEL_DPI = 160

# 面板颜色（RGB 0-255）
COLOR_A = (51, 230, 51)        # green
COLOR_B = (255, 51, 51)        # red
COLOR_B0 = (255, 140, 0)       # orange for suite2p iscell=0
COLOR_C = (51, 153, 255)       # blue
COLOR_AB = (255, 217, 26)      # yellow
COLOR_AC = (191, 77, 255)      # purple
COLOR_BC = (255, 128, 26)      # orange
COLOR_ABC = (255, 255, 255)    # white
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)

SHOW_TIME_TEXT = True
SHOW_FRAME_TEXT = True
SHOW_STIM_STATUS = True

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
    lab, n = ndi.label(mask.astype(bool))
    return int(n)


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_trial_fps(trial_dir: Path, prefix: str) -> float:
    meta_path = trial_dir / f"{prefix}_metadata.json"
    if not meta_path.exists():
        metas = sorted(trial_dir.glob("*_metadata.json"), key=lambda p: natural_key(p.name))
        meta_path = metas[0] if metas else meta_path
    meta = load_json(meta_path)
    fps = meta.get("temporal_calibration", {}).get("fps", None)
    try:
        fps = float(fps)
        if fps > 0:
            return fps
    except Exception:
        pass
    return 1.0


def load_stim_events(trial_dir: Path, prefix: str) -> pd.DataFrame | None:
    path = trial_dir / f"{prefix}_stim_events.csv"
    if not path.exists():
        candidates = sorted(trial_dir.glob("*_stim_events.csv"), key=lambda p: natural_key(p.name))
        path = candidates[0] if candidates else path
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not {"start_time_sec", "end_time_sec"}.issubset(df.columns):
        return None
    df = df.copy()
    df["start_time_sec"] = pd.to_numeric(df["start_time_sec"], errors="coerce")
    df["end_time_sec"] = pd.to_numeric(df["end_time_sec"], errors="coerce")
    df = df.dropna(subset=["start_time_sec", "end_time_sec"])
    df = df[df["end_time_sec"] > df["start_time_sec"]]
    if df.empty:
        return None
    return df.sort_values("start_time_sec").reset_index(drop=True)


def estimate_display_range(movie: np.ndarray) -> tuple[float, float]:
    T = movie.shape[0]
    if CONTRAST_SAMPLE_FRAMES is None or T <= CONTRAST_SAMPLE_FRAMES:
        sample = movie
    else:
        idx = np.linspace(0, T - 1, int(CONTRAST_SAMPLE_FRAMES)).astype(int)
        sample = movie[idx]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.0, 1.0
    if CONTRAST_MODE == "percentile":
        lo, hi = np.percentile(finite, [CONTRAST_PMIN, CONTRAST_PMAX])
    else:
        lo, hi = np.min(finite), np.max(finite)
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize_frame_global(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    out = (frame - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def binary_outline(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    eroded = ndi.binary_erosion(m)
    return m & (~eroded)


def draw_outline_rgb(base_rgb: np.ndarray, mask: np.ndarray, color, width: int = 1) -> np.ndarray:
    out = base_rgb.copy()
    outline = binary_outline(mask)
    if width > 1:
        outline = ndi.binary_dilation(outline, iterations=width - 1)
    out[outline] = np.array(color, dtype=np.uint8)
    return out


def render_panel(gray01: np.ndarray,
                 key: str,
                 masks: dict[str, np.ndarray],
                 has_c: bool,
                 B1_only: np.ndarray | None = None,
                 B0_only: np.ndarray | None = None) -> np.ndarray:
    rgb = (np.dstack([gray01, gray01, gray01]) * 255).astype(np.uint8)
    # 默认按 overlap 组绘制 outline
    if key == "A_only":
        return draw_outline_rgb(rgb, masks[key], COLOR_A, OUTLINE_WIDTH)
    if key == "B_only":
        if SUITE2P_DISPLAY_MODE == "split" and B1_only is not None and B0_only is not None:
            rgb = draw_outline_rgb(rgb, B1_only, COLOR_B, OUTLINE_WIDTH)
            rgb = draw_outline_rgb(rgb, B0_only, COLOR_B0, OUTLINE_WIDTH)
            return rgb
        return draw_outline_rgb(rgb, masks[key], COLOR_B, OUTLINE_WIDTH)
    if key == "C_only":
        return draw_outline_rgb(rgb, masks[key], COLOR_C, OUTLINE_WIDTH)
    return draw_outline_rgb(rgb, masks[key], {
        "AB": COLOR_AB,
        "AC": COLOR_AC,
        "BC": COLOR_BC,
        "ABC": COLOR_ABC,
    }[key], OUTLINE_WIDTH)


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


def build_summary_rows(trial_name: str, masks: dict[str, np.ndarray], has_c: bool,
                       actual_fps: float, output_fps: float, frame_stride: int) -> pd.DataFrame:
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
            "actual_fps": actual_fps,
            "playback_speed_multiplier": PLAYBACK_SPEED_MULTIPLIER,
            "output_video_fps": output_fps,
            "frame_stride": frame_stride,
            "suite2p_display_mode": SUITE2P_DISPLAY_MODE,
        })
    rows.extend([
        {"trial": trial_name, "group": "A_total", "label": "pixelwise_total", "n_pixels": int(np.sum(masks.get("A_only", 0)) + np.sum(masks.get("AB", 0)) + (np.sum(masks.get("AC", 0)) if has_c else 0) + (np.sum(masks.get("ABC", 0)) if has_c else 0)), "n_components": np.nan, "actual_fps": actual_fps, "playback_speed_multiplier": PLAYBACK_SPEED_MULTIPLIER, "output_video_fps": output_fps, "frame_stride": frame_stride, "suite2p_display_mode": SUITE2P_DISPLAY_MODE},
        {"trial": trial_name, "group": "B_total", "label": "suite2p_total", "n_pixels": int(np.sum(masks.get("B_only", 0)) + np.sum(masks.get("AB", 0)) + (np.sum(masks.get("BC", 0)) if has_c else 0) + (np.sum(masks.get("ABC", 0)) if has_c else 0)), "n_components": np.nan, "actual_fps": actual_fps, "playback_speed_multiplier": PLAYBACK_SPEED_MULTIPLIER, "output_video_fps": output_fps, "frame_stride": frame_stride, "suite2p_display_mode": SUITE2P_DISPLAY_MODE},
    ])
    if has_c:
        rows.append({"trial": trial_name, "group": "C_total", "label": "aqua_total", "n_pixels": int(np.sum(masks.get("C_only", 0)) + np.sum(masks.get("AC", 0)) + np.sum(masks.get("BC", 0)) + np.sum(masks.get("ABC", 0))), "n_components": np.nan, "actual_fps": actual_fps, "playback_speed_multiplier": PLAYBACK_SPEED_MULTIPLIER, "output_video_fps": output_fps, "frame_stride": frame_stride, "suite2p_display_mode": SUITE2P_DISPLAY_MODE})
    return pd.DataFrame(rows)


def save_static_panel(out_png: Path, mean_img: np.ndarray, masks: dict[str, np.ndarray], has_c: bool,
                      trial_name: str, B1_only: np.ndarray | None = None, B0_only: np.ndarray | None = None):
    lo, hi = estimate_display_range(mean_img[None, ...])
    gray = normalize_frame_global(mean_img, lo, hi)
    layout, grid = panel_layout(has_c)
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows, ncols, figsize=PANEL_FIGSIZE)
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx, (title, key) in enumerate(layout):
        panel = render_panel(gray, key, masks, has_c, B1_only=B1_only, B0_only=B0_only)
        axes[idx].imshow(panel)
        axes[idx].set_title(f"{title}\npx={int(masks[key].sum())}", fontsize=10)
        axes[idx].axis("off")

    if len(axes) > len(layout):
        axes[len(layout)].imshow((np.dstack([gray, gray, gray]) * 255).astype(np.uint8))
        axes[len(layout)].set_title("Mean image", fontsize=10)
        axes[len(layout)].axis("off")

    fig.suptitle(f"Method overlap - {trial_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)


def _frame_text_lines(t_sec: float, frame_idx: int, stim_df: pd.DataFrame | None):
    lines = []
    if SHOW_TIME_TEXT:
        lines.append(f"t = {t_sec:.2f} s")
    if SHOW_FRAME_TEXT:
        lines.append(f"frame = {frame_idx}")
    if SHOW_STIM_STATUS:
        is_on = False
        if stim_df is not None and not stim_df.empty:
            is_on = bool(np.any((stim_df["start_time_sec"].values <= t_sec) & (t_sec < stim_df["end_time_sec"].values)))
        lines.append("stim ON" if is_on else "stim OFF")
    return lines


def _draw_text_block(img_uint8: np.ndarray, lines: list[str], x=8, y=8) -> np.ndarray:
    img = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(img)
    cur_y = y
    for line in lines:
        bbox = draw.textbbox((x, cur_y), line)
        draw.rectangle((bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1), fill=TEXT_BG)
        draw.text((x, cur_y), line, fill=TEXT_COLOR)
        cur_y = bbox[3] + 4
    return np.array(img)


def compute_output_fps_and_stride(actual_fps: float) -> tuple[float, int]:
    desired = max(float(actual_fps) * float(PLAYBACK_SPEED_MULTIPLIER), 1.0)
    if desired <= MAX_OUTPUT_VIDEO_FPS:
        return desired, 1
    stride = int(math.ceil(desired / float(MAX_OUTPUT_VIDEO_FPS)))
    output_fps = desired / stride
    return float(output_fps), max(1, stride)


def save_video(out_mp4: Path, movie: np.ndarray, masks: dict[str, np.ndarray], has_c: bool,
               actual_fps: float, stim_df: pd.DataFrame | None,
               B1_only: np.ndarray | None = None, B0_only: np.ndarray | None = None):
    layout, grid = panel_layout(has_c)
    nrows, ncols = grid
    T = movie.shape[0]
    lo, hi = estimate_display_range(movie)
    out_fps, frame_stride = compute_output_fps_and_stride(actual_fps)

    frame_indices = list(range(0, T, frame_stride))
    if MAX_VIDEO_FRAMES is not None:
        frame_indices = frame_indices[: int(MAX_VIDEO_FRAMES)]

    writer = iio.imopen(str(out_mp4), "w", plugin=VIDEO_PLUGIN)
    writer.init_video_stream(VIDEO_CODEC, fps=out_fps)

    try:
        for frame_idx in frame_indices:
            gray = normalize_frame_global(movie[frame_idx], lo, hi)
            panels = []
            for title, key in layout:
                panel = render_panel(gray, key, masks, has_c, B1_only=B1_only, B0_only=B0_only)
                panel = _draw_text_block(panel, [title] + _frame_text_lines(frame_idx / actual_fps, frame_idx, stim_df))
                panels.append(panel)

            ref = (np.dstack([gray, gray, gray]) * 255).astype(np.uint8)
            ref = _draw_text_block(ref, ["Raw frame"] + _frame_text_lines(frame_idx / actual_fps, frame_idx, stim_df))
            panels.append(ref)

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

    return out_fps, frame_stride


def load_suite2p_masks(bench_root: Path):
    roi_mask_path = bench_root / "suite2p" / "roi_mask.tif"
    roi_label_path = bench_root / "suite2p" / "roi_label_map.tif"
    roi_summary_path = bench_root / "suite2p" / "roi_summary.csv"

    roi_mask = load_binary_mask(roi_mask_path)
    if roi_mask is None:
        return None, None, None

    if SUITE2P_DISPLAY_MODE == "all":
        return roi_mask, None, None

    if not roi_label_path.exists() or not roi_summary_path.exists():
        # 回退：没有 label 或 summary 时只能用总 mask
        return roi_mask, None, None

    labels = np.asarray(tf.imread(str(roi_label_path)))
    df = pd.read_csv(roi_summary_path)
    if "roi_id" not in df.columns or "iscell" not in df.columns:
        return roi_mask, None, None

    iscell_ids = set(df.loc[pd.to_numeric(df["iscell"], errors="coerce").fillna(0).astype(int) == 1, "roi_id"].astype(int).tolist())
    noncell_ids = set(df.loc[pd.to_numeric(df["iscell"], errors="coerce").fillna(0).astype(int) == 0, "roi_id"].astype(int).tolist())

    B1 = np.isin(labels, list(iscell_ids)) if len(iscell_ids) > 0 else np.zeros(labels.shape, dtype=bool)
    B0 = np.isin(labels, list(noncell_ids)) if len(noncell_ids) > 0 else np.zeros(labels.shape, dtype=bool)

    if SUITE2P_DISPLAY_MODE == "iscell_only":
        return B1, B1, B0
    # split: 集合运算用 all；显示时分开 B1/B0
    return (B1 | B0), B1, B0

# ============================================================
# 单个 trial
# ============================================================

def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    movie_path = Path(inputs["movie_path"])
    prefix = inputs["prefix"]

    bench_root = trial_dir / "benchmark"
    out_dir = bench_root
    overlap_dir = out_dir / "overlap_masks"

    png_out = out_dir / "all_methods_overlap.png"
    mp4_out = out_dir / "all_methods_overlap_movie.mp4"
    csv_out = out_dir / "comparison_summary.csv"

    # inputs
    A_path = bench_root / "pixelwise" / "averaged" / "patch_mask.tif"
    C_path = bench_root / "aqua" / "event_mask.tif"

    if not A_path.exists():
        print(f"[SKIP] {trial_dir.name}: missing A mask")
        return False

    A = load_binary_mask(A_path)
    B, B1, B0 = load_suite2p_masks(bench_root)
    if B is None:
        print(f"[SKIP] {trial_dir.name}: missing B mask")
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

    C = load_binary_mask(C_path) if C_path.exists() else None

    if A.shape != B.shape:
        raise ValueError(f"A/B mask shape mismatch: {A.shape} vs {B.shape}")
    if C is not None and C.shape != A.shape:
        raise ValueError(f"C mask shape mismatch: {C.shape} vs {A.shape}")
    if B1 is not None and B1.shape != A.shape:
        B1 = None
    if B0 is not None and B0.shape != A.shape:
        B0 = None

    masks = compute_overlap_masks(A, B, C)
    has_c = C is not None

    # extra split masks for inspection
    if SUITE2P_DISPLAY_MODE == "split" and B1 is not None and B0 is not None:
        save_mask(overlap_dir / "B_iscell1_mask.tif", B1)
        save_mask(overlap_dir / "B_iscell0_mask.tif", B0)

    for key, mask in masks.items():
        save_mask(overlap_dir / f"{key}_mask.tif", mask)

    actual_fps = get_trial_fps(trial_dir, prefix)
    output_fps, frame_stride = compute_output_fps_and_stride(actual_fps)
    summary_df = build_summary_rows(trial_dir.name, masks, has_c, actual_fps, output_fps, frame_stride)
    summary_df.to_csv(csv_out, index=False)

    movie = load_movie(movie_path)
    mean_img = np.mean(movie, axis=0, dtype=np.float32)
    save_static_panel(png_out, mean_img, masks, has_c, trial_dir.name, B1_only=B1 & masks.get("B_only", False) if B1 is not None and "B_only" in masks else None, B0_only=B0 & masks.get("B_only", False) if B0 is not None and "B_only" in masks else None)

    if SAVE_VIDEO:
        stim_df = load_stim_events(trial_dir, prefix)
        try:
            output_fps, frame_stride = save_video(
                mp4_out,
                movie,
                masks,
                has_c,
                actual_fps=actual_fps,
                stim_df=stim_df,
                B1_only=B1 & masks.get("B_only", False) if B1 is not None and "B_only" in masks else None,
                B0_only=B0 & masks.get("B_only", False) if B0 is not None and "B_only" in masks else None,
            )
        except Exception as e:
            print(f"[WARN] {trial_dir.name}: video export failed but masks/images were saved: {e}")

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
    print(f"ROOT_DIR                  : {root_dir}")
    print(f"EXISTING_MODE             : {EXISTING_MODE}")
    print(f"SAVE_VIDEO                : {SAVE_VIDEO}")
    print(f"PLAYBACK_SPEED_MULTIPLIER : {PLAYBACK_SPEED_MULTIPLIER}")
    print(f"MAX_OUTPUT_VIDEO_FPS      : {MAX_OUTPUT_VIDEO_FPS}")
    print(f"SUITE2P_DISPLAY_MODE      : {SUITE2P_DISPLAY_MODE}")
    print(f"CONTRAST_MODE             : {CONTRAST_MODE}")
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
