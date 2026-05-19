#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05e_make_roi_overlay_video.py

目的
----
把 ROI 结果叠加到 movie 上生成验证视频，帮助肉眼判断：
这些 ROI 到底是不是你看到在闪的地方。

当前支持两类 ROI：
1. pixelwise (05a 输出)
2. suite2p   (05b 输出)

默认输出两类视频：
- pixelwise_overlay.mp4
- suite2p_overlay.mp4

也可选输出并排对比视频：
- roi_compare_side_by_side.mp4

输入优先级
---------
movie:
1. *_spatial_highpass_movie.tif
2. *_brightness_corrected_movie.tif
3. *_corrected_movie.tif

ROI:
- benchmark/pixelwise/averaged/patch_label_map.tif
- benchmark/suite2p/roi_label_map.tif + roi_summary.csv

输出
----
trial_dir/benchmark/videos/
    pixelwise_overlay.mp4
    suite2p_overlay.mp4
    roi_compare_side_by_side.mp4   (可选)

说明
----
- ROI 用 outline，而不是填充，避免挡住细胞闪烁
- suite2p 默认只画 iscell=1
- 支持对比度增强
- 支持显示时间、frame、stim ON/OFF
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import imageio.v3 as iio
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
MAKE_PIXELWISE_VIDEO = True
MAKE_SUITE2P_VIDEO = True
MAKE_SIDE_BY_SIDE_VIDEO = True

PLAYBACK_SPEED_MULTIPLIER = 30.0
MAX_OUTPUT_VIDEO_FPS = 60.0
MAX_VIDEO_FRAMES = None           # None = 全部帧；可设整数限制输出长度（抽帧后）
VIDEO_PLUGIN = "pyav"
VIDEO_CODEC = "libx264"

# 画图
OUTLINE_WIDTH = 1
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
PIXELWISE_COLOR = (51, 230, 51)   # green
SUITE2P_COLOR = (255, 51, 51)     # red
SUITE2P_NONCELL_COLOR = (255, 140, 0)  # orange, only used when KEEP_SUITE2P_NONCELL = True

# suite2p 过滤
SUITE2P_ISCELL_ONLY = True
KEEP_SUITE2P_NONCELL = False

# 对比度增强
CONTRAST_PMIN = 1.0
CONTRAST_PMAX = 99.5
CONTRAST_SAMPLE_FRAMES = 200

SHOW_TIME_TEXT = True
SHOW_FRAME_TEXT = True
SHOW_STIM_STATUS = True
SHOW_TITLE_TEXT = True


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


def discover_trials(root_dir: Path):
    trial_dirs = set()
    for pattern in ["*_spatial_highpass_movie.tif", "*_brightness_corrected_movie.tif", "*_corrected_movie.tif"]:
        for p in root_dir.rglob(pattern):
            if pattern == "*_corrected_movie.tif" and "_brightness_" in p.name:
                continue
            trial_dirs.add(p.parent)

    trials = []
    for trial_dir in sorted(trial_dirs, key=lambda p: natural_key(str(p))):
        hp = sorted(trial_dir.glob("*_spatial_highpass_movie.tif"), key=lambda p: natural_key(p.name))
        bright = sorted(trial_dir.glob("*_brightness_corrected_movie.tif"), key=lambda p: natural_key(p.name))
        corr = sorted(trial_dir.glob("*_corrected_movie.tif"), key=lambda p: natural_key(p.name))
        corr = [p for p in corr if "_brightness_" not in p.name]

        if len(hp) > 0:
            movie_path = hp[0]
            prefix = movie_path.name.replace("_spatial_highpass_movie.tif", "").replace("_spatial_highpass_movie.tiff", "")
            movie_source = "spatial_highpass"
        elif len(bright) > 0:
            movie_path = bright[0]
            prefix = movie_path.name.replace("_brightness_corrected_movie.tif", "").replace("_brightness_corrected_movie.tiff", "")
            movie_source = "brightness_corrected"
        elif len(corr) > 0:
            movie_path = corr[0]
            prefix = movie_path.name.replace("_corrected_movie.tif", "").replace("_corrected_movie.tiff", "")
            movie_source = "corrected"
        else:
            continue

        trials.append({
            "trial_dir": trial_dir,
            "movie_path": movie_path,
            "prefix": prefix,
            "movie_source": movie_source,
        })
    return trials


def load_movie(movie_path: Path) -> np.ndarray:
    with tf.TiffFile(movie_path) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Movie must be 3D (T,Y,X), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_label_map(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = tf.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Label map must be 2D: {path}")
    return arr.astype(np.int32, copy=False)


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

    lo, hi = np.percentile(finite, [CONTRAST_PMIN, CONTRAST_PMAX])
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


def labelmap_to_outline(label_map: np.ndarray) -> np.ndarray:
    return binary_outline(label_map > 0)


def filter_suite2p_label_map(label_map: np.ndarray, summary_csv: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    返回：
    - filtered_label_map: 只保留 iscell=1（或全部，取决于配置）
    - noncell_label_map : 只保留 iscell=0（若 KEEP_SUITE2P_NONCELL=True，否则 None）
    """
    if not summary_csv.exists():
        return label_map, None

    df = pd.read_csv(summary_csv)
    if "roi_id" not in df.columns or "iscell" not in df.columns:
        return label_map, None

    keep_ids = set()
    noncell_ids = set()
    for _, row in df.iterrows():
        roi_id = int(row["roi_id"])
        iscell = int(row["iscell"])
        if iscell == 1:
            keep_ids.add(roi_id)
        else:
            noncell_ids.add(roi_id)

    if not SUITE2P_ISCELL_ONLY:
        filtered = label_map.copy()
    else:
        filtered = np.where(np.isin(label_map, list(keep_ids)), label_map, 0)

    noncell = None
    if KEEP_SUITE2P_NONCELL:
        noncell = np.where(np.isin(label_map, list(noncell_ids)), label_map, 0)

    return filtered.astype(np.int32, copy=False), None if noncell is None else noncell.astype(np.int32, copy=False)


def compute_output_fps_and_stride(actual_fps: float) -> tuple[float, int]:
    desired = max(float(actual_fps) * float(PLAYBACK_SPEED_MULTIPLIER), 1.0)
    if desired <= MAX_OUTPUT_VIDEO_FPS:
        return desired, 1
    stride = int(math.ceil(desired / float(MAX_OUTPUT_VIDEO_FPS)))
    output_fps = desired / stride
    return float(output_fps), max(1, stride)


def _frame_text_lines(title: str, t_sec: float, frame_idx: int, stim_df: pd.DataFrame | None):
    lines = []
    if SHOW_TITLE_TEXT:
        lines.append(title)
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


def draw_outline_rgb(base_rgb: np.ndarray, outline_mask: np.ndarray, color, width: int = 1) -> np.ndarray:
    out = base_rgb.copy()
    outline = outline_mask.astype(bool)
    if width > 1:
        outline = ndi.binary_dilation(outline, iterations=width - 1)
    out[outline] = np.array(color, dtype=np.uint8)
    return out


def render_overlay_frame(gray01: np.ndarray, outline_mask: np.ndarray, color, title: str,
                         t_sec: float, frame_idx: int, stim_df: pd.DataFrame | None,
                         extra_outline: np.ndarray | None = None, extra_color=None) -> np.ndarray:
    rgb = (np.dstack([gray01, gray01, gray01]) * 255).astype(np.uint8)
    rgb = draw_outline_rgb(rgb, outline_mask, color, OUTLINE_WIDTH)
    if extra_outline is not None and extra_color is not None:
        rgb = draw_outline_rgb(rgb, extra_outline, extra_color, OUTLINE_WIDTH)
    lines = _frame_text_lines(title, t_sec, frame_idx, stim_df)
    rgb = _draw_text_block(rgb, lines)
    return rgb


def write_video(out_mp4: Path, frames: list[np.ndarray], output_fps: float):
    writer = iio.imopen(str(out_mp4), "w", plugin=VIDEO_PLUGIN)
    writer.init_video_stream(VIDEO_CODEC, fps=output_fps)
    try:
        for frame in frames:
            writer.write_frame(frame)
    finally:
        writer.close()


def make_single_method_video(out_mp4: Path,
                             movie: np.ndarray,
                             outline_mask: np.ndarray,
                             color,
                             title: str,
                             actual_fps: float,
                             stim_df: pd.DataFrame | None,
                             extra_outline: np.ndarray | None = None,
                             extra_color=None):
    lo, hi = estimate_display_range(movie)
    output_fps, frame_stride = compute_output_fps_and_stride(actual_fps)
    frame_indices = list(range(0, movie.shape[0], frame_stride))
    if MAX_VIDEO_FRAMES is not None:
        frame_indices = frame_indices[: int(MAX_VIDEO_FRAMES)]

    frames = []
    for frame_idx in frame_indices:
        gray = normalize_frame_global(movie[frame_idx], lo, hi)
        t_sec = frame_idx / max(actual_fps, 1e-6)
        frame = render_overlay_frame(
            gray01=gray,
            outline_mask=outline_mask,
            color=color,
            title=title,
            t_sec=t_sec,
            frame_idx=frame_idx,
            stim_df=stim_df,
            extra_outline=extra_outline,
            extra_color=extra_color,
        )
        frames.append(frame)

    write_video(out_mp4, frames, output_fps)
    return {"output_fps": output_fps, "frame_stride": frame_stride, "n_frames_written": len(frames)}


def make_side_by_side_video(out_mp4: Path,
                            movie: np.ndarray,
                            pw_outline: np.ndarray,
                            s2p_outline: np.ndarray,
                            actual_fps: float,
                            stim_df: pd.DataFrame | None,
                            s2p_extra_outline: np.ndarray | None = None):
    lo, hi = estimate_display_range(movie)
    output_fps, frame_stride = compute_output_fps_and_stride(actual_fps)
    frame_indices = list(range(0, movie.shape[0], frame_stride))
    if MAX_VIDEO_FRAMES is not None:
        frame_indices = frame_indices[: int(MAX_VIDEO_FRAMES)]

    frames = []
    for frame_idx in frame_indices:
        gray = normalize_frame_global(movie[frame_idx], lo, hi)
        t_sec = frame_idx / max(actual_fps, 1e-6)

        left = render_overlay_frame(
            gray01=gray,
            outline_mask=pw_outline,
            color=PIXELWISE_COLOR,
            title="pixelwise",
            t_sec=t_sec,
            frame_idx=frame_idx,
            stim_df=stim_df,
        )

        right = render_overlay_frame(
            gray01=gray,
            outline_mask=s2p_outline,
            color=SUITE2P_COLOR,
            title="suite2p",
            t_sec=t_sec,
            frame_idx=frame_idx,
            stim_df=stim_df,
            extra_outline=s2p_extra_outline,
            extra_color=SUITE2P_NONCELL_COLOR,
        )

        canvas = np.concatenate([left, right], axis=1)
        frames.append(canvas)

    write_video(out_mp4, frames, output_fps)
    return {"output_fps": output_fps, "frame_stride": frame_stride, "n_frames_written": len(frames)}


# ============================================================
# 主处理
# ============================================================
def process_single_trial(info: dict):
    trial_dir = Path(info["trial_dir"])
    prefix = info["prefix"]
    movie_path = Path(info["movie_path"])
    movie_source = info["movie_source"]

    benchmark_dir = trial_dir / "benchmark"
    videos_dir = benchmark_dir / "videos"
    safe_makedirs(videos_dir)

    if EXISTING_MODE == "overwrite":
        clear_dir_contents(videos_dir)
        safe_makedirs(videos_dir)

    pixelwise_label_path = benchmark_dir / "pixelwise" / "averaged" / "patch_label_map.tif"
    suite2p_label_path = benchmark_dir / "suite2p" / "roi_label_map.tif"
    suite2p_summary_path = benchmark_dir / "suite2p" / "roi_summary.csv"

    has_pixelwise = pixelwise_label_path.exists()
    has_suite2p = suite2p_label_path.exists()

    if not has_pixelwise and not has_suite2p:
        print(f"[SKIP] {trial_dir.name}: no ROI outputs found")
        return False

    print(f"\n=== Trial: {trial_dir.name} ===")
    print(f"    movie: {movie_path.name}")
    print(f"    source: {movie_source}")

    movie = load_movie(movie_path)
    fps = get_trial_fps(trial_dir, prefix)
    stim_df = load_stim_events(trial_dir, prefix)

    pw_outline = None
    if has_pixelwise:
        pw_label = load_label_map(pixelwise_label_path)
        pw_outline = labelmap_to_outline(pw_label)

    s2p_outline = None
    s2p_noncell_outline = None
    if has_suite2p:
        s2p_label = load_label_map(suite2p_label_path)
        s2p_label, s2p_noncell = filter_suite2p_label_map(s2p_label, suite2p_summary_path)
        s2p_outline = labelmap_to_outline(s2p_label)
        if s2p_noncell is not None:
            s2p_noncell_outline = labelmap_to_outline(s2p_noncell)

    if MAKE_PIXELWISE_VIDEO and pw_outline is not None:
        out_mp4 = videos_dir / "pixelwise_overlay.mp4"
        info_vid = make_single_method_video(
            out_mp4=out_mp4,
            movie=movie,
            outline_mask=pw_outline,
            color=PIXELWISE_COLOR,
            title="pixelwise",
            actual_fps=fps,
            stim_df=stim_df,
        )
        print(f"    saved: {out_mp4.name} ({info_vid})")

    if MAKE_SUITE2P_VIDEO and s2p_outline is not None:
        out_mp4 = videos_dir / "suite2p_overlay.mp4"
        info_vid = make_single_method_video(
            out_mp4=out_mp4,
            movie=movie,
            outline_mask=s2p_outline,
            color=SUITE2P_COLOR,
            title="suite2p",
            actual_fps=fps,
            stim_df=stim_df,
            extra_outline=s2p_noncell_outline,
            extra_color=SUITE2P_NONCELL_COLOR if s2p_noncell_outline is not None else None,
        )
        print(f"    saved: {out_mp4.name} ({info_vid})")

    if MAKE_SIDE_BY_SIDE_VIDEO and pw_outline is not None and s2p_outline is not None:
        out_mp4 = videos_dir / "roi_compare_side_by_side.mp4"
        info_vid = make_side_by_side_video(
            out_mp4=out_mp4,
            movie=movie,
            pw_outline=pw_outline,
            s2p_outline=s2p_outline,
            actual_fps=fps,
            stim_df=stim_df,
            s2p_extra_outline=s2p_noncell_outline,
        )
        print(f"    saved: {out_mp4.name} ({info_vid})")

    return True


def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 72)
    print("05e ROI overlay video")
    print("=" * 72)
    print(f"ROOT_DIR: {root_dir}")

    trials = discover_trials(root_dir)
    print(f"Found {len(trials)} trial(s) with movie inputs")

    ok_n = 0
    fail_n = 0
    for info in trials:
        try:
            ok = process_single_trial(info)
            if ok:
                ok_n += 1
            else:
                fail_n += 1
        except Exception as e:
            fail_n += 1
            print(f"[FAILED] {info['trial_dir']}: {e}")

    print("\n" + "=" * 72)
    print(f"Done. success={ok_n}, failed={fail_n}")
    print("=" * 72)


if __name__ == "__main__":
    main()