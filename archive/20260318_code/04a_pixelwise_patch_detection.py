#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04a_pixelwise_patch_detection.py

目的
----
在 motion-corrected movie 上，基于刺激前后时间窗做 pixel-wise response map，
同时输出：
1. averaged：跨全部有效刺激平均后的响应图与 patch
2. per_stim：每次刺激单独的响应图与 patch

输出目录：
trial_dir/
└── benchmark/
    └── pixelwise/
        ├── averaged/
        │   ├── onset_response_map.tif
        │   ├── sustained_response_map.tif
        │   ├── offset_response_map.tif
        │   ├── patch_mask.tif
        │   ├── patch_label_map.tif
        │   ├── patch_summary.csv
        │   └── patch_overlay.png
        └── per_stim/
            ├── stim_001/
            │   ├── onset_response_map.tif
            │   ├── sustained_response_map.tif
            │   ├── offset_response_map.tif
            │   ├── patch_mask.tif
            │   ├── patch_label_map.tif
            │   ├── patch_summary.csv
            │   └── patch_overlay.png
            └── ...
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

# ============================================================
# 全局参数区
# ============================================================
ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "overwrite"   # "skip" / "overwrite"

# 时间窗（秒）
BASELINE_SEC = 1.5
ONSET_SEC = 1.0
OFFSET_SEC = 1.0
MIN_SUSTAINED_FRAMES = 1

# 响应图计算方式："delta" / "dff" / "zscore"
RESPONSE_MODE = "zscore"

# averaged patch 检测参数
THRESHOLD_STD = 3.0
MIN_PATCH_AREA = 15
MAX_PATCH_AREA = 5000
USE_ABS_RESPONSE = False

# per-stim 检测通常更噪，默认更严格一点
SAVE_PER_STIM = True
PER_STIM_THRESHOLD_STD = 3.5
PER_STIM_MIN_PATCH_AREA = 20
PER_STIM_MAX_PATCH_AREA = 5000
PER_STIM_USE_ABS_RESPONSE = False

BINARY_OPENING_ITER = 1
BINARY_CLOSING_ITER = 1

OVERLAY_FIGSIZE = (8, 8)
OVERLAY_DPI = 150
MAX_PATCHES_TO_ANNOTATE = 150


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


def get_trial_metadata(json_path: Path):
    meta = {"fs": 1.0, "pixel_size_um": None, "json_found": False}
    if json_path is None or (not json_path.exists()):
        return meta

    meta["json_found"] = True
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fps = data.get("temporal_calibration", {}).get("fps", None)
        meta["fs"] = float(fps) if fps is not None else 1.0

        physical = data.get("physical_size", {})
        dims = data.get("dimensions", {})

        # 修复 #13 联动：优先读新字段 pixel_size_um（01 修复后直接写入），
        # 回退到旧字段 fov_width_um / width 反推（向后兼容）
        if physical.get("pixel_size_um") is not None:
            meta["pixel_size_um"] = float(physical["pixel_size_um"])
        else:
            fov_width_um = physical.get("fov_width_um", physical.get("width", None))
            width_px = dims.get("width_pixel", None)
            if fov_width_um is not None and width_px not in (None, 0):
                meta["pixel_size_um"] = float(fov_width_um) / float(width_px)
    except Exception as e:
        print(f"    [Warning] Failed to read metadata: {json_path.name} ({e})")
    return meta


def validate_stim_events(stim_df: pd.DataFrame) -> pd.DataFrame:
    required = ["start_time_sec", "end_time_sec"]
    for c in required:
        if c not in stim_df.columns:
            raise ValueError(f"stim_events.csv missing required column: {c}")

    stim_df = stim_df.copy()
    stim_df["start_time_sec"] = pd.to_numeric(stim_df["start_time_sec"], errors="coerce")
    stim_df["end_time_sec"] = pd.to_numeric(stim_df["end_time_sec"], errors="coerce")
    stim_df = stim_df.dropna(subset=["start_time_sec", "end_time_sec"])
    stim_df = stim_df[stim_df["end_time_sec"] > stim_df["start_time_sec"]]
    stim_df = stim_df.sort_values("start_time_sec").reset_index(drop=True)
    return stim_df


def find_trial_inputs(trial_dir: Path):
    movie_files = sorted(trial_dir.glob("*_corrected_movie.tif"), key=lambda p: natural_key(p.name))
    stim_files = sorted(trial_dir.glob("*_stim_events.csv"), key=lambda p: natural_key(p.name))
    json_files = sorted(trial_dir.glob("*_metadata.json"), key=lambda p: natural_key(p.name))

    if len(movie_files) == 0 or len(stim_files) == 0:
        return None

    movie_path = movie_files[0]
    stim_path = stim_files[0]
    json_path = json_files[0] if len(json_files) > 0 else None
    prefix = movie_path.name.replace("_corrected_movie.tif", "")

    return {
        "trial_dir": trial_dir,
        "prefix": prefix,
        "movie_path": movie_path,
        "stim_path": stim_path,
        "json_path": json_path,
    }


def discover_trials(root_dir: Path):
    movie_files = sorted(root_dir.rglob("*_corrected_movie.tif"), key=lambda p: natural_key(str(p)))
    trials = []
    for movie_path in movie_files:
        inputs = find_trial_inputs(movie_path.parent)
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


def save_float32_tiff(path: Path, image: np.ndarray):
    tf.imwrite(str(path), np.asarray(image, dtype=np.float32), imagej=True)


def save_uint16_tiff(path: Path, image: np.ndarray):
    tf.imwrite(str(path), np.asarray(image, dtype=np.uint16), imagej=True)


def save_uint8_tiff(path: Path, image: np.ndarray):
    tf.imwrite(str(path), np.asarray(image, dtype=np.uint8), imagej=True)


def sec_to_frame(sec: float, fps: float) -> int:
    return int(round(float(sec) * float(fps)))


def robust_std(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad)


def noise_sigma_from_subthreshold(combined_map: np.ndarray) -> float:
    """
    修复 #7：从 combined_map 的"非响应区域"估计噪声 sigma。

    问题背景
    --------
    原版对整张 combined_map 做 MAD，但 combined_map = nanmax(onset, sustained, offset)
    是偏态分布（大多数像素接近 0，少数响应像素值很大）。
    对偏态分布做 MAD 会得到极小的 sigma（因为中位数附近的 MAD 很小），
    导致 threshold = N * sigma 过低，检出大量假阳性 patch。

    修复策略
    --------
    只用低于中位数的像素（即"非响应"的背景像素）来估计噪声 sigma。
    这些像素的分布近似对称，MAD 估计是可靠的。
    等价于：sigma ≈ MAD of (map[map <= median(map)])，再乘以 1.4826 换算为 std。

    如果子阈值像素不足（极端情况），回退到全图 std。
    """
    arr = np.asarray(combined_map, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0

    med = float(np.median(finite))
    # 只取低于中位数的像素作为"背景"
    background = finite[finite <= med]

    if background.size < 10:
        # 背景像素太少，回退到全图 std
        sigma = float(np.std(finite))
        return max(sigma, 1e-6)

    mad = float(np.median(np.abs(background - med)))
    sigma = 1.4826 * mad

    if sigma <= 0:
        sigma = float(np.std(finite))

    return max(float(sigma), 1e-6)


def frame_window_to_mean(movie: np.ndarray, start: int, end: int):
    n_frames = movie.shape[0]
    start = max(0, min(int(start), n_frames))
    end = max(0, min(int(end), n_frames))
    if end <= start:
        return None
    return np.mean(movie[start:end], axis=0, dtype=np.float32)


def compute_response_map(base_img: np.ndarray, win_img: np.ndarray, base_std_img: np.ndarray, mode: str):
    if base_img is None or win_img is None:
        return None
    eps = 1e-6
    delta = win_img - base_img
    if mode == "delta":
        return delta
    if mode == "dff":
        denom = np.maximum(np.abs(base_img), eps)
        return delta / denom
    if mode == "zscore":
        if base_std_img is None:
            return None
        denom = np.maximum(base_std_img, eps)
        return delta / denom
    raise ValueError(f"Unknown RESPONSE_MODE: {mode}")


def compute_single_event_maps(movie: np.ndarray, fps: float, start_sec: float, end_sec: float):
    n_frames = movie.shape[0]
    start_frame = max(0, min(sec_to_frame(start_sec, fps), n_frames))
    end_frame = max(0, min(sec_to_frame(end_sec, fps), n_frames))
    if end_frame <= start_frame:
        return None

    baseline_len = max(1, sec_to_frame(BASELINE_SEC, fps))
    onset_len = max(1, sec_to_frame(ONSET_SEC, fps))
    offset_len = max(1, sec_to_frame(OFFSET_SEC, fps))

    base_start = max(0, start_frame - baseline_len)
    base_end = start_frame
    onset_start = start_frame
    onset_end = min(n_frames, start_frame + onset_len)
    sustained_start = onset_end
    sustained_end = end_frame
    offset_start = end_frame
    offset_end = min(n_frames, end_frame + offset_len)

    if base_end <= base_start:
        return None

    baseline_stack = movie[base_start:base_end]
    base_img = np.mean(baseline_stack, axis=0, dtype=np.float32)
    base_std_img = np.std(baseline_stack, axis=0, dtype=np.float32)

    onset_img = frame_window_to_mean(movie, onset_start, onset_end)
    sustained_img = None
    if (sustained_end - sustained_start) >= MIN_SUSTAINED_FRAMES:
        sustained_img = frame_window_to_mean(movie, sustained_start, sustained_end)
    offset_img = frame_window_to_mean(movie, offset_start, offset_end)

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "base_start": base_start,
        "base_end": base_end,
        "onset_start": onset_start,
        "onset_end": onset_end,
        "sustained_start": sustained_start,
        "sustained_end": sustained_end,
        "offset_start": offset_start,
        "offset_end": offset_end,
        "onset_map": compute_response_map(base_img, onset_img, base_std_img, RESPONSE_MODE),
        "sustained_map": compute_response_map(base_img, sustained_img, base_std_img, RESPONSE_MODE),
        "offset_map": compute_response_map(base_img, offset_img, base_std_img, RESPONSE_MODE),
    }


def nanmean_stack(arrays, shape_2d):
    valid = [a for a in arrays if a is not None]
    if len(valid) == 0:
        return np.full(shape_2d, np.nan, dtype=np.float32)
    stack = np.stack(valid, axis=0).astype(np.float32, copy=False)
    return np.nanmean(stack, axis=0).astype(np.float32, copy=False)


def build_combined_detection_map(onset_map: np.ndarray, sustained_map: np.ndarray, offset_map: np.ndarray, use_abs: bool):
    maps = []
    for m in [onset_map, sustained_map, offset_map]:
        if m is None:
            continue
        maps.append(np.abs(m) if use_abs else m)
    if len(maps) == 0:
        raise ValueError("No valid response maps available for detection.")
    stack = np.stack(maps, axis=0)
    combined = np.nanmax(stack, axis=0)
    return combined.astype(np.float32, copy=False)


def detect_patches_from_map(combined_map: np.ndarray,
                            threshold_std: float,
                            min_area: int,
                            max_area: int):
    finite_vals = combined_map[np.isfinite(combined_map)]
    if finite_vals.size == 0:
        shape = combined_map.shape
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.int32), 0.0

    # 修复 #7：用子阈值像素估计噪声 sigma，而非全图 MAD
    # 原版对偏态 combined_map 做全图 MAD，sigma 过小导致阈值过低、假阳性过多
    sigma = noise_sigma_from_subthreshold(combined_map)

    threshold = float(threshold_std) * sigma
    binary = combined_map > threshold

    if BINARY_OPENING_ITER > 0:
        binary = ndi.binary_opening(binary, iterations=BINARY_OPENING_ITER)
    if BINARY_CLOSING_ITER > 0:
        binary = ndi.binary_closing(binary, iterations=BINARY_CLOSING_ITER)

    label_map, n_labels = ndi.label(binary)
    if n_labels == 0:
        shape = combined_map.shape
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.int32), threshold

    out_label = np.zeros_like(label_map, dtype=np.int32)
    new_id = 0
    for lab in range(1, n_labels + 1):
        mask = label_map == lab
        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue
        new_id += 1
        out_label[mask] = new_id

    return out_label > 0, out_label, threshold


def summarize_patches(label_map: np.ndarray,
                      onset_map: np.ndarray,
                      sustained_map: np.ndarray,
                      offset_map: np.ndarray,
                      combined_map: np.ndarray,
                      pixel_size_um: float = None):
    rows = []
    n_patches = int(label_map.max())
    for patch_id in range(1, n_patches + 1):
        mask = label_map == patch_id
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        area_px = int(mask.sum())
        centroid_y = float(np.mean(ys))
        centroid_x = float(np.mean(xs))

        row = {
            "patch_id": patch_id,
            "centroid_x": round(centroid_x, 3),
            "centroid_y": round(centroid_y, 3),
            "area_px": area_px,
            "bbox_xmin": int(xs.min()),
            "bbox_xmax": int(xs.max()),
            "bbox_ymin": int(ys.min()),
            "bbox_ymax": int(ys.max()),
            "mean_onset_response": float(np.nanmean(onset_map[mask])) if onset_map is not None else np.nan,
            "max_onset_response": float(np.nanmax(onset_map[mask])) if onset_map is not None else np.nan,
            "mean_sustained_response": float(np.nanmean(sustained_map[mask])) if sustained_map is not None else np.nan,
            "max_sustained_response": float(np.nanmax(sustained_map[mask])) if sustained_map is not None else np.nan,
            "mean_offset_response": float(np.nanmean(offset_map[mask])) if offset_map is not None else np.nan,
            "max_offset_response": float(np.nanmax(offset_map[mask])) if offset_map is not None else np.nan,
            "mean_combined_response": float(np.nanmean(combined_map[mask])),
            "max_combined_response": float(np.nanmax(combined_map[mask])),
        }
        row["area_um2"] = float(area_px * (pixel_size_um ** 2)) if pixel_size_um is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["max_combined_response", "area_px"], ascending=[False, False]).reset_index(drop=True)
    return df


def make_overlay_figure(mean_img: np.ndarray,
                        label_map: np.ndarray,
                        patch_summary: pd.DataFrame,
                        title: str,
                        out_png: Path):
    fig, ax = plt.subplots(figsize=OVERLAY_FIGSIZE)
    ax.imshow(mean_img, cmap="gray")

    contours = np.zeros_like(label_map, dtype=bool)
    eroded = ndi.binary_erosion(label_map > 0)
    contours[(label_map > 0) & (~eroded)] = True
    yy, xx = np.where(contours)
    ax.scatter(xx, yy, s=1, c="lime", alpha=0.9)

    if patch_summary is not None and not patch_summary.empty:
        for _, row in patch_summary.head(MAX_PATCHES_TO_ANNOTATE).iterrows():
            x0 = row["bbox_xmin"]
            y0 = row["bbox_ymin"]
            w = row["bbox_xmax"] - row["bbox_xmin"] + 1
            h = row["bbox_ymax"] - row["bbox_ymin"] + 1
            rect = Rectangle((x0, y0), w, h, fill=False, edgecolor="yellow", linewidth=0.6, alpha=0.8)
            ax.add_patch(rect)
            ax.text(row["centroid_x"], row["centroid_y"], str(int(row["patch_id"])),
                    color="yellow", fontsize=6, ha="center", va="center")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=OVERLAY_DPI, bbox_inches="tight")
    plt.close(fig)


def write_empty_outputs(out_dir: Path, shape_hw, mean_img: np.ndarray, title: str):
    h, w = shape_hw
    empty = np.zeros((h, w), dtype=np.float32)
    save_float32_tiff(out_dir / "onset_response_map.tif", empty)
    save_float32_tiff(out_dir / "sustained_response_map.tif", empty)
    save_float32_tiff(out_dir / "offset_response_map.tif", empty)
    save_uint8_tiff(out_dir / "patch_mask.tif", np.zeros((h, w), dtype=np.uint8))
    save_uint16_tiff(out_dir / "patch_label_map.tif", np.zeros((h, w), dtype=np.uint16))
    pd.DataFrame().to_csv(out_dir / "patch_summary.csv", index=False)
    make_overlay_figure(mean_img, np.zeros((h, w), dtype=np.int32), pd.DataFrame(), title, out_dir / "patch_overlay.png")


def _map_or_zeros(m, shape_hw) -> np.ndarray:
    """
    修复 #1：per_stim 模式下 sustained_map 可能为 None
    （刺激时长 < MIN_SUSTAINED_FRAMES 时 compute_single_event_maps 返回 None）。
    save_patch_bundle 直接对 None 调用 np.nan_to_num 会报 AttributeError。
    此辅助函数统一处理：None → 全零数组，ndarray → nan_to_num。
    """
    if m is None:
        return np.zeros(shape_hw, dtype=np.float32)
    return np.nan_to_num(np.asarray(m, dtype=np.float32), nan=0.0)


def save_patch_bundle(out_dir: Path,
                      onset_map: np.ndarray,
                      sustained_map: np.ndarray,
                      offset_map: np.ndarray,
                      patch_mask: np.ndarray,
                      patch_label_map: np.ndarray,
                      patch_summary: pd.DataFrame,
                      mean_img: np.ndarray,
                      title: str):
    safe_makedirs(out_dir)
    shape_hw = mean_img.shape[:2]
    save_float32_tiff(out_dir / "onset_response_map.tif",     _map_or_zeros(onset_map,     shape_hw))
    save_float32_tiff(out_dir / "sustained_response_map.tif", _map_or_zeros(sustained_map, shape_hw))
    save_float32_tiff(out_dir / "offset_response_map.tif",    _map_or_zeros(offset_map,    shape_hw))
    save_uint8_tiff(out_dir / "patch_mask.tif", patch_mask.astype(np.uint8) * 255)

    max_label = int(np.max(patch_label_map)) if patch_label_map.size > 0 else 0
    if max_label <= np.iinfo(np.uint16).max:
        save_uint16_tiff(out_dir / "patch_label_map.tif", patch_label_map.astype(np.uint16))
    else:
        tf.imwrite(str(out_dir / "patch_label_map.tif"), patch_label_map.astype(np.uint32), imagej=False)

    patch_summary.to_csv(out_dir / "patch_summary.csv", index=False)
    make_overlay_figure(mean_img, patch_label_map, patch_summary, title, out_dir / "patch_overlay.png")


def run_detection_for_maps(onset_map, sustained_map, offset_map,
                           pixel_size_um,
                           threshold_std,
                           min_area,
                           max_area,
                           use_abs,
                           extra_fields: dict):
    combined_map = build_combined_detection_map(onset_map, sustained_map, offset_map, use_abs=use_abs)
    patch_mask, patch_label_map, threshold = detect_patches_from_map(
        combined_map,
        threshold_std=threshold_std,
        min_area=min_area,
        max_area=max_area,
    )
    patch_summary = summarize_patches(
        label_map=patch_label_map,
        onset_map=onset_map,
        sustained_map=sustained_map,
        offset_map=offset_map,
        combined_map=combined_map,
        pixel_size_um=pixel_size_um,
    )
    if patch_summary is None or patch_summary.empty:
        patch_summary = pd.DataFrame(columns=[
            "patch_id", "centroid_x", "centroid_y", "area_px", "bbox_xmin", "bbox_xmax",
            "bbox_ymin", "bbox_ymax", "mean_onset_response", "max_onset_response",
            "mean_sustained_response", "max_sustained_response", "mean_offset_response",
            "max_offset_response", "mean_combined_response", "max_combined_response", "area_um2"
        ])
    for k, v in extra_fields.items():
        patch_summary[k] = v
    patch_summary["threshold_std"] = threshold_std
    patch_summary["threshold_value"] = threshold
    patch_summary["use_abs_response"] = use_abs
    return patch_mask, patch_label_map, patch_summary, threshold


# ============================================================
# 单个 trial 主流程
# ============================================================

def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    prefix = inputs["prefix"]
    movie_path = Path(inputs["movie_path"])
    stim_path = Path(inputs["stim_path"])
    json_path = Path(inputs["json_path"]) if inputs["json_path"] is not None else None

    pixelwise_root = trial_dir / "benchmark" / "pixelwise"
    averaged_dir = pixelwise_root / "averaged"
    per_stim_root = pixelwise_root / "per_stim"

    if pixelwise_root.exists() and EXISTING_MODE == "skip":
        print(f"[SKIP] {trial_dir.name} -> benchmark/pixelwise already exists")
        return True
    if pixelwise_root.exists() and EXISTING_MODE == "overwrite":
        clear_dir_contents(pixelwise_root)

    safe_makedirs(averaged_dir)
    if SAVE_PER_STIM:
        safe_makedirs(per_stim_root)

    print(f"\n=== Trial: {trial_dir.name} ===")
    print(f"    movie: {movie_path.name}")
    print(f"    stim : {stim_path.name}")
    print(f"    meta : {json_path.name if json_path is not None else 'NOT FOUND'}")

    meta = get_trial_metadata(json_path) if json_path is not None else {"fs": 1.0, "pixel_size_um": None}
    fps = float(meta.get("fs", 1.0))
    pixel_size_um = meta.get("pixel_size_um", None)
    print(f"    fps  : {fps:.4f}")
    if pixel_size_um is not None:
        print(f"    px   : {pixel_size_um:.4f} um/px")

    movie = load_movie(movie_path)
    n_frames, h, w = movie.shape
    mean_img = np.mean(movie, axis=0, dtype=np.float32)
    print(f"    movie shape: T={n_frames}, Y={h}, X={w}")

    stim_df = validate_stim_events(pd.read_csv(stim_path))
    if stim_df.empty:
        print("    [Info] No valid stim events. Writing empty outputs.")
        write_empty_outputs(averaged_dir, (h, w), mean_img, f"Pixelwise averaged - {trial_dir.name}")
        if SAVE_PER_STIM:
            safe_makedirs(per_stim_root)
        return True

    onset_maps = []
    sustained_maps = []
    offset_maps = []
    valid_events = []

    for stim_idx, row in stim_df.iterrows():
        out = compute_single_event_maps(movie, fps, float(row["start_time_sec"]), float(row["end_time_sec"]))
        if out is None:
            continue

        valid_events.append((stim_idx + 1, row, out))
        onset_maps.append(out["onset_map"])
        sustained_maps.append(out["sustained_map"])
        offset_maps.append(out["offset_map"])

        if SAVE_PER_STIM:
            stim_dir = per_stim_root / f"stim_{stim_idx + 1:03d}"
            patch_mask, patch_label_map, patch_summary, threshold = run_detection_for_maps(
                onset_map=out["onset_map"],
                sustained_map=out["sustained_map"],
                offset_map=out["offset_map"],
                pixel_size_um=pixel_size_um,
                threshold_std=PER_STIM_THRESHOLD_STD,
                min_area=PER_STIM_MIN_PATCH_AREA,
                max_area=PER_STIM_MAX_PATCH_AREA,
                use_abs=PER_STIM_USE_ABS_RESPONSE,
                extra_fields={
                    "trial": trial_dir.name,
                    "prefix": prefix,
                    "output_mode": "per_stim",
                    "stim_id": stim_idx + 1,
                    "stim_start_sec": float(row["start_time_sec"]),
                    "stim_end_sec": float(row["end_time_sec"]),
                    "response_mode": RESPONSE_MODE,
                },
            )
            save_patch_bundle(
                stim_dir,
                out["onset_map"],
                out["sustained_map"],
                out["offset_map"],
                patch_mask,
                patch_label_map,
                patch_summary,
                mean_img,
                f"Pixelwise per-stim {stim_idx + 1:03d} - {trial_dir.name}",
            )
            print(f"    per_stim stim_{stim_idx + 1:03d}: threshold={threshold:.4f}, n_patches={int(patch_label_map.max())}")

    if len(valid_events) == 0:
        raise RuntimeError("No valid stimulus events remained after frame-window conversion.")

    onset_map = nanmean_stack(onset_maps, (h, w))
    sustained_map = nanmean_stack(sustained_maps, (h, w))
    offset_map = nanmean_stack(offset_maps, (h, w))

    patch_mask, patch_label_map, patch_summary, threshold = run_detection_for_maps(
        onset_map=onset_map,
        sustained_map=sustained_map,
        offset_map=offset_map,
        pixel_size_um=pixel_size_um,
        threshold_std=THRESHOLD_STD,
        min_area=MIN_PATCH_AREA,
        max_area=MAX_PATCH_AREA,
        use_abs=USE_ABS_RESPONSE,
        extra_fields={
            "trial": trial_dir.name,
            "prefix": prefix,
            "output_mode": "averaged",
            "n_valid_stim_used": len(valid_events),
            "response_mode": RESPONSE_MODE,
        },
    )

    save_patch_bundle(
        averaged_dir,
        onset_map,
        sustained_map,
        offset_map,
        patch_mask,
        patch_label_map,
        patch_summary,
        mean_img,
        f"Pixelwise averaged - {trial_dir.name}",
    )

    print(f"    averaged threshold : {threshold:.4f}")
    print(f"    averaged n_patches : {int(patch_label_map.max())}")
    print(f"    saved to           : {pixelwise_root}")
    return True


# ============================================================
# main
# ============================================================

def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("04a Pixelwise patch detection")
    print("=" * 70)
    print(f"ROOT_DIR                 : {root_dir}")
    print(f"EXISTING_MODE            : {EXISTING_MODE}")
    print(f"RESPONSE_MODE            : {RESPONSE_MODE}")
    print(f"BASELINE_SEC             : {BASELINE_SEC}")
    print(f"ONSET_SEC                : {ONSET_SEC}")
    print(f"OFFSET_SEC               : {OFFSET_SEC}")
    print(f"THRESHOLD_STD            : {THRESHOLD_STD}")
    print(f"MIN_PATCH_AREA           : {MIN_PATCH_AREA}")
    print(f"SAVE_PER_STIM            : {SAVE_PER_STIM}")
    print(f"PER_STIM_THRESHOLD_STD   : {PER_STIM_THRESHOLD_STD}")
    print(f"PER_STIM_MIN_PATCH_AREA  : {PER_STIM_MIN_PATCH_AREA}")
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
