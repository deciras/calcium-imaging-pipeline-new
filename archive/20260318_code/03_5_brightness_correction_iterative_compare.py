#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_5_brightness_correction_three_modes.py

目的
----
在 motion-corrected movie 上做全局亮度校正原型，并同时输出多种校正版 movie，
便于人工比较哪种最适合当前数据：

1. subtract_mean   : 每帧减去全图均值，再加参考亮度
2. subtract_median : 每帧减去全图中位数，再加参考亮度
3. divide_mean_1   : 全图均值除法校正，迭代 1 次
4. divide_mean_2   : 全图均值除法校正，迭代 2 次
5. divide_mean_3   : 全图均值除法校正，迭代 3 次

同时输出：
- 全局亮度轨迹
- before/after 预览图
- compare_all_modes.mp4：把 raw + 所有校正版放在一起对比

输入
----
trial_dir/
    *_corrected_movie.tif
    *_metadata.json
    *_stim_events.csv   (可选)

输出
----
trial_dir/
├── *_global_brightness_trace.csv
├── *_global_brightness_trace.png
├── *_brightness_corrected_subtract_mean_movie.tif
├── *_brightness_corrected_subtract_median_movie.tif
├── *_brightness_corrected_divide_mean1_movie.tif
├── *_brightness_corrected_divide_mean2_movie.tif
├── *_brightness_corrected_divide_mean3_movie.tif
└── brightness_correction/
    ├── preview_subtract_mean.png
    ├── preview_subtract_median.png
    ├── preview_divide_mean_1.png
    ├── preview_divide_mean_2.png
    ├── preview_divide_mean_3.png
    ├── compare_all_modes.mp4
    └── correction_summary.json
"""

import json
import re
import shutil
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf

# ============================================================
# 全局参数
# ============================================================
ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
EXISTING_MODE = "overwrite"   # "skip" / "overwrite"

# 保存 float32，避免校正后动态范围丢失
SAVE_FLOAT32 = True

# 参考亮度如何取
# "prestim_mean" / "prestim_median" / "global_mean" / "global_median"
REFERENCE_MODE = "prestim_mean"

FIG_DPI = 150
PREVIEW_FIGSIZE = (10, 8)

# 预览图里选第几个刺激来展示局部帧（0-based）
PREVIEW_STIM_INDEX = 0
PREVIEW_EXTRA_SEC = 3.0

# 对比视频
MAKE_COMPARISON_VIDEO = True
PLAYBACK_SPEED_MULTIPLIER = 20.0
MAX_OUTPUT_VIDEO_FPS = 60.0
VIDEO_CONTRAST_PMIN = 1.0
VIDEO_CONTRAST_PMAX = 99.5
VIDEO_TEXT_COLOR = (255, 255, 0)

OUT_SUFFIXES = {
    "subtract_mean": "_brightness_corrected_subtract_mean_movie.tif",
    "subtract_median": "_brightness_corrected_subtract_median_movie.tif",
    "divide_mean_1": "_brightness_corrected_divide_mean1_movie.tif",
    "divide_mean_2": "_brightness_corrected_divide_mean2_movie.tif",
    "divide_mean_3": "_brightness_corrected_divide_mean3_movie.tif",
}


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


def find_trial_inputs(trial_dir: Path):
    movie_files = sorted(trial_dir.glob("*_corrected_movie.tif"), key=lambda p: natural_key(p.name))
    json_files = sorted(trial_dir.glob("*_metadata.json"), key=lambda p: natural_key(p.name))
    stim_files = sorted(trial_dir.glob("*_stim_events.csv"), key=lambda p: natural_key(p.name))

    if len(movie_files) == 0:
        return None

    movie_path = movie_files[0]
    json_path = json_files[0] if json_files else None
    stim_path = stim_files[0] if stim_files else None
    prefix = movie_path.name.replace("_corrected_movie.tif", "")

    return {
        "trial_dir": trial_dir,
        "prefix": prefix,
        "movie_path": movie_path,
        "json_path": json_path,
        "stim_path": stim_path,
    }


def discover_trials(root_dir: Path):
    movie_files = sorted(root_dir.rglob("*_corrected_movie.tif"), key=lambda p: natural_key(str(p)))
    trials = []
    seen = set()
    for movie_path in movie_files:
        name = movie_path.name
        # 显式排除误生成的 brightness 版本
        if "_brightness_" in name:
            continue
        if movie_path.parent in seen:
            continue
        inputs = find_trial_inputs(movie_path.parent)
        if inputs is not None:
            trials.append(inputs)
            seen.add(movie_path.parent)
    return trials


def get_trial_metadata(json_path: Path | None):
    meta = {"fs": 1.0, "json_found": False}
    if json_path is None or (not json_path.exists()):
        return meta

    meta["json_found"] = True
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fps = data.get("temporal_calibration", {}).get("fps", None)
        meta["fs"] = float(fps) if fps is not None else 1.0
    except Exception as e:
        print(f"    [Warning] Failed to read metadata: {json_path.name} ({e})")
    return meta


def validate_stim_events(stim_df: pd.DataFrame | None):
    if stim_df is None:
        return None
    required = ["start_time_sec", "end_time_sec"]
    for c in required:
        if c not in stim_df.columns:
            return None

    stim_df = stim_df.copy()
    stim_df["start_time_sec"] = pd.to_numeric(stim_df["start_time_sec"], errors="coerce")
    stim_df["end_time_sec"] = pd.to_numeric(stim_df["end_time_sec"], errors="coerce")
    stim_df = stim_df.dropna(subset=["start_time_sec", "end_time_sec"])
    stim_df = stim_df[stim_df["end_time_sec"] > stim_df["start_time_sec"]]
    stim_df = stim_df.sort_values("start_time_sec").reset_index(drop=True)
    return stim_df if not stim_df.empty else None


def load_movie(movie_path: Path) -> np.ndarray:
    with tf.TiffFile(movie_path) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Movie must be 3D (T,Y,X), got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def framewise_mean(movie: np.ndarray) -> np.ndarray:
    return movie.mean(axis=(1, 2), dtype=np.float64)


def framewise_median(movie: np.ndarray) -> np.ndarray:
    return np.median(movie, axis=(1, 2)).astype(np.float64)


def reference_from_trace(trace: np.ndarray, fps: float, stim_df: pd.DataFrame | None):
    if REFERENCE_MODE == "global_mean":
        return float(np.mean(trace))
    if REFERENCE_MODE == "global_median":
        return float(np.median(trace))

    if stim_df is not None and len(stim_df) > 0:
        first_start = float(stim_df.loc[0, "start_time_sec"])
        n_pre = int(round(first_start * fps))
        if n_pre >= 3:
            pre = trace[:n_pre]
            if REFERENCE_MODE == "prestim_mean":
                return float(np.mean(pre))
            if REFERENCE_MODE == "prestim_median":
                return float(np.median(pre))

    if REFERENCE_MODE.endswith("mean"):
        return float(np.mean(trace))
    return float(np.median(trace))


def correct_movie_subtract(movie: np.ndarray, gtrace: np.ndarray, gref: float) -> np.ndarray:
    corr = movie.astype(np.float32, copy=True)
    corr -= gtrace[:, None, None].astype(np.float32)
    corr += np.float32(gref)
    return corr


def correct_movie_divide(movie: np.ndarray, gtrace: np.ndarray, gref: float) -> np.ndarray:
    eps = 1e-6
    scale = (np.float32(gref) / np.maximum(gtrace.astype(np.float32), eps))
    corr = movie.astype(np.float32, copy=True)
    corr *= scale[:, None, None]
    return corr


def iterative_divide_mean(movie: np.ndarray, n_iter: int, gref: float) -> np.ndarray:
    corr = movie.astype(np.float32, copy=True)
    for _ in range(n_iter):
        gtrace = framewise_mean(corr)
        corr = correct_movie_divide(corr, gtrace, gref)
    return corr


def save_movie(path: Path, movie: np.ndarray):
    arr = np.asarray(movie, dtype=np.float32 if SAVE_FLOAT32 else np.uint16)
    tf.imwrite(str(path), arr, imagej=False, bigtiff=True)


def save_trace_csv(path: Path, time_axis: np.ndarray, mean_trace: np.ndarray, median_trace: np.ndarray, stim_trace: np.ndarray | None = None):
    df = pd.DataFrame({
        "frame": np.arange(len(time_axis), dtype=int),
        "time_sec": np.round(time_axis, 6),
        "global_mean": mean_trace,
        "global_median": median_trace,
    })
    if stim_trace is not None:
        df["stim_on"] = stim_trace.astype(int)
    df.to_csv(path, index=False)


def stim_boolean_trace(n_frames: int, fps: float, stim_df: pd.DataFrame | None):
    if stim_df is None or stim_df.empty:
        return None
    out = np.zeros(n_frames, dtype=bool)
    for _, row in stim_df.iterrows():
        s = max(0, min(n_frames, int(round(float(row["start_time_sec"]) * fps))))
        e = max(0, min(n_frames, int(round(float(row["end_time_sec"]) * fps))))
        if e > s:
            out[s:e] = True
    return out


def plot_global_brightness_trace(out_png: Path, time_axis: np.ndarray, mean_trace: np.ndarray, median_trace: np.ndarray, stim_df: pd.DataFrame | None, title: str):
    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, mean_trace, color="tab:blue", linewidth=1.2, label="Global mean")
    plt.plot(time_axis, median_trace, color="black", linewidth=1.0, label="Global median")

    if stim_df is not None and not stim_df.empty:
        for i, row in stim_df.iterrows():
            s = float(row["start_time_sec"])
            e = float(row["end_time_sec"])
            plt.axvspan(s, e, color="orange", alpha=0.18, label="Stimulus" if i == 0 else None)

    plt.xlabel("Time (s)")
    plt.ylabel("Global brightness")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def get_preview_frame_indices(n_frames: int, fps: float, stim_df: pd.DataFrame | None):
    if stim_df is None or stim_df.empty:
        center = n_frames // 2
        return max(0, center - 1), center, min(n_frames - 1, center + 1)

    idx = min(max(PREVIEW_STIM_INDEX, 0), len(stim_df) - 1)
    row = stim_df.loc[idx]
    s = float(row["start_time_sec"])
    e = float(row["end_time_sec"])

    pre_t = max(0.0, s - PREVIEW_EXTRA_SEC)
    post_t = min((n_frames - 1) / fps, e + PREVIEW_EXTRA_SEC)

    pre_f = max(0, min(n_frames - 1, int(round(pre_t * fps))))
    stim_f = max(0, min(n_frames - 1, int(round(s * fps))))
    post_f = max(0, min(n_frames - 1, int(round(post_t * fps))))
    return pre_f, stim_f, post_f


def percentile_stretch(img: np.ndarray, pmin=1.0, pmax=99.5):
    x = np.asarray(img, dtype=np.float32)
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y


def plot_before_after_preview(out_png: Path,
                              raw_movie: np.ndarray,
                              corr_movie: np.ndarray,
                              fps: float,
                              stim_df: pd.DataFrame | None,
                              mode_name: str):
    pre_f, stim_f, post_f = get_preview_frame_indices(raw_movie.shape[0], fps, stim_df)

    fig, axes = plt.subplots(2, 3, figsize=PREVIEW_FIGSIZE)

    imgs = [
        (raw_movie[pre_f],  f"Raw pre (frame {pre_f})"),
        (raw_movie[stim_f], f"Raw stim (frame {stim_f})"),
        (raw_movie[post_f], f"Raw post (frame {post_f})"),
        (corr_movie[pre_f],  f"{mode_name} pre"),
        (corr_movie[stim_f], f"{mode_name} stim"),
        (corr_movie[post_f], f"{mode_name} post"),
    ]

    for ax, (img, ttl) in zip(axes.flat, imgs):
        ax.imshow(percentile_stretch(img), cmap="gray")
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Brightness correction preview - {mode_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (percentile_stretch(img, VIDEO_CONTRAST_PMIN, VIDEO_CONTRAST_PMAX) * 255).astype(np.uint8)


def _add_text_panel(panel: np.ndarray, lines: list[str]):
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil)
        y = 10
        for line in lines:
            draw.text((10, y), line, fill=VIDEO_TEXT_COLOR)
            y += 18
        return np.array(pil)
    except Exception:
        return panel


def make_comparison_video(out_mp4: Path,
                          raw_movie: np.ndarray,
                          movies: dict,
                          fps: float,
                          stim_df: pd.DataFrame | None):
    n_frames = raw_movie.shape[0]

    target_fps = max(1.0, fps * PLAYBACK_SPEED_MULTIPLIER)
    step = max(1, int(np.ceil(target_fps / MAX_OUTPUT_VIDEO_FPS)))
    output_fps = target_fps / step

    names = ["raw", "subtract_mean", "subtract_median", "divide_mean_1", "divide_mean_2", "divide_mean_3"]

    stim_bool = stim_boolean_trace(n_frames, fps, stim_df)

    with imageio.get_writer(str(out_mp4), fps=output_fps) as writer:
        for fidx in range(0, n_frames, step):
            panels = []
            for name in names:
                img = raw_movie[fidx] if name == "raw" else movies[name][fidx]
                panel = _to_uint8(img)
                panel = np.stack([panel, panel, panel], axis=-1)

                lines = [name, f"frame={fidx}", f"t={fidx / max(fps, 1e-6):.2f}s"]
                if stim_bool is not None:
                    lines.append("stim ON" if stim_bool[fidx] else "stim OFF")
                panel = _add_text_panel(panel, lines)
                panels.append(panel)

            row1 = np.concatenate(panels[:3], axis=1)
            row2 = np.concatenate(panels[3:], axis=1)
            canvas = np.concatenate([row1, row2], axis=0)
            writer.append_data(canvas)

    return {"output_fps": float(output_fps), "frame_step": int(step)}


def process_single_trial(inputs: dict):
    trial_dir = Path(inputs["trial_dir"])
    prefix = inputs["prefix"]
    movie_path = Path(inputs["movie_path"])
    json_path = Path(inputs["json_path"]) if inputs["json_path"] is not None else None
    stim_path = Path(inputs["stim_path"]) if inputs["stim_path"] is not None else None

    out_dir = trial_dir / "brightness_correction"
    if out_dir.exists() and EXISTING_MODE == "skip":
        print(f"[SKIP] {trial_dir.name} -> brightness correction already exists")
        return True
    if out_dir.exists() and EXISTING_MODE == "overwrite":
        clear_dir_contents(out_dir)
    safe_makedirs(out_dir)

    print(f"\n=== Trial: {trial_dir.name} ===")
    print(f"    movie: {movie_path.name}")
    print(f"    stim : {stim_path.name if stim_path is not None else 'NOT FOUND'}")
    print(f"    meta : {json_path.name if json_path is not None else 'NOT FOUND'}")

    meta = get_trial_metadata(json_path)
    fps = float(meta.get("fs", 1.0))
    print(f"    fps  : {fps:.4f}")

    stim_df = None
    if stim_path is not None and stim_path.exists():
        try:
            stim_df = validate_stim_events(pd.read_csv(stim_path))
        except Exception as e:
            print(f"    [Warning] Failed to parse stim events: {e}")
            stim_df = None

    movie = load_movie(movie_path)
    n_frames, h, w = movie.shape
    print(f"    movie shape: T={n_frames}, Y={h}, X={w}")

    mean_trace = framewise_mean(movie)
    median_trace = framewise_median(movie)
    time_axis = np.arange(n_frames, dtype=np.float64) / max(fps, 1e-6)
    stim_bool = stim_boolean_trace(n_frames, fps, stim_df)

    trace_csv = trial_dir / f"{prefix}_global_brightness_trace.csv"
    trace_png = trial_dir / f"{prefix}_global_brightness_trace.png"

    save_trace_csv(trace_csv, time_axis, mean_trace, median_trace, stim_bool)
    plot_global_brightness_trace(
        trace_png,
        time_axis,
        mean_trace,
        median_trace,
        stim_df,
        title=f"Global brightness trace - {trial_dir.name}",
    )

    ref_mean = reference_from_trace(mean_trace, fps, stim_df)
    ref_median = reference_from_trace(median_trace, fps, stim_df)

    movies = {
        "subtract_mean": correct_movie_subtract(movie, mean_trace, ref_mean),
        "subtract_median": correct_movie_subtract(movie, median_trace, ref_median),
        "divide_mean_1": iterative_divide_mean(movie, 1, ref_mean),
        "divide_mean_2": iterative_divide_mean(movie, 2, ref_mean),
        "divide_mean_3": iterative_divide_mean(movie, 3, ref_mean),
    }

    outputs = {}
    for mode_name, corr_movie in movies.items():
        out_movie = trial_dir / f"{prefix}{OUT_SUFFIXES[mode_name]}"
        save_movie(out_movie, corr_movie)

        preview_png = out_dir / f"preview_{mode_name}.png"
        plot_before_after_preview(preview_png, movie, corr_movie, fps, stim_df, mode_name)

        outputs[mode_name] = {
            "movie": str(out_movie),
            "preview_png": str(preview_png),
            "min": float(np.min(corr_movie)),
            "max": float(np.max(corr_movie)),
            "mean": float(np.mean(corr_movie)),
        }
        print(f"    saved {mode_name}: {out_movie.name}")

    if MAKE_COMPARISON_VIDEO:
        video_path = out_dir / "compare_all_modes.mp4"
        try:
            video_info = make_comparison_video(video_path, movie, movies, fps, stim_df)
            outputs["comparison_video"] = {
                "path": str(video_path),
                **video_info,
            }
            print(f"    saved comparison video: {video_path.name}")
        except Exception as e:
            outputs["comparison_video"] = {"error": str(e)}
            print(f"    [Warning] comparison video failed: {e}")

    summary = {
        "trial": trial_dir.name,
        "prefix": prefix,
        "fps": fps,
        "n_frames": int(n_frames),
        "shape": [int(n_frames), int(h), int(w)],
        "reference_mode": REFERENCE_MODE,
        "ref_mean": float(ref_mean),
        "ref_median": float(ref_median),
        "inputs": {
            "movie_path": str(movie_path),
            "metadata_path": str(json_path) if json_path is not None else None,
            "stim_path": str(stim_path) if stim_path is not None else None,
        },
        "outputs": outputs,
    }

    with open(out_dir / "correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return True


def main():
    root_dir = Path(ROOT_DIR).resolve()
    print("=" * 70)
    print("03.5 Brightness correction (3 modes + iterative divide)")
    print("=" * 70)
    print(f"ROOT_DIR        : {root_dir}")
    print(f"EXISTING_MODE   : {EXISTING_MODE}")
    print(f"REFERENCE_MODE  : {REFERENCE_MODE}")
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