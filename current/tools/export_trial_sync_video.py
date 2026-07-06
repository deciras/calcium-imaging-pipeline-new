#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出单个 trial 的同步查看视频：

- 上方左侧：原始记录 stack
- 上方右侧：stim analog stack
- 下方：stimulus brightness 轨迹 + 当前刺激角度
- 最底部：播放进度条

这个脚本只消费 step-01 / step-02 的现成输出，不会重跑 pipeline。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import cv2

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class TrialPaths:
    trial_id: str
    raw_movie: Path
    analog_movie: Path
    metadata_json: Path
    stim_events_csv: Path
    brightness_trace_csv: Path


@dataclass(frozen=True)
class StimWindow:
    start_sec: float
    end_sec: float
    angle_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a synchronized raw/analog trial video.")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root.")
    parser.add_argument("--trial-id", required=True, help="Trial ID, e.g. 20260428_Euprymna_retina_25x_0001.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output mp4 path. Default: DATA_ROOT/18_reports/trial_sync_videos/<trial_id>_raw_analog_sync.mp4",
    )
    parser.add_argument("--output-fps", type=float, default=8.0, help="Output video fps. Default: 8.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 1 = real time, 60 = 60x real time. Default: 1.",
    )
    parser.add_argument("--display-step", type=int, default=2, help="Spatial downsample step for display. Default: 2.")
    parser.add_argument("--max-frames", type=int, help="Export only the first N frames.")
    return parser.parse_args()


def resolve_trial_paths(data_root: Path, trial_id: str) -> TrialPaths:
    step01_dir = data_root / "01_oir_to_tif" / trial_id
    step02_dir = data_root / "02_stim_map" / trial_id
    if not step01_dir.exists() or not step02_dir.exists():
        trial_date = trial_id.split("_", 1)[0]
        dated_step01_dir = data_root / "01_oir_to_tif" / trial_date / trial_id
        dated_step02_dir = data_root / "02_stim_map" / trial_date / trial_id
        if dated_step01_dir.exists() and dated_step02_dir.exists():
            step01_dir = dated_step01_dir
            step02_dir = dated_step02_dir
    raw_movie = step01_dir / f"{trial_id}_Max_Proj.tif"
    analog_movie = step01_dir / f"{trial_id}_Stim_Analog.tif"
    metadata_json = step01_dir / f"{trial_id}_metadata.json"
    stim_events_csv = step02_dir / f"{trial_id}_stim_events.csv"
    brightness_trace_csv = step02_dir / f"{trial_id}_brightness_trace.csv"
    for path in (raw_movie, analog_movie, metadata_json, stim_events_csv, brightness_trace_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    return TrialPaths(
        trial_id=trial_id,
        raw_movie=raw_movie,
        analog_movie=analog_movie,
        metadata_json=metadata_json,
        stim_events_csv=stim_events_csv,
        brightness_trace_csv=brightness_trace_csv,
    )


def load_fps(metadata_json: Path) -> float:
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    temporal = metadata.get("temporal_calibration", {})
    fps = temporal.get("fps")
    if fps is None:
        frame_interval = temporal.get("frame_interval_sec")
        if frame_interval:
            fps = 1.0 / float(frame_interval)
    if fps is None or float(fps) <= 0:
        raise ValueError(f"Could not read a valid fps from metadata: {metadata_json}")
    return float(fps)


def load_stack(path: Path) -> np.ndarray:
    try:
        return tf.memmap(path)
    except Exception:
        return tf.imread(path)


def sample_levels(stack: np.ndarray, sample_count: int = 24) -> tuple[float, float]:
    n_frames = int(stack.shape[0])
    if n_frames <= sample_count:
        sample_idx = np.arange(n_frames)
    else:
        sample_idx = np.linspace(0, n_frames - 1, num=sample_count, dtype=int)
    sampled = np.asarray(stack[sample_idx], dtype=np.float32)
    low = float(np.percentile(sampled, 1))
    high = float(np.percentile(sampled, 99.5))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(sampled))
        high = float(np.max(sampled))
    if high <= low:
        high = low + 1.0
    return low, high


def normalize_frame(frame: np.ndarray, low: float, high: float, display_step: int) -> np.ndarray:
    display = np.asarray(frame[::display_step, ::display_step], dtype=np.float32)
    scaled = (display - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def load_stim_windows(stim_events_csv: Path) -> list[StimWindow]:
    stim = pd.read_csv(stim_events_csv)
    windows: list[StimWindow] = []
    for _, row in stim.iterrows():
        start_sec = pd.to_numeric(row.get("start_time_sec"), errors="coerce")
        end_sec = pd.to_numeric(row.get("end_time_sec"), errors="coerce")
        if pd.isna(start_sec) or pd.isna(end_sec):
            continue
        angle = pd.to_numeric(row.get("pol_angle"), errors="coerce")
        angle_label = f"{int(round(angle))} deg" if pd.notna(angle) else "unknown"
        windows.append(StimWindow(float(start_sec), float(end_sec), angle_label))
    return windows


def current_angle_label(stim_windows: list[StimWindow], time_sec: float) -> str:
    for window in stim_windows:
        if window.start_sec <= time_sec <= window.end_sec:
            return window.angle_label
    return "no stim"


def default_output_path(data_root: Path, trial_id: str) -> Path:
    out_dir = data_root / "18_reports" / "trial_sync_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{trial_id}_raw_analog_sync.mp4"


def choose_frame_indices(total_frames: int, acquisition_fps: float, output_fps: float, speed: float, max_frames: int | None) -> np.ndarray:
    if output_fps <= 0:
        raise ValueError("output_fps must be > 0.")
    if speed <= 0:
        raise ValueError("speed must be > 0.")
    stride = speed * acquisition_fps / output_fps
    indices = np.floor(np.arange(0, total_frames, stride)).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)
    if len(indices) == 0:
        indices = np.array([0], dtype=int)
    if max_frames is not None:
        indices = indices[: max(1, int(max_frames))]
    return indices


def build_figure(
    trial_id: str,
    brightness: pd.DataFrame,
    stim_windows: list[StimWindow],
    raw_first: np.ndarray,
    analog_first: np.ndarray,
    output_fps: float,
    acquisition_fps: float,
    effective_speed: float,
) -> tuple[plt.Figure, dict[str, object]]:
    fig = plt.Figure(figsize=(12, 8), dpi=120, constrained_layout=True)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[12, 4, 1.2])
    raw_ax = fig.add_subplot(gs[0, 0])
    analog_ax = fig.add_subplot(gs[0, 1])
    trace_ax = fig.add_subplot(gs[1, :])
    progress_ax = fig.add_subplot(gs[2, :])

    raw_im = raw_ax.imshow(raw_first, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    analog_im = analog_ax.imshow(analog_first, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    for ax, title in ((raw_ax, "Raw recording stack"), (analog_ax, "Stim analog stack")):
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    brightness_time = pd.to_numeric(brightness["time_sec"], errors="coerce").to_numpy(dtype=float)
    brightness_value = pd.to_numeric(brightness["brightness"], errors="coerce").to_numpy(dtype=float)
    trace_ax.plot(brightness_time, brightness_value, color="#2667a8", linewidth=1.2)
    for window in stim_windows:
        trace_ax.axvspan(window.start_sec, window.end_sec, color="#f4c95d", alpha=0.25, linewidth=0)
    current_line = trace_ax.axvline(0.0, color="#c44536", linewidth=2)
    current_point = trace_ax.scatter([brightness_time[0]], [brightness_value[0]], s=36, color="#c44536", zorder=5)
    y_max = float(np.nanmax(brightness_value)) if len(brightness_value) else 1.0
    y_min = float(np.nanmin(brightness_value)) if len(brightness_value) else 0.0
    y_span = max(y_max - y_min, 1.0)
    angle_text = trace_ax.text(
        brightness_time[0] if len(brightness_time) else 0.0,
        y_max + y_span * 0.04,
        "",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#7a1f15",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#c44536", alpha=0.9),
    )
    trace_ax.set_title("Stimulus trace (brightness / analog)", fontsize=13)
    trace_ax.set_xlabel("Time (s)")
    trace_ax.set_ylabel("Brightness")
    trace_ax.grid(alpha=0.2, linewidth=0.5)
    if len(brightness_time):
        trace_ax.set_xlim(float(np.nanmin(brightness_time)), float(np.nanmax(brightness_time)))
    trace_ax.set_ylim(y_min - y_span * 0.05, y_max + y_span * 0.16)

    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis("off")
    progress_bg = Rectangle((0.0, 0.18), 1.0, 0.64, facecolor="#e6e6e6", edgecolor="#999999", linewidth=1)
    progress_fg = Rectangle((0.0, 0.18), 0.0, 0.64, facecolor="#3d9970", edgecolor="none")
    progress_ax.add_patch(progress_bg)
    progress_ax.add_patch(progress_fg)
    progress_text = progress_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=11, color="black")

    fig.suptitle(
        f"Trial {trial_id} | output {output_fps:.1f} fps | playback {effective_speed:.2f}x",
        fontsize=14,
    )
    return fig, {
        "canvas": canvas,
        "raw_im": raw_im,
        "analog_im": analog_im,
        "trace_line": current_line,
        "trace_point": current_point,
        "angle_text": angle_text,
        "progress_fg": progress_fg,
        "progress_text": progress_text,
        "brightness_time": brightness_time,
        "brightness_value": brightness_value,
        "trace_ymax": y_max,
        "trace_yspan": y_span,
    }


def render_video(
    trial_paths: TrialPaths,
    output_path: Path,
    output_fps: float,
    speed: float,
    display_step: int,
    max_frames: int | None,
) -> None:
    raw_stack = load_stack(trial_paths.raw_movie)
    analog_stack = load_stack(trial_paths.analog_movie)
    if raw_stack.shape[0] != analog_stack.shape[0]:
        raise ValueError(f"Raw and analog stack frame counts do not match: {raw_stack.shape[0]} vs {analog_stack.shape[0]}")

    acquisition_fps = load_fps(trial_paths.metadata_json)
    total_frames = int(raw_stack.shape[0])
    frame_indices = choose_frame_indices(total_frames, acquisition_fps, output_fps, speed, max_frames)
    effective_speed = speed

    brightness = pd.read_csv(trial_paths.brightness_trace_csv)
    stim_windows = load_stim_windows(trial_paths.stim_events_csv)
    if not stim_windows:
        raise ValueError(f"{trial_paths.trial_id} has no usable stimulus windows for angle labels.")

    raw_low, raw_high = sample_levels(raw_stack[frame_indices])
    analog_low, analog_high = sample_levels(analog_stack[frame_indices])
    raw_first = normalize_frame(raw_stack[0], raw_low, raw_high, display_step)
    analog_first = normalize_frame(analog_stack[0], analog_low, analog_high, display_step)

    fig, artists = build_figure(
        trial_id=trial_paths.trial_id,
        brightness=brightness,
        stim_windows=stim_windows,
        raw_first=raw_first,
        analog_first=analog_first,
        output_fps=output_fps,
        acquisition_fps=acquisition_fps,
        effective_speed=effective_speed,
    )
    canvas: FigureCanvasAgg = artists["canvas"]  # type: ignore[assignment]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.draw()
    first_rgba = np.asarray(canvas.buffer_rgba())
    height, width = first_rgba.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    try:
        n_output_frames = len(frame_indices)
        for output_idx, frame_idx in enumerate(frame_indices):
            time_sec = frame_idx / acquisition_fps
            raw_frame = normalize_frame(raw_stack[frame_idx], raw_low, raw_high, display_step)
            analog_frame = normalize_frame(analog_stack[frame_idx], analog_low, analog_high, display_step)
            artists["raw_im"].set_data(raw_frame)
            artists["analog_im"].set_data(analog_frame)
            artists["trace_line"].set_xdata([time_sec, time_sec])
            brightness_time = artists["brightness_time"]
            brightness_value = artists["brightness_value"]
            brightness_idx = int(np.clip(np.searchsorted(brightness_time, time_sec, side="left"), 0, len(brightness_time) - 1))
            artists["trace_point"].set_offsets([[brightness_time[brightness_idx], brightness_value[brightness_idx]]])

            angle_label = current_angle_label(stim_windows, time_sec)
            label_x = brightness_time[brightness_idx] if len(brightness_time) else time_sec
            label_y = artists["trace_ymax"] + artists["trace_yspan"] * 0.05
            artists["angle_text"].set_position((label_x, label_y))
            artists["angle_text"].set_text(f"Current AoLP: {angle_label}")

            progress = 0.0 if total_frames <= 1 else frame_idx / (total_frames - 1)
            artists["progress_fg"].set_width(progress)
            artists["progress_text"].set_text(
                f"output frame {output_idx + 1}/{n_output_frames} | source frame {frame_idx + 1}/{total_frames} | t = {time_sec:6.1f} s | {progress * 100:5.1f}%"
            )

            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    trial_paths = resolve_trial_paths(args.data_root, args.trial_id)
    output_path = args.output if args.output else default_output_path(args.data_root, args.trial_id)
    render_video(
        trial_paths=trial_paths,
        output_path=output_path,
        output_fps=float(args.output_fps),
        speed=float(args.speed),
        display_step=max(1, int(args.display_step)),
        max_frames=args.max_frames,
    )
    print(output_path)


if __name__ == "__main__":
    main()
