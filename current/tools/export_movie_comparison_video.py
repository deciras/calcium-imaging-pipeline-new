#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出同一 trial 的双窗 movie 对比视频。

支持场景：
- 原始 vs motion corrected
- motion corrected vs spatial high-pass

这个脚本只读取现成的 TIFF movie 和 metadata，不会重跑 pipeline。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import cv2

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a side-by-side movie comparison video.")
    parser.add_argument("--left-movie", type=Path, required=True, help="Path to the left movie.")
    parser.add_argument("--right-movie", type=Path, required=True, help="Path to the right movie.")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Step-01 metadata.json for acquisition fps.")
    parser.add_argument("--left-label", required=True, help="Left panel title.")
    parser.add_argument("--right-label", required=True, help="Right panel title.")
    parser.add_argument("--trial-id", required=True, help="Trial ID.")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path.")
    parser.add_argument("--output-fps", type=float, default=60.0, help="Output video fps. Default: 60.")
    parser.add_argument("--speed", type=float, default=60.0, help="Playback speed multiplier. Default: 60.")
    parser.add_argument("--display-step", type=int, default=2, help="Spatial downsample step for display. Default: 2.")
    parser.add_argument("--max-frames", type=int, help="Export only the first N output frames.")
    return parser.parse_args()


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


def sample_levels(stack: np.ndarray, sample_idx: np.ndarray) -> tuple[float, float]:
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


def choose_frame_indices(total_frames: int, acquisition_fps: float, output_fps: float, speed: float, max_frames: int | None) -> np.ndarray:
    if output_fps <= 0:
        raise ValueError("output_fps must be > 0.")
    if speed <= 0:
        raise ValueError("speed must be > 0.")
    stride = max(speed * acquisition_fps / output_fps, 1.0)
    indices = np.floor(np.arange(0, total_frames, stride)).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)
    if len(indices) == 0:
        indices = np.array([0], dtype=int)
    indices = np.unique(indices)
    if max_frames is not None:
        indices = indices[: max(1, int(max_frames))]
    return indices


def build_figure(
    trial_id: str,
    left_label: str,
    right_label: str,
    left_first: np.ndarray,
    right_first: np.ndarray,
    output_fps: float,
    effective_speed: float,
) -> tuple[plt.Figure, dict[str, object]]:
    fig = plt.Figure(figsize=(12, 6.8), dpi=120, constrained_layout=True)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[14, 1.5])
    left_ax = fig.add_subplot(gs[0, 0])
    right_ax = fig.add_subplot(gs[0, 1])
    progress_ax = fig.add_subplot(gs[1, :])

    left_im = left_ax.imshow(left_first, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    right_im = right_ax.imshow(right_first, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    for ax, title in ((left_ax, left_label), (right_ax, right_label)):
        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis("off")
    progress_bg = Rectangle((0.0, 0.18), 1.0, 0.64, facecolor="#e6e6e6", edgecolor="#999999", linewidth=1)
    progress_fg = Rectangle((0.0, 0.18), 0.0, 0.64, facecolor="#2e8b57", edgecolor="none")
    progress_ax.add_patch(progress_bg)
    progress_ax.add_patch(progress_fg)
    progress_text = progress_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=12, color="black")

    fig.suptitle(f"Trial {trial_id} | output {output_fps:.1f} fps | playback {effective_speed:.2f}x", fontsize=15)
    return fig, {
        "canvas": canvas,
        "left_im": left_im,
        "right_im": right_im,
        "progress_fg": progress_fg,
        "progress_text": progress_text,
    }


def render_comparison_video(
    left_movie: Path,
    right_movie: Path,
    metadata_json: Path,
    left_label: str,
    right_label: str,
    trial_id: str,
    output: Path,
    output_fps: float,
    speed: float,
    display_step: int,
    max_frames: int | None,
) -> None:
    left_stack = load_stack(left_movie)
    right_stack = load_stack(right_movie)
    if left_stack.shape != right_stack.shape:
        raise ValueError(f"Left/right movie shapes do not match: {left_stack.shape} vs {right_stack.shape}")
    if left_stack.ndim != 3:
        raise ValueError(f"Movie must be a T x H x W array, got: {left_stack.shape}")

    acquisition_fps = load_fps(metadata_json)
    total_frames = int(left_stack.shape[0])
    frame_indices = choose_frame_indices(total_frames, acquisition_fps, output_fps, speed, max_frames)
    effective_speed = (frame_indices[1] - frame_indices[0]) * acquisition_fps / output_fps if len(frame_indices) > 1 else speed

    left_low, left_high = sample_levels(left_stack, frame_indices)
    right_low, right_high = sample_levels(right_stack, frame_indices)
    left_first = normalize_frame(left_stack[frame_indices[0]], left_low, left_high, display_step)
    right_first = normalize_frame(right_stack[frame_indices[0]], right_low, right_high, display_step)

    fig, artists = build_figure(
        trial_id=trial_id,
        left_label=left_label,
        right_label=right_label,
        left_first=left_first,
        right_first=right_first,
        output_fps=output_fps,
        effective_speed=effective_speed,
    )
    canvas: FigureCanvasAgg = artists["canvas"]  # type: ignore[assignment]

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.draw()
    first_rgba = np.asarray(canvas.buffer_rgba())
    height, width = first_rgba.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output}")

    try:
        n_output_frames = len(frame_indices)
        for output_idx, frame_idx in enumerate(frame_indices):
            left_frame = normalize_frame(left_stack[frame_idx], left_low, left_high, display_step)
            right_frame = normalize_frame(right_stack[frame_idx], right_low, right_high, display_step)
            artists["left_im"].set_data(left_frame)
            artists["right_im"].set_data(right_frame)

            progress = 0.0 if total_frames <= 1 else frame_idx / (total_frames - 1)
            time_sec = frame_idx / acquisition_fps
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
    render_comparison_video(
        left_movie=args.left_movie,
        right_movie=args.right_movie,
        metadata_json=args.metadata_json,
        left_label=args.left_label,
        right_label=args.right_label,
        trial_id=args.trial_id,
        output=args.output,
        output_fps=float(args.output_fps),
        speed=float(args.speed),
        display_step=max(1, int(args.display_step)),
        max_frames=args.max_frames,
    )
    print(args.output)


if __name__ == "__main__":
    main()
