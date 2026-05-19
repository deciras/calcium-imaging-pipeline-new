#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
刺激信号分析脚本 v3.2
功能：
1. 从 *_Stim_Analog.tif 提取刺激模拟信号
2. 精确识别每个刺激的起始和结束时间点（考虑边界帧的部分刺激）
3. 为每个 trial 生成:
   - trialID_brightness_trace.csv
   - trialID_stim_trace.png
   - trialID_stim_map.csv
   - trialID_stim_events.csv   ← 列名含 trialID，供 06/07 按列匹配
4. 为每个日期文件夹额外生成汇总版:
   - stim_map.csv
   - stim_events.csv           ← 同样含 trialID 列，06/07 可按 trialID 筛选
5. 支持自动清理旧输出

修复说明 (v3.2)
--------------
- 修复 #8：nostim 判断从硬编码绝对阈值 (b_max-b_min < 50) 改为相对阈值
  (range < NOSTIM_REL_THRESH * b_max)，对不同相机/增益更鲁棒。
  NOSTIM_REL_THRESH 暴露为配置参数，默认 0.05（5%）。
- 说明 trialID 列命名约定：06/07 脚本通过 trialID 列匹配当前 trial，
  请勿修改该列名。
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

# 配置 matplotlib 支持中文
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ================== 配置参数 ==================
DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus'

# 是否自动删除旧输出
CLEAN_TRIAL_OUTPUTS = True
CLEAN_DAY_OUTPUTS = True

# ------------------------------------------------------------------
# 修复 #8：nostim 判断阈值
# 原版：b_max - b_min < 50（绝对值，对不同相机/增益不鲁棒）
# 修复：range < NOSTIM_REL_THRESH * b_max（相对值）
# 含义：亮度变化幅度不足最大亮度的 NOSTIM_REL_THRESH 倍，则判定为无刺激
# 默认 0.05（5%）；如果你的刺激信号很弱，可适当降低（如 0.02）
# ------------------------------------------------------------------
NOSTIM_REL_THRESH = 0.05


# ================== 工具函数 ==================
def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"    Removed old file -> {path}")


def clean_trial_outputs(trial_dir):
    """
    删除该步骤在单个 trial 内可能生成的所有文件。
    用通配符而不是精确 trial_id，避免历史命名残留。
    """
    if not CLEAN_TRIAL_OUTPUTS:
        return 0

    patterns = [
        "*_brightness_trace.csv",
        "*_stim_trace.png",
        "*_stim_map.csv",
        "*_stim_events.csv",
    ]

    removed = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(trial_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)
                print(f"    Removed old file -> {path}")
                removed += 1
    return removed


def clean_day_outputs(date_folder):
    """
    删除 day 层汇总文件。
    """
    if not CLEAN_DAY_OUTPUTS:
        return 0

    targets = [
        os.path.join(date_folder, "stim_map.csv"),
        os.path.join(date_folder, "stim_events.csv"),
    ]

    removed = 0
    for path in targets:
        if os.path.exists(path):
            os.remove(path)
            print(f"    Removed old file -> {path}")
            removed += 1
    return removed


# ================== 核心函数 ==================
def estimate_precise_boundary(brightness_trace, frame_idx, baseline, peak, is_start=True):
    """
    估算刺激在边界帧内的精确时间点
    """
    if frame_idx < 0 or frame_idx >= len(brightness_trace):
        return 0.0 if is_start else 1.0

    brightness = brightness_trace[frame_idx]

    if peak - baseline > 0:
        ratio = (brightness - baseline) / (peak - baseline)
        ratio = np.clip(ratio, 0.0, 1.0)
    else:
        ratio = 0.5

    if is_start:
        return 1.0 - ratio
    else:
        return ratio


def analyze_stim_blocks(brightness_trace, dt, dynamic_thresh, baseline, peak):
    """
    分析刺激块，精确计算每个刺激的起止时间
    """
    is_stim_on = brightness_trace > dynamic_thresh
    stim_indices = np.where(is_stim_on)[0]

    if len(stim_indices) == 0:
        return []

    diffs = np.diff(stim_indices)
    split_points = np.where(diffs > 1)[0] + 1
    blocks = np.split(stim_indices, split_points)

    stim_blocks = []

    for block_idx, block in enumerate(blocks):
        if len(block) == 0:
            continue

        start_frame = block[0]
        end_frame = block[-1]

        start_offset = estimate_precise_boundary(
            brightness_trace, start_frame, baseline, peak, is_start=True
        )
        end_offset = estimate_precise_boundary(
            brightness_trace, end_frame, baseline, peak, is_start=False
        )

        start_time = (start_frame + start_offset) * dt
        end_time = (end_frame + end_offset) * dt
        duration = end_time - start_time

        stim_blocks.append({
            'stim_index': block_idx + 1,
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'start_time_sec': round(start_time, 3),
            'end_time_sec': round(end_time, 3),
            'duration_sec': round(duration, 3)
        })

    return stim_blocks


def plot_stim_trace(brightness_trace, time_axis, stim_blocks, dynamic_thresh,
                    trial_id, output_path):
    """
    绘制刺激曲线图
    """
    total_time = time_axis[-1] if len(time_axis) > 0 else 1
    max_width = 50
    fig_width = min(max(12, total_time / 10), max_width)

    max_points = 10000
    if len(brightness_trace) > max_points:
        downsample_factor = len(brightness_trace) // max_points + 1
        brightness_trace_plot = brightness_trace[::downsample_factor]
        time_axis_plot = time_axis[::downsample_factor]
        print(f"    [Info] Downsampling trace for plotting: "
              f"{len(brightness_trace)} -> {len(brightness_trace_plot)} points")
    else:
        brightness_trace_plot = brightness_trace
        time_axis_plot = time_axis

    fig, ax = plt.subplots(figsize=(fig_width, 4))

    ax.plot(time_axis_plot, brightness_trace_plot, 'b-', linewidth=0.8,
            label='Brightness', alpha=0.7)

    ax.axhline(y=dynamic_thresh, color='r', linestyle='--', linewidth=1,
               label=f'Threshold ({dynamic_thresh:.1f})')

    for block in stim_blocks:
        start_t = block['start_time_sec']
        end_t = block['end_time_sec']
        ax.axvspan(start_t, end_t, alpha=0.3, color='orange',
                   label='Stimulus' if block == stim_blocks[0] else '')

        if len(stim_blocks) <= 50:
            mid_t = (start_t + end_t) / 2
            y_pos = ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else dynamic_thresh * 1.1
            ax.text(mid_t, y_pos, f"#{block['stim_index']}",
                    ha='center', va='top', fontsize=8, fontweight='bold')

    ax.set_xlabel('Time (sec)', fontsize=11)
    ax.set_ylabel('Brightness (a.u.)', fontsize=11)
    ax.set_title(f'Stimulus Trace - {trial_id}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    except ValueError as e:
        print(f"    [Warning] Failed to save full resolution plot: {e}")
        try:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"    [Info] Saved with reduced DPI (100)")
        except Exception:
            print(f"    [Error] Could not save plot for {trial_id}")

    plt.close()


def build_summary_row(trial_id, stim_blocks):
    # ------------------------------------------------------------------
    # 注意：trialID 列名是 06/07 脚本按列匹配的关键，请勿修改。
    # 06 的 find_trial_stim_row 和 07 的 subset_stim_events_for_trial
    # 都会尝试用 "trialID" 列匹配当前 trial 的文件夹名或 prefix。
    # ------------------------------------------------------------------
    summary_row = {
        "trialID": trial_id,
        "stim_type": "",
        "number_of_stim": len(stim_blocks),
        "initial_delay_sec": "",
        "avg_duration_sec": "",
        "avg_interval_sec": "",
        "strength": "",
        "pol_angle_list": ""
    }

    if len(stim_blocks) > 0:
        initial_delay = stim_blocks[0]['start_time_sec']
        summary_row['initial_delay_sec'] = round(initial_delay, 2)

        durations = [b['duration_sec'] for b in stim_blocks]
        summary_row['avg_duration_sec'] = round(float(np.mean(durations)), 2)

        if len(stim_blocks) > 1:
            intervals = [
                stim_blocks[i + 1]['start_time_sec'] - stim_blocks[i]['start_time_sec']
                for i in range(len(stim_blocks) - 1)
            ]
            summary_row['avg_interval_sec'] = round(float(np.mean(intervals)), 2)
    else:
        summary_row['stim_type'] = "nostim"

    return summary_row


def build_events_rows(trial_id, stim_blocks):
    # trialID 列名同上，供下游脚本匹配
    events_rows = []
    for block in stim_blocks:
        events_rows.append({
            'trialID': trial_id,
            'stim_index': block['stim_index'],
            'start_time_sec': block['start_time_sec'],
            'end_time_sec': block['end_time_sec'],
            'duration_sec': block['duration_sec'],
            'stim_type': '',
            'pol_angle': '',
            'strength': ''
        })
    return events_rows


def save_trial_outputs(trial_dir, trial_id, summary_row, events_rows):
    """
    为单个 trial 保存 stim_map / stim_events
    """
    summary_path = os.path.join(trial_dir, f"{trial_id}_stim_map.csv")
    events_path = os.path.join(trial_dir, f"{trial_id}_stim_events.csv")

    df_summary = pd.DataFrame([summary_row])
    df_summary = df_summary.reindex(columns=[
        "trialID", "stim_type", "number_of_stim", "initial_delay_sec",
        "avg_duration_sec", "avg_interval_sec", "strength", "pol_angle_list"
    ])
    df_summary.to_csv(summary_path, index=False)

    df_events = pd.DataFrame(events_rows)
    df_events = df_events.reindex(columns=[
        "trialID", "stim_index", "start_time_sec", "end_time_sec",
        "duration_sec", "stim_type", "pol_angle", "strength"
    ])
    df_events.to_csv(events_path, index=False)

    print(f"    Saved trial summary -> {summary_path}")
    print(f"    Saved trial events  -> {events_path}")


def save_day_outputs(date_folder, all_summary_data, all_events_data):
    """
    保存 day 层汇总文件
    """
    if all_summary_data:
        df_summary = pd.DataFrame(all_summary_data)
        df_summary = df_summary.reindex(columns=[
            "trialID", "stim_type", "number_of_stim", "initial_delay_sec",
            "avg_duration_sec", "avg_interval_sec", "strength", "pol_angle_list"
        ])
        summary_path = os.path.join(date_folder, 'stim_map.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"\n✓ Saved day summary to: {summary_path}")

    if all_events_data:
        df_events = pd.DataFrame(all_events_data)
        df_events = df_events.reindex(columns=[
            "trialID", "stim_index", "start_time_sec", "end_time_sec",
            "duration_sec", "stim_type", "pol_angle", "strength"
        ])
        events_path = os.path.join(date_folder, 'stim_events.csv')
        df_events.to_csv(events_path, index=False)
        print(f"✓ Saved day events to: {events_path}")


def is_nostim(b_max, b_min):
    """
    修复 #8：判断是否为无刺激 trial。
    原版：b_max - b_min < 50（绝对阈值，对不同相机/增益不鲁棒）
    修复：range < NOSTIM_REL_THRESH * b_max（相对阈值）
    b_max <= 0 时（异常数据）也判定为 nostim。
    """
    if b_max <= 0:
        print(f"    [Warning] b_max={b_max:.1f} <= 0, treating as nostim.")
        return True
    rel_range = (b_max - b_min) / b_max
    if rel_range < NOSTIM_REL_THRESH:
        print(f"    [Info] Relative range {rel_range:.3f} < {NOSTIM_REL_THRESH} "
              f"(b_max={b_max:.1f}, b_min={b_min:.1f}), treating as nostim.")
        return True
    return False


def process_single_trial(trial_dir, trial_id):
    """
    处理单个 trial

    Returns
    -------
    summary_row : dict
    events_rows : list of dict
    success : bool
    """
    clean_trial_outputs(trial_dir)

    stim_files = glob.glob(os.path.join(trial_dir, "*_Stim_Analog.tif"))
    json_files = glob.glob(os.path.join(trial_dir, "*_metadata.json"))

    if not stim_files or not json_files:
        return None, [], False

    # 1. 加载元数据
    with open(json_files[0], 'r') as f:
        meta = json.load(f)
        dt = meta.get('temporal_calibration', {}).get('frame_interval_sec', 1.0)
        nZ = meta.get('dimensions', {}).get('z_slices', 1)

    # dt=0 保护（fps=0 时 frame_interval_sec 也为 0）
    if dt <= 0:
        print(f"    [Warning] frame_interval_sec={dt} in metadata, defaulting to 1.0s. "
              f"Check Fiji metadata export for {trial_id}.")
        dt = 1.0

    # 2. 提取亮度序列
    with tiff.TiffFile(stim_files[0]) as tif:
        images = tif.asarray()
        brightness_trace = np.mean(images, axis=(1, 2))
        time_axis = np.arange(len(brightness_trace)) * dt

    # 3. 保存 brightness_trace
    trace_df = pd.DataFrame({
        'frame': np.arange(len(brightness_trace)),
        'time_sec': np.round(time_axis, 3),
        'brightness': brightness_trace
    })
    trace_path = os.path.join(trial_dir, f"{trial_id}_brightness_trace.csv")
    trace_df.to_csv(trace_path, index=False)
    print(f"    Saved brightness trace -> {trace_path}")

    # 4. 判断是否为无刺激 trial（修复 #8：改用相对阈值）
    b_max = float(np.max(brightness_trace))
    b_min = float(np.min(brightness_trace))

    if is_nostim(b_max, b_min):
        summary_row = {
            "trialID": trial_id,
            "stim_type": "nostim",
            "number_of_stim": 0,
            "initial_delay_sec": "",
            "avg_duration_sec": "",
            "avg_interval_sec": "",
            "strength": "",
            "pol_angle_list": ""
        }
        events_rows = []
        save_trial_outputs(trial_dir, trial_id, summary_row, events_rows)
        return summary_row, events_rows, True

    dynamic_thresh = b_min + 1 / (nZ + 1) * (b_max - b_min)
    baseline = b_min
    peak = b_max

    # 5. 分析刺激块
    stim_blocks = analyze_stim_blocks(brightness_trace, dt, dynamic_thresh, baseline, peak)

    # 6. 生成可视化图
    plot_path = os.path.join(trial_dir, f"{trial_id}_stim_trace.png")
    plot_stim_trace(brightness_trace, time_axis, stim_blocks, dynamic_thresh, trial_id, plot_path)
    print(f"    Saved stim plot       -> {plot_path}")

    # 7. 构建表
    summary_row = build_summary_row(trial_id, stim_blocks)
    events_rows = build_events_rows(trial_id, stim_blocks)

    # 8. 保存 trial-level CSV
    save_trial_outputs(trial_dir, trial_id, summary_row, events_rows)

    return summary_row, events_rows, True


def process_date_folders(root_path):
    """
    处理所有日期文件夹
    """
    date_folders = [
        f for f in glob.glob(os.path.join(root_path, 'Processed_TIF_*'))
        if os.path.isdir(f)
    ]
    date_folders.sort()

    for date_folder in date_folders:
        print(f"\n{'=' * 60}")
        print(f"Processing Date Folder: {os.path.basename(date_folder)}")
        print(f"{'=' * 60}")

        clean_day_outputs(date_folder)

        all_summary_data = []
        all_events_data = []

        trial_dirs = [d for d in glob.glob(os.path.join(date_folder, '*')) if os.path.isdir(d)]
        trial_dirs.sort()

        for trial_dir in tqdm(trial_dirs, desc="Processing Trials"):
            trial_id = os.path.basename(trial_dir)

            summary_row, events_rows, success = process_single_trial(trial_dir, trial_id)

            if success:
                if summary_row is not None:
                    all_summary_data.append(summary_row)
                if events_rows:
                    all_events_data.extend(events_rows)
                print(f"  ✓ {trial_id} | {len(events_rows)} stimuli detected")
            else:
                print(f"  ✗ {trial_id} | Skipped (missing files)")

        save_day_outputs(date_folder, all_summary_data, all_events_data)

        print(f"\n{'=' * 60}\n")


# ================== 主程序 ==================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stimulus Analyzer v3.2")
    print("=" * 60 + "\n")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path does not exist: {DATA_PATH}")
        raise SystemExit(1)

    process_date_folders(DATA_PATH)

    print("\n" + "=" * 60)
    print("All processing complete!")
    print("=" * 60 + "\n")
