#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
刺激信号分析脚本 v2.0
功能：
1. 从 *_Stim_Analog.tif 提取刺激模拟信号
2. 精确识别每个刺激的起始和结束时间点（考虑边界帧的部分刺激）
3. 生成详细的刺激事件表 (stim_events.csv)
4. 生成概览表 (stim_map.csv)
5. 为每个trial生成刺激曲线可视化图
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

# 配置matplotlib支持中文
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ================== 配置参数 ==================
DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus'

# ================== 核心函数 ==================

def estimate_precise_boundary(brightness_trace, frame_idx, baseline, peak, is_start=True):
    """
    估算刺激在边界帧内的精确时间点
    
    Parameters:
    -----------
    brightness_trace : array
        完整的亮度序列
    frame_idx : int
        边界帧的索引
    baseline : float
        基线亮度值
    peak : float
        刺激峰值亮度
    is_start : bool
        True表示起始边界，False表示结束边界
    
    Returns:
    --------
    offset : float
        帧内偏移量 (0-1之间)，表示刺激在该帧内开始/结束的相对位置
    """
    if frame_idx < 0 or frame_idx >= len(brightness_trace):
        return 0.0 if is_start else 1.0
    
    brightness = brightness_trace[frame_idx]
    
    # 计算该帧的刺激占比
    if peak - baseline > 0:
        ratio = (brightness - baseline) / (peak - baseline)
        ratio = np.clip(ratio, 0.0, 1.0)
    else:
        ratio = 0.5  # 如果无法区分，默认取中点
    
    if is_start:
        # 起始帧：假设刺激从该帧的某个位置开始
        # ratio越大，说明这一帧刺激占比越多，实际开始越早
        return 1.0 - ratio
    else:
        # 结束帧：假设刺激在该帧的某个位置结束
        # ratio越大，说明这一帧刺激占比越多，实际结束越晚
        return ratio


def analyze_stim_blocks(brightness_trace, dt, dynamic_thresh, baseline, peak):
    """
    分析刺激块，精确计算每个刺激的起止时间
    
    Returns:
    --------
    stim_blocks : list of dict
        每个刺激的详细信息
    """
    is_stim_on = brightness_trace > dynamic_thresh
    stim_indices = np.where(is_stim_on)[0]
    
    if len(stim_indices) == 0:
        return []
    
    # 识别连续的刺激块
    diffs = np.diff(stim_indices)
    split_points = np.where(diffs > 1)[0] + 1
    blocks = np.split(stim_indices, split_points)
    
    stim_blocks = []
    
    for block_idx, block in enumerate(blocks):
        if len(block) == 0:
            continue
        
        start_frame = block[0]
        end_frame = block[-1]
        
        # 估算起始帧内的精确偏移
        start_offset = estimate_precise_boundary(
            brightness_trace, start_frame, baseline, peak, is_start=True
        )
        
        # 估算结束帧内的精确偏移
        end_offset = estimate_precise_boundary(
            brightness_trace, end_frame, baseline, peak, is_start=False
        )
        
        # 计算精确的时间点
        start_time = (start_frame + start_offset) * dt
        end_time = (end_frame + end_offset) * dt
        duration = end_time - start_time
        
        stim_blocks.append({
            'stim_index': block_idx + 1,
            'start_frame': start_frame,
            'end_frame': end_frame,
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
    # 自动调整图片宽度，避免超过matplotlib限制
    total_time = time_axis[-1] if len(time_axis) > 0 else 1
    max_width = 50  # 最大宽度（英寸）
    fig_width = min(max(12, total_time / 10), max_width)  # 根据时长动态调整，但不超过50英寸
    
    # 如果数据点太多，进行降采样以提高绘图效率
    max_points = 10000
    if len(brightness_trace) > max_points:
        downsample_factor = len(brightness_trace) // max_points + 1
        brightness_trace_plot = brightness_trace[::downsample_factor]
        time_axis_plot = time_axis[::downsample_factor]
        print(f"    [Info] Downsampling trace for plotting: {len(brightness_trace)} -> {len(brightness_trace_plot)} points")
    else:
        brightness_trace_plot = brightness_trace
        time_axis_plot = time_axis
    
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    
    # 绘制亮度曲线
    ax.plot(time_axis_plot, brightness_trace_plot, 'b-', linewidth=0.8, label='Brightness', alpha=0.7)
    
    # 绘制阈值线
    ax.axhline(y=dynamic_thresh, color='r', linestyle='--', linewidth=1, 
               label=f'Threshold ({dynamic_thresh:.1f})')
    
    # 标记每个刺激块
    for block in stim_blocks:
        start_t = block['start_time_sec']
        end_t = block['end_time_sec']
        ax.axvspan(start_t, end_t, alpha=0.3, color='orange', 
                   label='Stimulus' if block == stim_blocks[0] else '')
        
        # 在刺激中点标注序号（仅当刺激数量不太多时标注）
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
    
    # 使用try-except捕获可能的保存错误
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    except ValueError as e:
        print(f"    [Warning] Failed to save full resolution plot: {e}")
        # 尝试使用更低的DPI
        try:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"    [Info] Saved with reduced DPI (100)")
        except:
            print(f"    [Error] Could not save plot for {trial_id}")
    
    plt.close()


def process_single_trial(trial_dir, trial_id):
    """
    处理单个trial
    
    Returns:
    --------
    summary_row : dict
        该trial的概览信息
    events_rows : list of dict
        该trial的详细刺激事件列表
    success : bool
        是否成功处理
    """
    stim_files = glob.glob(os.path.join(trial_dir, "*_Stim_Analog.tif"))
    json_files = glob.glob(os.path.join(trial_dir, "*_metadata.json"))
    
    if not stim_files or not json_files:
        return None, [], False
    
    # 1. 加载元数据
    with open(json_files[0], 'r') as f:
        meta = json.load(f)
        dt = meta.get('temporal_calibration', {}).get('frame_interval_sec', 1.0)
        nZ = meta.get('dimensions', {}).get('z_slices', 1)
    
    # 2. 提取亮度序列
    with tiff.TiffFile(stim_files[0]) as tif:
        images = tif.asarray()
        brightness_trace = np.mean(images, axis=(1, 2))
        time_axis = np.arange(len(brightness_trace)) * dt
    
    # 3. 保存完整的 brightness_trace
    trace_df = pd.DataFrame({
        'frame': np.arange(len(brightness_trace)),
        'time_sec': np.round(time_axis, 3),
        'brightness': brightness_trace
    })
    trace_df.to_csv(os.path.join(trial_dir, f"{trial_id}_brightness_trace.csv"), index=False)
    
    # 4. 动态计算阈值
    b_max = np.max(brightness_trace)
    b_min = np.min(brightness_trace)
    if b_max - b_min < 50:
        # 如果信号变化太小，认为无刺激
        return {
            "trialID": trial_id,
            "stim_type": "nostim",
            "number_of_stim": 0,
            "initial_delay_sec": "",
            "avg_duration_sec": "",
            "avg_interval_sec": "",
            "strength": "",
            "pol_angle_list": ""
        }, [], True
    
    # 20% 动态阈值
    dynamic_thresh = b_min + 1 / (nZ + 1) * (b_max - b_min)
    baseline = b_min
    peak = b_max
    
    # 5. 分析刺激块
    stim_blocks = analyze_stim_blocks(brightness_trace, dt, dynamic_thresh, baseline, peak)
    
    # 6. 生成可视化图
    plot_path = os.path.join(trial_dir, f"{trial_id}_stim_trace.png")
    plot_stim_trace(brightness_trace, time_axis, stim_blocks, dynamic_thresh, 
                    trial_id, plot_path)
    
    # 7. 构建概览行
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
        # 初始延迟
        initial_delay = stim_blocks[0]['start_time_sec']
        summary_row['initial_delay_sec'] = round(initial_delay, 2)
        
        # 平均持续时间
        durations = [b['duration_sec'] for b in stim_blocks]
        avg_duration = np.mean(durations)
        summary_row['avg_duration_sec'] = round(avg_duration, 2)
        
        # 平均间隔 (onset-to-onset)
        if len(stim_blocks) > 1:
            intervals = [stim_blocks[i+1]['start_time_sec'] - stim_blocks[i]['start_time_sec'] 
                        for i in range(len(stim_blocks)-1)]
            avg_interval = np.mean(intervals)
            summary_row['avg_interval_sec'] = round(avg_interval, 2)
    else:
        summary_row['stim_type'] = "nostim"
    
    # 8. 构建详细事件行
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
    
    return summary_row, events_rows, True


def process_date_folders(root_path):
    """
    处理所有日期文件夹
    """
    date_folders = [f for f in glob.glob(os.path.join(root_path, 'Processed_TIF_*')) 
                    if os.path.isdir(f)]
    
    for date_folder in date_folders:
        print(f"\n{'='*60}")
        print(f"Processing Date Folder: {os.path.basename(date_folder)}")
        print(f"{'='*60}")
        
        all_summary_data = []
        all_events_data = []
        
        trial_dirs = [d for d in glob.glob(os.path.join(date_folder, '*')) if os.path.isdir(d)]
        trial_dirs.sort()
        
        for trial_dir in tqdm(trial_dirs, desc="Processing Trials"):
            trial_id = os.path.basename(trial_dir)
            
            summary_row, events_rows, success = process_single_trial(trial_dir, trial_id)
            
            if success:
                all_summary_data.append(summary_row)
                all_events_data.extend(events_rows)
                print(f"  ✓ {trial_id} | {len(events_rows)} stimuli detected")
            else:
                print(f"  ✗ {trial_id} | Skipped (missing files)")
        
        # 保存概览CSV
        if all_summary_data:
            df_summary = pd.DataFrame(all_summary_data)
            column_order = ["trialID", "stim_type", "number_of_stim", "initial_delay_sec", 
                          "avg_duration_sec", "avg_interval_sec", "strength", "pol_angle_list"]
            df_summary = df_summary.reindex(columns=column_order)
            summary_path = os.path.join(date_folder, 'stim_map.csv')
            df_summary.to_csv(summary_path, index=False)
            print(f"\n✓ Saved summary to: {summary_path}")
        
        # 保存详细事件CSV
        if all_events_data:
            df_events = pd.DataFrame(all_events_data)
            column_order = ["trialID", "stim_index", "start_time_sec", "end_time_sec", 
                          "duration_sec", "stim_type", "pol_angle", "strength"]
            df_events = df_events.reindex(columns=column_order)
            events_path = os.path.join(date_folder, 'stim_events.csv')
            df_events.to_csv(events_path, index=False)
            print(f"✓ Saved detailed events to: {events_path}")
        
        print(f"\n{'='*60}\n")


# ================== 主程序 ==================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Stimulus Analyzer v2.0")
    print("="*60 + "\n")
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path does not exist: {DATA_PATH}")
        exit(1)
    
    process_date_folders(DATA_PATH)
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60 + "\n")