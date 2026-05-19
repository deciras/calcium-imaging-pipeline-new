#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 还没试过，等我伪造一些数据（）


"""
dF/F0 计算脚本 v2.0
适配新的 metadata 结构（包含详细的 stim_events 列表）

文件结构:
motion_correction_128/ (ROOT_DIR)
└── Processed_TIF_20250115/          # 日期文件夹
    └── Trial_001/                   # 单次实验文件夹
        ├── Results.csv              # 【必须】由 ImageJ/Fiji 导出的原始荧光数据
        ├── Trial_001_updated.json   # 【必须】包含 FPS 和刺激参数的元数据
        └── Overlay Elements of ... .csv # 【可选】用于将 "Mean1" 映射为具体的 "ROI_Name"

修改说明：
1. 自动识别最后 5 列作为 Background ROI
2. 计算这 5 列的行均值（每一帧的背景水平）
3. 用所有 ROI 减去该背景均值，实现空间背景校正
4. 基于校正后的信号计算 dF/F0
5. 适配新的 metadata 结构（stim_events 列表）
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ================= 配置区域 =================
ROOT_DIR = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_Suite2p_Results' 
OFFSET = 5.0  # 防止除零的偏移量
LOWER = -0.8  # dF/F0 的下限
NUM_BG_COLS = 5  # 背景 ROI 数量
# ===========================================


def get_trial_context(json_path):
    """
    从 JSON 文件中提取试验上下文信息
    适配新的 metadata 结构
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 获取 FPS（优先从 temporal_calibration，其次从顶层 fps）
    fps = data.get('temporal_calibration', {}).get('fps')
    if fps is None:
        fps = data.get('fps', 30.0)
    fps = float(fps)
    
    # 2. 获取刺激信息
    stimulation = data.get('stimulation', {})
    stim_params = stimulation.get('stim_parameters', {})
    stim_events = stimulation.get('stim_events', [])
    
    # 3. 获取 initial_delay（优先从 stim_parameters，如果没有则从第一个事件计算）
    initial_delay = stim_params.get('initial_delay_sec')
    if initial_delay is None and stim_events:
        # 从第一个刺激事件的开始时间获取
        initial_delay = stim_events[0].get('start_time_sec', 5.0)
    if initial_delay is None:
        initial_delay = 5.0
    
    return {
        'fps': fps,
        'initial_delay_sec': float(initial_delay),
        'folder_path': json_path.parent,
        'stim_params': stim_params,
        'stim_events': stim_events,
        'stim_type': stimulation.get('stim_type', 'unknown')
    }


def get_name_mapping(folder_path):
    """
    从 Overlay Elements CSV 文件中读取 ROI 名称映射
    将 Mean1, Mean2... 映射为实际的 ROI 名称
    """
    pattern = "Overlay Elements of *.csv"
    match = list(folder_path.glob(pattern))
    if not match:
        return None
    
    try:
        df_meta = pd.read_csv(match[0])
        name_col = next((c for c in df_meta.columns if 'Name' in c or 'Label' in c), None)
        if name_col:
            return {f"Mean{i+1}": str(name).strip() for i, name in enumerate(df_meta[name_col])}
    except Exception as e:
        print(f"  [警告] 读取 Overlay Elements 文件失败: {e}")
    
    return None


def visualize_bg_comparison(time_axis, pure_raw, bg_subtracted, roi_name, save_path, stim_events=None):
    """
    绘制双 Y 轴对比图：
    左轴 (Blue): 原始荧光 (Pure Raw, NO BG subtraction)
    右轴 (Red): 背景减除后的荧光 (Background Subtracted)
    可选：标记刺激时间段
    """
    fig, ax1 = plt.subplots(figsize=(12, 4))
    
    # --- 1. 绘制原始数据 (左轴 - 蓝色) ---
    ax1.plot(time_axis, pure_raw, color='blue', alpha=0.7, label='Original Raw', linewidth=1.2)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Original Intensity", color='blue', fontweight='bold', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- 2. 绘制背景减除后的数据 (右轴 - 红色) ---
    ax2 = ax1.twinx()
    ax2.plot(time_axis, bg_subtracted, color='red', alpha=0.7, label='BG Subtracted', linewidth=1.2)
    ax2.set_ylabel('Subtracted Intensity', color='red', fontweight='bold', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # --- 3. 标记刺激时间段（如果提供） ---
    if stim_events:
        for event in stim_events:
            start_t = event.get('start_time_sec')
            end_t = event.get('end_time_sec')
            if start_t is not None and end_t is not None:
                ax1.axvspan(start_t, end_t, alpha=0.15, color='orange', 
                           label='Stimulus' if event == stim_events[0] else '')
    
    plt.title(f"Background Subtraction Comparison: {roi_name}", fontsize=12, fontweight='bold')
    
    # 合并图例
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, lbs1 + lbs2, loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def visualize_df_f(time_axis, df_f_data, roi_name, save_path, stim_events=None, ctx=None):
    """
    绘制 dF/F0 曲线，标记刺激时间段和 baseline 区域
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 绘制 dF/F0 曲线
    ax.plot(time_axis, df_f_data, color='black', linewidth=1.2, label='dF/F0')
    
    # 标记 baseline 区域
    baseline_end = ctx['initial_delay_sec'] if ctx else 5.0
    ax.axvspan(0, baseline_end, alpha=0.15, color='gray', label='Baseline (F0)')
    
    # 标记刺激时间段
    if stim_events:
        for event in stim_events:
            start_t = event.get('start_time_sec')
            end_t = event.get('end_time_sec')
            stim_idx = event.get('stim_index')
            if start_t is not None and end_t is not None:
                ax.axvspan(start_t, end_t, alpha=0.2, color='orange', 
                          label='Stimulus' if event == stim_events[0] else '')
                # 在刺激中点标注序号（如果刺激数量不太多）
                if len(stim_events) <= 20 and stim_idx is not None:
                    mid_t = (start_t + end_t) / 2
                    ax.text(mid_t, ax.get_ylim()[1] * 0.95, f"#{stim_idx}", 
                           ha='center', va='top', fontsize=8, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("dF/F0", fontsize=11, fontweight='bold')
    ax.set_title(f"dF/F0 Response: {roi_name}", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def process_raw_to_df_f(csv_path, ctx):
    """
    处理原始荧光数据，计算 dF/F0
    """
    # 读取所有 Mean 列
    df_full = pd.read_csv(csv_path)
    roi_cols = [c for c in df_full.columns if c.startswith('Mean')]
    
    if not roi_cols:
        print(f"  [错误] 未找到任何 Mean 列")
        return None
    
    # 提取所有 ROI 数据 (包含最后的背景列)
    all_roi_data = df_full[roi_cols].copy()
    
    # 区分"神经元列"和"背景列"
    neuron_cols = roi_cols[:-NUM_BG_COLS] if len(roi_cols) > NUM_BG_COLS else roi_cols
    bg_cols = roi_cols[-NUM_BG_COLS:] if len(roi_cols) > NUM_BG_COLS else []
    
    print(f"  [Info] 总 ROI: {len(roi_cols)}, 神经元 ROI: {len(neuron_cols)}, 背景 ROI: {len(bg_cols)}")

    # --- 计算背景减除 ---
    if bg_cols:
        bg_mean = all_roi_data[bg_cols].mean(axis=1)
        roi_corrected = all_roi_data[neuron_cols].sub(bg_mean, axis=0)
        print(f"  [Info] 已使用 {len(bg_cols)} 个背景 ROI 进行背景校正")
    else:
        roi_corrected = all_roi_data[neuron_cols]
        print(f"  [Info] 无背景 ROI，跳过背景校正")

    # --- 计算 dF/F0 ---
    fps = ctx['fps']
    baseline_frames = int(ctx['initial_delay_sec'] * fps)
    if baseline_frames <= 0:
        baseline_frames = 1
    
    # 确保 baseline 不超过总帧数
    baseline_frames = min(baseline_frames, len(roi_corrected))
    
    print(f"  [Info] Baseline 帧数: {baseline_frames} (前 {ctx['initial_delay_sec']:.2f} 秒)")
    
    # 计算 F0（baseline 的平均值）
    f0 = roi_corrected.iloc[:baseline_frames, :].mean()
    
    # 计算 dF/F0
    df_f = roi_corrected.sub(f0, axis=1).div(f0 + OFFSET, axis=1)
    df_f = df_f.clip(lower=LOWER)

    # --- 绘图：背景校正对比 ---
    trace_save_dir = ctx['folder_path'] / "Traces_BG_Correction"
    trace_save_dir.mkdir(exist_ok=True)
    
    mapping = get_name_mapping(ctx['folder_path'])
    time_axis = np.arange(len(all_roi_data)) / fps
    stim_events = ctx.get('stim_events', [])

    for col in neuron_cols:
        roi_display_name = mapping.get(col, col) if mapping else col
        safe_name = "".join([c for c in roi_display_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        
        # 背景校正对比图
        bg_comp_path = trace_save_dir / f"{safe_name}_BG_Comp.pdf"
        visualize_bg_comparison(
            time_axis, 
            all_roi_data[col],      # 原始值
            roi_corrected[col],     # 背景减除后的值
            roi_display_name, 
            bg_comp_path,
            stim_events=stim_events
        )
    
    # --- 绘图：dF/F0 曲线 ---
    df_f_save_dir = ctx['folder_path'] / "Traces_dF_F0"
    df_f_save_dir.mkdir(exist_ok=True)
    
    for col in neuron_cols:
        roi_display_name = mapping.get(col, col) if mapping else col
        safe_name = "".join([c for c in roi_display_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        
        # dF/F0 曲线图
        df_f_path = df_f_save_dir / f"{safe_name}_dF_F0.pdf"
        visualize_df_f(
            time_axis,
            df_f[col],
            roi_display_name,
            df_f_path,
            stim_events=stim_events,
            ctx=ctx
        )

    # --- 导出 CSV 时使用 Mapping ---
    if mapping:
        df_f.columns = [mapping.get(col, col) for col in df_f.columns]
    
    return df_f


def main():
    """
    主函数：递归处理所有 *_updated.json 文件
    """
    root = Path(ROOT_DIR)
    
    if not root.exists():
        print(f"错误：根目录不存在 {ROOT_DIR}")
        return
    
    found_files = list(root.rglob("*_updated.json"))
    
    if not found_files:
        print(f"未找到任何 *_updated.json 文件")
        return
    
    print(f"找到 {len(found_files)} 个待处理文件")
    print("="*60)
    
    success_count = 0
    error_count = 0
    
    for j_file in found_files:
        results_path = j_file.parent / "Results.csv"
        
        if not results_path.exists():
            print(f"[跳过] {j_file.parent.name}: 未找到 Results.csv")
            continue
        
        print(f"\n[处理] {j_file.parent.name}")
        
        try:
            ctx = get_trial_context(j_file)
            print(f"  [Info] FPS: {ctx['fps']}, Initial Delay: {ctx['initial_delay_sec']:.2f}s")
            print(f"  [Info] 刺激类型: {ctx['stim_type']}, 刺激数量: {len(ctx['stim_events'])}")
            
            df_f_res = process_raw_to_df_f(results_path, ctx)
            
            if df_f_res is not None:
                out_name = j_file.name.replace("_updated.json", "_preprocessed.csv")
                output_path = j_file.parent / out_name
                df_f_res.to_csv(output_path, index=False)
                print(f"  [成功] 保存至: {out_name}")
                success_count += 1
            else:
                print(f"  [错误] 处理失败")
                error_count += 1
                
        except Exception as e:
            print(f"  [错误] {j_file.parent.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    print("\n" + "="*60)
    print(f"处理完成！成功: {success_count}, 失败: {error_count}")


if __name__ == "__main__":
    main()