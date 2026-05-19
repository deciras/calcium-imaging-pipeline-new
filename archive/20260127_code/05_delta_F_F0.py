"""
文件结构:
motion_correction_128/ (ROOT_DIR)
└── Processed_TIF_20250115/          # 日期文件夹
    └── Trial_001/                   # 单次实验文件夹
        ├── Results.csv              # 【必须】由 ImageJ/Fiji 导出的原始荧光数据
        ├── Trial_001_updated.json   # 【必须】包含 FPS 和刺激参数的元数据
        └── Overlay Elements of ... .csv # 【可选】用于将 "Mean1" 映射为具体的 "ROI_Name"
"""

# 修改说明：
# 1. 自动识别最后 5 列作为 Background ROI。
# 2. 计算这 5 列的行均值（每一帧的背景水平）。
# 3. 用所有 ROI 减去该背景均值，实现空间背景校正。
# 4. 基于校正后的信号计算 dF/F0。

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ================= 配置区域 =================
ROOT_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/Processed_TIF' 
OFFSET = 5.0  
LOWER = -0.8  
# ===========================================

def get_trial_context(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fps = data.get('fps', 30.0)
    stim_params = data.get('stimulation', {}).get('stim_parameters', {})
    initial_delay = stim_params.get('initial_delay_sec')
    if initial_delay is None: initial_delay = 5.0
    return {'fps': fps, 'initial_delay_sec': float(initial_delay), 'folder_path': json_path.parent}

def get_name_mapping(folder_path):
    pattern = "Overlay Elements of *.csv"
    match = list(folder_path.glob(pattern))
    if not match: return None
    try:
        df_meta = pd.read_csv(match[0])
        name_col = next((c for c in df_meta.columns if 'Name' in c or 'Label' in c), None)
        if name_col:
            return {f"Mean{i+1}": str(name).strip() for i, name in enumerate(df_meta[name_col])}
    except: pass
    return None

def visualize_bg_comparison(time_axis, pure_raw, bg_subtracted, roi_name, save_path):
    """
    绘制双 Y 轴对比图：
    左轴 (Black): 原始荧光 (Pure Raw, NO BG subtraction)
    右轴 (red): 背景减除后的荧光 (Background Subtracted)
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # --- 1. 绘制原始数据 (左轴 - 黑色) ---
    ax1.plot(time_axis, pure_raw, color='blue', alpha=0.7, label='Original Raw', linewidth=1.2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Original Intensity", color='blue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- 2. 绘制背景减除后的数据 (右轴 - 蓝色) ---
    ax2 = ax1.twinx()
    ax2.plot(time_axis, bg_subtracted, color='red', alpha=0.7, label='BG Subtracted', linewidth=1.2)
    ax2.set_ylabel('Subtracted Intensity', color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f"Background Subtraction Comparison: {roi_name}")
    
    # 合并图例
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, lbs1 + lbs2, loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def process_raw_to_df_f(csv_path, ctx):
    # 读取所有 Mean 列
    df_full = pd.read_csv(csv_path)
    roi_cols = [c for c in df_full.columns if c.startswith('Mean')]
    if not roi_cols: return None
    
    # 提取所有 ROI 数据 (包含最后的背景列)
    all_roi_data = df_full[roi_cols].copy()
    num_bg_cols = 5
    
    # 区分“神经元列”和“背景列”
    neuron_cols = roi_cols[:-num_bg_cols] if len(roi_cols) > num_bg_cols else roi_cols
    bg_cols = roi_cols[-num_bg_cols:] if len(roi_cols) > num_bg_cols else []

    # --- 计算用于 dF/F0 的背景减除数据 ---
    if bg_cols:
        bg_mean = all_roi_data[bg_cols].mean(axis=1)
        roi_corrected = all_roi_data[neuron_cols].sub(bg_mean, axis=0)
    else:
        roi_corrected = all_roi_data[neuron_cols]

    # --- 计算 dF/F0 ---
    fps = ctx['fps']
    baseline_frames = int(ctx['initial_delay_sec'] * fps)
    if baseline_frames <= 0: baseline_frames = 1
    
    f0 = roi_corrected.iloc[:baseline_frames, :].mean()
    df_f = roi_corrected.sub(f0, axis=1).div(f0 + OFFSET, axis=1)
    df_f = df_f.clip(lower=LOWER)

# --- 绘图逻辑：Pure Raw vs BG-Subtracted ---
    trace_save_dir = ctx['folder_path'] / "Traces_BG_Correction"
    trace_save_dir.mkdir(exist_ok=True)
    
    mapping = get_name_mapping(ctx['folder_path'])
    time_axis = np.arange(len(all_roi_data)) / fps

    for col in neuron_cols:
        roi_display_name = mapping.get(col, col) if mapping else col
        safe_name = "".join([c for c in roi_display_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        save_path = trace_save_dir / f"{safe_name}_BG_Comp.pdf"
        
        # 修改此处调用：传入 all_roi_data[col] 和 roi_corrected[col]
        visualize_bg_comparison(
            time_axis, 
            all_roi_data[col],     # 原始值
            roi_corrected[col],    # 背景减除后的值
            roi_display_name, 
            save_path
        )

    # 导出 CSV 时使用 Mapping
    if mapping:
        df_f.columns = [mapping.get(col, col) for col in df_f.columns]
    return df_f

def main():
    root = Path(ROOT_DIR)
    if not root.exists(): return
    found_files = list(root.rglob("*_updated.json"))
    for j_file in found_files:
        results_path = j_file.parent / "Results.csv"
        if not results_path.exists(): continue
        print(f"Processing: {j_file.parent.name}")
        try:
            ctx = get_trial_context(j_file)
            df_f_res = process_raw_to_df_f(results_path, ctx)
            if df_f_res is not None:
                out_name = j_file.name.replace("_updated.json", "_preprocessed.csv")
                df_f_res.to_csv(j_file.parent / out_name, index=False)
        except Exception as e:
            print(f"  [Error] {j_file.parent.name}: {e}")

if __name__ == "__main__":
    main()