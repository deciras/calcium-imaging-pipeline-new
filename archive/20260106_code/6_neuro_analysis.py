import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from scipy.optimize import curve_fit

# --- 配置部分 ---
ROOT_DIR = r'I:\Calcium_imaging_process\motion_correction_128'

def get_meta(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    fps = meta.get('temporal_calibration', {}).get('fps', 0.8333)
    stim_type = meta.get('stimulation', {}).get('stim_type', 'nostim')
    params = meta.get('stimulation', {}).get('stim_parameters', {})
    return fps, stim_type, params

def analyze_trace(trace, fps, stim_type, params):
    """深度量化神经元响应特征"""
    time_axis = np.arange(len(trace)) / fps
    results = {}
    
    # --- 基础统计 (所有组通用) ---
    results['Mean_dFF'] = np.mean(trace)
    results['Std_dFF'] = np.std(trace)
    
    if stim_type == 'nostim' or not params.get('initial_delay_sec'):
        # 对照组：主要关注自发放电频率或基线稳定性
        results['Is_Control'] = True
        results['Peak_dFF'] = np.max(trace)
        results['Status'] = "Control/Baseline"
    else:
        # 刺激组：计算刺激锁定响应
        results['Is_Control'] = False
        delay = float(params.get('initial_delay_sec', 0))
        duration = float(params.get('duration_sec', 1.2))
        interval = float(params.get('interval_sec', 60))
        n_stim = int(params.get('number_of_stim', 1))
        
        # 提取基线期 (刺激开始前)
        baseline_trace = trace[time_axis < delay]
        b_mean = np.mean(baseline_trace) if len(baseline_trace)>0 else 0
        b_std = np.std(baseline_trace) if len(baseline_trace)>0 else 0.01
        
        # 循环计算每一次刺激的响应
        trial_peaks = []
        for i in range(n_stim):
            t_start = delay + i * interval
            t_end = t_start + duration + 5.0  # 观察刺激后5秒内的反应
            
            mask = (time_axis >= t_start) & (time_axis < t_end)
            if any(mask):
                trial_peaks.append(np.max(trace[mask]))
        
        # 核心指标
        results['Peak_dFF'] = np.max(trial_peaks) if trial_peaks else np.max(trace)
        # 响应显著性 (Z-Score)
        results['Z_Score'] = (results['Peak_dFF'] - b_mean) / b_std
        # 响应可靠性 (有多少次刺激引起了超过 3*std 的反应)
        results['Reliability'] = sum(1 for p in trial_peaks if (p - b_mean) > 3*b_std) / n_stim if n_stim > 0 else 0
        results['Status'] = "Stimulated"

    return results

def main():
    root = Path(ROOT_DIR)
    all_summary = []

    # 寻找所有的预处理数据
    for csv_file in root.rglob("*_preprocessed.csv"):
        json_file = csv_file.parent / csv_file.name.replace('_preprocessed.csv', '_updated.json')
        if not json_file.exists(): continue
        
        print(f"正在深度分析: {csv_file.parent.name}")
        fps, stim_type, params = get_meta(json_file)
        df_f = pd.read_csv(csv_file)
        
        trial_data = []
        for roi_name in df_f.columns:
            metrics = analyze_trace(df_f[roi_name].values, fps, stim_type, params)
            metrics['ROI_ID'] = roi_name
            metrics['FileName'] = csv_file.parent.name
            metrics['Stim_Type'] = stim_type
            trial_data.append(metrics)
            all_summary.append(metrics)
        
        # 保存单个文件夹结果
        pd.DataFrame(trial_data).to_csv(csv_file.parent / "neuro_metrics_detailed.csv", index=False)

    # 保存总汇总表
    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(root / "Global_Neuro_Analysis_Results.csv", index=False)
        print(f"\n成功！汇总了 {len(summary_df)} 个神经元数据至根目录。")

if __name__ == "__main__":
    main()