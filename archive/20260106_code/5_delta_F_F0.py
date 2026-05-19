import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

# --- 配置部分 ---
ROOT_DIR = r'I:\Calcium_imaging_process\motion_correction_128'

def get_trial_context(json_path):
    """从 JSON 提取元数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return {
        "trial_id": meta.get('filename'),
        "fps": meta.get('temporal_calibration', {}).get('fps', 0.8333),
        "params": meta.get('stimulation', {}).get('stim_parameters', {})
    }

def get_name_mapping(folder_path):
    """适配映射文件名：Overlay Elements of ... .csv"""
    folder = Path(folder_path)
    # 查找符合命名规则的文件
    mapping_files = list(folder.glob("Overlay Elements of *.csv"))
    
    if not mapping_files:
        return {}
    
    try:
        df_map = pd.read_csv(mapping_files[0])
        # 将 Index (0-based) 映射为 Mean_ID (1-based)
        mapping = {}
        for _, row in df_map.iterrows():
            mean_key = "Mean" + str(int(row['Index']) + 1)
            mapping[mean_key] = row['Name']
        return mapping
    except Exception as e:
        print(f"      [警告] 映射解析失败: {e}")
        return {}

def process_raw_to_df_f(results_path, ctx):
    df_raw = pd.read_csv(results_path)
    folder_path = Path(results_path).parent
    
    # 1. 提取并自然排序 Mean 列
    all_cols = df_raw.columns.tolist()
    mean_cols = [c for c in all_cols if re.search(r'Mean\d+$', c)]
    mean_cols.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    if len(mean_cols) <= 5: return None

    # 2. 区分背景 (最后5列) 和神经元
    bg_cols = mean_cols[-5:]
    neuron_cols = mean_cols[:-5]
    
    # --- 修复 NoneType 报错的核心：安全提取参数 ---
    fps = ctx['fps']
    p = ctx['params']
    
    def safe_float(key, default):
        val = p.get(key)
        # 检查是否为 None 或空字符串
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # 尝试读取延迟时间，如果 JSON 里没有就默认 5.0 秒
    delay_sec = safe_float('initial_delay_sec', 5.0)
    baseline_frames = max(int(delay_sec * fps), 5)

    # 3. 计算背景和基线
    bg_mean_per_frame = df_raw[bg_cols].mean(axis=1) 
    f0_raw = df_raw[neuron_cols].iloc[:baseline_frames, :].mean() 
    bg_baseline_mean = bg_mean_per_frame.iloc[:baseline_frames].mean() 

    # 4. 信噪比检查
    min_signal_threshold = bg_baseline_mean * 1.1 
    valid_rois = f0_raw[f0_raw > min_signal_threshold].index.tolist()
    
    if not valid_rois:
        return None

    # 5. Delta F / F0 计算
    roi_corrected = df_raw[valid_rois].sub(bg_mean_per_frame, axis=0)
    f0_corrected = roi_corrected.iloc[:baseline_frames, :].mean()
    
    offset = 5.0 
    df_f = roi_corrected.sub(f0_corrected, axis=1).div(f0_corrected + offset, axis=1)
    df_f = df_f.clip(lower=-0.8) #

    # 6. 替换表头
    mapping = get_name_mapping(folder_path)
    if mapping:
        df_f.columns = [mapping.get(col, col) for col in df_f.columns]

    return df_f

def main():
    root = Path(ROOT_DIR)
    # 递归查找所有子目录下的 _updated.json
    for j_file in root.rglob("*_updated.json"):
        results_path = j_file.parent / "Results.csv"
        if not results_path.exists(): continue
            
        try:
            # 运行前清除上一次产生的 _preprocessed.csv
            for old in j_file.parent.glob("*_preprocessed.csv"):
                old.unlink()

            ctx = get_trial_context(j_file)
            df_f_sequence = process_raw_to_df_f(results_path, ctx)
            
            if df_f_sequence is not None:
                # 生成新的预处理文件名
                out_name = j_file.name.replace('_updated.json', '_preprocessed.csv')
                out_path = j_file.parent / out_name
                df_f_sequence.to_csv(out_path, index=False)
                print(f"    [完成] {j_file.parent.name} | 有效ROI: {df_f_sequence.shape[1]}")
            else:
                print(f"    [跳过] {j_file.parent.name}: 无有效 ROI")
        except Exception as e:
            print(f"    [失败] {j_file.name}: {e}")

if __name__ == "__main__":
    main()