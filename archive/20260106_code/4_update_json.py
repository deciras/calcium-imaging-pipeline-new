import pandas as pd
import json
import os
import ast
from pathlib import Path

# --- 配置部分 ---
ROOT_DIR = r'I:\Calcium_imaging_process\motion_correction_128'  # 包含所有 Processed_TIF_YYYYMMDD 的上层目录
SUFFIX = "_updated.json"  # 新生成文件的后缀，用于区分和后续删除

def process_folder(folder_path):
    """处理单个 Processed_TIF 文件夹"""
    csv_file = folder_path / 'stim_map.csv'
    if not csv_file.exists():
        print(f"跳过：{folder_path.name} (未找到 stim_map.csv)")
        return

    print(f"正在处理文件夹: {folder_path.name}")

    # 1. 删除该文件夹下所有旧的更新文件 (*_updated.json)
    for old_file in folder_path.rglob(f"*{SUFFIX}"):
        old_file.unlink()
        # print(f"  已清理旧文件: {old_file.name}")

    # 2. 读取 CSV 数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"  [错误] 读取 CSV 失败: {e}")
        return

    # 3. 遍历 CSV 行并生成新 JSON
    for _, row in df.iterrows():
        trial_id = str(row['trialID']).strip()
        if not trial_id: continue

        # 寻找原始 JSON 文件 (排除我们自己生成的 _updated.json)
        # 假设原文件名为 trialID_metadata.json
        original_json = folder_path / trial_id / f"{trial_id}_metadata.json"
        
        # 如果 trialID 文件夹不存在，尝试在当前目录下直接寻找
        if not original_json.exists():
            original_json = folder_path / f"{trial_id}_metadata.json"

        if original_json.exists():
            new_json_path = original_json.parent / f"{trial_id}{SUFFIX}"
            create_updated_json(original_json, new_json_path, row)
            print(f"  [生成] {new_json_path.name}")
        else:
            print(f"  [警告] 未找到原始文件: {trial_id}_metadata.json")


def create_updated_json(src_path, dest_path, row_data):
    """读取原 JSON，对每个字段进行空值检查并合并"""
    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 获取刺激类型，如果为空则设为 "none"
    stim_type = str(row_data['stim_type']).lower() if not pd.isna(row_data['stim_type']) else "none"

    # 2. 解析 pol_angle_list，如果为空或格式错误则设为 None (null)
    pol_list_str = row_data.get('pol_angle_list', None)
    pol_list = None
    if isinstance(pol_list_str, str) and pol_list_str.strip():
        try:
            pol_list = ast.literal_eval(pol_list_str)
        except:
            pol_list = None

    # 3. 构造参数字典，对每个数值条目进行 pd.isna 检查
    # 如果 CSV 单元格是空的，这些值在 JSON 中会变成 null
    params = {
        "duration_sec": float(row_data['duration_sec']) if not pd.isna(row_data.get('duration_sec')) else None,
        "initial_delay_sec": float(row_data['initial_delay_sec']) if not pd.isna(row_data.get('initial_delay_sec')) else None,
        "interval_sec": float(row_data['interval_sec']) if not pd.isna(row_data.get('interval_sec')) else None,
        "number_of_stim": int(row_data['number_of_stim']) if not pd.isna(row_data.get('number_of_stim')) else None,
        "strength": float(row_data['strength']) if not pd.isna(row_data.get('strength')) else None,
        "pol_angle_list": pol_list
    }

    # 4. 更新数据结构
    data['stimulation'] = {
        "stim_type": stim_type,
        "stim_parameters": params
    }

    # 5. 写入新文件
    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def update_json_files():
    root = Path(ROOT_DIR)
    # 搜索所有以 Processed_TIF_ 开头的文件夹
    tif_folders = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('Processed_TIF_')]
    
    if not tif_folders:
        print(f"在 {root.absolute()} 下未找到 Processed_TIF_ 文件夹")
        return

    for folder in tif_folders:
        process_folder(folder)

    print("\n所有操作已完成。")

if __name__ == "__main__":
    update_json_files()