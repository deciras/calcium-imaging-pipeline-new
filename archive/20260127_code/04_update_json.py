import pandas as pd
import json
import os
import ast
from pathlib import Path

# ================= 配置区域 =================
# 包含所有 Processed_TIF_YYYYMMDD 的上层目录
ROOT_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/Processed_TIF'  
SUFFIX = "_updated.json"  # 新生成文件的后缀
# ===========================================

def process_folder(folder_path):
    """处理单个 Processed_TIF 文件夹"""
    csv_file = folder_path / 'stim_map.csv'
    
    # 检查 CSV 是否存在且不是隐藏文件
    if not csv_file.exists() or csv_file.name.startswith('.'):
        print(f"跳过：{folder_path.name} (未找到有效的 stim_map.csv)")
        return

    print(f"正在处理文件夹: {folder_path.name}")

    # 1. 清理旧的更新文件，同时跳过 ._ 隐藏文件
    for old_file in folder_path.rglob(f"*{SUFFIX}"):
        if not old_file.name.startswith('.'):
            try:
                old_file.unlink()
            except Exception as e:
                print(f"  [警告] 无法删除旧文件 {old_file.name}: {e}")

    # 2. 读取 CSV 数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"  [错误] 读取 CSV 失败: {e}")
        return

    # 3. 遍历 CSV 行并生成新 JSON
    for _, row in df.iterrows():
        trial_id = str(row['trialID']).strip()
        if not trial_id or trial_id == 'nan': 
            continue

        # 寻找原始 JSON 文件
        # 优先级 1: trialID/trialID_metadata.json
        original_json = folder_path / trial_id / f"{trial_id}_metadata.json"
        
        # 优先级 2: folder_path/trialID_metadata.json
        if not original_json.exists():
            original_json = folder_path / f"{trial_id}_metadata.json"

        # 检查文件存在性，并排除 ._ 隐藏文件
        if original_json.exists() and not original_json.name.startswith('.'):
            new_json_path = original_json.parent / f"{trial_id}{SUFFIX}"
            create_updated_json(original_json, new_json_path, row)
            print(f"  [生成] {new_json_path.name}")
        else:
            # 如果不是因为隐藏文件导致的缺失，则打印警告
            if not original_json.name.startswith('.'):
                print(f"  [警告] 未找到原始文件: {trial_id}_metadata.json")


def create_updated_json(src_path, dest_path, row_data):
    """读取原 JSON，对每个字段进行空值检查，并执行 180-角度 的镜像转换"""
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 获取刺激类型
        stim_type = str(row_data['stim_type']).lower() if not pd.isna(row_data['stim_type']) else "none"

        # 2. 解析 pol_angle_list 并执行镜像转换 (180 - angle)
        pol_list_str = row_data.get('pol_angle_list', None)
        pol_list = None
        if isinstance(pol_list_str, str) and pol_list_str.strip():
            try:
                # 处理 CSV 中可能的字符串形式列表，如 "[0, 90, 180]"
                raw_list = ast.literal_eval(pol_list_str)
                if isinstance(raw_list, list):
                    # --- 核心修改：执行镜像转换 ---
                    pol_list = [180 - float(angle) for angle in raw_list]
                else:
                    pol_list = raw_list
            except Exception as e:
                print(f"  [警告] 解析角度列表失败: {e}")
                pol_list = None

        # 3. 构造参数字典，处理空值(NaN)为 None(null)
        params = {
            "duration_sec": float(row_data['duration_sec']) if not pd.isna(row_data.get('duration_sec')) else None,
            "initial_delay_sec": float(row_data['initial_delay_sec']) if not pd.isna(row_data.get('initial_delay_sec')) else None,
            "interval_sec": float(row_data['interval_sec']) if not pd.isna(row_data.get('interval_sec')) else None,
            "number_of_stim": int(row_data['number_of_stim']) if not pd.isna(row_data.get('number_of_stim')) else None,
            "strength": float(row_data['strength']) if not pd.isna(row_data.get('strength')) else None,
            "pol_angle_list": pol_list
        }

        # 4. 更新或创建 stimulation 字段
        data['stimulation'] = {
            "stim_type": stim_type,
            "stim_parameters": params
        }

        # 5. 写入新文件
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"  [错误] 处理文件 {src_path.name} 时出错: {e}")


def update_json_files():
    root = Path(ROOT_DIR)
    
    if not root.exists():
        print(f"错误：根目录不存在 {ROOT_DIR}")
        return

    # 搜索所有以 Processed_TIF_ 开头的文件夹，同时排除隐藏文件夹
    tif_folders = [
        d for d in root.iterdir() 
        if d.is_dir() and d.name.startswith('Processed_TIF_') and not d.name.startswith('.')
    ]
    
    if not tif_folders:
        print(f"在 {root.absolute()} 下未找到有效的 Processed_TIF_ 文件夹")
        return

    print(f"找到 {len(tif_folders)} 个待处理文件夹。\n")

    for folder in tif_folders:
        process_folder(folder)

    print("\n所有操作已完成。")

if __name__ == "__main__":
    update_json_files()