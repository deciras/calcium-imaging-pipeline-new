#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
刺激数据更新脚本 v2.0
功能：
1. 根据手动更新的 stim_map.csv，更新 stim_events.csv 中的 stim_type, pol_angle, strength
2. 处理 180-angle 的镜像转换
3. 更新 JSON 文件，添加详细的刺激事件列表

结束之后需要画ROI,但按理来说应该suite2p生成了ROI
"""

import pandas as pd
import json
import os
import ast
from pathlib import Path

# ================= 配置区域 =================
# 包含所有 Processed_TIF_YYYYMMDD 的上层目录
ROOT_DIR = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_Suite2p_Results'  
SUFFIX = "_updated.json"  # 新生成文件的后缀
# ===========================================


def mirror_angle(angle):
    """执行镜像转换：180 - angle"""
    if angle is None or pd.isna(angle):
        return None
    return 180 - float(angle)


def parse_angle_list(angle_str):
    """
    解析角度列表字符串，支持多种格式
    返回列表或单个值
    """
    if pd.isna(angle_str) or not angle_str or str(angle_str).strip() == '':
        return None
    
    angle_str = str(angle_str).strip()
    
    try:
        # 尝试解析为列表格式 "[0, 90, 180]"
        parsed = ast.literal_eval(angle_str)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        else:
            return [float(parsed)]
    except:
        try:
            # 尝试解析为单个数值
            return [float(angle_str)]
        except:
            print(f"  [警告] 无法解析角度: {angle_str}")
            return None


def update_stim_events_csv(folder_path, stim_map_df):
    """
    根据 stim_map.csv 更新 stim_events.csv
    """
    events_file = folder_path / 'stim_events.csv'
    
    if not events_file.exists():
        print(f"  [跳过] 未找到 stim_events.csv")
        return None
    
    # 读取 stim_events.csv
    try:
        events_df = pd.read_csv(events_file)
    except Exception as e:
        print(f"  [错误] 读取 stim_events.csv 失败: {e}")
        return None
    
    print(f"  [更新] stim_events.csv ({len(events_df)} 条记录)")
    
    # 为每个 trial 更新信息
    for trial_id in events_df['trialID'].unique():
        # 从 stim_map 中找到该 trial 的信息
        map_row = stim_map_df[stim_map_df['trialID'] == trial_id]
        
        if map_row.empty:
            print(f"    [警告] {trial_id} 在 stim_map.csv 中未找到")
            continue
        
        map_row = map_row.iloc[0]
        
        # 获取基本信息
        stim_type = str(map_row['stim_type']) if not pd.isna(map_row['stim_type']) else ''
        strength = float(map_row['strength']) if not pd.isna(map_row.get('strength')) else None
        
        # 解析角度列表
        pol_angles = parse_angle_list(map_row.get('pol_angle_list'))
        
        # 获取该 trial 的所有事件
        trial_events = events_df[events_df['trialID'] == trial_id]
        num_stims = len(trial_events)
        
        # 更新每个刺激事件
        for idx, (event_idx, event_row) in enumerate(trial_events.iterrows()):
            # 更新 stim_type
            events_df.at[event_idx, 'stim_type'] = stim_type
            
            # 更新 strength
            if strength is not None:
                events_df.at[event_idx, 'strength'] = strength
            
            # 更新 pol_angle（带镜像转换）
            if pol_angles is not None:
                if len(pol_angles) == 1:
                    # 所有刺激使用同一个角度
                    original_angle = pol_angles[0]
                elif idx < len(pol_angles):
                    # 为每个刺激分配不同角度
                    original_angle = pol_angles[idx]
                else:
                    # 角度数量不足，循环使用
                    original_angle = pol_angles[idx % len(pol_angles)]
                
                # 执行镜像转换
                mirrored_angle = mirror_angle(original_angle)
                events_df.at[event_idx, 'pol_angle'] = mirrored_angle
    
    # 保存更新后的 stim_events.csv
    try:
        events_df.to_csv(events_file, index=False)
        print(f"  [保存] stim_events.csv 已更新")
        return events_df
    except Exception as e:
        print(f"  [错误] 保存 stim_events.csv 失败: {e}")
        return None


def create_updated_json(src_path, dest_path, row_data, events_df=None):
    """
    读取原 JSON，添加刺激信息和详细的事件列表
    """
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 获取刺激类型
        stim_type = str(row_data['stim_type']).lower() if not pd.isna(row_data['stim_type']) else "none"

        # 2. 解析 pol_angle_list 并执行镜像转换
        pol_list_str = row_data.get('pol_angle_list', None)
        pol_list = parse_angle_list(pol_list_str)
        
        # 执行镜像转换
        if pol_list is not None:
            pol_list_mirrored = [mirror_angle(angle) for angle in pol_list]
        else:
            pol_list_mirrored = None

        # 3. 构造基本参数字典
        params = {
            "duration_sec": float(row_data['avg_duration_sec']) if not pd.isna(row_data.get('avg_duration_sec')) else None,
            "initial_delay_sec": float(row_data['initial_delay_sec']) if not pd.isna(row_data.get('initial_delay_sec')) else None,
            "interval_sec": float(row_data['avg_interval_sec']) if not pd.isna(row_data.get('avg_interval_sec')) else None,
            "number_of_stim": int(row_data['number_of_stim']) if not pd.isna(row_data.get('number_of_stim')) else None,
            "strength": float(row_data['strength']) if not pd.isna(row_data.get('strength')) else None,
            "pol_angle_list": pol_list_mirrored
        }

        # 4. 添加详细的刺激事件列表（如果有 stim_events.csv 数据）
        stim_events_list = None
        if events_df is not None:
            trial_id = str(row_data['trialID']).strip()
            trial_events = events_df[events_df['trialID'] == trial_id]
            
            if not trial_events.empty:
                stim_events_list = []
                for _, event_row in trial_events.iterrows():
                    event_dict = {
                        "stim_index": int(event_row['stim_index']) if not pd.isna(event_row['stim_index']) else None,
                        "start_time_sec": float(event_row['start_time_sec']) if not pd.isna(event_row['start_time_sec']) else None,
                        "end_time_sec": float(event_row['end_time_sec']) if not pd.isna(event_row['end_time_sec']) else None,
                        "duration_sec": float(event_row['duration_sec']) if not pd.isna(event_row['duration_sec']) else None,
                        "stim_type": str(event_row['stim_type']) if not pd.isna(event_row['stim_type']) else None,
                        "pol_angle": float(event_row['pol_angle']) if not pd.isna(event_row['pol_angle']) else None,
                        "strength": float(event_row['strength']) if not pd.isna(event_row['strength']) else None
                    }
                    stim_events_list.append(event_dict)

        # 5. 更新或创建 stimulation 字段
        data['stimulation'] = {
            "stim_type": stim_type,
            "stim_parameters": params,
            "stim_events": stim_events_list  # 详细的刺激事件列表
        }

        # 6. 写入新文件
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"  [错误] 处理文件 {src_path.name} 时出错: {e}")


def process_folder(folder_path):
    """处理单个 Processed_TIF 文件夹"""
    csv_file = folder_path / 'stim_map.csv'
    
    # 检查 CSV 是否存在且不是隐藏文件
    if not csv_file.exists() or csv_file.name.startswith('.'):
        print(f"跳过：{folder_path.name} (未找到有效的 stim_map.csv)")
        return

    print(f"\n正在处理文件夹: {folder_path.name}")

    # 1. 清理旧的更新文件
    for old_file in folder_path.rglob(f"*{SUFFIX}"):
        if not old_file.name.startswith('.'):
            try:
                old_file.unlink()
                print(f"  [删除] 旧文件 {old_file.name}")
            except Exception as e:
                print(f"  [警告] 无法删除旧文件 {old_file.name}: {e}")

    # 2. 读取 stim_map.csv
    try:
        stim_map_df = pd.read_csv(csv_file)
        print(f"  [读取] stim_map.csv ({len(stim_map_df)} 条记录)")
    except Exception as e:
        print(f"  [错误] 读取 stim_map.csv 失败: {e}")
        return

    # 3. 更新 stim_events.csv
    events_df = update_stim_events_csv(folder_path, stim_map_df)

    # 4. 遍历 CSV 行并生成新 JSON
    json_count = 0
    for _, row in stim_map_df.iterrows():
        trial_id = str(row['trialID']).strip()
        if not trial_id or trial_id == 'nan': 
            continue

        # 寻找原始 JSON 文件
        # 优先级 1: trialID/trialID_metadata.json
        original_json = folder_path / trial_id / f"{trial_id}_metadata.json"
        
        # 优先级 2: folder_path/trialID_metadata.json
        if not original_json.exists():
            original_json = folder_path / f"{trial_id}_metadata.json"

        # 检查文件存在性
        if original_json.exists() and not original_json.name.startswith('.'):
            new_json_path = original_json.parent / f"{trial_id}{SUFFIX}"
            create_updated_json(original_json, new_json_path, row, events_df)
            json_count += 1
            print(f"  [生成] {new_json_path.name}")
        else:
            if not original_json.name.startswith('.'):
                print(f"  [警告] 未找到原始文件: {trial_id}_metadata.json")
    
    print(f"  [完成] 生成了 {json_count} 个更新的 JSON 文件")


def update_json_files():
    """主函数"""
    root = Path(ROOT_DIR)
    
    if not root.exists():
        print(f"错误：根目录不存在 {ROOT_DIR}")
        return

    # 搜索所有以 Processed_TIF_ 开头的文件夹
    tif_folders = [
        d for d in root.iterdir() 
        if d.is_dir() and d.name.startswith('Processed_TIF_') and not d.name.startswith('.')
    ]
    
    if not tif_folders:
        print(f"在 {root.absolute()} 下未找到有效的 Processed_TIF_ 文件夹")
        return

    print(f"找到 {len(tif_folders)} 个待处理文件夹。")
    print("="*60)

    for folder in tif_folders:
        process_folder(folder)

    print("\n" + "="*60)
    print("所有操作已完成。")


if __name__ == "__main__":
    update_json_files()