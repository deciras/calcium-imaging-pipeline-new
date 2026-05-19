import os
import shutil
import csv
from tqdm import tqdm

# ================= 配置区域 =================
# 请在这里指定你的主文件夹路径（建议使用 r'' 原始字符串防止转义）
DATA_ROOT = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus'
# ===========================================

def organize_and_incremental_csv(base_path):
    root_dir = os.path.abspath(base_path)
    csv_path = os.path.join(root_dir, "metadata.csv")
    
    headers = [
        "Trial_ID", "Time_Per_Frame", "Num_Flashes", 
        "Stim_Start_s", "Stim_Duration_s", "Stim_Interval_s", "Polarization_Angles"
    ]

    if not os.path.exists(root_dir):
        print(f"错误: 路径不存在 -> {root_dir}")
        return

    all_found_trial_ids = set()

    print(f"开始递归扫描并处理: {root_dir}")

    # 1. 递归遍历所有文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 找出当前目录下所有的 .oir 文件，确定该层级下有哪些 Trial_ID
        oir_files = [f for f in filenames if f.endswith('.oir')]
        if not oir_files:
            continue
            
        # 提取当前目录下所有的 Trial ID
        current_tids = [os.path.splitext(f)[0] for f in oir_files]
        all_found_trial_ids.update(current_tids)
        
        # --- 核心改进：按长度从长到短排序 ---
        # 这样 file_0001 会比 file 先被处理，避免 file 吞掉 file_0001 的文件
        current_tids.sort(key=len, reverse=True)
        
        # 记录当前目录中哪些文件已经被“领走”了，防止一个文件被移动两次
        processed_files = set()
        
        for tid in current_tids:
            # 检查当前文件夹名是否已经是 tid (说明已经归类过了)
            if os.path.basename(dirpath) == tid:
                continue
                
            # 确定要移动的文件：
            # 1. 精确匹配 tid.oir
            # 2. 匹配以 tid_ 开头的文件
            target_files = [
                f for f in filenames 
                if (f == f"{tid}.oir" or f.startswith(f"{tid}_")) 
                and f not in processed_files
                and os.path.isfile(os.path.join(dirpath, f))
            ]
            
            if target_files:
                target_folder = os.path.join(dirpath, tid)
                target_folder.replace(' ', '')  # 去掉空格，防止后续处理出问题
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                for f in target_files:
                    try:
                        shutil.move(os.path.join(dirpath, f), os.path.join(target_folder, f.replace(' ','')))
                        processed_files.add(f)
                    except Exception as e:
                        print(f"移动文件 {f} 失败: {e}")
                
                print(f"已完成归纳: {tid} (位于 {os.path.relpath(dirpath, root_dir)})")

    # 2. 处理 CSV 增量更新
    existing_ids = []
    file_exists = os.path.exists(csv_path)

    if file_exists:
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_ids = [row["Trial_ID"] for row in reader if row.get("Trial_ID")]
        except Exception as e:
            print(f"读取旧 CSV 出错: {e}")

    # 找出不在 CSV 里的新 ID
    new_ids = [tid for tid in sorted(list(all_found_trial_ids)) if tid not in existing_ids]

    # 3. 写入/追加 CSV
    if not new_ids and file_exists:
        print("CSV 已是最新，无需添加。")
    else:
        try:
            # 'a' 模式追加，'w' 模式新建
            mode = 'a' if file_exists else 'w'
            with open(csv_path, mode=mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                for tid in new_ids:
                    writer.writerow({"Trial_ID": tid})
            
            if new_ids:
                print(f"CSV 更新成功: 新增了 {len(new_ids)} 条记录。")
            else:
                print(f"CSV 已初始化。")
        except Exception as e:
            print(f"写入 CSV 失败: {e}")

if __name__ == "__main__":
    for dirpath, dirnames, filenames in os.walk(DATA_ROOT):
        for dirname in tqdm(dirnames, desc=f"Organizing {dirpath}"):
            base_path = os.path.join(dirpath, dirname)
            organize_and_incremental_csv(base_path)