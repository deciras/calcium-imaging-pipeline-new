import os
import shutil
import csv
from tqdm import tqdm

# ================= 配置区域 =================
# 请在这里指定你的主文件夹路径（建议使用 r'' 原始字符串防止转义）
DATA_ROOT = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus'
# ===========================================


def organize_and_incremental_csv(base_path):
    """
    对单个日期文件夹（base_path）内的 .oir 文件进行归纳整理，
    并增量更新该文件夹下的 metadata.csv。

    修复说明
    --------
    原版在 __main__ 里用 os.walk 遍历所有子目录，对每个子目录都调用本函数，
    而本函数内部又有 os.walk 递归，导致同一个 trial 被重复处理。
    修复后：本函数只处理传入的单个 base_path（不再内部递归），
    __main__ 负责找到所有日期文件夹后逐个调用。

    Trial_ID 去空格一致性修复
    -------------------------
    原版 target_folder 去空格，但 all_found_trial_ids 里的 tid 保留原始空格，
    导致 CSV 增量更新时可能重复写入同一 trial。
    修复后：tid 在提取时统一去空格，保证 CSV 里的 Trial_ID 与文件夹名一致。
    """
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

    print(f"开始扫描并处理: {root_dir}")

    # ----------------------------------------------------------------
    # 1. 只扫描 base_path 这一层（不递归），整理 .oir 文件到子文件夹
    # ----------------------------------------------------------------
    dirpath = root_dir
    filenames = os.listdir(dirpath)
    oir_files = [f for f in filenames if f.endswith('.oir') and os.path.isfile(os.path.join(dirpath, f))]

    if oir_files:
        # 提取 Trial ID，统一去空格，保证与后续文件夹名一致
        current_tids = [os.path.splitext(f)[0].replace(' ', '') for f in oir_files]
        all_found_trial_ids.update(current_tids)

        # 按长度从长到短排序：避免 "file" 吞掉 "file_0001" 的文件
        # 同时对原始文件名做对应排序
        tid_file_pairs = sorted(
            zip(current_tids, oir_files),
            key=lambda x: len(x[0]),
            reverse=True
        )

        processed_files = set()

        for tid, orig_filename in tid_file_pairs:
            # 如果当前文件夹名已经是 tid，说明已归类，跳过
            if os.path.basename(dirpath) == tid:
                continue

            # 找出属于这个 tid 的所有文件（精确匹配 tid.oir 或 tid_ 开头）
            # 注意：这里用去空格后的 tid 去匹配去空格后的文件名
            target_files = [
                f for f in filenames
                if (
                    f.replace(' ', '') == f"{tid}.oir"
                    or f.replace(' ', '').startswith(f"{tid}_")
                )
                and f not in processed_files
                and os.path.isfile(os.path.join(dirpath, f))
            ]

            if target_files:
                target_folder = os.path.join(dirpath, tid)  # tid 已去空格
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)

                for f in target_files:
                    dst_name = f.replace(' ', '')
                    try:
                        shutil.move(
                            os.path.join(dirpath, f),
                            os.path.join(target_folder, dst_name)
                        )
                        processed_files.add(f)
                    except Exception as e:
                        print(f"移动文件 {f} 失败: {e}")

                print(f"已完成归纳: {tid} (位于 {os.path.relpath(dirpath, root_dir)})")

    # 同时收集已存在的子文件夹中的 trial ID（已归类过的）
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            tid = entry.name.replace(' ', '')
            # 只收录看起来像 trial 文件夹的（里面有 .oir 文件）
            has_oir = any(
                f.endswith('.oir')
                for f in os.listdir(entry.path)
                if os.path.isfile(os.path.join(entry.path, f))
            )
            if has_oir:
                all_found_trial_ids.add(tid)

    # ----------------------------------------------------------------
    # 2. 处理 CSV 增量更新
    # ----------------------------------------------------------------
    existing_ids = []
    file_exists = os.path.exists(csv_path)

    if file_exists:
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_ids = [row["Trial_ID"] for row in reader if row.get("Trial_ID")]
        except Exception as e:
            print(f"读取旧 CSV 出错: {e}")

    # 找出不在 CSV 里的新 ID（Trial_ID 已统一去空格）
    new_ids = [tid for tid in sorted(all_found_trial_ids) if tid not in existing_ids]

    # ----------------------------------------------------------------
    # 3. 写入/追加 CSV
    # ----------------------------------------------------------------
    if not new_ids and file_exists:
        print("CSV 已是最新，无需添加。")
    else:
        try:
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
                print("CSV 已初始化。")
        except Exception as e:
            print(f"写入 CSV 失败: {e}")


def find_date_folders(data_root):
    """
    找出 DATA_ROOT 下所有直接子文件夹作为"日期文件夹"。
    不递归，避免重复处理。
    """
    date_folders = []
    try:
        for entry in os.scandir(data_root):
            if entry.is_dir():
                date_folders.append(entry.path)
    except Exception as e:
        print(f"扫描 DATA_ROOT 失败: {e}")
    return sorted(date_folders)


if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"错误: DATA_ROOT 不存在 -> {DATA_ROOT}")
    else:
        date_folders = find_date_folders(DATA_ROOT)
        print(f"找到 {len(date_folders)} 个日期文件夹")

        for date_folder in tqdm(date_folders, desc="Organizing date folders"):
            organize_and_incremental_csv(date_folder)
