#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理 pipeline 后续步骤生成的文件
只保留 03 的基础数据结构
"""

import shutil
from pathlib import Path


ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"


# 只保留这些文件
KEEP_SUFFIX = [
    "_corrected_movie.tif",
    "_metadata.json",
    "_stim_events.csv",
    "_stim_map.csv",
    "_stim_trace.png",
]


# 强制删除这些文件
REMOVE_PATTERNS = [
    "_brightness_corrected_movie.tif",
    "_brightness_adjusted_movie.tif",
    "_brightness_brightness_corrected_movie.tif",
    "_brightness_global_brightness_trace.csv",
    "_brightness_global_brightness_trace.png",
]


# 删除所有子目录
KEEP_DIRS = []


def should_keep(file_name):

    # 强制删除
    for pat in REMOVE_PATTERNS:
        if pat in file_name:
            return False

    # 保留正常文件
    for suf in KEEP_SUFFIX:
        if file_name.endswith(suf):
            return True

    return False


def clean_trial_dir(trial_path):

    print(f"\nCleaning trial: {trial_path}")

    for item in trial_path.iterdir():

        if item.is_file():

            if not should_keep(item.name):
                print("remove file:", item.name)
                item.unlink()

        elif item.is_dir():

            if item.name not in KEEP_DIRS:
                print("remove dir :", item.name)
                shutil.rmtree(item)


def find_trial_dirs(root):

    """
    通过 *_corrected_movie.tif 自动识别 trial 目录
    """
    trial_dirs = set()

    for movie in root.rglob("*_corrected_movie.tif"):
        trial_dirs.add(movie.parent)

    return sorted(trial_dirs)


def main():

    root = Path(ROOT_DIR)

    trial_dirs = find_trial_dirs(root)

    print(f"\nFound {len(trial_dirs)} trial directories\n")

    for trial in trial_dirs:
        clean_trial_dir(trial)


if __name__ == "__main__":
    main()