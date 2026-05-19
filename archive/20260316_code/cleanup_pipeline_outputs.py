#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理 pipeline 后续步骤生成的文件
只保留基础数据结构
"""

import os
from pathlib import Path
import shutil


ROOT_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"


KEEP_SUFFIX = [
    "_brightness_trace.csv",
    "_corrected_movie.tif",
    "_metadata.json",
    "_stim_events.csv",
    "_stim_map.csv",
    "_stim_trace.png"
]


KEEP_DIRS = []


def should_keep(file_name):

    for suf in KEEP_SUFFIX:
        if file_name.endswith(suf):
            return True

    return False


def clean_trial_dir(trial_path):

    print(f"\nCleaning: {trial_path}")

    for item in trial_path.iterdir():

        if item.is_file():

            if not should_keep(item.name):
                print("remove file:", item)
                item.unlink()

        elif item.is_dir():

            if item.name not in KEEP_DIRS:
                print("remove dir :", item)
                shutil.rmtree(item)


def main():

    root = Path(ROOT_DIR)

    for path in root.rglob("*"):

        if path.is_dir():

            files = list(path.glob("*_corrected_movie.tif"))

            if len(files) > 0:
                clean_trial_dir(path)


if __name__ == "__main__":
    main()