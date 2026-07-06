#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理 pipeline 后续步骤生成的文件。

默认只预览，不执行删除；需要显式传 `--execute`。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


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


def clean_trial_dir(trial_path: Path, *, execute: bool) -> tuple[int, int]:
    removed_files = 0
    removed_dirs = 0

    print(f"\nCleaning trial: {trial_path}")

    for item in trial_path.iterdir():

        if item.is_file():

            if not should_keep(item.name):
                print(("remove file:" if execute else "would remove file:"), item.name)
                if execute:
                    item.unlink()
                removed_files += 1

        elif item.is_dir():

            if item.name not in KEEP_DIRS:
                print(("remove dir :" if execute else "would remove dir :"), item.name)
                if execute:
                    shutil.rmtree(item)
                removed_dirs += 1

    return removed_files, removed_dirs


def find_trial_dirs(root: Path) -> list[Path]:

    """
    通过 *_corrected_movie.tif 自动识别 trial 目录
    """
    trial_dirs = set()

    for movie in root.rglob("*_corrected_movie.tif"):
        trial_dirs.add(movie.parent)

    return sorted(trial_dirs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="清理指定 DATA_ROOT 下由 pipeline 后续步骤生成的 trial 内文件。"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="要清理的根目录，例如 DATA_ROOT/03_motion_correct 或某个包含 corrected movie 的输出根目录。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真的执行删除。默认只预览。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root_dir.expanduser().resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 2
    if not root.is_dir():
        print(f"Root is not a directory: {root}")
        return 2

    trial_dirs = find_trial_dirs(root)

    print(f"\nFound {len(trial_dirs)} trial directories\n")
    if not args.execute:
        print("Preview mode only. Re-run with --execute to delete files.\n")

    total_files = 0
    total_dirs = 0
    for trial in trial_dirs:
        removed_files, removed_dirs = clean_trial_dir(trial, execute=args.execute)
        total_files += removed_files
        total_dirs += removed_dirs

    print(f"\nSummary: {'removed' if args.execute else 'would remove'} {total_files} files and {total_dirs} directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
