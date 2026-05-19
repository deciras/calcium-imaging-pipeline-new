#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

正式版 benchmark pipeline 入口。

当前主线：
    04_spatial_highpass.py
    05a_pixelwise_patch_detection_schema_aligned.py
    05b_suite2p_roi_detection_schema_aligned.py
    05d_compare_methods_enhanced.py

说明
----
1. 本脚本只负责串联“正式主线”脚本，不包含实验性分支
   （如 03_5 / 03_7 / 03_8 等）。
2. 默认按顺序依次执行，每一步失败即停止。
3. 使用当前 Python 解释器运行子脚本，方便直接在对应 conda 环境下启动。

建议
----
- 跑之前先确认当前环境正确：
    - 04b 推荐在 suite2p_test / suite2p 0.14.4 环境下运行
- 如果只想跑其中一部分，可在 PIPELINE_STEPS 里临时注释掉对应脚本
"""

import sys
import time
import traceback
import subprocess
from pathlib import Path

# -------------------------------------------------------------------
# 配置区域
# -------------------------------------------------------------------
CODE_DIR = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260505_code"

PIPELINE_STEPS = [
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260505_code/00_oir_file_manager.py",    
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260505_code/01_fiji_totif_ini.py",
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260505_code/02_generate_stim_map.py",
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260505_code/03_motion_correct_func_caiman.py",
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260505_code/04_spatial_highpass.py",
]

STOP_ON_FAILURE = True


def banner(text: str, char: str = "="):
    line = char * 72
    print(f"\n{line}\n{text}\n{line}")


def run_script(script_path: Path) -> bool:
    banner(f"正在启动: {script_path.name}")

    start_time = time.time()

    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            cwd=str(script_path.parent),
        )
        process.wait()

        elapsed = time.time() - start_time

        if process.returncode == 0:
            print(f"\n[成功] {script_path.name} 运行完成 (耗时: {elapsed:.2f}s)")
            return True

        print(f"\n[失败] {script_path.name} 退出代码为 {process.returncode}")
        return False

    except Exception as e:
        print(f"\n[异常] 运行 {script_path.name} 时出错: {e}")
        traceback.print_exc()
        return False


def validate_pipeline(code_dir: Path, steps: list[str]) -> list[Path]:
    resolved = []

    if not code_dir.exists():
        raise FileNotFoundError(f"CODE_DIR 不存在: {code_dir}")

    for step in steps:
        path = code_dir / step
        if not path.exists():
            raise FileNotFoundError(f"找不到脚本文件: {path}")
        resolved.append(path)

    return resolved


def main():
    code_dir = Path(CODE_DIR).resolve()

    banner("正式版 benchmark pipeline")
    print(f"当前工作路径: {code_dir}")
    print("执行顺序:")
    for i, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"  {i}. {step}")

    try:
        pipeline_paths = validate_pipeline(code_dir, PIPELINE_STEPS)
    except Exception as e:
        print(f"\n[配置错误] {e}")
        sys.exit(1)

    total_start_time = time.time()
    success_count = 0

    for script_path in pipeline_paths:
        ok = run_script(script_path)

        if ok:
            success_count += 1
            continue

        if STOP_ON_FAILURE:
            print(f"\n停止流水线：{script_path.name} 执行失败。")
            sys.exit(1)

    total_elapsed = time.time() - total_start_time

    banner("所有任务已完成", "#")
    print(f"成功步骤数: {success_count}/{len(pipeline_paths)}")
    print(f"总耗时: {total_elapsed / 60:.2f} 分钟")


if __name__ == "__main__":
    main()
