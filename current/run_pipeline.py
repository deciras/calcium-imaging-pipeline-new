#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

Suite2p-only calcium imaging pipeline entry point.

Current main line
-----------------
00_oir_file_manager.py
01_fiji_totif_ini.py
02_generate_stim_map.py
03_motion_correct_func_caiman.py
04_spatial_highpass.py
05_suite2p_roi_detection_schema_aligned_connected.py

Purpose
-------
1. Run only the stable preprocessing + suite2p ROI detection path.
2. Do not run benchmark branches such as pixelwise detection / method comparison / overlay videos.
3. Let each sub-script handle its own skip / overwrite logic.
4. Allow each step to run in its required conda environment.
5. Stream child-process logs in real time.

Notes
-----
- This file only controls which scripts are launched and in which environment.
- Real skip behavior must still be fixed inside 02 / 03 / 04 / 05b.
- After the front half is stable, later steps can be appended:
    06_extract_dff_from_suite2p.py
    07_neuro_analysis.py
    08_make_suite2p_response_overlay_video.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

CODE_DIR = Path(
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/"
    "yifeiding/calcium_imaging/olympus_code/20260513_code"
)

# env:
#   - None / "" / "current" means use the current Python interpreter.
#   - Other strings are conda environment names, e.g. "caiman", "suite2p_test".
PIPELINE_STEPS = [
    {
        "name": "00 organize OIR files",
        "script": "00_oir_file_manager.py",
        "env": "current",
        "enabled": True,
    },
    {
        "name": "01 Fiji OIR to TIF",
        "script": "01_fiji_totif_ini.py",
        "env": "fiji_env",
        "enabled": True,
    },
    {
        "name": "02 generate stimulus map",
        "script": "02_generate_stim_map.py",
        "env": "current",
        "enabled": True,
    },
    {
        "name": "03 CaImAn motion correction",
        "script": "03_motion_correct_func_caiman.py",
        "env": "caiman",
        "enabled": True,
    },
    {
        "name": "04 spatial high-pass",
        "script": "04_spatial_highpass.py",
        "env": "caiman",
        "enabled": True,
    },
    {
        "name": "05 suite2p ROI detection",
        "script": "05_suite2p_roi_detection_schema_aligned_connected.py",
        "env": "suite2p_test",
        "enabled": True,
    },
]

STOP_ON_FAILURE = True

# Usually "conda" is enough. If `conda run` cannot be found, set an absolute path, e.g.:
# CONDA_BIN = "/home/yifei/anaconda3/bin/conda"
CONDA_BIN = "conda"


# ============================================================
# Utility functions
# ============================================================

def banner(text: str, char: str = "=") -> None:
    line = char * 72
    print(f"\n{line}\n{text}\n{line}", flush=True)


def normalize_env_name(env_name: str | None) -> str | None:
    """
    Normalize env config.

    Returns
    -------
    None
        Use current Python interpreter.
    str
        Conda environment name.
    """
    if env_name is None:
        return None

    env_name = str(env_name).strip()
    if env_name == "":
        return None

    if env_name.lower() in {"current", "same", "self"}:
        return None

    return env_name


def enabled_steps(steps: list[dict]) -> list[dict]:
    """
    Return only enabled pipeline steps.
    Missing `enabled` is treated as True for backward compatibility.
    """
    return [step for step in steps if bool(step.get("enabled", True))]


def build_command(script_path: Path, env_name: str | None) -> list[str]:
    """
    Build the command used to run one script.
    """
    env_name = normalize_env_name(env_name)

    if env_name is None:
        return [sys.executable, str(script_path)]

    return [
        CONDA_BIN,
        "run",
        "-n",
        env_name,
        "--no-capture-output",
        "python",
        str(script_path),
    ]


def check_conda_available(steps: list[dict]) -> None:
    """
    Check whether conda is available if any enabled step needs a conda env.
    """
    need_conda = any(
        normalize_env_name(step.get("env")) is not None
        for step in enabled_steps(steps)
    )

    if not need_conda:
        return

    conda_path = shutil.which(CONDA_BIN)
    if conda_path is None:
        raise FileNotFoundError(
            f"找不到 conda 命令: {CONDA_BIN}\n"
            "解决办法：\n"
            "1. 确认当前 shell 能运行 conda；或\n"
            "2. 把 CONDA_BIN 改成 conda 的绝对路径，例如 "
            "/home/yifei/anaconda3/bin/conda"
        )

    print(f"[检查] conda 命令: {conda_path}", flush=True)


def validate_pipeline(code_dir: Path, steps: list[dict]) -> list[dict]:
    """
    Validate CODE_DIR and all enabled step scripts.

    Returns
    -------
    list[dict]
        Normalized step configs with absolute script paths.
    """
    resolved_steps = []
    code_dir = code_dir.resolve()

    if not code_dir.exists():
        raise FileNotFoundError(f"CODE_DIR 不存在: {code_dir}")

    if not code_dir.is_dir():
        raise NotADirectoryError(f"CODE_DIR 不是文件夹: {code_dir}")

    for i, step in enumerate(enabled_steps(steps), start=1):
        if not isinstance(step, dict):
            raise TypeError(f"第 {i} 个 PIPELINE_STEPS 项不是 dict: {step}")

        if "script" not in step:
            raise KeyError(f"第 {i} 个 PIPELINE_STEPS 项缺少 script 字段")

        script_raw = step["script"]
        env_name = normalize_env_name(step.get("env"))
        step_name = str(step.get("name", script_raw))

        script_path = Path(script_raw)
        if not script_path.is_absolute():
            script_path = code_dir / script_path

        script_path = script_path.resolve()

        if not script_path.exists():
            raise FileNotFoundError(f"找不到脚本文件: {script_path}")

        if not script_path.is_file():
            raise FileNotFoundError(f"脚本路径不是文件: {script_path}")

        resolved_steps.append(
            {
                "name": step_name,
                "script": script_path,
                "env": env_name,
            }
        )

    if len(resolved_steps) == 0:
        raise ValueError("没有启用任何 pipeline step。请检查 PIPELINE_STEPS 里的 enabled 配置。")

    return resolved_steps


def run_script(step: dict) -> bool:
    """
    Run one pipeline step.
    """
    script_path = Path(step["script"])
    env_name = normalize_env_name(step.get("env"))
    step_name = str(step.get("name", script_path.name))

    banner(f"正在启动: {step_name}")

    cmd = build_command(script_path, env_name)
    start_time = time.time()

    print(f"[步骤] {step_name}", flush=True)
    print(f"[脚本] {script_path}", flush=True)
    print(f"[工作目录] {script_path.parent}", flush=True)
    print(f"[运行环境] {env_name if env_name else 'current'}", flush=True)
    print(f"[命令] {' '.join(cmd)}", flush=True)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            cwd=str(script_path.parent),
        )

        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code == 0:
            print(
                f"\n[成功] {step_name} 运行完成 "
                f"(脚本: {script_path.name}, 耗时: {elapsed:.2f}s)",
                flush=True,
            )
            return True

        print(
            f"\n[失败] {step_name} 退出代码为 {return_code} "
            f"(脚本: {script_path.name}, 耗时: {elapsed:.2f}s)",
            flush=True,
        )
        return False

    except KeyboardInterrupt:
        print(f"\n[中断] 用户手动中断: {step_name}", flush=True)
        raise

    except Exception as e:
        print(f"\n[异常] 运行 {step_name} 时出错: {e}", flush=True)
        traceback.print_exc()
        return False


# ============================================================
# Main
# ============================================================

def main() -> None:
    banner("Suite2p-only calcium imaging pipeline")

    print(f"CODE_DIR: {CODE_DIR.resolve()}", flush=True)
    print(f"当前启动 Python: {sys.executable}", flush=True)

    active_steps = enabled_steps(PIPELINE_STEPS)

    print("\n执行顺序:", flush=True)
    for i, step in enumerate(active_steps, start=1):
        env_name = normalize_env_name(step.get("env"))
        env_label = env_name if env_name else "current"
        step_name = str(step.get("name", step.get("script", "unnamed")))
        print(f"  {i}. [{env_label}] {step_name} -> {step['script']}", flush=True)

    try:
        check_conda_available(PIPELINE_STEPS)
        pipeline_steps = validate_pipeline(CODE_DIR, PIPELINE_STEPS)

    except Exception as e:
        print(f"\n[配置错误] {e}", flush=True)
        sys.exit(1)

    total_start_time = time.time()
    success_count = 0

    for step in pipeline_steps:
        ok = run_script(step)

        if ok:
            success_count += 1
            continue

        if STOP_ON_FAILURE:
            print(f"\n停止流水线：{step['name']} 执行失败。", flush=True)
            sys.exit(1)

    total_elapsed = time.time() - total_start_time

    banner("所有启用任务已完成", "#")
    print(f"成功步骤数: {success_count}/{len(pipeline_steps)}", flush=True)
    print(f"总耗时: {total_elapsed / 60:.2f} 分钟", flush=True)


if __name__ == "__main__":
    main()