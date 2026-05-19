#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

正式版 calcium imaging / benchmark pipeline 入口。

功能
----
1. 串联正式主线脚本。
2. 每一步可以指定不同 conda 环境运行。
3. 每一步失败后默认停止。
4. 支持相对路径 / 绝对路径。
5. 实时输出子脚本日志。
"""

import sys
import time
import shutil
import traceback
import subprocess
from pathlib import Path


# ============================================================
# 配置区域
# ============================================================

CODE_DIR = Path(
    "/mnt/50d357b2-473b-4533-8c74-1db99704876a/"
    "yifeiding/calcium_imaging/olympus_code/20260513_code"
)

# env:
#   - None / "" / "current" 表示使用当前 Python 解释器
#   - 其他字符串表示 conda 环境名，例如 "caiman", "suite2p_test"
PIPELINE_STEPS = [
    {
        "script": "00_oir_file_manager.py",
        "env": "current",
    },
    {
        "script": "01_fiji_totif_ini.py",
        "env": "fiji_env",
    },
    {
        "script": "02_generate_stim_map.py",
        "env": "current",
    },
    {
        "script": "03_motion_correct_func_caiman.py",
        "env": "caiman",
    },
    {
        "script": "04_spatial_highpass.py",
        "env": "caiman",
    },
    {
        "script": "05b_suite2p_roi_detection_schema_aligned_connected.py",
        "env": "suite2p_test",
    },

]

STOP_ON_FAILURE = True

# conda 命令路径。一般写 "conda" 即可。
# 如果 conda run 找不到，可以改成：
# "/home/yifei/anaconda3/bin/conda"
CONDA_BIN = "conda"


# ============================================================
# 工具函数
# ============================================================

def banner(text: str, char: str = "=") -> None:
    line = char * 72
    print(f"\n{line}\n{text}\n{line}", flush=True)


def normalize_env_name(env_name: str | None) -> str | None:
    """
    将 env 配置统一为:
      None = 当前解释器
      str  = conda 环境名
    """
    if env_name is None:
        return None

    env_name = str(env_name).strip()

    if env_name == "":
        return None

    if env_name.lower() in {"current", "same", "self"}:
        return None

    return env_name


def build_command(script_path: Path, env_name: str | None) -> list[str]:
    """
    根据 env_name 构建运行命令。
    """
    env_name = normalize_env_name(env_name)

    if env_name is None:
        return [
            sys.executable,
            str(script_path),
        ]

    return [
        CONDA_BIN,
        "run",
        "-n",
        env_name,
        "--no-capture-output",
        "python",
        str(script_path),
    ]


def check_conda_available() -> None:
    """
    如果 pipeline 中用了 conda 环境，先检查 conda 命令是否存在。
    """
    need_conda = any(
        normalize_env_name(step.get("env")) is not None
        for step in PIPELINE_STEPS
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
    检查 CODE_DIR 和每个脚本是否存在。
    返回标准化后的 pipeline step 列表。
    """
    resolved_steps = []

    code_dir = code_dir.resolve()

    if not code_dir.exists():
        raise FileNotFoundError(f"CODE_DIR 不存在: {code_dir}")

    if not code_dir.is_dir():
        raise NotADirectoryError(f"CODE_DIR 不是文件夹: {code_dir}")

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise TypeError(f"第 {i} 个 PIPELINE_STEPS 项不是 dict: {step}")

        if "script" not in step:
            raise KeyError(f"第 {i} 个 PIPELINE_STEPS 项缺少 script 字段")

        script_raw = step["script"]
        env_name = normalize_env_name(step.get("env"))

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
                "script": script_path,
                "env": env_name,
            }
        )

    return resolved_steps


def run_script(script_path: Path, env_name: str | None = None) -> bool:
    """
    运行单个脚本。
    """
    env_name = normalize_env_name(env_name)

    banner(f"正在启动: {script_path.name}")

    cmd = build_command(script_path, env_name)
    start_time = time.time()

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
                f"\n[成功] {script_path.name} 运行完成 "
                f"(耗时: {elapsed:.2f}s)",
                flush=True,
            )
            return True

        print(
            f"\n[失败] {script_path.name} 退出代码为 {return_code} "
            f"(耗时: {elapsed:.2f}s)",
            flush=True,
        )
        return False

    except KeyboardInterrupt:
        print(f"\n[中断] 用户手动中断: {script_path.name}", flush=True)
        raise

    except Exception as e:
        print(f"\n[异常] 运行 {script_path.name} 时出错: {e}", flush=True)
        traceback.print_exc()
        return False


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    banner("正式版 calcium imaging pipeline")

    print(f"CODE_DIR: {CODE_DIR.resolve()}", flush=True)
    print(f"当前启动 Python: {sys.executable}", flush=True)

    print("\n执行顺序:", flush=True)
    for i, step in enumerate(PIPELINE_STEPS, start=1):
        env_name = normalize_env_name(step.get("env"))
        env_label = env_name if env_name else "current"
        print(f"  {i}. [{env_label}] {step['script']}", flush=True)

    try:
        check_conda_available()
        pipeline_steps = validate_pipeline(CODE_DIR, PIPELINE_STEPS)

    except Exception as e:
        print(f"\n[配置错误] {e}", flush=True)
        sys.exit(1)

    total_start_time = time.time()
    success_count = 0

    for step in pipeline_steps:
        script_path = step["script"]
        env_name = step["env"]

        ok = run_script(script_path, env_name=env_name)

        if ok:
            success_count += 1
            continue

        if STOP_ON_FAILURE:
            print(f"\n停止流水线：{script_path.name} 执行失败。", flush=True)
            sys.exit(1)

    total_elapsed = time.time() - total_start_time

    banner("所有任务已完成", "#")
    print(f"成功步骤数: {success_count}/{len(pipeline_steps)}", flush=True)
    print(f"总耗时: {total_elapsed / 60:.2f} 分钟", flush=True)


if __name__ == "__main__":
    main()