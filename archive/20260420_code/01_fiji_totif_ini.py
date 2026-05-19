import subprocess
import os
import sys

# --- 配置区域 ---
# 1. Linux 工作站上 Fiji 可执行文件的路径
FIJI_PATH = "/home/yifei/Fiji/fiji-linux-x64"

# 2. Fiji worker 脚本路径
WORKER_SCRIPT = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260318_code/01_fiji_totif_worker.py"

# 3. 数据主目录（建议选择包含多个日期文件夹的总根目录）
DATA_ROOT = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus"

# 4. 设置线程数（当前 worker 内部未实际并行，仅保留参数位）
MAX_THREADS = 30

# 5. 运行模式
# "skip"      已完成文件跳过
# "overwrite" 删除旧输出并重跑
MODE = "skip"


def validate_paths(fiji_bin, script_path, root_dir):
    errors = []

    if not os.path.exists(fiji_bin):
        errors.append("Fiji binary not found: {}".format(fiji_bin))
    elif not os.path.isfile(fiji_bin):
        errors.append("Fiji path is not a file: {}".format(fiji_bin))

    if not os.path.exists(script_path):
        errors.append("Worker script not found: {}".format(script_path))
    elif not os.path.isfile(script_path):
        errors.append("Worker script path is not a file: {}".format(script_path))

    if not os.path.exists(root_dir):
        errors.append("Data root not found: {}".format(root_dir))
    elif not os.path.isdir(root_dir):
        errors.append("Data root is not a directory: {}".format(root_dir))

    return errors


def run_fiji_task(fiji_bin, script_path, root_dir, extension=".oir", threads=8, mode="skip"):
    """
    通过 subprocess 调用 Fiji 的 Headless 模式运行脚本
    """
    abs_root = os.path.abspath(root_dir)
    abs_script = os.path.abspath(script_path)

    command = [
        fiji_bin,
        "--headless",
        "--run", abs_script,
        "rootDir='{}',ext='{}',threads={},mode='{}'".format(
            abs_root, extension, threads, mode
        )
    ]

    print("\n" + "=" * 60)
    print("Launching Fiji Task...")
    print("Target Directory : {}".format(abs_root))
    print("Worker Script    : {}".format(abs_script))
    print("File Extension   : {}".format(extension))
    print("Mode             : {}".format(mode))
    print("=" * 60)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()

    if process.returncode == 0:
        print("\nSUCCESS: Fiji task completed.")
        return 0
    else:
        print("\nERROR: Fiji task failed with return code {}".format(process.returncode))
        return process.returncode


if __name__ == "__main__":
    path_errors = validate_paths(FIJI_PATH, WORKER_SCRIPT, DATA_ROOT)

    if path_errors:
        print("\nCONFIGURATION ERROR(S):")
        for err in path_errors:
            print(" - " + err)
        sys.exit(1)

    exit_code = run_fiji_task(
        FIJI_PATH,
        WORKER_SCRIPT,
        DATA_ROOT,
        extension=".oir",
        threads=MAX_THREADS,
        mode=MODE
    )
    sys.exit(exit_code)