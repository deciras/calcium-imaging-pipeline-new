import subprocess
import os
import sys
import traceback

# --- 配置区域 ---
# 1. Linux 工作站上 Fiji 可执行文件的路径
# 通常是 /path/to/Fiji.app/ImageJ-linux64
FIJI_PATH = "/home/yifei/Fiji/fiji-linux-x64"

# 2. 刚才保存的 worker.py 的路径
WORKER_SCRIPT = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/olympus_code/20260131_code/01_fiji_totif_worker.py"

# 3. 你要处理的数据主目录
DATA_ROOT = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus"

# 4. 设置线程数 (根据工作站核心数设置)
MAX_THREADS = 30


def run_fiji_task(fiji_bin, script_path, root_dir, extension=".oir", threads=8):
    """
    通过 subprocess 调用 Fiji 的 Headless 模式运行脚本
    """
    # 确保路径是绝对路径
    abs_root = os.path.abspath(root_dir)
    abs_script = os.path.abspath(script_path)
    
    # 构造 Fiji 命令
    # --headless: 无界面模式
    # --run: 运行指定的脚本
    # 后面跟着的是传给脚本的参数
    command = [
        fiji_bin, 
        "--headless", 
        "--run", abs_script, 
        "rootDir='{}',ext='{}',threads={}".format(abs_root, extension, threads)
    ]
    
    print("\n" + "="*50)
    print("Launching Fiji Task...")
    print("Target Directory: {}".format(abs_root))
    print("="*50)
    
    # 执行命令并实时打印输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    process.wait()
    
    if process.returncode == 0:
        print("\nSUCCESS: Fiji task completed.")
    else:
        print("\nERROR: Fiji task failed with return code {}".format(process.returncode))

if __name__ == "__main__":
    # ---- 执行 ---
    if not os.path.exists(FIJI_PATH):
        print("Error: Fiji binary not found at " + FIJI_PATH)
    else:
        run_fiji_task(FIJI_PATH, WORKER_SCRIPT, DATA_ROOT, extension=".oir", threads=MAX_THREADS)