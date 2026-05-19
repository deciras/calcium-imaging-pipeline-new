import subprocess
import os
import sys

def run_fiji_task(fiji_bin, script_path, root_dir, extension=".oir", threads=8, 
                   do_proj=True, do_layers=True, do_stack=True):
    """
    通过 subprocess 调用 Fiji
    """
    abs_root = os.path.abspath(root_dir)
    abs_script = os.path.abspath(script_path)
    
    # 将 Python 的 True/False 转换为 Fiji 参数识别的字符串
    proj_str = "true" if do_proj else "false"
    layers_str = "true" if do_layers else "false"
    stack_str = "true" if do_stack else "false"
    
    # 构造 Fiji 命令参数字符串
    arg_string = "rootDir='{}',ext='{}',threads={},doZProj={},doZLayers={},doStack={}".format(
        abs_root, extension, threads, proj_str, layers_str, stack_str
    )
    
    command = [
        fiji_bin, 
        "-Xmx64G",      # [新增] 分配 64GB 内存给 Fiji，根据你的服务器内存调整 (例如 -Xmx128G)
        "--headless", 
        "--run", abs_script, 
        arg_string
    ]
    
    print("\n" + "="*50)
    print("Launching Fiji Task...")
    print("Tasks: Projection={}, Layers={}, Stack={}".format(do_proj, do_layers, do_stack))
    print("="*50)
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    process.wait()
    
    if process.returncode == 0:
        print("\nSUCCESS: Fiji task completed.")
    else:
        print("\nERROR: Fiji task failed.")

if __name__ == "__main__":
    # --- 配置区域 ---
    FIJI_PATH = "/home/yifei/Fiji/fiji-linux-x64"
    WORKER_SCRIPT = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/20251231_code/1_fiji_totif.py"
    DATA_ROOT = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2025_olympus"
    
    MAX_THREADS = 16
    
    # --- 任务开关 ---
    SAVE_PROJECTION = True   # 是否生成平均投影
    SAVE_LAYERS     = False   # 是否生成分层TIF
    SAVE_STACK      = False   # 是否生成完整Stack (带Z轴)

    # ---- 执行 ---
    if not os.path.exists(FIJI_PATH):
        print("Error: Fiji binary not found at " + FIJI_PATH)
    else:
        run_fiji_task(
            FIJI_PATH, 
            WORKER_SCRIPT, 
            DATA_ROOT, 
            extension=".oir", 
            threads=MAX_THREADS,
            do_proj=SAVE_PROJECTION,
            do_layers=SAVE_LAYERS,
            do_stack=SAVE_STACK
        )