import subprocess
import os
import sys
import time
from pathlib import Path

CODE_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/code'

def run_script(script_name):
    """运行单个脚本并捕获其输出"""
    print(f"\n{'='*60}")
    print(f"正在启动: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 使用当前 Python 解释器运行脚本
        # bufsize=1 表示行缓冲，这样可以实时看到子脚本的 print 输出
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        
        # 等待脚本完成
        process.wait()
        
        if process.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\n[成功] {script_name} 运行完成 (耗时: {elapsed:.2f}s)")
            return True
        else:
            print(f"\n[失败] {script_name} 退出代码为 {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[异常] 运行 {script_name} 时出错: {e}")
        return False

def main():
    # 定义脚本执行顺序
    
    os.chdir(CODE_DIR)
    print(f'当前工作路径为: {os.getcwd()}')
    
    pipeline = [
        # "00_oir_file_manager.py",
        # "01_fiji_totif_ini.py",
        # "02_motion_correct_suite2p.py",
        #"04_update_json.py",
        #"05_delta_F_F0.py",
        #"06_neuro_analysis.py",
        "07_5_retina_leiden_clustering_analysis.py",
        # "08_map_significant_ROI_video.py",
        "09_generate_annotated_video.py"
    ]
    
    total_start_time = time.time()
    print("开始执行钙成像分析完整流水线...")

    for script in pipeline:
        # 检查文件是否存在
        if not Path(script).exists():
            print(f"错误：找不到脚本文件 {script}，请确保它在当前目录下。")
            sys.exit(1)
        
        # 运行脚本
        success = run_script(script)
        
        if not success:
            print(f"\n停止流水线：{script} 执行失败。请检查该脚本的错误信息。")
            sys.exit(1)

    total_elapsed = time.time() - total_start_time
    print(f"\n{'#'*60}")
    print(f"所有任务已成功完成！")
    print(f"总耗时: {total_elapsed/60:.2f} 分钟")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()