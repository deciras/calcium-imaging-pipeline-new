import os
import json
import shutil
import re
import time
import traceback
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import tifffile as tf
import suite2p
from suite2p.run_s2p import run_s2p

# ============================================================
# 配置区域
# ============================================================
# 注意：这里的 DATA_PATH 应指向第 3 步生成的 Results 文件夹
DATA_PATH = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"
# 是否删除 suite2p 运行的中间二进制文件
DELETE_INTERNAL = False
# 开启 ROI 检测
ROI_DETECT = True
# 进程与线程配置
N_WORKERS = 2
S2P_NTHREADS = 8
BATCH_SIZE = 200

def set_low_level_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def get_params_from_json(json_path):
    n_planes, fs, n_channels = 1, 1.0, 1
    if not os.path.exists(json_path):
        return n_planes, fs, n_channels
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        fs = float(data.get("temporal_calibration", {}).get("fps", 1.0))
    except Exception:
        pass
    return n_planes, fs, n_channels

# ============================================================
# 子进程任务：仅处理 ROI Detection
# ============================================================
def _process_roi_detection(
    corrected_tif_path: str,
    root_dir: str,
    ops_template: dict,
):
    set_low_level_thread_env()

    # 获取路径信息
    rel_dir = os.path.relpath(os.path.dirname(corrected_tif_path), root_dir)
    tif_name = os.path.basename(corrected_tif_path)
    # 假设 ID 是从文件名中提取，例如 XXX_corrected_movie.tif
    exp_id = tif_name.replace("_corrected_movie.tif", "")

    # 日志设置
    log_dir = os.path.join(root_dir, "_logs_roi")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"roi_{os.getpid()}_{exp_id}.log")

    def log(msg: str):
        with open(log_path, "a") as f:
            f.write(f"{time.ctime()}: {msg}\n")

    try:
        log(f"=== ROI DETECTION START ===")
        log(f"Input TIF: {corrected_tif_path}")

        # 查找同目录下的 metadata 供 FS 参考
        parent_folder = os.path.dirname(os.path.dirname(corrected_tif_path)) # 往上跳一级找 JSON
        json_files = [f for f in os.listdir(parent_folder) if f.endswith("_metadata.json")]
        
        n_planes, fs, n_channels = 1, 1.0, 1
        if json_files:
            json_path = os.path.join(parent_folder, json_files[0])
            n_planes, fs, n_channels = get_params_from_json(json_path)
            log(f"Found metadata: FS={fs}")

        # 配置 ops
        ops = suite2p.default_ops()
        ops.update(ops_template)
        ops.update({"fs": fs})

        # 设置保存路径（就在当前校正文件的目录下新建 suite2p 文件夹）
        save_path = os.path.join(os.path.dirname(corrected_tif_path), "roi_results")
        
        db = {
            "data_path": [os.path.dirname(corrected_tif_path)],
            "tiff_list": [tif_name],
            "save_path0": save_path,
            "nchannels": n_channels,
            "nplanes": n_planes,
        }

        log("[RUN] run_s2p (ROI detection only) starting...")
        run_s2p(ops=ops, db=db)
        log("[RUN] ROI detection finished.")

        return ("ok", exp_id, "ROI detection complete")

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log(traceback.format_exc())
        return ("error", exp_id, str(e))

# ============================================================
# 主函数
# ============================================================
def run_roi_detection_only(root_dir, n_workers, s2p_nthreads, batch_size):
    set_low_level_thread_env()

    # 配置 ROI 检测参数
    ops_template = {
        "do_registration": 0,      # 关键：跳过运动校正
        "roidetect": True,         # 关键：开启 ROI 检测
        "reg_tif": False,          # 不需要再生成校正后的 TIF
        "delete_bin": 0,           # 保留二进制文件以便 GUI 查看
        "nthreads": int(s2p_nthreads),
        "batch_size": int(batch_size),
        "save_mat": True,          # 保存为 .mat 方便其他软件读取
        "combined": False,
    }

    # 扫描所有校正后的文件
    all_corrected_tifs = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("_corrected_movie.tif"):
                all_corrected_tifs.append(os.path.join(r, f))
    
    all_corrected_tifs.sort()

    if not all_corrected_tifs:
        print(f"未在 {root_dir} 下找到 *_corrected_movie.tif")
        return

    print(f"找到 {len(all_corrected_tifs)} 个校正后的文件，开始 ROI 检测...")

    ctx = mp.get_context("spawn")
    futures = []
    with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=ctx) as ex:
        for p in all_corrected_tifs:
            futures.append(
                ex.submit(_process_roi_detection, p, root_dir, ops_template)
            )

        for fu in tqdm(as_completed(futures), total=len(futures), desc="ROI Detection"):
            status, exp_id, msg = fu.result()
            tqdm.write(f"[{status.upper()}] {exp_id}: {msg}")

if __name__ == "__main__":
    run_roi_detection_only(
        root_dir=DATA_PATH,
        n_workers=N_WORKERS,
        s2p_nthreads=S2P_NTHREADS,
        batch_size=BATCH_SIZE
    )
    print("ROI Detection Task Done.")