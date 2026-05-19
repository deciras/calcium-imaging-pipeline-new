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
# 输入数据路径
DATA_PATH = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2025_olympus"
# 删除中间计算结果
DELETE_INTERNAL = False
# 是否跳过 ROI 检测，直接用 GUI 手动注册
ROI_DETECT = True
# 最小帧数（时间序列）要求
MIN_FRAMES = 2
# 并行工作进程数
N_WORKERS = 2
# suite2p 内部线程数
S2P_NTHREADS = 8
# suite2p 处理批次大小
BATCH_SIZE = 200


# ============================================================
# 0) 强制限制底层 BLAS/OpenMP 线程数
#    重要：必须在导入 heavy compute 前/尽早设置更好（这里也还来得及）
# ============================================================
def set_low_level_thread_env():
    """
    外层多进程 + 内层 suite2p nthreads + BLAS/OpenMP 默认线程
    很容易造成：线程爆炸 / 竞争 / 随机崩溃（BrokenProcessPool）
    所以把 BLAS/OpenMP 系列都锁到 1。
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# 1) 自然排序：保证 reg_tif_2 < reg_tif_10
# ============================================================
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", s)]


# ============================================================
# 2) 读 metadata（可选）
# ============================================================
def get_params_from_json(json_path):
    n_planes, fs, n_channels = 1, 1.0, 1
    if not os.path.exists(json_path):
        return n_planes, fs, n_channels

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        dims = data.get("dimensions", {})
        n_planes = 1  # Max_Proj 当单 plane 时间序列
        n_channels = 1
        fs = float(data.get("temporal_calibration", {}).get("fps", 1.0))
    except Exception:
        pass

    return n_planes, fs, n_channels


# ============================================================
# 3) TIFF sanity check：是不是多帧时间序列
# ============================================================
def sanity_check_tiff_is_time_series_2d(tif_path, min_frames=2):
    try:
        with tf.TiffFile(tif_path) as tif:
            n_pages = len(tif.pages)
            if n_pages < min_frames:
                return False, f"Only {n_pages} page(s)"

            series = tif.series[0]
            ndim = getattr(series, "ndim", None)
            shape = getattr(series, "shape", None)

            if ndim == 3:
                return True, f"OK: shape={shape} (T×Y×X)"
            elif ndim == 2:
                return True, f"OK: multipage 2D (pages={n_pages}, frame={shape})"
            else:
                return False, f"Unsupported ndim={ndim}, shape={shape}"
    except Exception as e:
        return False, f"Failed to read tiff: {e}"


# ============================================================
# 4) 合并 reg_tif chunks 为一个完整 tiff（顺序正确）
# ============================================================
def merge_reg_tifs_to_one(plane_reg_dir, out_path, delete_chunks=True):
    chunk_files = [
        fn for fn in os.listdir(plane_reg_dir)
        if fn.lower().endswith((".tif", ".tiff"))
    ]
    if not chunk_files:
        raise FileNotFoundError(f"No reg_tif chunks in {plane_reg_dir}")

    chunk_files = sorted(chunk_files, key=natural_key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tf.TiffWriter(out_path, bigtiff=True) as tw:
        for fn in chunk_files:
            arr = tf.imread(os.path.join(plane_reg_dir, fn))
            if arr.ndim == 2:
                tw.write(arr)
            elif arr.ndim == 3:
                for i in range(arr.shape[0]):
                    tw.write(arr[i])
            else:
                raise ValueError(f"Unexpected ndim={arr.ndim} in {fn}")

    if delete_chunks:
        for fn in chunk_files:
            os.remove(os.path.join(plane_reg_dir, fn))


# ============================================================
# 5) 子进程任务：处理一个 Max_Proj
# ============================================================
def _process_one_maxproj(
    max_proj_path: str,
    root_tif_dir: str,
    internal_results_root: str,
    organized_results_root: str,
    delete_internal: bool,
    min_frames: int,
    ops_template: dict,
    delete_chunk_tifs: bool = True,
):
    """
    这个函数跑在子进程里：
    1) suite2p run_s2p
    2) 合并 reg_tif
    3) 清理 internal
    同时：写 per-task log，方便定位崩溃点
    """
    # 每个子进程也设置一次（保险）
    set_low_level_thread_env()

    rel_dir = os.path.relpath(os.path.dirname(max_proj_path), root_tif_dir)
    stack_name = os.path.basename(max_proj_path)
    exp_id = stack_name.replace("_Max_Proj.tif", "").replace("_Max_Proj.tiff", "")

    # 日志目录：最终结果根目录下的 _logs
    log_dir = os.path.join(organized_results_root, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{os.getpid()}_{exp_id}.log")

    def log(msg: str):
        with open(log_path, "a") as f:
            f.write(msg.rstrip() + "\n")

    try:
        log(f"=== START {time.ctime()} ===")
        log(f"max_proj_path: {max_proj_path}")
        log(f"rel_dir      : {rel_dir}")
        log(f"exp_id       : {exp_id}")

        json_path = max_proj_path.replace("_Max_Proj.tif", "_metadata.json").replace(
            "_Max_Proj.tiff", "_metadata.json"
        )
        log(f"json_path    : {json_path}")

        ok, info = sanity_check_tiff_is_time_series_2d(max_proj_path, min_frames=min_frames)
        if not ok:
            log(f"[SKIP] sanity_check failed: {info}")
            return ("skip", exp_id, info)

        n_planes, fs, n_channels = get_params_from_json(json_path)
        log(f"params: n_planes={n_planes}, fs={fs}, n_channels={n_channels}")

        # 每个任务独立 ops（避免串台）
        ops = suite2p.default_ops()
        ops.update(ops_template)
        ops.update({"nplanes": n_planes, "fs": fs, "nchannels": n_channels})

        # 内部临时输出（suite2p原生结构）
        temp_save_path = os.path.join(internal_results_root, rel_dir)

        # 最终整理输出（干净结构）
        final_exp_folder = os.path.join(organized_results_root, rel_dir)

        log(f"temp_save_path : {temp_save_path}")
        log(f"final_exp_folder: {final_exp_folder}")

        db = {
            "data_path": [os.path.dirname(max_proj_path)],
            "tiff_list": [stack_name],
            "save_path0": temp_save_path,
            "nchannels": n_channels,
            "nplanes": n_planes,
        }

        log("[RUN] run_s2p starting...")
        run_s2p(ops=ops, db=db)
        log("[RUN] run_s2p finished.")

        # 整理输出
        os.makedirs(final_exp_folder, exist_ok=True)
        if os.path.exists(json_path):
            shutil.copy(json_path, final_exp_folder)
            log("[COPY] metadata copied.")

        # 合并 reg_tif
        search_base = os.path.join(temp_save_path, "suite2p")
        for p in range(n_planes):
            plane_reg_dir = os.path.join(search_base, f"plane{p}", "reg_tif")
            if not os.path.exists(plane_reg_dir):
                log(f"[WARN] reg_tif dir missing: {plane_reg_dir}")
                continue

            plane_dst_dir = os.path.join(final_exp_folder, f"Plane{p}")
            out_path = os.path.join(plane_dst_dir, f"{exp_id}_P{p}_Reg_full.tif")

            log(f"[MERGE] {plane_reg_dir} -> {out_path}")
            merge_reg_tifs_to_one(plane_reg_dir, out_path, delete_chunks=delete_chunk_tifs)
            log("[MERGE] done.")

        # 清理 internal
        if delete_internal and os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)
            log("[CLEAN] temp_save_path removed.")

        log(f"=== DONE {time.ctime()} ===")
        return ("ok", exp_id, info)

    except Exception as e:
        log("[ERROR] Exception occurred!")
        log(str(e))
        log(traceback.format_exc())
        return ("error", exp_id, str(e))


# ============================================================
# 6) 主函数：并行调度
# ============================================================
def run_suite2p_on_maxproj_2d_tiffs(
    root_tif_dir,
    delete_internal=True,
    roidetect=True,
    min_frames=2,
    n_workers=2,
    s2p_nthreads=8,
    batch_size=200,
):
    """
    参数解释：
    - root_tif_dir:
        输入数据根目录（递归寻找 *_Max_Proj.tif）
    - internal_results_root:
        suite2p 原生输出（临时，中间文件多，结构深）
    - organized_results_root:
        你最终要用的输出目录（干净、结构跟输入一致）
    - n_workers:
        同时跑多少个实验（多进程）
    - s2p_nthreads:
        每个实验内部 suite2p 用多少线程（多线程）
    - batch_size:
        suite2p 处理帧的分块大小（它会影响 reg_tif 碎块数量，但我们后面会合并）
    """
    # 主进程也设置一次线程环境
    set_low_level_thread_env()

    root_tif_dir = os.path.abspath(root_tif_dir)
    parent_dir = os.path.dirname(root_tif_dir)
    base_name = os.path.basename(root_tif_dir)

    internal_results_root = os.path.join(parent_dir, f"{base_name}_Suite2p_Internal")
    organized_results_root = os.path.join(parent_dir, f"{base_name}_Suite2p_Results")

    # suite2p 配置模板（包含所有解决顽固位移的重要参数）
    ops_template = {
        # --- 基础运行控制 ---
        "do_registration": 1,        # 强制运行运动校正 
        "roidetect": roidetect,          # 跳过自动检测，由你手动 register [cite: 3, 8]
        "reg_tif": True,             # 直接保存校正后的 TIF 文件 [cite: 9]
        "delete_bin": 0,             # 保留原始二进制文件以便后续在 GUI 查看
        
        # --- 非刚性校正核心 (解决局部晃动) ---
        "nonrigid": True,            # 开启非刚性校正 [cite: 9]
        "block_size": [128, 128],      # 减小块大小（默认128），针对视网膜等组织的精细扭曲
        "snr_thresh": 0.25,           # 降低信噪比阈值，防止暗区域因信号弱而对齐失败
        "maxregshiftNR": 10,         # 增加非刚性块的最大允许位移（像素）
        
        # --- 刚性校正优化 (解决大跳动) ---
        "nimg_init": 500,            # 增加用于计算参考帧的帧数，使对齐基准更稳 [cite: 3]
        "maxregshift": 0.2,          # 允许的最大整体位移（长宽的20%），防止大幅度跳动对不齐
        "smooth_sigma": 1.5,         # 增加高斯平滑半径，过滤噪声，让算法更看重细胞结构
        
        # --- 性能与多线程 ---
        "batch_size": int(batch_size),
        "nthreads": int(s2p_nthreads),
        "combined": False,
    }

    # 扫描所有 Max_Proj
    all_Maxproj = []
    for r, _, files in os.walk(root_tif_dir):
        for f in files:
            if f.endswith("_Max_Proj.tif") or f.endswith("_Max_Proj.tiff"):
                all_Maxproj.append(os.path.join(r, f))
    all_Maxproj.sort()

    if not all_Maxproj:
        print(f"[No files] {root_tif_dir} 下没找到 *_Max_Proj.tif(f)")
        return

    # ✅ 关键：用 spawn（避免 fork + numba/openmp 的不稳定）
    ctx = mp.get_context("spawn")

    futures = []
    with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=ctx) as ex:
        for p in all_Maxproj:
            futures.append(
                ex.submit(
                    _process_one_maxproj,
                    p,
                    root_tif_dir,
                    internal_results_root,
                    organized_results_root,
                    delete_internal,
                    min_frames,
                    ops_template,
                )
            )

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Suite2p (spawn)"):
            status, exp_id, msg = fu.result()
            tqdm.write(f"[{status.upper()}] {exp_id}: {msg}")

    print(f"Done. Logs are in: {os.path.join(organized_results_root, '_logs')}")

    return organized_results_root


if __name__ == "__main__":

    organized_results_root = run_suite2p_on_maxproj_2d_tiffs(
        root_tif_dir=DATA_PATH,
        delete_internal=DELETE_INTERNAL,
        roidetect=ROI_DETECT,
        min_frames=MIN_FRAMES,
        n_workers=N_WORKERS,
        s2p_nthreads=S2P_NTHREADS,
        batch_size=BATCH_SIZE
    )

    subprocess.run(["bash", "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/20260127_code/03_move_tif.sh", organized_results_root])

    print("Done.")