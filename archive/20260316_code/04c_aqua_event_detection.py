# Recommended suite2p version: 0.14.4
# ROI detection in 0.14.5 gave abnormal results on this dataset.
# conda activate suite2p_test

import os
import json
import re
import time
import math
import shutil
import traceback
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tf
import suite2p
from suite2p.run_s2p import default_ops, run_s2p

# ============================================================
# 配置区域
# ============================================================
# 输入数据根目录（递归查找 *_corrected_movie.tif）
DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected'

# 是否做 ROI 检测
ROI_DETECT = True

# 最小帧数（时间序列）要求
MIN_FRAMES = 2

# 并行工作进程数
N_WORKERS = 2

# suite2p 内部线程数
S2P_NTHREADS = 8

# suite2p batch_size
BATCH_SIZE = 500

# 已有 suite2p 文件夹时的处理方式
SUITE2P_MODE = "overwrite"
# "skip"       有 suite2p 就跳过
# "overwrite"  删除旧 suite2p 重新跑
# "keep"       保留旧结果继续跑

# 细胞真实直径（单位：µm）
CELL_DIAMETER_UM = 5.0

# 自动计算出的像素直径最小值，避免算得过小
MIN_DIAMETER_PX = 4

# 是否把自动计算出的 diameter 稍微放大一点
# 例如 5 µm / 0.994 ≈ 5 px；如果想更保守一点可以设 1.1 或 1.2
DIAMETER_SCALE_FACTOR = 1.0

# ROI detection 这套参数偏“多检出一些候选 ROI”
THRESHOLD_SCALING = 0.85
MAX_OVERLAP = 0.85
SNR_THRESH = 1.0
HIGH_PASS = 40
SPATIAL_HP_DETECT = 10

MAX_ITERATIONS = 20
INNER_NEUROPIL_RADIUS = 2
MIN_NEUROPIL_PIXELS = 350
TAU = 1.0


# ============================================================
# 0) 强制限制底层 BLAS/OpenMP 线程数
# ============================================================
def set_low_level_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# 1) 自然排序
# ============================================================
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", s)]


# ============================================================
# 2) 读 metadata
# ============================================================
def get_trial_metadata(json_path):
    """
    从 metadata 读取:
      - fs
      - nplanes
      - nchannels
      - pixel_size_um
      - diameter_px (根据 CELL_DIAMETER_UM 自动推算)
    """
    meta = {
        "nplanes": 1,
        "fs": 1.0,
        "nchannels": 1,
        "pixel_size_um": None,
        "diameter_px": None,
        "json_found": False,
    }

    if not os.path.exists(json_path):
        return meta

    meta["json_found"] = True

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # 时间信息
        meta["fs"] = float(
            data.get("temporal_calibration", {}).get("fps", 1.0)
        )

        # 这里继续按你原来的 pipeline 处理：suite2p 跑的是单个 corrected movie
        # 所以默认 nplanes=1, nchannels=1
        meta["nplanes"] = 1
        meta["nchannels"] = 1

        # 像素尺寸
        physical = data.get("physical_size", {})
        dims = data.get("dimensions", {})

        width_um = physical.get("width", None)
        width_px = dims.get("width_pixel", None)

        if width_um is not None and width_px not in (None, 0):
            pixel_size_um = float(width_um) / float(width_px)
            meta["pixel_size_um"] = pixel_size_um

            diameter_px = CELL_DIAMETER_UM / pixel_size_um
            diameter_px = diameter_px * DIAMETER_SCALE_FACTOR
            diameter_px = int(round(diameter_px))
            diameter_px = max(MIN_DIAMETER_PX, diameter_px)
            meta["diameter_px"] = diameter_px

    except Exception:
        pass

    return meta


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
# 4) 每个 trial 的 ops 模板
# ============================================================
def build_ops_template(
    roidetect=True,
    batch_size=500,
    s2p_nthreads=8,
    diameter_px=5,
):
    """
    为单个 trial 构建 suite2p ops 模板。

    注意：
    - do_registration=False，因为上游已经做过 motion correction
    - diameter_px 是根据 metadata 自动换算出来的“像素单位细胞直径”
    - 其余 ROI detection 参数统一从文件开头的全局配置区读取
    """
    return {
        # ----------------------------
        # registration-related
        # ----------------------------
        "do_registration": False,
        "nonrigid": True,
        "block_size": [128, 128],
        "snr_thresh": SNR_THRESH,
        "maxregshiftNR": 5,
        "nimg_init": 300,
        "maxregshift": 0.1,
        "smooth_sigma": 1.15,
        "smooth_sigma_time": 0,

        # ----------------------------
        # ROI detection
        # ----------------------------
        "roidetect": roidetect,
        "sparse_mode": False,
        "diameter": [int(diameter_px), int(diameter_px)],
        "connected": True,
        "threshold_scaling": THRESHOLD_SCALING,
        "max_overlap": MAX_OVERLAP,
        "max_iterations": MAX_ITERATIONS,

        # ----------------------------
        # filtering / detection
        # ----------------------------
        "high_pass": HIGH_PASS,
        "spatial_hp_detect": SPATIAL_HP_DETECT,

        # ----------------------------
        # neuropil extraction
        # ----------------------------
        "allow_overlap": False,
        "neuropil_extract": True,
        "inner_neuropil_radius": INNER_NEUROPIL_RADIUS,
        "min_neuropil_pixels": MIN_NEUROPIL_PIXELS,

        # ----------------------------
        # deconvolution / runtime
        # ----------------------------
        "tau": TAU,
        "batch_size": int(batch_size),
        "nthreads": int(s2p_nthreads),
        "combined": True,

        # ----------------------------
        # output control
        # ----------------------------
        "reg_tif": False,
        "delete_bin": 0,
        "move_bin": False,
        "keep_movie_raw": False,
    }


# ============================================================
# 5) 子进程任务：处理一个 corrected_movie
# ============================================================
def _process_one_trial(
    corrected_movie_path: str,
    min_frames: int,
    roidetect: bool,
    batch_size: int,
    s2p_nthreads: int,
    suite2p_mode: str,
):
    set_low_level_thread_env()

    trial_dir = os.path.dirname(corrected_movie_path)
    stack_name = os.path.basename(corrected_movie_path)

    exp_id = (stack_name
              .replace("_corrected_movie.tif", "")
              .replace("_corrected_movie.tiff", ""))

    log_path = os.path.join(trial_dir, "suite2p_run.log")

    def log(msg: str):
        with open(log_path, "a") as f:
            f.write(msg.rstrip() + "\n")

    try:
        log(f"=== START {time.ctime()} ===")
        log(f"corrected_movie_path : {corrected_movie_path}")
        log(f"exp_id               : {exp_id}")
        # log(f"suite2p_version      : {suite2p.__version__}")

        suite2p_dir = os.path.join(trial_dir, "suite2p")

        # ----------------------------------
        # 检测已有 suite2p 结果
        # ----------------------------------
        if os.path.exists(suite2p_dir):
            if suite2p_mode == "skip":
                msg = "suite2p folder exists, skipping."
                log(f"[SKIP] {msg}")
                return ("skip", exp_id, msg, None)

            elif suite2p_mode == "overwrite":
                log("[INFO] removing existing suite2p folder...")
                shutil.rmtree(suite2p_dir)

            elif suite2p_mode == "keep":
                log("[INFO] suite2p folder exists, keeping old results.")

            else:
                raise ValueError(f"Unknown suite2p_mode: {suite2p_mode}")

        ok, info = sanity_check_tiff_is_time_series_2d(
            corrected_movie_path,
            min_frames=min_frames
        )

        if not ok:
            log(f"[SKIP] sanity_check failed: {info}")
            return ("skip", exp_id, info, None)

        log(f"sanity_check         : {info}")

        json_path = corrected_movie_path \
            .replace("_corrected_movie.tif", "_metadata.json") \
            .replace("_corrected_movie.tiff", "_metadata.json")

        meta = get_trial_metadata(json_path)

        n_planes = meta["nplanes"]
        fs = meta["fs"]
        n_channels = meta["nchannels"]
        pixel_size_um = meta["pixel_size_um"]

        if meta["diameter_px"] is None:
            diameter_px = max(MIN_DIAMETER_PX, int(round(CELL_DIAMETER_UM)))
        else:
            diameter_px = meta["diameter_px"]

        ops_template = build_ops_template(
            roidetect=roidetect,
            batch_size=batch_size,
            s2p_nthreads=s2p_nthreads,
            diameter_px=diameter_px,
        )

        log(f"json_found           : {meta['json_found']}")
        log(f"params: n_planes={n_planes}, fs={fs}, n_channels={n_channels}")
        log(f"pixel_size_um        : {pixel_size_um}")
        log(f"cell_diameter_um     : {CELL_DIAMETER_UM}")
        log(f"diameter_px          : {diameter_px}")
        log(f"ops[diameter]        : {ops_template['diameter']}")
        log(f"ops[threshold_scaling]: {ops_template['threshold_scaling']}")
        log(f"ops[max_overlap]     : {ops_template['max_overlap']}")
        log(f"ops[high_pass]       : {ops_template['high_pass']}")
        log(f"ops[spatial_hp_detect]: {ops_template['spatial_hp_detect']}")
        log(f"ops[snr_thresh]      : {ops_template['snr_thresh']}")
        log(f"db[data_path]        : {[trial_dir]}")
        log(f"db[tiff_list]        : {[stack_name]}")

        ops = default_ops()
        ops.update(ops_template)
        ops.update({
            "nplanes": n_planes,
            "fs": fs,
            "nchannels": n_channels,
        })

        db = {
            "data_path": [trial_dir],
            "tiff_list": [stack_name],
            "save_path0": trial_dir,
            "nchannels": n_channels,
            "nplanes": n_planes,
        }

        log("[RUN] run_s2p starting...")
        run_s2p(ops=ops, db=db)
        log("[RUN] run_s2p finished.")
        log(f"=== DONE {time.ctime()} ===")

        return ("ok", exp_id, "success", diameter_px)

    except Exception as e:
        log("[ERROR] Exception occurred!")
        log(str(e))
        log(traceback.format_exc())
        return ("error", exp_id, str(e), None)


# ============================================================
# 6) 主函数：并行调度
# ============================================================
def run_suite2p_inplace(
    root_tif_dir,
    roidetect=True,
    min_frames=2,
    n_workers=2,
    s2p_nthreads=8,
    batch_size=500,
    suite2p_mode="skip",
):
    set_low_level_thread_env()

    root_tif_dir = os.path.abspath(root_tif_dir)
    all_trials = []

    for r, dirs, files in os.walk(root_tif_dir):
        dirs[:] = [d for d in dirs if d != "suite2p"]

        for f in sorted(files, key=natural_key):
            if f.endswith("_corrected_movie.tif") or f.endswith("_corrected_movie.tiff"):
                all_trials.append(os.path.join(r, f))

    if not all_trials:
        print(f"[No files] {root_tif_dir} 下没找到 *_corrected_movie.tif(f)")
        return

    print(f"找到 {len(all_trials)} 个 trial，开始处理...")

    ctx = mp.get_context("spawn")
    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=ctx) as ex:
        for p in all_trials:
            futures.append(
                ex.submit(
                    _process_one_trial,
                    p,
                    min_frames,
                    roidetect,
                    batch_size,
                    s2p_nthreads,
                    suite2p_mode,
                )
            )

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Suite2p (inplace)"):
            status, exp_id, msg, diameter_px = fu.result()

            tqdm.write(f"[{status.upper()}] {exp_id}: {msg}")

            results.append({
                "trial": exp_id,
                "status": status,
                "message": msg,
                "diameter_px": diameter_px,
            })

    # -------------------------------------------------
    # 统计 ROI 数量
    # -------------------------------------------------
    for r in results:
        trial = r["trial"]
        stat_path = None

        for p in all_trials:
            if trial in p:
                trial_dir = os.path.dirname(p)
                stat_path = os.path.join(trial_dir, "suite2p", "plane0", "stat.npy")
                break

        if stat_path and os.path.exists(stat_path):
            try:
                stat = np.load(stat_path, allow_pickle=True)
                r["n_rois"] = len(stat)
            except Exception:
                r["n_rois"] = -1
        else:
            r["n_rois"] = 0

    # -------------------------------------------------
    # 保存 summary
    # -------------------------------------------------
    summary_path = os.path.join(root_tif_dir, "suite2p_summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)

    print("\nSummary saved:")
    print(summary_path)

    print("\nROI count preview:")
    cols = [c for c in ["trial", "status", "diameter_px", "n_rois"] if c in df.columns]
    print(df[cols].head())

    print("\nAll done.")


if __name__ == "__main__":
    run_suite2p_inplace(
        root_tif_dir=DATA_PATH,
        roidetect=ROI_DETECT,
        min_frames=MIN_FRAMES,
        n_workers=N_WORKERS,
        s2p_nthreads=S2P_NTHREADS,
        batch_size=BATCH_SIZE,
        suite2p_mode=SUITE2P_MODE,
    )

    print("Done.")