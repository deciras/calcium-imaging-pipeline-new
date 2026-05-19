# Recommended suite2p version: 0.14.4
# ROI detection in 0.14.5 gave abnormal results on this dataset.
# conda activate suite2p_test
#
# 04b: run suite2p, then export benchmark-ready outputs to:
#   trial_dir/benchmark/suite2p/
#       roi_mask.tif
#       roi_label_map.tif
#       roi_summary.csv
#       roi_overlay.png

import os
import json
import re
import time
import shutil
import traceback
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tf
import suite2p
from suite2p.run_s2p import default_ops, run_s2p
import importlib.metadata

import matplotlib.pyplot as plt

# ============================================================
# 修复 #17：运行时版本检查
# ============================================================
_RECOMMENDED_S2P_VERSION = "0.14.4"

try:
    _s2p_ver = importlib.metadata.version("suite2p")
except Exception:
    _s2p_ver = getattr(suite2p, "__version__", "unknown")

if _s2p_ver != _RECOMMENDED_S2P_VERSION:
    print(
        f"\n{'!'*70}\n"
        f"[WARNING] suite2p version mismatch!\n"
        f"  Installed : {_s2p_ver}\n"
        f"  Recommended: {_RECOMMENDED_S2P_VERSION}\n"
        f"  ROI detection in 0.14.5 gave abnormal results on this dataset.\n"
        f"  Activate the correct env: conda activate suite2p_test\n"
        f"{'!'*70}\n"
    )

# ============================================================
# 配置区域
# ============================================================
DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected'

ROI_DETECT = True
MIN_FRAMES = 2
N_WORKERS = 2
S2P_NTHREADS = 8
BATCH_SIZE = 500

SUITE2P_MODE = "skip"
# "skip" / "overwrite" / "keep"

CELL_DIAMETER_UM = 5.0
MIN_DIAMETER_PX = 4
DIAMETER_SCALE_FACTOR = 1.0

THRESHOLD_SCALING = 0.85
MAX_OVERLAP = 0.85
SNR_THRESH = 1.0
HIGH_PASS = 40
SPATIAL_HP_DETECT = 10

MAX_ITERATIONS = 20
INNER_NEUROPIL_RADIUS = 2
MIN_NEUROPIL_PIXELS = 350
TAU = 1.0

# benchmark export
EXPORT_BENCHMARK = True
BENCHMARK_OVERWRITE = True
OVERLAY_FIGSIZE = (8, 8)
OVERLAY_DPI = 150

# ============================================================
# 0) 强制限制底层 BLAS/OpenMP 线程数
# ============================================================
def set_low_level_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# 1) 通用工具
# ============================================================
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", s)]


def safe_makedirs(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clear_dir_contents(path: str | Path):
    path = Path(path)
    if not path.exists():
        return
    for p in path.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            print(f"    [Warning] Failed to remove {p}: {e}")


def save_uint8_tiff(path: str | Path, image: np.ndarray):
    tf.imwrite(str(path), np.asarray(image, dtype=np.uint8), imagej=True)


def save_uint16_tiff(path: str | Path, image: np.ndarray):
    tf.imwrite(str(path), np.asarray(image, dtype=np.uint16), imagej=True)


# ============================================================
# 2) 读 metadata
# ============================================================
def get_trial_metadata(json_path):
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

        meta["fs"] = float(
            data.get("temporal_calibration", {}).get("fps", 1.0)
        )

        meta["nplanes"] = 1
        meta["nchannels"] = 1

        physical = data.get("physical_size", {})
        dims = data.get("dimensions", {})

        if physical.get("pixel_size_um") is not None:
            pixel_size_um = float(physical["pixel_size_um"])
            meta["pixel_size_um"] = pixel_size_um
        else:
            fov_width_um = physical.get("fov_width_um", physical.get("width", None))
            width_px = dims.get("width_pixel", None)
            if fov_width_um is not None and width_px not in (None, 0):
                pixel_size_um = float(fov_width_um) / float(width_px)
                meta["pixel_size_um"] = pixel_size_um
            else:
                pixel_size_um = None

        if pixel_size_um is not None:
            diameter_px = CELL_DIAMETER_UM / pixel_size_um
            diameter_px = diameter_px * DIAMETER_SCALE_FACTOR
            diameter_px = int(round(diameter_px))
            diameter_px = max(MIN_DIAMETER_PX, diameter_px)
            meta["diameter_px"] = diameter_px

    except Exception:
        pass

    return meta


# ============================================================
# 3) TIFF sanity check
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
    return {
        "do_registration": False,
        "nonrigid": True,
        "block_size": [128, 128],
        "snr_thresh": SNR_THRESH,
        "maxregshiftNR": 5,
        "nimg_init": 300,
        "maxregshift": 0.1,
        "smooth_sigma": 1.15,
        "smooth_sigma_time": 0,

        "roidetect": roidetect,
        "sparse_mode": False,
        "diameter": [int(diameter_px), int(diameter_px)],
        "connected": True,
        "threshold_scaling": THRESHOLD_SCALING,
        "max_overlap": MAX_OVERLAP,
        "max_iterations": MAX_ITERATIONS,

        "high_pass": HIGH_PASS,
        "spatial_hp_detect": SPATIAL_HP_DETECT,

        "allow_overlap": False,
        "neuropil_extract": True,
        "inner_neuropil_radius": INNER_NEUROPIL_RADIUS,
        "min_neuropil_pixels": MIN_NEUROPIL_PIXELS,

        "tau": TAU,
        "batch_size": int(batch_size),
        "nthreads": int(s2p_nthreads),
        "combined": True,

        "reg_tif": False,
        "delete_bin": 0,
        "move_bin": False,
        "keep_movie_raw": False,
    }


# ============================================================
# 5) benchmark export
# ============================================================
def _find_plane0_dir(trial_dir: str | Path) -> Path | None:
    trial_dir = Path(trial_dir)
    plane0 = trial_dir / "suite2p" / "plane0"
    if plane0.exists():
        return plane0
    return None


def _load_iscell(plane0_dir: Path, n_stat: int):
    iscell_path = plane0_dir / "iscell.npy"
    if not iscell_path.exists():
        return np.ones(n_stat, dtype=bool), np.full(n_stat, np.nan, dtype=float)

    iscell = np.load(iscell_path, allow_pickle=True)
    if iscell.ndim != 2 or iscell.shape[0] != n_stat:
        return np.ones(n_stat, dtype=bool), np.full(n_stat, np.nan, dtype=float)

    flags = iscell[:, 0].astype(bool)
    prob = iscell[:, 1].astype(float) if iscell.shape[1] >= 2 else np.full(n_stat, np.nan, dtype=float)
    return flags, prob


def _build_roi_maps(stat, Ly: int, Lx: int):
    roi_mask = np.zeros((Ly, Lx), dtype=np.uint8)
    roi_label_map = np.zeros((Ly, Lx), dtype=np.int32)

    for i, s in enumerate(stat, start=1):
        xpix = np.asarray(s.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(s.get("ypix", []), dtype=np.int32)
        if len(xpix) == 0 or len(ypix) == 0:
            continue
        good = (xpix >= 0) & (xpix < Lx) & (ypix >= 0) & (ypix < Ly)
        xpix = xpix[good]
        ypix = ypix[good]
        if len(xpix) == 0:
            continue
        roi_mask[ypix, xpix] = 255
        roi_label_map[ypix, xpix] = i

    return roi_mask, roi_label_map


def _summarize_rois(stat, iscell_flag, prob, pixel_size_um, trial_name, prefix):
    rows = []
    for i, s in enumerate(stat, start=1):
        xpix = np.asarray(s.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(s.get("ypix", []), dtype=np.int32)
        if len(xpix) == 0 or len(ypix) == 0:
            area_px = 0
            centroid_x = np.nan
            centroid_y = np.nan
            bbox_xmin = bbox_xmax = bbox_ymin = bbox_ymax = np.nan
        else:
            area_px = int(len(xpix))
            centroid_x = float(np.mean(xpix))
            centroid_y = float(np.mean(ypix))
            bbox_xmin = int(np.min(xpix))
            bbox_xmax = int(np.max(xpix))
            bbox_ymin = int(np.min(ypix))
            bbox_ymax = int(np.max(ypix))

        med = s.get("med", [np.nan, np.nan])
        med_y = float(med[0]) if len(med) > 0 else np.nan
        med_x = float(med[1]) if len(med) > 1 else np.nan

        row = {
            "trial": trial_name,
            "prefix": prefix,
            "roi_id": i,
            "suite2p_index": i - 1,
            "iscell": int(bool(iscell_flag[i - 1])) if i - 1 < len(iscell_flag) else 1,
            "iscell_prob": float(prob[i - 1]) if i - 1 < len(prob) and np.isfinite(prob[i - 1]) else np.nan,
            "area_px": area_px,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bbox_xmin": bbox_xmin,
            "bbox_xmax": bbox_xmax,
            "bbox_ymin": bbox_ymin,
            "bbox_ymax": bbox_ymax,
            "med_x": med_x,
            "med_y": med_y,
        }
        row["area_um2"] = float(area_px * (pixel_size_um ** 2)) if pixel_size_um is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["iscell", "area_px", "roi_id"], ascending=[False, False, True]).reset_index(drop=True)
    return df


def _plot_roi_overlay(mean_img, stat, iscell_flag, out_png: Path, trial_name: str):
    fig, ax = plt.subplots(figsize=OVERLAY_FIGSIZE)
    ax.imshow(mean_img, cmap="gray")

    for i, s in enumerate(stat):
        xpix = np.asarray(s.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(s.get("ypix", []), dtype=np.int32)
        if len(xpix) == 0 or len(ypix) == 0:
            continue
        color = "lime" if iscell_flag[i] else "red"
        ax.scatter(xpix, ypix, s=0.5, c=color, alpha=0.7)

    ax.set_title(f"Suite2p ROI overlay - {trial_name}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=OVERLAY_DPI, bbox_inches="tight")
    plt.close(fig)


def export_suite2p_benchmark(trial_dir: str | Path, prefix: str, pixel_size_um=None):
    plane0_dir = _find_plane0_dir(trial_dir)
    if plane0_dir is None:
        raise FileNotFoundError(f"suite2p/plane0 not found: {trial_dir}")

    stat_path = plane0_dir / "stat.npy"
    ops_path = plane0_dir / "ops.npy"

    if not stat_path.exists():
        raise FileNotFoundError(f"stat.npy not found: {stat_path}")
    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found: {ops_path}")

    stat = np.load(stat_path, allow_pickle=True)
    ops = np.load(ops_path, allow_pickle=True).item()

    Ly = int(ops.get("Ly"))
    Lx = int(ops.get("Lx"))
    mean_img = ops.get("meanImg", None)
    if mean_img is None:
        mean_img = np.zeros((Ly, Lx), dtype=np.float32)
    else:
        mean_img = np.asarray(mean_img, dtype=np.float32)

    iscell_flag, prob = _load_iscell(plane0_dir, len(stat))
    roi_mask, roi_label_map = _build_roi_maps(stat, Ly=Ly, Lx=Lx)
    roi_summary = _summarize_rois(
        stat=stat,
        iscell_flag=iscell_flag,
        prob=prob,
        pixel_size_um=pixel_size_um,
        trial_name=Path(trial_dir).name,
        prefix=prefix,
    )

    out_dir = Path(trial_dir) / "benchmark" / "suite2p"
    safe_makedirs(out_dir)
    if BENCHMARK_OVERWRITE:
        clear_dir_contents(out_dir)
        safe_makedirs(out_dir)

    save_uint8_tiff(out_dir / "roi_mask.tif", roi_mask)

    max_label = int(np.max(roi_label_map)) if roi_label_map.size > 0 else 0
    if max_label <= np.iinfo(np.uint16).max:
        save_uint16_tiff(out_dir / "roi_label_map.tif", roi_label_map.astype(np.uint16))
    else:
        tf.imwrite(str(out_dir / "roi_label_map.tif"), roi_label_map.astype(np.uint32), imagej=False)

    roi_summary.to_csv(out_dir / "roi_summary.csv", index=False)
    _plot_roi_overlay(mean_img, stat, iscell_flag, out_dir / "roi_overlay.png", Path(trial_dir).name)

    return {
        "benchmark_dir": str(out_dir),
        "n_rois": int(len(stat)),
        "n_iscell": int(np.sum(iscell_flag)),
    }


# ============================================================
# 6) 子进程任务：处理一个 corrected_movie
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

        suite2p_dir = os.path.join(trial_dir, "suite2p")

        if os.path.exists(suite2p_dir):
            if suite2p_mode == "skip":
                msg = "suite2p folder exists, skipping."
                log(f"[SKIP] {msg}")
                return ("skip", exp_id, msg, None, 0, 0)

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
            return ("skip", exp_id, info, None, 0, 0)

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

        n_rois = 0
        n_iscell = 0
        if EXPORT_BENCHMARK:
            bench_info = export_suite2p_benchmark(
                trial_dir=trial_dir,
                prefix=exp_id,
                pixel_size_um=pixel_size_um,
            )
            n_rois = bench_info["n_rois"]
            n_iscell = bench_info["n_iscell"]
            log(f"[BENCHMARK] exported to: {bench_info['benchmark_dir']}")
            log(f"[BENCHMARK] n_rois={n_rois}, n_iscell={n_iscell}")

        log(f"=== DONE {time.ctime()} ===")

        return ("ok", exp_id, "success", diameter_px, n_rois, n_iscell)

    except Exception as e:
        log("[ERROR] Exception occurred!")
        log(str(e))
        log(traceback.format_exc())
        return ("error", exp_id, str(e), None, 0, 0)


# ============================================================
# 7) 主函数：并行调度
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
            status, exp_id, msg, diameter_px, n_rois, n_iscell = fu.result()

            tqdm.write(f"[{status.upper()}] {exp_id}: {msg}")

            results.append({
                "trial": exp_id,
                "status": status,
                "message": msg,
                "diameter_px": diameter_px,
                "n_rois": n_rois,
                "n_iscell": n_iscell,
            })

    summary_path = os.path.join(root_tif_dir, "suite2p_summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)

    print("\nSummary saved:")
    print(summary_path)

    print("\nROI count preview:")
    cols = [c for c in ["trial", "status", "diameter_px", "n_rois", "n_iscell"] if c in df.columns]
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
