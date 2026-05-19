import os
import re
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 配置区域
# ============================================================
# 输入根目录：递归查找 suite2p/plane0/F.npy
DATA_PATH = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"

# neuropil 扣除系数
NEUROPIL_COEF = 0.7

# baseline 相关
DEFAULT_BASELINE_SEC = 10.0     # 没有 stim 信息时，默认用前 10 秒
MIN_BASELINE_FRAMES = 5         # baseline 至少要有这么多帧
BASELINE_QUANTILE = 0.1         # 如果想改成分位数基线，可设 0.1 / 0.2；None 表示均值

# 数值稳定项
EPS = 1e-6

# 是否裁剪极端负值，避免 F0 很小时 dF/F 爆炸
CLIP_MIN_DFF = -1.0
CLIP_MAX_DFF = None

# 输出图
SAVE_PREVIEW_PLOT = True
PREVIEW_MAX_ROIS = 20

# 文件名
PREPROCESSED_SUFFIX = "_preprocessed.csv"
ROI_SUMMARY_SUFFIX = "_roi_summary.csv"
PREVIEW_SUFFIX = "_dff_preview.png"


# ============================================================
# 工具函数
# ============================================================
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", s)]


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_json(json_path: Path):
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_fps_from_metadata(meta: dict, default_fps=1.0) -> float:
    fps = None
    if isinstance(meta, dict):
        fps = meta.get("temporal_calibration", {}).get("fps", None)
        if fps is None:
            fps = meta.get("fps", None)
    fps = safe_float(fps, default_fps)
    if fps is None or fps <= 0:
        fps = default_fps
    return float(fps)


def find_trial_dir_from_plane0(plane0_dir: Path) -> Path:
    # .../trial/suite2p/plane0 -> .../trial
    return plane0_dir.parent.parent


def infer_trial_prefix(trial_dir: Path) -> str:
    """
    从 corrected movie / metadata 文件推断统一前缀。
    例如：
      trial_xxx_corrected_movie.tif -> trial_xxx
      trial_xxx_metadata.json       -> trial_xxx
    """
    tif_files = sorted(
        trial_dir.glob("*_corrected_movie.tif"),
        key=lambda p: natural_key(p.name)
    )
    if tif_files:
        return tif_files[0].name.replace("_corrected_movie.tif", "").replace("_corrected_movie.tiff", "")

    json_files = sorted(
        trial_dir.glob("*_metadata.json"),
        key=lambda p: natural_key(p.name)
    )
    if json_files:
        return json_files[0].name.replace("_metadata.json", "")

    return trial_dir.name


def find_metadata_json(trial_dir: Path, prefix: str) -> Path:
    p = trial_dir / f"{prefix}_metadata.json"
    if p.exists():
        return p

    candidates = sorted(trial_dir.glob("*_metadata.json"), key=lambda x: natural_key(x.name))
    if candidates:
        return candidates[0]

    return p


def find_trial_stim_row(trial_dir: Path, prefix: str):
    """
    尝试从 date-level stim_events.csv 里找到当前 trial 的刺激行。
    这里只做“最小可用版”：
    - 优先按 trial_folder / trial / exp_id / file_name / recording_name 等常见列匹配
    - 找不到就返回 None
    """
    date_dir = trial_dir.parent
    stim_events_path = date_dir / "stim_events.csv"
    if not stim_events_path.exists():
        return None, None

    try:
        df = pd.read_csv(stim_events_path)
    except Exception:
        return None, stim_events_path

    if df.empty:
        return None, stim_events_path

    candidate_cols = [
        "trial_folder", "trial", "exp_id", "file_name", "recording_name", "name"
    ]

    key_values = {
        "trial_folder": trial_dir.name,
        "trial": trial_dir.name,
        "exp_id": prefix,
        "file_name": prefix,
        "recording_name": prefix,
        "name": prefix,
    }

    for col in candidate_cols:
        if col in df.columns:
            sub = df[df[col].astype(str) == str(key_values[col])]
            if len(sub) > 0:
                return sub.copy(), stim_events_path

    return None, stim_events_path


def get_first_stim_start_sec(trial_dir: Path, prefix: str, meta: dict):
    """
    baseline 优先级：
    1) stim_events.csv 里的 start_time_sec 最小值
    2) metadata 里的 stimulation.stim_parameters.initial_delay_sec
    3) None
    """
    stim_df, _ = find_trial_stim_row(trial_dir, prefix)
    if stim_df is not None and "start_time_sec" in stim_df.columns:
        vals = pd.to_numeric(stim_df["start_time_sec"], errors="coerce").dropna()
        if len(vals) > 0:
            return float(vals.min())

    stim = meta.get("stimulation", {}) if isinstance(meta, dict) else {}
    params = stim.get("stim_parameters", {}) if isinstance(stim, dict) else {}
    delay = safe_float(params.get("initial_delay_sec", None), None)
    if delay is not None and delay >= 0:
        return float(delay)

    return None


def compute_baseline_frame_count(n_frames: int, fps: float, first_stim_start_sec):
    if first_stim_start_sec is not None:
        n_base = int(np.floor(first_stim_start_sec * fps))
        if n_base >= MIN_BASELINE_FRAMES:
            return min(n_base, n_frames)

    fallback = int(round(DEFAULT_BASELINE_SEC * fps))
    fallback = max(fallback, MIN_BASELINE_FRAMES)
    return min(fallback, n_frames)


def compute_f0_from_baseline(Fcorr: np.ndarray, n_base_frames: int):
    """
    Fcorr: shape = [n_roi, n_frames]
    返回每个 ROI 的 F0, shape = [n_roi, 1]
    """
    baseline = Fcorr[:, :n_base_frames]

    if BASELINE_QUANTILE is None:
        F0 = np.mean(baseline, axis=1, keepdims=True)
    else:
        F0 = np.quantile(baseline, BASELINE_QUANTILE, axis=1, keepdims=True)

    # 避免 0 或负数
    F0 = np.where(np.abs(F0) < EPS, EPS, F0)
    F0 = np.where(F0 <= 0, EPS, F0)
    return F0


def clip_dff_values(dff: np.ndarray):
    if CLIP_MIN_DFF is not None:
        dff = np.maximum(dff, CLIP_MIN_DFF)
    if CLIP_MAX_DFF is not None:
        dff = np.minimum(dff, CLIP_MAX_DFF)
    return dff


def save_preview_plot(dff_df: pd.DataFrame, fps: float, out_path: Path):
    if dff_df.empty:
        return

    n_show = min(PREVIEW_MAX_ROIS, dff_df.shape[1])
    cols = list(dff_df.columns[:n_show])

    time_axis = np.arange(len(dff_df)) / fps if fps > 0 else np.arange(len(dff_df))

    fig_h = max(4, 0.6 * n_show)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, fig_h), sharex=True)

    if n_show == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(time_axis, dff_df[col].values, lw=0.8)
        ax.set_ylabel(col, rotation=0, labelpad=18, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("dF/F preview", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# 单个 trial 处理
# ============================================================
def process_one_trial(plane0_dir: Path):
    trial_dir = find_trial_dir_from_plane0(plane0_dir)
    prefix = infer_trial_prefix(trial_dir)

    F_path = plane0_dir / "F.npy"
    Fneu_path = plane0_dir / "Fneu.npy"
    iscell_path = plane0_dir / "iscell.npy"
    stat_path = plane0_dir / "stat.npy"

    if not F_path.exists():
        return {
            "trial": trial_dir.name,
            "prefix": prefix,
            "status": "skip",
            "message": "F.npy not found",
            "n_cell": 0,
        }

    if not Fneu_path.exists():
        return {
            "trial": trial_dir.name,
            "prefix": prefix,
            "status": "skip",
            "message": "Fneu.npy not found",
            "n_cell": 0,
        }

    if not iscell_path.exists():
        return {
            "trial": trial_dir.name,
            "prefix": prefix,
            "status": "skip",
            "message": "iscell.npy not found",
            "n_cell": 0,
        }

    try:
        meta_path = find_metadata_json(trial_dir, prefix)
        meta = load_json(meta_path)
        fps = get_fps_from_metadata(meta, default_fps=1.0)

        F = np.load(F_path, allow_pickle=True)
        Fneu = np.load(Fneu_path, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)

        if F.ndim != 2 or Fneu.ndim != 2:
            raise ValueError(f"F/Fneu shape invalid: F={F.shape}, Fneu={Fneu.shape}")

        if F.shape != Fneu.shape:
            raise ValueError(f"F and Fneu shape mismatch: {F.shape} vs {Fneu.shape}")

        if iscell.ndim != 2 or iscell.shape[1] < 1:
            raise ValueError(f"iscell.npy shape invalid: {iscell.shape}")

        if iscell.shape[0] != F.shape[0]:
            raise ValueError(f"iscell length mismatch: {iscell.shape[0]} vs n_roi={F.shape[0]}")

        iscell_flag = iscell[:, 0].astype(bool)
        prob = iscell[:, 1].astype(float) if iscell.shape[1] >= 2 else np.full(F.shape[0], np.nan)

        cell_idx = np.where(iscell_flag)[0]
        if len(cell_idx) == 0:
            return {
                "trial": trial_dir.name,
                "prefix": prefix,
                "status": "skip",
                "message": "No cell ROI in iscell.npy",
                "n_cell": 0,
            }

        # 只保留 cell ROI，顺序和 05 的 RoiSet.zip 一致
        F_cell = F[cell_idx, :]
        Fneu_cell = Fneu[cell_idx, :]
        prob_cell = prob[cell_idx]

        # neuropil subtraction
        Fcorr = F_cell - NEUROPIL_COEF * Fneu_cell

        n_frames = Fcorr.shape[1]
        first_stim_start_sec = get_first_stim_start_sec(trial_dir, prefix, meta)
        n_base_frames = compute_baseline_frame_count(
            n_frames=n_frames,
            fps=fps,
            first_stim_start_sec=first_stim_start_sec,
        )

        F0 = compute_f0_from_baseline(Fcorr, n_base_frames)
        dff = (Fcorr - F0) / (F0 + EPS)
        dff = clip_dff_values(dff)

        # 输出 DataFrame：行=frame，列=0001,0002,...
        roi_names = [f"{i+1:04d}" for i in range(len(cell_idx))]
        dff_df = pd.DataFrame(dff.T, columns=roi_names)

        out_csv = trial_dir / f"{prefix}{PREPROCESSED_SUFFIX}"
        dff_df.to_csv(out_csv, index=False)

        # ROI summary，便于之后核对 suite2p 原始 index
        summary_df = pd.DataFrame({
            "roi_name": roi_names,
            "suite2p_index": cell_idx,
            "iscell_prob": prob_cell,
            "f0": F0.flatten(),
        })
        if stat_path.exists():
            try:
                stat = np.load(stat_path, allow_pickle=True)
                npix_list = []
                med_y = []
                med_x = []
                for idx in cell_idx:
                    s = stat[idx]
                    npix_list.append(len(s.get("xpix", [])))
                    med = s.get("med", [np.nan, np.nan])
                    med_y.append(med[0] if len(med) > 0 else np.nan)
                    med_x.append(med[1] if len(med) > 1 else np.nan)
                summary_df["npix"] = npix_list
                summary_df["med_y"] = med_y
                summary_df["med_x"] = med_x
            except Exception:
                pass

        out_summary = trial_dir / f"{prefix}{ROI_SUMMARY_SUFFIX}"
        summary_df.to_csv(out_summary, index=False)

        if SAVE_PREVIEW_PLOT:
            out_plot = trial_dir / f"{prefix}{PREVIEW_SUFFIX}"
            save_preview_plot(dff_df, fps=fps, out_path=out_plot)

        return {
            "trial": trial_dir.name,
            "prefix": prefix,
            "status": "ok",
            "message": "success",
            "n_cell": len(cell_idx),
            "fps": fps,
            "baseline_frames": n_base_frames,
            "first_stim_start_sec": first_stim_start_sec,
        }

    except Exception as e:
        return {
            "trial": trial_dir.name,
            "prefix": prefix,
            "status": "error",
            "message": f"{e}\n{traceback.format_exc()}",
            "n_cell": 0,
        }


# ============================================================
# 主流程
# ============================================================
def find_all_plane0_dirs(root_dir: Path):
    plane0_dirs = []
    for p in root_dir.rglob("plane0"):
        if p.is_dir() and (p / "F.npy").exists():
            plane0_dirs.append(p)
    plane0_dirs = sorted(plane0_dirs, key=lambda x: natural_key(str(x)))
    return plane0_dirs


def main():
    root_dir = Path(DATA_PATH).resolve()
    plane0_dirs = find_all_plane0_dirs(root_dir)

    if not plane0_dirs:
        print(f"[No files] No suite2p/plane0/F.npy found under: {root_dir}")
        return

    print(f"Found {len(plane0_dirs)} trial(s).")

    results = []
    for plane0_dir in plane0_dirs:
        trial_dir = find_trial_dir_from_plane0(plane0_dir)
        print(f"\nProcessing: {trial_dir}")
        res = process_one_trial(plane0_dir)
        print(f"  [{res['status'].upper()}] {res['prefix']}: {res['message']}")
        results.append(res)

    summary_df = pd.DataFrame(results)
    summary_path = root_dir / "06_dff_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary saved:")
    print(summary_path)

    ok_df = summary_df[summary_df["status"] == "ok"]
    if not ok_df.empty:
        print("\nSuccess preview:")
        cols = [c for c in ["trial", "prefix", "n_cell", "fps", "baseline_frames", "first_stim_start_sec"] if c in ok_df.columns]
        print(ok_df[cols].head())

    print("\nDone.")


if __name__ == "__main__":
    main()