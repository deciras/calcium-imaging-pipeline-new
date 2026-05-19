import os
import json
import re
import glob
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 配置
# ============================================================
DATA_PATH = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected"

# stimulus 前 baseline 窗长（秒）
BASELINE_SEC = 1.0

# 判定 responsive 的 zscore 阈值
Z_THRESHOLD = 2.0

# 是否保存 heatmap
SAVE_HEATMAP = True

# 是否保存每个 ROI 单独的 trace 图
SAVE_TRACE_WITH_STIM = True

# 每个 trial 内 trace 图输出文件夹名
TRACE_FOLDER_NAME = "Trace_With_Stim"

# heatmap 最多显示多少个 ROI；None = 全部显示
HEATMAP_MAX_ROIS = None

# 是否自动清理旧输出
CLEAN_TRIAL_OUTPUTS = True


# ============================================================
# 工具函数
# ============================================================
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", s)]


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_fps_from_metadata(meta_path: Path) -> float:
    meta = load_json(meta_path)
    fps = meta.get("temporal_calibration", {}).get("fps", None)
    if fps is None:
        raise ValueError(f"fps not found in metadata: {meta_path}")
    return float(fps)


def infer_prefix_from_preprocessed(preprocessed_path: Path) -> str:
    name = preprocessed_path.name
    if name.endswith("_preprocessed.csv"):
        return name.replace("_preprocessed.csv", "")
    return preprocessed_path.stem


def clean_trial_outputs(trial_folder: Path):
    """
    清理本步骤在单个 trial 目录下生成的所有输出
    """
    if not CLEAN_TRIAL_OUTPUTS:
        return 0

    patterns = [
        "*_event_response.csv",
        "*_response_summary.csv",
        "*_heatmap.png",
    ]

    removed = 0
    for pattern in patterns:
        for path in glob.glob(str(trial_folder / pattern)):
            if os.path.isfile(path):
                os.remove(path)
                print(f"  Removed old file -> {path}")
                removed += 1

    trace_dir = trial_folder / TRACE_FOLDER_NAME
    if trace_dir.exists() and trace_dir.is_dir():
        shutil.rmtree(trace_dir)
        print(f"  Removed old folder -> {trace_dir}")
        removed += 1

    return removed


def load_stim_events_for_trial(trial_folder: Path, prefix: str):
    """
    优先读取 trial 内的:
      prefix_stim_events.csv
    如果没有，再 fallback 到 day 层:
      stim_events.csv
    """
    trial_folder = Path(trial_folder)

    trial_stim_path = trial_folder / f"{prefix}_stim_events.csv"
    if trial_stim_path.exists():
        return pd.read_csv(trial_stim_path), trial_stim_path

    day_folder = trial_folder.parent
    day_stim_path = day_folder / "stim_events.csv"
    if day_stim_path.exists():
        return pd.read_csv(day_stim_path), day_stim_path

    return None, None


def subset_stim_events_for_trial(stim_df: pd.DataFrame, trial_folder: Path, prefix: str, stim_path: Path):
    """
    如果读到的是 trial-level stim_events，直接返回全表。
    如果读到的是 day-level stim_events，则尝试按 trialID 等列筛当前 trial。
    """
    if stim_df is None or stim_df.empty:
        return None

    if stim_path is not None and stim_path.name.endswith("_stim_events.csv") and stim_path.parent == trial_folder:
        return stim_df.copy()

    candidate_cols = [
        "trialID",
        "trial_folder",
        "trial",
        "exp_id",
        "file_name",
        "recording_name",
        "name",
    ]

    trial_name = trial_folder.name
    has_any_id_col = any(col in stim_df.columns for col in candidate_cols)

    if not has_any_id_col:
        return stim_df.copy()

    for col in candidate_cols:
        if col in stim_df.columns:
            sub1 = stim_df[stim_df[col].astype(str) == str(trial_name)]
            if len(sub1) > 0:
                return sub1.copy()

            sub2 = stim_df[stim_df[col].astype(str) == str(prefix)]
            if len(sub2) > 0:
                return sub2.copy()

    return None


def validate_stim_events(stim_df: pd.DataFrame):
    required = ["start_time_sec", "end_time_sec"]
    for c in required:
        if c not in stim_df.columns:
            raise ValueError(f"stim_events.csv missing required column: {c}")

    stim_df = stim_df.copy()
    stim_df["start_time_sec"] = pd.to_numeric(stim_df["start_time_sec"], errors="coerce")
    stim_df["end_time_sec"] = pd.to_numeric(stim_df["end_time_sec"], errors="coerce")

    stim_df = stim_df.dropna(subset=["start_time_sec", "end_time_sec"])
    stim_df = stim_df[stim_df["end_time_sec"] > stim_df["start_time_sec"]]
    stim_df = stim_df.sort_values("start_time_sec").reset_index(drop=True)

    return stim_df


def compute_zscore_from_baseline(stim_signal: np.ndarray, baseline: np.ndarray):
    mu = float(np.mean(baseline))
    sd = float(np.std(baseline))

    if sd <= 0:
        return 0.0

    peak = float(np.max(stim_signal))
    return (peak - mu) / sd


def get_baseline_window(start_frame: int, fps: float):
    baseline_len = max(1, int(round(BASELINE_SEC * fps)))
    base_start = max(0, start_frame - baseline_len)
    base_end = start_frame
    return base_start, base_end


def build_event_results(dff_df: pd.DataFrame, stim_df: pd.DataFrame, fps: float):
    results = []
    n_frames = len(dff_df)

    for stim_id, row in stim_df.iterrows():
        start_frame = int(round(float(row["start_time_sec"]) * fps))
        end_frame = int(round(float(row["end_time_sec"]) * fps))

        start_frame = max(0, min(start_frame, n_frames))
        end_frame = max(0, min(end_frame, n_frames))

        if end_frame <= start_frame:
            continue

        base_start, base_end = get_baseline_window(start_frame, fps)

        extra_cols = {}
        for c in stim_df.columns:
            if c not in ["start_time_sec", "end_time_sec"]:
                extra_cols[c] = row[c]

        for roi in dff_df.columns:
            trace = dff_df[roi].values

            baseline = trace[base_start:base_end]
            stim_signal = trace[start_frame:end_frame]

            if len(baseline) == 0 or len(stim_signal) == 0:
                continue

            peak = float(np.max(stim_signal))
            mean_resp = float(np.mean(stim_signal))
            baseline_mean = float(np.mean(baseline))
            z = float(compute_zscore_from_baseline(stim_signal, baseline))
            responsive = int(z > Z_THRESHOLD)

            out = {
                "roi": roi,
                "stim_id": int(stim_id),
                "start_time_sec": float(row["start_time_sec"]),
                "end_time_sec": float(row["end_time_sec"]),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "baseline_start_frame": int(base_start),
                "baseline_end_frame": int(base_end),
                "baseline_mean": baseline_mean,
                "peak_dff": peak,
                "mean_dff": mean_resp,
                "zscore": z,
                "responsive": responsive,
            }
            out.update(extra_cols)
            results.append(out)

    return pd.DataFrame(results)


def build_response_summary(event_df: pd.DataFrame):
    if event_df is None or event_df.empty:
        return pd.DataFrame()

    summary = event_df.groupby("roi").agg(
        mean_peak=("peak_dff", "mean"),
        max_peak=("peak_dff", "max"),
        mean_dff=("mean_dff", "mean"),
        max_zscore=("zscore", "max"),
        responsive_n=("responsive", "sum"),
        n_stim=("responsive", "count"),
    ).reset_index()

    summary["responsive_fraction"] = summary["responsive_n"] / summary["n_stim"]
    return summary


def save_heatmap(dff_df: pd.DataFrame, stim_df: pd.DataFrame | None, fps: float, out_path: Path):
    if dff_df.empty:
        return

    if HEATMAP_MAX_ROIS is not None:
        dff_plot = dff_df.iloc[:, :HEATMAP_MAX_ROIS]
    else:
        dff_plot = dff_df

    mat = dff_plot.values.T  # ROI x frame

    plt.figure(figsize=(10, 6))
    plt.imshow(mat, aspect="auto", interpolation="nearest")

    # 只有有 stim 时才画
    if stim_df is not None and not stim_df.empty:
        for _, row in stim_df.iterrows():
            s = int(round(float(row["start_time_sec"]) * fps))
            e = int(round(float(row["end_time_sec"]) * fps))
            plt.axvspan(s, e, alpha=0.2)

    plt.xlabel("Frame")
    plt.ylabel("ROI")
    plt.title("dF/F heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_single_roi_trace(trace: np.ndarray, roi_name: str, prefix: str,
                          stim_df: pd.DataFrame | None, fps: float, out_path: Path):
    """
    单个 ROI 画一张 trace 图。
    如果没有 stim，则只画 trace，不画刺激区间/线。
    """
    n_frames = len(trace)
    time_axis = np.arange(n_frames) / fps

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, trace, linewidth=0.9)

    if stim_df is not None and not stim_df.empty:
        for _, row in stim_df.iterrows():
            s = float(row["start_time_sec"])
            e = float(row["end_time_sec"])
            plt.axvspan(s, e, alpha=0.15)
            plt.axvline(s, linestyle="--", linewidth=0.8)
            plt.axvline(e, linestyle="--", linewidth=0.8)

    plt.xlabel("Time (sec)")
    plt.ylabel("dF/F")
    plt.title(f"{prefix} | ROI {roi_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_all_roi_traces(dff_df: pd.DataFrame, prefix: str, stim_df: pd.DataFrame | None,
                        fps: float, trace_dir: Path):
    """
    为每个 ROI 单独保存一张 trace 图
    """
    trace_dir.mkdir(parents=True, exist_ok=True)

    for roi in dff_df.columns:
        trace = dff_df[roi].values
        out_path = trace_dir / f"{roi}_trace_with_stim.png"
        save_single_roi_trace(trace, roi, prefix, stim_df, fps, out_path)


# ============================================================
# 单个 trial 分析
# ============================================================
def analyze_trial(preprocessed_path: Path):
    trial_folder = preprocessed_path.parent
    prefix = infer_prefix_from_preprocessed(preprocessed_path)

    clean_trial_outputs(trial_folder)

    meta_path = trial_folder / f"{prefix}_metadata.json"
    if not meta_path.exists():
        meta_files = sorted(trial_folder.glob("*_metadata.json"), key=lambda p: natural_key(p.name))
        if len(meta_files) == 0:
            print(f"[SKIP] No metadata json found: {trial_folder}")
            return
        meta_path = meta_files[0]

    try:
        fps = get_fps_from_metadata(meta_path)
    except Exception as e:
        print(f"[SKIP] Failed to read fps from metadata: {meta_path}")
        print(f"       {e}")
        return

    try:
        dff_df = pd.read_csv(preprocessed_path)
    except Exception as e:
        print(f"[SKIP] Failed to read preprocessed csv: {preprocessed_path}")
        print(f"       {e}")
        return

    if dff_df.empty:
        print(f"[SKIP] Empty dff csv: {preprocessed_path}")
        return

    stim_df_all, stim_path = load_stim_events_for_trial(trial_folder, prefix)

    # 允许没有 stim
    stim_df = None
    if stim_df_all is not None:
        stim_df = subset_stim_events_for_trial(stim_df_all, trial_folder, prefix, stim_path)

        if stim_df is not None and not stim_df.empty:
            try:
                stim_df = validate_stim_events(stim_df)
            except Exception as e:
                print(f"[WARN] Invalid stim_events.csv: {stim_path}")
                print(f"       {e}")
                stim_df = None

            if stim_df is not None and stim_df.empty:
                stim_df = None

    print(f"[INFO] Processing trial: {trial_folder.name}")
    if stim_df is not None:
        print(f"[INFO] Using stim_events: {stim_path}")
        print(f"[INFO] Matched {len(stim_df)} stimulus event(s)")
    else:
        print(f"[INFO] No valid stimulus events found. Will plot traces without stim overlay.")

    try:
        # 如果有 stim，生成 event_df 和 summary_df
        if stim_df is not None:
            event_df = build_event_results(dff_df, stim_df, fps)
            if event_df.empty:
                print(f"[WARN] No valid event response rows generated: {trial_folder}")
                event_df = pd.DataFrame()
                summary_df = pd.DataFrame()
            else:
                summary_df = build_response_summary(event_df)
        else:
            event_df = pd.DataFrame()
            summary_df = pd.DataFrame()

        event_out = trial_folder / f"{prefix}_event_response.csv"
        summary_out = trial_folder / f"{prefix}_response_summary.csv"
        heatmap_out = trial_folder / f"{prefix}_heatmap.png"
        trace_dir = trial_folder / TRACE_FOLDER_NAME

        # 没有 stim 时也允许保存空表，保持流程一致
        event_df.to_csv(event_out, index=False)
        summary_df.to_csv(summary_out, index=False)

        if SAVE_HEATMAP:
            save_heatmap(dff_df, stim_df, fps, heatmap_out)

        if SAVE_TRACE_WITH_STIM:
            save_all_roi_traces(dff_df, prefix, stim_df, fps, trace_dir)

        print(f"[OK] Saved:")
        print(f"     {event_out}")
        print(f"     {summary_out}")
        if SAVE_HEATMAP:
            print(f"     {heatmap_out}")
        if SAVE_TRACE_WITH_STIM:
            print(f"     {trace_dir}")

    except Exception as e:
        print(f"[ERROR] Failed on trial: {trial_folder}")
        print(e)
        print(traceback.format_exc())


# ============================================================
# 主函数
# ============================================================
def main():
    root = Path(DATA_PATH).resolve()

    preprocessed_files = sorted(
        root.rglob("*_preprocessed.csv"),
        key=lambda p: natural_key(str(p))
    )

    print(f"Found {len(preprocessed_files)} trials")

    if len(preprocessed_files) == 0:
        return

    for preprocessed_path in preprocessed_files:
        analyze_trial(preprocessed_path)


if __name__ == "__main__":
    main()