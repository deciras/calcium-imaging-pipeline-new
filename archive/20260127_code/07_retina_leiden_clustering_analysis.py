import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
from tqdm import tqdm
from matplotlib.patches import Polygon
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import shutil


# ================= 配置区域 =================
ROOT_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/Processed_TIF'

# —— 07（原有 slicing + clustering）参数 ——
PRE_STIM_SEC = 0
PRE_STIM_PLOT_SEC = 5.0
POST_STIM_SEC = 10.0
N_NEIGHBORS = 15
RESOLUTION = 0.6
MIN_DFF_THR = 0.01
TRANSIENT_THRESHOLD = 0.5
TAU = 0.5
PALETTE_NAME = 'Dark2'      # 调色板名称

# —— 新增：07 长 trace 可视化/检测参数 ——
GENERATE_LONG_TRACE_PLOTS = True      # 是否生成每个 Trial 的“整段长 trace”ROI 图
LONG_TRACE_SUBFOLDER_NAME = "ROI_Long_Traces"  # 输出子文件夹名（位于 trial 文件夹下）
LONG_TRACE_FORMAT = "pdf"             # 图片格式：pdf/pdf/svg 等
LONG_TRACE_DPI = 220

# 检测窗口：刺激结束后额外观察秒数
EXTRA_SEC_LONG_TRACE = 10.0

# Z 分数阈值（你要求新增 SIGMA_THRESH）
SIGMA_THRESH = 3.0

# 时间衰减阈值
TEMPORAL_SCORE_THRESH = 0.01

# 是否在图上标注角度文本
ANNOTATE_ANGLE_TEXT = True

# 刺激背景色
STIM_SHADE_COLOR = '#f1c40f'
STIM_SHADE_ALPHA = 0.12

# dF/F0 颜色、spike 颜色
DFF_COLOR = '#2980b9'
SPIKE_COLOR = '#7f8c8d'
RED_DOT_COLOR = '#e74c3c'
# ===========================================

warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks")


# ----------------------------------------------------------------
# 0. 通用工具函数（新增）
# ----------------------------------------------------------------

def is_valid(path: Path) -> bool:
    """过滤 macOS 系统自动生成的隐藏索引文件"""
    return not path.name.startswith("._") and not path.name.startswith(".")


def safe_float(value, default=0.0) -> float:
    """安全转换浮点数，处理 JSON 中的 null 或非数字类型"""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def robust_baseline_stats(trace: np.ndarray) -> tuple[float, float]:
    """
    计算基线 median 与鲁棒 std（MAD * 1.4826）
    """
    if trace is None or len(trace) == 0:
        return 0.0, 0.0001
    b_median = float(np.median(trace))
    mad = float(np.median(np.abs(trace - b_median)))
    b_std = mad * 1.4826
    if b_std == 0:
        b_std = 0.0001
    return b_median, b_std


def ensure_dir_clean(dir_path: Path, clean: bool = False) -> None:
    """
    创建目录；clean=True 则先删除旧目录再创建
    """
    if clean and dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# 1. 07 原有基础工具函数（严格对齐原脚本，未删减）
# ----------------------------------------------------------------

def calculate_dynamics(time_axis, trace, stim_dur):
    """计算动力学特征"""
    stim_mask = (time_axis >= 0) & (time_axis <= stim_dur)
    if len(trace[stim_mask]) == 0 or np.max(trace[stim_mask]) <= 0:
        return 0.0, 1.0, "Noise"
    peak_idx = np.argmax(trace[stim_mask])
    peak_val = trace[stim_mask][peak_idx]
    end_val = trace[stim_mask][-1]
    adaptation_ratio = end_val / peak_val if peak_val > 0 else 1.0
    d_type = "Sustained" if adaptation_ratio > TRANSIENT_THRESHOLD else "Transient"
    return time_axis[stim_mask][peak_idx], adaptation_ratio, d_type


def malus_law(theta, A, phi, B):
    # theta 以度为单位，phi 为首选相位
    return A * (np.cos(np.deg2rad(theta - phi))**2) + B


def fit_malus(angles, responses):
    """进行 Malus 拟合并返回参数与 R2"""
    try:
        # 初始参数猜测: [振幅, 峰值相位, 基值]
        p0 = [np.max(responses) - np.min(responses), angles[np.argmax(responses)], np.min(responses)]
        popt, _ = curve_fit(
            malus_law,
            angles,
            responses,
            p0=p0,
            bounds=([0, 0, -np.inf], [np.inf, 180, np.inf])
        )

        # 计算 R2
        y_pred = malus_law(angles, *popt)
        r2 = r2_score(responses, y_pred)
        return popt, r2, y_pred
    except Exception:
        return None, 0, None


def calculate_advanced_stats(time_axis, traces_by_angle, stim_dur):
    """
    计算 ANOVA 和 互相关特征
    traces_by_angle: 字典 {angle: [trace1, trace2, ...]}
    """
    # 1. ANOVA 计算
    angle_groups = []
    for ang in sorted(traces_by_angle.keys()):
        resps = [np.mean(t[(time_axis >= 0) & (time_axis <= stim_dur)]) for t in traces_by_angle[ang]]
        angle_groups.append(resps)

    f_stat, p_val = stats.f_oneway(*angle_groups) if len(angle_groups) > 1 else (0, 1)

    # 2. 互相关分析 (取所有 trace 的平均进行计算)
    # 注：这里保持你原来的“简化逻辑”，不做删改，以免影响旧功能
    all_traces = [t for group in angle_groups for t in traces_by_angle[list(traces_by_angle.keys())[0]]]
    _ = all_traces  # 保持变量存在（对齐原脚本风格），不改变逻辑
    avg_trace = np.mean(np.concatenate(list(traces_by_angle.values())), axis=0)

    stim_mask = np.where((time_axis >= 0) & (time_axis <= stim_dur), 1, 0)

    corr = np.correlate(avg_trace - np.mean(avg_trace), stim_mask - np.mean(stim_mask), mode='full')
    lags = np.arange(-len(avg_trace) + 1, len(avg_trace))

    positive_lags = lags > 0
    max_corr_idx = np.argmax(corr[positive_lags])
    best_lag_frames = lags[positive_lags][max_corr_idx]
    dt = time_axis[1] - time_axis[0]
    latency = best_lag_frames * dt
    max_corr_coeff = np.max(corr) / (np.std(avg_trace) * np.std(stim_mask) * len(avg_trace))

    return f_stat, p_val, latency, max_corr_coeff


def plot_half_polar(angles, responses, title, save_path, color='blue'):
    """绘制专业 0-180 度半圆极坐标图"""
    combined = sorted(zip(angles, responses), key=lambda x: x[0])
    p_angles = [x[0] for x in combined if 0 <= x[0] <= 180]
    p_resps = [x[1] for x in combined if 0 <= x[0] <= 180]
    if not p_angles:
        return
    rads = np.deg2rad(p_angles)
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection='polar')
    # 半极坐标范围
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # r 轴刻度位置（避免堆叠）
    ax.set_rlabel_position(135)

    # 旋转 r 轴刻度数字
    for label in ax.get_yticklabels():
        label.set_rotation(60)
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')

    # r 轴单位标注
    '''
    ax.text(
        np.deg2rad(135),
        ax.get_rmax() * 1.1,
        "dF/F0",
        rotation=60,
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )
    '''

    ax.plot(rads, p_resps, marker='o', color=color, linewidth=2)
    ax.fill(rads, p_resps, color=color, alpha=0.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title(title, pad=15, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def slice_single_trial(trial_path):
    """【不减行】完整的 JSON 与 CSV 匹配与切片逻辑"""
    j_files = [f for f in trial_path.glob("*_updated.json") if not f.name.startswith('.')]
    csv_files = [f for f in trial_path.glob("*_preprocessed.csv") if not f.name.startswith('.')]
    if not j_files or not csv_files:
        print(f"  [Skip] 缺少必要文件: {trial_path.name}")
        return None
    try:
        with open(j_files[0], 'r', encoding='utf-8') as f:
            meta = json.load(f)

        fps = meta['temporal_calibration'].get('fps', 1.0)
        s_p = meta['stimulation']['stim_parameters']
        angles = s_p['pol_angle_list']
        stim_start_time = s_p.get('stim_start_time', 0)  # 刺激开始时间（单位：秒）
        dur = s_p.get('duration_sec') or s_p.get('stim_duration_sec') or 1.0
        inter = s_p.get('interval_sec', 0)
        delay = s_p.get('initial_delay_sec', 0)

        dff = pd.read_csv(csv_files[0])
        if 'Unnamed: 0' in dff.columns:
            dff = dff.drop(columns='Unnamed: 0')

        pre_f = int(PRE_STIM_PLOT_SEC * fps)  # 切片前提前 5 秒
        total_f = int((PRE_STIM_PLOT_SEC + dur + POST_STIM_SEC) * fps)

        t_slices, t_meta = [], []
        stim_start_frame = int(stim_start_time * fps)  # 计算刺激开始的帧数

        for i, ang in enumerate(angles):
            s_start = int((delay + i * (dur + inter)) * fps) - pre_f
            s_end = s_start + total_f

            if s_start < 0:
                s_start = 0
            if s_end > len(dff):
                continue

            for roi in dff.columns:
                # 只取刺激开始后的数据
                t_slices.append(dff[roi].iloc[s_start + stim_start_frame: s_end + stim_start_frame].values)
                t_meta.append({
                    'Trial': trial_path.name, 'ROI': roi, 'Angle': ang,
                    'FPS': fps, 'Stim_Dur': dur, 'Repeat_Idx': i
                })

        return np.array(t_slices), t_meta

    except Exception as e:
        print(f"  [Error] {trial_path.name} 处理异常: {e}")
        return None


def extract_spikes_and_corr(time_axis, dff_trace, stim_dur, fps):
    """
    改用指数衰减法计算时序相关性得分（原 07 slicing 流程用）
    """
    alpha = np.exp(-1.0 / (fps * 0.5))
    spikes = np.maximum(0, dff_trace[1:] - alpha * dff_trace[:-1])
    spikes = np.insert(spikes, 0, 0)

    stim_start_idx = np.argmin(np.abs(time_axis - 0))
    window_frames = int(2.0 * fps)
    window_data = spikes[stim_start_idx: stim_start_idx + window_frames]

    if len(window_data) > 0 and np.max(window_data) > 0:
        max_idx = int(np.argmax(window_data))
        max_val = float(window_data[max_idx])
        dt = max_idx / fps
        temporal_score = max_val * np.exp(-dt * TAU)
        latency = dt
    else:
        temporal_score, latency = 0, 0

    return spikes, temporal_score, latency


# ----------------------------------------------------------------
# 2. 07 原有可视化核心：UMAP 阴影圈与居中 Tuning 布局（未删减）
# ----------------------------------------------------------------

def plot_umap_with_hulls(adata, out_path):
    """绘制带阴影圈且包含坐标轴的 UMAP，清晰展现 Cluster 范围"""
    fig, ax = plt.subplots(figsize=(8, 7))

    sc.pl.umap(
        adata,
        color='leiden',
        size=150,
        palette=PALETTE_NAME,
        legend_loc='on data',
        frameon=True,
        show=False,
        ax=ax,
        alpha=0.6
    )

    ax.set_xlabel("UMAP 1", fontsize=12, labelpad=10)
    ax.set_ylabel("UMAP 2", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=10, bottom=True, left=True)

    coords = adata.obsm['X_umap']
    clusters = adata.obs['leiden']
    colors = sns.color_palette(PALETTE_NAME, n_colors=len(clusters.unique()))

    for i, cluster in enumerate(sorted(clusters.unique())):
        cluster_coords = coords[clusters == cluster]
        if len(cluster_coords) > 2:
            try:
                hull = ConvexHull(cluster_coords)
                points = cluster_coords[hull.vertices]
                polygon = Polygon(points, alpha=0.15, color=colors[i], zorder=0)
                ax.add_patch(polygon)
            except Exception:
                pass

    plt.title("UMAP Clusters with Convex Hulls", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_visual_analysis(all_data, meta_df, cluster_df, save_dir):
    """
    执行深度可视化分析（原 07 功能），不删减
    """
    (save_dir / "ROI_Details").mkdir(exist_ok=True)
    (save_dir / "Cluster_Summaries").mkdir(exist_ok=True)

    unique_angles = sorted(meta_df['Angle'].unique())
    fps = meta_df['FPS'].iloc[0]
    time_axis = np.arange(all_data.shape[1]) / fps  # 确保从 0 开始

    stim_dur = meta_df['Stim_Dur'].iloc[0]
    advanced_results = []

    for _, row in tqdm(cluster_df.iterrows(), total=len(cluster_df), desc="ROI Analysis"):
        rid, cid = row['ROI'], row['Cluster']
        roi_meta = meta_df[meta_df['FullID'] == rid]

        ang_resps, ang_traces, anova_groups = [], [], []

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, len(unique_angles), height_ratios=[1, 1.2], hspace=0.4)

        for i, ang in enumerate(unique_angles):
            indices = roi_meta[roi_meta['Angle'] == ang].index
            raw_traces = all_data[indices]
            plot_mask = (time_axis >= 0) & (time_axis <= stim_dur + POST_STIM_SEC / 2)
            m_trace = np.max(raw_traces[:, plot_mask], axis=0)  # 注意这里的时间索引

            ang_traces.append(m_trace)

            trial_resps = [np.max(t[(time_axis >= 0) & (time_axis <= stim_dur + POST_STIM_SEC / 2)]) for t in raw_traces]
            anova_groups.append(trial_resps)
            ang_resps.append(np.mean(trial_resps))

            ax_t = fig.add_subplot(gs[0, i])
            ax_t.plot(time_axis[plot_mask], m_trace, color='black', lw=1.2)
            ax_t.axvspan(0, stim_dur, color='orange', alpha=0.1)
            ax_t.set_title(f"{int(ang)}°")
            
            # 计算所有 trace 的最小值和最大值
            y_min = np.min([np.min(m_trace) for m_trace in ang_traces])
            y_max = np.max([np.max(m_trace) for m_trace in ang_traces])

            # 给 y 轴加一些外扩的空间
            y_margin = (y_max - y_min) * 0.1  # 外扩 10%

            # 设置统一的 y 轴范围
            y_min -= y_margin
            y_max += y_margin

            # 设置所有子图的 y 轴范围一致
            for ax in fig.get_axes():
                ax.set_ylim(y_min, y_max)

            if i == 0:
                ax_t.set_ylabel("dF/F")
            sns.despine(ax=ax_t)

        f_stat, p_val = stats.f_oneway(*anova_groups) if len(anova_groups) > 1 else (0, 1)
        popt, r2, _ = fit_malus(np.array(unique_angles), np.array(ang_resps))

        avg_full_trace = np.mean(ang_traces, axis=0)
        spikes, s_score, s_lat = extract_spikes_and_corr(time_axis, avg_full_trace, stim_dur, fps)

        ax_tuning = fig.add_subplot(gs[1, :len(unique_angles)//2])
        ax_tuning.plot(unique_angles, ang_resps, 'o', color='teal', label='Data')
        if popt is not None:
            fine_x = np.linspace(0, 180, 100)
            ax_tuning.plot(fine_x, malus_law(fine_x, *popt), '--', color='red', label=f'Malus R²={r2:.3f}')

        title_str = (
            f"ANOVA p={p_val:.4f} | Malus R²={r2:.3f}\n"
            f"Exp_Score: {s_score:.3f} | Latency: {s_lat:.2f}s"
        )
        ax_tuning.set_title(title_str, fontsize=10)
        ax_tuning.set_xlabel("Angle")
        ax_tuning.set_ylabel("Mean dF/F")
        ax_tuning.tick_params(axis='x', rotation=60)
        ax_tuning.legend()
        sns.despine(ax=ax_tuning)

        ax_polar = fig.add_subplot(gs[1, len(unique_angles)//2:], projection='polar')

        # 半极坐标范围
        ax_polar.set_thetamin(0)
        ax_polar.set_thetamax(180)

        # r 轴刻度位置（避免堆叠）
        ax_polar.set_rlabel_position(135)

        # 旋转 r 轴刻度数字
        for label in ax_polar.get_yticklabels():
            label.set_rotation(60)
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')

        # r 轴单位标注
        '''
        ax_polar.text(
            np.deg2rad(135),
            ax_polar.get_rmax() * 1.1,
            "dF/F0",
            rotation=60,
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
        '''


        rads = np.deg2rad(unique_angles)
        ax_polar.plot(rads, ang_resps, marker='o', color='teal', lw=2)
        ax_polar.fill(rads, ang_resps, color='teal', alpha=0.2)
        ax_polar.set_thetamin(0)
        ax_polar.set_thetamax(180)
        ax_polar.set_title(f"ROI: {rid} (Cluster {cid})", pad=25)

        plt.savefig(save_dir / "ROI_Details" / f"Integrated_{rid}_C{cid}.pdf", bbox_inches='tight')
        plt.close()

        advanced_results.append({
            'ROI': rid,
            'Cluster': cid,
            'ANOVA_p': p_val,
            'Malus_R2': r2,
            'Temporal_Score': s_score,
            'Latency': s_lat,
            'Pref_Angle': unique_angles[np.argmax(ang_resps)]
        })

    # === 开始 Cluster 汇总分析 ===
    palette = sns.color_palette(PALETTE_NAME, n_colors=len(cluster_df['Cluster'].unique()))
    
    for i, cid in enumerate(sorted(cluster_df['Cluster'].unique())):
        c_rois = cluster_df[cluster_df['Cluster'] == cid]['ROI']
        c_meta = meta_df[meta_df['FullID'].isin(c_rois)]

        # 初始化当前 Cluster 的 Tuning 数据
        c_tuning_mean = []
        c_tuning_std = []

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, len(unique_angles), height_ratios=[1, 1.2], hspace=0.4)

        # 遍历每个角度，计算该 Cluster 的平均 trace 和响应值
        for j, ang in enumerate(unique_angles):
            roi_avg_list = []
            resp_list = []
            
            for r in c_rois:
                idx = c_meta[(c_meta['Angle'] == ang) & (c_meta['FullID'] == r)].index
                if not idx.empty:
                    # 获取该 ROI 在该角度下的所有重复 trial 的平均 trace
                    raw_traces = all_data[idx]
                    avg_trace_roi = np.mean(raw_traces, axis=0)
                    roi_avg_list.append(avg_trace_roi)
                    
                    # 计算响应值 (用于 Tuning Curve)
                    # 确保 mask 长度不超过 trace 长度
                    resp_mask = (time_axis[:len(avg_trace_roi)] >= 0) & (time_axis[:len(avg_trace_roi)] <= stim_dur)
                    resp_list.append(np.mean(avg_trace_roi[resp_mask]))

            # 计算 Cluster 在该角度下的群体均值
            if roi_avg_list:
                arr = np.array(roi_avg_list)
                m_t = np.mean(arr, axis=0)
                s_t = np.std(arr, axis=0)
                c_tuning_mean.append(np.mean(resp_list))
                c_tuning_std.append(np.std(resp_list))
                
                # 绘制 Top Row: Trace 图
                current_time_axis = time_axis[:len(m_t)]
                ax_t = fig.add_subplot(gs[0, j])
                ax_t.plot(current_time_axis, m_t, color=palette[i], lw=1.5)
                ax_t.fill_between(current_time_axis, m_t - s_t, m_t + s_t, alpha=0.2, color=palette[i])
                ax_t.axvspan(0, stim_dur, color='gray', alpha=0.1)
                ax_t.set_title(f"{int(ang)}°")
                if j == 0: ax_t.set_ylabel("Mean dF/F")
                sns.despine(ax=ax_t)
            else:
                c_tuning_mean.append(0)
                c_tuning_std.append(0)

        # 绘制 Bottom Row: Tuning Curve (对齐后的 unique_angles)
        ax_tuning = fig.add_subplot(gs[1, :len(unique_angles)//2])
        # 此时 len(unique_angles) 必然等于 len(c_tuning_mean)
        ax_tuning.errorbar(unique_angles, c_tuning_mean, yerr=c_tuning_std, fmt='o', color=palette[i], capsize=5, label='Cluster Mean')
        
        popt_c, r2_c, _ = fit_malus(np.array(unique_angles), np.array(c_tuning_mean))
        if popt_c is not None:
            fine_x = np.linspace(0, 180, 100)
            ax_tuning.plot(fine_x, malus_law(fine_x, *popt_c), '--', color='black', label=f'Malus R²={r2_c:.3f}')
        
        ax_tuning.set_title(f"Cluster {cid} Population Tuning")
        ax_tuning.set_xlabel("Angle")
        ax_tuning.legend()
        sns.despine(ax=ax_tuning)

        # 绘制 Bottom Row: Polar Plot
        ax_polar = fig.add_subplot(gs[1, len(unique_angles)//2:], projection='polar')
        ax_polar.set_thetamin(0)
        ax_polar.set_thetamax(180)
        ax_polar.plot(np.deg2rad(unique_angles), c_tuning_mean, marker='o', color=palette[i], lw=2)
        ax_polar.fill(np.deg2rad(unique_angles), c_tuning_mean, color=palette[i], alpha=0.2)
        ax_polar.set_title(f"Cluster {cid} (N={len(c_rois)})", pad=20)

        plt.savefig(save_dir / "Cluster_Summaries" / f"Cluster_{cid}_Integrated.pdf", bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(12, 7))

    unique_clusters = sorted(cluster_df['Cluster'].unique(), key=lambda x: int(x))
    palette = sns.color_palette(PALETTE_NAME, n_colors=len(unique_clusters))

    for i, cid in enumerate(unique_clusters):
        c_rois = cluster_df[cluster_df['Cluster'] == cid]['ROI']
        c_meta = meta_df[meta_df['FullID'].isin(c_rois)]

        c_tuning_mean = []
        c_tuning_sd = []

        for ang in unique_angles:
            roi_avg_list = []
            for r in c_rois:
                idx = c_meta[(c_meta['Angle'] == ang) & (c_meta['FullID'] == r)].index
                if not idx.empty:
                    trace_slice = all_data[idx]
                    resp_val = np.mean(trace_slice[:, (time_axis >= 0) & (time_axis <= stim_dur)])
                    roi_avg_list.append(resp_val)

            if roi_avg_list:
                c_tuning_mean.append(np.mean(roi_avg_list))
                c_tuning_sd.append(np.std(roi_avg_list))
            else:
                c_tuning_mean.append(0)
                c_tuning_sd.append(0)

        means = np.array(c_tuning_mean)
        sds = np.array(c_tuning_sd)
        angles_arr = np.array(unique_angles)

        plt.plot(angles_arr, means, marker='o', linewidth=2, color=palette[i], label=f'Cluster {cid}', zorder=3)
        plt.fill_between(angles_arr, means - sds, means + sds, color=palette[i], alpha=0.15, zorder=2)

    plt.title("Population Tuning Curves Comparison (Mean ± SD)", fontsize=14, pad=15)
    plt.xlabel("Angle (Degree)", fontsize=12)
    plt.ylabel("Mean dF/F Response", fontsize=12)
    plt.xticks(unique_angles, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_dir / "Cluster_Summaries" / "All_Clusters_Tuning_Comparison_with_SD.pdf", dpi=300)
    plt.close()

    return pd.DataFrame(advanced_results)


# ----------------------------------------------------------------
# 3. 07 原有主流程逻辑（聚类分析，未删减）
# ----------------------------------------------------------------

def process_day(day_dir):
    print(f"\n>>>>>>> 开始处理日期目录: {day_dir.name} <<<<<<<")
    trial_paths = sorted([t for t in day_dir.glob("*") if t.is_dir() and not t.name.startswith('.')])

    all_slices, all_meta = [], []
    for tp in tqdm(trial_paths, desc="数据加载中"):
        res = slice_single_trial(tp)
        if res:
            # 使用切片后的数据，确保只使用刺激开始后的部分
            all_slices.append(res[0])  # 切片数据（从刺激后开始）
            all_meta.extend(res[1])  # 元数据

    if not all_slices:
        print(f"  [Abort] {day_dir.name} 目录下未找到有效 Trial 数据。")
        return

    min_len = min([s.shape[1] for s in all_slices])
    all_data = np.vstack([s[:, :min_len] for s in all_slices])  # 拼接所有数据
    meta_df = pd.DataFrame(all_meta)
    meta_df['FullID'] = meta_df['Trial'] + "_" + meta_df['ROI']
    unique_angles = sorted(meta_df['Angle'].unique())


    agg_features, valid_ids = [], []
    for rid in meta_df['FullID'].unique():
        roi_meta = meta_df[meta_df['FullID'] == rid]
        if len(roi_meta['Angle'].unique()) < len(unique_angles):
            continue

        wave = []
        for ang in unique_angles:
            m_wave = np.mean(all_data[roi_meta[roi_meta['Angle'] == ang].index], axis=0)
            wave.extend(m_wave)

        if np.std(wave) > MIN_DFF_THR:
            agg_features.append(wave)
            valid_ids.append(rid)

    if not valid_ids:
        print("  [Abort] 该日期下无 ROI 通过标准过滤。")
        return

    X = np.array(agg_features)
    out_dir = day_dir / "Slicing_Analysis"
    out_dir.mkdir(exist_ok=True)

    X_norm = StandardScaler().fit_transform(X)
    adata = sc.AnnData(X=X_norm.astype(np.float32))

    sc.pp.neighbors(adata, n_neighbors=min(N_NEIGHBORS, len(valid_ids) - 1), use_rep='X')
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, resolution=RESOLUTION)

    plot_umap_with_hulls(adata, out_dir / "UMAP_Cluster_Hulls.pdf")

    cluster_df = pd.DataFrame({'ROI': valid_ids, 'Cluster': adata.obs['leiden'].values})
    adv_metrics = run_visual_analysis(all_data, meta_df, cluster_df, out_dir)

    adv_metrics.to_csv(out_dir / "Final_Dynamics_Metrics.csv", index=False)
    print(f"--- {day_dir.name} 处理成功完成 ---")


# ----------------------------------------------------------------
# 4. 新增：07 长 trace（每个 Trial 每个 ROI 一张图，双 y 轴 + 红点）
# ----------------------------------------------------------------

def get_long_trace_meta(json_path: Path) -> dict:
    """
    读取 trial 的 *_updated.json，返回 fps、angles、stim 参数等。
    兼容 05/06/07 中出现的两套 fps 字段写法。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    fps = None
    if isinstance(meta, dict):
        fps = meta.get('temporal_calibration', {}).get('fps', None)
        if fps is None:
            fps = meta.get('fps', None)
    fps = safe_float(fps, 1.0)

    stim = meta.get('stimulation', {}) if isinstance(meta, dict) else {}
    params = stim.get('stim_parameters', {}) if isinstance(stim, dict) else {}
    stim_type = stim.get('stim_type', 'nostim') if isinstance(stim, dict) else 'nostim'

    angles = params.get('pol_angle_list', [])
    if angles is None:
        angles = []
    try:
        angles = [float(a) for a in angles]
    except Exception:
        angles = []

    delay = safe_float(params.get('initial_delay_sec', 0.0), 0.0)
    duration = safe_float(params.get('duration_sec', None), None)
    if duration is None:
        duration = safe_float(params.get('stim_duration_sec', 1.0), 1.0)
    interval = safe_float(params.get('interval_sec', 0.0), 0.0)

    n_stim = params.get('number_of_stim', None)
    if n_stim is None:
        n_stim = len(angles)
    n_stim = int(safe_float(n_stim, 0))

    return {
        "fps": fps,
        "stim_type": stim_type,
        "params": params,
        "angles": angles,
        "initial_delay_sec": delay,
        "duration_sec": float(duration),
        "interval_sec": float(interval),
        "number_of_stim": int(n_stim),
    }


def build_stim_windows_from_meta(meta: dict) -> list[tuple[int, int]]:
    """
    基于 long trace 的真实时间轴（从 0 秒开始到 trial 结束）构建刺激窗（帧索引）
    """
    fps = float(meta["fps"])
    delay = float(meta["initial_delay_sec"])
    dur = float(meta["duration_sec"])
    interval = float(meta["interval_sec"])
    n_stim = int(meta["number_of_stim"])

    windows: list[tuple[int, int]] = []
    for i in range(n_stim):
        s = int((delay + i * interval) * fps)
        e = int((delay + i * interval + dur) * fps)
        windows.append((s, e))
    return windows


def infer_spikes_from_dff(trace: np.ndarray, fps: float, spike_tau_for_alpha: float = 1) -> np.ndarray:
    """
    如果没有 *_spikes.csv，则用指数差分法临时推一个 spikes（不影响 06 生成的 spikes 文件逻辑）
    """
    if trace is None or len(trace) == 0:
        return np.array([], dtype=float)
    alpha = float(np.exp(-1.0 / (fps * spike_tau_for_alpha))) if fps > 0 else 0.0
    spikes = np.maximum(0, trace[1:] - alpha * trace[:-1])
    spikes = np.insert(spikes, 0, 0.0)
    return spikes.astype(float)


def detect_stim_locked_responses_long_trace(
    trace: np.ndarray,
    spikes: np.ndarray,
    stim_windows: list[tuple[int, int]],
    fps: float,
    tau: float,
    sigma_thresh: float,
    extra_sec: float
) -> dict:
    """
    对每个刺激窗口：
    1) 在 [stim_start, stim_end + extra] 中找 spikes 最大值点，得到 dt
    2) temporal_score = spike_amp * exp(-dt / tau)   （你要求的 e^(-dt/TAU)）
    3) 同时在同窗口计算 dF/F 的峰值 z-score，要求 z >= SIGMA_THRESH
    返回：
      - valid_resp_frames: list[int]  (用于红点标注在 dF/F 上)
      - per_stim_details: list[dict]  (保留细节便于你后续导出)
    """
    if trace is None or len(trace) == 0:
        return {"valid_resp_frames": [], "per_stim_details": []}

    n = len(trace)
    if spikes is None or len(spikes) != n:
        spikes = np.zeros(n, dtype=float)

    b_median, b_std = robust_baseline_stats(trace)

    valid_resp_frames: list[int] = []
    per_stim_details: list[dict] = []

    extra_frames = int(extra_sec * fps) if fps > 0 else 0
    tau = float(tau) if tau is not None else 0.5
    if tau <= 0:
        tau = 0.5

    for idx, (s, e) in enumerate(stim_windows):
        s = int(max(0, s))
        e = int(max(s, e))
        if s >= n:
            per_stim_details.append({
                "stim_index": idx,
                "stim_start": s,
                "stim_end": e,
                "search_end": s,
                "spike_amp": 0.0,
                "dt_sec": 0.0,
                "temporal_score": 0.0,
                "dff_peak": float(b_median),
                "z_score": 0.0,
                "is_valid": 0
            })
            continue

        search_end = int(min(n, e + extra_frames))
        if search_end <= s:
            search_end = min(n, s + 1)

        # —— spikes 最大值定位（用于 dt + 衰减得分）——
        spike_win = spikes[s:search_end]
        if len(spike_win) > 0:
            spike_rel = int(np.argmax(spike_win))
            spike_amp = float(spike_win[spike_rel])
            dt_sec = float(spike_rel / fps) if fps > 0 else 0.0
            temporal_score = spike_amp * float(np.exp(-dt_sec / tau))
            resp_frame_spike = int(s + spike_rel)
        else:
            spike_amp, dt_sec, temporal_score, resp_frame_spike = 0.0, 0.0, 0.0, int(s)

        # —— dF/F 峰值与 z-score ——（用于 SIGMA_THRESH 筛选）
        dff_win = trace[s:search_end]
        if len(dff_win) > 0:
            dff_peak = float(np.max(dff_win))
            z_score = float((dff_peak - b_median) / b_std)
        else:
            dff_peak, z_score = float(b_median), 0.0

        is_valid = 1 if (z_score >= sigma_thresh and spike_amp > 0 and temporal_score > TEMPORAL_SCORE_THRESH) else 0

        # 红点标注：用“spike 最大点”的时间（更贴近刺激锁定的快速事件），标在 dF/F 上
        if is_valid == 1 and 0 <= resp_frame_spike < n:
            valid_resp_frames.append(resp_frame_spike)

        per_stim_details.append({
            "stim_index": idx,
            "stim_start": int(s),
            "stim_end": int(e),
            "search_end": int(search_end),
            "spike_amp": float(spike_amp),
            "dt_sec": float(dt_sec),
            "temporal_score": float(temporal_score),
            "dff_peak": float(dff_peak),
            "z_score": float(z_score),
            "is_valid": int(is_valid)
        })

    return {
        "valid_resp_frames": valid_resp_frames,
        "per_stim_details": per_stim_details,
        "baseline_median": float(b_median),
        "baseline_std": float(b_std)
    }


def plot_long_trace_dual_axis(
    roi_name: str,
    trace: np.ndarray,
    spikes: np.ndarray,
    fps: float,
    stim_windows: list[tuple[int, int]],
    angles: list[float],
    valid_resp_frames: list[int],
    baseline_median: float,
    save_path: Path
) -> None:
    """
    画一张整段长 trace：
      - 左轴：dF/F0
      - 右轴：spikes（vlines）
      - 刺激窗底色
      - 红点标注 valid_resp_frames（标在 dF/F0 上）
    """
    if trace is None or len(trace) == 0:
        return
    n = len(trace)
    if spikes is None or len(spikes) != n:
        spikes = np.zeros(n, dtype=float)

    time_axis = np.arange(n) / fps if fps > 0 else np.arange(n)

    fig, ax1 = plt.subplots(figsize=(18, 6))

    # dF/F0（左轴）
    ax1.plot(time_axis, trace, color=DFF_COLOR, lw=1.4, alpha=0.95, label='dF/F0', zorder=2)
    ax1.fill_between(time_axis, trace, baseline_median, color=DFF_COLOR, alpha=0.10, zorder=1)
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('dF/F0', fontsize=12, fontweight='bold', color=DFF_COLOR)
    ax1.tick_params(axis='y', labelcolor=DFF_COLOR)
    ax1.grid(True, linestyle='--', alpha=0.25)

    # spikes（右轴）
    ax2 = ax1.twinx()
    ax2.vlines(time_axis, [0], spikes, colors=SPIKE_COLOR, alpha=0.35, lw=0.8, label='Spikes', zorder=3)
    ax2.set_ylabel('Spike amplitude', fontsize=12, fontweight='bold', color=SPIKE_COLOR)
    ax2.tick_params(axis='y', labelcolor=SPIKE_COLOR)

    # 调整右轴范围
    smax = float(np.max(spikes)) if len(spikes) > 0 else 0.0
    if smax > 0:
        ax2.set_ylim(0, smax * 3.0)
    else:
        ax2.set_ylim(0, 1.0)

    # 刺激背景 + 角度文本
    y_min, y_max = ax1.get_ylim()
    for j, (s, e) in enumerate(stim_windows):
        ax1.axvspan(s / fps, e / fps, color=STIM_SHADE_COLOR, alpha=STIM_SHADE_ALPHA, zorder=0)
        if ANNOTATE_ANGLE_TEXT and j < len(angles):
            ax1.text(
                (s + e) / (2 * fps),
                y_max * 0.95,
                f"{angles[j]}°",
                ha='center',
                va='top',
                color='#c0392b',
                fontsize=10,
                fontweight='bold'
            )

    # 红点标注（标在 dF/F0 上）
    if valid_resp_frames:
        resp_frames = [int(f) for f in valid_resp_frames if 0 <= int(f) < n]
        if resp_frames:
            resp_times = [time_axis[f] for f in resp_frames]
            resp_vals = [trace[f] for f in resp_frames]
            ax1.scatter(
                resp_times,
                resp_vals,
                s=35,
                color=RED_DOT_COLOR,
                edgecolor='white',
                linewidth=0.6,
                alpha=0.95,
                label='Stim-locked response',
                zorder=5
            )

    title = f"ROI: {roi_name} | Long Trace (dF/F0 + Spikes) | SIGMA_THRESH={SIGMA_THRESH}, TAU={TAU}"
    ax1.set_title(title, fontsize=14, pad=10)

    # 合并图例
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper left', frameon=True)

    sns.despine(ax=ax1, top=True, right=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=LONG_TRACE_DPI)
    plt.close()


def find_trial_input_files(trial_path: Path) -> dict:
    """
    在 trial 文件夹中定位：
      - *_updated.json
      - *_preprocessed.csv
      - *_spikes.csv（可选，优先）
    """
    json_files = sorted([p for p in trial_path.glob("*_updated.json") if is_valid(p)])
    pre_files = sorted([p for p in trial_path.glob("*_preprocessed.csv") if is_valid(p)])

    spikes_files = sorted([p for p in trial_path.glob("*_spikes.csv") if is_valid(p)])
    spikes_file = spikes_files[0] if spikes_files else None

    if not json_files or not pre_files:
        return {"ok": False, "json": None, "pre": None, "spikes": None}

    # 通常一对一，但这里取第一个（与 05/06/07 的默认产物一致）
    return {
        "ok": True,
        "json": json_files[0],
        "pre": pre_files[0],
        "spikes": spikes_file
    }


def generate_long_trace_plots_for_trial(trial_path: Path) -> None:
    """
    生成该 trial 下所有 ROI 的长 trace 图：
      trial_path / LONG_TRACE_SUBFOLDER_NAME / {ROI}.{LONG_TRACE_FORMAT}
    同时输出一个 detection_summary.csv，记录每个 ROI 每个刺激窗的检测指标。
    """
    files = find_trial_input_files(trial_path)
    if not files["ok"]:
        return

    meta = get_long_trace_meta(files["json"])
    fps = float(meta["fps"])
    angles = meta.get("angles", [])
    stim_windows = build_stim_windows_from_meta(meta)

    # 输出目录：trial 文件夹内的子文件夹
    out_dir = trial_path / LONG_TRACE_SUBFOLDER_NAME
    ensure_dir_clean(out_dir, clean=False)

    # 读取 dF/F0
    dff = pd.read_csv(files["pre"])
    if 'Unnamed: 0' in dff.columns:
        dff = dff.drop(columns='Unnamed: 0')

    # 读取 spikes（若无则现场推）
    spikes_df = None
    if files["spikes"] is not None and files["spikes"].exists():
        spikes_df = pd.read_csv(files["spikes"])
        if 'Unnamed: 0' in spikes_df.columns:
            spikes_df = spikes_df.drop(columns='Unnamed: 0')

    detection_rows = []

    for roi_name in tqdm(dff.columns, desc=f"LongTrace {trial_path.name}", leave=False):
        trace = dff[roi_name].values.astype(float)

        if spikes_df is not None and roi_name in spikes_df.columns:
            spikes = spikes_df[roi_name].values.astype(float)
        else:
            spikes = infer_spikes_from_dff(trace, fps=fps, spike_tau_for_alpha=1)

        det = detect_stim_locked_responses_long_trace(
            trace=trace,
            spikes=spikes,
            stim_windows=stim_windows,
            fps=fps,
            tau=TAU,
            sigma_thresh=SIGMA_THRESH,
            extra_sec=EXTRA_SEC_LONG_TRACE
        )

        # 画图
        safe_name = "".join([c for c in str(roi_name) if c.isalnum() or c in (' ', '_', '-', '.', '(', ')')]).strip()
        save_path = out_dir / f"{safe_name}.{LONG_TRACE_FORMAT}"

        plot_long_trace_dual_axis(
            roi_name=str(roi_name),
            trace=trace,
            spikes=spikes,
            fps=fps,
            stim_windows=stim_windows,
            angles=angles,
            valid_resp_frames=det["valid_resp_frames"],
            baseline_median=det["baseline_median"],
            save_path=save_path
        )

        # 记录检测详情（每个 ROI * 每个 stim）
        for d in det["per_stim_details"]:
            detection_rows.append({
                "Trial": trial_path.name,
                "ROI": str(roi_name),
                "stim_index": d["stim_index"],
                "stim_start_frame": d["stim_start"],
                "stim_end_frame": d["stim_end"],
                "search_end_frame": d["search_end"],
                "spike_amp": d["spike_amp"],
                "dt_sec": d["dt_sec"],
                "temporal_score": d["temporal_score"],
                "dff_peak": d["dff_peak"],
                "z_score": d["z_score"],
                "is_valid": d["is_valid"]
            })

    # 输出检测表
    if detection_rows:
        det_df = pd.DataFrame(detection_rows)
        det_df.to_csv(out_dir / "detection_summary.csv", index=False)


def generate_long_trace_plots_for_day(day_dir: Path) -> None:
    """
    对某个日期目录下所有 Trial 执行长 trace 绘图
    """
    trial_paths = sorted([t for t in day_dir.glob("*") if t.is_dir() and is_valid(t)])
    for tp in trial_paths:
        try:
            generate_long_trace_plots_for_trial(tp)
        except Exception as e:
            print(f"  [LongTrace Error] {tp.name}: {e}")


# ----------------------------------------------------------------
# 5. 入口
# ----------------------------------------------------------------

if __name__ == "__main__":
    day_folders = sorted([d for d in Path(ROOT_DIR).iterdir() if d.is_dir() and "Processed_TIF_" in d.name])

    for folder in day_folders:
        # 原 07 功能：切片聚类分析
        process_day(folder)

        # 新增：每个 trial 的整段长 trace ROI 图
        if GENERATE_LONG_TRACE_PLOTS:
            generate_long_trace_plots_for_day(folder)
