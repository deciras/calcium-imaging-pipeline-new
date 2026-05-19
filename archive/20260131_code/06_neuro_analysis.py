import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 根目录路径
ROOT_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/Processed_TIF'
# 是否保存可视化图表
SAVE_PLOTS = True
# Z-Score 响应识别阈值
SIGMA_THRESH = 3.0          
# 刺激后观察窗口长度（秒）
EXTRA_SEC = 10              
# 响应可靠性阈值
REL_THRESH = 0.05            
          

# 设置绘图风格
sns.set_theme(style="ticks")
# 设置字体以支持标准学术绘图
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ===========================================

def is_valid(path):
    """过滤 macOS 系统自动生成的隐藏索引文件"""
    return not path.name.startswith("._")

def safe_float(value, default=0.0):
    """安全转换浮点数，处理 JSON 中的 null 或非数字类型"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_meta(json_path):
    """从元数据中提取参数：FPS、刺激类型、参数、角度列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    fps = meta.get('temporal_calibration', {}).get('fps', 0.8333)
    stim_type = meta.get('stimulation', {}).get('stim_type', 'nostim')
    params = meta.get('stimulation', {}).get('stim_parameters', {})
    angles = params.get('pol_angle_list', []) 
    return fps, stim_type, params, angles

def difference_deconvolution(y):
    """差分法识别脉冲近似"""
    if len(y) < 2: 
        return np.zeros_like(y)
    diff = np.diff(y, prepend=y[0])
    return np.maximum(0, diff)

def analyze_trace(trace, fps, stim_type, params):
    """
    核心分析逻辑：
    1. 计算差分信号 (原本的差分法)。
    2. 判定响应可靠性 (基于 dF/F 信号的 Z-Score)。
    """
    # 使用原本的差分法识别脉冲近似
    inferred_s = difference_deconvolution(trace)
    
    # 安全获取刺激参数
    delay = safe_float(params.get('initial_delay_sec'), 0.0)
    duration = safe_float(params.get('duration_sec'), 1.0)
    interval = safe_float(params.get('interval_sec'), 0.0)
    n_stim = int(safe_float(params.get('number_of_stim'), 0))
    
    # 计算每个刺激周期的起始和结束帧数
    stim_windows = [(int((delay + i * interval) * fps), 
                     int((delay + i * interval + duration) * fps)) for i in range(n_stim)]
    
    # 计算基线与噪声标准差 (用于响应判定)
    b_median = np.median(trace)
    b_std = np.median(np.abs(trace - b_median)) * 1.4826
    if b_std == 0: b_std = 0.0001
    
    stim_responses = []
    trial_peaks = []

    # 遍历每个刺激周期进行分析
    for i, (s, e) in enumerate(stim_windows):
        if s < len(trace):
            # 设定观察窗口：从刺激开始到刺激结束后 EXTRA_SEC 秒
            search_end = min(len(trace), e + int(EXTRA_SEC * fps))
            window_trace = trace[s:search_end]
            
            # 记录该窗口内的最大 dF/F 峰值
            local_peak = np.max(window_trace) if len(window_trace) > 0 else b_median
            trial_peaks.append(local_peak)
            
            # 响应判定：峰值偏离基线的程度是否超过 SIGMA_THRESH 倍噪声
            is_responsive = 1 if (local_peak - b_median) > SIGMA_THRESH * b_std else 0
            stim_responses.append(is_responsive)
        else:
            # 如果刺激起始点超出数据范围，记为无响应
            stim_responses.append(0)
    
    # 汇总统计指标
    if trial_peaks and stim_type != 'nostim' and n_stim > 0:
        peak_avg = np.mean(trial_peaks)
        rel = sum(stim_responses) / n_stim
        status = "Stimulated" if rel > REL_THRESH else "Control"
        z_score = (peak_avg - b_median) / b_std
    else:
        peak_avg, rel, z_score, status = 0, 0, 0, "Control"

    return {
        'trace': trace, 
        'spikes': inferred_s, 
        'valid_erg_indices': [], # 已根据要求清空，不再记录红点位置
        'stim_windows': stim_windows, 
        'stim_responses': stim_responses,
        'Peak_dFF': peak_avg, 
        'Z_Score': z_score, 
        'Reliability': rel, 
        'Status': status,
        'b_median': b_median
    }

def plot_combined_trace(roi_id, metrics, fps, angles, save_dir):
    """
    可视化：展示 dF/F 迹线和差分 Spikes。
    根据要求，不再绘制红点标注。
    """
    trace = metrics['trace']
    spikes = metrics['spikes']
    time_axis = np.arange(len(trace)) / fps
    
    # 创建双 Y 轴图表
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    # --- 1. 绘制左轴：dF/F 信号 ---
    ax1.plot(time_axis, trace, color="#2980b9", lw=1.5, alpha=0.9, label='dF/F Trace', zorder=2)
    # 填充基线以下的透明色块以增加可读性
    ax1.fill_between(time_axis, trace, metrics['b_median'], color='#3498db', alpha=0.1, zorder=1)
    ax1.set_ylabel(r'$\Delta F/F$', fontsize=13, color='#2980b9', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    
    # --- 2. 绘制右轴：差分脉冲信号 (Spikes) ---
    ax2 = ax1.twinx()
    # 使用垂直线段展示脉冲强度
    ax2.vlines(time_axis, [0], spikes, colors='#95a5a6', alpha=0.4, lw=0.8, label='Spikes', zorder=3)
    ax2.set_ylabel('Spike Amplitude (Difference)', fontsize=13, color='#7f8c8d')
    ax2.tick_params(axis='y', labelcolor='#7f8c8d')
    
    # 动态调整右轴高度，防止 Spikes 线条遮挡上半部分角度标注
    if len(spikes) > 0 and np.max(spikes) > 0:
        ax2.set_ylim(0, np.max(spikes) * 3)
    else:
        ax2.set_ylim(0, 1)

    # --- 3. 标注刺激背景和角度文本 ---
    y_min, y_max = ax1.get_ylim()
    for j, (s, e) in enumerate(metrics.get('stim_windows', [])):
        # 绘制黄色透明背景
        ax1.axvspan(s/fps, e/fps, color='#f1c40f', alpha=0.12, zorder=0)
        # 在窗口上方标注对应的角度值
        if j < len(angles):
            ax1.text((s+e)/(2*fps), y_max * 0.95, f"{angles[j]}°", 
                     ha='center', color='#c0392b', fontsize=10, fontweight='bold')

    # 设置轴标签和标题
    ax1.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title(f"ROI: {roi_id} | Reliability: {metrics['Reliability']:.2f} | Status: {metrics['Status']}", 
                 fontsize=15, pad=10)
    
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, shadow=True)

    # 去除多余边框
    sns.despine(ax=ax1, top=True, right=False)
    
    plt.tight_layout()
    # 保存图片并关闭画布以节省内存
    plt.savefig(save_dir / f"{roi_id}_combined.pdf", dpi=200)
    plt.close()

def plot_trial_heatmap(roi_data_list, stim_windows, angles, fps, save_path, title, use_spikes=False, reference_order=None):
    """
    绘制热图：
    - 横坐标改为时间 (秒)
    - 左侧显示层次聚类树 (Spikes图也开启)
    """
    if not roi_data_list: return None
    data_key = 'spikes' if use_spikes else 'trace'
    
    # 1. 构建 DataFrame
    df_heatmap = pd.DataFrame([d[data_key] for d in roi_data_list], index=[d['ROI_ID'] for d in roi_data_list])
    df_heatmap = df_heatmap.fillna(0)
    
    # 2. 将列索引转换为时间（秒）
    # 原本列名是 0, 1, 2... 转换为 0.0, 1.2, 2.4... (取决于 FPS)
    time_columns = np.arange(df_heatmap.shape[1]) / fps
    df_heatmap.columns = [f"{t:.1f}" for t in time_columns]

    # 3. 聚类设置
    # 如果提供了参考顺序（通常用于让 Spikes 的顺序跟随 dF/F），则关闭聚类并重新排序
    # 否则（如 dF/F 第一次画图时），开启 row_cluster=True 显示左侧树
    should_cluster_row = True if reference_order is None else False
    if reference_order is not None:
        valid_order = [r for r in reference_order if r in df_heatmap.index]
        df_heatmap = df_heatmap.reindex(valid_order)

    cmap = 'magma' if not use_spikes else 'rocket'
    
    try:
        # 4. 绘图
        # row_cluster=True 会在左侧绘制 Dendrogram
        g = sns.clustermap(df_heatmap, 
                           cmap=cmap, 
                           col_cluster=False, 
                           row_cluster=should_cluster_row, 
                           standard_scale=0, 
                           figsize=(14, 10), 
                           yticklabels=len(df_heatmap) < 50,
                           xticklabels=max(1, len(time_columns) // 10)) # 每隔10个显示一个时间标签
        
        ax = g.ax_heatmap
        
        # 5. 转换刺激窗口的坐标（从帧数到时间轴上的位置）
        # 在 clustermap 中，即便 X 轴标签改了，内部坐标通常还是 0 到 N-1
        for j, (s, e) in enumerate(stim_windows):
            ax.axvline(x=s, color='white', linestyle='-', alpha=0.8, lw=1.5)
            if j < len(angles):
                ax.text((s + e) / 2, -0.5, f"{angles[j]}°", color='white', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', 
                        bbox=dict(facecolor='#c0392b', alpha=0.9, edgecolor='none', boxstyle='round'))
        
        # 设置坐标轴标签
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight='bold')
        g.fig.suptitle(title, fontsize=15, y=1.02, fontweight='bold')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        # 返回排序后的索引，供下一个热图同步顺序
        if should_cluster_row:
            return [df_heatmap.index[i] for i in g.dendrogram_row.reordered_ind]
        else:
            return list(df_heatmap.index)

    except Exception as e:
        print(f"   [!] Heatmap Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print(f"\n>>> 启动神经分析与可视化流程 | 根目录: {ROOT_DIR}")
    root = Path(ROOT_DIR)
    day_dirs = sorted([d for d in root.glob("Processed_TIF_*") if d.is_dir() and is_valid(d)])
    
    for day_dir in day_dirs:
        print(f"\n{'='*70}\n[*] 处理日期: {day_dir.name}")
        csv_files = list(day_dir.rglob("*_preprocessed.csv"))
        day_trial_summary = []
        
        for csv_file in csv_files:
            trial_path = csv_file.parent
            json_file = trial_path / csv_file.name.replace('_preprocessed.csv', '_updated.json')
            if not json_file.exists(): continue
            
            print(f"\n    >> 正在分析 Trial: {trial_path.name}")
            fps, stim_type, params, angles = get_meta(json_file)
            
            # 创建统一保存目录
            all_traces_dir = trial_path / "All_Traces"
            if all_traces_dir.exists(): shutil.rmtree(all_traces_dir)
            all_traces_dir.mkdir(parents=True)

            df_f = pd.read_csv(csv_file)
            trial_all_data = []
            
            for roi_name in tqdm(df_f.columns, desc="       ROI 分析", leave=False):
                metrics = analyze_trace(df_f[roi_name].values, fps, stim_type, params)
                
                # 记录结果
                responsive_angles = [str(angles[i]) + "°" for i, res in enumerate(metrics['stim_responses']) if res == 1 and i < len(angles)]
                day_trial_summary.append({
                    'Trial_ID': trial_path.name,
                    'ROI_ID': roi_name,
                    'Status': metrics['Status'],
                    'Reliability': metrics['Reliability'],
                    'Peak_dFF': metrics['Peak_dFF'],
                    'Responsive_Stimuli': "; ".join(responsive_angles),
                    'ERG_Spike_Count': len(metrics['valid_erg_indices'])
                })
                
                metrics.update({'ROI_ID': roi_name})
                trial_all_data.append(metrics)
                
                if SAVE_PLOTS:
                    plot_combined_trace(roi_name, metrics, fps, angles, all_traces_dir)

            # 导出 spikes.csv
            spikes_df = pd.DataFrame({d['ROI_ID']: d['spikes'] for d in trial_all_data})
            spikes_csv_path = trial_path / csv_file.name.replace('_preprocessed.csv', '_spikes.csv')
            spikes_df.to_csv(spikes_csv_path, index=False)

            # 生成热图
            if SAVE_PLOTS and trial_all_data:
                win = trial_all_data[0]['stim_windows']
                fname_dff = trial_path / f"Heatmap_All_dFF.pdf"
                
                # 第一次调用，不传 reference_order，会自动聚类并显示左侧树
                ordered_roi_names = plot_trial_heatmap(
                    trial_all_data, win, angles, fps, fname_dff, "dF/F: All"
                )
                
                if ordered_roi_names is not None:
                    fname_spikes = trial_path / f"Heatmap_All_Spikes.pdf"
                    # 第二次调用 Spikes，传入 dF/F 的排序结果，保持两者顺序一致
                    # 如果你希望 Spikes 图也独立聚类，就把最后一个参数改为 None
                    plot_trial_heatmap(
                        trial_all_data, win, angles, fps, fname_spikes, "Spikes: All", 
                        use_spikes=True, reference_order=ordered_roi_names
                    )

        if day_trial_summary:
            summary_out = day_dir / "neuro_analysis_results.csv"
            pd.DataFrame(day_trial_summary).to_csv(summary_out, index=False)

    print(f"\n{'='*70}\n>>> 全量分析任务已顺利完成。")

if __name__ == "__main__":
    main()