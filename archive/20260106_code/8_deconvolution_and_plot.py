import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

# 尝试导入 OASIS (请确保在你的新环境 oasis_env 中运行)
try:
    from oasis.functions import deconvolve
    HAS_OASIS = True
except ImportError:
    HAS_OASIS = False
    print(" [提示] 未检测到 OASIS 库，将使用差分法模拟神经活动。")

# --- 配置部分 ---
ROOT_DIR = r'I:\Calcium_imaging_process\motion_correction_128'

def get_stim_events(json_path):
    """解析刺激参数"""
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    fps = meta['temporal_calibration']['fps']
    p = meta['stimulation']['stim_parameters']
    
    delay = float(p.get('initial_delay_sec', 0))
    interval = float(p.get('interval_sec', 60))
    duration = float(p.get('duration_sec', 1.2))
    angles = p.get('pol_angle_list', [])
    
    events = []
    for i, ang in enumerate(angles):
        t_start = delay + i * interval
        events.append({
            'angle': ang,
            'start_f': int(t_start * fps),
            'end_f': int((t_start + duration + 5.0) * fps), # 寻找刺激后5秒内的峰值
            'stim_t': t_start
        })
    return fps, events

def process_deconvolution(y):
    """OASIS 去卷积：从钙信号提取放电(Spikes)"""
    if HAS_OASIS:
        # c: 去卷积后的钙浓度（去噪）, s: 神经活动（spikes）
        c, s, b, g, lam = deconvolve(y, penalty=1)
        return c, s
    else:
        s = np.diff(y, prepend=y[0])
        return y, np.maximum(s, 0)

def plot_deep_analysis(df_f, events, fps, save_dir):
    """绘制深度分析图，标注红点(Peak)和刺激关系"""
    time_axis = np.arange(len(df_f)) / fps
    plot_dir = save_dir / "ROI_Traces"
    plot_dir.mkdir(exist_ok=True)
    
    for roi in df_f.columns:
        trace = df_f[roi].values
        c_denoised, spikes = process_deconvolution(trace)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Top: 钙信号与红点(Peak) ---
        ax1.plot(time_axis, trace, color='lightgray', label='Raw $\Delta F/F_0$', alpha=0.5)
        ax1.plot(time_axis, c_denoised, color='tab:blue', label='Deconvolved Calcium', linewidth=1.5)
        
        for ev in events:
            # 1. 红色阴影：刺激持续时间
            ax1.axvspan(ev['stim_t'], ev['stim_t'] + 1.2, color='red', alpha=0.15)
            
            # 2. 寻找并画出红点 (Peak)
            win = trace[ev['start_f'] : ev['end_f']]
            if len(win) > 0:
                p_idx = np.argmax(win) + ev['start_f']
                p_time, p_val = p_idx / fps, trace[p_idx]
                
                # 画红点
                ax1.scatter(p_time, p_val, color='red', s=40, edgecolors='black', zorder=10)
                
                # 画绿色延迟线 (刺激开始 -> 达峰)
                ax1.hlines(y=p_val, xmin=ev['stim_t'], xmax=p_time, colors='green', linestyles='--', alpha=0.4)
                
                # 标注角度
                ax1.text(ev['stim_t'], ax1.get_ylim()[1]*0.8, f"{ev['angle']}°", fontsize=9)

        ax1.set_title(f"ROI Physiological Analysis: {roi}")
        ax1.set_ylabel("$\Delta F/F_0$")
        ax1.legend(loc='upper right')

        # --- Bottom: 推断的神经放电 ---
        ax2.fill_between(time_axis, spikes, color='black', alpha=0.8)
        ax2.set_ylabel("Spiking Activity")
        ax2.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(plot_dir / f"{roi}_deconv.png", dpi=200)
        plt.close()

def main():
    root = Path(ROOT_DIR)
    for csv_file in root.rglob("*_preprocessed.csv"):
        json_file = csv_file.parent / csv_file.name.replace('_preprocessed.csv', '_updated.json')
        if not json_file.exists(): continue
        
        # 1. --- 清理逻辑 ---
        # 删除旧图文件夹
        old_plot_dir = csv_file.parent / "ROI_Traces"
        if old_plot_dir.exists():
            shutil.rmtree(old_plot_dir)
            print(f" [清理] 已删除旧图: {csv_file.parent.name}")
        
        # 删除旧的分析结果 CSV
        old_csv = csv_file.parent / "deconv_results.csv"
        if old_csv.exists(): old_csv.unlink()

        # 2. --- 分析逻辑 ---
        with open(json_file, 'r') as f:
            if json.load(f)['stimulation']['stim_type'] != 'pol': continue
            
        print(f" [运行] 正在生成深度分析图: {csv_file.parent.name}")
        fps, events = get_stim_events(json_file)
        df_f = pd.read_csv(csv_file)
        
        plot_deep_analysis(df_f, events, fps, csv_file.parent)

if __name__ == "__main__":
    main()