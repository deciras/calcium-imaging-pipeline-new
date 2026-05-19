import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# --- 配置部分 ---
ROOT_DIR = r'I:\Calcium_imaging_process\motion_correction_128'

def get_stim_events(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    fps = meta['temporal_calibration']['fps']
    params = meta['stimulation']['stim_parameters']
    
    delay = float(params.get('initial_delay_sec', 0))
    duration = float(params.get('duration_sec', 1.2))
    interval = float(params.get('interval_sec', 60))
    angles = params.get('pol_angle_list', [])
    
    events = []
    for i, angle in enumerate(angles):
        t_start = delay + i * interval
        events.append({
            'angle': angle,
            'start_f': int(t_start * fps),
            'end_f': int((t_start + duration + 5.0) * fps)
        })
    return fps, events, delay

def calculate_osi_and_angle(angle_groups):
    """
    使用向量平均法 (Vector Averaging) 计算 OSI 和 偏好角度
    对于 180° 周期的数据，角度需要乘以 2 进行计算
    """
    angles_rad = np.deg2rad([int(a) for a in angle_groups.keys()])
    responses = np.array([np.mean(v) for v in angle_groups.values()])
    
    # 转换为向量 (2*theta 映射到 360 度空间)
    vx = np.sum(responses * np.cos(2 * angles_rad))
    vy = np.sum(responses * np.sin(2 * angles_rad))
    
    osi = np.sqrt(vx**2 + vy**2) / np.sum(responses) if np.sum(responses) > 0 else 0
    pref_rad = 0.5 * np.arctan2(vy, vx)
    pref_deg = np.rad2deg(pref_rad) % 180  # 回射到 0-180 度
    
    return round(osi, 3), round(pref_deg, 1)

def analyze_tuning_full(df_f, events, fps, delay):
    stats = []
    for roi in df_f.columns:
        trace = df_f[roi].values
        baseline = trace[:int(delay * fps)]
        b_std = np.std(baseline) if np.std(baseline) > 0 else 0.01
        b_mean = np.mean(baseline)

        angle_groups = {}
        for ev in events:
            clip = trace[ev['start_f'] : ev['end_f']]
            resp = np.max(clip) if len(clip) > 0 else 0
            if ev['angle'] not in angle_groups: angle_groups[ev['angle']] = []
            angle_groups[ev['angle']].append(resp)
        
        # 计算核心统计量
        osi, pref_ang = calculate_osi_and_angle(angle_groups)
        max_resp = np.max([np.mean(v) for v in angle_groups.values()])
        z_score = (max_resp - b_mean) / b_std
        
        res = {"ROI_ID": roi, "OSI": osi, "Pref_Angle": pref_ang, "Z_Score": z_score, "Is_Responder": 1 if z_score > 3 else 0}
        for ang, v in angle_groups.items():
            res[f"Ang{ang}_mean"] = np.mean(v)
            res[f"Ang{ang}_sem"] = np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0
        stats.append(res)
    return pd.DataFrame(stats)

def plot_master_visuals(df, save_dir):
    responders = df[df['Is_Responder'] == 1].copy()
    if responders.empty: return

    # --- 1. Polar Plot (群体分布) ---
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    
    # 偏振角度在生物学上通常是 180° 周期，但 Polar plot 是 360°
    # 我们绘制两个对称的点来表示偏振轴
    rads = np.deg2rad(responders['Pref_Angle'])
    osis = responders['OSI']
    
    ax.scatter(rads, osis, c='crimson', alpha=0.6, label='Neurons')
    ax.scatter(rads + np.pi, osis, c='crimson', alpha=0.6) # 对称点
    
    ax.set_ylim(0, 1)
    ax.set_title("Population Polarization Preference (OSI vs Angle)", pad=20)
    plt.savefig(save_dir / "population_polar_plot.png", dpi=300)
    plt.close()

    # --- 2. Tuning Curves with SEM ---
    mean_cols = [c for c in df.columns if "_mean" in c]
    angles = sorted([int(c.replace("Ang", "").replace("_mean", "")) for c in mean_cols])
    
    plt.figure(figsize=(8, 5))
    for _, row in responders.nlargest(5, 'OSI').iterrows():
        means = [row[f"Ang{a}_mean"] for a in angles]
        sems = [row[f"Ang{a}_sem"] for a in angles]
        plt.errorbar(angles, means, yerr=sems, fmt='-o', capsize=3, label=f"{row['ROI_ID']} (OSI:{row['OSI']})")
    
    plt.title("Tuning Curves of High-OSI Responders")
    plt.xlabel("Angle (Deg)")
    plt.ylabel("$\Delta F/F_0$")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "tuning_best_responders.png", dpi=300)
    plt.close()

def main():
    root = Path(ROOT_DIR)
    for csv_file in root.rglob("*_preprocessed.csv"):
        json_file = csv_file.parent / csv_file.name.replace('_preprocessed.csv', '_updated.json')
        if not json_file.exists(): continue
        with open(json_file, 'r') as f:
            meta = json.load(f)
            if meta['stimulation']['stim_type'] != 'pol': continue
            
        print(f"Master分析运行中: {csv_file.parent.name}")
        fps, events, delay = get_stim_events(json_file)
        df_f = pd.read_csv(csv_file)
        
        stats_df = analyze_tuning_full(df_f, events, fps, delay)
        stats_df.to_csv(csv_file.parent / "polar_tuning_master_results.csv", index=False)
        plot_master_visuals(stats_df, csv_file.parent)

if __name__ == "__main__":
    main()