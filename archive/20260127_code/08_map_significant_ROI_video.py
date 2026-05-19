import pandas as pd
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
import re
import warnings
from datetime import datetime
from read_roi import read_roi_zip

# ================= 配置区域 =================
ROOT_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/Processed_TIF'
VIDEO_FPS = 60  
# ===========================================

warnings.filterwarnings("ignore")

def cleanup_existing_videos(trial_path):
    """
    清除该 Trial 文件夹下所有已存在的 mp4 视频
    """
    existing_videos = list(trial_path.glob("*.mp4"))
    for v in existing_videos:
        try:
            v.unlink()
        except Exception as e:
            print(f"无法删除旧视频 {v.name}: {e}")

def get_significant_rois(trial_path):
    sig_dir = trial_path / "Retina_Malus_Analysis" / "Significant"
    if not sig_dir.exists(): return []
    fnames = [f.stem for f in sig_dir.glob("*.png")]
    roi_ids = [re.search(r'(\d{4}-\d{4})', f).group(1) for f in fnames if re.search(r'(\d{4}-\d{4})', f)]
    return list(set(roi_ids))

def draw_video_frame(img_2d, rois_polygons, f_idx, dff_series, spike_series, stim_info, fps_img, all_stims):
    """
    绘制视频帧：含真实轮廓、实时刺激角度、全长静态 Trace（含刺激背景带与文字）+ 扫描线
    """
    if img_2d.ndim > 2:
        img_2d = np.squeeze(img_2d)

    fig = plt.figure(figsize=(16, 10), facecolor='black')
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1.2, 0.8])
    
    # 1. 钙成像底图 (左侧)
    ax_img = fig.add_subplot(gs[0:2, 0])
    v_min, v_max = np.percentile(img_2d, [1, 99.5])
    ax_img.imshow(img_2d, cmap='gray', vmin=v_min, vmax=v_max)
    
    for rid, poly_coords in rois_polygons.items():
        poly = plt.Polygon(poly_coords, closed=True, edgecolor='#00FF00', facecolor='none', lw=1.2)
        ax_img.add_patch(poly)
        centroid = np.mean(poly_coords, axis=0)
        ax_img.text(centroid[0], centroid[1], rid, color='#00FF00', fontsize=7, fontweight='bold', ha='center')
    ax_img.axis('off')

    # 2. 右上角实时刺激大字
    ax_stim = fig.add_subplot(gs[0, 1])
    ax_stim.set_facecolor('black')
    if stim_info['active']:
        ax_stim.add_patch(plt.Rectangle((0, 0.2), 1, 0.6, color='#CC0000', alpha=0.9))
        ax_stim.text(0.5, 0.5, f"{stim_info['angle']}°", color='yellow', 
                     fontsize=60, fontweight='bold', ha='center', va='center')
    ax_stim.axis('off')

    # 3. 底部全长 Trace (含静态扫描线和刺激标注)
    ax_trace = fig.add_subplot(gs[2, :])
    ax_trace.set_facecolor('#080808')
    
    curr_t = f_idx / fps_img
    time_axis = np.arange(len(dff_series)) / fps_img
    
    ax_trace.plot(time_axis, dff_series, color='#00FFFF', lw=1.2, alpha=0.9)
    if np.max(spike_series) > 0:
        spike_display = spike_series * (np.max(dff_series) * 0.4 / np.max(spike_series))
        ax_trace.fill_between(time_axis, spike_display, color='#FF5722', alpha=0.5)
    
    y_limit = ax_trace.get_ylim()
    for s_start, s_end, s_ang in all_stims:
        ax_trace.axvspan(s_start, s_end, color='red', alpha=0.15, zorder=0)
        ax_trace.text((s_start + s_end)/2, y_limit[1]*0.9, f"{s_ang}°", 
                      color='red', fontsize=8, ha='center', fontweight='bold')
    
    ax_trace.axvline(curr_t, color='white', lw=2, zorder=10) 
    ax_trace.set_xlim(0, time_axis[-1])
    ax_trace.tick_params(colors='white', labelsize=9)
    ax_trace.set_xlabel("Time (seconds)", color='white')
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def main():
    root = Path(ROOT_DIR)
    day_dirs = sorted([d for d in root.glob("Processed_TIF_*") if d.is_dir()])
    generated_videos = []

    for day_dir in day_dirs:
        for trial_path in [d for d in day_dir.iterdir() if d.is_dir()]:
            sig_rois = get_significant_rois(trial_path)
            if not sig_rois: continue
            
            # --- 新增：每次处理前先清空该子文件夹下的旧视频 ---
            cleanup_existing_videos(trial_path)
            
            try:
                tif_path = list(trial_path.glob("*_Reg_full.tif"))[0]
                dff_path = list(trial_path.glob("*_preprocessed.csv"))[0]
                spike_path = list(trial_path.glob("*_spikes.csv"))[0]
                json_path = list(trial_path.glob("*_updated.json"))[0]
                roi_zip_path = list(trial_path.glob("ROISet.zip"))[0]
            except IndexError: continue

            print(f"\nProcessing: {trial_path.name}")
            
            roi_data = read_roi_zip(str(roi_zip_path))
            rois_polygons = {n: np.column_stack((i['x'], i['y'])) for n, i in roi_data.items() if n in sig_rois}

            with tifffile.TiffFile(str(tif_path)) as tif:
                stack = tif.asarray()
                if stack.ndim == 2: stack = np.array([p.asarray() for p in tif.pages])
            stack = np.squeeze(stack)
            if stack.ndim == 3:
                if stack.shape[0] == stack.shape[1]: stack = np.transpose(stack, (2, 0, 1))
                elif np.argmax(stack.shape) != 0: stack = np.moveaxis(stack, np.argmax(stack.shape), 0)
            
            df_dff, df_spike = pd.read_csv(dff_path), pd.read_csv(spike_path)
            with open(json_path, 'r') as f: meta = json.load(f)
            fps_img = meta['temporal_calibration']['fps']
            params = meta['stimulation']['stim_parameters']
            
            all_stims = []
            if params.get('pol_angle_list'):
                d, i, dur = params.get('initial_delay_sec', 0), params.get('interval_sec', 0), params.get('duration_sec', 0)
                for idx, a in enumerate(params['pol_angle_list']):
                    s_start = d + idx * i
                    all_stims.append((s_start, s_start + dur, a))

            rep_roi = sig_rois[0]
            d_trace, s_trace = df_dff[rep_roi].values, df_spike[rep_roi].values
            
            video_name = trial_path / f"{trial_path.name}_Final_Mapping.mp4"
            writer = None

            for f_idx in tqdm(range(len(stack)), desc="Rendering"):
                curr_t = f_idx / fps_img
                active, ang = False, 0
                for s_start, s_end, s_ang in all_stims:
                    if s_start <= curr_t <= s_end:
                        active, ang = True, s_ang
                        break
                
                frame_bgr = draw_video_frame(stack[f_idx], rois_polygons, f_idx, 
                                            d_trace, s_trace, {'active': active, 'angle': ang}, 
                                            fps_img, all_stims)
                
                if writer is None:
                    h, w, _ = frame_bgr.shape
                    writer = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (w, h))
                writer.write(frame_bgr)
            
            if writer: 
                writer.release()
                generated_videos.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | SUCCESS | {trial_path.name} -> {video_name.name}")

    # 写入 Log
    log_path = root / "video_generation_log.txt"
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if generated_videos: f.write("\n".join(generated_videos) + "\n")
        else: f.write("No videos were generated.\n")
        f.write(f"{'='*60}\n")
    
    print(f"\n>>> 日志已更新: {log_path}")

if __name__ == "__main__":
    main()