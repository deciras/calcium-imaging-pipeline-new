import pandas as pd
import numpy as np
import json
import cv2
import shutil
from pathlib import Path
import tifffile
from tqdm import tqdm
import warnings
from read_roi import read_roi_zip
import re

# ================= 配置区域 =================
ROOT_DIR = '/Volumes/Yifei_Ding/Calcium_imaging_process/Processed_TIF'
DEST_NAME = 'All_Trial_Videos'  
VIDEO_FPS = 60          
LINE_THICKNESS = 1      

# 定义不同分类的颜色 (BGR格式)
COLOR_MAP = {
    'Stimulated': (0, 255, 100),  # 绿色
    'Control': (48, 60, 232),     # 红色
    'Default': (0, 165, 255)      # 橙色 (未匹配到分类时)
}
STIM_TEXT_COLOR = (0, 165, 255) 
# ===========================================

warnings.filterwarnings("ignore")

def get_roi_status_map(day_dir):
    """读取该日期下的 neuro_analysis_results.csv，建立 ROI_ID -> Status 映射"""
    csv_path = day_dir / "neuro_analysis_results.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        # 建立映射表，处理可能的空格
        return dict(zip(df['ROI_ID'].astype(str), df['Status'].astype(str)))
    except:
        return {}

def get_roi_index_from_csv(trial_path):
    csv_file = next(trial_path.glob("Overlay Elements*.csv"), None)
    if not csv_file: return None
    try:
        df = pd.read_csv(csv_file)
        if 'Index' in df.columns:
            df_sorted = df.drop_duplicates(subset=['Name']).sort_values('Index')
            return df_sorted['Name'].tolist()
        return df['Name'].unique().tolist()
    except: return None

def get_filtered_rois_with_status(trial_path, status_map):
    zip_path = trial_path / "RoiSet.zip"
    if not zip_path.exists(): return []
    csv_names = get_roi_index_from_csv(trial_path)
    try:
        rois = read_roi_zip(zip_path)
        # 按照 CSV 索引排序获取所有 ROI 名称
        keys = [n for n in csv_names if n in rois] if csv_names else sorted(rois.keys())
        
        roi_data_list = []
        num_total = len(keys)
        
        for i, name in enumerate(keys):
            d = rois[name]
            pts = None
            # 几何形状解析
            if d['type'] in ['polygon', 'freehand', 'traced', 'polyline']:
                pts = np.array(list(zip(d['x'], d['y'])), dtype=np.int32).reshape((-1, 1, 2))
            elif d['type'] == 'oval':
                c = (int(d['left'] + d['width']/2), int(d['top'] + d['height']/2))
                a = (int(d['width']/2), int(d['height']/2))
                pts = cv2.ellipse2Poly(c, a, 0, 0, 360, 5).reshape((-1, 1, 2))
            
            if pts is not None:
                # 颜色判定逻辑：
                # 最后 5 个 ROI 为背景，画橙色 (0, 165, 255)
                # 其余所有正常 ROI 画绿色 (0, 255, 100)
                if i >= (num_total - 5):
                    color = (0, 165, 255)  # 橙色 (BGR)
                else:
                    color = (0, 255, 100)  # 绿色 (BGR)
                
                roi_data_list.append({'pts': pts, 'color': color})
        return roi_data_list
    except: return []

def process_trial_video(trial_path, status_map):
    tif_file = next(trial_path.glob("*.tif"), None)
    json_file = next(trial_path.glob("*_updated.json"), None)
    if not tif_file or not json_file: return None

    with tifffile.TiffFile(str(tif_file)) as tif:
        n_pages = len(tif.pages)
        first_page = tif.pages[0].asarray()
        h, w = first_page.shape[:2]
        
        # 增强对比度 (1% - 99.5%)
        p_low, p_high = np.percentile(first_page, [1.0, 99.5])
        
        with open(json_file, 'r') as f: meta = json.load(f)
        fps_img = meta.get('temporal_calibration', {}).get('fps', 10.0)
        stim_data = (meta.get('stimulation') or {}).get('stim_parameters') or {}
        angles = stim_data.get('pol_angle_list')
        
        has_stim = isinstance(angles, list) and len(angles) > 0
        text_hold_sec = 0
        if has_stim:
            dur, inter, delay = stim_data.get('duration_sec', 0), stim_data.get('interval_sec', 0), stim_data.get('initial_delay_sec', 0)
            text_hold_sec = inter / 2.0 if inter > 0 else 2.0
        
        # 获取带颜色的 ROI 列表
        roi_info = get_filtered_rois_with_status(trial_path, status_map)
        
        video_path = trial_path / f"{trial_path.name}_Annotated_Final.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (w, h))
        pbar = tqdm(total=n_pages, desc=f"  -> {trial_path.name}", leave=False, unit="fr")
        
        for i in range(n_pages):
            raw_frame = tif.pages[i].asarray()
            if raw_frame.ndim == 3: raw_frame = raw_frame[:,:,0]
            
            frame_enhanced = np.clip((raw_frame - p_low) * 255.0 / (p_high - p_low + 1e-5), 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2BGR)
            curr_t = i / fps_img
            
            # 【核心修改】使用 ROI 各自的颜色绘制
            for item in roi_info:
                cv2.polylines(frame_bgr, [item['pts']], True, item['color'], LINE_THICKNESS, cv2.LINE_AA)
            
            if has_stim:
                active, show_text, curr_ang = False, False, 0
                for j, ang in enumerate(angles):
                    s_s, s_e = delay + j * inter, delay + j * inter + dur
                    if s_s <= curr_t <= s_e:
                        active, show_text, curr_ang = True, True, ang
                        break
                    elif s_e < curr_t <= (s_e + text_hold_sec):
                        show_text, curr_ang = True, ang
                        break
                if show_text:
                    cv2.putText(frame_bgr, f"STIM: {int(curr_ang)} DEG", (15, 60), 1, 1.2, STIM_TEXT_COLOR, 2)
                    if active: cv2.circle(frame_bgr, (w-25, 25), 12, STIM_TEXT_COLOR, -1)
            
            cv2.putText(frame_bgr, f"Time: {curr_t:.2f}s", (15, 30), 1, 1.0, (255, 255, 255), 1)
            writer.write(frame_bgr)
            pbar.update(1)
            
        writer.release()
        pbar.close()
        return video_path

def main():
    root = Path(ROOT_DIR)
    dest_root = root / DEST_NAME
    dest_root.mkdir(exist_ok=True)
    
    day_folders = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Processed_TIF_")])
    
    for day_dir in tqdm(day_folders, desc="Overall Days"):
        # 获取该日期的状态映射表
        status_map = get_roi_status_map(day_dir)
        
        target_day_dir = dest_root / day_dir.name
        trials = sorted([t for t in day_dir.glob("*") if t.is_dir()])
        if not trials: continue
        target_day_dir.mkdir(exist_ok=True)
        
        for t_path in trials:
            try:
                v_file = process_trial_video(t_path, status_map)
                if v_file and v_file.exists():
                    shutil.copy2(v_file, target_day_dir / v_file.name)
            except Exception as e:
                print(f"\n[Error] {t_path.name}: {e}")

if __name__ == "__main__":
    main()