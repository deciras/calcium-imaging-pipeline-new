# conda activate caiman

import cv2
import holoviews as hv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import shutil
import json
import glob
from datetime import datetime
from tqdm import tqdm

import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.motion_correction import MotionCorrect


# ================== 配置参数 ==================
DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus'

# 是否自动清理 _normcorrected 中旧输出
CLEAN_TRIAL_OUTPUTS = True
CLEAN_DAY_OUTPUTS = True

# Set OpenCV to use a single thread
cv2.setNumThreads(0)

hv.extension('bokeh')

# Optional: Set logging level for better debugging
logging.basicConfig(level=logging.INFO)

# set up logging
logfile = None  # Replace with a path if you want to log to a file
logger = logging.getLogger('caiman')
logger.setLevel(logging.WARNING)
logfmt = logging.Formatter(
    '%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s'
)
if logfile is not None:
    handler = logging.FileHandler(logfile)
else:
    handler = logging.StreamHandler()
handler.setFormatter(logfmt)
logger.addHandler(handler)

# set env variables in case they weren't already set
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

starting_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
print(starting_time)


# ================== 工具函数 ==================
def remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  Removed old folder -> {path}")
        else:
            os.remove(path)
            print(f"  Removed old file -> {path}")


def clean_trial_outputs(output_folder_path):
    """
    清理 _normcorrected 中单个 trial 目录下由本步骤生成的文件。
    用通配符避免历史命名残留。
    """
    if not CLEAN_TRIAL_OUTPUTS:
        return 0

    patterns = [
        "*_corrected_movie.tif",
        "*_corrected_movie.tiff",
        "*_metadata.json",
        "*_brightness_trace.csv",
        "*_stim_events.csv",
        "*_stim_map.csv",
        "*_stim_trace.png",
        "*_motion_correct_figure.png",
        "memmap_*",
    ]

    removed = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_folder_path, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  Removed old folder -> {path}")
                removed += 1
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  Removed old file -> {path}")
                removed += 1
    return removed


def clean_day_outputs(output_date_folder_path):
    """
    清理 _normcorrected 中 day 层汇总文件。
    """
    if not CLEAN_DAY_OUTPUTS:
        return 0

    targets = [
        os.path.join(output_date_folder_path, "stim_events.csv"),
        os.path.join(output_date_folder_path, "stim_map.csv"),
    ]

    removed = 0
    for path in targets:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed old file -> {path}")
            removed += 1
    return removed


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  Copied -> {dst}")
        return True
    return False


def read_fps_from_metadata(metadata_path, default_fps=2.0):
    """
    从同 trial 的 *_metadata.json 中读取 temporal_calibration.fps。
    读不到或 fps=null（01 修复后写入的）时回退到 default_fps。
    """
    if not os.path.exists(metadata_path):
        print(f"[WARN] metadata not found, fallback fps={default_fps}: {metadata_path}")
        return float(default_fps)

    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)

        fps = data.get("temporal_calibration", {}).get("fps", None)

        # 修复 #6 联动：01 修复后 fps=null 时这里会读到 None，正确 fallback
        if fps is None:
            print(f"[WARN] fps is null in metadata (frame_interval=0?), "
                  f"fallback fps={default_fps}: {metadata_path}")
            return float(default_fps)

        fps = float(fps)
        if fps <= 0:
            print(f"[WARN] fps={fps} <= 0 in metadata, fallback fps={default_fps}: {metadata_path}")
            return float(default_fps)

        print(f"[INFO] Read fps={fps} from metadata: {metadata_path}")
        return fps

    except Exception as e:
        print(f"[WARN] Failed to read metadata fps, fallback fps={default_fps}: {metadata_path}")
        print(f"[WARN] {e}")
        return float(default_fps)


# ================== 核心函数 ==================
def caiman_motion_correction(movie_paths: list, cluster):
    """
    对单个 trial 做运动校正。

    修复 #2：cluster 不再在函数内部创建/销毁，由调用方（batch_motion_correction）
    统一管理生命周期，通过参数传入。这样批量处理时不会重复创建 cluster。

    修复 #11：删除无效的 movie_orig.resize(fz=0.2)，该行对实际运动校正无影响，
    只是浪费内存加载时间。

    修复 #12：路径拼接统一改为 os.path.join()，不再使用 // 。
    """
    print(f"Already set movie paths: {movie_paths}")

    movie_name = os.path.splitext(os.path.basename(movie_paths[0]))[0]
    movie_name = movie_name.replace('_Max_Proj', '')

    # root_folder_path: /mnt/.../calcium_imaging
    root_folder_path = os.path.dirname(DATA_PATH)

    # output_root_folder_name: 2026_olympus_normcorrected
    output_root_folder_name = os.path.basename(DATA_PATH) + '_normcorrected'

    # output_root_folder_path: /mnt/.../calcium_imaging/2026_olympus_normcorrected
    output_root_folder_path = os.path.join(root_folder_path, output_root_folder_name)
    if not os.path.isdir(output_root_folder_path):
        os.mkdir(output_root_folder_path)

    # input_date_folder_name: Processed_TIF_20260204
    input_date_folder_name = os.path.basename(os.path.dirname(os.path.dirname(movie_paths[0])))

    # output_date_folder_path: .../2026_olympus_normcorrected/Processed_TIF_20260204
    output_date_folder_path = os.path.join(output_root_folder_path, input_date_folder_name)
    if not os.path.isdir(output_date_folder_path):
        os.mkdir(output_date_folder_path)

    # output_folder_path: .../2026_olympus_normcorrected/Processed_TIF_20260204/movie_name
    output_folder_path = os.path.join(output_date_folder_path, movie_name)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    # 先清理旧 trial 输出
    clean_trial_outputs(output_folder_path)

    # 修复 #11：删除无效的 movie_orig 加载和 resize
    # 原版：
    #   movie_orig = cm.load_movie_chain(movie_paths)
    #   movie_orig.resize(fz=0.2)
    # 这两行对后续 MotionCorrect（使用 movie_paths 原始文件）没有任何影响，
    # 只是浪费内存，已删除。

    print(f"You have {psutil.cpu_count()} CPUs available in your current environment")

    # dataset dependent parameters
    metadata_path = movie_paths[0].replace('_Max_Proj.tif', '_metadata.json').replace('_Max_Proj.tiff', '_metadata.json')
    frate = read_fps_from_metadata(metadata_path, default_fps=2.0)
    decay_time = 2.5  # 钙信号上升的时间

    print(f"[INFO] Using frate={frate} Hz for {movie_paths[0]}")

    # 运动矫正参数
    motion_correct = True
    pw_rigid = True
    gSig_filt = (5, 5)
    max_shifts = (20, 20)
    strides = (48, 48)
    overlaps = (24, 24)
    max_deviation_rigid = 3
    border_nan = 'copy'

    mc_dict = {
        'fnames': movie_paths,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    parameters = params.CNMFParams(params_dict=mc_dict)

    if motion_correct:
        mot_correct = MotionCorrect(movie_paths, dview=cluster, **parameters.get_group('motion'))
        mot_correct.motion_correct(save_movie=True)
        fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig

        if pw_rigid:
            bord_px = np.ceil(
                np.maximum(
                    np.max(np.abs(mot_correct.x_shifts_els)),
                    np.max(np.abs(mot_correct.y_shifts_els))
                )
            ).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)
            plt.plot(mot_correct.shifts_rig)
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')
            plt.gcf().set_size_inches(6, 3)
            # 修复 #12：路径拼接改为 os.path.join
            plt.savefig(os.path.join(output_folder_path, f"{movie_name}_motion_correct_figure.png"))
            plt.close()

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(
            fname_mc,
            base_name=os.path.join(output_folder_path, 'memmap_'),
            order='C',
            border_to_0=bord_px
        )
    else:
        fname_new = cm.save_memmap(
            movie_paths,
            base_name=os.path.join(output_folder_path, 'memmap_'),
            order='C',
            border_to_0=0,
            dview=cluster
        )

    movie_corrected = cm.load(mot_correct.mmap_file)  # load motion corrected movie
    # 修复 #12：路径拼接改为 os.path.join
    corrected_movie_path = os.path.join(output_folder_path, f"{movie_name}_corrected_movie.tif")
    movie_corrected.save(corrected_movie_path, bigtiff=True)
    print(f"  Saved corrected movie -> {corrected_movie_path}")

    # -------- 复制 trial-level 文件 --------
    input_trial_folder = os.path.dirname(movie_paths[0])

    trial_files_to_copy = [
        (os.path.join(input_trial_folder, f"{movie_name}_metadata.json"),
         os.path.join(output_folder_path, f"{movie_name}_metadata.json")),
        (os.path.join(input_trial_folder, f"{movie_name}_brightness_trace.csv"),
         os.path.join(output_folder_path, f"{movie_name}_brightness_trace.csv")),
        (os.path.join(input_trial_folder, f"{movie_name}_stim_events.csv"),
         os.path.join(output_folder_path, f"{movie_name}_stim_events.csv")),
        (os.path.join(input_trial_folder, f"{movie_name}_stim_map.csv"),
         os.path.join(output_folder_path, f"{movie_name}_stim_map.csv")),
        (os.path.join(input_trial_folder, f"{movie_name}_stim_trace.png"),
         os.path.join(output_folder_path, f"{movie_name}_stim_trace.png")),
    ]

    for src, dst in trial_files_to_copy:
        copy_if_exists(src, dst)


def batch_motion_correction(input_folder, output_folder):
    """
    修复 #2：cluster 生命周期提升到本函数层。
    所有 trial 共享同一个 cluster，处理完毕后用 try/finally 确保关闭。
    原版在 caiman_motion_correction 内部用 locals() 检测 cluster，
    每次函数调用都是新的栈帧，locals() 永远找不到上一次的 cluster，
    导致批量处理时 cluster 不断累积而不释放。
    """
    # 递归寻找输入文件夹中所有以 Max_Proj.tiff 或 Max_Proj.tif 结尾的 TIFF 文件
    tiff_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('Max_Proj.tiff') or file.endswith('Max_Proj.tif'):
                tiff_files.append(os.path.join(root, file))

    tiff_files.sort()
    print(f"Found {len(tiff_files)} TIFF files in folder: {input_folder}")

    # 先准备 day 层输出目录
    input_date_folder_path = input_folder
    output_date_folder_path = os.path.join(output_folder, os.path.basename(input_date_folder_path))
    if not os.path.isdir(output_date_folder_path):
        os.mkdir(output_date_folder_path)

    # 清理 day 层旧汇总输出
    clean_day_outputs(output_date_folder_path)

    if not tiff_files:
        print("No TIFF files found, skipping cluster setup.")
        return

    # ------------------------------------------------------------------
    # 修复 #2：在 batch 层统一创建 cluster，用 try/finally 确保关闭
    # ------------------------------------------------------------------
    num_processors_to_use = None
    print("Setting up CaImAn cluster...")
    _, cluster, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing',
        n_processes=num_processors_to_use,
        ignore_preexisting=False
    )
    print(f"Successfully set up cluster with {n_processes} processes")

    try:
        for tiff_file in tqdm(tiff_files, desc="Processing TIFF files"):
            movie_paths = [tiff_file]
            caiman_motion_correction(movie_paths, cluster)

    finally:
        # 无论成功还是异常，都确保 cluster 被关闭
        try:
            cm.stop_server(dview=cluster)
            print("CaImAn cluster stopped.")
        except Exception as e:
            print(f"[WARN] Failed to stop cluster: {e}")

    # 复制 day-level stim_events.csv 和 stim_map.csv
    copy_if_exists(
        os.path.join(input_date_folder_path, 'stim_events.csv'),
        os.path.join(output_date_folder_path, 'stim_events.csv')
    )
    copy_if_exists(
        os.path.join(input_date_folder_path, 'stim_map.csv'),
        os.path.join(output_date_folder_path, 'stim_map.csv')
    )


if __name__ == '__main__':

    # 遍历 DATA_PATH，找到所有 Processed_TIF_ 开头的文件夹，并对每个文件夹进行运动矫正
    input_folders = []
    for item in os.listdir(DATA_PATH):
        item_path = os.path.join(DATA_PATH, item)
        if os.path.isdir(item_path) and item.startswith('Processed_TIF'):
            input_folders.append(item_path)

    input_folders.sort()
    for i in input_folders:
        print(f"Found folder: {i}")

    output_folder = os.path.join(
        os.path.dirname(DATA_PATH),
        os.path.basename(DATA_PATH) + '_normcorrected'
    )
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for input_folder in tqdm(input_folders, desc="Processing folders"):
        batch_motion_correction(input_folder, output_folder)
