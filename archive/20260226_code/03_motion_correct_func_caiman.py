# conda activate caiman

import cv2
import holoviews as hv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import shutil
from datetime import datetime
import sys
from tqdm import tqdm

import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.motion_correction import MotionCorrect


# ================== 配置参数 ==================
# DATA_PATH : /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus
DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus'


# Set OpenCV to use a single thread
cv2.setNumThreads(0)

hv.extension('bokeh')

# Optional: Set logging level for better debugging
logging.basicConfig(level=logging.INFO)

# set up logging
logfile = None # Replace with a path if you want to log to a file
logger = logging.getLogger('caiman')
logger.setLevel(logging.WARNING)
logfmt = logging.Formatter('%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s')
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

# 处理一个文件的函数
def caiman_motion_correction(movie_paths:list):

    print(f"Already set movie paths: {movie_paths}")

    movie_name = os.path.splitext(os.path.basename(movie_paths[0]))[0]
    movie_name = movie_name.replace('_Max_Proj', '')

    # root_folder_path: /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging
    root_folder_path = os.path.dirname(DATA_PATH)

    # output_root_folder_name: 2026_olympus_normcorrected
    output_root_folder_name = os.path.basename(DATA_PATH) + '_normcorrected'

    # output_root_folder_path: /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected
    output_root_folder_path = os.path.join(root_folder_path, output_root_folder_name)
    if not os.path.isdir(output_root_folder_path):
        os.mkdir(output_root_folder_path)

    # print(f"Output root folder path: {output_root_folder_path}")

    # input_date_folder_name: Processed_TIF_20260204
    input_date_folder_name = os.path.basename(os.path.dirname(os.path.dirname(movie_paths[0])))

    # output_date_folder_path: /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected/Processed_TIF_20260204
    output_date_folder_path = os.path.join(output_root_folder_path, input_date_folder_name)
    if not os.path.isdir(output_date_folder_path):
        os.mkdir(output_date_folder_path)
        # print(f"Already created folder: {output_date_folder_path}")

    # output_folder_path: /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected/Processed_TIF_20260204/movie_name
    output_folder_path = os.path.join(output_date_folder_path, movie_name)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
        # print(f"Already created folder: {output_folder_path}")

    movie_orig = cm.load_movie_chain(movie_paths) 
    downsampling_ratio = 0.2  # subsample 5x
    movie_orig.resize(fz=downsampling_ratio)

    print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
    num_processors_to_use = None

    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=num_processors_to_use, 
                                                    ignore_preexisting=False)
    print(f"Successfully set up cluster with {n_processes} processes")

    # dataset dependent parameters
    frate = 2                       # 帧率
    decay_time = 2.5                 # 钙信号上升的时间

    # 运动矫正参数
    motion_correct = True    # flag for performing motion correction
    pw_rigid = True         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (5, 5)       # 细胞移动像素大小的标准差
    max_shifts = (20, 20)      # 细胞移动最大的像素，别调太大
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'      # replicate values along the boundaries

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
        # do motion correction rigid
        mot_correct = MotionCorrect(movie_paths, dview=cluster, **parameters.get_group('motion'))
        mot_correct.motion_correct(save_movie=True)
        fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
                                        np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)
            # Plot shifts
            plt.plot(mot_correct.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')
            plt.gcf().set_size_inches(6,3)
            plt.savefig(f"{output_folder_path}//{movie_name}_motion_correct_figure.png")

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name=f'{output_folder_path}//memmap_', order='C',
                                border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(movie_paths, base_name=f'{output_folder_path}//memmap_',
                                order='C', border_to_0=0, dview=dview)

    movie_corrected = cm.load(mot_correct.mmap_file) # load motion corrected movie
    '''ds_ratio = 0.2
    concatenate_movie = cm.concatenate([movie_orig.resize(1, 1, ds_ratio) - mot_correct.min_mov*mot_correct.nonneg_movie,
                    movie_corrected.resize(1, 1, ds_ratio)], 
                    axis=2)

    concatenate_movie.save(f"{output_folder_path}//{movie_name}_concatenate_movie.tif")'''

    movie_corrected.save(f"{output_folder_path}//{movie_name}_corrected_movie.tif", bigtiff=True)

    shutil.copy(f"{os.path.dirname(movie_paths[0])}//{movie_name}_metadata.json", f"{output_folder_path}//{movie_name}_metadata.json")
    shutil.copy(f"{os.path.dirname(movie_paths[0])}//{movie_name}_brightness_trace.csv", f"{output_folder_path}//{movie_name}_brightness_trace.csv")


def batch_motion_correction(input_folder, output_folder):
    
    # 递归寻找输入文件夹中所有以Max_Proj.tiff或Max_Proj.tif结尾的TIFF文件
    # 并将其路径存入movie_paths列表中
    tiff_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('Max_Proj.tiff') or file.endswith('Max_Proj.tif'):
                tiff_files.append(os.path.join(root, file))

    tiff_files.sort()  # 可选：对文件进行排序，确保处理顺序一致
    print(f"Found {len(tiff_files)} TIFF files in folder: {input_folder}")

    for tiff_file in tqdm(tiff_files, desc="Processing TIFF files"):
        movie_path = tiff_file
        movie_paths = [movie_path]
        
        # 调用caiman_motion_correction函数进行运动矫正
        caiman_motion_correction(movie_paths)

        # 复制stim_events.csv和stim_map.csv文件到输出文件夹中
        input_date_folder_path  = os.path.dirname(os.path.dirname(movie_path))
        output_date_folder_path = os.path.join(output_folder, os.path.basename(input_date_folder_path))
        if not os.path.isdir(output_date_folder_path):
            os.mkdir(output_date_folder_path)
        
        shutil.copy(f'{input_date_folder_path}//stim_events.csv', f'{output_date_folder_path}//stim_events.csv')
        shutil.copy(f'{input_date_folder_path}//stim_map.csv', f'{output_date_folder_path}//stim_map.csv')

    return



if __name__ == '__main__':

    # 遍历DATA_PATH，找到所有Pocessed_TIF_开头的文件夹，并对每个文件夹进行运动矫正
    input_folders = []
    for item in os.listdir(DATA_PATH):
        item_path = os.path.join(DATA_PATH, item)
        if os.path.isdir(item_path) and item.startswith('Processed_TIF'):
            input_folders.append(item_path)
    
    input_folders.sort()  # 可选：对文件夹进行排序，确保处理顺序一致
    for i in input_folders:
        print(f"Found folder: {i}")

    output_folder = os.path.join(os.path.dirname(DATA_PATH), os.path.basename(DATA_PATH) + '_normcorrected')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for input_folder in tqdm(input_folders, desc="Processing folders"):
        batch_motion_correction(input_folder, output_folder)

