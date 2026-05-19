import numpy as np
import tifffile as tiff
from tqdm import tqdm
import os

# 统一亮度的函数
def adjust_brightness(frame, target_brightness):
    current_brightness = np.mean(frame)
    adjustment_factor = target_brightness - current_brightness
    adjusted_frame = frame + adjustment_factor
    return adjusted_frame

def get_tiff_files(directory, recursive=True):
    """
    获取指定目录中的所有 .tif 和 .tiff 文件路径。

    参数：
        directory (str): 要查找的目录路径。
        recursive (bool): 是否递归查找子目录，默认为 True。

    返回：
        list: 包含所有找到的 TIFF 文件路径的列表。
    """
    tiff_files = []

    if recursive:
        # 使用 os.walk 递归查找
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.tif', '.tiff')):
                    file_path = os.path.join(root, filename)
                    tiff_files.append(file_path)
                    # logging.info(f'TIFF file path added: {file_path}')
    else:
        # 非递归，只查找当前目录
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.tif', '.tiff')):
                    file_path = entry.path
                    tiff_files.append(file_path)
                    # logging.info(f'TIFF file path added: {file_path}')

    return tiff_files

def process_tiff_file(tiff_file_path, suite2p_compatible=False):
    # 使用 TiffFile 读取元数据
    with tiff.TiffFile(tiff_file_path) as tif:
        if len(tif.pages) == 0:
            print(f'No pages found in {tiff_file_path}. Skipping...')
            return None
        
        # 获取第一帧的形状
        frame_shape = tif.pages[0].shape
        num_channels = frame_shape[-1] if len(frame_shape) == 3 else 1  # 检查通道数

        if num_channels == 3:  # 三通道图像
            print(f'Skipping {tiff_file_path} (3-channel image)')
            return None
        elif num_channels != 1:  # 非单通道图像
            print(f'Skipping {tiff_file_path} (not a single-channel image)')
            return None
        
        # 读取所有帧
        frames = tif.asarray()
        print(f'Already loaded TIFF file from path: {tiff_file_path}')

        # 计算均值亮度
        overall_average_brightness = np.mean([np.mean(frame) for frame in tqdm(frames, desc='Calculating frame average:')])
        
        # 调整所有帧的亮度
        adjusted_frames = np.array([adjust_brightness(frame, overall_average_brightness) for frame in tqdm(frames, desc='Adjusting frame average:')])

        tiff_file_name = os.path.basename(tiff_file_path)

        # 保存调整后的帧为新的多帧 TIFF 文件
        save_folder = os.path.dirname(tiff_file_path) + "_brightness_adjusted"
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        # 每个文件单独创建一个文件夹，为了适应suite2p
        if suite2p_compatible:
            save_path_folder = os.path.join(save_folder, f'{tiff_file_name[:-4]}_brightness_adjusted')
            if not os.path.isdir(save_path_folder):
                os.mkdir(save_path_folder)
        else:
            save_path_folder = save_folder

        save_tiff_file_path = f'{save_path_folder}/{tiff_file_name[:-4]}_brightness_adjusted.tif'
        tiff.imwrite(save_tiff_file_path, adjusted_frames)
        print(f'Already saved adjusted TIFF file to: {save_tiff_file_path}')

def main(tiff_file_paths: list, suite2p_compatible=False):
    # 顺序处理 TIFF 文件
    for tiff_file_path in tqdm(tiff_file_paths, desc="Processing TIFF files"):
        process_tiff_file(tiff_file_path, suite2p_compatible=suite2p_compatible)

if __name__ == "__main__":
    tiff_file_folders = [
        '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/20251014/tiff'
    ]

    for tiff_file_folder in tqdm(tiff_file_folders, desc="Processing TIFF folders"):    
        tiff_file_paths = get_tiff_files(tiff_file_folder)
        main(tiff_file_paths, suite2p_compatible=False)