import os
import json
import shutil
from tqdm import tqdm
import suite2p
from suite2p.run_s2p import run_s2p

import tifffile as tf

def convert_to_bigtiff(input_path, output_path):
    # 使用 memmap 模式读取，避免内存溢出
    with tf.TiffFile(input_path) as tif:
        data = tif.asarray()
        tf.imwrite(output_path, data, bigtiff=True, compression='zlib') # 建议加压缩
        print(f"转换完成: {output_path}")

def get_params_from_json(json_path):
    """从元数据 JSON 提取参数"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        dims = data.get('dimensions', {})
        n_planes = dims.get('z_slices', 1)
        fs = data.get('temporal_calibration', {}).get('fps', 1.0)
        n_channels = dims.get('channels', 1)
        total_frames = dims.get('total_frames', None) 
        return n_planes, fs, n_channels, total_frames
    except Exception:
        return 1, 1.0, 1, None

def run_suite2p_bigtiff_adapted(root_tif_dir, delete_internal=True):
    root_tif_dir = os.path.abspath(root_tif_dir)
    parent_dir = os.path.dirname(root_tif_dir)
    base_name = os.path.basename(root_tif_dir)
    
    internal_results_root = os.path.join(parent_dir, f"{base_name}_Suite2p_Internal")
    organized_results_root = os.path.join(parent_dir, f"{base_name}_Suite2p_Results")

    # 基础配置
    ops = suite2p.default_ops()
    ops.update({
        'do_registration': 1,
        'nonrigid': True,    
        'reg_tif': True,     
        'delete_bin': 0,
        'batch_size': 50,       
        'nthreads': 8,          
        'combined': False,      
    })

    all_stacks = []
    for r, d, fs in os.walk(root_tif_dir):
        for f in fs:
            if f.endswith("_Stack.tif"):
                all_stacks.append(os.path.join(r, f))
    all_stacks.sort()

    for stack_path in tqdm(all_stacks, desc="Total Progress"):
        rel_dir = os.path.relpath(os.path.dirname(stack_path), root_tif_dir)
        stack_name = os.path.basename(stack_path)
        exp_id = stack_name.replace("_Stack.tif", "")
        json_path = stack_path.replace("_Stack.tif", "_metadata.json")
        
        n_planes, fs, n_channels, total_frames = get_params_from_json(json_path)

        if total_frames is None:
            tqdm.write(f" [Skip]: {exp_id} - No metadata found.")
            continue

        # 更新特定实验的 ops
        ops.update({'nplanes': n_planes, 'fs': fs, 'nchannels': n_channels})
        temp_save_path = os.path.join(internal_results_root, rel_dir, exp_id)
        final_exp_folder = os.path.join(organized_results_root, rel_dir, exp_id)
        
        # [核心适配]: BigTIFF 适配
        # 通过 frames_per_file 告诉 Suite2p 即使 TIF 索引损坏，也要按此帧数读取数据。
        db = {
            'data_path': [os.path.dirname(stack_path)], 
            'tiff_list': [stack_name], 
            'save_path0': temp_save_path,
            'nchannels': n_channels,
            'frames_per_file': [total_frames] 
        }

        try:
            tqdm.write(f"\n[Processing]: {exp_id} (BigTIFF Mode)")
            convert_to_bigtiff(stack_path, stack_path.replace(".tif", "_BIGTIFF.tif"))
            run_s2p(ops=ops, db=db)
        except Exception as e:
            tqdm.write(f" [Error]: {exp_id} -> {e}")
            continue

        # 整理结果
        os.makedirs(final_exp_folder, exist_ok=True)
        if os.path.exists(json_path):
            shutil.copy(json_path, os.path.join(final_exp_folder, os.path.basename(json_path)))

        search_base = os.path.join(temp_save_path, 'suite2p')
        for p in range(n_planes):
            plane_reg_dir = os.path.join(search_base, f'plane{p}', 'reg_tif')
            if os.path.exists(plane_reg_dir):
                plane_dst_dir = os.path.join(final_exp_folder, f"Plane{p}")
                os.makedirs(plane_dst_dir, exist_ok=True)
                for tf in os.listdir(plane_reg_dir):
                    if tf.endswith('.tif'):
                        shutil.move(os.path.join(plane_reg_dir, tf), 
                                    os.path.join(plane_dst_dir, f"{exp_id}_P{p}_Reg.tif"))
        
        if delete_internal and os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)

if __name__ == "__main__":
    DATA_PATH = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2025_olympus/Processed_TIF_test"
    run_suite2p_bigtiff_adapted(DATA_PATH, delete_internal=True)