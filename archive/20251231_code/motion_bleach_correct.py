import os
import numpy as np
import tifffile as tf
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
from tqdm import tqdm

def exp_func(t, a, k, c):
    """指数衰减模型"""
    return a * np.exp(-k * t) + c

def stable_process(file_path, save_path):
    print(f"\nReading: {os.path.basename(file_path)}")
    # 使用 memmap=True 可以防止大文件直接撑爆内存
    with tf.TiffFile(file_path) as tif:
        video = tif.asarray().astype(np.float32)
    
    T, Y, X = video.shape

    # --- 1. 指数漂白矫正 ---
    print("  Step 1: Bleach Correction (Exponential)...")
    xdata = np.arange(T)
    ydata = np.mean(video, axis=(1, 2))
    # 初始猜测: a=振幅, k=衰减常数, c=基线
    p0 = (ydata[0] - ydata[-1], 0.0005, ydata[-1])
    try:
        popt, _ = curve_fit(exp_func, xdata, ydata, p0=p0, maxfev=2000)
        fit_curve = exp_func(xdata, *popt)
        # 归一化增益
        correction = fit_curve[0] / fit_curve
        video = video * correction[:, np.newaxis, np.newaxis]
    except Exception as e:
        print(f"  [Skip] Bleach fitting failed: {e}")

    # --- 2. 刚性运动矫正 (Phase Correlation) ---
    print(f"  Step 2: Rigid Registration (T={T})...")
    # 使用前 50 帧的平均图作为稳健模板
    template = np.mean(video[:50], axis=0)
    
    corrected_video = np.zeros_like(video)
    corrected_video[0] = video[0]
    
    # 遍历后续帧
    for t in tqdm(range(T), desc="  Registering frames"):
        # 计算位移 (upsample_factor=10 提供亚像素精度)
        # shifts 结果为 [row_shift, col_shift]
        shifts, error, diffphase = phase_cross_correlation(
            template, video[t], upsample_factor=10
        )
        
        # 使用 skimage.transform.warp 进行平移，这比 scipy.shift 更稳定
        # 注意：skimage 的变换矩阵使用的是 (x, y)，而 shifts 是 (row, col) 即 (y, x)
        tform = AffineTransform(translation=(-shifts[1], -shifts[0]))
        corrected_video[t] = warp(video[t], tform, mode='reflect', preserve_range=True)

    # --- 3. 保存结果 ---
    # 裁剪掉可能的负值并转回 uint16
    print(f"  Saving to: {os.path.basename(save_path)}")
    final_output = np.clip(corrected_video, 0, 65535).astype(np.uint16)
    tf.imwrite(save_path, final_output, photometric='minisblack')

# --- 自动批处理逻辑 ---
def main():
    # 获取当前脚本所在目录作为起点，或者指定路径
    root_dir = "/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2025_olympus/Processed_TIF_20251205"
    
    for subdir, dirs, files in os.walk(root_dir):
        # 寻找所有 Z001...Z009 的文件，排除掉已经处理过的
        z_files = [f for f in files if "Z" in f and f.endswith(".tif") and "_corrected" not in f]
        
        for f in z_files:
            in_p = os.path.join(subdir, f)
            out_p = os.path.join(subdir, f.replace(".tif", "_corrected.tif"))
            
            # 如果已经存在结果文件，跳过，方便断点续传
            if os.path.exists(out_p):
                continue
                
            try:
                stable_process(in_p, out_p)
            except Exception as e:
                print(f"Error processing {in_p}: {e}")

if __name__ == "__main__":
    main()