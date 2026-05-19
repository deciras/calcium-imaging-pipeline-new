# @File (label="Select Root Directory", style="directory") rootDir
# @String (label="File Extension", value=".oir") ext
# @Integer (label="Max Threads", value=8) threads

import os
import json
from java.io import File
from java.lang import System
from ij import IJ, ImagePlus
from ij.plugin import ZProjector, Duplicator
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from threading import Thread

def save_metadata_json(imp, output_dir_path, title):
    cal = imp.getCalibration()

    # 强制转成 Python 类型（避免 java.lang.Double 等导致 json.dumps 失败）
    width_px  = int(imp.getWidth())
    height_px = int(imp.getHeight())
    nC = int(imp.getNChannels())
    nZ = int(imp.getNSlices())
    nT = int(imp.getNFrames())
    total_frames = int(imp.getStackSize())

    frame_interval = float(cal.frameInterval) if cal.frameInterval is not None else 0.0
    fps = (1.0 / frame_interval) if frame_interval != 0 else 0.0

    px_w = float(cal.pixelWidth)  if cal.pixelWidth  is not None else 0.0
    px_h = float(cal.pixelHeight) if cal.pixelHeight is not None else 0.0
    unit = str(cal.getUnit())

    meta_dict = {
        "filename": str(title),
        "dimensions": {
            "width_pixel": width_px,
            "height_pixel": height_px,
            "channels": nC,
            "z_slices": nZ,
            "timepoints": nT,
            "total_frames": total_frames
        },
        "temporal_calibration": {
            "frame_interval_sec": frame_interval,
            "fps": fps
        },
        "physical_size": {
            "width": float(width_px * px_w),
            "height": float(height_px * px_h),
            "unit": unit
        }
    }

    json_path = os.path.join(output_dir_path, str(title) + "_metadata.json")

    try:
        # ✅ 先序列化，确保不会“创建空文件后才失败”
        json_str = json.dumps(meta_dict, indent=4)

        with open(json_path, "w") as f:
            f.write(json_str)

        print("Metadata JSON exported: " + json_path)
    except Exception as e:
        print("Failed to save JSON metadata: " + str(e))

def process_oir(file_path, output_dir_path):
    print("-" * 30)
    print("Opening (Full RAM Mode): " + file_path)
    
    options = ImporterOptions()
    options.setId(file_path)
    options.setGroupFiles(True) 
    options.setOpenAllSeries(False)
    options.setVirtual(False) 
    
    try:
        imps = BF.openImagePlus(options)
        if not imps:
            print("Failed to load: " + file_path)
            return
        imp = imps[0]
        imp.setDisplayMode(IJ.GRAYSCALE)
        
        title = imp.getShortTitle()
        nZ = imp.getNSlices()
        nT = imp.getNFrames()
        nC = imp.getNChannels()

        # 保存元数据
        save_metadata_json(imp, output_dir_path, title)
        
        print("Metadata: {} Channels, {} Z-Slices, {} Timepoints".format(nC, nZ, nT))


        # --- 1. 处理 Max Projection ---
        if nZ > 1:
            # 只有1个通道的情况
            if nC == 1:
                print("Calculating Max Projection for single channel...")
                zp = ZProjector(imp)
                zp.setMethod(ZProjector.MAX_METHOD)
                zp.setStopSlice(nZ)
                zp.doHyperStackProjection(True) 
                imp_max = zp.getProjection()
                
                proj_path = os.path.join(output_dir_path, title + "_Max_Proj.tif")
                IJ.saveAsTiff(imp_max, proj_path)
                imp_max.close()
                print("Projection saved.")

            # 有2个通道的情况
            elif nC == 2:
                print("Calculating Max Projections for each channel...")
                
                # --- 通道 1 ---
                imp_c1 = Duplicator().run(imp, 1, 1, 1, nZ, 1, nT)   # 注意这里是 nZ
                zp1 = ZProjector(imp_c1)
                zp1.setMethod(ZProjector.MAX_METHOD)
                zp1.setStartSlice(1)
                zp1.setStopSlice(nZ)
                zp1.doHyperStackProjection(True)
                proj_c1 = zp1.getProjection()
                IJ.saveAsTiff(proj_c1, os.path.join(output_dir_path, title + "_Max_Proj.tif"))
                proj_c1.close()
                imp_c1.close()

                # --- 通道 2 ---
                imp_c2 = Duplicator().run(imp, 2, 2, 1, nZ, 1, nT)   # 注意这里是 nZ
                zp2 = ZProjector(imp_c2)
                zp2.setMethod(ZProjector.MAX_METHOD)
                zp2.setStartSlice(1)
                zp2.setStopSlice(nZ)
                zp2.doHyperStackProjection(True)
                proj_c2 = zp2.getProjection()
                IJ.saveAsTiff(proj_c2, os.path.join(output_dir_path, title + "_Stim_Analog.tif"))
                proj_c2.close()
                imp_c2.close()

            # 多于2个通道的情况
            else:
                print("Multiple channels (>2) detected. Skipping Max Projection.")

        else:
            print("Only one Z-Slice detected. Saving individual channels as TIF.")
            # 如果不是 Z-Stack，分别保存两个通道
            if nC >= 1:
                imp_c1 = Duplicator().run(imp, 1, 1, 1, 1, 1, nT)
                IJ.saveAsTiff(imp_c1, os.path.join(output_dir_path, title + "_Max_Proj.tif"))
                imp_c1.close()
            if nC >= 2:
                imp_c2 = Duplicator().run(imp, 2, 2, 1, 1, 1, nT)
                IJ.saveAsTiff(imp_c2, os.path.join(output_dir_path, title + "_Stim_Analog.tif"))
                imp_c2.close()
            print("Images saved as TIF.")
            
        imp.close()
      
    except Exception as e:
        print("Processing error for " + file_path + ": " + str(e))
        # traceback.print_exc()

# --- 主循环逻辑 ---
try:
    root_path = rootDir.getAbsolutePath()
    for root, dirs, files in os.walk(root_path):
        if "Processed_TIF_" in root: continue # 跳过输出目录
        for f in files:
            if f.lower().endswith(ext.lower()):
                file_path = os.path.join(root, f)
                parent_dir_name = os.path.basename(os.path.dirname(root))
                output_base = os.path.join(root_path, "Processed_TIF_" + parent_dir_name)
                target_dir = os.path.join(output_base, os.path.basename(root))
                if not os.path.exists(target_dir): os.makedirs(target_dir)
                process_oir(file_path, target_dir)
                System.gc() # 清理内存

except Exception as main_e:
    print("Main loop error: " + str(main_e))
    traceback.print_exc()

print("\n" + "="*30 + "\nBATCH PROCESSING COMPLETE\n" + "="*30)