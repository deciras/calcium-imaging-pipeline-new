# @File (label="Select Root Directory", style="directory") rootDir
# @String (label="File Extension", value=".oir") ext
# @Integer (label="Max Threads", value=8) threads
# @Boolean (label="Generate Z-Projection", value=true) doZProj
# @Boolean (label="Generate Z-Layers", value=true) doZLayers
# @Boolean (label="Generate Full Stack", value=true) doStack

import os
import json
from java.io import File
from java.lang import System
from ij import IJ, ImagePlus
from ij.plugin import ZProjector, Duplicator
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from threading import Thread
from ij import ImageStack, ImagePlus
from ij.plugin import ZProjector

def save_metadata_json(imp, output_dir_path, title):
    """
    提取元数据并保存为 JSON 文件
    """
    cal = imp.getCalibration()
    
    # [关键修改]: 获取物理总帧数。Fiji 在打开 .oir 时能准确识别所有帧，
    # 即使后续保存的 TIF 超过 4GB，这个数字在 JSON 里也是准确的。
    total_frames = imp.getStackSize() 
    
    # 构建元数据字典
    meta_dict = {
        "filename": title,
        "dimensions": {
            "width_pixel": imp.getWidth(),
            "height_pixel": imp.getHeight(),
            "channels": imp.getNChannels(),
            "z_slices": imp.getNSlices(),
            "timepoints": imp.getNFrames(),
            "total_frames": total_frames  # [新增] 存入总帧数
        },
        "temporal_calibration": {
            "frame_interval_sec": cal.frameInterval,
            "fps": (1.0 / cal.frameInterval) if cal.frameInterval != 0 else 0
        },
        "physical_size": {
            "width": imp.getWidth() * cal.pixelWidth,
            "height": imp.getHeight() * cal.pixelHeight,
            "unit": cal.getUnit()
        }
    }
    
    json_path = os.path.join(output_dir_path, title + "_metadata.json")
    
    try:
        # 在 Jython 中使用 json.dumps 序列化
        with open(json_path, "w") as f:
            f.write(json.dumps(meta_dict, indent=4))
        print("Metadata JSON exported: " + json_path)
    except Exception as e:
        print("Failed to save JSON metadata: " + str(e))


def z_project_hyperstack(imp):
    """
    对 HyperStack（多个通道/时间点）进行逐通道逐时间 Z-平均，组合成新堆栈返回
    """
    nC = imp.getNChannels()
    nZ = imp.getNSlices()
    nT = imp.getNFrames()

    width, height = imp.getWidth(), imp.getHeight()
    stack = ImageStack(width, height)
    
    for t in range(1, nT + 1):
        for c in range(1, nC + 1):
            # Duplicator(C, C, Z1, Z2, T, T)
            dup = Duplicator().run(imp, c, c, 1, nZ, t, t)
            zp = ZProjector(dup)
            zp.setMethod(ZProjector.AVG_METHOD)
            zp.setStartSlice(1)
            zp.setStopSlice(nZ)
            zp.doProjection()
            proj = zp.getProjection()
            stack.addSlice("C{}_T{}".format(c, t), proj.getProcessor())
            proj.close()
            dup.close()
    
    proj_imp = ImagePlus("Projected", stack)
    proj_imp.setDimensions(nC, 1, nT)  # C, Z, T
    # proj_imp.setDisplayMode(IJ.GRAYSCALE)
    return proj_imp



def save_z_layer(imp_source, z, nC, nT, title, output_dir_path):
    """
    提取单个 Z 层并保存
    """
    try:
        imp_z = Duplicator().run(imp_source, 1, nC, z, z, 1, nT)
        imp_z.setDisplayMode(IJ.GRAYSCALE)
        z_name = "{}_Z{:03d}.tif".format(title, z)
        save_path = os.path.join(output_dir_path, z_name)
        IJ.saveAsTiff(imp_z, save_path)
        imp_z.close()
    except Exception as e:
        print("Error in Thread-Z{}: {}".format(z, str(e)))

def process_oir(file_path, output_dir_path):
    print("-" * 30)
    print("Opening: " + file_path)
    
    options = ImporterOptions()
    options.setId(file_path)
    options.setGroupFiles(True) 
    options.setOpenAllSeries(False)
    options.setVirtual(True) 
    
    try:
        imps = BF.openImagePlus(options)
        if not imps:
            print("Failed to load: " + file_path)
            return
        imp = imps[0]
        imp.setDisplayMode(IJ.GRAYSCALE)
        
        title = imp.getShortTitle()

        # --- 核心修改：导出 JSON 元数据 ---
        save_metadata_json(imp, output_dir_path, title)

        # --- 保存原始 Stack ---
        if doStack:
            IJ.saveAsTiff(imp, os.path.join(output_dir_path, title + "_Stack.tif"))
            print("Stack saved.")

        # --- Z-Projection ---
        if doZProj and imp.getNSlices() > 1:
            try:
                print("Calculating HyperStack Z-Projection...")
                proj_imp = z_project_hyperstack(imp)
                
                proj_path = os.path.join(output_dir_path, title + "_Avg_Proj.tif")
                IJ.saveAsTiff(proj_imp, proj_path)
                proj_imp.close()
                print("Projection saved.")
            except Exception as e:
                print("Z-Projection error: " + str(e))


        # --- 多线程分层导出 ---
        if doZLayers:
            nZ = imp.getNSlices()
            nC = imp.getNChannels()
            nT = imp.getNFrames()
            for i in range(0, nZ, threads):
                batch = range(i + 1, min(i + threads + 1, nZ + 1))
                active_threads = [Thread(target=save_z_layer, args=(imp, z, nC, nT, title, output_dir_path)) for z in batch]
                for t in active_threads: t.start()
                for t in active_threads: t.join()
                System.gc()
            print("Z-Layers saved.")

        imp.close()
    except Exception as e:
        print("Processing error for " + file_path + ": " + str(e))

# --- 主循环逻辑 ---
root_path = rootDir.getAbsolutePath()
for root, dirs, files in os.walk(root_path):
    files.sort()
    for f in files:
        if f.lower().endswith(ext.lower()):
            file_path = os.path.join(root, f)
            # 获取父文件夹名（日期）
            date_str = os.path.basename(os.path.dirname(root))
            output_folder_date = File(root_path, "Processed_TIF_" + date_str)
            if not output_folder_date.exists():
                output_folder_date.mkdirs()
            
            process_oir(file_path, output_folder_date.getPath())

print("\n" + "="*30)
print("BATCH PROCESSING COMPLETE")
print("="*30)