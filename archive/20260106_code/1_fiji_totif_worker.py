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


def save_z_layer(imp_source, z, nC, nT, title, output_dir_path):
    """
    Thread task: Extract a single Z-layer and save as TIF.
    """
    try:
        # Duplicate the specific Z-slice across all timepoints
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
    print("Opening (Full RAM Mode): " + file_path)
    
    options = ImporterOptions()
    options.setId(file_path)
    options.setGroupFiles(True) 
    options.setOpenAllSeries(False)
    # Loading into RAM for faster multi-threaded access
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

        save_metadata_json(imp, output_dir_path, title)
        
        print("Metadata: {} Channels, {} Z-Slices, {} Timepoints".format(nC, nZ, nT))

        # --- 1. Z-Projection (Average Intensity) ---
        if nZ > 1:
            print("Calculating Z-Projection...")
            zp = ZProjector(imp)
            zp.setMethod(ZProjector.MAX_METHOD)
            zp.setStopSlice(nZ)
            zp.doHyperStackProjection(True) 
            imp_max = zp.getProjection()
            imp_max.setDisplayMode(IJ.GRAYSCALE)
            
            proj_path = os.path.join(output_dir_path, title + "_Max_Proj.tif")
            IJ.saveAsTiff(imp_max, proj_path)
            imp_max.close()
            print("Projection saved.")

        # --- 2. Multi-threaded Z-Layer Export ---
        '''print("Exporting Z-layers using {} parallel threads...".format(threads))
        z_indices = range(1, nZ + 1)
        
        # Batching threads to prevent memory spikes
        for i in range(0, len(z_indices), threads):
            batch = z_indices[i : i + threads]
            active_threads = []
            
            for z in batch:
                t = Thread(target=save_z_layer, 
                           args=(imp, z, nC, nT, title, output_dir_path))
                active_threads.append(t)
                t.start()
            
            # Wait for the current batch to finish
            for t in active_threads:
                t.join()
            
            # Explicit Garbage Collection to stabilize RAM
            System.gc() '''

        imp.close()
        print("Successfully processed: " + title)
        print("Saved to: " + output_dir_path)

    except Exception as e:
        print("Processing error for " + file_path + ": " + str(e))

# --- Main Logic ---
root_path = rootDir.getAbsolutePath()

for root, dirs, files in os.walk(root_path):
    for f in files:
        if f.lower().endswith(ext.lower()):
            file_path = os.path.join(root, f)
            
            # create Processed_TIF_{date}
            date_str = os.path.basename(os.path.dirname(root))
            output_folder_date = File(root_path, "Processed_TIF_" + date_str)
            if not output_folder_date.exists():
                output_folder_date.mkdirs()

            # Create a specific folder for each dataset
            output_folder_obj = File(output_folder_date, os.path.basename(root))
            if not output_folder_obj.exists():
                output_folder_obj.mkdirs()
            
            process_oir(file_path, output_folder_obj.getPath())

print("\n" + "="*30)
print("BATCH PROCESSING COMPLETE")
print("="*30)