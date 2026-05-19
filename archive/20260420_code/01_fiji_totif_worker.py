# @File (label="Select Root Directory", style="directory") rootDir
# @String (label="File Extension", value=".oir") ext
# @Integer (label="Max Threads", value=8) threads
# @String (label="Mode", value="skip") mode

import os
import glob
import json
from java.lang import System
from ij import IJ
from ij.plugin import ZProjector, Duplicator
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
import traceback


def file_nonempty(path):
    return os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_file_if_exists(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            print("[INFO] Removed old file: " + path)
        except Exception as e:
            print("[WARNING] Failed to remove old file: {} | {}".format(path, str(e)))


def get_expected_outputs(output_dir_path, title):
    return {
        "metadata": os.path.join(output_dir_path, title + "_metadata.json"),
        "max_proj": os.path.join(output_dir_path, title + "_Max_Proj.tif"),
        "stim_analog": os.path.join(output_dir_path, title + "_Stim_Analog.tif"),
    }


def is_already_processed(output_dir_path, title):
    """
    不再严格依赖 title 精确匹配。
    只要当前 trial 输出目录下已经存在:
      - 任意非空 *_metadata.json
      - 任意非空 *_Max_Proj.tif
    就认为该 trial 已经处理过，可 skip。

    这样可兼容历史版本里 title 命名不一致的问题
    （例如 basename vs imp.getShortTitle）。
    """
    meta_files = glob.glob(os.path.join(output_dir_path, "*_metadata.json"))
    maxproj_files = glob.glob(os.path.join(output_dir_path, "*_Max_Proj.tif"))

    meta_ok = any(file_nonempty(p) for p in meta_files)
    max_ok = any(file_nonempty(p) for p in maxproj_files)

    print("[DEBUG] Checking existing outputs for: " + title)
    print("[DEBUG] output_dir   : " + output_dir_path)
    print("[DEBUG] meta_files   : {}".format(meta_files))
    print("[DEBUG] maxproj_files: {}".format(maxproj_files))
    print("[DEBUG] meta_ok={}, max_ok={}".format(meta_ok, max_ok))

    return meta_ok and max_ok


def clear_previous_outputs(output_dir_path, title):
    outputs = get_expected_outputs(output_dir_path, title)
    remove_file_if_exists(outputs["metadata"])
    remove_file_if_exists(outputs["max_proj"])
    remove_file_if_exists(outputs["stim_analog"])


def save_metadata_json(imp, output_dir_path, title):
    cal = imp.getCalibration()

    width_px = int(imp.getWidth())
    height_px = int(imp.getHeight())
    nC = int(imp.getNChannels())
    nZ = int(imp.getNSlices())
    nT = int(imp.getNFrames())
    total_frames = int(imp.getStackSize())

    frame_interval = float(cal.frameInterval) if cal.frameInterval is not None else 0.0
    px_w = float(cal.pixelWidth) if cal.pixelWidth is not None else 0.0
    px_h = float(cal.pixelHeight) if cal.pixelHeight is not None else 0.0
    unit = str(cal.getUnit())

    if frame_interval > 0:
        fps = 1.0 / frame_interval
    else:
        fps = None
        print("[WARNING] frame_interval=0 for '{}', fps set to null in metadata. Check Olympus calibration settings.".format(title))

    fov_width_um = float(width_px * px_w)
    fov_height_um = float(height_px * px_h)

    if width_px > 0 and px_w > 0:
        pixel_size_um = float(px_w)
    else:
        pixel_size_um = None

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
            "fov_width_um": fov_width_um,
            "fov_height_um": fov_height_um,
            "pixel_size_um": pixel_size_um,
            "unit": unit,
            # backward compatibility
            "width": fov_width_um,
            "height": fov_height_um,
        }
    }

    json_path = os.path.join(output_dir_path, str(title) + "_metadata.json")

    try:
        json_str = json.dumps(meta_dict, indent=4)
        with open(json_path, "w") as f:
            f.write(json_str)
        print("Metadata JSON exported: " + json_path)
    except Exception as e:
        print("Failed to save JSON metadata: " + str(e))


def save_single_channel_projection(imp, output_path, nZ):
    zp = ZProjector(imp)
    zp.setMethod(ZProjector.MAX_METHOD)
    zp.setStartSlice(1)
    zp.setStopSlice(nZ)
    zp.doHyperStackProjection(True)
    proj = zp.getProjection()
    IJ.saveAsTiff(proj, output_path)
    proj.close()


def process_oir(file_path, output_dir_path, run_mode):
    print("-" * 40)
    print("Input file : " + file_path)
    print("Output dir : " + output_dir_path)

    title = os.path.splitext(os.path.basename(file_path))[0]

    if run_mode.lower() == "skip" and is_already_processed(output_dir_path, title):
        print("[SKIP] Already converted: " + title)
        return

    if run_mode.lower() == "overwrite":
        clear_previous_outputs(output_dir_path, title)

    options = ImporterOptions()
    options.setId(file_path)
    options.setGroupFiles(True)
    options.setOpenAllSeries(False)
    options.setVirtual(False)

    try:
        imps = BF.openImagePlus(options)
        if not imps:
            print("[ERROR] Failed to load: " + file_path)
            return

        imp = imps[0]
        imp.setDisplayMode(IJ.GRAYSCALE)

        actual_title = imp.getShortTitle()
        nZ = imp.getNSlices()
        nT = imp.getNFrames()
        nC = imp.getNChannels()

        if actual_title != title:
            print("[INFO] Filename title = {}, ImageJ short title = {}".format(title, actual_title))

        print("Metadata: {} Channels, {} Z-Slices, {} Timepoints".format(nC, nZ, nT))

        save_metadata_json(imp, output_dir_path, title)

        max_proj_path = os.path.join(output_dir_path, title + "_Max_Proj.tif")
        stim_analog_path = os.path.join(output_dir_path, title + "_Stim_Analog.tif")

        if nZ > 1:
            if nC == 1:
                print("Calculating Max Projection for single channel...")
                zp = ZProjector(imp)
                zp.setMethod(ZProjector.MAX_METHOD)
                zp.setStopSlice(nZ)
                zp.doHyperStackProjection(True)
                imp_max = zp.getProjection()
                IJ.saveAsTiff(imp_max, max_proj_path)
                imp_max.close()
                print("Projection saved: " + max_proj_path)

            elif nC == 2:
                print("Calculating Max Projections for each channel...")

                imp_c1 = Duplicator().run(imp, 1, 1, 1, nZ, 1, nT)
                save_single_channel_projection(imp_c1, max_proj_path, nZ)
                imp_c1.close()
                print("Saved channel 1 projection: " + max_proj_path)

                imp_c2 = Duplicator().run(imp, 2, 2, 1, nZ, 1, nT)
                save_single_channel_projection(imp_c2, stim_analog_path, nZ)
                imp_c2.close()
                print("Saved channel 2 projection: " + stim_analog_path)

            else:
                print("[WARNING] {} channels detected (>2). Saving channel 1 as Max_Proj.tif; channels 2+ are ignored.".format(nC))
                imp_c1 = Duplicator().run(imp, 1, 1, 1, nZ, 1, nT)
                save_single_channel_projection(imp_c1, max_proj_path, nZ)
                imp_c1.close()
                print("Saved channel 1 projection: " + max_proj_path)

        else:
            print("Only one Z-Slice detected. Saving individual channels as TIF.")

            if nC >= 1:
                imp_c1 = Duplicator().run(imp, 1, 1, 1, 1, 1, nT)
                IJ.saveAsTiff(imp_c1, max_proj_path)
                imp_c1.close()
                print("Saved channel 1 image: " + max_proj_path)

            if nC >= 2:
                imp_c2 = Duplicator().run(imp, 2, 2, 1, 1, 1, nT)
                IJ.saveAsTiff(imp_c2, stim_analog_path)
                imp_c2.close()
                print("Saved channel 2 image: " + stim_analog_path)

        imp.close()
        print("[DONE] " + title)

    except Exception as e:
        print("[ERROR] Processing error for " + file_path + ": " + str(e))
        traceback.print_exc()


try:
    root_path = rootDir.getAbsolutePath()
    run_mode = str(mode).strip().lower()

    if run_mode not in ["skip", "overwrite"]:
        print("[WARNING] Unknown mode '{}', fallback to 'skip'".format(run_mode))
        run_mode = "skip"

    print("=" * 40)
    print("Fiji batch conversion started")
    print("Root path : " + root_path)
    print("Extension : " + ext)
    print("Threads   : " + str(threads))
    print("Mode      : " + run_mode)
    print("=" * 40)

    for root, dirs, files in os.walk(root_path):
        if "Processed_TIF_" in root:
            continue

        for f in files:
            if not f.lower().endswith(ext.lower()):
                continue

            file_path = os.path.join(root, f)

            parent_dir_name = os.path.basename(os.path.dirname(root))
            output_base = os.path.join(root_path, "Processed_TIF_" + parent_dir_name)
            target_dir = os.path.join(output_base, os.path.basename(root))
            ensure_dir(target_dir)

            process_oir(file_path, target_dir, run_mode)
            System.gc()

except Exception as main_e:
    print("Main loop error: " + str(main_e))
    traceback.print_exc()

print("\n" + "=" * 30 + "\nBATCH PROCESSING COMPLETE\n" + "=" * 30)