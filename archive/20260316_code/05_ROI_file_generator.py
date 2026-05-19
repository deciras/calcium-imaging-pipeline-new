import os
import re
import traceback
import numpy as np
import roifile

from skimage import measure
from roifile import ROI_TYPE

DATA_PATH = '/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus_normcorrected'


def natural_key(s):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", s)]


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"  Removed old file -> {path}")


def _load_iscell_flags(folder, n_stat):
    """
    读取 iscell.npy
    返回:
        iscell_flag: bool array, True=cell, False=non-cell
        prob: float array, classifier probability
    """
    iscell_path = os.path.join(folder, "iscell.npy")

    if not os.path.exists(iscell_path):
        print("  WARNING: iscell.npy not found, all ROIs will be treated as cell")
        return np.ones(n_stat, dtype=bool), np.full(n_stat, np.nan, dtype=float)

    iscell = np.load(iscell_path, allow_pickle=True)

    if iscell.ndim != 2 or iscell.shape[1] < 1:
        raise ValueError(f"Invalid iscell.npy shape: {iscell.shape}")

    if iscell.shape[0] != n_stat:
        raise ValueError(
            f"iscell.npy length ({iscell.shape[0]}) does not match stat.npy length ({n_stat})"
        )

    iscell_flag = iscell[:, 0].astype(bool)

    if iscell.shape[1] >= 2:
        prob = iscell[:, 1].astype(float)
    else:
        prob = np.full(n_stat, np.nan, dtype=float)

    return iscell_flag, prob


def _build_local_mask(xpix, ypix, pad=2):
    """
    根据像素点构建局部二值 mask
    返回:
        mask, xmin, ymin, pad
    """
    xpix = np.asarray(xpix, dtype=np.int32)
    ypix = np.asarray(ypix, dtype=np.int32)

    if len(xpix) == 0 or len(ypix) == 0:
        return None, None, None, None

    xmin, xmax = int(xpix.min()), int(xpix.max())
    ymin, ymax = int(ypix.min()), int(ypix.max())

    w = (xmax - xmin + 1) + 2 * pad
    h = (ymax - ymin + 1) + 2 * pad

    mask = np.zeros((h, w), dtype=np.uint8)
    xx = xpix - xmin + pad
    yy = ypix - ymin + pad
    mask[yy, xx] = 1

    return mask, xmin, ymin, pad


def _make_polygon_roi_from_coords(coords, name):
    """
    用 polygon 坐标创建 ImageJ polygon ROI
    coords: Nx2, [x, y]
    """
    coords = np.asarray(coords, dtype=np.float32)

    if coords.shape[0] < 3:
        return None

    # 去掉闭合轮廓重复终点
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    if coords.shape[0] < 3:
        return None

    roi = roifile.ImagejRoi.frompoints(coords, name=name)
    roi.roitype = ROI_TYPE.POLYGON
    return roi


def _polygon_from_pixels(xpix, ypix, name, simplify_tolerance=1.0):
    """
    把像素集合转成 Fiji/ImageJ polygon ROI
    优先:
        mask -> find_contours -> polygon
    失败则回退:
        convex hull polygon
    """
    xpix = np.asarray(xpix, dtype=np.int32)
    ypix = np.asarray(ypix, dtype=np.int32)

    if len(xpix) == 0 or len(ypix) == 0:
        return None

    mask, xmin, ymin, pad = _build_local_mask(xpix, ypix, pad=2)
    if mask is None:
        return None

    # 1) 优先取外轮廓
    try:
        contours = measure.find_contours(mask.astype(float), level=0.5)

        if len(contours) > 0:
            contour = max(contours, key=len)

            # contour: (row, col) -> (y, x)
            y = contour[:, 0] + ymin - pad
            x = contour[:, 1] + xmin - pad

            coords = np.column_stack((x, y)).astype(np.float32)

            try:
                coords = measure.approximate_polygon(coords, tolerance=simplify_tolerance)
            except Exception:
                pass

            roi = _make_polygon_roi_from_coords(coords, name=name)
            if roi is not None:
                return roi
    except Exception:
        pass

    # 2) fallback: convex hull
    try:
        from scipy.spatial import ConvexHull

        pts = np.column_stack((xpix, ypix)).astype(np.float32)
        if pts.shape[0] >= 3:
            hull = ConvexHull(pts)
            coords = pts[hull.vertices]
            roi = _make_polygon_roi_from_coords(coords, name=name)
            if roi is not None:
                return roi
    except Exception:
        pass

    return None


def _get_neuropil_pixels_from_stat(s, Lx):
    """
    同时兼容两种字段:
      - ipix_neuropil
      - neuropil_mask
    返回:
      xneu, yneu 或 (None, None)
    """
    ipix = None

    if "ipix_neuropil" in s:
        ipix = np.asarray(s["ipix_neuropil"])
    elif "neuropil_mask" in s:
        ipix = np.asarray(s["neuropil_mask"])

    if ipix is None or len(ipix) == 0:
        return None, None

    ipix = ipix.astype(np.int64)
    yneu = ipix // int(Lx)
    xneu = ipix % int(Lx)
    return xneu, yneu


def _write_zip_if_not_empty(zip_path, rois, label):
    if rois:
        roifile.roiwrite(zip_path, rois)
        print(f"  Saved {label} -> {zip_path}  (n={len(rois)})")
    else:
        print(f"  No {label} generated")


def stat_to_separate_roisets(stat_path):
    """
    输入:
      .../suite2p/planeX/stat.npy

    输出:
      RoiSet.zip            -> iscell == 1
      NonCellRoiSet.zip     -> iscell == 0
      NeuropilSet.zip       -> polygon neuropil ROI
      roi_index_mapping.csv -> suite2p 原始 index 与 ROI 名字映射表
    """
    import pandas as pd

    folder = os.path.dirname(stat_path)
    trial_folder = os.path.dirname(os.path.dirname(folder))

    cell_zip = os.path.join(trial_folder, "RoiSet.zip")
    noncell_zip = os.path.join(trial_folder, "NonCellRoiSet.zip")
    neuropil_zip = os.path.join(trial_folder, "NeuropilSet.zip")
    mapping_csv = os.path.join(trial_folder, "roi_index_mapping.csv")

    # 每次运行先删除旧文件
    _remove_if_exists(cell_zip)
    _remove_if_exists(noncell_zip)
    _remove_if_exists(neuropil_zip)
    _remove_if_exists(mapping_csv)

    stat = np.load(stat_path, allow_pickle=True)

    ops_path = os.path.join(folder, "ops.npy")
    if not os.path.exists(ops_path):
        raise FileNotFoundError(f"ops.npy not found in {folder}")

    ops = np.load(ops_path, allow_pickle=True).item()
    Lx = int(ops["Lx"])

    iscell_flag, prob = _load_iscell_flags(folder, len(stat))

    cell_rois = []
    noncell_rois = []
    neuropil_rois = []
    mapping_rows = []

    cell_count = 0
    noncell_count = 0
    neuropil_count = 0

    for i, s in enumerate(stat):
        xpix = np.asarray(s.get("xpix", []))
        ypix = np.asarray(s.get("ypix", []))

        # 先准备映射行
        row = {
            "suite2p_index": int(i),
            "iscell": int(bool(iscell_flag[i])),
            "iscell_prob": float(prob[i]) if not np.isnan(prob[i]) else np.nan,
            "cell_roi_name": "",
            "noncell_roi_name": "",
            "neuropil_roi_name": "",
            "has_main_roi_pixels": int(len(xpix) > 0 and len(ypix) > 0),
            "has_neuropil_pixels": 0,
        }

        # ===== 主 ROI =====
        if len(xpix) > 0 and len(ypix) > 0:
            if iscell_flag[i]:
                cell_count += 1
                roi_name = f"{cell_count:04d}"

                roi = _polygon_from_pixels(
                    xpix=xpix,
                    ypix=ypix,
                    name=roi_name,
                    simplify_tolerance=0.8
                )

                if roi is not None:
                    cell_rois.append(roi)
                    row["cell_roi_name"] = roi_name

            else:
                noncell_count += 1
                roi_name = f"{noncell_count:04d}"

                roi = _polygon_from_pixels(
                    xpix=xpix,
                    ypix=ypix,
                    name=roi_name,
                    simplify_tolerance=0.8
                )

                if roi is not None:
                    noncell_rois.append(roi)
                    row["noncell_roi_name"] = roi_name

        # ===== neuropil =====
        xneu, yneu = _get_neuropil_pixels_from_stat(s, Lx)
        if xneu is not None and yneu is not None and len(xneu) > 0:
            row["has_neuropil_pixels"] = 1

            neuropil_count += 1
            neu_name = f"{neuropil_count:04d}"

            roi_neu = _polygon_from_pixels(
                xpix=xneu,
                ypix=yneu,
                name=neu_name,
                simplify_tolerance=1.2
            )

            if roi_neu is not None:
                neuropil_rois.append(roi_neu)
                row["neuropil_roi_name"] = neu_name

        mapping_rows.append(row)

    _write_zip_if_not_empty(cell_zip, cell_rois, "cell RoiSet")
    _write_zip_if_not_empty(noncell_zip, noncell_rois, "non-cell RoiSet")
    _write_zip_if_not_empty(neuropil_zip, neuropil_rois, "neuropil RoiSet")

    df_map = pd.DataFrame(mapping_rows)
    df_map.to_csv(mapping_csv, index=False)
    print(f"  Saved mapping -> {mapping_csv}  (n={len(df_map)})")

    print(
        f"  Summary: cells={len(cell_rois)}, noncells={len(noncell_rois)}, neuropil={len(neuropil_rois)}"
    )


def convert_all_suite2p(root_dir):
    root_dir = os.path.abspath(root_dir)
    stat_files = []

    for r, dirs, files in os.walk(root_dir):
        dirs[:] = sorted(dirs, key=natural_key)

        if "stat.npy" in files and os.path.basename(r).startswith("plane"):
            stat_files.append(os.path.join(r, "stat.npy"))

    if not stat_files:
        print("No stat.npy found.")
        return

    print(f"Found {len(stat_files)} stat.npy files\n")

    for stat_path in sorted(stat_files, key=natural_key):
        print(f"Processing: {stat_path}")
        try:
            stat_to_separate_roisets(stat_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":

    if not os.path.exists(DATA_PATH):
        print("Root path does not exist!")
    else:
        convert_all_suite2p(DATA_PATH)