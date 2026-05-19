#!/bin/bash

# 这个文件的作用是把所有的tif从plane0文件夹里提出来，并且删掉空的plane0文件夹

# 设置包含多个 Processed_TIF_YYYYMMDD 文件夹的父目录路径
BASE_DIR="/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2025_olympus/Processed_TIF_avg_proj_focus_Suite2p_Results"  # 替换为你的父目录路径

# 遍历所有 Processed_TIF_YYYYMMDD 文件夹
for dir in "$BASE_DIR"/Processed_TIF_*; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        
        # 在每个 Processed_TIF_YYYYMMDD 文件夹中，找到所有 Plane0 文件夹
        find "$dir" -type d -name "Plane0" | while read plane_dir; do
            echo "Processing Plane0 folder: $plane_dir"
            
            # 移动所有 .tif 文件到父目录
            find "$plane_dir" -type f -name "*.tif" -exec mv {} "$plane_dir"/.. \;
            
            # 删除空的 Plane0 文件夹
            rmdir "$plane_dir"
        done
    fi
done

echo "Processing completed."

# /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/20260106_code/3_move_tif.sh