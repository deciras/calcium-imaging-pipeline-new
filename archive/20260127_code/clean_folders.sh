# 1. 先把所有的特定文件夹移入废纸篓
find . -type d \( \
    -name "Control_Traces" \
    -o -name "Stimulated_Traces" \
    -o -name "Retina_Malus_Analysis" \
    -o -name "Slicing_Analysis*" \
    -o -name "All_Trial_Videos" \
    -o -name "ROI_Visual_Analysis" \
    -o -name "All_Traces" \
    -o -name "Analysis_Plots" \
    -o -name "Traces_Orig" \
    -o -name "Traces_BG_Correction" \
    -o -name "Traces_Neuropil_Corrected" \
    -o -name "ROI_Long_Traces" \
    -o -name "ROI_Full_Traces" \
    \) -exec trash {} +



# 2. 再清理文件夹以外的多余文件（保留你指定的那些）
find . -type f \
    ! -name "Overlay Elements of *_Reg_full.csv" \
    ! -name "stim_map.csv" \
    ! -name "Results.csv" \
    ! -name "RoiSet.zip" \
    ! -name "*.tif" \
    ! -name "*.json" \
    ! -name ".DS_Store" \
    ! -name "._*" \
    -exec trash {} +

