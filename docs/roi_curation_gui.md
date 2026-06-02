# 人工 ROI 校对 GUI 说明

这个工具用于人工标注/校对 ROI。目标是少一点折磨，多一点可控：能看视频、能看 ROI、能手动剔除错的 ROI，也能用 freehand 或椭圆补画 suite2p 漏掉的细胞。

当前推荐主线是：05e 人工标注/校对，之后 06 读取人工结果。suite2p 可以只作为可选参考层，用来“捡”少数可用 ROI；05b/05c 那套“学习历史手画 ROI 再自动筛选”的路线保留为可选实验功能，不再作为主线。

如果 suite2p 的 ROI 质量长期不稳定，05e 可以直接作为主力手动标注工具。suite2p 只负责可选预标注，甚至可以完全跳过。

## 打开方式

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 05e \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --trial-id 20260428_Euprymna_retina2_25x \
  --movie-kind corrected
```

Linux 工作站上也用同一套命令。只需要把 `--data-root` 改成 Linux 上真实数据的位置。

如果不写 `--data-root`，会弹出窗口让你选择数据根目录。

## 窗口怎么看

- `Movie source`：选择当前查看的图像来源：
  - `raw (01 converted TIFF)`：01 转出的主成像 TIFF，文件名通常是 `*_Max_Proj.tif`
  - `motion corrected (03)`：03 运动矫正后的 movie
  - `spatial high-pass (04)`：04 空间高通滤波后的 movie
- `Files`：显示当前数据根目录里找到的 trial。可以点 `Load selected` 打开选中的 trial，也可以点 `Next` 切到下一个。
- 右侧控制区可以上下滚动，屏幕较小时按钮不会被挤出窗口。
- 左边窗口：默认只显示视频，不显示 suite2p ROI。
- `show suite2p refs`：需要参考 suite2p 时再打开；打开时才读取 suite2p ROI。默认只显示 suite2p 原本 accepted 的 ROI；勾选 `show rejected` 才显示 rejected ROI。未选中的参考 ROI 是较细的半透明橙色，选中的参考 ROI 是绿色，当前查看的是黄色。
- 右边窗口：显示最终会保存的 ROI，也就是手画 ROI 和已经选中的 suite2p ROI。
- 右边窗口：切到 `draw freehand ROI` 后，可以按住鼠标或数位板笔直接圈画 ROI。
- 右边窗口：切到 `draw ellipse ROI` 后，可以拖出一个椭圆 ROI。
- 下方滑条：拖动时间帧。
- 右侧列表：已有 suite2p ROI 列表。
- Trace 图：显示当前选中 ROI 的 `F`、`Fneu`、`F - 0.7Fneu` 和 dF/F。

## 怎么从 suite2p 里捡 ROI

1. 打开 `show suite2p refs`。
2. 在左边视频里单击一个 suite2p ROI，可以查看它的 trace。
3. 双击这个 suite2p ROI，才会加入/移出最终人工 ROI set。
4. 最后只有被选中的 suite2p ROI 会保存；其他 suite2p ROI 都不会进入后续分析。

这个操作不会改 suite2p 原始文件，只会另存一个新的人工 ROI set。

选中的 suite2p ROI 会出现在右边窗口。右边也可以点选这些 ROI，右键可以从最终 ROI set 里移除。

`reject` 和 `delete` 主要用于手动画 ROI：

- `reject`：保留记录，但标记为后续不用。
- `delete`：从人工 ROI set 中移除。

常用快捷键：

- `Delete` 或 `Backspace`：删除当前 ROI
- `X`：reject 当前 ROI
- `K`：keep 当前 ROI
- `Ctrl+Z` 或 `U`：撤销上一步 keep/reject/delete/add
- `S`：显示/隐藏 suite2p 参考层
- `方向键`：跳到上下左右方向最近的 ROI
- `Esc`：撤销当前手绘线最后一段
- `Enter`：完成当前手绘/椭圆 ROI

切换 trial 或退出窗口前，如果当前 ROI 修改还没有保存，会提示是否保存。

## 怎么手动画新 ROI

自由手绘：

1. 把 `right image mode` 切成 `draw freehand ROI`。
2. 在右边视频上按住鼠标或数位板笔，沿着细胞边缘圈一圈。
3. 松开后自动生成 ROI。
4. Trace 图会立刻显示这个手画 ROI 的信号。

椭圆：

1. 把 `right image mode` 切成 `draw ellipse ROI`。
2. 在右边视频上按住并拖出椭圆。
3. 松开后自动生成 ROI。
4. Trace 图会立刻显示这个椭圆 ROI 的信号。

手画 ROI 是 freehand 轨迹或椭圆，不是圆形。新 ROI 的 neuropil 先用 ROI 周围一圈像素近似估计，所以可以马上看 `F - 0.7Fneu` 和 dF/F。

## 保存了什么

输出位置：

```text
DATA_ROOT/05e_roi_manual_curation/<trial>/
```

主要文件：

- `<trial>_manual_roi_set.json`：新的 ROI set，只包含手画 ROI 和被选中的 suite2p 参考 ROI
- `<trial>_iscell_manual.npy`：被选中的 suite2p 参考 ROI 标记
- `<trial>_selected_suite2p_indices.csv`：最终被捡出来的 suite2p ROI 编号
- `<trial>_deleted_suite2p_indices.csv`：从人工校对输出中删除的 suite2p ROI 编号
- `<trial>_manual_added_rois.json`：手动画的新 ROI 坐标
- `<trial>_manual_added_roi_traces.csv`：手画 ROI 的 trace
- `suite2p_compatible/plane0/stat.npy`：兼容 suite2p 风格的 ROI 文件
- `suite2p_compatible/plane0/iscell.npy`：兼容 suite2p 风格的 ROI 标记
- `suite2p_compatible/plane0/F.npy` / `Fneu.npy`：如果 trace 长度一致，会一起导出
- `RoiSet.zip`：兼容 Fiji/ImageJ 的 ROI set
- `<trial>_manual_curation_summary.json`：本次校对摘要

## 当前限制

- 这一版 05e 先负责人工校对和保存结果。
- 06 还没有默认读取 05e 的手画新 ROI；下一步可以把 05e 输出接进 06。
- 手画 ROI 的 neuropil 是近似值，用来辅助判断 trace，不等同于 suite2p 的 neuropil mask。
