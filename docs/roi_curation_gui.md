# 人工 ROI 校对 GUI 说明

这个工具用于 suite2p 跑完之后人工检查 ROI。目标是少一点折磨，多一点可控：能看视频、能看 ROI、能手动剔除错的 ROI，也能用多边形补画 suite2p 漏掉的细胞。

当前推荐主线是：05 跑 suite2p，05e 人工校对，之后 06 读取人工结果。05b/05c 那套“学习历史手画 ROI 再自动筛选”的路线保留为可选实验功能，不再作为主线。

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

## 窗口怎么看

- 左边窗口：视频加 suite2p ROI 的原始形状。点一个 ROI，就会选中它。
- 右边窗口：干净视频。切到 `draw polygon ROI` 后，可以在这里手动画新 ROI。
- 下方滑条：拖动时间帧。
- 右侧列表：已有 suite2p ROI 列表。
- Trace 图：显示当前选中 ROI 的 `F`、`Fneu`、`F - 0.7Fneu` 和 dF/F。

## 怎么校对已有 ROI

1. 在左边视频或右侧 ROI 列表里选中一个 ROI。
2. 看它的位置和 trace。
3. 像细胞就点 `Keep selected`。
4. 不像细胞但希望保留一条人工判断记录，就点 `Reject selected`。
5. 确定不想让它进入后续人工校对结果，就点 `Delete selected`。

这个操作不会改 suite2p 原始文件，只会另存一个人工校对版。

`reject` 和 `delete` 的区别：

- `reject`：保留记录，但标记为后续不用。
- `delete`：从人工校对输出中移除。suite2p 原始文件不会被改写。

常用快捷键：

- `Delete` 或 `Backspace`：删除当前 ROI
- `X`：reject 当前 ROI
- `K`：keep 当前 ROI
- `Ctrl+Z` 或 `U`：撤销上一步 keep/reject/delete/add
- `Esc`：撤销当前多边形的最后一个点
- `Enter`：完成当前多边形 ROI

## 怎么手动画新 ROI

1. 把 `right image mode` 切成 `draw polygon ROI`。
2. 在右边视频上沿着细胞边界连续点几个点。
3. 点错可以按 `Undo polygon point`。
4. 画完点 `Finish polygon ROI`。
5. Trace 图会立刻显示这个手画 ROI 的信号。

手画 ROI 是多边形，不是圆形。新 ROI 的 neuropil 先用 ROI 周围一圈像素近似估计，所以可以马上看 `F - 0.7Fneu` 和 dF/F。

## 保存了什么

输出位置：

```text
DATA_ROOT/05e_roi_manual_curation/<trial>/
```

主要文件：

- `<trial>_iscell_manual.npy`：已有 suite2p ROI 的人工 keep/reject 结果
- `<trial>_kept_suite2p_indices.csv`：最终保留的 suite2p ROI 编号
- `<trial>_rejected_suite2p_indices.csv`：被 reject 但未 delete 的 suite2p ROI 编号
- `<trial>_deleted_suite2p_indices.csv`：从人工校对输出中删除的 suite2p ROI 编号
- `<trial>_manual_added_rois.json`：手动画的新 ROI 多边形坐标
- `<trial>_manual_added_roi_traces.csv`：手画 ROI 的 trace
- `<trial>_manual_curation_summary.json`：本次校对摘要

## 当前限制

- 这一版 05e 先负责人工校对和保存结果。
- 06 还没有默认读取 05e 的手画新 ROI；下一步可以把 05e 输出接进 06。
- 手画 ROI 的 neuropil 是近似值，用来辅助判断 trace，不等同于 suite2p 的 neuropil mask。
