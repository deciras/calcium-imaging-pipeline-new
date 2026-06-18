# 人工 ROI 校对 GUI 说明

这个工具用于人工标注/校对 ROI。目标是少一点折磨，多一点可控：能看视频、能看 ROI、能手动剔除错的 ROI，也能用 freehand 或椭圆补画 suite2p 漏掉的细胞。

当前推荐主线是：05 跑 suite2p 候选 ROI，`manual` 人工标注/校对，之后 06 和后续分析继续处理。suite2p 可以只作为参考层，用来“捡”少数可用 ROI；05b/05c/05d 那套“自动学习历史手画 ROI 再筛选”的路线已经归档，不再作为主线。

如果 suite2p 的 ROI 质量长期不稳定，manual GUI 可以直接作为主力手动标注工具。suite2p 只负责可选预标注。

## 打开方式

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --trial-id 20260428_Euprymna_retina2_25x \
  --movie-kind corrected
```

Linux 工作站上也用同一套命令。只需要把 `--data-root` 改成 Linux 上真实数据的位置。

如果不写 `--data-root`，会弹出窗口让你选择数据根目录。

GUI 会在各步骤输出文件夹下递归搜索 movie 和 suite2p 结果，所以同时支持本地测试集这种一层结构：
`03_motion_correct/<trial>/...`，也支持工作站整理后的日期结构：
`03_motion_correct/<date>/<trial>/...`。manual ROI 保存时也会保留相同的相对层级，例如：
`05e_roi_manual_curation/<date>/<trial>/...`，这样后续 06 可以自动接上。

## 窗口怎么看

- `Movie source`：选择当前查看的图像来源：
  - `raw (01 converted TIFF)`：01 转出的主成像 TIFF，文件名通常是 `*_Max_Proj.tif`
  - `motion corrected (03)`：03 运动矫正后的 movie
  - `spatial high-pass (04)`：04 空间高通滤波后的 movie
- 切换 `Movie source` 只是换当前 trial 的显示底片，不会另存一套 ROI，也不会要求保存。手画 ROI 和已选 suite2p ROI 会继续保留。
- `Display` 里的 `black` / `white` 类似 Fiji 的 brightness/contrast：只改变画面显示，不改变原始 movie，也不改变 trace。
- `Files`：显示当前数据根目录里找到的 trial。可以点 `Load selected` 打开选中的 trial，也可以点 `Next` 切到下一个。
- 右侧控制区可以上下滚动，屏幕较小时按钮不会被挤出窗口。
- 左边窗口：默认只显示视频，不显示 suite2p ROI。
- `show suite2p refs`：需要参考 suite2p 时再打开；打开时才读取 suite2p ROI。默认只显示当前 suite2p 自己标成 `iscell=1` 的 ROI 候选；勾选 `show current suite2p iscell=0 refs` 后，也会显示当前 suite2p 输出里仍然存在、但标成 `iscell=0` 的候选。未选中的参考 ROI 是较细的半透明橙色，选中的参考 ROI 是绿色，当前查看的是黄色。
- `show current suite2p iscell=0 refs`：只控制当前 suite2p 原始 `iscell=0` 候选是否显示；这只是 suite2p 给当前候选库的 label，不是人工校对结果里的状态。旧保存里有、但当前 suite2p 候选库里已经没有对应形状的 ROI 不属于这里，会作为 `source=manual` 载入。
- `show final ROIs on right`：只控制右边最终 ROI set 是否显示。关掉后，右边会只显示底片和当前正在画的线；已选 ROI 不会丢，只是暂时隐藏。
- 右边窗口：显示最终会保存的 ROI，也就是手画 ROI 和已经选中的 suite2p ROI。
- `ROI table`：统一管理最终 ROI set。`source=suite2p` 的 `id` 是 suite2p 原始 index；`source=manual` 的 `id` 是 manual ROI index，两套编号不交叉。
- `Removed this session`：临时存放本次打开 GUI 后从最终 ROI set 里移除的 ROI。这个临时仓库不写入最终 ROI 输出，也不会被 06 读取；选中后点 `Restore removed` 可以放回 `ROI table`。
- `ROI table` 和 `Removed this session` 都支持多选。可以按住 `Ctrl` 点选多行，或用 `Shift` 选择一段连续行，然后批量 `Remove selected` 或 `Restore removed`。
- 右边最终 ROI 图在 `select ROI` 模式下支持 `Shift` 拖框选择多个 final ROI；框选后会同步选中 `ROI table` 里的对应行。
- 右边窗口：切到 `draw freehand ROI` 后，可以按住鼠标或数位板笔直接圈画 ROI。
- 右边窗口：切到 `draw ellipse ROI` 后，可以拖出一个椭圆 ROI。
- 下方滑条：拖动时间帧；`Play/Pause` 可以播放/暂停，旁边的 `fps` 可以调播放速度。
- 右侧表格：最终 ROI set 和本次临时移除的 ROI。
- Trace 图：显示当前选中 ROI 的 `F`、`Fneu`、`F - 0.7Fneu` 和 dF/F。

Trace 来源：

- suite2p 候选 ROI 和手画 ROI 的 trace 都按 ROI 形状从 motion-corrected movie 重新计算；如果没有 `03_motion_correct`，才回退到当前显示底片。
- 因此 05 默认也使用 motion-corrected movie 生成 suite2p 候选；spatial high-pass 主要作为 GUI 中辅助查看边界的显示底片。
- 因此切换当前显示的 `Movie source` 或调 brightness/contrast，只影响你看图，不会改变 trace。

## 怎么从 suite2p 里捡 ROI

1. 打开 `show suite2p refs`。
2. 在左边视频里单击一个 suite2p ROI，可以查看它的 trace。
3. 双击这个 suite2p ROI，才会加入/移出最终人工 ROI set。
4. 最后只有被选中的 suite2p ROI 会保存；其他 suite2p ROI 都不会进入后续分析。

这个操作不会改 suite2p 原始文件，只会另存一个新的人工 ROI set。

选中的 suite2p ROI 会出现在右边窗口，即使左边的 `show suite2p refs` 关掉也会保留显示。右边也可以点选这些 ROI，右键可以从最终 ROI set 里移除。右边手画的 ROI 也可以右键直接删除；右键会直接处理点到的 ROI，不需要先选中，也不需要先切回 `select ROI` 模式。

如果关掉 `show final ROIs on right`，右边已有 ROI 会暂时隐藏，鼠标也不会点中这些隐藏 ROI，避免误删。

最终 ROI set 里的 ROI 一视同仁，只有来源不同：

- `source=suite2p`：来自 suite2p 候选，保留 suite2p 原始 index。
- `source=manual`：手画 ROI，或 suite2p 重跑后旧保存 ROI 被导入为 manual，保留 manual ROI index。
- `Remove selected` / `Delete selected`：把当前选中的一个或多个 ROI 从最终 ROI set 移除，不修改 suite2p 原始输出。
- `Restore removed`：把本次临时移除的一个或多个 ROI 放回最终 ROI set。
- `Undo`：撤销最近一次添加、移除或选择状态变化。

保存后，`suite2p_compatible/plane0/` 里的 ROI 会作为一个统一的 final ROI set 输出。GUI 会按空间位置给 final ROI 重新排序：先按 `y_mean`，再按 `x_mean`，最后按 ROI 面积。06 之后的 `roi_id=1..N` 来自这个统一顺序，不再表示 suite2p 原始编号或 manual 绘制顺序。

后续 06-16 对所有 final ROI 使用同一套计算规则：同一个 trace movie、同一种 neuropil mask、同一种 dF/F 和 feature 计算。`roi_source`、`roi_type`、`manual_roi_id`、`suite2p_original_id`、`previous_suite2p_original_id` 只作为查询和追溯字段保留，不参与 ROI 选择、trace 提取、dF/F、响应分析或聚类。

常用快捷键：

- `Delete` 或 `Backspace`：把当前选中的 ROI 从最终 ROI set 移除
- `X`：把当前选中的 ROI 从最终 ROI set 移除
- `K`：把当前 suite2p ref 捡入最终 ROI set
- `Space`：播放/暂停
- `Ctrl+Z` 或 `U`：撤销上一步添加、移除或选择变化
- `S`：显示/隐藏 suite2p 参考层
- `方向键`：有 ROI 被选中时，跳到上下左右方向最近的 ROI；没有选中 ROI 时，左右键逐帧前后移动
- `Esc`：撤销当前手绘线最后一段
- `Enter`：完成当前手绘/椭圆 ROI

播放不会因为你正在画 ROI 或点选 ROI 就自动停下。点 `Pause`、按空格，或者开始拖动时间滑条时会暂停。播放时是一帧一帧前进并循环到开头；如果电脑一时忙不过来，它会变慢，而不是跳过中间帧。

切换 trial 或退出窗口前，如果当前 ROI 修改还没有保存，会提示是否保存。

如果这个 trial 之前已经保存过人工校对结果，再次打开时会自动读回上次保存的校对状态，包括已选的 suite2p ROI、手画 ROI 和已移除的参考 ROI。也就是说可以分多次慢慢校对，不需要一次做完。

人工校对结果按 trial 保存，不按 `Movie source` 分开保存。`raw`、`motion corrected` 和 `spatial high-pass` 只是不同查看方式，最终指向同一套人工 ROI 校对结果。

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

手画 ROI 是 freehand 轨迹或椭圆，不是圆形。freehand 和椭圆的预览线都用细线显示。新 ROI 只把形状和位置当作人工结果；trace、Fneu 和 dF/F 会重新计算。当前 Fneu 用 ROI 周围一圈像素近似估计，所以可以马上看 `F - 0.7Fneu` 和 dF/F。

## 保存了什么

输出位置仍然沿用旧文件夹名，避免破坏之前保存过的人工校对结果：

```text
DATA_ROOT/05e_roi_manual_curation/<trial>/
```

主要文件：

- `<trial>_manual_roi_set.json`：新的 ROI set，只包含手画 ROI 和被选中的 suite2p 参考 ROI
- `<trial>_iscell_manual.npy`：被选中的 suite2p 参考 ROI 标记
- `<trial>_selected_suite2p_indices.csv`：最终被捡出来的 suite2p ROI 编号
- `<trial>_deleted_suite2p_indices.csv`：兼容旧保存格式，当前人工结果不再依赖这个文件
- `<trial>_manual_added_rois.json`：手动画的新 ROI 坐标
- `<trial>_manual_added_roi_traces.csv`：手画 ROI 的 trace
- `suite2p_compatible/plane0/stat.npy`：兼容 suite2p 风格的 ROI 文件
- `suite2p_compatible/plane0/iscell.npy`：兼容 suite2p 风格的 ROI 标记
- `suite2p_compatible/plane0/F.npy` / `Fneu.npy`：如果 trace 长度一致，会一起导出；这些 trace 按 ROI 形状从 motion-corrected movie 重新计算
- `suite2p_compatible/plane0/stat.npy` 中每个 ROI 会带来源 metadata；真正手画或旧 suite2p 导入的 ROI 进入 06 后 `suite2p_original_id=-1`，旧 suite2p 编号只保存在 `previous_suite2p_original_id`
- `RoiSet.zip`：兼容 Fiji/ImageJ 的 ROI set
- `<trial>_manual_curation_summary.json`：本次校对摘要

重新打开同一个 trial 时，GUI 会优先读取这些已保存结果，方便继续调整。

如果 05 重新跑过 suite2p，导致旧保存里的 suite2p ROI 编号/形状和当前 suite2p 候选库对不上，GUI 会把旧保存的 suite2p ROI 形状导入为 manual ROI，避免之前校对过的 ROI 因候选库更新而丢失。这些 ROI 不会显示在 `show current suite2p iscell=0 refs` 里，因为它们已经不是当前 suite2p 候选库的一部分。

## 当前限制

- 这一版 manual GUI 先负责人工校对和保存结果。
- 06 默认会优先读取 manual GUI 保存的 `suite2p_compatible/plane0/`；没有 manual 结果的 trial 才回退到 05 suite2p。
- 手画 ROI 的 neuropil 是近似值，用来辅助判断 trace，不等同于 suite2p 的完整 neuropil mask；旧校对结果重新加载时只保留 ROI 形状和位置，trace / Fneu / dF/F 会重新计算。
