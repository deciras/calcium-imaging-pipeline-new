# run_pipeline.py 命令行调用说明

这份文件记录如何从命令行运行当前维护版 pipeline。日常建议都从总入口运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 02 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

## 基本格式

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 步骤编号 \
  --data-root 数据文件夹 \
  --action skip
```

常用参数：

- `--steps`：选择要运行的步骤，例如 `00`、`01`、`02`、`03`、`04`，也可以写 `01,02`
- `--data-root`：数据总文件夹，例如 `test_dataset`
- `--action skip`：默认安全模式，已有结果就跳过
- `--action overwrite`：重新生成该步骤输出，只在确认要重跑时使用
- `--dry-run`：只显示将要运行的命令，不真正启动步骤
- `--step-dry-run`：启动子步骤自己的 dry-run，不真正处理数据
- `--output-root`：指定输出总文件夹；不写时默认等于 `data-root`

## 查看会运行什么

```bash
python3 current/run_pipeline.py \
  --steps 04 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --dry-run
```

## Step 00：整理原始 OIR 文件

预览：

```bash
python3 current/run_pipeline.py \
  --steps 00 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --step-dry-run
```

执行：

```bash
python3 current/run_pipeline.py \
  --steps 00 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

## Step 01：Fiji / Bio-Formats 转 TIFF

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 01 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --stim-export-mode both \
  --projection-mode auto \
  --fiji-memory 12g
```

确认要重转 01 时：

```bash
python3 current/run_pipeline.py \
  --steps 01 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --stim-export-mode both \
  --projection-mode auto \
  --fiji-memory 12g
```

说明：

- `--stim-export-mode both`：同时导出 projected stim analog 和 raw stim analog 文件夹
- `--projection-mode auto`：自动平衡速度和内存
- `--fiji-memory 12g`：给 Fiji 的 Java 内存；Linux 工作站内存更大时可用 `32g`

## Step 02：提取刺激时间

推荐运行：

```bash
python3 current/run_pipeline.py \
  --steps 02 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --analog-source auto \
  --raw-z-strategy planes
```

说明：

- `--analog-source auto`：优先读取 `_Stim_Analog_Raw/`，没有 raw 时回退到 `_Stim_Analog.tif`
- `--raw-z-strategy planes`：把 raw z 平面展开成更密的时间轴，适合短 pulse
- 图会同时输出 PNG 和可编辑 PDF

## Step 03：运动校正

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 03 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

确认要重跑运动校正时：

```bash
python3 current/run_pipeline.py \
  --steps 03 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite
```

说明：03 很耗时，平时优先用 `skip`。

## Step 04：空间高通滤波

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 04 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --sigma-px 12 \
  --dpi 150
```

确认要重跑 04 时：

```bash
python3 current/run_pipeline.py \
  --steps 04 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --sigma-px 12 \
  --dpi 150
```

可选参数：

- `--sigma-px 12`：空间背景的高斯模糊尺度，越大越保留大尺度背景；默认 12
- `--clip-negative`：把高通后的负值裁成 0；不加这个参数则保留负值
- `--dpi 150`：preview 图分辨率；默认 150

04 输出到：

```text
DATA_ROOT/04_spatial_highpass/
```

每个 trial 会生成：

- `*_spatial_highpass_movie.tif`
- `preview_before_after.png`
- `preview_before_after.pdf`
- `spatial_highpass_summary.json`
- metadata 和刺激 CSV 表格

## Step 05：suite2p ROI 检测

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 05 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

使用更保守的 ROI 参数运行：

```bash
python3 current/run_pipeline.py \
  --steps 05 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --cell-diameter-um 5 \
  --diameter-scale 1.2 \
  --threshold-scaling 1.4 \
  --suite2p-threads 4 \
  --n-workers 1 \
  --overlay-dpi 150
```

确认要用新参数重跑 05 时：

```bash
python3 current/run_pipeline.py \
  --steps 05 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --cell-diameter-um 5 \
  --diameter-scale 1.2 \
  --threshold-scaling 1.4 \
  --suite2p-threads 4 \
  --n-workers 1 \
  --overlay-dpi 150
```

说明：

- 05 默认优先读取 `04_spatial_highpass/`，没有 04 时才读取 `03_motion_correct/`
- `--threshold-scaling` 越大，ROI 检测越保守，通常 ROI 会更少
- `--cell-diameter-um 5` 使用当前估计的细胞直径
- `--diameter-scale 1.2` 会让 suite2p 的检测直径略大于按像素尺寸换算出的 5 um
- `--n-workers 1` 表示一次只跑一个 trial，最稳、最省内存
- `--suite2p-threads 4` 控制 suite2p 内部线程数；Linux 工作站可根据 CPU/内存适当提高
- overlay 图会同时输出 PNG 和可编辑 PDF

05 输出到：

```text
DATA_ROOT/05_suite2p_roi_detection/
```

重要输出：

- `suite2p/plane0/stat.npy`
- `suite2p/plane0/ops.npy`
- `suite2p/plane0/F.npy`
- `suite2p/plane0/Fneu.npy`
- `suite2p/plane0/iscell.npy`
- `benchmark/suite2p/roi_summary.csv`
- `benchmark/suite2p/roi_overlay.png`
- `benchmark/suite2p/roi_overlay.pdf`
- `suite2p_trial_summary.json`

## Step 05b：从手画 ROI 建立 prior（可选实验功能）

05b 会读取历史手画 ROI 文件夹里的 `RoiSet.zip`。如果同一文件夹里有 ImageJ/Fiji 导出的 `Results.csv` 和 `Overlay Elements*.csv`，它还会把手画 ROI 的 trace 特征一起学进去。

现在主线更倾向于用 05e 做人工校对。05b 保留为可选实验功能，不再是必须步骤。

推荐运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 05b \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --manual-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/20250901 \
  --action overwrite \
  --manual-trace-mode summary
```

说明：

- `--manual-root`：手画 ROI 数据所在文件夹
- `--manual-trace-mode summary`：只保存 trace 摘要，文件较小，日常推荐
- `--manual-trace-mode full`：额外保存每个 ROI 的逐帧 trace，文件会更大
- 05b 不会改动原始手画 ROI 文件夹

主要输出：

- `manual_roi_prior.json`
- `manual_roi_table.csv`
- `manual_roi_shape_summary.csv`
- `manual_roi_trace_summary.csv`
- shape / trace 分布图 PNG/PDF

## Step 05c：ROI 质量筛选（可选实验功能）

05c 会读取 05 的 suite2p ROI，再用 05b 从历史手画 ROI 得到的形状和 trace 信息做质控。它不会修改 05 的原始 suite2p 输出，只会生成新的 curated 文件夹。

现在主线更倾向于人工校对：05 跑 suite2p，05e 打开 GUI 校对，之后 06 读取 05e 结果。05c 保留为可选实验功能。

重要：05c 默认不相信 suite2p 的 `iscell` 分类。也就是说，suite2p 说某个 ROI 不是 cell，05c 仍然可以把它救回来。只有手动加 `--require-suite2p-iscell` 时，才会要求 suite2p 也认为它是 cell。

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 05c \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --trace-prior-mode trace-report
```

确认要重跑 05c 时：

```bash
python3 current/run_pipeline.py \
  --steps 05c \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --trace-prior-mode trace-report \
  --trace-source raw
```

更保守但可能漏掉真实细胞的运行方式：

```bash
python3 current/run_pipeline.py \
  --steps 05c \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --require-suite2p-iscell
```

常用参数：

- `--min-quality-score 0.75`：默认要求 4 个形状检查中至少 3 个通过
- `--rule-padding-fraction 0`：默认不放宽手画 ROI 的 p05-p95 范围
- `--trace-prior-mode trace-report`：默认推荐；计算 trace 质量，但不直接用 trace 硬过滤
- `--trace-prior-mode shape-and-trace`：严格模式；形状和 trace 一起决定是否保留，可能漏掉真实细胞，需谨慎
- `--trace-source raw`：默认用 suite2p 的 `F.npy`，更接近 ImageJ 手画 ROI 的 `Results.csv`
- `--trace-source neuropil-corrected`：使用 `F - 0.7 * Fneu` 计算 trace 质量
- `--require-suite2p-iscell`：只保留 suite2p 原本也判为 cell 的 ROI；更保守，但不会救回 suite2p 漏判的 ROI

05c 输出到：

```text
DATA_ROOT/05c_roi_quality_filter/
```

重要输出：

- `roi_quality_summary.csv`
- `<trial>_roi_quality_table.csv`
- `<trial>_accepted_suite2p_indices.csv`
- `<trial>_rejected_suite2p_indices.csv`
- `<trial>_iscell_curated.npy`
- `<trial>_roi_quality_overlay.png`
- `<trial>_roi_quality_overlay.pdf`
- `<trial>_roi_quality_trace_qc.png`
- `<trial>_roi_quality_trace_qc.pdf`
- `<trial>_roi_quality_ranked_candidates.csv`
- `<trial>_roi_quality_summary.json`

## Step 05d：寻找 suite2p 漏画的 ROI 候选

05d 解决的是另一个问题：如果 suite2p 根本没有画出某个肉眼可见细胞，05c 没法从不存在的 ROI 里把它救回来。05d 会独立看图像本身，提出一批“可能是漏掉的 ROI”的候选。

推荐运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 05d \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite
```

更明确地指定参数：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 05d \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --step-args '--image-source max_proj --peak-z-threshold 4.0 --max-suite2p-overlap 0.25 --max-candidates 800 --extract-traces'
```

说明：

- 05d 不使用 suite2p 的 `iscell` 分类
- 05d 会参考 suite2p 已经画出的 ROI 位置，用来标记哪些候选是“新的/可能漏掉的”
- `--peak-z-threshold` 越高，候选越少、越保守
- `--max-suite2p-overlap 0.25` 表示和已有 suite2p ROI 重叠超过 25% 的候选不算 novel
- 这一版输出的是候选列表，不会自动替代 05c 进入 06

主要输出：

- `<trial>_independent_roi_candidates.csv`
- `<trial>_independent_roi_candidates_summary.json`
- `<trial>_independent_roi_candidate_label_map.tif`
- `<trial>_independent_roi_candidate_overlay.png`
- `<trial>_independent_roi_candidate_overlay.pdf`

## Step 05e：打开人工 ROI 校对 GUI

05e 是一个交互式窗口，不是批量计算步骤。它用来在 suite2p 之后人工检查 ROI：左边看电影和可选的 suite2p 参考 ROI，右边看最终会保存的人工 ROI set；可以把已有 ROI 标成 keep/reject，也可以在右边用 freehand 或椭圆画新的 ROI。

打开一个 trial：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 05e \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --trial-id 20260428_Euprymna_retina2_25x \
  --movie-kind corrected
```

Linux 工作站也是同样格式，只需要把 `--data-root` 改成工作站上的数据路径。

窗口里：

- `Movie source`：可以在 raw / motion corrected / spatial high-pass 之间切换
- `Files`：显示当前数据根目录里找到的 trial；可用 `Load selected` 或 `Next` 切换
- 右侧控制区可以上下滚动，屏幕较小时按钮不会被挤出窗口
- 左边：默认只显示视频，不显示 suite2p ROI
- `show suite2p refs`：需要参考 suite2p 时再打开；打开时才读取 suite2p ROI；默认只显示 suite2p 原本 accepted 的 ROI，勾选 `show rejected` 才显示 rejected ROI
- suite2p 未选中的参考 ROI 是较细的半透明橙色，选中的参考 ROI 是绿色
- 左边单击 suite2p ROI：只查看 trace
- 左边双击 suite2p ROI：加入/移出最终人工 ROI set
- 右边显示最终会保存的 ROI；可以在右边点选手画 ROI 或已选 suite2p ROI，右键直接删除/移除
- 右边：切到 `draw freehand ROI` 后，可以按住鼠标或数位板笔直接圈画，松开后自动生成 ROI
- 右边：切到 `draw ellipse ROI` 后，可以拖出一个椭圆 ROI
- 下方 `Play/Pause`：播放或暂停 movie；旁边 `fps` 可以调播放速度
- `Keep selected` / `Reject selected`：保留或标记不用当前 ROI
- `Delete selected`：从人工校对输出里删除当前 ROI
- Trace 面板：显示当前 ROI 的 `F`、`Fneu`、`F - 0.7Fneu` 和 dF/F
- `Save manual curation`：保存结果，但不会改写 suite2p 原始输出

快捷键：

- `Delete` 或 `Backspace`：删除当前 ROI
- `X`：reject 当前 ROI
- `K`：keep 当前 ROI
- `Space`：播放/暂停
- `Ctrl+Z` 或 `U`：撤销上一步 keep/reject/delete/add
- `S`：显示/隐藏 suite2p 参考层
- `方向键`：有 ROI 被选中时跳到最近 ROI；没有选中 ROI 时，左右键逐帧前后移动
- `Esc`：撤销当前手绘线最后一段
- `Enter`：完成当前手绘/椭圆 ROI；freehand 模式通常松开鼠标/笔就会自动完成

播放状态下可以继续点选、删除、reject 或手画 ROI。点 `Pause`、按空格，或者开始拖动时间滑条时会暂停。GUI 会按帧前进并循环播放；如果电脑处理不过来，会播放得慢一点，但不会主动跳过中间帧。

05e 输出到：

```text
DATA_ROOT/05e_roi_manual_curation/<trial>/
```

主要输出：

- `<trial>_manual_roi_set.json`
- `<trial>_iscell_manual.npy`
- `<trial>_selected_suite2p_indices.csv`
- `<trial>_deleted_suite2p_indices.csv`
- `<trial>_manual_added_rois.json`
- `<trial>_manual_added_roi_traces.csv`
- `suite2p_compatible/plane0/stat.npy`
- `suite2p_compatible/plane0/iscell.npy`
- `RoiSet.zip`
- `<trial>_manual_curation_summary.json`

注意：

- `--movie-kind raw` 打开 01 转出的 `*_Max_Proj.tif`
- `--movie-kind corrected` 打开 03 运动矫正后的 `*_corrected_movie.tif`
- `--movie-kind spatial-highpass` 打开 04 空间高通滤波后的 `*_spatial_highpass_movie.tif`
- 手动画的新 ROI 现在支持 freehand 和椭圆
- freehand 模式线条更细，适合数位板
- suite2p ROI 显示为原本的像素边界，不再简化成圆圈
- suite2p ROI 默认不显示，需要时才作为参考层打开；最后只保存被选中的 suite2p ROI
- 切换 trial 或退出前，如果有未保存修改，会提示是否保存
- `reject` 表示保留记录但后续不用；`delete` 表示从人工 ROI set 中移除
- 手画 ROI 的 neuropil 是用 ROI 周围一圈像素近似估计的，适合人工判断 trace 是否像细胞
- 播放控制已经接入 GUI，可以边播放 movie 边校对 ROI
- 这一版 05e 先负责“校对和保存”，还没有自动接入 06；后续可以让 06 优先读取 05e 的人工校对结果
- 05e 可以逐步发展成主力手动 ROI 标注工具；suite2p 可以只作为可选预标注

## Step 06：提取 dF/F

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 06 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --neuropil-coeff 0.7 \
  --f0-mode percentile \
  --f0-percentile 10
```

预演但不生成输出：

```bash
python3 current/run_pipeline.py \
  --steps 06 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --neuropil-coeff 0.7 \
  --f0-mode percentile \
  --f0-percentile 10 \
  --step-dry-run
```

确认要重做 06 时：

```bash
python3 current/run_pipeline.py \
  --steps 06 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --neuropil-coeff 0.7 \
  --f0-mode percentile \
  --f0-percentile 10
```

说明：

- 06 使用 suite2p 的 `Fneu.npy` 做 neuropil correction
- 默认公式为 `F_corrected = F - 0.7 * Fneu`
- 默认 `--roi-source auto`：有 05c curated ROI 时优先用 05c，没有 05c 时回退到 suite2p `iscell.npy`
- 如果需要保留 non-cell ROI，可以通过单步脚本或 `--step-args '--include-noncell'` 传入
- 图会同时输出 PNG 和可编辑 PDF

## Step 07：检测 calcium events

预演但不生成输出：

```bash
python3 current/run_pipeline.py \
  --steps 07 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --event-method robust-threshold \
  --event-threshold-sigma 3.0 \
  --step-dry-run
```

正式运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 07 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --event-method robust-threshold \
  --event-threshold-sigma 4.0
```

确认要重做 07 时：

```bash
python3 current/run_pipeline.py \
  --steps 07 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite \
  --event-method robust-threshold \
  --event-threshold-sigma 3.0
```

说明：

- 07 检测的是 calcium events，不等同于 electrophysiological spikes
- 默认方法是 `robust-threshold`
- `--event-threshold-sigma` 越大，event 检测越保守
- 也可以使用 `--event-method find-peaks`
- raster 和 example 图会同时输出 PNG 和可编辑 PDF

## Step 08：刺激响应分析

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 08 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --baseline-sec 5 \
  --response-sec 10
```

输出到：

```text
DATA_ROOT/08_stim_response/
```

主要输出：

- `<trial>_stim_response_table.csv`
- `<trial>_roi_response_summary.csv`
- `<trial>_peri_stimulus_tensor.npy`
- `<trial>_psth_by_roi.csv`
- `<trial>_response_type_map.csv`
- PNG/PDF QC 图

## Step 09：角度调谐分析

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 09 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --angle-period 180
```

输出到：

```text
DATA_ROOT/09_angle_tuning/
```

主要输出：

- `<trial>_angle_response_table.csv`
- `<trial>_angle_tuning_summary.csv`
- `<trial>_preferred_angle_by_roi.csv`
- angle heatmap / polar plot / OSI distribution 的 PNG/PDF

## Step 10：population feature matrix

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 10 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

输出到：

```text
DATA_ROOT/10_population_features/
```

主要输出：

- `<trial>_roi_feature_matrix.csv`
- `<trial>_roi_feature_matrix_zscored.csv`
- `<trial>_trace_matrix.npy`
- `<trial>_response_matrix.npy`
- `<trial>_angle_response_matrix.npy`
- `<trial>_feature_description.json`

## Step 11：population similarity

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 11 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --similarity-source features \
  --min-corr 0.3 \
  --knn 10
```

主要输出：

- `<trial>_feature_similarity_matrix.npy`
- `<trial>_trace_correlation_matrix.npy`
- `<trial>_response_correlation_matrix.npy`
- `<trial>_distance_matrix.npy`
- `<trial>_similarity_edges.csv`
- similarity heatmap PNG/PDF

## Step 12：hierarchical clustering

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 12 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --n-clusters 6
```

主要输出：

- `<trial>_hierarchical_cluster_labels.csv`
- `<trial>_hierarchical_cluster_summary.csv`
- dendrogram PNG/PDF
- clustered heatmap PNG/PDF
- cluster mean traces PNG/PDF

## Step 14：PCA 降维

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 14 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

主要输出：

- `<trial>_pca_embedding.csv`
- `<trial>_pca_variance.csv`
- `<trial>_embedding_summary.json`
- PCA plot PNG/PDF

## Step 13：Leiden community detection

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 13 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --leiden-resolution 1.0
```

说明：

- 13 依赖 optional packages：`igraph` 和 `leidenalg`
- 如果没装，会写出 skipped summary，但不会让 pipeline 失败

主要输出：

- `<trial>_leiden_labels.csv`
- `<trial>_graph_edges.csv`
- `<trial>_leiden_community_summary.csv`
- `<trial>_leiden_summary.json`

## Step 15：跨 trial 汇总

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 15 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

主要输出：

- `all_trials_roi_summary.csv`
- `all_trials_event_summary.csv`
- `all_trials_stim_response_summary.csv`
- `all_trials_angle_tuning_summary.csv`
- `cross_trial_qc_summary.csv`
- `cross_trial_summary.json`
- cross-trial QC PNG/PDF
- `all_trials_roi_features.csv`
- `all_trials_event_table.csv`
- `all_trials_stim_response_table.csv`
- `all_trials_angle_response_table.csv`
- `top_responsive_rois.csv`
- `top_event_rois.csv`
- `top_angle_selective_rois.csv`

## Step 16：HTML report

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 16 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

主要输出：

- `DATA_ROOT/16_reports/global_report.html`
- `DATA_ROOT/16_reports/<trial>_report.html`

`global_report.html` 里包含：

- cross-trial QC 表
- top responsive ROI 预览
- top event ROI 预览
- top angle ROI 预览
- 15 导出的 CSV 表链接

## 连续运行多个步骤

例如运行 02 到 04：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 02,03,04 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --analog-source auto \
  --raw-z-strategy planes \
  --sigma-px 12
```

更建议大数据时一步一步运行，方便检查每一步输出。

## 最常用的安全习惯

先看命令，不运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 04 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --dry-run
```

已有结果不重做：

```bash
--action skip
```

确认要重做当前步骤：

```bash
--action overwrite
```

不要在不确定时用 `overwrite`。
