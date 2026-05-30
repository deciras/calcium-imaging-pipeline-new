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

## Step 05c：ROI 质量筛选

05c 会读取 05 的 suite2p ROI，再用 05b 从历史手画 ROI 得到的形状范围做筛选。它不会修改 05 的原始 suite2p 输出，只会生成新的 curated 文件夹。

安全运行：

```bash
python3 current/run_pipeline.py \
  --steps 05c \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

确认要重跑 05c 时：

```bash
python3 current/run_pipeline.py \
  --steps 05c \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite
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
- `<trial>_roi_quality_summary.json`

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
