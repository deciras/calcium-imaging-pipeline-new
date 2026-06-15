# Calcium Imaging Pipeline

这是一个给头足类视网膜钙成像数据用的处理流程。

简单说，它做的事情是：

1. 整理 Olympus `.oir` 原始文件
2. 用 Fiji / Bio-Formats 把 `.oir` 转成 TIFF
3. 从刺激通道和刺激记录里整理刺激时间
4. 做运动校正
5. 用 suite2p 生成 ROI 候选
6. 用人工 GUI 校对 ROI，也可以手画补 ROI
7. 从 motion-corrected movie 重新提取 dF/F
8. 检测 calcium events
9. 分析刺激响应、角度调谐、群体特征、聚类和降维
10. 最后生成跨 trial 汇总和 HTML 报告

这个仓库只放代码，不放真实实验数据。

真实的 `.oir`、`.tif`、`.npy`、suite2p 输出、大 movie 和中间结果都不要提交到 GitHub。

## 现在怎么用

日常建议都从总入口运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 06,07,08 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

这里最重要的是三件事：

- `cd ...`：先进入代码仓库
- `--steps`：选择要跑哪几步
- `--data-root`：告诉程序实验数据在哪里

## 最安全的运行方式

平时优先用：

```bash
--action skip
```

意思是：如果这一部已经有结果，就跳过，不重复生成。

只有在明确想重做某一步时，才用：

```bash
--action overwrite
```

`overwrite` 只会清理当前步骤自己生成的文件，不应该删除原始数据。但真实数据很大，还是建议谨慎使用。

如果只是想看看会发生什么，不想真的处理数据，用：

```bash
--step-dry-run
```

例如：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 01 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --step-dry-run
```

## 文件夹结构

推荐一个实验数据文件夹长这样：

```text
test_dataset/
  00_original_files/
  01_oir_to_tif/
  02_stim_map/
  03_motion_correct/
  04_spatial_highpass/
  05_suite2p_roi_detection/
  05e_roi_manual_curation/
  06_dff/
  07_events/
  08_stim_response/
  09_angle_tuning/
  10_population_features/
  11_population_similarity/
  12_hierarchical_clustering/
  13_leiden/
  14_dimensionality_reduction/
  15_cross_trial_summary/
  16_reports/
```

每一步的结果都放在自己的文件夹里。这样比较容易检查，也不容易把原始数据和中间结果混在一起。

## 推荐主线

现在流程分三段：

```text
premanual：00 -> 01 -> 02 -> 03 -> 04 -> 05
manual：人工 ROI 校对
postmanual：06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16
```

一口气跑自动前半段：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

打开人工 ROI 校对 GUI：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset
```

一口气跑自动后半段：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip
```

05 默认用 `03_motion_correct` 跑 suite2p；`04_spatial_highpass` 仍然生成，主要给 manual GUI 显示 ROI 边界。manual GUI 保存人工校对结果后，06 会优先读取 manual 的最终 ROI set；没有 manual 结果的 trial 才回退到 suite2p `iscell.npy`。无论 ROI 来源是哪一个，06 都默认从 `03_motion_correct` 重新抽 F、Fneu 和 dF/F。

## 一口气跑后半段

如果前面的 TIFF、运动校正、suite2p 都已经做好，可以从 06 跑到 16：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 06,07,08,09,10,11,12,13,14,15,16 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --roi-source auto \
  --event-method robust-threshold \
  --event-threshold-sigma 4 \
  --baseline-sec 5 \
  --response-sec 10 \
  --angle-period 180 \
  --similarity-source features \
  --min-corr 0.3 \
  --knn 10 \
  --n-clusters 6 \
  --leiden-resolution 1.0
```

这个命令不会重做已有结果，因为用了 `--action skip`。

## 每一步在干什么

### 00 整理原始文件

把 `.oir` 和相关文件整理到 `00_original_files/`。

适合在一批新数据刚放进数据文件夹时运行。

### 01 `.oir` 转 TIFF

用 Fiji / Bio-Formats 读取 Olympus `.oir`。

主要输出：

- calcium imaging movie 的 TIFF
- stimulus analog channel 的 TIFF
- metadata JSON

如果要保留更细的刺激通道信息，可以用：

```bash
--stim-export-mode both
```

这样会同时保存投影版和 raw 版刺激通道。

### 02 提取刺激时间

整理刺激时间。

它会尽量结合两类信息：

- 显微镜刺激 analog 通道
- 外部刺激程序记录

这一步会输出：

- 真实电压/亮度 trace 图
- 刺激时间示意图
- `stim_events.csv`
- pulse trial 的 `stim_pulse_events.csv`

### 03 运动校正

用 CaImAn 做 motion correction。

这一步比较耗时，真实数据上建议先 dry-run，再正式跑。

### 04 空间高通滤波

去掉一些大尺度背景，让后续 suite2p 找 ROI 时更专注于局部信号。

### 05 suite2p 自动找 ROI

用 suite2p 做 ROI detection。现在它主要负责生成 ROI 候选库，不再负责最终可信分类。

默认输入是 `03_motion_correct`。`04_spatial_highpass` 仍然会生成，主要用于 manual GUI 里看 ROI 边界。

注意：目前更推荐 suite2p `0.14.4`。之前 `0.14.5` 在这批数据上出现过异常 ROI 行为。

### manual 人工校对 ROI

这是唯一需要手动操作的步骤。它会打开 GUI，让你从 suite2p 候选 ROI 里挑选可信 ROI，也可以手画 freehand 或椭圆 ROI。

输出文件夹仍叫 `05e_roi_manual_curation/`，这是为了兼容以前已经保存过的人工校对结果。

### 06 提取 dF/F

读取 ROI 形状并计算 dF/F。

默认做法是：

- 有 manual GUI 保存结果时，ROI 位置和形状来自人工校对结果
- 没有 manual 结果时，ROI 位置和形状来自 05 suite2p
- F、Fneu、dF/F 从 `03_motion_correct` 的 movie 重新计算

默认：

```text
F_corrected = F - 0.7 * Fneu
```

如果某次临时想强制只用 suite2p ROI，可以用 `--roi-source suite2p`。如果想回退到 suite2p 自己的 `F.npy/Fneu.npy`，可以给单步脚本传 `--trace-source suite2p`。

### 07 检测 calcium events

从 dF/F trace 里找 calcium events。

这里的 event 只是钙信号事件，不等于电生理 spike。

目前推荐稍微严格一点：

```bash
--event-threshold-sigma 4
```

### 08 刺激响应分析

计算每个 ROI 对每次刺激的响应。

会比较：

- 刺激前 baseline
- 刺激后的 response
- event rate 变化
- response peak
- latency

也会给 ROI 一个初步 response type，比如 onset、sustained、offset、suppressed、nonresponsive。

### 09 角度调谐分析

按刺激角度整理 ROI 的响应。

主要用于看 polarization angle tuning。

默认：

```bash
--angle-period 180
```

因为 polarization angle 通常 0 度和 180 度等价。

### 10 生成 ROI 特征表

把前面几步的结果合成一个总的 ROI feature matrix。

后面的聚类、相似性、降维都依赖这一步。

### 11 ROI 相似性

计算 ROI 和 ROI 之间有多像。

可以基于：

- trace
- stimulus response
- feature matrix

当前默认主要用 feature similarity。

### 12 层次聚类

根据 ROI feature 做 hierarchical clustering。

这一步会输出 cluster labels 和聚类图。

### 13 Leiden community detection

基于 ROI similarity graph 做 Leiden community detection。

这一步需要额外包：

- `igraph`
- `leidenalg`

如果没装，程序不会崩，会写一个 skipped summary。

### 14 PCA / 降维

用 PCA 把 ROI feature matrix 降到二维或少数几个维度，方便看群体结构。

UMAP 是可选项，没装也不影响 PCA。

### 15 跨 trial 汇总

把所有 trial 的结果合并成总表。

重要输出包括：

- `cross_trial_qc_summary.csv`
- `all_trials_roi_features.csv`
- `all_trials_event_table.csv`
- `all_trials_stim_response_table.csv`
- `all_trials_angle_response_table.csv`
- `all_trials_cluster_labels.csv`
- `top_responsive_rois.csv`
- `top_event_rois.csv`
- `top_angle_selective_rois.csv`

这些表很适合后面手动筛选候选细胞。

### 16 HTML 报告

生成可以直接打开看的报告。

最重要的是：

```text
DATA_ROOT/16_reports/global_report.html
```

里面会有：

- 每个 trial 的链接
- 跨 trial QC 表
- top ROI 预览
- 主要图
- 重要 CSV 表链接

## 常用命令

### 看有哪些步骤

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py --list-steps
```

### 只跑 15 和 16，更新总表和报告

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 15,16 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite
```

### 只重新生成报告

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 16 \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action overwrite
```

### 按当前主线尽量安全地跑

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --stim-export-mode both \
  --projection-mode auto \
  --metadata-mode update-missing \
  --analog-source auto \
  --raw-z-strategy planes \
  --sigma-px 12 \
  --cell-diameter-um 5 \
  --diameter-scale 1.2 \
  --threshold-scaling 1.4 \
  --suite2p-threads 4 \
  --n-workers 1
```

然后打开人工 ROI 校对：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset
```

人工校对完成后，再跑后半段：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Users/dingyifei/Documents/calcium-imaging-pipeline-new/test_dataset \
  --action skip \
  --roi-source auto \
  --neuropil-coeff 0.7 \
  --f0-mode percentile \
  --f0-percentile 10 \
  --event-method robust-threshold \
  --event-threshold-sigma 4 \
  --baseline-sec 5 \
  --response-sec 10 \
  --angle-period 180 \
  --similarity-source features \
  --min-corr 0.3 \
  --knn 10 \
  --n-clusters 6 \
  --leiden-resolution 1.0
```

## Conda 环境

目前主要用三个环境：

- `fiji_env`：运行 Fiji launcher
- `caiman`：运行 CaImAn、图像处理、分析和画图
- `suite2p`：运行 suite2p ROI detection

环境配置文件在：

```text
envs/
```

如果是在新电脑或 Linux 工作站上配置环境，可以先看：

```text
envs/README.md
```

## 不要提交到 GitHub 的东西

不要提交：

- `.oir`
- `.oib`
- `.tif`
- `.tiff`
- `.npy`
- `.npz`
- suite2p 输出
- motion corrected movie
- raw imaging data
- 任何大型中间结果

GitHub 只放代码、说明文档、环境配置和运行日志。

## 运行日志

运行日志在：

```text
docs/run_logs/
```

这些日志记录了每次改了什么、跑了什么、结果如何。

## 当前状态

当前主线已经接到：

```text
00 -> 01 -> 02 -> 03 -> 04 -> 05 -> manual -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16
```

其中 13 Leiden 在没有 `igraph/leidenalg` 时会自动 soft-skip。

当前 test dataset 已经验证过：

- 06 到 16 可以连续 `skip` 运行
- 15 可以生成跨 trial 总表
- 16 可以生成 HTML report

## 这个 pipeline 的使用原则

最重要的原则是：

```text
不要不小心重跑大步骤。
```

所以：

- 平时用 `--action skip`
- 不确定时先用 `--step-dry-run`
- 只有明确要重做某一步时才用 `--action overwrite`
- 真实数据跑完后，先看每一步 summary，再相信后面的分析
