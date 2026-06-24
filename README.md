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
  --data-root /Volumes/Yifei_Ding/20260617_test \
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
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip \
  --step-dry-run
```

## 文件夹结构

推荐一个实验数据文件夹长这样：

```text
20260617_test/
  00_original_files/
  00_stim_logs_raw/
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
  10_trace_plots/
  11_population_features/
  12_stimulus_slice_features/
  13_population_similarity/
  14_hierarchical_clustering/
  15_leiden/
  16_dimensionality_reduction/
  17_cross_trial_summary/
  18_reports/
```

每一步的结果都放在自己的文件夹里。这样比较容易检查，也不容易把原始数据和中间结果混在一起。

约定两台机器都保持一致：

- `00_original_files/` 只放显微镜原始数据，例如日期/session 文件夹、`.oir` 和同名原始附件。
- `00_stim_logs_raw/` 只放刺激控制程序导出的原始参数合集，例如 `20260428_motor_rotation/` 里的 `timestamp_log_*.csv`、`stim_map_*.csv`、`experiment_config_*.json` 和 `*_angle_list.txt`。
- `stim_logs/` 如果存在，只当作兼容旧步骤的 flat 索引或 symlink 文件夹；不要把它当作原始归档。

当前测试数据集路径是：

```text
/Volumes/Yifei_Ding/20260617_test
```

## 推荐主线

现在流程分三段：

```text
premanual：00 -> 01 -> 02 -> 03 -> 04 -> 05
manual：人工 ROI 校对
postmanual：06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18
```

一口气跑自动前半段：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip
```

打开人工 ROI 校对 GUI：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Volumes/Yifei_Ding/20260617_test
```

一口气跑自动后半段：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip
```

05 默认用 `03_motion_correct` 跑 suite2p；`04_spatial_highpass` 仍然生成，主要给 manual GUI 显示 ROI 边界。manual GUI 保存人工校对结果后，06 会优先读取 manual 的最终 ROI set；没有 manual 结果的 trial 才回退到 suite2p `iscell.npy`。无论 ROI 来源是哪一个，06 都默认从 `03_motion_correct` 重新抽 F、Fneu 和 dF/F。

## 一口气跑后半段

如果前面的 TIFF、运动校正、suite2p 和 manual ROI 都已经做好，可以从 06 跑到 18：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 06,07,08,09,10,11,12,13,14,15,16,17,18 \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip \
  --roi-source auto \
  --event-method robust-threshold \
  --event-threshold-sigma 4 \
  --baseline-sec 5 \
  --response-sec 10 \
  --angle-period 180 \
  --similarity-source slices \
  --cluster-source slices \
  --embedding-source slices \
  --min-corr 0.3 \
  --knn 10 \
  --n-clusters 6 \
  --leiden-resolution 1.0
```

这个命令不会重做已有结果，因为用了 `--action skip`。

默认的 population/cluster 主线现在是 stimulus-slice based：

- 08 先按刺激 onset 对齐，输出 peri-stimulus tensor。onset/offset 时间优先使用 analog-derived columns，其次才回退到 protocol/MCU 时间。
- 12 把每个 ROI x stimulus slice 用刺激前 baseline 做 robust normalization：baseline median 作为中心，baseline MAD/std 作为尺度，最小尺度默认 0.02 dF/F。
- 12 对每个 slice 输出 `stimulus_evoked_flag` 和 `stimulus_response_score`。默认 flag 需要同时满足 amplitude、latency、recovery 三类证据：
  - `amplitude_pass`：normalized peak z >= 3，或者 mean z >= 1.5 且 positive AUC >= 1.0。
  - `latency_pass`：刺激 onset 后 sustained crossing 在 3 s 内出现。
  - `recovery_pass`：刺激 offset 后 6 s 内回到 abs(z) <= 1，或 offset 窗口平均 abs(z) <= 1。没有可用 offset 窗口时不把 recovery 当作硬失败。
- 13 similarity、14 hierarchical clustering、16 PCA/UMAP 默认使用 12 的 normalized stimulus-slice feature matrix。`--cluster-source traces` 可以切回完整 trace clustering，主要作为漂白/自发活动/组织状态 QC。

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
DATA_ROOT/18_reports/global_report.html
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

### 只跑 17 和 18，更新总表和报告

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 17,18 \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action overwrite
```

### 只重新生成报告

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 18 \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action overwrite
```

### 按当前主线尽量安全地跑

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip \
  --stim-export-mode both \
  --projection-mode auto \
  --metadata-mode skip \
  --analog-source auto \
  --raw-z-strategy planes \
  --sigma-px 12 \
  --cell-diameter-um 5 \
  --diameter-scale 1.2 \
  --threshold-scaling 1.4 \
  --suite2p-threads 4 \
  --n-workers 1
```

`--metadata-mode skip` keeps step 01 from reopening OIR files when TIFF outputs
already exist. Use `--metadata-mode update-missing` only when you specifically
want to refresh existing metadata JSON.

然后打开人工 ROI 校对：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Volumes/Yifei_Ding/20260617_test
```

人工校对完成后，再跑后半段：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
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
  --similarity-source slices \
  --cluster-source slices \
  --embedding-source slices \
  --min-corr 0.3 \
  --knn 10 \
  --n-clusters 6 \
  --leiden-resolution 1.0
```

## Conda 环境

目前主要用三个环境：

- `fiji_env`：运行 Fiji launcher
- `caiman`：运行 CaImAn motion correction 和前段图像处理
- `postmanual_analysis`：运行 06-18 后半段分析、聚类和画图
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

第一原则是：

```text
Mac 和 Linux workstation 立即同步化。
```

也就是说，除了系统本身不同、绝对路径不同、少量平台入口脚本不同之外，其余内容都应该一致：代码逻辑、步骤顺序、数据目录结构、文件命名、参数含义、README 说明和运行日志都要保持同步。不要在 Mac 和 Linux 上维护两套不同规则。

第二原则是：

```text
不要不小心重跑大步骤。
```

所以：

- 平时用 `--action skip`
- 不确定时先用 `--step-dry-run`
- 只有明确要重做某一步时才用 `--action overwrite`
- 真实数据跑完后，先看每一步 summary，再相信后面的分析

## 后半段各步详细说明（中文）

下面这一段是现在推荐主线里，`06 -> 18` 每一步更具体的用途、原理和默认判定思路。

### 06 提取 dF/F

这一步做两件事：

1. 决定这次 trial 最终使用哪些 ROI  
2. 用这些 ROI 从 motion-corrected movie 重新提取 `F_raw`、`Fneu`、`F_corrected` 和 `dF/F`

默认 ROI 选择逻辑：

- 优先 final manual ROI
- 如果没有 final manual ROI，则回退到 suite2p `iscell.npy`

默认 trace 逻辑：

- ROI 位置/形状来自 manual 或 suite2p
- 但荧光时间序列默认从 `03_motion_correct` 的 corrected movie 重新抽取，而不是直接盲信 suite2p 的原始 `F.npy`

默认 neuropil 校正：

- `F_corrected = F_raw - 0.7 * Fneu`

默认 F0 / dF/F：

- 按 `--f0-mode` 计算 baseline
- 最终用 `(F_corrected - F0) / max(F0, eps)` 生成 dF/F

关于 stimulus sidecar：

- 06 现在会优先使用 `02_stim_map/` 的最新 `stim_events / stim_map / stim_pulse_events`
- 只有 `02` 缺失时才回退到本 trial 输入目录里的旧副本
- 这样可以避免 “02 重跑以后，06/08 还继续读旧刺激信息” 的问题

### 07 检测 calcium events

这一步不是做 stimulus slicing，而是做全 trace 的 event detection。

默认思路是先在每条 ROI dF/F 上找比较稳健的瞬时活动，再输出：

- `event_table.csv`
- `event_binary.npy`
- `event_mask.npy`
- `event_rate_by_roi.csv`
- `event_summary.csv/json`

现在 AUC 计算已经统一成兼容新旧 NumPy 的写法，不再依赖旧 `np.trapz` API。

### 08 按刺激切 ROI trace

这一步才是真正开始做 “ROI x stimulus slice”。

逻辑顺序应该是：

1. 先拿刺激时间  
2. 按刺激时间把每个 ROI 的 dF/F 切成 peri-stimulus slices  
3. 再在这些切片上算 response table

默认切片窗口：

- 刺激前 baseline：`10 s`
- 刺激后 response window：`6 s`
- 刺激结束后 offset/recovery window：`6 s`

关键原则：

- **只要有可用 stimulus timing，就应该先切片**
- **有没有明显钙活动，是切完以后再判，不应该先把切片筛掉**

刺激时间来源现在的优先级：

1. `02_stim_map/` 最新 sidecar  
2. 旧目录副本（仅当前者缺失时）

这次修复的核心就是把 step 08 从“可能读旧 sidecar”改成了“优先读 step 02 最新 stimulus timing”。

### 09 角度调谐

这一步基于 step 08 的 stimulus-locked response 统计每个 ROI 的 AoLP tuning。

需要注意：

- 偏振角是 `0-180 deg` 周期，不是 `0-360 deg`
- 所以 preferred angle / angle difference 的统计必须按 axial data 处理

当前默认更偏保守：

- 不是只要有一点差别就叫 tuning
- 要求 response 和角度选择性都达到一定稳定度

### 10 ROI trace plots

这一步主要是画单 ROI 或多 ROI trace 供人工检查。

用途更偏 QC：

- 看漂白
- 看自发波动
- 看 stimulus 时段附近的原始动态
- 看某个 ROI 的峰值是不是合理

### 11 population features

这一步把 ROI 的整条 trace 或基础统计量先整理成 population-level feature。

它更像是一个“全局背景行为”入口，适合看：

- baseline activity
- 波动强弱
- 漂移
- 自发同步

### 12 stimulus-slice features

这一步是当前功能聚类主线的核心。

它把 step 08 的 peri-stimulus tensor 变成：

- `ROI x stimulus slice` 的 response table
- `ROI x stimulus slice` 的 feature matrix
- slice-level `0/1` 判定

默认 normalized 规则：

- 每个 `ROI x stimulus slice` 用它自己的 pre-stimulus baseline 做 normalization
- baseline median 作为中心
- baseline MAD/std 作为尺度
- 最小尺度默认 `0.02 dF/F`

默认 `stimulus_evoked_flag` 规则：

必须同时满足三组证据：

1. `amplitude_pass`
   - peak z >= 3
   - 或 mean z >= 1.5 且 positive AUC >= 1.0
2. `latency_pass`
   - onset 后 3 s 内出现 sustained crossing
3. `recovery_pass`
   - offset 后 6 s 内回到 `abs(z) <= 1`
   - 或 offset window 的平均 `abs(z) <= 1`

也就是说：

- 先切片
- 再判 `0/1`

而不是反过来。

这一步现在还会给每个 `ROI x stimulus slice` 生成单独 panel：

- 上面：normalized trace
- 下面：raw dF/F
- 图上直接写 `evoked=0/1`
- 以及 `amp / lat / rec` 三个子判定和数值证据

### 13 population similarity

默认使用 step 12 的 stimulus-slice feature matrix 来算 ROI 之间的相似性。

这一步现在更像：

- 功能相似性的基础图
- 为后面的 hierarchical clustering / Leiden / embedding 提供输入

如果想看完整 trace 的相似性，而不是 stimulus-slice 功能相似性，可以显式切回 trace source，但那更偏 QC。

### 14 hierarchical clustering

这一步做层次聚类。

当前主线推荐：

- 用 normalized stimulus-slice features 聚类

而不是直接用整条 trace 聚类，因为整条 trace 更容易被：

- 漂白
- baseline drift
- spontaneous activity
- recording state

这些东西主导。

### 15 Leiden community detection

这一步是图聚类版本的功能分群。

它依赖 step 13 的相似性图。如果某个 trial：

- 没有足够有效边
- feature matrix 太空

现在会更倾向于 clean skip，而不是整步失败。

### 16 dimensionality reduction

这一步做 PCA / UMAP 之类的降维。

用途主要是：

- 看 cluster 是否分开
- 看不同 response pattern 的连续性
- 看 trial 内功能结构是否有明显主轴

默认也优先用 stimulus-slice features。

### 17 cross-trial summary

这一步把很多单 trial summary 汇总到跨 trial 级别。

适合回答的问题包括：

- 每个 trial 有多少 ROI
- 有多少响应细胞
- 哪些 trial 没 stimulus
- 哪些 trial clustering / embedding 被跳过

### 18 HTML report

最后把前面多步结果汇总成一个 HTML 报告。

这个报告不是只放文件链接，而是尽量让人能直接检查：

- trial context
- clustering / similarity
- 响应统计
- QC 异常

## 这次与 stimulus slicing 直接相关的修复

这次已经确认并修复了两条关键问题：

1. `08_stim_response_analysis.py`
   - 现在优先读 `02_stim_map/` 的最新 stimulus sidecar
   - 不再默认继续吃 `06_dff/` 里可能过期的副本

2. `06_extract_dff.py`
   - 现在也优先同步 `02_stim_map/` 的最新 sidecar
   - 保证 06 目录里的刺激信息不容易滞后

如果以后再看到：

- `n_stimulus_slices = 0`
- 但你明明知道这个 trial 有刺激

第一件要检查的事情应该就是：

- `02_stim_map/<trial>/*_stim_events.csv` 是否有 event rows
- `08_stim_response/<trial>/*_stim_response_summary.json` 里的 `n_stim_events` 是否仍然是 0
