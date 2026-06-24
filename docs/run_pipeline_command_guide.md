# run_pipeline.py 命令行说明

这份文件只写当前推荐主线。日常运行时，先进入代码库：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new
```

最常用格式：

```bash
python3 current/run_pipeline.py \
  --steps 步骤 \
  --data-root 数据文件夹 \
  --action skip
```

`--action skip` 是安全模式：已有结果就跳过，不会覆盖。只有确认要重算某一步时，再用 `--action overwrite`。

当前测试数据集路径：

```text
/Volumes/Yifei_Ding/20260617_test
```

本地和工作站的数据根目录都按同一套规则放置：`00_original_files/` 存显微镜原始数据，`00_stim_logs_raw/` 存刺激控制程序导出的原始参数合集。`stim_logs/` 如果存在，只是旧步骤兼容用的 flat index。

## 当前主流程

现在流程分成三段：

```text
自动前半段：00 -> 01 -> 02 -> 03 -> 04 -> 05
人工校对：manual
自动后半段：06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18
```

意思是：

- 00-05 负责把原始数据整理、转 TIFF、提取刺激、运动校正、生成 high-pass 底片、跑 suite2p 候选 ROI。
- `manual` 是唯一需要人工操作的步骤，用 GUI 挑选 suite2p 候选 ROI，也可以手画补 ROI。
- 06-18 负责从最终 ROI 抽 trace、算 dF/F、event、刺激响应、角度调谐、stimulus-slice feature、population 分析和报告。

重要逻辑：

- 05 默认用 `03_motion_correct` 跑 suite2p；`04_spatial_highpass` 仍然生成，主要给 manual GUI 显示 ROI 边界。
- 05 产生的 suite2p 结果只当作 ROI 位置和形状候选。
- 06 默认使用 `--roi-source auto`：优先读取 manual GUI 保存的最终 ROI set；没有 manual 结果的 trial 才回退到 suite2p `iscell.npy`。旧 05c curated ROI 路线已归档，不再作为 current 主线的默认来源。
- 06 默认从 `03_motion_correct` 重新抽 F、Fneu 和 dF/F，不直接相信 high-pass 图上的 suite2p trace。
- 12 默认把 08 的 peri-stimulus tensor 转成 normalized stimulus-slice feature matrix，并为每个 ROI x stimulus slice 输出 `stimulus_evoked_flag`。判定标准同时看 amplitude、onset latency 和 offset recovery；后续 13 similarity、14 hierarchical clustering、16 PCA/UMAP 默认使用这些 slice features。

## 一次跑自动前半段

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip
```

如果只想看看会运行什么，不真正运行：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --dry-run
```

## 打开人工 ROI 校对 GUI

打开文件选择界面：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Volumes/Yifei_Ding/20260617_test
```

直接打开某个 trial：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps manual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --trial-id 20260428_Euprymna_retina2_25x
```

GUI 里可以切换显示底片：

- `raw`：原始转换后的成像 TIFF
- `motion corrected`：运动校正后图像，适合看真实 trace
- `spatial high-pass`：高通图像，适合看 ROI 边界

保存结果仍然是同一套人工 ROI，不会因为切换底片而分开保存。

## 一次跑自动后半段

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip
```

如果只想先看计划：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --dry-run
```

后半段的推荐功能聚类入口是 stimulus slices：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps postmanual \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip \
  --similarity-source slices \
  --cluster-source slices \
  --embedding-source slices \
  --cluster-normalization both
```

## 单独运行某一步

例如只跑 02：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 02 \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action skip
```

例如重新跑 05 suite2p：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 05 \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action overwrite
```

例如重新跑 06 dF/F：

```bash
cd /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new

python3 current/run_pipeline.py \
  --steps 06 \
  --data-root /Volumes/Yifei_Ding/20260617_test \
  --action overwrite
```

06 默认会优先使用 manual GUI 的最终 ROI set；没有 manual 结果时回退到 suite2p `iscell.npy`，并从 `03_motion_correct` 重新抽 trace。如果临时想强制只用 suite2p ROI，可以加：

```bash
--roi-source suite2p
```

如果临时想用 suite2p 自己的 `F.npy/Fneu.npy`，可以加：

```bash
--step-args "--trace-source suite2p"
```

## 常用安全参数

- `--dry-run`：只显示总入口会启动什么，不真正运行。
- `--step-dry-run`：真正进入子步骤，但子步骤只预览，不处理大数据。
- `--action skip`：已有结果跳过，推荐日常使用。
- `--action overwrite`：清掉本步骤旧输出并重做，确认要重跑时使用。
- `--verbose`：显示更多运行信息。

## 旧的 05b/05c/05d

之前做过的 05b、05c、05d 是“根据历史手画 ROI 自动筛选/补候选”的实验路线。现在主线改成：

```text
05 suite2p 候选 ROI -> manual 人工校对 -> 06 后续分析
```

所以 05b/05c/05d 已从 `current/` 归档到 `archive/legacy_roi_quality_steps/`，不再作为日常 pipeline 步骤。
