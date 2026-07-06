# Calcium Imaging Pipeline

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This repository contains a stepwise calcium-imaging analysis pipeline oriented
toward cephalopod retina datasets, with an emphasis on traceable intermediate
outputs, manual ROI curation, stimulus-aligned analysis, and report-ready
artifacts.

本仓库提供一个分步式钙成像分析 pipeline，当前主要面向头足类 retina
数据，强调中间结果可追溯、ROI 可人工校对、刺激对齐分析清晰、最终输出适合汇总与展示。

## 中文

## 项目定位

这个 pipeline 的目标不是“一键把 movie 跑完”，而是把原始实验数据稳定地整理成四类结果：

- 可追溯的分步中间产物
- 可人工审阅和修改的 ROI 候选与 final ROI
- 可解释的单 trial 定量分析
- 可直接用于汇报和比较的跨 trial 汇总与 HTML 报告

当前主线默认面向这类数据场景：

- 原始成像文件来自 Olympus `.oir`
- 样品多为 whole-mount retina
- 刺激信息可能来自显微镜 analog 通道，也可能来自外部刺激控制程序记录
- 后续分析重点包括 stimulus response、AoLP tuning、population feature 和 cross-trial summary

## 设计原则

- 原始数据和每一步输出严格分开，不覆盖原文件。
- 自动步骤负责生成候选结果，人工步骤负责决定最终 ROI。
- 最终 trace 从 motion-corrected movie 重新提取，而不是直接照搬 ROI detection 时的临时信号。
- 后半段尽量先把每个 trial 标准化成表格和 feature，再做群体比较和报告。

## 工作流总览

```text
原始 OIR / 刺激记录
  -> 00 整理原始文件
  -> 01 OIR -> TIFF
  -> 02 生成 stim map / stim events
  -> 03 CaImAn motion correction
  -> 04 spatial high-pass
  -> 05 suite2p / cellpose ROI candidates
  -> manual 人工 ROI 校对
  -> 06 重新提取 F/Fneu/dF/F
  -> 07 calcium event detection
  -> 08 stimulus response analysis
  -> 09 AoLP tuning
  -> 10-16 population / clustering / embedding / QC
  -> 17 cross-trial summary
  -> 18 HTML report
```

主线按三段组织：

- `premanual`: `00 -> 05`
- `manual`: 人工 ROI 校对
- `postmanual`: `06 -> 18`

这样分是为了把“候选 ROI 生成”“最终 ROI 决定”“固定 ROI 后的定量分析”三件事明确拆开，避免把自动 ROI detection 误差直接带进最终结论。

## 兼容的前半段输入场景

到 `manual` 为止，这个 pipeline 现在兼容三类常见 setup：

- 有 stim analog，也有外部刺激记录
- 没有 stim analog，但有外部刺激记录
- 没有 stim analog，也没有可对齐的刺激记录，只想先完成 motion correction、ROI candidate 和 manual ROI curation

也就是说，前半段不再把“必须有 analog 通道”当成默认前提。没有 analog 时，step 02 仍会写出明确标记的兼容 sidecar，例如 `nostim` 或 `no_analog_input`，保证 `03-05` 和 manual GUI 还能继续使用。

## 数据目录结构

推荐数据根目录如下：

```text
DATA_ROOT/
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

约定：

- `00_original_files/` 只放显微镜原始数据
- `00_stim_logs_raw/` 只放刺激控制程序导出的原始参数合集
- `stim_logs/` 如果存在，只作为兼容旧逻辑的 flat index 或 symlink 层

## 快速开始

建议日常都从总入口运行：

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps postmanual \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip
```

最常用参数：

- `--steps`: 指定步骤、步骤组或逗号列表
- `--data-root`: 指定实验数据根目录
- `--action`: `skip` / `overwrite`
- `--dry-run`: 只预览将执行什么
- `--step-dry-run`: 进入子步骤但做轻量预览
- `--trial-id`: 只处理指定 trial
- `--date-id`: 只处理指定日期，例如 `20251205`

示例：只跑某一天的前半段

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 01,02,03,04,05 \
  --data-root /absolute/path/to/DATA_ROOT \
  --date-id 20251205 \
  --action skip
```

## 主要代码入口

- [current/README.md](./current/README.md)
  - 当前 `current/` 代码布局与各步骤入口说明
- [linux_workstation/README.md](./linux_workstation/README.md)
  - Linux 工作站运行说明
- [envs/README.md](./envs/README.md)
  - 推荐 conda 环境命名与检查方式
- [docs/run_pipeline_command_guide.md](./docs/run_pipeline_command_guide.md)
  - 常用运行命令示例
- [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)
  - Dashboard GUI 使用说明

## 图示

- 可编辑流程图：
  - [docs/figures/calcium_imaging_pipeline_flowchart.svg](./docs/figures/calcium_imaging_pipeline_flowchart.svg)
  - [docs/figures/calcium_imaging_pipeline_flowchart.pptx](./docs/figures/calcium_imaging_pipeline_flowchart.pptx)

## 当前边界

- `manual` 仍然是单 trial GUI，不是自动日期批处理 GUI。
- `13` 之后的汇总类步骤还没有全部扩展成真正的日期级筛选。
- 这个仓库只放代码、文档、小型配置和必要图示，不放真实实验数据。
- 对 `nostim` trial，可以继续跑到 manual；但 `06` 之后的 stimulus-locked 分析只有在确实有可解释刺激时间时才有意义。

## 发布边界

不要提交这些内容到 GitHub：

- 原始显微图像：`.oir`, `.tif`, `.tiff`, `.czi`, `.nd2`, `.lif`
- 大型数组与中间结果：`.npy`, `.npz`, `.h5`, `.mat`
- suite2p 大输出
- 真实实验数据目录
- 临时缓存和机器本地配置

## English

## What This Repository Is For

This pipeline is not meant to be a black-box “run the whole movie and trust the
answer” workflow. Instead, it turns raw imaging experiments into:

- traceable stepwise intermediate outputs
- manually reviewable and editable ROI sets
- interpretable single-trial quantitative outputs
- cross-trial summaries and HTML reports suitable for presentation

The current maintained path is mainly designed for:

- Olympus `.oir` source files
- whole-mount retina imaging
- stimulus information from either microscope analog channels or external stimulus logs
- downstream analyses such as stimulus response, AoLP tuning, population features, and cross-trial summaries

## Core Design Ideas

- Raw data and generated outputs are kept separate.
- Automated steps generate candidates; manual curation determines the final ROI set.
- Final traces are re-extracted from the motion-corrected movie rather than copied from temporary detection-stage signals.
- Post-manual analysis first standardizes each trial into tables and features, then performs population-level comparison and reporting.

## Workflow Overview

```text
Raw OIR / stimulus records
  -> 00 organize raw files
  -> 01 OIR -> TIFF
  -> 02 generate stim map / stim events
  -> 03 CaImAn motion correction
  -> 04 spatial high-pass
  -> 05 suite2p / cellpose ROI candidates
  -> manual manual ROI curation
  -> 06 re-extract F/Fneu/dF/F
  -> 07 calcium event detection
  -> 08 stimulus response analysis
  -> 09 AoLP tuning
  -> 10-16 population / clustering / embedding / QC
  -> 17 cross-trial summary
  -> 18 HTML report
```

The maintained workflow is grouped into three phases:

- `premanual`: `00 -> 05`
- `manual`: manual ROI curation
- `postmanual`: `06 -> 18`

This separation keeps candidate generation, final ROI decisions, and
post-curation quantitative analysis clearly distinct.

## Supported Pre-Manual Input Scenarios

Up to `manual`, the pipeline currently supports three common setups:

- microscope stimulus analog plus external stimulus logs
- no stimulus analog but usable external stimulus logs
- no stimulus analog and no alignable stimulus record, when the goal is only to reach motion correction, ROI candidates, and manual ROI curation

In other words, the pre-manual path no longer assumes that an analog channel is
required. When analog input is missing, step 02 still writes explicit fallback
sidecars such as `nostim` or `no_analog_input`, so steps `03-05` and the manual
GUI can still proceed.

## Recommended Data Layout

```text
DATA_ROOT/
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

Conventions:

- `00_original_files/` stores microscope raw files only
- `00_stim_logs_raw/` stores exported raw stimulus-program records
- `stim_logs/`, if present, is only a compatibility layer such as a flat index or symlink set

## Quick Start

Use the top-level launcher for normal work:

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps postmanual \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip
```

Most useful options:

- `--steps`: step, step group, or comma-separated list
- `--data-root`: experiment data root
- `--action`: `skip` or `overwrite`
- `--dry-run`: preview commands only
- `--step-dry-run`: enter child steps but keep them lightweight
- `--trial-id`: restrict work to selected trial IDs
- `--date-id`: restrict work to one date such as `20251205`

Example: run only one date through the pre-manual path

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 01,02,03,04,05 \
  --data-root /absolute/path/to/DATA_ROOT \
  --date-id 20251205 \
  --action skip
```

## Key Entry Points

- [current/README.md](./current/README.md)
  - layout of the maintained code under `current/`
- [linux_workstation/README.md](./linux_workstation/README.md)
  - Linux workstation runner notes
- [envs/README.md](./envs/README.md)
  - recommended conda environment naming and checks
- [docs/run_pipeline_command_guide.md](./docs/run_pipeline_command_guide.md)
  - common launcher command examples
- [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)
  - dashboard GUI usage

## Figures

- Editable pipeline flowchart:
  - [docs/figures/calcium_imaging_pipeline_flowchart.svg](./docs/figures/calcium_imaging_pipeline_flowchart.svg)
  - [docs/figures/calcium_imaging_pipeline_flowchart.pptx](./docs/figures/calcium_imaging_pipeline_flowchart.pptx)

## Current Limits

- `manual` is still a single-trial GUI, not a date-batch GUI.
- The later summary steps after `13` are not all date-filtered yet.
- This repository is for code, docs, small config files, and figures, not for real experiment data.
- `nostim` trials can continue through manual curation, but stimulus-locked analysis after step `06` is only meaningful when interpretable stimulus timing exists.

## Public Repository Boundary

Do not commit:

- raw microscope image data such as `.oir`, `.tif`, `.tiff`, `.czi`, `.nd2`, `.lif`
- large numerical intermediates such as `.npy`, `.npz`, `.h5`, `.mat`
- full suite2p outputs
- real experiment data directories
- transient caches and machine-local settings
