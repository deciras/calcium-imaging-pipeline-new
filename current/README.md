# Current Code Layout

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This page describes the maintained code under `current/`: the launcher, step
scripts, GUI entry points, and a few implementation conventions that matter for
daily use.

本页说明 `current/` 目录下当前维护中的代码布局，包括总入口、各步骤脚本、GUI 入口，以及日常使用时最关键的实现约定。

## 中文

## 目录角色

- `run_pipeline.py`
  - 整个 numbered pipeline 的统一入口
  - 负责步骤编号、参数透传、步骤组和后台执行
- `preprocess/`
  - 原始文件整理、Fiji 转换、stim map、motion correction、spatial high-pass
- `roi/`
  - ROI candidate generation 与 manual ROI curation GUI
- `analysis/`
  - `06-18` 的 post-manual 定量分析、汇总和报告
- `tools/`
  - 维护、导出和辅助脚本，不属于常规 numbered pipeline 主线

## 常用步骤组

- `premanual`: `00-05`
- `manual`: 打开人工 ROI 校对 GUI
- `basic-analysis`: `06-09`
- `core-analysis`: `06,08,09,10-14`
- `postmanual`: `06-18`

查看当前可运行步骤：

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py --list-steps
```

## 当前的重要约定

- `02_stim_map/` 是 stimulus timing 的权威来源。
  - `06` 和 `08` 不应该继续优先使用过期副本。
- 自动 ROI detection 只生成候选 ROI。
  - final ROI 以 manual curation 输出为准。
- step 06 默认优先使用 final manual ROI。
  - 如果某个 trial 还没有 final ROI，会回退到 suite2p `iscell.npy`。
- `--trial-id` 可限制单个或多个 trial。
  - 对支持的步骤，多个 trial 用逗号分隔。
- `--date-id` 可限制单个日期。
  - 当前主要覆盖 `01-12` 的常规处理链。

## GUI 相关

- `pipeline_dashboard_gui.py`
  - pipeline dashboard 图形入口
  - 现在支持 `Date id` 输入框，可直接生成 `--date-id`
- `roi/05_manual_roi_curation_gui.py`
  - 人工 ROI 校对 GUI
  - 支持同步缩放/平移、单独视图、拖拽平移、显式保存与关闭

## Post-manual QC 图

step 06 可以输出基础 QC 图，常见包括：

- `*_dff_heatmap_sorted.png/pdf`
- `*_dff_trace_examples_with_stim.png/pdf`
- `*_dff_distribution.png/pdf`

`--plot-level` 说明：

- `none`: 不输出新增 QC 图
- `basic`: 输出核心快速 QC 图
- `full`: 再增加更诊断性的图

示例：

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06,07,08,09,10,11,12,13,14,15,16,17,18 \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip \
  --plot-level basic
```

只刷新 step 06 图而不重抽 trace：

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06 \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip \
  --refresh-plots \
  --plot-level basic
```

## 一些实现边界

- `manual` 仍然按单个 trial 工作，不会自动把日期展开成 GUI 批处理。
- `05c` 的老 curated-ROI 路线是归档旧逻辑，不是当前默认来源。
- `13/14/16` 默认使用 stimulus-slice feature，而不是 full-trace clustering。

## English

## Folder Roles

- `run_pipeline.py`
  - top-level launcher for the numbered pipeline
  - handles step IDs, argument forwarding, step groups, and background execution
- `preprocess/`
  - raw-file organization, Fiji conversion, stimulus maps, motion correction, spatial high-pass outputs
- `roi/`
  - ROI candidate generation and the manual ROI curation GUI
- `analysis/`
  - post-manual quantitative analysis, summaries, and reports for steps `06-18`
- `tools/`
  - maintenance, export, and helper utilities outside the main numbered path

## Useful Step Groups

- `premanual`: `00-05`
- `manual`: launch the manual ROI curation GUI
- `basic-analysis`: `06-09`
- `core-analysis`: `06,08,09,10-14`
- `postmanual`: `06-18`

List currently runnable steps:

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py --list-steps
```

## Important Current Conventions

- `02_stim_map/` is the authoritative source of stimulus timing.
  - Steps `06` and `08` should not prefer stale copied sidecars.
- Automated ROI detection only generates candidate ROIs.
  - Final ROIs come from manual curation outputs.
- Step 06 prefers final manual ROIs by default.
  - Trials without final manual ROIs fall back to suite2p `iscell.npy`.
- `--trial-id` can restrict supported steps to one or more trial IDs.
  - Multiple trial IDs are comma-separated.
- `--date-id` can restrict supported steps to a single date.
  - The current main coverage is the normal `01-12` chain.

## GUI Entry Points

- `pipeline_dashboard_gui.py`
  - dashboard GUI launcher
  - now includes a `Date id` field that emits `--date-id`
- `roi/05_manual_roi_curation_gui.py`
  - manual ROI curation GUI
  - supports synchronized zoom/pan, single-view mode, drag-to-pan, and explicit save/close behavior

## Post-Manual QC Plots

Step 06 can write compact QC plots such as:

- `*_dff_heatmap_sorted.png/pdf`
- `*_dff_trace_examples_with_stim.png/pdf`
- `*_dff_distribution.png/pdf`

`--plot-level`:

- `none`: disable the added QC plots
- `basic`: core quick QC figures
- `full`: additional diagnostic figures

Example:

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06,07,08,09,10,11,12,13,14,15,16,17,18 \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip \
  --plot-level basic
```

Refresh step-06 figures without re-extracting traces:

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06 \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip \
  --refresh-plots \
  --plot-level basic
```

## Current Boundaries

- `manual` still works one trial at a time and is not a date-batch GUI.
- The old `05c` curated-ROI route is archived legacy code, not the current default.
- Steps `13/14/16` default to stimulus-slice features rather than full-trace clustering.
