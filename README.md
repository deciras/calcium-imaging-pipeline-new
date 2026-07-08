# Calcium Imaging Pipeline

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This repository contains a stepwise calcium-imaging analysis pipeline oriented
toward cephalopod retina datasets. It is designed for traceable intermediate
outputs, manual ROI curation, stimulus-aware analysis, and report-ready trial
and cross-trial summaries.

本仓库提供一个分步式钙成像分析 pipeline，当前主要面向头足类 retina
数据，强调中间结果可追溯、ROI 可人工校对、刺激信息可对齐、分析结果可解释，并能最终产出适合比较和展示的 trial / cross-trial 报告。

## 中文

## 1. 这个仓库是做什么的

这套 pipeline 的目标不是“一键跑完 movie 然后直接给结论”，而是把一份实验数据稳定地拆成多个可检查、可复现、可局部返工的阶段。

更具体地说，它希望把原始实验数据逐步变成以下几类结果：

- 可追溯的中间产物
  - 例如 TIFF、metadata、stim map、motion corrected movie、high-pass movie
- 可人工审阅的 ROI
  - 自动检测只生成候选 ROI，真正用于后续分析的是人工确认后的 ROI set
- 可解释的单 trial 定量分析
  - 例如 dF/F、event、stimulus response、AoLP tuning
- 可比较的跨 trial 汇总
  - 例如 feature table、similarity、clustering、cross-trial summary
- 可直接展示给人看的结果
  - 例如 HTML 报告、流程图、QC 图、导出视频

这意味着它更接近“可审核的数据整理流程”，而不是黑箱分析脚本。

## 2. 当前主要面向什么数据

当前维护主线默认面向这类数据场景：

- 原始成像文件来自 Olympus `.oir`
- 样品多为 whole-mount retina
- 刺激信息可能来自：
  - 显微镜模拟通道
  - 外部刺激控制程序导出的记录
  - 只有实验记录、没有自动 stim log 的早期数据
- 后续分析重点通常包括：
  - stimulus response
  - AoLP tuning
  - population features
  - clustering / dimensionality reduction
  - cross-trial comparison

虽然代码现在已经兼容更宽松的输入场景，但它不是一个“任何钙成像数据都直接适配”的通用框架。当前设计仍然明显带有这类 retina / polarization 实验的工作流特征。

## 3. 设计原则

这套 pipeline 的核心原则有几条：

### 3.1 原始数据与输出严格分开

- 原始 `.oir`、刺激原始记录与各步骤输出目录分层放置
- 不覆盖原始数据
- 不把中间结果写回原始目录

### 3.2 自动步骤只负责生成候选结果

- 自动 ROI detection 不是最终真值
- 人工 ROI curation 才决定 final ROI set
- 后面的定量分析应当尽量围绕 final ROI，而不是围绕自动检测阶段的临时结果

### 3.3 下游分析尽量围绕标准化 sidecar 与表格

- stimulus timing 尽量统一到 `02_stim_map/`
- ROI traces、slice features、trial metadata、summary tables 都尽量标准化输出
- 跨 trial 分析优先读取结构化表格，而不是重新猜原始文件关系

### 3.4 允许局部返工，而不是逼迫全量重跑

- 可以只重跑某一步
- 可以按 trial 处理
- 现在也可以按日期处理一部分步骤
- GUI 与 CLI 都尽量围绕“续跑、抽查、返工”而不是“每次全量重建”

## 4. 为什么要分成 `premanual`、`manual`、`postmanual`

维护主线分成三段：

```text
premanual: 00 -> 01 -> 02 -> 03 -> 04 -> 05
manual:    人工 ROI 校对
postmanual: 06 -> 18
```

这样分不是为了命名好看，而是因为这三类任务的性质不同：

- `premanual`
  - 把原始数据转换成“可供判断”的标准输入
  - 包括整理原始文件、转 TIFF、生成 stim map、motion correction、ROI 候选生成
- `manual`
  - 把“候选 ROI”变成“真正用于后续分析的 ROI set”
- `postmanual`
  - 在 final ROI 已确定的前提下做定量分析和汇总

如果不这样分，常见问题是：

- 自动 ROI detection 的误差直接污染最终分析
- ROI 还没定好就过早做 tuning / clustering
- 想重跑后半段时，不得不把前半段一起重来

## 5. 工作流总览

```text
原始 OIR / 刺激记录
  -> 00 整理原始文件
  -> 01 OIR -> TIFF
  -> 02 提取刺激时间并生成 stim map / stim events
  -> 03 CaImAn motion correction
  -> 04 spatial high-pass
  -> 05 suite2p / cellpose ROI candidates
  -> manual 人工 ROI 校对
  -> 06 从 corrected movie 重新提取 F/Fneu/dF/F
  -> 07 calcium event detection
  -> 08 stimulus response analysis
  -> 09 AoLP tuning analysis
  -> 10 trace QC plots
  -> 11 population features
  -> 12 stimulus-slice features
  -> 13 population similarity
  -> 14 hierarchical clustering
  -> 15 Leiden community detection
  -> 16 dimensionality reduction
  -> 17 cross-trial summary
  -> 18 HTML report
```

## 6. 到 `manual` 为止兼容哪些输入场景

前半段现在兼容三类常见 setup：

- 有 stim analog，也有外部刺激记录
- 没有 stim analog，但有外部刺激记录
- 没有 stim analog，也没有可对齐刺激记录，只想先做到 motion correction、ROI candidate 和 manual ROI curation

这意味着：

- 前半段不再把“必须有 analog 通道”当作默认前提
- step 02 在没有 analog 时仍然会写出统一 sidecar
- 如果既没有 analog 也没有 stim log，可以明确走 `nostim / no_analog_input` 兼容路径

但是要注意：

- `06` 之后并不是所有分析都对 `nostim` trial 有意义
- 尤其 stimulus-locked response 和 AoLP tuning 只有在确实有可解释刺激时间时才值得信任

## 7. 推荐目录结构

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

补充约定：

- `00_original_files/`
  - 只放显微镜原始数据及其同层原始附件
- `00_stim_logs_raw/`
  - 只放刺激控制程序导出的原始参数合集
- `stim_logs/`
  - 如果存在，只作为兼容旧逻辑的 flat index / symlink 层
- `18_reports/`
  - 更偏向最终汇总、HTML 报告、演示用视频等结果

## 8. 各步骤到底在做什么

下面按“输入 / 输出 / 作用 / 风险点”说明。

### 00 整理原始文件

输入：

- 一个新拷入的数据根目录
- 原始 `.oir`
- 同批 sidecar / 附件

输出：

- 标准化后的 `00_original_files/`

作用：

- 把后续步骤依赖的 trial 命名和目录结构先稳定下来

风险点：

- 这是最容易因为路径理解错误而把原始目录结构改坏的一步
- 因此它应该尽量保守，默认不递归乱动，不自动覆盖

### 01 OIR -> TIFF

输入：

- `00_original_files/` 中的 `.oir`
- Fiji / Bio-Formats

输出：

- 主成像 TIFF
- 刺激模拟通道 TIFF（如果存在）
- metadata JSON

作用：

- 让后续步骤统一读取 TIFF 和 metadata，而不是每一步都直接重新碰 `.oir`

风险点：

- Fiji 路径、Java 内存、Bio-Formats 版本都可能影响稳定性
- 大 movie 时容易碰到内存问题

### 02 生成 stim map / stim events

输入：

- step 01 的 metadata
- 可选的 stim analog TIFF
- 可选的 `00_stim_logs_raw/`
- 可选的手工补录 stim log

输出：

- `*_brightness_trace.csv`
- `*_stim_map.csv`
- `*_stim_events.csv`
- `*_stim_pulse_events.csv`
- 诊断图

作用：

- 把“实验时到底发生了什么刺激”统一写成可直接消费的时间表

风险点：

- 计划刺激与实际刺激未必一致
- 没有 analog 时只能走 fallback
- 早期手工记录若单位或时序理解错，会直接污染后续 stimulus-locked 分析

### 03 motion correction

输入：

- step 01 输出的 movie

输出：

- corrected movie
- motion correction 相关 sidecar

作用：

- 尽量把组织漂移从后续 ROI / trace 中剥离

风险点：

- 运动校正失败时，下游所有 ROI 和 dF/F 都会受影响
- 需要结合 QC 图和实际观察判断 corrected movie 是否可信

### 04 spatial high-pass

输入：

- corrected movie

输出：

- high-pass movie

作用：

- 主要服务于 ROI 边界显示和候选检测

风险点：

- 这个输出更偏向“帮助识别边界”，不代表真实生理信号强弱

### 05 ROI candidates

输入：

- corrected movie / high-pass movie

输出：

- suite2p 或 cellpose 的候选 ROI

作用：

- 给 manual curation 一个起点

风险点：

- 自动检测的误检、漏检、碎片化、神经纤维/非细胞结构混入都很常见
- 不应把这里的结果直接当 final ROI

### manual ROI curation

输入：

- ROI candidates
- reference frame / high-pass frame / ROI overlay

输出：

- final ROI set

作用：

- 决定后续所有定量分析到底围绕哪些 ROI 展开

风险点：

- 这是整个流程里最依赖人工判断的一步
- ROI 身份不确定性必须保留，不应强行把所有 ROI 都解释为 photoreceptor

### 06 dF/F extraction

输入：

- corrected movie
- final ROI

输出：

- `F`
- `Fneu`
- `dF/F`
- QC 图

作用：

- 从最终 ROI 出发重新抽取 trace，而不是复用检测阶段的临时 trace

风险点：

- baseline 估计、neuropil correction、ROI source 选择都会影响结果

### 07 events

输入：

- dF/F trace

输出：

- calcium event 表格与相关 sidecar

作用：

- 对后续统计或可视化提供事件级描述

风险点：

- event detection 对阈值和 SNR 非常敏感

### 08 stimulus response

输入：

- dF/F
- `02_stim_map/` 中的 stimulus timing

输出：

- stimulus-locked 响应表格
- response summary
- peri-stimulus slice

作用：

- 真正把 ROI trace 与 stimulus timing 对齐

风险点：

- 如果 stimulus timing 错了，这一步后面的解读会整体跑偏

### 09 AoLP tuning

输入：

- stimulus-aligned response

输出：

- angle tuning summary
- preferred angle / OSI 等指标

作用：

- 评估偏振角响应特征

风险点：

- AoLP 是 0-180° axial data，不是普通 0-360° direction data
- 如果统计方法或角度周期处理错了，会直接得出错误 preferred angle

### 10-16 population / clustering / embedding

输入：

- traces
- response tables
- slice features
- ROI-level summary

输出：

- QC 图
- feature 表
- similarity
- clustering
- embedding

作用：

- 从单 ROI / 单 trial 结果走向群体比较和模式识别

风险点：

- 如果前面 ROI 或 stimulus timing 有偏差，这一段会把偏差“统计化”
- 聚类结果漂亮不等于生物学解释可靠

### 17 cross-trial summary

输入：

- 多个 trial 的标准化输出

输出：

- cross-trial summary

作用：

- 把多个 trial 放到同一口径下比较

风险点：

- trial 间样本状态、成像条件、刺激条件不一致时，不能只看 summary 数字而忽略实验背景

### 18 HTML report

输入：

- 前面各步骤的汇总结果

输出：

- HTML 报告

作用：

- 提供一个便于浏览、演示和检查的可视化入口

风险点：

- HTML 报告适合“看结构”和“看异常”，不应替代对原始 QC 与关键表格的判断

## 9. 常用运行方式

### 9.1 日常总入口

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps postmanual \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip
```

### 9.2 只预览，不执行

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps premanual \
  --data-root /absolute/path/to/DATA_ROOT \
  --dry-run
```

### 9.3 只处理指定 trial

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06,07,08,09 \
  --data-root /absolute/path/to/DATA_ROOT \
  --trial-id 20251205_Euprymna_retina_0002 \
  --action skip
```

### 9.4 只处理指定日期

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 01,02,03,04,05 \
  --data-root /absolute/path/to/DATA_ROOT \
  --date-id 20251205 \
  --action skip
```

### 9.5 GUI 启动

相关说明见：

- [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)

相关入口见：

- macOS:
  - [current/launch_pipeline_dashboard.command](./current/launch_pipeline_dashboard.command)
- Linux:
  - [linux_workstation/launch_pipeline_dashboard.sh](./linux_workstation/launch_pipeline_dashboard.sh)
  - [linux_workstation/launch_pipeline_dashboard.desktop](./linux_workstation/launch_pipeline_dashboard.desktop)
- Windows:
  - [windows/launch_pipeline_dashboard.bat](./windows/launch_pipeline_dashboard.bat)
  - [windows/launch_pipeline_dashboard.ps1](./windows/launch_pipeline_dashboard.ps1)

## 10. 最重要的几个参数

- `--steps`
  - 选择步骤、步骤组或逗号列表
- `--data-root`
  - 指向实验数据根目录
- `--action`
  - `skip` / `overwrite`
- `--dry-run`
  - 只预览总入口会做什么
- `--step-dry-run`
  - 进入子步骤，但子步骤只做轻量预览
- `--trial-id`
  - 限制到单个或多个 trial
- `--date-id`
  - 限制到某个日期

日常更推荐：

- `--action skip`
  - 续跑最安全
- 先 `--dry-run`
  - 再正式跑

## 11. 主代码入口与相关说明文件

- [current/README.md](./current/README.md)
  - 当前维护主线 `current/` 的代码布局说明
- [linux_workstation/README.md](./linux_workstation/README.md)
  - Linux 工作站入口说明
- [windows/README.md](./windows/README.md)
  - Windows 薄启动层说明
- [envs/README.md](./envs/README.md)
  - conda 环境命名和检查说明
- [docs/run_pipeline_command_guide.md](./docs/run_pipeline_command_guide.md)
  - 常用命令示例
- [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)
  - dashboard GUI 使用说明
- [archive/README.md](./archive/README.md)
  - 历史归档说明

## 12. 跨平台原则

这个仓库不应该维护三套彼此复制的主代码。

推荐结构是：

- 一份共享主代码
  - `current/run_pipeline.py`
  - `current/pipeline_dashboard_gui.py`
  - `current/preprocess/`
  - `current/roi/`
  - `current/analysis/`
- 每个平台一层很薄的入口和适配
  - macOS
  - Linux
  - Windows

应该按平台分开的主要是：

- 启动方式
- 默认路径
- `conda` / Fiji 定位
- 桌面快捷方式

不应该按平台复制的主要是：

- pipeline 主逻辑
- dashboard 主界面逻辑
- trial / date 筛选规则
- stim log、ROI、dF/F、response、tuning、summary 逻辑

## 13. 这个仓库放什么，不放什么

这个仓库适合放：

- 代码
- 文档
- 小型配置
- 启动脚本
- 小型图示
- 运行日志
- 历史代码快照

不应该放：

- 原始显微图像
  - `.oir`, `.tif`, `.tiff`, `.czi`, `.nd2`, `.lif`
- 大型数值结果
  - `.npy`, `.npz`, `.h5`, `.mat`
- suite2p 大输出
- 真实实验数据目录
- 临时缓存
- 机器本地配置

## 14. 当前边界与已知限制

- `manual` 仍然是单 trial GUI，不是日期级批处理 GUI
- `13` 之后的汇总步骤还没有全部扩展成真正的日期级筛选
- 对 `nostim` trial，可以继续跑到 manual；但很多 stimulus-locked 分析并不自动变得有意义
- Windows 现在只有薄启动层，不维护独立 Windows 主代码
- 历史实验记录转 manual stim log 的自动补录仍然需要人工核查单位和时序

## 15. 哪些结果应该谨慎解释

以下结果尤其需要谨慎：

- 自动 ROI detection 输出
  - 不是 final ROI truth
- stimulus-locked response
  - 强依赖 stimulus timing 是否正确
- AoLP tuning
  - 强依赖角度统计是否按 axial data 处理
- clustering / embedding
  - 容易把技术偏差包装成“结构”
- cross-trial summary
  - 容易忽略 trial 间样品状态与条件差异

更稳妥的做法是始终区分：

- 观察到的现象
- 数据处理层面的结构
- 真正可以支持的生物学解释

## 16. 图示与可展示输出

- 可编辑流程图：
  - [docs/figures/calcium_imaging_pipeline_flowchart.svg](./docs/figures/calcium_imaging_pipeline_flowchart.svg)
  - [docs/figures/calcium_imaging_pipeline_flowchart.pptx](./docs/figures/calcium_imaging_pipeline_flowchart.pptx)

此外，仓库里也逐步加入了：

- dashboard 入口
- movie 导出工具
- trial sync video 导出工具
- 手工 stim log 生成工具

## 17. 如果第一次接手这个仓库，建议先看什么

推荐顺序：

1. 先看本 README
2. 再看 [current/README.md](./current/README.md)
3. 再看 [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)
4. 如果主要在 Linux 跑，接着看 [linux_workstation/README.md](./linux_workstation/README.md)
5. 如果主要想看历史变化，再看 [archive/README.md](./archive/README.md)

如果只是想尽快开始跑：

1. 先准备 `DATA_ROOT`
2. 先 dry-run
3. 先跑 `premanual`
4. 做 manual ROI curation
5. 再跑 `postmanual`

## English

## 1. What This Repository Is For

This pipeline is not intended to be a black-box “run the movie and trust the
answer” workflow. Instead, it breaks one experiment into multiple stages that
can be inspected, rerun, and audited independently.

Its practical goal is to turn raw experimental data into:

- traceable intermediate artifacts
- manually reviewable ROI sets
- interpretable single-trial quantitative outputs
- cross-trial summaries
- report-ready and presentation-ready outputs

In other words, it is closer to an auditable analysis workflow than a
single-pass script bundle.

## 2. What Kind of Data It Mainly Targets

The current maintained path is mainly designed for:

- Olympus `.oir` source files
- whole-mount retina imaging
- stimulus information from:
  - microscope analog channels
  - external stimulus-program logs
  - older experiments with only hand-written notes and no auto-generated stim log
- downstream analyses such as:
  - stimulus response
  - AoLP tuning
  - population features
  - clustering / dimensionality reduction
  - cross-trial summaries

The code is now more tolerant than before, but it is still not a completely
generic calcium-imaging framework. The current workflow retains strong retina /
polarization-experiment assumptions.

## 3. Core Design Principles

### 3.1 Raw data and outputs stay separate

- raw `.oir` files, raw stimulus logs, and generated outputs live in different layers
- original data should not be overwritten
- intermediate outputs should not be written back into the raw-data layer

### 3.2 Automated steps generate candidates, not truth

- automated ROI detection is not final truth
- manual ROI curation determines the final ROI set
- downstream quantitative analysis should be built around final ROIs, not around temporary detection-stage signals

### 3.3 Downstream analysis should prefer standardized sidecars and tables

- stimulus timing should flow through `02_stim_map/`
- ROI traces, slice features, trial metadata, and summary tables should be standardized outputs
- cross-trial analysis should consume structured outputs instead of re-inferring file relationships from raw directories

### 3.4 Partial reruns should be normal

- one step can be rerun without rebuilding everything
- one trial can be processed independently
- many steps can now be limited by date
- both the CLI and GUI are meant for resume, inspection, and targeted rework

## 4. Why The Workflow Is Split Into `premanual`, `manual`, and `postmanual`

The maintained workflow is split into:

```text
premanual: 00 -> 05
manual: manual ROI curation
postmanual: 06 -> 18
```

This is not just naming convenience. These stages represent fundamentally
different tasks:

- `premanual`
  - converts raw experiments into standardized inputs and candidate ROIs
- `manual`
  - converts candidate ROIs into the final ROI set
- `postmanual`
  - performs quantitative analysis only after the final ROI set is fixed

Without that separation, common failure modes are:

- automated ROI-detection errors leaking directly into final results
- tuning or clustering being run before ROI identity is stable
- downstream reruns forcing unnecessary rebuilds of the upstream chain

## 5. Workflow Overview

```text
Raw OIR / stimulus records
  -> 00 organize raw files
  -> 01 OIR -> TIFF
  -> 02 generate stim map / stim events
  -> 03 CaImAn motion correction
  -> 04 spatial high-pass
  -> 05 suite2p / cellpose ROI candidates
  -> manual manual ROI curation
  -> 06 re-extract F/Fneu/dF/F from corrected movies
  -> 07 calcium event detection
  -> 08 stimulus response analysis
  -> 09 AoLP tuning analysis
  -> 10 trace QC plots
  -> 11 population features
  -> 12 stimulus-slice features
  -> 13 population similarity
  -> 14 hierarchical clustering
  -> 15 Leiden community detection
  -> 16 dimensionality reduction
  -> 17 cross-trial summary
  -> 18 HTML report
```

## 6. Supported Input Scenarios Up To `manual`

The pre-manual path currently supports three common setups:

- microscope stimulus analog plus external stimulus logs
- no stimulus analog but usable external stimulus logs
- no stimulus analog and no alignable stimulus record, when the goal is only to reach motion correction, ROI candidates, and manual ROI curation

This means:

- the pre-manual path no longer assumes an analog channel is mandatory
- step 02 still writes standardized sidecars when analog is missing
- older no-analog or no-stim datasets can still proceed through the front half of the pipeline

However:

- not every `nostim` trial meaningfully supports post-manual stimulus-locked analyses
- especially response alignment and AoLP tuning depend on interpretable stimulus timing

## 7. Recommended Data Layout

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

- `00_original_files/`
  - microscope raw files and raw trial-layer attachments only
- `00_stim_logs_raw/`
  - exported raw records from the external stimulus-control program
- `stim_logs/`
  - legacy compatibility layer only, such as a flat index or symlink layer
- `18_reports/`
  - final report-like outputs such as HTML reports and demonstration videos

## 8. What Each Step Actually Does

### 00 Organize raw files

Input:

- a freshly copied data root
- raw `.oir` files
- same-batch raw sidecars

Output:

- standardized `00_original_files/`

Purpose:

- stabilize trial naming and directory layout before any analysis

Main risk:

- this is the step most likely to damage layout if the path assumptions are wrong

### 01 OIR -> TIFF

Input:

- `.oir` files under `00_original_files/`
- Fiji / Bio-Formats

Output:

- main imaging TIFF
- stimulus analog TIFF when present
- metadata JSON

Purpose:

- convert all later reading to stable TIFF + metadata instead of repeatedly touching `.oir`

Main risk:

- Fiji path, Java memory, and Bio-Formats behavior affect stability

### 02 Stimulus map generation

Input:

- step-01 metadata
- optional stimulus analog TIFF
- optional `00_stim_logs_raw/`
- optional manually reconstructed stim logs

Output:

- `*_brightness_trace.csv`
- `*_stim_map.csv`
- `*_stim_events.csv`
- `*_stim_pulse_events.csv`
- diagnostic figures

Purpose:

- convert “what was supposed to happen” and “what probably did happen during imaging” into one standardized timing representation

Main risk:

- planned stimulus and actual stimulus may diverge
- fallback interpretation is only as good as the available metadata or notes

### 03 Motion correction

Input:

- step-01 movie

Output:

- corrected movie
- motion-correction sidecars

Purpose:

- reduce drift before ROI analysis and trace extraction

Main risk:

- if motion correction is poor, all downstream ROI and dF/F analysis become harder to trust

### 04 Spatial high-pass

Input:

- corrected movie

Output:

- high-pass movie

Purpose:

- mainly improves ROI boundary visibility

Main risk:

- this is an aid for visualization and detection, not a direct physiological signal representation

### 05 ROI candidates

Input:

- corrected movie / high-pass movie

Output:

- suite2p or cellpose ROI candidates

Purpose:

- provide a starting point for manual curation

Main risk:

- false positives, merged cells, split cells, fibers, and non-cell structures are common

### Manual ROI curation

Input:

- ROI candidates
- reference views and overlays

Output:

- final ROI set

Purpose:

- define which ROIs are actually used downstream

Main risk:

- this stage is strongly judgment-dependent and ROI identity uncertainty should be preserved

### 06 dF/F extraction

Input:

- corrected movie
- final ROI set

Output:

- `F`
- `Fneu`
- `dF/F`
- QC plots

Purpose:

- re-extract traces from the final ROI set rather than reusing detection-stage traces

Main risk:

- baseline estimation, neuropil correction, and ROI source choice all matter

### 07 Events

Input:

- dF/F traces

Output:

- calcium event tables and related sidecars

Purpose:

- provide event-level descriptors for later interpretation and plotting

Main risk:

- event detection is sensitive to thresholding and SNR

### 08 Stimulus response

Input:

- dF/F
- stimulus timing from `02_stim_map/`

Output:

- stimulus-locked response tables
- response summaries
- peri-stimulus slices

Purpose:

- align ROI activity to stimulus timing

Main risk:

- incorrect stimulus timing propagates directly into all downstream stimulus-aware analysis

### 09 AoLP tuning

Input:

- stimulus-aligned responses

Output:

- tuning summaries
- preferred angle and selectivity metrics

Purpose:

- quantify polarization-angle response behavior

Main risk:

- AoLP is 0-180 degree axial data, not ordinary 0-360 direction data

### 10-16 Population / clustering / embedding

Input:

- traces
- response tables
- slice features
- ROI-level summaries

Output:

- QC figures
- feature tables
- similarity matrices
- cluster assignments
- low-dimensional embeddings

Purpose:

- move from single-ROI or single-trial description toward population structure

Main risk:

- upstream bias can become more persuasive, not less, once summarized statistically

### 17 Cross-trial summary

Input:

- standardized outputs from multiple trials

Output:

- cross-trial summary tables

Purpose:

- compare trials under one output schema

Main risk:

- trial-to-trial differences in condition or sample quality can be hidden if only summary numbers are read

### 18 HTML report

Input:

- summarized outputs from earlier steps

Output:

- HTML report

Purpose:

- provide a human-browsable inspection and presentation layer

Main risk:

- the report is useful for browsing, not a replacement for raw QC or key tables

## 9. Common Ways To Run The Pipeline

### 9.1 Normal launcher path

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps postmanual \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip
```

### 9.2 Preview only

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps premanual \
  --data-root /absolute/path/to/DATA_ROOT \
  --dry-run
```

### 9.3 Restrict to one trial

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06,07,08,09 \
  --data-root /absolute/path/to/DATA_ROOT \
  --trial-id 20251205_Euprymna_retina_0002 \
  --action skip
```

### 9.4 Restrict to one date

```bash
python3 /absolute/path/to/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 01,02,03,04,05 \
  --data-root /absolute/path/to/DATA_ROOT \
  --date-id 20251205 \
  --action skip
```

### 9.5 GUI launch

See:

- [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)

Launchers:

- macOS:
  - [current/launch_pipeline_dashboard.command](./current/launch_pipeline_dashboard.command)
- Linux:
  - [linux_workstation/launch_pipeline_dashboard.sh](./linux_workstation/launch_pipeline_dashboard.sh)
  - [linux_workstation/launch_pipeline_dashboard.desktop](./linux_workstation/launch_pipeline_dashboard.desktop)
- Windows:
  - [windows/launch_pipeline_dashboard.bat](./windows/launch_pipeline_dashboard.bat)
  - [windows/launch_pipeline_dashboard.ps1](./windows/launch_pipeline_dashboard.ps1)

## 10. The Most Important CLI Arguments

- `--steps`
  - choose a step, step group, or comma-separated list
- `--data-root`
  - experiment data root
- `--action`
  - `skip` or `overwrite`
- `--dry-run`
  - preview what the launcher would run
- `--step-dry-run`
  - enter child steps in lightweight preview mode
- `--trial-id`
  - restrict work to selected trial IDs
- `--date-id`
  - restrict work to one date

Recommended default habits:

- prefer `--action skip`
- run `--dry-run` first

## 11. Main Code Entry Points And Companion Docs

- [current/README.md](./current/README.md)
  - maintained code layout under `current/`
- [linux_workstation/README.md](./linux_workstation/README.md)
  - Linux workstation runner notes
- [windows/README.md](./windows/README.md)
  - Windows thin-launcher notes
- [envs/README.md](./envs/README.md)
  - conda environment naming and checks
- [docs/run_pipeline_command_guide.md](./docs/run_pipeline_command_guide.md)
  - common command examples
- [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)
  - dashboard GUI usage
- [archive/README.md](./archive/README.md)
  - historical archive guide

## 12. Cross-Platform Principle

This repository should not maintain three duplicated copies of the main code
for macOS, Linux, and Windows.

The intended split is:

- one shared code path
  - `current/run_pipeline.py`
  - `current/pipeline_dashboard_gui.py`
  - `current/preprocess/`
  - `current/roi/`
  - `current/analysis/`
- one thin launcher layer per platform
  - macOS
  - Linux
  - Windows

Platform-specific differences should mainly be:

- launch style
- default paths
- `conda` / Fiji discovery
- desktop shortcuts

Shared logic should include:

- pipeline logic
- dashboard logic
- trial / date filtering
- stimulus-log, ROI, dF/F, response, tuning, and summary logic

## 13. What Belongs In This Repository And What Does Not

Appropriate contents:

- source code
- documentation
- small config files
- launcher scripts
- small diagrams
- run logs
- historical code snapshots

Do not commit:

- raw microscope image data
  - `.oir`, `.tif`, `.tiff`, `.czi`, `.nd2`, `.lif`
- large numeric intermediates
  - `.npy`, `.npz`, `.h5`, `.mat`
- full suite2p outputs
- real experiment data directories
- temporary caches
- machine-local settings

## 14. Current Limits

- `manual` is still a single-trial GUI, not a true date-batch GUI
- the later summary steps after `13` are not all date-filtered yet
- `nostim` compatibility does not automatically make all post-manual analyses meaningful
- Windows currently has a thin launcher layer, not an independently maintained Windows codebase
- hand-written stim-note conversion still requires human review of timing units and interpretation

## 15. Which Results Need Careful Interpretation

Be especially careful with:

- automated ROI detection outputs
- stimulus-locked response summaries
- AoLP tuning outputs
- clustering and embedding structure
- cross-trial summaries

It is safer to keep separate:

- observed phenomena
- data-processing structure
- actual biological interpretation

## 16. Figures And Presentation Outputs

- Editable pipeline flowchart:
  - [docs/figures/calcium_imaging_pipeline_flowchart.svg](./docs/figures/calcium_imaging_pipeline_flowchart.svg)
  - [docs/figures/calcium_imaging_pipeline_flowchart.pptx](./docs/figures/calcium_imaging_pipeline_flowchart.pptx)

The repository also includes or is gradually adding:

- dashboard launchers
- movie-export helpers
- trial-sync video export helpers
- manual stim-log generation helpers

## 17. Suggested Reading Order For New Users

Recommended order:

1. read this README
2. read [current/README.md](./current/README.md)
3. read [docs/pipeline_dashboard_gui.md](./docs/pipeline_dashboard_gui.md)
4. if running mainly on Linux, read [linux_workstation/README.md](./linux_workstation/README.md)
5. if you need older behavior, then read [archive/README.md](./archive/README.md)

If the main goal is simply to start running:

1. prepare `DATA_ROOT`
2. dry-run first
3. run `premanual`
4. do manual ROI curation
5. run `postmanual`
