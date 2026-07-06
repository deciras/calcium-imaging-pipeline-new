# Linux Workstation Runner

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This folder is the Linux workstation entry layer for running the maintained
pipeline on Ubuntu-style desktop or overnight machines.

本目录提供 Linux 工作站上的 pipeline 运行入口，主要面向 Ubuntu 类桌面或 overnight 跑批机器。

## 中文

## 核心原则

Linux workstation 和 Mac 应该尽量保持同一套规则。除了操作系统差异、绝对路径、Fiji/conda 位置和少量平台入口脚本不同之外，代码逻辑、步骤顺序、数据目录结构、参数含义和文档说明都应一致。

## 先设置绝对路径

```bash
REPO_ROOT=/absolute/path/to/calcium-imaging-pipeline-new
DATA_ROOT=/absolute/path/to/DATA_ROOT
```

假设后续命令都在代码目录执行：

```bash
cd "$REPO_ROOT"
```

## 适用范围

这套工作站入口主要覆盖：

- `00`: 原始 OIR 整理
- `01`: Fiji / Bio-Formats 转 TIFF
- `02`: 生成 stim map
- `03`: motion correction
- `04`: spatial high-pass
- `05`: suite2p ROI candidate generation

`manual` GUI 可以在 Linux 上单独开，但不是这套 overnight runner 的默认目标。

## 需要准备什么

- conda 环境名与主仓库约定一致：
  - `fiji_env`
  - `caiman`
  - `suite2p`
- 已安装 Fiji / Bio-Formats
- 运行前指定 Fiji 程序位置：

```bash
export FIJI_BIN=/path/to/Fiji.app/ImageJ-linux64
```

或：

```bash
export FIJI_BIN=/path/to/Fiji/fiji-linux-x64
```

## 先检查环境

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/check_linux_setup.sh
```

如果需要先导出当前机器 conda 环境概况：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/export_conda_envs.sh
```

## 先 dry-run

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT" --dry-run
```

这个命令只显示计划执行的步骤，不会处理真实数据。

## 刺激记录整理

建议把外部刺激程序导出的原始记录统一放到：

```text
DATA_ROOT/00_stim_logs_raw/
```

预览整理计划：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/prepare_stim_logs.sh "$DATA_ROOT" --organize-raw
```

确认后执行：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/prepare_stim_logs.sh "$DATA_ROOT" --organize-raw --overwrite --execute
```

## 清理旧中间结果

预览：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/clean_generated_outputs.sh "$DATA_ROOT"
```

执行：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/clean_generated_outputs.sh "$DATA_ROOT" --execute
```

## 正式跑 00-05

推荐放到 `tmux` 或 `screen`：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

默认使用：

```bash
ACTION=skip
```

也就是已有输出会跳过，适合续跑。

明确重跑时：

```bash
ACTION=overwrite bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

## 常用环境变量

```bash
FIJI_MEMORY=64G \
SUITE2P_THREADS=8 \
N_WORKERS=1 \
NUM_THREADS=1 \
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

含义：

- `FIJI_MEMORY`: Fiji Java 内存
- `SUITE2P_THREADS`: suite2p 线程数
- `N_WORKERS`: 同时处理多少个 trial
- `NUM_THREADS`: 每个 worker 内部 BLAS/OpenMP 线程数

## English

## Core Principle

The Linux workstation and the Mac copy should stay aligned as one workflow.
Paths, OS-specific launchers, and Fiji/conda locations may differ, but code
logic, step ordering, data layout, parameter meaning, and documentation should
stay synchronized.

## Set Absolute Paths First

```bash
REPO_ROOT=/absolute/path/to/calcium-imaging-pipeline-new
DATA_ROOT=/absolute/path/to/DATA_ROOT
```

The commands below assume:

```bash
cd "$REPO_ROOT"
```

## Scope

These workstation runners mainly cover:

- `00`: organize raw OIR files
- `01`: Fiji / Bio-Formats TIFF conversion
- `02`: stimulus-map generation
- `03`: motion correction
- `04`: spatial high-pass outputs
- `05`: suite2p ROI candidate generation

The `manual` GUI can also run on Linux, but it is not the main target of this
overnight runner layer.

## Requirements

- conda environment names aligned with the main repository:
  - `fiji_env`
  - `caiman`
  - `suite2p`
- Fiji / Bio-Formats installed
- `FIJI_BIN` exported before use:

```bash
export FIJI_BIN=/path/to/Fiji.app/ImageJ-linux64
```

or:

```bash
export FIJI_BIN=/path/to/Fiji/fiji-linux-x64
```

## Check the Setup First

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/check_linux_setup.sh
```

If you want a quick conda-environment export first:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/export_conda_envs.sh
```

## Dry Run First

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT" --dry-run
```

This previews the planned work without processing real data.

## Organize Stimulus Logs

Keep raw external stimulus-program records under:

```text
DATA_ROOT/00_stim_logs_raw/
```

Preview:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/prepare_stim_logs.sh "$DATA_ROOT" --organize-raw
```

Execute:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/prepare_stim_logs.sh "$DATA_ROOT" --organize-raw --overwrite --execute
```

## Clean Older Generated Outputs

Preview:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/clean_generated_outputs.sh "$DATA_ROOT"
```

Execute:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/clean_generated_outputs.sh "$DATA_ROOT" --execute
```

## Run Steps 00-05

Recommended inside `tmux` or `screen`:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

Default behavior:

```bash
ACTION=skip
```

That means existing outputs are skipped for safe resume behavior.

Explicit rerun:

```bash
ACTION=overwrite bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

## Common Environment Variables

```bash
FIJI_MEMORY=64G \
SUITE2P_THREADS=8 \
N_WORKERS=1 \
NUM_THREADS=1 \
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

Meaning:

- `FIJI_MEMORY`: Fiji Java memory
- `SUITE2P_THREADS`: suite2p worker threads
- `N_WORKERS`: number of trials processed in parallel
- `NUM_THREADS`: BLAS/OpenMP threads used inside each worker
