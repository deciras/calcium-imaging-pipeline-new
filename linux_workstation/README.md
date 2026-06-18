# Linux Workstation Runner

这份入口是给 Linux 工作站 overnight 跑前半段用的。

核心原则：Linux workstation 和 Mac 应该即刻同步化。除了操作系统差异、绝对路径、Fiji/conda 位置和少量平台入口脚本不同之外，代码逻辑、步骤顺序、数据目录结构、文件命名、参数含义、README 说明和运行日志都要一致。不要在两台机器上维护两套不同规则。

你当前工作站上的代码目录是：

```bash
REPO_ROOT=/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/calcium-imaging-pipeline-new
```

如果原始日期文件夹、`*_motor_rotation`、旧版 `Processed_TIF_*` 都放在代码库的上一级，可以这样设置数据目录：

```bash
DATA_ROOT=/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/2026_olympus
```

本地和工作站统一使用同一套数据结构：

```text
DATA_ROOT/
  00_original_files/
  00_stim_logs_raw/
  01_oir_to_tif/
  02_stim_map/
  ...
```

其中 `00_original_files/` 只放显微镜原始数据，`00_stim_logs_raw/` 只放刺激控制程序导出的原始参数合集。

后面的命令都假设先进入代码目录：

```bash
cd "$REPO_ROOT"
```

范围：

- `00`: 整理 OIR 原始文件
- `01`: Fiji / Bio-Formats 转 TIFF
- `02`: 生成刺激 map
- `03`: motion correction
- `04`: spatial high-pass 底片
- `05`: suite2p ROI 候选生成

`05` 只跑 suite2p，不打开人工 GUI。manual GUI 也可以在 Linux 上单独启动，但入口是 `launch_manual_gui.sh`。

## 工作站需要准备什么

Linux 上需要有这几个 conda 环境，名字要和当前总入口一致：

- `fiji_env`: 用来启动 Fiji 转换
- `caiman`: 用来跑 02-04，也用于 manual GUI
- `suite2p`: 用来跑 05

还需要安装 Fiji / Bio-Formats，并在运行前指定 Fiji 程序位置：

```bash
export FIJI_BIN=/path/to/Fiji.app/ImageJ-linux64
```

如果你的 Fiji 目录是普通 Linux 解压版，也可能是：

```bash
export FIJI_BIN=/path/to/Fiji/fiji-linux-x64
```

## 先检查环境

进入代码库：

```bash
cd /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/calcium-imaging-pipeline-new
```

检查 conda、环境名、Fiji 路径和总入口：

```bash
bash linux_workstation/check_linux_setup.sh
```

如果不确定两台电脑的 conda 环境名字是否一致，可以先导出当前机器所有环境：

```bash
bash linux_workstation/export_conda_envs.sh
```

默认会写到代码库上一级的 `calcium_imaging/` 目录，例如：

```text
/mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/conda_env_exports_20260616_003000/
```

也可以指定导出目录：

```bash
bash linux_workstation/export_conda_envs.sh /mnt/50d357b2-473b-4533-8c74-1db99704876a/yifeiding/calcium_imaging/conda_env_exports_workstation
```

导出的内容包括 `conda_env_list.txt`、每个环境的 `.yml` 和包列表 `.txt`。

## 先 dry-run

把 `$DATA_ROOT` 换成工作站上的实验数据目录：

```bash
bash linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT" --dry-run
```

这个命令只显示会跑哪些步骤，不处理数据。

## 整理刺激参数合集

你的原始目录里有类似这样的刺激控制记录。建议把它们统一放在 `00_stim_logs_raw/`：

```text
00_stim_logs_raw/
  20260428_motor_rotation/
    timestamp_log_20260428_170422.csv
    stim_map_20260428_170422.csv
    experiment_config_20260428_170422.json
    20260428_170422_angle_list.txt
```

02 现在默认优先从 `$DATA_ROOT/00_stim_logs_raw/` 读取这些记录；如果老数据只有 `$DATA_ROOT/stim_logs/`，才会回退到旧入口。可以先预览整理计划：

```bash
bash linux_workstation/prepare_stim_logs.sh "$DATA_ROOT" --organize-raw
```

确认后移动顶层 `*_motor_rotation` 到 `00_stim_logs_raw/`，并创建/刷新兼容旧逻辑用的 `stim_logs/`：

```bash
bash linux_workstation/prepare_stim_logs.sh "$DATA_ROOT" --organize-raw --overwrite --execute
```

默认在 Linux 上用 symlink，不复制大批文件。`stim_logs/` 只是 flat index，不是原始刺激参数归档；原始合集仍以 `00_stim_logs_raw/` 为准。早期只有 `mcu_config_*.json`、没有 `experiment_config_*.json` 的记录，会自动生成一个兼容 02 的 `experiment_config_*.json`。

`run_00_05_suite2p.sh` 默认会在正式运行前自动执行这一步。想关闭可以：

```bash
PREPARE_STIM_LOGS=0 bash linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

## 清理旧中间结果

如果工作站上已经有之前跑出来的中间结果，先预览会删什么：

```bash
bash linux_workstation/clean_generated_outputs.sh "$DATA_ROOT"
```

默认只清理 01-05 输出，也会清理旧版 `Processed_TIF_*` 输出；不碰原始 `.oir` 数据，也不碰 `00_original_files`。

确认无误后再真正删除：

```bash
bash linux_workstation/clean_generated_outputs.sh "$DATA_ROOT" --execute
```

如果想把 manual GUI 和 06-16 后续分析输出也一起删：

```bash
bash linux_workstation/clean_generated_outputs.sh "$DATA_ROOT" --scope all-generated --execute
```

只有在你确认原始数据还有另一份备份时，才考虑加：

```bash
--include-organized-raw
```

## Overnight 跑 00-05

推荐放在 `tmux` 或 `screen` 里跑：

```bash
bash linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

默认是安全续跑模式：

```bash
ACTION=skip
```

也就是说，已经完成的 trial 会跳过。日志会写到：

```text
$DATA_ROOT/pipeline_logs/
```

如果你确认要重跑 00-05，可以显式覆盖：

```bash
ACTION=overwrite bash linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

## 工作站参数

常用参数用环境变量控制：

```bash
FIJI_MEMORY=64G \
SUITE2P_THREADS=8 \
N_WORKERS=1 \
NUM_THREADS=1 \
bash linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

含义：

- `FIJI_MEMORY`: Fiji Java 内存
- `SUITE2P_THREADS`: suite2p 自己用的线程数
- `N_WORKERS`: 同时处理几个 trial
- `NUM_THREADS`: 每个 worker 内部的 BLAS/OpenMP 线程数

默认 `METADATA_MODE=skip`，这样已有 step 01 输出时不会重新打开 OIR
刷新 metadata。只有需要补 metadata JSON 时再显式设置
`METADATA_MODE=update-missing`。

如果机器内存不是特别大，建议先保持 `N_WORKERS=1`，让 suite2p 在单个 trial 内多线程。

Linux runner 会自动创建并使用：

```text
$DATA_ROOT/.tmp
$DATA_ROOT/.cache
```

这些目录用于 Python、CaImAn、pynwb、matplotlib 等包的临时文件和缓存，避免 Linux 上误用 macOS 风格的 `/private/tmp`。如果需要改到别的位置，可以在启动前设置：

```bash
PIPELINE_TMPDIR=/tmp/calcium_pipeline_tmp \
PIPELINE_CACHE_DIR=/tmp/calcium_pipeline_cache \
bash linux_workstation/run_00_05_suite2p.sh "$DATA_ROOT"
```

## 单独打开 manual GUI

GUI 不属于 overnight 00-05。前半段跑完后，如果你想在 Linux 上看 ROI，可以单独启动：

```bash
bash linux_workstation/launch_manual_gui.sh "$DATA_ROOT"
```

直接打开某个 trial：

```bash
bash linux_workstation/launch_manual_gui.sh "$DATA_ROOT" 20260428_Euprymna_retina2_25x
```

如果工作站没有图形桌面，可以继续把 00-05 结果同步回本地，再在本地打开 GUI。

## 当前 ROI 主线

- `04_spatial_highpass` 仍然会生成，主要给 GUI 显示边界。
- `05_suite2p_roi_detection` 默认从 `03_motion_correct` 读 movie 跑 suite2p。
- suite2p 输出只作为 ROI 候选库。
- manual GUI 保存的最终 ROI set 才是后续分析优先使用的结果。
