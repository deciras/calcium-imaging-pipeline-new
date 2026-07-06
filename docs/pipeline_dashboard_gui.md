# Pipeline Dashboard GUI

<p align="right">
  <a href="#中文"><img alt="切换到中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-1f6feb?style=for-the-badge"></a>
  <a href="#english"><img alt="Switch to English" src="https://img.shields.io/badge/English-24292f?style=for-the-badge"></a>
</p>

This page describes the dashboard GUI that launches and monitors
`current/run_pipeline.py` without turning the GUI itself into the compute host.

本页说明用于启动和监控 `current/run_pipeline.py` 的 dashboard GUI。它本身只是控制器，不承担重计算。

## 中文

## 这个 dashboard 是做什么的

dashboard 是 `current/run_pipeline.py` 的图形控制层。

- 负责组织参数、预览命令、启动步骤、显示日志、终止子进程
- 不直接在 Qt 主线程里跑 Fiji、CaImAn、suite2p 或 post-manual analysis
- 重计算仍然在子进程里执行，所以窗口在运行过程中尽量保持可响应

## 启动方式

### macOS

推荐从代码目录直接启动：

```bash
conda run -n caiman python /absolute/path/to/calcium-imaging-pipeline-new/current/pipeline_dashboard_gui.py
```

也可以用双击入口：

- [current/launch_pipeline_dashboard.command](../current/launch_pipeline_dashboard.command)

### Linux workstation

推荐使用 Linux 专用薄启动层：

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/launch_pipeline_dashboard.sh "$DATA_ROOT"
```

还可以配合桌面入口：

- [linux_workstation/launch_pipeline_dashboard.desktop](../linux_workstation/launch_pipeline_dashboard.desktop)

### Windows

当前不维护一套独立 Windows 主代码，只提供薄包装启动层：

- [windows/launch_pipeline_dashboard.bat](../windows/launch_pipeline_dashboard.bat)
- [windows/launch_pipeline_dashboard.ps1](../windows/launch_pipeline_dashboard.ps1)
- [windows/README.md](../windows/README.md)

Windows 路径和 conda 安装位置差异较大，因此这层主要负责：

- 寻找 `conda.exe`
- 指定 dashboard 环境名
- 从仓库根目录启动 `current/pipeline_dashboard_gui.py`

## 典型使用流程

1. 设置 `Data root`
2. 如果默认值不对，设置 `Conda` 和 `Fiji` 路径
3. 选择步骤组，例如 `00-05 premanual`、`manual GUI`、`06-18 postmanual`
4. 用 `Plan` 只打印解析后的命令
5. 用 `Start` 正式运行并查看实时日志
6. 用 `Stop` 终止子进程

## 重要字段

- `Data root`
  - 当前实验数据根目录
- `Conda`
  - 运行 pipeline 子进程时使用的 conda 可执行文件
- `Fiji`
  - step 01 使用的 Fiji 路径
- `Trial id`
  - 只处理单个或多个指定 trial
- `Date id`
  - 只处理指定日期，例如 `20251205`
- `Extra args`
  - 透传给总入口的附加参数

## post-manual 相关控制

`Analysis input` 会影响：

- step 13 similarity
- step 14 hierarchical clustering
- step 16 PCA / UMAP

可选项：

- `Stimulus slices`
  - 推荐默认值
  - 基于 step 12 的 normalized peri-stimulus slice features
- `Full traces`
  - 使用完整 dF/F trace
  - 更适合把 drift、bleaching、自发活动或组织状态变化当作 QC 视图来看
- `Summary features`
  - 使用 step 11 的 scalar ROI metrics
- `Response scalars only`
  - 在支持的步骤中使用更紧凑的 response matrix

`Cluster scaling` 只控制 step 14：

- normalized
- raw/source-scale
- both

## dashboard 和记忆行为

- dashboard 会记住最近一次通过 `Plan`、`Start` 或 `Open manual GUI` 使用的 `Data root`
- 下次打开时，会优先回到上次工作目录

## manual GUI 入口

`Open manual GUI` 按钮并不是另写一套 GUI，它只是通过已有的 `manual` pipeline step 启动：

- [current/roi/05_manual_roi_curation_gui.py](../current/roi/05_manual_roi_curation_gui.py)

manual GUI 的行为和实现仍然独立于 dashboard。

## 默认 Fiji 路径和内存

默认 Fiji 路径：

- macOS: `/Applications/Fiji.app`
- Linux: `~/Fiji`, `~/Fiji.app`, `/opt/Fiji`, `/opt/Fiji.app`, `/usr/local/Fiji`

默认 Fiji memory：

- macOS: `16G`
- Linux workstation: `64G`

Fiji 字段保持可编辑：

- 用 `Browse` 选择其他 Fiji 安装位置
- 用 `Set default` 把当前路径保存为这台电脑之后的默认值

在 macOS 上，把 `/Applications/Fiji.app` 存成默认值是可以的；step 01 真正启动时会解析到 app bundle 里的可执行文件。

Fiji 路径优先级：

1. 在 dashboard 里通过 `Set default` 保存的路径
2. 环境变量 `FIJI_BIN` 或 `FIJI_PATH`
3. 平台默认路径

## 跨平台原则

这个 dashboard 不应该在 macOS、Linux、Windows 各维护一份主逻辑。

更合理的边界是：

- 共享一份主 GUI 代码：
  - [current/pipeline_dashboard_gui.py](../current/pipeline_dashboard_gui.py)
- 每个平台各自维护很薄的一层启动脚本：
  - macOS `.command`
  - Linux `.sh` / `.desktop`
  - Windows `.bat` / `.ps1`

应该分开的主要是：

- 启动方式
- 默认路径
- `conda` / Fiji 定位
- 桌面快捷方式

不应该分开的主要是：

- GUI 主逻辑
- 参数拼装规则
- `--trial-id` / `--date-id` 等 pipeline 控制逻辑

## English

## What The Dashboard Does

The dashboard is the GUI control layer for `current/run_pipeline.py`.

- It assembles arguments, previews commands, starts steps, shows logs, and stops child processes.
- It does not run Fiji, CaImAn, suite2p, or post-manual analysis directly on the Qt main thread.
- Heavy work still happens in child processes, so the window remains responsive during runs.

## Launch Options

### macOS

Recommended direct launch:

```bash
conda run -n caiman python /absolute/path/to/calcium-imaging-pipeline-new/current/pipeline_dashboard_gui.py
```

You can also use the double-click launcher:

- [current/launch_pipeline_dashboard.command](../current/launch_pipeline_dashboard.command)

### Linux workstation

Recommended Linux wrapper:

```bash
bash /absolute/path/to/calcium-imaging-pipeline-new/linux_workstation/launch_pipeline_dashboard.sh "$DATA_ROOT"
```

Optional desktop entry:

- [linux_workstation/launch_pipeline_dashboard.desktop](../linux_workstation/launch_pipeline_dashboard.desktop)

### Windows

The repository does not maintain a separate Windows copy of the dashboard logic.
Instead, it provides thin Windows launchers:

- [windows/launch_pipeline_dashboard.bat](../windows/launch_pipeline_dashboard.bat)
- [windows/launch_pipeline_dashboard.ps1](../windows/launch_pipeline_dashboard.ps1)
- [windows/README.md](../windows/README.md)

These wrappers mainly handle:

- locating `conda.exe`
- selecting the dashboard environment
- launching `current/pipeline_dashboard_gui.py` from the repository root

## Typical Workflow

1. Set `Data root`
2. Set `Conda` and `Fiji` paths if needed
3. Choose a step group such as `00-05 premanual`, `manual GUI`, or `06-18 postmanual`
4. Use `Plan` to print the resolved command only
5. Use `Start` to run and monitor live logs
6. Use `Stop` to terminate the child process

## Important Fields

- `Data root`
  - experiment data root
- `Conda`
  - conda executable used for child processes
- `Fiji`
  - Fiji path for step 01
- `Trial id`
  - restrict work to selected trial IDs
- `Date id`
  - restrict work to one date such as `20251205`
- `Extra args`
  - extra arguments forwarded to the launcher

## Post-Manual Controls

`Analysis input` affects:

- step 13 similarity
- step 14 hierarchical clustering
- step 16 PCA / UMAP

Options:

- `Stimulus slices`
  - recommended default
  - based on normalized peri-stimulus slice features from step 12
- `Full traces`
  - uses complete dF/F traces
  - useful as a QC view for drift, bleaching, spontaneous activity, or tissue-state effects
- `Summary features`
  - uses scalar ROI metrics from step 11
- `Response scalars only`
  - uses compact response matrices where supported

`Cluster scaling` affects step 14 only:

- normalized
- raw/source-scale
- both

## Persistence Behavior

- The dashboard remembers the most recent `Data root` used by `Plan`, `Start`, or `Open manual GUI`.
- The next launch reopens at that previous working location.

## Manual GUI Entry

The `Open manual GUI` button does not launch a second independent GUI stack. It
uses the existing `manual` pipeline step:

- [current/roi/05_manual_roi_curation_gui.py](../current/roi/05_manual_roi_curation_gui.py)

The manual GUI remains a separate implementation from the dashboard.

## Default Fiji Paths And Memory

Default Fiji paths:

- macOS: `/Applications/Fiji.app`
- Linux: `~/Fiji`, `~/Fiji.app`, `/opt/Fiji`, `/opt/Fiji.app`, `/usr/local/Fiji`

Default Fiji memory:

- macOS: `16G`
- Linux workstation: `64G`

The Fiji field remains editable:

- use `Browse` to choose another install
- use `Set default` to save that path for future launches on the same machine

On macOS, saving `/Applications/Fiji.app` is valid; the conversion step resolves
the actual executable inside the app bundle before launching Fiji.

Fiji path priority:

1. path saved with `Set default`
2. `FIJI_BIN` or `FIJI_PATH`
3. platform default

## Cross-Platform Principle

The dashboard should not maintain three separate copies of the main logic for
macOS, Linux, and Windows.

The intended split is:

- one shared GUI implementation:
  - [current/pipeline_dashboard_gui.py](../current/pipeline_dashboard_gui.py)
- one thin launcher layer per platform:
  - macOS `.command`
  - Linux `.sh` / `.desktop`
  - Windows `.bat` / `.ps1`

The parts that should differ by platform are mostly:

- launch style
- default paths
- `conda` / Fiji discovery
- desktop shortcuts

The parts that should stay shared are:

- dashboard GUI logic
- argument assembly
- pipeline control behavior such as `--trial-id` and `--date-id`
