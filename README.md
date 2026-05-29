# 钙成像数据处理流程

这个仓库保存的是头足类视网膜钙成像实验的数据处理代码，主要用于 `Euprymna berryi` 离体视网膜的 Olympus `.oir` 成像数据。

仓库里只放代码，不放真实成像数据。真实 `.oir`、`.tif/.tiff`、suite2p 输出和中间大文件都不应该提交到 GitHub。

## 当前目标

这个项目的重点不是重新设计整套分析，而是把已有流程变得更安全、更容易复现：

- 去掉写死的电脑路径
- 每一步默认跳过已有结果，而不是覆盖
- 支持 dry-run，先预览将要做什么
- 每一步检查输入是否存在
- 每一步打印处理、跳过、失败的总结
- 大文件处理保持保守，避免无意义重复计算
- 输出文件按步骤集中管理，避免散落在原始数据文件夹里

## 代码结构

当前维护的代码在：

```text
current/
```

历史版本在：

```text
archive/
```

除非明确需要追溯旧逻辑，日常请只使用 `current/`。

环境说明在：

```text
envs/
```

## 推荐数据结构

原始数据目录可以是类似这样的结构：

```text
test_dataset/
  00_original_files/
    metadata.csv
    trial_A/
      trial_A.oir
      trial_A_00001
      ...
    trial_B/
      trial_B.oir
      trial_B_00001
      ...
  01_oir_to_tif/
  02_stim_map/
  03_motion_correct/
  04_spatial_highpass/
  05_suite2p_roi_detection/
```

这样原始数据和每一步生成的结果按步骤并列，不会混在一起。`00` 会把根目录下已有的原始 trial 文件夹自动收进 `00_original_files/`。

## 总入口

推荐从总入口运行：

```bash
python3 current/run_pipeline.py --steps 00 --data-root ../test_dataset --step-dry-run
```

常用参数：

- `--steps 00`：只运行指定步骤，也可以写 `01,02`
- `--data-root ../test_dataset`：指定数据目录
- `--output-root PATH`：指定统一输出目录；不写时默认是 `DATA_ROOT`
- `--dry-run`：只显示总入口将要调用什么命令，不启动子脚本
- `--step-dry-run`：启动子脚本，但让子脚本自己只预览、不改文件
- `--action skip`：遇到已有结果时跳过，默认就是 `skip`
- `--action overwrite`：只覆盖当前步骤自己生成的文件，谨慎使用
- `--projection-mode auto`：第 `01` 步投影策略，默认自动平衡速度和内存
- `--stim-export-mode projected`：第 `01` 步刺激通道导出策略，默认保存省空间的投影版
- `--metadata-mode update-missing`：第 `01` 步遇到已有 TIFF 时，只补缺失的轻量 metadata，不重转 TIFF

## 当前步骤

### 00 整理 `.oir` 文件

脚本：

```text
current/00_oir_file_manager.py
```

作用：

- 把同一个 trial 的 `.oir` 和伴随文件放进同名 trial 文件夹
- 生成或更新 `metadata.csv`
- 默认不覆盖已有文件

预览：

```bash
python3 current/run_pipeline.py --steps 00 --data-root ../test_dataset --step-dry-run
```

真正执行：

```bash
python3 current/run_pipeline.py --steps 00 --data-root ../test_dataset
```

### 01 Fiji / Bio-Formats 转 TIFF

脚本：

```text
current/01_fiji_totif_ini.py
current/01_fiji_totif_worker.py
```

作用：

- 用 Fiji/Bio-Formats 读取 Olympus `.oir`
- 输出 `_Max_Proj.tif`
- 如果有刺激模拟通道，输出 `_Stim_Analog.tif`
- 输出 `_metadata.json`，其中会尽量保存 `.oir` 内部的绝对采集开始时间 `acquisition.start_time`

默认输出到：

```text
DATA_ROOT/01_oir_to_tif/
```

Mac 上会尝试自动寻找：

```text
/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx
```

预览：

```bash
python3 current/run_pipeline.py --steps 01 --data-root ../test_dataset --step-dry-run
```

投影策略：

- `--projection-mode auto`：默认。小文件用较快策略，大文件自动切到省内存策略
- `--projection-mode safe`：逐时间点投影，慢一些，但峰值内存低，适合 Mac 和大文件
- `--projection-mode fast`：整通道复制后投影，快一些，但可能占用大量 Java heap，适合内存充足的工作站

例如 Linux 工作站内存充足时可以尝试：

```bash
python3 current/run_pipeline.py --steps 01 --data-root /path/to/data --fiji-memory 32g --projection-mode auto
```

刺激通道导出策略：

- `--stim-export-mode projected`：默认。保存当前使用的 `_Stim_Analog.tif`，体积较小，适合常规流程
- `--stim-export-mode raw`：保存 `_Stim_Analog_Raw/` 文件夹，不做 Z projection；每个时间点保存一个小 z-stack TIFF，后续可用于更精细地从刺激通道检测 pulse，但文件会明显变多
- `--stim-export-mode both`：同时保存 projected 和 raw 两种版本

建议平时继续用默认值。只有在需要追踪很短的 flash/pulse，而且愿意多占用磁盘空间时，才用 `raw` 或 `both`。

注意：当前 step 02 仍默认读取 `_Stim_Analog.tif`。`_Stim_Analog_Raw/` 是为下一步“从未投影刺激通道逐 flash 检测”预留的输入；在 step 02 支持 raw 检测之前，常规流程请使用 `projected` 或 `both`，不要只用 `raw`。

如果某个 Olympus `.oir` 是分卷保存的，step 01 会先检查旁边的 `_00001`、`_00002` 等分卷是否连续。缺分卷的 trial 会被标记为 invalid/skipped，不生成 TIFF，也会清理本步骤残留输出，避免后续误用不完整数据。

metadata 刷新策略：

- `--metadata-mode update-missing`：默认。已有 TIFF 时跳过 TIFF 生成，但如果 `_metadata.json` 缺少 `acquisition.start_time`，会从 `.oir` 中轻量读取并补上
- `--metadata-mode skip`：完全不改已有 metadata
- `--metadata-mode refresh`：已有 TIFF 时也刷新可轻量更新的 metadata 字段

这个刷新不会重写 `_Max_Proj.tif` 或 `_Stim_Analog.tif`。

### 02 提取刺激时间

脚本：

```text
current/02_generate_stim_map.py
```

作用：

- 从 `_Stim_Analog.tif` 提取亮度曲线
- 识别刺激开始和结束时间
- 可选读取刺激控制程序日志，把刺激模式、角度和 MCU 时间并入结果
- 输出 brightness trace、stim map、stim events 和实际电压预览图
- 对有刺激事件的 trial 输出 `*_stim_schematic.png`，在同一条时间轴上画出刺激时间示意
- 对 pulse 模式额外输出 `stim_pulse_events.csv`，把每个 flash 单独列出来
- 对 pulse 模式额外保留 `*_stim_pulse_trace.png`，把 packet 里的多个 flash 画成示意线图

默认输入：

```text
DATA_ROOT/01_oir_to_tif/
```

默认输出：

```text
DATA_ROOT/02_stim_map/
```

预览：

```bash
python3 current/run_pipeline.py --steps 02 --data-root ../test_dataset --step-dry-run
```

如果有类似 `timestamp_log_*.csv`、`stim_map_*.csv`、`experiment_config_*.json` 的刺激控制日志，可以放在：

```text
DATA_ROOT/test_dataset_motor_rotation/
```

也可以额外传入日志文件夹：

```bash
python3 current/run_pipeline.py \
  --steps 02 \
  --data-root ../test_dataset \
  --stim-log-root /path/to/stim_logs
```

这个匹配不依赖显微镜 trial 文件名和刺激日志文件名完全一致。脚本会优先使用 `_metadata.json` 里的 `.oir` 绝对采集开始时间，和刺激日志里的 host timestamp 对齐；匹配到刺激日志的 trial 才会被认为有刺激，匹配不到的 trial 标记为 `nostim`。电压阈值检测只作为 analog 质量检查和图上参考，不再决定“有没有刺激”。匹配成功后，`stim_events.csv` 会额外包含 `stim_protocol_id`、`pol_angle`、`mcu_onset_sec`、`mcu_offset_sec`、`host_timestamp`、`timing_source`、`protocol_match_delta_sec` 等列。

`stim_events.csv` 表示根据刺激日志和绝对时间戳匹配到的刺激事件。对 pulse 模式，每个 flash 会另存到 `stim_pulse_events.csv`。如果 analog 通道能可靠检测到 packet，会同时记录 analog 辅助时间；如果不能可靠检测到，则仍会使用刺激日志的绝对时间生成 pulse 时间表。

图像输出里，`*_stim_trace.png` 保留实际从 stim analog 读到的电压/亮度波形；`*_stim_schematic.png` 是刺激时间示意图，sustain trial 画连续刺激块，pulse trial 画 packet 和每个 flash。对 pulse trial，`*_stim_pulse_trace.png` 也会保留一份，方便和旧输出习惯兼容。每张 figure 都会同时保存 `.png` 和 `.pdf`：PNG 用于快速预览，PDF 用于后续编辑。电压图和示意图都保持单行完整时间轴；刺激角度使用固定的半圆色盘映射，同一角度在所有 trial 中颜色一致。

如果 step 01 已经生成 `_Stim_Analog_Raw/`，step 02 默认会优先读取 raw 文件夹：

```bash
python3 current/run_pipeline.py \
  --steps 02 \
  --data-root ../test_dataset \
  --action overwrite \
  --analog-source auto \
  --raw-z-strategy planes
```

- `--analog-source auto`：默认。优先使用 `_Stim_Analog_Raw/`，没有 raw 时才回退到 `_Stim_Analog.tif`
- `--analog-source raw`：强制使用 raw；如果 raw 不完整就报错
- `--analog-source projected`：强制使用旧的 `_Stim_Analog.tif`
- `--raw-z-strategy planes`：默认。把每个时间点里的 z 平面按顺序展开成更密的时间轴，尽量保留短 pulse
- `--raw-z-strategy max-range`：从 raw z-stack 中选择随时间变化最大的单个 z 平面作为电压曲线

### 03 CaImAn 运动校正

脚本：

```text
current/03_motion_correct_func_caiman.py
```

作用：

- 从 `01_oir_to_tif/` 读取 `_Max_Proj.tif`
- 用 CaImAn 做 motion correction
- 输出到 `03_motion_correct/`
- 复制 metadata 和 step 02 的刺激 CSV 表格，方便后续步骤继续追踪；刺激曲线图只保留在 `02_stim_map/`

预览：

```bash
python3 current/run_pipeline.py --steps 03 --data-root ../test_dataset --step-dry-run
```

真实运行前建议先 dry-run。该步骤会生成 corrected movie，计算和存储成本都明显高于 `00-02`。

### 04 空间高通滤波

脚本：

```text
current/04_spatial_highpass.py
```

作用：

- 从 `03_motion_correct/` 读取 corrected movie
- 逐帧做空间高通：

```text
highpass = frame - gaussian_blur(frame, sigma)
```

- 输出到 `04_spatial_highpass/`
- 逐帧读取、逐帧写出，避免一次性把整部 movie 放进内存

预览：

```bash
python3 current/run_pipeline.py --steps 04 --data-root ../test_dataset --step-dry-run
```

### 05 suite2p ROI 检测

脚本：

```text
current/05_suite2p_roi_detection_schema_aligned_connected.py
```

作用：

- 优先从 `04_spatial_highpass/` 读取 movie
- 如果 step 04 还没跑，则回退到 `03_motion_correct/`
- suite2p 输出写入 `05_suite2p_roi_detection/`
- 不在输入 movie 文件夹里原地生成 `suite2p/`
- 默认保守参数，继续偏向“少一些但更可信的 ROI”
- 导出简单 benchmark 文件：
  - `roi_mask.tif`
  - `roi_label_map.tif`
  - `roi_summary.csv`
  - `roi_overlay.png`

预览：

```bash
python3 current/run_pipeline.py --steps 05 --data-root ../test_dataset --step-dry-run
```

注意：

- 当前推荐 suite2p `0.14.4`
- `0.14.5` 曾在这批数据上产生异常 ROI 行为
- 脚本会检查 suite2p 版本；如需版本不匹配时直接失败，可加 `--step-args '--strict-suite2p-version'`

### 后续步骤

后续步骤正在逐步改造：

- ROI 文件导出
- dF/F 提取
- stimulus response analysis

suite2p 对环境更敏感，不建议在未 dry-run 检查前直接跑真实数据。

## Conda 环境

这台 Mac 上已经看到几个可用环境：

- `fiji_env`：Fiji/ImageJ 相关
- `caiman`：CaImAn、图像处理、`matplotlib`、`tifffile`
- `suite2p`：suite2p，当前偏好版本是 `0.14.4`

环境配置草案在：

```text
envs/
```

检查环境：

```bash
conda env list
conda run -n fiji_env python -c "import imageio, tifffile, numpy"
conda run -n caiman python -c "import caiman, cv2, tifffile, numpy"
conda run -n suite2p python -c "import suite2p; print(suite2p.__version__)"
```

## 安全原则

请不要提交这些文件：

- `.oir`
- `.oib`
- `.tif`
- `.tiff`
- `.npy`
- `.npz`
- suite2p 输出文件夹
- 大型中间结果和视频

运行真实数据前，优先使用：

```bash
--dry-run
--step-dry-run
```

确认路径和输出位置正确后，再去掉 dry-run 执行。

## 当前开发状态

已经完成：

- `00`：支持 CLI、dry-run、skip/overwrite、安全整理 test 数据
- `01`：支持 Mac/Linux 路径配置、dry-run、新输出结构
- `02`：支持 CLI、dry-run、skip/overwrite、新输出结构
- `run_pipeline.py`：支持总入口参数和步骤选择

待继续：

- 改造 `03` 运动校正
- 改造 `04` 空间高通
- 改造 `05` suite2p ROI 检测
- 后续再接回 `06/07` 分析步骤
