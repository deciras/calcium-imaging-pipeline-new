# Current Code Layout

`run_pipeline.py` is the main launcher. Step scripts are grouped by role:

- `preprocess/`: data organization, Fiji conversion, stimulus maps, motion correction, and spatial high-pass movies.
- `roi/`: suite2p ROI candidate generation and the manual ROI curation GUI.
- `analysis/`: post-manual trace extraction, response analysis, population analysis, summaries, and reports.
- `tools/`: maintenance utilities that are not part of the normal numbered pipeline.

Use `python3 current/run_pipeline.py --list-steps` to see the runnable steps.

Useful step groups:

- `premanual`: steps 00-05, through suite2p ROI candidate generation.
- `manual`: open the manual ROI curation GUI.
- `basic-analysis`: steps 06-09, from final ROI trace extraction through angle tuning.
- `core-analysis`: steps 06, 08, 09, and 10-14; the recommended compact post-manual path for tuning, trace plots, stimulus-slice features, and response-pattern clustering.
- `postmanual`: steps 06-18, all automated analysis after manual ROI curation.

For GUI integration, `basic-analysis` can be limited to one trial with `--trial-id`.

## Post-Manual QC Plot Output

Step 06 can write compact QC figures next to each trial's normal numeric
outputs. The unified plot controls are currently wired only to step 06; later
06-18 steps should be added one script at a time rather than receiving unknown
options.

- `<trial_id>_dff_heatmap_sorted.png/pdf`: ROI-by-time dF/F heatmap, sorted by activity, with stimulus timing when available.
- `<trial_id>_dff_trace_examples_with_stim.png/pdf`: deterministic representative ROI traces with stimulus timing.
- `<trial_id>_dff_distribution.png/pdf`: ROI-level max dF/F, trace SD, F0, and SNR-like distributions.
- With `--plot-level full`, step 06 also writes neuropil correction examples and an ROI spatial activity map when coordinates are available.

Run the post-manual chain with basic QC plots:

```bash
python3 /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06,07,08,09,10,11,12,13,14,15,16,17,18 \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip \
  --plot-level basic
```

Refresh step-06 QC plots from existing numeric outputs without re-extracting
traces:

```bash
python3 /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new/current/run_pipeline.py \
  --steps 06 \
  --data-root /absolute/path/to/DATA_ROOT \
  --action skip \
  --refresh-plots \
  --plot-level basic
```

`--plot-level none` disables the new QC figures, `basic` writes the core quick
QC plots, and `full` adds more diagnostic example plots. `--plot-format` accepts
`png`, `pdf`, or `both`, and `--plot-dpi` controls figure resolution.

Step 06 ROI source defaults to `auto`: final manual curation output is preferred,
and trials without final manual ROI fall back to suite2p `iscell.npy`. The old
05c curated-ROI route is archived legacy code, not a current default source.
Use `--roi-source manual` only when trials without final manual ROI should be
skipped explicitly.

Post-manual analysis defaults:

- Step 08 uses a 10 s pre-stimulus baseline, a 6 s response window, raw dF/F responses, `z >= 3`, and a minimum baseline standard deviation of 0.02 dF/F. Optional smoothing can be enabled with `--smooth-method rolling-median --smooth-window-sec 3`.
- Peak detection in steps 08 and `trace` is global per ROI: candidate peaks are detected on the full trace using a robust sigma threshold, then only peaks inside stimulus response windows are used. `response_peak` follows this global-peak rule; `window_peak` is kept as the raw window maximum for auditing.
- Step 09 marks angle-selective ROIs only when OSI is at least 0.3 and the preferred angle is reliable in at least half of repeats.
- Step `trace` plots raw dF/F and finds peak markers from raw dF/F by default. Single-ROI plots use a full dynamic y-axis per ROI so large peaks are visible instead of clipped. It writes both real-scale ROI plots in `roi_traces/` and normalized ROI plots in `roi_traces_normalized/`; use `--trace-scale dff` or `--trace-scale normalized` to write only one set. Optional robust clipped display can be enabled with `--y-axis-mode robust`, and optional smoothed overlays/peak finding with `--smooth-method rolling-median --smooth-window-sec 3`. It does not use deconvolution by default.
- Step 12 converts step-08 peri-stimulus tensors into stimulus-slice features and slice-level response calls. Each ROI x stimulus slice is normalized by its own pre-stimulus baseline median and robust MAD/std scale. The default `stimulus_evoked_flag` requires amplitude, fast onset, and recovery evidence: peak z >= 3 or mean z >= 1.5 with positive AUC >= 1.0; sustained crossing within 3 s after onset; and recovery to abs(z) <= 1 within 6 s after offset when an offset window is available. The softer `stimulus_response_score` is also saved for ranking and QC.
- Steps 13, 14, and 16 default to stimulus-slice features (`--similarity-source slices`, `--cluster-source slices`, `--embedding-source slices`). This is the main functional clustering route. Full-trace clustering remains available with `--cluster-source traces` as a QC view for drift, bleaching, spontaneous activity, or tissue-state effects. Use `--cluster-normalization both` to write normalized primary cluster outputs plus raw/source-scale cluster-control outputs.

Trial naming workflow:

- Use `tools/generate_trial_id_manifest.py` to generate a proposed canonical naming manifest before renaming any data or outputs.
- The script does not rename files. It writes:
  - `00_trial_metadata/trial_id_manifest.csv`
  - `00_trial_metadata/trial_id_manifest_collisions.csv`
  - `00_trial_metadata/trial_id_manifest_excluded.csv`
  - `00_trial_metadata/trial_id_manifest_summary.json`
- Trials from clearly separate non-retina series such as `WB_cells` are excluded from the main manifest and written to the excluded table instead of participating in collision counting.
- The proposed canonical name currently combines:
  - date
  - species
  - preparation
  - magnification
  - chloride condition
  - planned stimulus mode
  - AoLP availability
  - replicate suffix

Example:

```bash
python3 /Users/dingyifei/Documents/calcium-imaging-pipeline-new/calcium-imaging-pipeline-new/current/tools/generate_trial_id_manifest.py \
  --data-root /absolute/path/to/DATA_ROOT
```

Manual GUI notes:

- Step `manual` opens `roi/05_manual_roi_curation_gui.py`.
- The GUI supports synchronized zoom/pan across the reference and edit views:
  - `Ctrl` / `Cmd` + wheel zooms.
  - Pinch gestures zoom on supported trackpads.
  - Wheel / two-finger scroll pans after zooming.
  - `drag to pan` enables left-button panning; middle-button panning also works.
- The view can show both images, only the reference view, or only the edit view, and the two image positions can be swapped.
- Frame rendering caches the normalized base frame and static ROI overlays, so playback and frame scrubbing do not redraw every suite2p ROI on every frame.
- Manual ROI traces are updated lazily after final ROI set edits and forced current before saving.
- Undoing a just-finished manual ROI clears the finished drawing preview line as well as removing the ROI.
- The GUI has explicit `Save and close` and `Close` buttons. Closing the window also asks whether to save unsaved ROI edits.

## 中文速查

如果只是想快速知道 `current/` 里现在每个核心入口负责什么，可以先看这一段。

### 总入口

- `run_pipeline.py`
  - 整个 numbered pipeline 的统一入口
  - 负责步骤编号、参数透传、步骤组（`premanual` / `manual` / `postmanual`）和后台运行

### preprocess

- `00_oir_file_manager.py`
  - 整理原始 `.oir` 及配套文件
- `01_fiji_totif_*.py`
  - 调 Fiji / Bio-Formats 把 `.oir` 转成 TIFF 和 metadata
- `02_generate_stim_map.py`
  - 从 stimulus analog 与外部刺激日志生成 `stim_events / stim_map / stim_pulse_events`
  - 现在还会写每个 analog block 的诊断图和 `stim_block_summary.csv`
- `03_motion_correct_func_caiman.py`
  - 做 motion correction
- `04_spatial_highpass.py`
  - 生成 high-pass movie，主要服务于 ROI 边界显示和检测

### roi / manual

- `05_*`
  - suite2p ROI candidate generation
- `05_manual_roi_curation_gui.py`
  - 人工 ROI 校对与补画

### analysis

- `06_extract_dff.py`
  - 用 final ROI 从 motion-corrected movie 重新抽 F/Fneu/dF/F
  - 现在优先从 `02_stim_map/` 读取最新 stimulus sidecar，而不是盲信旧副本
- `07_detect_events.py`
  - 在全 trace 上做 calcium event detection
- `08_stim_response_analysis.py`
  - 真正开始按 stimulus timing 切 ROI trace
  - 现在优先读取 `02_stim_map/` 最新刺激时间；这是修复 `n_stimulus_slices = 0` 的关键
- `09_angle_tuning_analysis.py`
  - AoLP tuning
- `10_trace_plots.py`
  - ROI trace QC 图
- `11_population_features.py`
  - population-level 基础 feature
- `12_stimulus_slice_features.py`
  - stimulus-slice normalization、feature matrix、`stimulus_evoked_flag`
  - 现在还会生成每个 `ROI x stimulus slice` 的单独判定 panel
- `13_population_similarity.py`
  - slice-based similarity
- `14_hierarchical_clustering.py`
  - 层次聚类
- `15_leiden_community_detection.py`
  - 图聚类
- `16_dimensionality_reduction.py`
  - PCA / UMAP
- `17_cross_trial_summary.py`
  - 跨 trial 汇总
- `18_report_generator.py`
  - HTML 报告

### tools

- `tools/generate_trial_id_manifest.py`
  - 生成 trial 命名 manifest，不直接重命名数据

## 当前最重要的两条实现原则

1. stimulus timing 的权威来源是 `02_stim_map/`
   - 后面 06、08 不应该继续优先吃过期副本

2. ROI stimulus slicing 先切、后判
   - 只要 trial 有可用 stimulus timing，就应该先切 peri-stimulus ROI trace
   - `evoked 0/1` 是 step 12 在切片之后根据 amplitude / latency / recovery 再判
