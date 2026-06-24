#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

Safe launcher for the maintained calcium imaging pipeline.

This file starts one or more step scripts. It does not delete data, does not
rewrite outputs by itself, and defaults to the local ``current/`` folder instead
of a hardcoded machine-specific path.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger("run_pipeline")


@dataclass(frozen=True)
class PipelineStep:
    """One runnable pipeline step."""

    step_id: str
    name: str
    script: str
    env: str | None = None
    enabled: bool = True
    accepts_data_root: bool = False
    accepts_layout: bool = False
    accepts_action: bool = False
    accepts_step_dry_run: bool = False
    accepts_output_root: bool = False
    accepts_fiji_memory: bool = False
    accepts_projection_mode: bool = False
    accepts_stim_export_mode: bool = False
    accepts_stim_log_root: bool = False
    accepts_metadata_mode: bool = False
    accepts_analog_source: bool = False
    accepts_raw_z_strategy: bool = False
    accepts_sigma_px: bool = False
    accepts_clip_negative: bool = False
    accepts_dpi: bool = False
    accepts_suite2p_options: bool = False
    accepts_dff_options: bool = False
    accepts_event_options: bool = False
    accepts_stim_response_options: bool = False
    accepts_angle_options: bool = False
    accepts_trace_options: bool = False
    accepts_similarity_options: bool = False
    accepts_clustering_options: bool = False
    accepts_embedding_options: bool = False
    accepts_leiden_options: bool = False
    accepts_roi_gui_options: bool = False
    accepts_qc_plot_options: bool = False
    accepts_trial_id: bool = False


# ``env=None`` means: run with the same Python that launched run_pipeline.py.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        step_id="00",
        name="整理 OIR 原始文件",
        script="preprocess/00_oir_file_manager.py",
        accepts_data_root=True,
        accepts_layout=True,
        accepts_action=True,
        accepts_step_dry_run=True,
    ),
    PipelineStep(
        step_id="01",
        name="Fiji 将 OIR 转为 TIF",
        script="preprocess/01_fiji_totif_ini.py",
        env="fiji_env",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_fiji_memory=True,
        accepts_projection_mode=True,
        accepts_stim_export_mode=True,
        accepts_metadata_mode=True,
    ),
    PipelineStep(
        step_id="02",
        name="生成刺激映射",
        script="preprocess/02_generate_stim_map.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_stim_log_root=True,
        accepts_analog_source=True,
        accepts_raw_z_strategy=True,
    ),
    PipelineStep(
        step_id="03",
        name="CaImAn 运动校正",
        script="preprocess/03_motion_correct_func_caiman.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
    PipelineStep(
        step_id="04",
        name="空间高通滤波",
        script="preprocess/04_spatial_highpass.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_sigma_px=True,
        accepts_clip_negative=True,
        accepts_dpi=True,
    ),
    PipelineStep(
        step_id="05",
        name="suite2p ROI 检测",
        script="roi/05_suite2p_roi_detection_schema_aligned_connected.py",
        env="suite2p",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_suite2p_options=True,
    ),
    PipelineStep(
        step_id="cellpose",
        name="Cellpose ROI 分割",
        script="roi/05_cellpose_roi_segmentation.py",
        env="czi_cellpose",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="manual",
        name="手动 ROI 筛选界面",
        script="roi/05_manual_roi_curation_gui.py",
        env="caiman",
        accepts_data_root=True,
        accepts_roi_gui_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="06",
        name="提取 dF/F",
        script="analysis/06_extract_dff.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_dff_options=True,
        accepts_qc_plot_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="07",
        name="检测钙事件",
        script="analysis/07_detect_events.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_event_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="08",
        name="刺激响应分析",
        script="analysis/08_stim_response_analysis.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_stim_response_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="09",
        name="偏振角调谐分析",
        script="analysis/09_angle_tuning_analysis.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_angle_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="10",
        name="绘制 ROI 曲线",
        script="analysis/10_trace_plots.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_trace_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="11",
        name="群体特征提取",
        script="analysis/11_population_features.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="12",
        name="刺激切片特征提取",
        script="analysis/12_stimulus_slice_features.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="13",
        name="群体相似性分析",
        script="analysis/13_population_similarity.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_similarity_options=True,
    ),
    PipelineStep(
        step_id="14",
        name="层次聚类",
        script="analysis/14_hierarchical_clustering.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_clustering_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="15",
        name="Leiden 社区检测",
        script="analysis/15_leiden_community_detection.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_leiden_options=True,
        accepts_trial_id=True,
    ),
    PipelineStep(
        step_id="16",
        name="降维分析",
        script="analysis/16_dimensionality_reduction.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_embedding_options=True,
    ),
    PipelineStep(
        step_id="17",
        name="跨 trial 汇总",
        script="analysis/17_cross_trial_summary.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
    PipelineStep(
        step_id="18",
        name="生成报告",
        script="analysis/18_report_generator.py",
        env="postmanual_analysis",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
)


@dataclass(frozen=True)
class ResolvedStep:
    """A pipeline step after its script path has been checked."""

    config: PipelineStep
    script_path: Path
    command: tuple[str, ...]


def configure_logging(verbose: bool) -> None:
    """Set up simple console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def banner(text: str, char: str = "=") -> None:
    line = char * 72
    LOGGER.info("\n%s\n%s\n%s", line, text, line)


def default_code_dir() -> Path:
    """Use the folder containing this launcher as the default code directory."""
    return Path(__file__).resolve().parent


def normalize_env_name(env_name: str | None) -> str | None:
    if env_name is None:
        return None

    clean = str(env_name).strip()
    if clean == "" or clean.lower() in {"current", "same", "self", "none"}:
        return None

    return clean


def default_stim_log_root(data_root: Path | None) -> Path | None:
    """Prefer the shared raw stimulus-log layout, with legacy fallback."""
    if data_root is None:
        return None

    for candidate_name in ("00_stim_logs_raw", "stim_logs"):
        candidate = data_root / candidate_name
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def parse_step_selector(raw_selector: str | None) -> set[str] | None:
    """
    Convert ``--steps`` text into lowercase selectors.

    Examples
    --------
    ``00,02,05``
    ``stim,suite2p``
    """
    if raw_selector is None:
        return None

    selectors = {
        item.strip().lower()
        for item in raw_selector.split(",")
        if item.strip()
    }
    if not selectors:
        raise ValueError("--steps was provided, but no valid step names were found.")

    step_groups = {
        "premanual": {"00", "01", "02", "03", "04", "05", "cellpose"},
        "pre-manual": {"00", "01", "02", "03", "04", "05", "cellpose"},
        "before-manual": {"00", "01", "02", "03", "04", "05", "cellpose"},
        "basic-analysis": {"06", "07", "08", "09"},
        "basicanalysis": {"06", "07", "08", "09"},
        "basic": {"06", "07", "08", "09"},
        "core-analysis": {"06", "08", "09", "10", "11", "12", "13", "14"},
        "core": {"06", "08", "09", "10", "11", "12", "13", "14"},
        "tuning-analysis": {"06", "08", "09", "10", "11", "12", "13", "14"},
        "postmanual": {"06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"},
        "post-manual": {"06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"},
        "after-manual": {"06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"},
        "manual-curation": {"manual"},
    }
    expanded: set[str] = set()
    for selector in selectors:
        expanded.update(step_groups.get(selector, {selector}))
    return expanded


def step_matches_selector(step: PipelineStep, selectors: set[str]) -> bool:
    text = f"{step.step_id} {step.name} {step.script}".lower()
    for selector in selectors:
        if selector == step.step_id:
            return True
        if selector.isdigit():
            continue
        if selector in text:
            return True
    return False


def select_steps(
    steps: tuple[PipelineStep, ...],
    raw_selector: str | None,
    from_step: str | None,
    to_step: str | None,
) -> list[PipelineStep]:
    """Apply CLI step filters."""
    active_steps = [step for step in steps if step.enabled]
    selectors = parse_step_selector(raw_selector)

    if selectors is not None:
        selected = [
            step for step in active_steps if step_matches_selector(step, selectors)
        ]
        if not selected:
            raise ValueError(
                f"No pipeline steps matched --steps={raw_selector!r}. "
                "Use --list-steps to see valid choices."
            )
    else:
        selected = active_steps

    if from_step is not None:
        from_step = from_step.strip().lower()
        matching_indexes = [
            i for i, step in enumerate(selected) if step_matches_selector(step, {from_step})
        ]
        if not matching_indexes:
            raise ValueError(f"No selected step matched --from-step={from_step!r}.")
        selected = selected[matching_indexes[0] :]

    if to_step is not None:
        to_step = to_step.strip().lower()
        matching_indexes = [
            i for i, step in enumerate(selected) if step_matches_selector(step, {to_step})
        ]
        if not matching_indexes:
            raise ValueError(f"No selected step matched --to-step={to_step!r}.")
        selected = selected[: matching_indexes[-1] + 1]

    if not selected:
        raise ValueError("No pipeline steps were selected.")

    return selected


def split_passthrough_args(raw_args: list[str] | None) -> list[str]:
    """Parse extra arguments that should be forwarded to every step script."""
    if not raw_args:
        return []

    parsed: list[str] = []
    for item in raw_args:
        parsed.extend(shlex.split(item))
    return parsed


def build_managed_step_args(
    step: PipelineStep,
    data_root: Path | None,
    output_root: Path | None,
    layout: str,
    action: str,
    step_dry_run: bool,
    fiji_memory: str | None,
    projection_mode: str | None,
    stim_export_mode: str | None,
    stim_log_root: Path | None,
    metadata_mode: str | None,
    analog_source: str | None,
    raw_z_strategy: str | None,
    sigma_px: float | None,
    clip_negative: bool,
    dpi: int | None,
    strict_suite2p_version: bool,
    n_workers: int | None,
    num_threads: int | None,
    suite2p_threads: int | None,
    batch_size: int | None,
    min_frames: int | None,
    cell_diameter_um: float | None,
    min_diameter_px: int | None,
    diameter_scale: float | None,
    threshold_scaling: float | None,
    max_overlap: float | None,
    snr_thresh: float | None,
    high_pass: int | None,
    spatial_hp_detect: int | None,
    max_iterations: int | None,
    inner_neuropil_radius: int | None,
    min_neuropil_pixels: int | None,
    tau: float | None,
    overlay_dpi: int | None,
    neuropil_coeff: float | None,
    trial_id: str | None,
    movie_kind: str | None,
    roi_source: str | None,
    f0_mode: str | None,
    f0_percentile: float | None,
    event_method: str | None,
    event_threshold_sigma: float | None,
    baseline_sec: float | None,
    response_sec: float | None,
    smooth_method: str | None,
    smooth_window_sec: float | None,
    peak_threshold_sigma: float | None,
    peak_min_distance_sec: float | None,
    angle_period: float | None,
    trace_response_sec: float | None,
    trace_max_rois: int | None,
    trace_scale: str | None,
    y_axis_mode: str | None,
    similarity_source: str | None,
    min_corr: float | None,
    knn: int | None,
    cluster_source: str | None,
    cluster_normalization: str | None,
    linkage_method: str | None,
    distance_metric: str | None,
    n_clusters: int | None,
    embedding_source: str | None,
    leiden_resolution: float | None,
    plot_level: str | None,
    plot_format: str | None,
    plot_dpi: int | None,
    refresh_plots: bool,
) -> list[str]:
    """
    Build arguments understood by known step scripts.

    This keeps common pipeline settings on the main launcher, so users do not
    need to remember which individual script needs which flag.
    """
    managed_args: list[str] = []

    if data_root is not None and step.accepts_data_root:
        managed_args.extend(["--data-root", str(data_root)])

    if output_root is not None and step.accepts_output_root:
        managed_args.extend(["--output-root", str(output_root)])

    if step.accepts_layout:
        managed_args.extend(["--layout", layout])

    if step.accepts_action:
        managed_args.extend(["--action", action])

    if step_dry_run and step.accepts_step_dry_run:
        managed_args.append("--dry-run")

    if trial_id is not None and step.accepts_trial_id:
        managed_args.extend(["--trial-id", trial_id])

    if fiji_memory and step.accepts_fiji_memory:
        managed_args.extend(["--fiji-memory", fiji_memory])

    if projection_mode and step.accepts_projection_mode:
        managed_args.extend(["--projection-mode", projection_mode])

    if stim_export_mode and step.accepts_stim_export_mode:
        managed_args.extend(["--stim-export-mode", stim_export_mode])

    if stim_log_root is not None and step.accepts_stim_log_root:
        managed_args.extend(["--stim-log-root", str(stim_log_root)])

    if metadata_mode and step.accepts_metadata_mode:
        managed_args.extend(["--metadata-mode", metadata_mode])

    if analog_source and step.accepts_analog_source:
        managed_args.extend(["--analog-source", analog_source])

    if raw_z_strategy and step.accepts_raw_z_strategy:
        managed_args.extend(["--raw-z-strategy", raw_z_strategy])

    if sigma_px is not None and step.accepts_sigma_px:
        managed_args.extend(["--sigma-px", str(sigma_px)])

    if clip_negative and step.accepts_clip_negative:
        managed_args.append("--clip-negative")

    if dpi is not None and step.accepts_dpi:
        managed_args.extend(["--dpi", str(dpi)])

    if step.accepts_suite2p_options:
        if strict_suite2p_version:
            managed_args.append("--strict-suite2p-version")
        suite2p_options = (
            ("--n-workers", n_workers),
            ("--num-threads", num_threads),
            ("--suite2p-threads", suite2p_threads),
            ("--batch-size", batch_size),
            ("--min-frames", min_frames),
            ("--cell-diameter-um", cell_diameter_um),
            ("--min-diameter-px", min_diameter_px),
            ("--diameter-scale", diameter_scale),
            ("--threshold-scaling", threshold_scaling),
            ("--max-overlap", max_overlap),
            ("--snr-thresh", snr_thresh),
            ("--high-pass", high_pass),
            ("--spatial-hp-detect", spatial_hp_detect),
            ("--max-iterations", max_iterations),
            ("--inner-neuropil-radius", inner_neuropil_radius),
            ("--min-neuropil-pixels", min_neuropil_pixels),
            ("--tau", tau),
            ("--overlay-dpi", overlay_dpi),
        )
        for option_name, option_value in suite2p_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_roi_gui_options:
        roi_gui_options = (
            ("--movie-kind", movie_kind),
            ("--neuropil-coeff", neuropil_coeff),
            ("--f0-percentile", f0_percentile),
        )
        for option_name, option_value in roi_gui_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_dff_options:
        dff_options = (
            ("--neuropil-coeff", neuropil_coeff),
            ("--roi-source", roi_source),
            ("--f0-mode", f0_mode),
            ("--f0-percentile", f0_percentile),
        )
        for option_name, option_value in dff_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_event_options:
        event_options = (
            ("--event-method", event_method),
            ("--event-threshold-sigma", event_threshold_sigma),
        )
        for option_name, option_value in event_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_stim_response_options:
        response_options = (
            ("--baseline-sec", baseline_sec),
            ("--response-sec", response_sec),
            ("--smooth-method", smooth_method),
            ("--smooth-window-sec", smooth_window_sec),
            ("--peak-threshold-sigma", peak_threshold_sigma),
            ("--peak-min-distance-sec", peak_min_distance_sec),
        )
        for option_name, option_value in response_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_angle_options and angle_period is not None:
        managed_args.extend(["--angle-period", str(angle_period)])

    if step.accepts_trace_options:
        trace_options = (
            ("--response-sec", trace_response_sec),
            ("--smooth-method", smooth_method),
            ("--smooth-window-sec", smooth_window_sec),
            ("--peak-threshold-sigma", peak_threshold_sigma),
            ("--peak-min-distance-sec", peak_min_distance_sec),
            ("--trace-scale", trace_scale),
            ("--y-axis-mode", y_axis_mode),
            ("--max-rois", trace_max_rois),
        )
        for option_name, option_value in trace_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_similarity_options:
        similarity_options = (
            ("--similarity-source", similarity_source),
            ("--min-corr", min_corr),
            ("--knn", knn),
        )
        for option_name, option_value in similarity_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_clustering_options:
        clustering_options = (
            ("--cluster-source", cluster_source),
            ("--cluster-normalization", cluster_normalization),
            ("--linkage-method", linkage_method),
            ("--distance-metric", distance_metric),
            ("--n-clusters", n_clusters),
        )
        for option_name, option_value in clustering_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_embedding_options and embedding_source is not None:
        managed_args.extend(["--embedding-source", embedding_source])

    if step.accepts_leiden_options and leiden_resolution is not None:
        managed_args.extend(["--resolution", str(leiden_resolution)])

    if step.accepts_qc_plot_options:
        qc_plot_options = (
            ("--plot-level", plot_level),
            ("--plot-format", plot_format),
            ("--plot-dpi", plot_dpi),
        )
        for option_name, option_value in qc_plot_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])
        if refresh_plots:
            managed_args.append("--refresh-plots")

    return managed_args


def build_command(
    script_path: Path,
    env_name: str | None,
    conda_bin: str,
    python_bin: str,
    passthrough_args: list[str],
) -> tuple[str, ...]:
    """Build the command used to launch one script."""
    env_name = normalize_env_name(env_name)

    if env_name is None:
        return tuple([python_bin, str(script_path), *passthrough_args])

    return tuple(
        [
            conda_bin,
            "run",
            "-n",
            env_name,
            "--no-capture-output",
            "python",
            str(script_path),
            *passthrough_args,
        ]
    )


def check_conda_available(steps: list[PipelineStep], conda_bin: str) -> None:
    """Fail early if any selected step needs conda but conda is unavailable."""
    needs_conda = any(normalize_env_name(step.env) is not None for step in steps)
    if not needs_conda:
        return

    conda_path = shutil.which(conda_bin)
    if conda_path is None:
        raise FileNotFoundError(
            f"Cannot find conda command: {conda_bin!r}. "
            "Either make conda available in this shell, or pass "
            "--conda-bin /path/to/conda."
        )

    LOGGER.info("使用 conda：%s", conda_path)


def resolve_steps(
    code_dir: Path,
    selected_steps: list[PipelineStep],
    conda_bin: str,
    python_bin: str,
    data_root: Path | None,
    output_root: Path | None,
    layout: str,
    action: str,
    step_dry_run: bool,
    fiji_memory: str | None,
    projection_mode: str | None,
    stim_export_mode: str | None,
    stim_log_root: Path | None,
    metadata_mode: str | None,
    analog_source: str | None,
    raw_z_strategy: str | None,
    sigma_px: float | None,
    clip_negative: bool,
    dpi: int | None,
    strict_suite2p_version: bool,
    n_workers: int | None,
    num_threads: int | None,
    suite2p_threads: int | None,
    batch_size: int | None,
    min_frames: int | None,
    cell_diameter_um: float | None,
    min_diameter_px: int | None,
    diameter_scale: float | None,
    threshold_scaling: float | None,
    max_overlap: float | None,
    snr_thresh: float | None,
    high_pass: int | None,
    spatial_hp_detect: int | None,
    max_iterations: int | None,
    inner_neuropil_radius: int | None,
    min_neuropil_pixels: int | None,
    tau: float | None,
    overlay_dpi: int | None,
    neuropil_coeff: float | None,
    trial_id: str | None,
    movie_kind: str | None,
    roi_source: str | None,
    f0_mode: str | None,
    f0_percentile: float | None,
    event_method: str | None,
    event_threshold_sigma: float | None,
    baseline_sec: float | None,
    response_sec: float | None,
    smooth_method: str | None,
    smooth_window_sec: float | None,
    peak_threshold_sigma: float | None,
    peak_min_distance_sec: float | None,
    angle_period: float | None,
    trace_response_sec: float | None,
    trace_max_rois: int | None,
    trace_scale: str | None,
    y_axis_mode: str | None,
    similarity_source: str | None,
    min_corr: float | None,
    knn: int | None,
    cluster_source: str | None,
    cluster_normalization: str | None,
    linkage_method: str | None,
    distance_metric: str | None,
    n_clusters: int | None,
    embedding_source: str | None,
    leiden_resolution: float | None,
    plot_level: str | None,
    plot_format: str | None,
    plot_dpi: int | None,
    refresh_plots: bool,
    passthrough_args: list[str],
) -> list[ResolvedStep]:
    """Check code directory and script files before launching anything."""
    code_dir = code_dir.expanduser().resolve()

    if not code_dir.exists():
        raise FileNotFoundError(f"Code directory does not exist: {code_dir}")
    if not code_dir.is_dir():
        raise NotADirectoryError(f"Code directory is not a folder: {code_dir}")

    resolved: list[ResolvedStep] = []
    for step in selected_steps:
        script_path = Path(step.script)
        if not script_path.is_absolute():
            script_path = code_dir / script_path
        script_path = script_path.resolve()

        if not script_path.exists():
            raise FileNotFoundError(
                f"Missing script for step {step.step_id}: {script_path}"
            )
        if not script_path.is_file():
            raise FileNotFoundError(
                f"Step {step.step_id} script path is not a file: {script_path}"
            )

        managed_args = build_managed_step_args(
            step=step,
            data_root=data_root,
            output_root=output_root,
            layout=layout,
            action=action,
            step_dry_run=step_dry_run,
            fiji_memory=fiji_memory,
            projection_mode=projection_mode,
            stim_export_mode=stim_export_mode,
            stim_log_root=stim_log_root,
            metadata_mode=metadata_mode,
            analog_source=analog_source,
            raw_z_strategy=raw_z_strategy,
            sigma_px=sigma_px,
            clip_negative=clip_negative,
            dpi=dpi,
            strict_suite2p_version=strict_suite2p_version,
            n_workers=n_workers,
            num_threads=num_threads,
            suite2p_threads=suite2p_threads,
            batch_size=batch_size,
            min_frames=min_frames,
            cell_diameter_um=cell_diameter_um,
            min_diameter_px=min_diameter_px,
            diameter_scale=diameter_scale,
            threshold_scaling=threshold_scaling,
            max_overlap=max_overlap,
            snr_thresh=snr_thresh,
            high_pass=high_pass,
            spatial_hp_detect=spatial_hp_detect,
            max_iterations=max_iterations,
            inner_neuropil_radius=inner_neuropil_radius,
            min_neuropil_pixels=min_neuropil_pixels,
            tau=tau,
            overlay_dpi=overlay_dpi,
            neuropil_coeff=neuropil_coeff,
            trial_id=trial_id,
            movie_kind=movie_kind,
            roi_source=roi_source,
            f0_mode=f0_mode,
            f0_percentile=f0_percentile,
            event_method=event_method,
            event_threshold_sigma=event_threshold_sigma,
            baseline_sec=baseline_sec,
            response_sec=response_sec,
            smooth_method=smooth_method,
            smooth_window_sec=smooth_window_sec,
            peak_threshold_sigma=peak_threshold_sigma,
            peak_min_distance_sec=peak_min_distance_sec,
            angle_period=angle_period,
            trace_response_sec=trace_response_sec,
            trace_max_rois=trace_max_rois,
            trace_scale=trace_scale,
            y_axis_mode=y_axis_mode,
            similarity_source=similarity_source,
            min_corr=min_corr,
            knn=knn,
            cluster_source=cluster_source,
            cluster_normalization=cluster_normalization,
            linkage_method=linkage_method,
            distance_metric=distance_metric,
            n_clusters=n_clusters,
            embedding_source=embedding_source,
            leiden_resolution=leiden_resolution,
            plot_level=plot_level,
            plot_format=plot_format,
            plot_dpi=plot_dpi,
            refresh_plots=refresh_plots,
        )
        command = build_command(
            script_path=script_path,
            env_name=step.env,
            conda_bin=conda_bin,
            python_bin=python_bin,
            passthrough_args=[*managed_args, *passthrough_args],
        )
        resolved.append(ResolvedStep(step, script_path, command))

    return resolved


def print_step_table(steps: list[PipelineStep]) -> None:
    LOGGER.info("可用步骤：")
    for step in steps:
        env_label = normalize_env_name(step.env) or "current"
        LOGGER.info(
            "  %s | %-26s | env=%-12s | %s",
            step.step_id,
            step.name,
            env_label,
            step.script,
        )


def print_plan(code_dir: Path, steps: list[ResolvedStep], dry_run: bool) -> None:
    LOGGER.info("代码目录：%s", code_dir.expanduser().resolve())
    LOGGER.info("启动器模式：%s", "仅展示计划" if dry_run else "执行命令")
    LOGGER.info("已选择步骤：")
    for index, step in enumerate(steps, start=1):
        env_label = normalize_env_name(step.config.env) or "current"
        LOGGER.info(
            "  %d. %s [%s] %s",
            index,
            step.config.step_id,
            env_label,
            step.config.name,
        )
        LOGGER.info("     脚本：%s", step.script_path)
        LOGGER.info("     命令：%s", subprocess.list2cmdline(step.command))


def run_step(step: ResolvedStep) -> bool:
    """Run one selected pipeline step."""
    step_label = f"{step.config.step_id} {step.config.name}"
    banner(f"开始步骤 {step_label}")
    start_time = time.time()

    LOGGER.info("脚本：%s", step.script_path)
    LOGGER.info("工作目录：%s", step.script_path.parent)
    LOGGER.info("命令：%s", subprocess.list2cmdline(step.command))

    child_env = os.environ.copy()
    tmp_root = Path(child_env.get("TMPDIR") or tempfile.gettempdir()).expanduser()
    cache_root = Path(child_env.get("XDG_CACHE_HOME") or tmp_root / "calcium_pipeline_cache").expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    child_env.setdefault("TMPDIR", str(tmp_root))
    child_env.setdefault("XDG_CACHE_HOME", str(cache_root))
    child_env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

    try:
        process = subprocess.Popen(
            list(step.command),
            cwd=str(step.script_path.parent),
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            env=child_env,
        )
        return_code = process.wait()
    except KeyboardInterrupt:
        LOGGER.error("运行步骤 %s 时被用户中断。", step_label)
        raise
    except OSError as exc:
        LOGGER.exception("Could not start step %s: %s", step_label, exc)
        return False

    elapsed = time.time() - start_time
    if return_code == 0:
        LOGGER.info("步骤 %s 成功完成，耗时 %.2f 秒。", step_label, elapsed)
        return True

    LOGGER.error(
        "步骤 %s 失败，退出代码 %s，耗时 %.2f 秒。",
        step_label,
        return_code,
        elapsed,
    )
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch selected calcium imaging pipeline steps safely. "
            "By default this uses the folder containing run_pipeline.py."
        )
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=default_code_dir(),
        help="Folder containing the step scripts. Default: this script's folder.",
    )
    parser.add_argument(
        "--steps",
        help=(
            "Comma-separated step IDs or text matches. Examples: 00,02,05 "
            "or stim,suite2p."
        ),
    )
    parser.add_argument(
        "--from-step",
        help="Run from the first selected step matching this ID/text.",
    )
    parser.add_argument(
        "--to-step",
        help="Run through the last selected step matching this ID/text.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show which commands would run. Do not launch step scripts.",
    )
    parser.add_argument(
        "--step-dry-run",
        action="store_true",
        help=(
            "Launch selected step scripts in their own dry-run mode when supported. "
            "Supported by maintained steps that expose a --dry-run option."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Root folder containing imaging data. Passed to steps that need it, "
            "currently step 00."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Root folder for generated pipeline outputs. For steps that support "
            "it, defaults inside the step to DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "direct", "children"),
        default="auto",
        help=(
            "Data layout for steps that organize files. auto: infer structure; "
            "direct: data-root itself has .oir files; children: child folders have .oir files."
        ),
    )
    parser.add_argument(
        "--action",
        choices=("skip", "overwrite"),
        default="skip",
        help="Conflict behavior for steps that support it. Default: skip.",
    )
    parser.add_argument(
        "--fiji-memory",
        help="Optional Fiji Java heap size for step 01, for example 12g or 16g.",
    )
    parser.add_argument(
        "--projection-mode",
        choices=("auto", "safe", "fast"),
        default="auto",
        help=(
            "Step 01 projection strategy. auto balances speed and memory; "
            "safe lowers memory use; fast can be quicker on high-memory machines."
        ),
    )
    parser.add_argument(
        "--stim-export-mode",
        choices=("projected", "raw", "both"),
        default="projected",
        help=(
            "Step 01 channel-2 export mode. projected keeps the current small "
            "Z-projected stim analog TIFF; raw/both can preserve the unprojected "
            "Z/T stim stack for more precise pulse timing, but may create large files."
        ),
    )
    parser.add_argument(
        "--stim-log-root",
        type=Path,
        help=(
            "Folder containing stimulus controller logs. If omitted, step 02 "
            "uses DATA_ROOT/00_stim_logs_raw when present, then falls back to "
            "DATA_ROOT/stim_logs."
        ),
    )
    parser.add_argument(
        "--analog-source",
        choices=("auto", "raw", "projected"),
        default="auto",
        help="Step 02 stim analog source. auto prefers raw outputs from step 01.",
    )
    parser.add_argument(
        "--raw-z-strategy",
        choices=("planes", "max-range", "mean", "max"),
        default="planes",
        help="Step 02 reduction strategy for raw stim analog z-stacks.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=("skip", "update-missing", "refresh"),
        default="skip",
        help=(
            "Step 01 metadata behavior when TIFF outputs already exist. "
            "Default leaves existing metadata untouched so skip mode does not reopen OIR files."
        ),
    )
    parser.add_argument(
        "--sigma-px",
        type=float,
        default=None,
        help="Step 04 Gaussian sigma for spatial high-pass background.",
    )
    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Step 04: clip negative high-pass values to zero.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Step 04 preview figure DPI.",
    )
    parser.add_argument(
        "--strict-suite2p-version",
        action="store_true",
        help="Step 05: fail if suite2p is not the recommended version 0.14.4.",
    )
    parser.add_argument("--n-workers", type=int, default=None, help="Step 05: trial-level parallel workers.")
    parser.add_argument("--num-threads", type=int, default=None, help="Step 05: low-level BLAS/OpenMP threads per worker.")
    parser.add_argument("--suite2p-threads", type=int, default=None, help="Step 05: suite2p nthreads setting.")
    parser.add_argument("--batch-size", type=int, default=None, help="Step 05: suite2p batch size.")
    parser.add_argument("--min-frames", type=int, default=None, help="Step 05: minimum movie frames required.")
    parser.add_argument("--cell-diameter-um", type=float, default=None, help="Step 05: expected cell diameter in microns.")
    parser.add_argument("--min-diameter-px", type=int, default=None, help="Step 05: lower bound for suite2p diameter in pixels.")
    parser.add_argument("--diameter-scale", type=float, default=None, help="Step 05: multiplier applied after converting cell diameter to pixels.")
    parser.add_argument("--threshold-scaling", type=float, default=None, help="Step 05: larger values make ROI detection more conservative.")
    parser.add_argument("--max-overlap", type=float, default=None, help="Step 05: suite2p max_overlap.")
    parser.add_argument("--snr-thresh", type=float, default=None, help="Step 05: suite2p SNR threshold.")
    parser.add_argument("--high-pass", type=int, default=None, help="Step 05: suite2p temporal high_pass setting.")
    parser.add_argument("--spatial-hp-detect", type=int, default=None, help="Step 05: suite2p spatial_hp_detect setting.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Step 05: suite2p max_iterations.")
    parser.add_argument("--inner-neuropil-radius", type=int, default=None, help="Step 05: suite2p inner_neuropil_radius.")
    parser.add_argument("--min-neuropil-pixels", type=int, default=None, help="Step 05: suite2p min_neuropil_pixels.")
    parser.add_argument("--tau", type=float, default=None, help="Step 05: suite2p calcium decay tau.")
    parser.add_argument("--overlay-dpi", type=int, default=None, help="Step 05: ROI overlay PNG/PDF figure DPI.")
    parser.add_argument("--neuropil-coeff", type=float, default=None, help="Steps manual/06: coefficient for Fneu subtraction.")
    parser.add_argument("--trial-id", default=None, help="Manual step: trial folder name to open in the manual ROI curation GUI.")
    parser.add_argument(
        "--movie-kind",
        choices=("raw", "corrected", "spatial-highpass"),
        default=None,
        help="Manual step: movie shown in the manual ROI curation GUI.",
    )
    parser.add_argument(
        "--roi-source",
        choices=("auto", "manual", "suite2p", "all"),
        default=None,
        help="Step 06: ROI source. auto uses manual curation when present, otherwise suite2p iscell.",
    )
    parser.add_argument(
        "--f0-mode",
        choices=("percentile", "rolling-percentile", "rolling-median", "stim-baseline"),
        default=None,
        help="Step 06: F0 estimation method.",
    )
    parser.add_argument("--f0-percentile", type=float, default=None, help="Step 06: percentile used by F0 estimation.")
    parser.add_argument(
        "--event-method",
        choices=("robust-threshold", "find-peaks"),
        default=None,
        help="Step 07: calcium event detection method.",
    )
    parser.add_argument(
        "--event-threshold-sigma",
        type=float,
        default=None,
        help="Step 07: robust threshold in sigma units.",
    )
    parser.add_argument("--baseline-sec", type=float, default=None, help="Step 08: baseline window before stimulus onset.")
    parser.add_argument("--response-sec", type=float, default=None, help="Step 08: response window after stimulus onset.")
    parser.add_argument("--smooth-method", choices=("rolling-median", "rolling-mean", "none"), default=None, help="Steps 08/trace: smoothing method for noisy dF/F response and peak detection.")
    parser.add_argument("--smooth-window-sec", type=float, default=None, help="Steps 08/trace: smoothing window in seconds.")
    parser.add_argument("--peak-threshold-sigma", type=float, default=None, help="Steps 08/trace: global peak threshold in robust sigma units.")
    parser.add_argument("--peak-min-distance-sec", type=float, default=None, help="Steps 08/trace: minimum distance between global peaks.")
    parser.add_argument("--angle-period", type=float, choices=(180.0, 360.0), default=None, help="Step 09: angular period for circular tuning.")
    parser.add_argument("--trace-response-sec", type=float, default=None, help="Trace step: response window after stimulus onset for peak markers.")
    parser.add_argument("--trace-max-rois", type=int, default=None, help="Trace step: maximum individual ROI trace PNGs to write; 0 means all.")
    parser.add_argument("--trace-scale", choices=("dff", "normalized", "both"), default=None, help="Trace step: write raw dF/F plots, normalized plots, or both.")
    parser.add_argument("--y-axis-mode", choices=("full", "robust"), default=None, help="Trace step: full dynamic y-axis per ROI, or robust clipped display.")
    parser.add_argument("--similarity-source", choices=("slices", "traces", "responses", "features"), default=None, help="Step 13: source for edge graph and heatmap.")
    parser.add_argument("--min-corr", type=float, default=None, help="Step 13: minimum similarity for graph edges.")
    parser.add_argument("--knn", type=int, default=None, help="Step 13: maximum neighbors per ROI.")
    parser.add_argument("--cluster-source", choices=("slices", "angle", "features", "responses", "traces"), default=None, help="Step 14: data source for hierarchical clustering.")
    parser.add_argument("--cluster-normalization", choices=("normalized", "raw", "both"), default=None, help="Step 14: cluster normalized data, raw data, or both.")
    parser.add_argument("--linkage-method", choices=("ward", "average", "complete"), default=None, help="Step 14: hierarchical linkage method.")
    parser.add_argument("--distance-metric", choices=("euclidean", "correlation", "cosine"), default=None, help="Step 14: distance metric.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Step 14: number of hierarchical clusters.")
    parser.add_argument("--embedding-source", choices=("slices", "features", "responses", "traces"), default=None, help="Step 16: data source for PCA/UMAP embedding.")
    parser.add_argument("--leiden-resolution", type=float, default=None, help="Step 15: Leiden resolution parameter.")
    parser.add_argument(
        "--plot-level",
        choices=("none", "basic", "full"),
        default=None,
        help="QC figures for steps that support them. Currently wired for step 06.",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf", "both"),
        default=None,
        help="Step 06+ QC figures: output image format. Default is step-specific, usually both.",
    )
    parser.add_argument("--plot-dpi", type=int, default=None, help="QC figure DPI for steps that support plot options.")
    parser.add_argument(
        "--refresh-plots",
        action="store_true",
        help="Rebuild plots from existing numeric outputs when the selected step supports it. Currently wired for step 06.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List known pipeline steps and exit.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep running later selected steps if one step fails.",
    )
    parser.add_argument(
        "--conda-bin",
        default="conda",
        help="Conda executable to use for steps that require a conda env.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable for steps that use the current environment.",
    )
    parser.add_argument(
        "--step-args",
        action="append",
        help=(
            "Extra arguments forwarded to every launched step. "
            "Example: --step-args '--dry-run'. Use only if those scripts support it."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more launcher details.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        selected_steps = select_steps(
            PIPELINE_STEPS,
            raw_selector=args.steps,
            from_step=args.from_step,
            to_step=args.to_step,
        )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    if args.list_steps:
        print_step_table([step for step in PIPELINE_STEPS if step.enabled])
        return 0

    passthrough_args = split_passthrough_args(args.step_args)
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    stim_log_root = (
        args.stim_log_root.expanduser().resolve()
        if args.stim_log_root
        else default_stim_log_root(data_root)
    )
    try:
        check_conda_available(selected_steps, args.conda_bin)
        resolved_steps = resolve_steps(
            code_dir=args.code_dir,
            selected_steps=selected_steps,
            conda_bin=args.conda_bin,
            python_bin=args.python_bin,
            data_root=data_root,
            output_root=output_root,
            layout=args.layout,
            action=args.action,
            step_dry_run=args.step_dry_run,
            fiji_memory=args.fiji_memory,
            projection_mode=args.projection_mode,
            stim_export_mode=args.stim_export_mode,
            stim_log_root=stim_log_root,
            metadata_mode=args.metadata_mode,
            analog_source=args.analog_source,
            raw_z_strategy=args.raw_z_strategy,
            sigma_px=args.sigma_px,
            clip_negative=args.clip_negative,
            dpi=args.dpi,
            strict_suite2p_version=args.strict_suite2p_version,
            n_workers=args.n_workers,
            num_threads=args.num_threads,
            suite2p_threads=args.suite2p_threads,
            batch_size=args.batch_size,
            min_frames=args.min_frames,
            cell_diameter_um=args.cell_diameter_um,
            min_diameter_px=args.min_diameter_px,
            diameter_scale=args.diameter_scale,
            threshold_scaling=args.threshold_scaling,
            max_overlap=args.max_overlap,
            snr_thresh=args.snr_thresh,
            high_pass=args.high_pass,
            spatial_hp_detect=args.spatial_hp_detect,
            max_iterations=args.max_iterations,
            inner_neuropil_radius=args.inner_neuropil_radius,
            min_neuropil_pixels=args.min_neuropil_pixels,
            tau=args.tau,
            overlay_dpi=args.overlay_dpi,
            neuropil_coeff=args.neuropil_coeff,
            trial_id=args.trial_id,
            movie_kind=args.movie_kind,
            roi_source=args.roi_source,
            f0_mode=args.f0_mode,
            f0_percentile=args.f0_percentile,
            event_method=args.event_method,
            event_threshold_sigma=args.event_threshold_sigma,
            baseline_sec=args.baseline_sec,
            response_sec=args.response_sec,
            smooth_method=args.smooth_method,
            smooth_window_sec=args.smooth_window_sec,
            peak_threshold_sigma=args.peak_threshold_sigma,
            peak_min_distance_sec=args.peak_min_distance_sec,
            angle_period=args.angle_period,
            trace_response_sec=args.trace_response_sec,
            trace_max_rois=args.trace_max_rois,
            trace_scale=args.trace_scale,
            y_axis_mode=args.y_axis_mode,
            similarity_source=args.similarity_source,
            min_corr=args.min_corr,
            knn=args.knn,
            cluster_source=args.cluster_source,
            cluster_normalization=args.cluster_normalization,
            linkage_method=args.linkage_method,
            distance_metric=args.distance_metric,
            n_clusters=args.n_clusters,
            embedding_source=args.embedding_source,
            leiden_resolution=args.leiden_resolution,
            plot_level=args.plot_level,
            plot_format=args.plot_format,
            plot_dpi=args.plot_dpi,
            refresh_plots=args.refresh_plots,
            passthrough_args=passthrough_args,
        )
    except Exception as exc:
        LOGGER.error("配置错误：%s", exc)
        return 1

    banner("Calcium imaging pipeline launcher")
    print_plan(args.code_dir, resolved_steps, args.dry_run)

    if args.dry_run:
        LOGGER.info("dry-run 完成，未启动任何步骤脚本。")
        return 0

    total_start = time.time()
    succeeded: list[str] = []
    failed: list[str] = []

    for step in resolved_steps:
        ok = run_step(step)
        label = f"{step.config.step_id} {step.config.name}"
        if ok:
            succeeded.append(label)
            continue

        failed.append(label)
        if not args.continue_on_failure:
            LOGGER.error("由于该步骤失败，停止后续执行：%s", label)
            break

    total_elapsed = time.time() - total_start
    banner("Pipeline summary", "#")
    LOGGER.info("成功步骤：%d/%d", len(succeeded), len(resolved_steps))
    if succeeded:
        LOGGER.info("成功的步骤：%s", ", ".join(succeeded))
    if failed:
        LOGGER.error("失败的步骤：%s", ", ".join(failed))
    LOGGER.info("启动器总耗时：%.2f 分钟", total_elapsed / 60.0)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
