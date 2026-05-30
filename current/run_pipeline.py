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
    accepts_manual_prior_options: bool = False
    accepts_roi_filter_options: bool = False
    accepts_dff_options: bool = False
    accepts_event_options: bool = False
    accepts_stim_response_options: bool = False
    accepts_angle_options: bool = False
    accepts_similarity_options: bool = False
    accepts_clustering_options: bool = False
    accepts_leiden_options: bool = False


# ``env=None`` means: run with the same Python that launched run_pipeline.py.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        step_id="00",
        name="organize OIR files",
        script="00_oir_file_manager.py",
        accepts_data_root=True,
        accepts_layout=True,
        accepts_action=True,
        accepts_step_dry_run=True,
    ),
    PipelineStep(
        step_id="01",
        name="Fiji OIR to TIF",
        script="01_fiji_totif_ini.py",
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
        name="generate stimulus map",
        script="02_generate_stim_map.py",
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
        name="CaImAn motion correction",
        script="03_motion_correct_func_caiman.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
    PipelineStep(
        step_id="04",
        name="spatial high-pass",
        script="04_spatial_highpass.py",
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
        name="suite2p ROI detection",
        script="05_suite2p_roi_detection_schema_aligned_connected.py",
        env="suite2p",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_suite2p_options=True,
    ),
    PipelineStep(
        step_id="05b",
        name="manual ROI prior",
        script="05b_manual_roi_prior.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_manual_prior_options=True,
    ),
    PipelineStep(
        step_id="05c",
        name="ROI quality filter",
        script="05c_roi_quality_filter.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_roi_filter_options=True,
    ),
    PipelineStep(
        step_id="06",
        name="extract dF/F",
        script="06_extract_dff.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_dff_options=True,
    ),
    PipelineStep(
        step_id="07",
        name="detect calcium events",
        script="07_detect_events.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_event_options=True,
    ),
    PipelineStep(
        step_id="08",
        name="stimulus response analysis",
        script="08_stim_response_analysis.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_stim_response_options=True,
    ),
    PipelineStep(
        step_id="09",
        name="angle tuning analysis",
        script="09_angle_tuning_analysis.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_angle_options=True,
    ),
    PipelineStep(
        step_id="10",
        name="population features",
        script="10_population_features.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
    PipelineStep(
        step_id="11",
        name="population similarity",
        script="11_population_similarity.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_similarity_options=True,
    ),
    PipelineStep(
        step_id="12",
        name="hierarchical clustering",
        script="12_hierarchical_clustering.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_clustering_options=True,
    ),
    PipelineStep(
        step_id="13",
        name="Leiden community detection",
        script="13_leiden_community_detection.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
        accepts_leiden_options=True,
    ),
    PipelineStep(
        step_id="14",
        name="dimensionality reduction",
        script="14_dimensionality_reduction.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
    PipelineStep(
        step_id="15",
        name="cross-trial summary",
        script="15_cross_trial_summary.py",
        env="caiman",
        accepts_data_root=True,
        accepts_action=True,
        accepts_step_dry_run=True,
        accepts_output_root=True,
    ),
    PipelineStep(
        step_id="16",
        name="report generation",
        script="16_report_generator.py",
        env="caiman",
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

    return selectors


def step_matches_selector(step: PipelineStep, selectors: set[str]) -> bool:
    text = f"{step.step_id} {step.name} {step.script}".lower()
    return any(selector == step.step_id or selector in text for selector in selectors)


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
    manual_root: Path | None,
    manual_trace_mode: str | None,
    manual_trace_frame_stride: int | None,
    min_quality_score: float | None,
    rule_padding_fraction: float | None,
    require_suite2p_iscell: bool,
    trace_prior_mode: str | None,
    trace_weight: float | None,
    neuropil_coeff: float | None,
    roi_source: str | None,
    f0_mode: str | None,
    f0_percentile: float | None,
    event_method: str | None,
    event_threshold_sigma: float | None,
    baseline_sec: float | None,
    response_sec: float | None,
    angle_period: float | None,
    similarity_source: str | None,
    min_corr: float | None,
    knn: int | None,
    n_clusters: int | None,
    leiden_resolution: float | None,
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

    if step.accepts_manual_prior_options:
        if manual_root is not None:
            managed_args.extend(["--manual-root", str(manual_root)])
        manual_prior_options = (
            ("--trace-mode", manual_trace_mode),
            ("--trace-frame-stride", manual_trace_frame_stride),
            ("--dpi", dpi),
        )
        for option_name, option_value in manual_prior_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_roi_filter_options:
        roi_filter_options = (
            ("--min-quality-score", min_quality_score),
            ("--rule-padding-fraction", rule_padding_fraction),
            ("--trace-prior-mode", trace_prior_mode),
            ("--trace-weight", trace_weight),
            ("--neuropil-coeff", neuropil_coeff),
        )
        for option_name, option_value in roi_filter_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])
        if require_suite2p_iscell:
            managed_args.append("--require-suite2p-iscell")

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
        )
        for option_name, option_value in response_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_angle_options and angle_period is not None:
        managed_args.extend(["--angle-period", str(angle_period)])

    if step.accepts_similarity_options:
        similarity_options = (
            ("--similarity-source", similarity_source),
            ("--min-corr", min_corr),
            ("--knn", knn),
        )
        for option_name, option_value in similarity_options:
            if option_value is not None:
                managed_args.extend([option_name, str(option_value)])

    if step.accepts_clustering_options and n_clusters is not None:
        managed_args.extend(["--n-clusters", str(n_clusters)])

    if step.accepts_leiden_options and leiden_resolution is not None:
        managed_args.extend(["--resolution", str(leiden_resolution)])

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

    LOGGER.info("Using conda: %s", conda_path)


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
    manual_root: Path | None,
    manual_trace_mode: str | None,
    manual_trace_frame_stride: int | None,
    min_quality_score: float | None,
    rule_padding_fraction: float | None,
    require_suite2p_iscell: bool,
    trace_prior_mode: str | None,
    trace_weight: float | None,
    neuropil_coeff: float | None,
    roi_source: str | None,
    f0_mode: str | None,
    f0_percentile: float | None,
    event_method: str | None,
    event_threshold_sigma: float | None,
    baseline_sec: float | None,
    response_sec: float | None,
    angle_period: float | None,
    similarity_source: str | None,
    min_corr: float | None,
    knn: int | None,
    n_clusters: int | None,
    leiden_resolution: float | None,
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
            manual_root=manual_root,
            manual_trace_mode=manual_trace_mode,
            manual_trace_frame_stride=manual_trace_frame_stride,
            min_quality_score=min_quality_score,
            rule_padding_fraction=rule_padding_fraction,
            require_suite2p_iscell=require_suite2p_iscell,
            trace_prior_mode=trace_prior_mode,
            trace_weight=trace_weight,
            neuropil_coeff=neuropil_coeff,
            roi_source=roi_source,
            f0_mode=f0_mode,
            f0_percentile=f0_percentile,
            event_method=event_method,
            event_threshold_sigma=event_threshold_sigma,
            baseline_sec=baseline_sec,
            response_sec=response_sec,
            angle_period=angle_period,
            similarity_source=similarity_source,
            min_corr=min_corr,
            knn=knn,
            n_clusters=n_clusters,
            leiden_resolution=leiden_resolution,
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
    LOGGER.info("Available steps:")
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
    LOGGER.info("Code directory: %s", code_dir.expanduser().resolve())
    LOGGER.info("Launcher mode: %s", "plan only" if dry_run else "run commands")
    LOGGER.info("Selected steps:")
    for index, step in enumerate(steps, start=1):
        env_label = normalize_env_name(step.config.env) or "current"
        LOGGER.info(
            "  %d. %s [%s] %s",
            index,
            step.config.step_id,
            env_label,
            step.config.name,
        )
        LOGGER.info("     script: %s", step.script_path)
        LOGGER.info("     command: %s", subprocess.list2cmdline(step.command))


def run_step(step: ResolvedStep) -> bool:
    """Run one selected pipeline step."""
    step_label = f"{step.config.step_id} {step.config.name}"
    banner(f"Starting step {step_label}")
    start_time = time.time()

    LOGGER.info("Script: %s", step.script_path)
    LOGGER.info("Working directory: %s", step.script_path.parent)
    LOGGER.info("Command: %s", subprocess.list2cmdline(step.command))

    child_env = os.environ.copy()
    child_env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    child_env.setdefault("XDG_CACHE_HOME", "/private/tmp")

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
        LOGGER.error("Interrupted by user while running step %s.", step_label)
        raise
    except OSError as exc:
        LOGGER.exception("Could not start step %s: %s", step_label, exc)
        return False

    elapsed = time.time() - start_time
    if return_code == 0:
        LOGGER.info("Step %s finished successfully in %.2f seconds.", step_label, elapsed)
        return True

    LOGGER.error(
        "Step %s failed with exit code %s after %.2f seconds.",
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
            "Folder containing stimulus controller logs. Passed to step 02 when "
            "provided."
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
        default="update-missing",
        help=(
            "Step 01 metadata behavior when TIFF outputs already exist. "
            "Default updates missing lightweight metadata without regenerating TIFFs."
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
    parser.add_argument("--manual-root", type=Path, default=None, help="Step 05b: folder containing historical RoiSet.zip files and matching trace exports.")
    parser.add_argument(
        "--manual-trace-mode",
        choices=("none", "summary", "full"),
        default=None,
        help="Step 05b: extract manual ROI trace summaries; full also writes per-frame traces.",
    )
    parser.add_argument("--manual-trace-frame-stride", type=int, default=None, help="Step 05b: use every Nth TIFF frame if traces must be extracted from a movie.")
    parser.add_argument("--min-quality-score", type=float, default=None, help="Step 05c: minimum combined manual-prior quality score.")
    parser.add_argument("--rule-padding-fraction", type=float, default=None, help="Step 05c: expand manual-prior p05-p95 ranges by this fraction.")
    parser.add_argument(
        "--trace-prior-mode",
        choices=("shape-only", "trace-report", "shape-and-trace"),
        default=None,
        help="Step 05c: whether trace features are reported only or also used for filtering.",
    )
    parser.add_argument("--trace-weight", type=float, default=None, help="Step 05c: weight of trace score in the combined ROI quality score.")
    parser.add_argument("--require-suite2p-iscell", action="store_true", help="Step 05c: require suite2p iscell==1 in addition to shape prior.")
    parser.add_argument("--neuropil-coeff", type=float, default=None, help="Steps 05c/06: coefficient for suite2p Fneu subtraction.")
    parser.add_argument(
        "--roi-source",
        choices=("auto", "curated", "suite2p", "all"),
        default=None,
        help="Step 06: ROI source. auto uses 05c curated labels when present.",
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
    parser.add_argument("--angle-period", type=float, choices=(180.0, 360.0), default=None, help="Step 09: angular period for circular tuning.")
    parser.add_argument("--similarity-source", choices=("traces", "responses", "features"), default=None, help="Step 11: source for edge graph and heatmap.")
    parser.add_argument("--min-corr", type=float, default=None, help="Step 11: minimum similarity for graph edges.")
    parser.add_argument("--knn", type=int, default=None, help="Step 11: maximum neighbors per ROI.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Step 12: number of hierarchical clusters.")
    parser.add_argument("--leiden-resolution", type=float, default=None, help="Step 13: Leiden resolution parameter.")
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
    stim_log_root = args.stim_log_root.expanduser().resolve() if args.stim_log_root else None
    manual_root = args.manual_root.expanduser().resolve() if args.manual_root else None

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
            manual_root=manual_root,
            manual_trace_mode=args.manual_trace_mode,
            manual_trace_frame_stride=args.manual_trace_frame_stride,
            min_quality_score=args.min_quality_score,
            rule_padding_fraction=args.rule_padding_fraction,
            require_suite2p_iscell=args.require_suite2p_iscell,
            trace_prior_mode=args.trace_prior_mode,
            trace_weight=args.trace_weight,
            neuropil_coeff=args.neuropil_coeff,
            roi_source=args.roi_source,
            f0_mode=args.f0_mode,
            f0_percentile=args.f0_percentile,
            event_method=args.event_method,
            event_threshold_sigma=args.event_threshold_sigma,
            baseline_sec=args.baseline_sec,
            response_sec=args.response_sec,
            angle_period=args.angle_period,
            similarity_source=args.similarity_source,
            min_corr=args.min_corr,
            knn=args.knn,
            n_clusters=args.n_clusters,
            leiden_resolution=args.leiden_resolution,
            passthrough_args=passthrough_args,
        )
    except Exception as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 1

    banner("Calcium imaging pipeline launcher")
    print_plan(args.code_dir, resolved_steps, args.dry_run)

    if args.dry_run:
        LOGGER.info("Dry-run complete. No step scripts were launched.")
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
            LOGGER.error("Stopping because this step failed: %s", label)
            break

    total_elapsed = time.time() - total_start
    banner("Pipeline summary", "#")
    LOGGER.info("Succeeded: %d/%d", len(succeeded), len(resolved_steps))
    if succeeded:
        LOGGER.info("Successful steps: %s", ", ".join(succeeded))
    if failed:
        LOGGER.error("Failed steps: %s", ", ".join(failed))
    LOGGER.info("Total launcher time: %.2f minutes", total_elapsed / 60.0)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
