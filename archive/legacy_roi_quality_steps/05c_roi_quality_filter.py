#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter suite2p ROIs using a manual ROI shape and trace prior.

Default layout:
  input : DATA_ROOT/05_suite2p_roi_detection/
  prior : DATA_ROOT/05b_manual_roi_prior/manual_roi_prior.json
  output: DATA_ROOT/05c_roi_quality_filter/

This step does not modify the original suite2p folder. It writes curated ROI
tables and an iscell_curated.npy file that can be used by later steps after the
filtering rules have been inspected.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("roi_quality_filter")

STEP_NAME = "05c_roi_quality_filter"
STEP_OUTPUT_PATTERNS = (
    "*_roi_quality_table.csv",
    "*_accepted_suite2p_indices.csv",
    "*_rejected_suite2p_indices.csv",
    "*_iscell_curated.npy",
    "*_roi_quality_summary.json",
    "*_roi_quality_overlay.png",
    "*_roi_quality_overlay.pdf",
    "*_roi_quality_trace_qc.png",
    "*_roi_quality_trace_qc.pdf",
    "*_roi_quality_ranked_candidates.csv",
    "*_metadata.json",
    "*_stim_events.csv",
    "*_stim_map.csv",
    "*_stim_pulse_events.csv",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    plane0_dir: Path
    stat_path: Path
    ops_path: Path
    iscell_path: Path
    f_path: Path
    fneu_path: Path
    metadata_path: Path | None
    stim_events_path: Path | None
    stim_map_path: Path | None
    stim_pulse_events_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_input_root(output_root: Path) -> Path:
    return output_root / "05_suite2p_roi_detection"


def default_prior_path(output_root: Path) -> Path:
    return output_root / "05b_manual_roi_prior" / "manual_roi_prior.json"


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def trial_output_dir(step_root: Path, trial: TrialInput) -> Path:
    if str(trial.rel_parent) in ("", "."):
        return step_root / trial.trial_id
    return step_root / trial.rel_parent / trial.trial_id


def find_first_existing(folder: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_trials(input_root: Path) -> list[TrialInput]:
    trials: list[TrialInput] = []
    for stat_path in sorted(input_root.rglob("suite2p/plane0/stat.npy")):
        plane0_dir = stat_path.parent
        input_dir = plane0_dir.parent.parent
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                plane0_dir=plane0_dir,
                stat_path=stat_path,
                ops_path=plane0_dir / "ops.npy",
                iscell_path=plane0_dir / "iscell.npy",
                f_path=plane0_dir / "F.npy",
                fneu_path=plane0_dir / "Fneu.npy",
                metadata_path=find_first_existing(input_dir, ("*_metadata.json",)),
                stim_events_path=find_first_existing(input_dir, ("*_stim_events.csv",)),
                stim_map_path=find_first_existing(input_dir, ("*_stim_map.csv",)),
                stim_pulse_events_path=find_first_existing(input_dir, ("*_stim_pulse_events.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_roi_quality_table.csv",
        out_dir / f"{trial_id}_accepted_suite2p_indices.csv",
        out_dir / f"{trial_id}_rejected_suite2p_indices.csv",
        out_dir / f"{trial_id}_iscell_curated.npy",
        out_dir / f"{trial_id}_roi_quality_summary.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def clean_step_outputs(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    removed = 0
    for pattern in STEP_OUTPUT_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
                LOGGER.info("Removed old step-05c folder: %s", path)
            elif path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                LOGGER.info("Removed old step-05c file: %s", path)
    return removed


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    LOGGER.info("Copied %s -> %s", src, dst)
    return True


def copy_sidecar_outputs(trial: TrialInput, out_dir: Path) -> None:
    copy_if_exists(trial.metadata_path, out_dir / f"{trial.trial_id}_metadata.json")
    copy_if_exists(trial.stim_events_path, out_dir / f"{trial.trial_id}_stim_events.csv")
    copy_if_exists(trial.stim_map_path, out_dir / f"{trial.trial_id}_stim_map.csv")
    copy_if_exists(trial.stim_pulse_events_path, out_dir / f"{trial.trial_id}_stim_pulse_events.csv")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ops(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = np.load(path, allow_pickle=True)
        item = value.item()
        return item if isinstance(item, dict) else {}
    except Exception:
        return {}


def load_iscell(path: Path, n_roi: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros(n_roi, dtype=bool), np.full(n_roi, np.nan, dtype=float)
    iscell = np.load(path, allow_pickle=True)
    if iscell.ndim != 2 or iscell.shape[0] != n_roi:
        raise ValueError(f"iscell.npy shape invalid: {iscell.shape}, expected rows={n_roi}")
    flags = iscell[:, 0].astype(bool)
    prob = iscell[:, 1].astype(float) if iscell.shape[1] >= 2 else np.full(n_roi, np.nan, dtype=float)
    return flags, prob


def scalar_prior(prior: dict, metric: str, key: str, fallback: float) -> float:
    try:
        value = float(prior["metrics"][metric][key])
        return value if np.isfinite(value) else fallback
    except Exception:
        return fallback


def build_rules(prior: dict, padding_fraction: float) -> dict:
    pad = max(0.0, float(padding_fraction))
    area_min = scalar_prior(prior, "area_px", "p05", 0.0)
    area_max = scalar_prior(prior, "area_px", "p95", np.inf)
    diam_min = scalar_prior(prior, "equiv_diameter_px", "p05", 0.0)
    diam_max = scalar_prior(prior, "equiv_diameter_px", "p95", np.inf)
    aspect_max = scalar_prior(prior, "aspect_ratio", "p95", np.inf)
    circularity_min = scalar_prior(prior, "circularity", "p05", 0.0)
    return {
        "area_px_min": area_min * (1.0 - pad),
        "area_px_max": area_max * (1.0 + pad),
        "equiv_diameter_px_min": diam_min * (1.0 - pad),
        "equiv_diameter_px_max": diam_max * (1.0 + pad),
        "aspect_ratio_max": aspect_max * (1.0 + pad),
        "circularity_min": max(0.0, circularity_min * (1.0 - pad)),
    }


def scalar_trace_prior(prior: dict, metric: str, key: str, fallback: float) -> float:
    try:
        value = float(prior["trace_metrics"][metric][key])
        return value if np.isfinite(value) else fallback
    except Exception:
        return fallback


def build_trace_rules(prior: dict, padding_fraction: float) -> dict:
    pad = max(0.0, float(padding_fraction))
    dff_min = scalar_trace_prior(prior, "trace_dff_p95", "p05", np.nan)
    snr_min = scalar_trace_prior(prior, "trace_robust_snr", "p05", np.nan)
    cv_min = scalar_trace_prior(prior, "trace_cv", "p05", np.nan)
    cv_max = scalar_trace_prior(prior, "trace_cv", "p95", np.nan)
    lag1_min = scalar_trace_prior(prior, "trace_lag1_autocorr", "p05", np.nan)
    return {
        "trace_dff_p95_min": dff_min * (1.0 - pad) if np.isfinite(dff_min) else np.nan,
        "trace_robust_snr_min": snr_min * (1.0 - pad) if np.isfinite(snr_min) else np.nan,
        "trace_cv_min": cv_min * (1.0 - pad) if np.isfinite(cv_min) else np.nan,
        "trace_cv_max": cv_max * (1.0 + pad) if np.isfinite(cv_max) else np.nan,
        "trace_lag1_autocorr_min": lag1_min * (1.0 - pad) if np.isfinite(lag1_min) else np.nan,
    }


def trace_prior_available(prior: dict) -> bool:
    return int(prior.get("n_trace_roi", 0) or 0) > 0 and bool(prior.get("trace_metrics"))


def roi_geometry(roi: dict) -> dict:
    xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
    ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
    valid = np.isfinite(xpix) & np.isfinite(ypix)
    xpix = xpix[valid]
    ypix = ypix[valid]
    area = int(len(xpix))
    if area == 0:
        return {
            "area_px": 0,
            "equiv_diameter_px": np.nan,
            "width_px": np.nan,
            "height_px": np.nan,
            "aspect_ratio": np.nan,
            "perimeter_px": np.nan,
            "circularity": np.nan,
            "x_mean": np.nan,
            "y_mean": np.nan,
        }

    xmin, xmax = int(np.min(xpix)), int(np.max(xpix))
    ymin, ymax = int(np.min(ypix)), int(np.max(ypix))
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    perimeter = pixel_perimeter(xpix, ypix)
    circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else np.nan
    equiv_diameter = 2.0 * math.sqrt(area / math.pi)
    aspect_ratio = max(width, height) / max(min(width, height), 1)
    return {
        "area_px": area,
        "equiv_diameter_px": float(equiv_diameter),
        "width_px": int(width),
        "height_px": int(height),
        "aspect_ratio": float(aspect_ratio),
        "perimeter_px": float(perimeter),
        "circularity": float(circularity) if np.isfinite(circularity) else np.nan,
        "x_mean": float(np.mean(xpix)),
        "y_mean": float(np.mean(ypix)),
    }


def pixel_perimeter(xpix: np.ndarray, ypix: np.ndarray) -> float:
    pixels = set(zip(xpix.tolist(), ypix.tolist()))
    edges = 0
    for x, y in pixels:
        if (x - 1, y) not in pixels:
            edges += 1
        if (x + 1, y) not in pixels:
            edges += 1
        if (x, y - 1) not in pixels:
            edges += 1
        if (x, y + 1) not in pixels:
            edges += 1
    return float(edges)


def score_roi(geom: dict, rules: dict) -> tuple[float, dict]:
    checks = {
        "area_ok": rules["area_px_min"] <= geom["area_px"] <= rules["area_px_max"],
        "diameter_ok": rules["equiv_diameter_px_min"] <= geom["equiv_diameter_px"] <= rules["equiv_diameter_px_max"],
        "aspect_ok": geom["aspect_ratio"] <= rules["aspect_ratio_max"],
        "circularity_ok": geom["circularity"] >= rules["circularity_min"],
    }
    valid_checks = [bool(v) for v in checks.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    score = float(np.mean(valid_checks)) if valid_checks else 0.0
    return score, checks


def summarize_trace(trace: np.ndarray) -> dict:
    trace = np.asarray(trace, dtype=float)
    finite = np.asarray(trace[np.isfinite(trace)], dtype=float)
    if finite.size == 0:
        return {
            "trace_n_frames": 0,
            "trace_valid_fraction": 0.0,
            "trace_cv": np.nan,
            "trace_robust_snr": np.nan,
            "trace_dff_p95": np.nan,
            "trace_event_fraction_z3": np.nan,
            "trace_lag1_autocorr": np.nan,
        }
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    p10 = float(np.percentile(finite, 10))
    p95 = float(np.percentile(finite, 95))
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    threshold = median + 3.0 * robust_sigma
    if finite.size > 1 and std > 0:
        lag1 = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
    else:
        lag1 = np.nan
    return {
        "trace_n_frames": int(trace.size),
        "trace_valid_fraction": float(finite.size / max(trace.size, 1)),
        "trace_mean": mean,
        "trace_std": std,
        "trace_cv": float(std / max(abs(mean), 1e-6)),
        "trace_p10": p10,
        "trace_median": median,
        "trace_p95": p95,
        "trace_robust_sigma": float(robust_sigma),
        "trace_robust_snr": float((p95 - median) / max(robust_sigma, 1e-6)),
        "trace_dff_p95": float((p95 - p10) / max(abs(p10), 1e-6)),
        "trace_event_fraction_z3": float(np.mean(finite > threshold)) if robust_sigma > 0 else 0.0,
        "trace_lag1_autocorr": lag1 if np.isfinite(lag1) else np.nan,
    }


def load_quality_traces(
    f_path: Path,
    fneu_path: Path,
    neuropil_coeff: float,
    trace_source: str,
) -> np.ndarray | None:
    if not f_path.exists():
        return None
    f = np.load(f_path, allow_pickle=False)
    if trace_source == "neuropil-corrected" and fneu_path.exists():
        fneu = np.load(fneu_path, allow_pickle=False)
        if fneu.shape == f.shape:
            return np.asarray(f - float(neuropil_coeff) * fneu, dtype=float)
    return np.asarray(f, dtype=float)


def score_trace(trace_features: dict, rules: dict) -> tuple[float, dict]:
    checks: dict[str, bool | None] = {}
    dff_min = rules.get("trace_dff_p95_min", np.nan)
    snr_min = rules.get("trace_robust_snr_min", np.nan)
    cv_min = rules.get("trace_cv_min", np.nan)
    cv_max = rules.get("trace_cv_max", np.nan)
    lag1_min = rules.get("trace_lag1_autocorr_min", np.nan)

    value = float(trace_features.get("trace_dff_p95", np.nan))
    checks["trace_dff_ok"] = bool(value >= dff_min) if np.isfinite(value) and np.isfinite(dff_min) else None
    value = float(trace_features.get("trace_robust_snr", np.nan))
    checks["trace_snr_ok"] = bool(value >= snr_min) if np.isfinite(value) and np.isfinite(snr_min) else None
    value = float(trace_features.get("trace_cv", np.nan))
    checks["trace_cv_ok"] = bool(cv_min <= value <= cv_max) if np.isfinite(value) and np.isfinite(cv_min) and np.isfinite(cv_max) else None
    value = float(trace_features.get("trace_lag1_autocorr", np.nan))
    checks["trace_lag1_ok"] = bool(value >= lag1_min) if np.isfinite(value) and np.isfinite(lag1_min) else None

    valid_checks = [bool(v) for v in checks.values() if v is not None]
    score = float(np.mean(valid_checks)) if valid_checks else np.nan
    return score, checks


def quality_label(score: float, accepted: bool) -> str:
    if accepted:
        return "accepted"
    if score >= 0.5:
        return "borderline"
    return "rejected"


def build_quality_table(
    stat: np.ndarray,
    iscell_flag: np.ndarray,
    iscell_prob: np.ndarray,
    traces: np.ndarray | None,
    rules: dict,
    trace_rules: dict,
    min_score: float,
    require_suite2p_iscell: bool,
    trial_id: str,
    trace_prior_mode: str,
    trace_weight: float,
) -> tuple[list[dict], np.ndarray]:
    rows = []
    accepted = np.zeros(len(stat), dtype=bool)
    use_trace = trace_prior_mode in {"trace-report", "shape-and-trace"} and traces is not None and traces.shape[0] == len(stat)
    weight = min(max(float(trace_weight), 0.0), 1.0)
    for idx, roi in enumerate(stat):
        geom = roi_geometry(roi if isinstance(roi, dict) else {})
        shape_score, shape_checks = score_roi(geom, rules)
        trace_features: dict = {}
        trace_score = np.nan
        trace_checks: dict[str, bool | None] = {}
        if use_trace:
            trace_features = summarize_trace(traces[idx])
            trace_score, trace_checks = score_trace(trace_features, trace_rules)
        if trace_prior_mode == "shape-and-trace" and np.isfinite(trace_score):
            score = float((1.0 - weight) * shape_score + weight * trace_score)
        else:
            score = shape_score
        pass_shape = score >= min_score
        pass_suite2p = bool(iscell_flag[idx]) if require_suite2p_iscell else True
        is_accepted = bool(pass_shape and pass_suite2p)
        accepted[idx] = is_accepted
        rows.append(
            {
                "trial_id": trial_id,
                "suite2p_original_id": idx,
                "suite2p_iscell": int(bool(iscell_flag[idx])),
                "suite2p_iscell_prob": float(iscell_prob[idx]) if np.isfinite(iscell_prob[idx]) else np.nan,
                "manual_prior_score": score,
                "shape_prior_score": shape_score,
                "trace_prior_score": float(trace_score) if np.isfinite(trace_score) else np.nan,
                "trace_prior_used": int(bool(use_trace and np.isfinite(trace_score))),
                "accepted": int(is_accepted),
                "quality_label": quality_label(score, is_accepted),
                **geom,
                **trace_features,
                **{key: int(bool(value)) for key, value in shape_checks.items()},
                **{key: (int(bool(value)) if value is not None else np.nan) for key, value in trace_checks.items()},
            }
        )
    return rows, accepted


def curated_iscell_array(accepted: np.ndarray, quality_rows: list[dict]) -> np.ndarray:
    scores = np.asarray([row["manual_prior_score"] for row in quality_rows], dtype=np.float32)
    out = np.zeros((len(accepted), 2), dtype=np.float32)
    out[:, 0] = accepted.astype(np.float32)
    out[:, 1] = scores
    return out


def mean_numeric(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            value = np.nan
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def numeric_values(rows: list[dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def count_pass(rows: list[dict], key: str) -> int:
    total = 0
    for row in rows:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            value = np.nan
        if np.isfinite(value) and value > 0:
            total += 1
    return total


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_indices(path: Path, trial_id: str, indices: np.ndarray) -> None:
    rows = [{"trial_id": trial_id, "suite2p_original_id": int(idx)} for idx in indices]
    write_csv(path, rows)


def save_overlay(
    ops: dict,
    stat: np.ndarray,
    accepted: np.ndarray,
    output_path: Path,
    trial_id: str,
    dpi: int,
    max_rois: int,
) -> None:
    import matplotlib.pyplot as plt

    mean_img = np.asarray(ops.get("meanImg", []), dtype=np.float32)
    if mean_img.ndim != 2:
        ly = int(ops.get("Ly", 0) or 0)
        lx = int(ops.get("Lx", 0) or 0)
        if ly <= 0 or lx <= 0:
            return
        mean_img = np.zeros((ly, lx), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_img, cmap="gray")
    order = np.arange(len(stat))
    if max_rois > 0 and len(order) > max_rois:
        order = order[:max_rois]
    for idx in order:
        roi = stat[idx]
        if not isinstance(roi, dict):
            continue
        xpix = np.asarray(roi.get("xpix", []), dtype=np.int32)
        ypix = np.asarray(roi.get("ypix", []), dtype=np.int32)
        if len(xpix) == 0:
            continue
        color = "lime" if accepted[idx] else "red"
        alpha = 0.7 if accepted[idx] else 0.28
        ax.scatter(xpix, ypix, s=0.45, c=color, alpha=alpha)
    ax.set_title(f"ROI quality filter - {trial_id}")
    ax.axis("off")
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_trace_qc_plot(
    quality_rows: list[dict],
    output_path: Path,
    trial_id: str,
    dpi: int,
) -> None:
    shape_score = numeric_values(quality_rows, "shape_prior_score")
    trace_score = numeric_values(quality_rows, "trace_prior_score")
    accepted = numeric_values(quality_rows, "accepted") > 0
    iscell = numeric_values(quality_rows, "suite2p_iscell") > 0
    valid = np.isfinite(shape_score) & np.isfinite(trace_score)
    if not np.any(valid):
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    colors = np.where(accepted[valid], "tab:green", "tab:red")
    ax.scatter(shape_score[valid], trace_score[valid], s=8, c=colors, alpha=0.45, edgecolors="none")
    ax.axvline(0.75, color="0.3", lw=1, ls="--", label="shape threshold")
    ax.set_xlabel("Shape prior score")
    ax.set_ylabel("Trace prior score")
    ax.set_title("ROI quality: shape vs trace")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 11)
    ax.hist(trace_score[valid & accepted], bins=bins, alpha=0.7, label="accepted", color="tab:green")
    ax.hist(trace_score[valid & ~accepted], bins=bins, alpha=0.45, label="rejected", color="tab:red")
    ax.set_xlabel("Trace prior score")
    ax.set_ylabel("ROI count")
    ax.set_title("Trace score distribution")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    trace_dff = numeric_values(quality_rows, "trace_dff_p95")
    trace_snr = numeric_values(quality_rows, "trace_robust_snr")
    valid2 = np.isfinite(trace_dff) & np.isfinite(trace_snr)
    if np.any(valid2):
        finite_dff = trace_dff[valid2]
        finite_snr = trace_snr[valid2]
        dff_limit = float(np.percentile(finite_dff, 99)) if finite_dff.size else 1.0
        snr_limit = float(np.percentile(finite_snr, 99)) if finite_snr.size else 1.0
        colors2 = np.where(accepted[valid2], "tab:green", "tab:red")
        ax.scatter(finite_dff, finite_snr, s=8, c=colors2, alpha=0.45, edgecolors="none")
        if np.isfinite(dff_limit) and dff_limit > 0:
            ax.set_xlim(left=0, right=dff_limit * 1.1)
        if np.isfinite(snr_limit) and snr_limit > 0:
            ax.set_ylim(bottom=0, top=snr_limit * 1.1)
    ax.set_xlabel("Trace dF/F-like amplitude")
    ax.set_ylabel("Robust trace SNR")
    ax.set_title("Trace feature space")

    ax = axes[1, 1]
    labels = ["all", "suite2p cell", "accepted", "trace score >= 0.75"]
    counts = [
        len(quality_rows),
        int(np.sum(iscell)),
        int(np.sum(accepted)),
        int(np.sum(valid & (trace_score >= 0.75))),
    ]
    ax.bar(labels, counts, color=["0.5", "tab:blue", "tab:green", "tab:purple"])
    ax.set_ylabel("ROI count")
    ax.set_title("ROI counts")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle(trial_id)
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def ranked_candidate_rows(quality_rows: list[dict], limit: int) -> list[dict]:
    def sort_key(row: dict) -> tuple[float, float, float]:
        return (
            float(row.get("accepted", 0) or 0),
            float(row.get("trace_prior_score", -1) or -1),
            float(row.get("shape_prior_score", -1) or -1),
        )

    keys = (
        "trial_id",
        "suite2p_original_id",
        "accepted",
        "suite2p_iscell",
        "suite2p_iscell_prob",
        "manual_prior_score",
        "shape_prior_score",
        "trace_prior_score",
        "area_px",
        "equiv_diameter_px",
        "circularity",
        "aspect_ratio",
        "x_mean",
        "y_mean",
        "trace_cv",
        "trace_robust_snr",
        "trace_dff_p95",
        "trace_event_fraction_z3",
        "trace_lag1_autocorr",
    )
    ordered = sorted(quality_rows, key=sort_key, reverse=True)
    if limit > 0:
        ordered = ordered[:limit]
    return [{key: row.get(key, np.nan) for key in keys} for row in ordered]


def process_trial(
    trial: TrialInput,
    out_dir: Path,
    prior: dict,
    args: argparse.Namespace,
) -> tuple[str, dict]:
    missing = [path.name for path in (trial.stat_path, trial.ops_path, trial.iscell_path) if not path.exists()]
    if missing:
        return "failed", {"trial_id": trial.trial_id, "status": "failed", "message": f"Missing inputs: {', '.join(missing)}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_outputs(trial, out_dir)
    stat = np.load(trial.stat_path, allow_pickle=True)
    ops = load_ops(trial.ops_path)
    iscell_flag, iscell_prob = load_iscell(trial.iscell_path, len(stat))
    rules = build_rules(prior, padding_fraction=args.rule_padding_fraction)
    trace_rules = build_trace_rules(prior, padding_fraction=args.rule_padding_fraction)
    traces = None
    if args.trace_prior_mode in {"trace-report", "shape-and-trace"} and trace_prior_available(prior):
        traces = load_quality_traces(
            trial.f_path,
            trial.fneu_path,
            neuropil_coeff=args.neuropil_coeff,
            trace_source=args.trace_source,
        )
    rows, accepted = build_quality_table(
        stat=stat,
        iscell_flag=iscell_flag,
        iscell_prob=iscell_prob,
        traces=traces,
        rules=rules,
        trace_rules=trace_rules,
        min_score=args.min_quality_score,
        require_suite2p_iscell=args.require_suite2p_iscell,
        trial_id=trial.trial_id,
        trace_prior_mode=args.trace_prior_mode,
        trace_weight=args.trace_weight,
    )

    accepted_indices = np.flatnonzero(accepted)
    rejected_indices = np.flatnonzero(~accepted)
    write_csv(out_dir / f"{trial.trial_id}_roi_quality_table.csv", rows)
    write_csv(out_dir / f"{trial.trial_id}_roi_quality_ranked_candidates.csv", ranked_candidate_rows(rows, args.rank_limit))
    write_indices(out_dir / f"{trial.trial_id}_accepted_suite2p_indices.csv", trial.trial_id, accepted_indices)
    write_indices(out_dir / f"{trial.trial_id}_rejected_suite2p_indices.csv", trial.trial_id, rejected_indices)
    np.save(out_dir / f"{trial.trial_id}_iscell_curated.npy", curated_iscell_array(accepted, rows))
    save_overlay(
        ops=ops,
        stat=stat,
        accepted=accepted,
        output_path=out_dir / f"{trial.trial_id}_roi_quality_overlay.png",
        trial_id=trial.trial_id,
        dpi=args.dpi,
        max_rois=args.overlay_max_rois,
    )
    save_trace_qc_plot(
        quality_rows=rows,
        output_path=out_dir / f"{trial.trial_id}_roi_quality_trace_qc.png",
        trial_id=trial.trial_id,
        dpi=args.dpi,
    )

    rescued = int(np.sum(accepted & ~iscell_flag))
    kept_suite2p = int(np.sum(accepted & iscell_flag))
    rejected_suite2p_cell = int(np.sum((~accepted) & iscell_flag))
    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_suite2p_roi": int(len(stat)),
        "n_suite2p_iscell": int(np.sum(iscell_flag)),
        "n_accepted": int(len(accepted_indices)),
        "n_rejected": int(len(rejected_indices)),
        "n_rescued_noncell_by_shape": rescued,
        "n_kept_suite2p_iscell": kept_suite2p,
        "n_rejected_suite2p_iscell": rejected_suite2p_cell,
        "accepted_fraction": float(len(accepted_indices) / max(len(stat), 1)),
        "min_quality_score": float(args.min_quality_score),
        "mean_shape_prior_score": mean_numeric(rows, "shape_prior_score"),
        "mean_trace_prior_score": mean_numeric(rows, "trace_prior_score"),
        "mean_manual_prior_score": mean_numeric(rows, "manual_prior_score"),
        "n_trace_prior_score_ge_075": int(np.sum(numeric_values(rows, "trace_prior_score") >= 0.75)),
        "n_trace_dff_ok": count_pass(rows, "trace_dff_ok"),
        "n_trace_snr_ok": count_pass(rows, "trace_snr_ok"),
        "n_trace_cv_ok": count_pass(rows, "trace_cv_ok"),
        "n_trace_lag1_ok": count_pass(rows, "trace_lag1_ok"),
        "trace_prior_mode": args.trace_prior_mode,
        "trace_prior_available": bool(trace_prior_available(prior)),
        "trace_prior_used": bool(traces is not None),
        "trace_source": args.trace_source,
        "trace_weight": float(args.trace_weight),
        "neuropil_coeff": float(args.neuropil_coeff),
        "require_suite2p_iscell": bool(args.require_suite2p_iscell),
        "rule_padding_fraction": float(args.rule_padding_fraction),
        "manual_prior_path": str(args.prior_path),
        "rules": rules,
        "trace_rules": trace_rules,
    }
    with (out_dir / f"{trial.trial_id}_roi_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter suite2p ROIs using a manual ROI shape and trace prior.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-05 suite2p root. Default: OUTPUT_ROOT/05_suite2p_roi_detection.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--prior-path", type=Path, help="manual_roi_prior.json. Default: OUTPUT_ROOT/05b_manual_roi_prior/manual_roi_prior.json.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-quality-score", type=float, default=0.75, help="Minimum combined quality score required for accepting an ROI.")
    parser.add_argument("--rule-padding-fraction", type=float, default=0.0, help="Expand manual prior p05-p95 ranges by this fraction.")
    parser.add_argument(
        "--trace-prior-mode",
        choices=("shape-only", "trace-report", "shape-and-trace"),
        default="trace-report",
        help="Compute trace quality features; shape-and-trace also uses them in the accept/reject score.",
    )
    parser.add_argument("--trace-weight", type=float, default=0.3, help="Weight of trace score in the combined manual prior score.")
    parser.add_argument(
        "--trace-source",
        choices=("raw", "neuropil-corrected"),
        default="raw",
        help="Trace source for 05c quality features. raw matches ImageJ manual ROI Results.csv more closely.",
    )
    parser.add_argument("--neuropil-coeff", type=float, default=0.7, help="F - coeff * Fneu used for trace quality features.")
    parser.add_argument("--require-suite2p-iscell", action="store_true", help="Require original suite2p iscell==1 in addition to shape prior.")
    parser.add_argument("--overlay-max-rois", type=int, default=2500)
    parser.add_argument("--rank-limit", type=int, default=500, help="Number of top ROI candidates to write in the ranked candidate table.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root).resolve()
    input_root = args.input_root.expanduser().resolve() if args.input_root else default_input_root(output_root).resolve()
    out_root = step_output_root(output_root)
    args.prior_path = args.prior_path.expanduser().resolve() if args.prior_path else default_prior_path(output_root).resolve()

    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1
    if not args.prior_path.exists():
        LOGGER.error("Manual ROI prior does not exist: %s", args.prior_path)
        return 1

    prior = load_json(args.prior_path)
    trials = discover_trials(input_root)
    summary = RunSummary(found=len(trials))
    rows: list[dict] = []

    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Prior path : %s", args.prior_path)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d suite2p trial(s).", len(trials))

    for trial in trials:
        out_dir = trial_output_dir(out_root, trial)
        if required_outputs_done(out_dir, trial.trial_id) and args.action == "skip":
            summary.skipped += 1
            LOGGER.info("[skip] %s", trial.trial_id)
            continue
        if args.dry_run:
            summary.processed += 1
            LOGGER.info("[dry-run] Would filter ROI quality for %s -> %s", trial.plane0_dir, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)

        try:
            status, row = process_trial(trial, out_dir, prior, args)
            rows.append(row)
            if status == "processed":
                summary.processed += 1
                LOGGER.info("[ok] %s: accepted=%s/%s", trial.trial_id, row.get("n_accepted"), row.get("n_suite2p_roi"))
            elif status == "skipped":
                summary.skipped += 1
                LOGGER.info("[skip] %s: %s", trial.trial_id, row.get("message"))
            else:
                summary.failed += 1
                LOGGER.error("[failed] %s: %s", trial.trial_id, row.get("message"))
        except Exception as exc:
            summary.failed += 1
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        write_csv(out_root / "roi_quality_summary.csv", rows)

    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
