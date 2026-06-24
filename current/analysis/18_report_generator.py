#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate lightweight HTML reports from pipeline outputs."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trial_context_utils import filter_table_by_excluded_trials, load_excluded_trial_ids


LOGGER = logging.getLogger("report_generator")
STEP_NAME = "18_reports"
STEP_OUTPUT_PATTERNS = ("*.html",)
TRIAL_STEPS = ("06_dff", "08_stim_response", "09_angle_tuning", "11_population_features", "12_stimulus_slice_features")


@dataclass(frozen=True)
class TrialTarget:
    rel_path: Path

    @property
    def trial_id(self) -> str:
        return self.rel_path.name

    @property
    def label(self) -> str:
        return self.rel_path.as_posix()

    @property
    def report_name(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.trial_id).strip("_")
        return f"{safe or self.trial_id}_report.html"

    @property
    def report_subdir(self) -> Path:
        return self.rel_path.parent


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_root(data_root: Path) -> Path:
    return data_root


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def clean_step_outputs(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    removed = 0
    for pattern in STEP_OUTPUT_PATTERNS:
        for path in out_dir.rglob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
    return removed


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except Exception:
        return path.as_posix()


def href_from(from_html: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_html.parent).replace(os.sep, "/")


def image_tag(path: Path, from_html: Path, caption: str) -> str:
    if not path.exists():
        return ""
    return (
        '<figure>'
        f'<img src="{html.escape(href_from(from_html, path))}" alt="{html.escape(caption)}">'
        f'<figcaption>{html.escape(caption)}</figcaption>'
        '</figure>'
    )


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table_html(df: pd.DataFrame, max_rows: int = 20, columns: list[str] | None = None) -> str:
    if df.empty:
        return "<p>No table available.</p>"
    shown = df.copy()
    if columns:
        keep = [col for col in columns if col in shown.columns]
        if keep:
            shown = shown[keep]
    shown = shown.head(max_rows).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(format_value)
    more = "" if len(df) <= max_rows else f'<p class="muted">Showing {max_rows} of {len(df)} rows.</p>'
    return shown.to_html(index=False, escape=True, classes="data-table sortable") + more


def metric_card(label: str, value, detail: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(format_value(value))}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div>'
        '</div>'
    )


def metric_grid(metrics: list[tuple[str, object, str]]) -> str:
    return '<div class="metrics">' + "".join(metric_card(*metric) for metric in metrics) + "</div>"


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge {html.escape(kind)}">{html.escape(text)}</span>'


def file_links(paths: list[Path], from_html: Path) -> str:
    items = []
    for path in paths:
        if path.exists():
            items.append(
                f'<li><a href="{html.escape(href_from(from_html, path))}">{html.escape(path.name)}</a></li>'
            )
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>No exports available.</p>"


def control_bar(
    search_placeholder: str,
    target_selector: str,
    extra_buttons: list[tuple[str, str]] | None = None,
) -> str:
    buttons = "".join(
        f'<button class="control-btn" type="button" data-filter="{html.escape(value)}">{html.escape(label)}</button>'
        for label, value in (extra_buttons or [])
    )
    return (
        f'<div class="controls" data-target="{html.escape(target_selector)}">'
        f'<input class="search-input" type="search" placeholder="{html.escape(search_placeholder)}">'
        f'{buttons}'
        '</div>'
    )


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def context_badges_from_row(qc_row: pd.DataFrame) -> str:
    pieces: list[str] = []
    planned_mode = str(scalar(qc_row, "planned_stimulus_mode", "") or "")
    measured_aolp = str(scalar(qc_row, "measured_has_AoLP", "") or "")
    reduced_chloride = str(scalar(qc_row, "reduced_chloride", "") or "")
    chloride_condition = str(scalar(qc_row, "chloride_condition", "") or "")
    if planned_mode and planned_mode != "unknown":
        pieces.append(badge(planned_mode.replace("_", " "), "ok" if planned_mode in {"pulse", "sustain"} else "neutral"))
    if measured_aolp == "yes":
        pieces.append(badge("AoLP", "ok"))
    elif measured_aolp == "no":
        pieces.append(badge("no AoLP", "neutral"))
    if reduced_chloride == "yes":
        pieces.append(badge((chloride_condition or "reduced chloride").replace("_", " "), "warn"))
    elif reduced_chloride == "no" and chloride_condition:
        pieces.append(badge(chloride_condition.replace("_", " "), "neutral"))
    return " ".join(pieces)


def canonical_label_html(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<span class="muted">{html.escape(text)}</span>'


def discover_trial_targets(output_root: Path) -> list[TrialTarget]:
    rel_paths: set[Path] = set()
    root = output_root / "06_dff"
    excluded_trial_ids = load_excluded_trial_ids(output_root)
    if not root.exists():
        return []
    for marker in root.rglob("*_dff_summary.json"):
        summary = load_json(marker)
        if summary.get("roi_source_used") == "manual_curation":
            rel_path = marker.parent.relative_to(root)
            if rel_path.name not in excluded_trial_ids:
                rel_paths.add(rel_path)
    return [TrialTarget(path) for path in sorted(rel_paths, key=lambda p: p.as_posix())]


def step_dir(output_root: Path, step_name: str, target: TrialTarget) -> Path:
    return output_root / step_name / target.rel_path


def trial_row(table: pd.DataFrame, target: TrialTarget) -> pd.DataFrame:
    if table.empty or "trial_id" not in table.columns:
        return pd.DataFrame()
    trial_match = table[table["trial_id"].astype(str) == target.trial_id]
    if "rel_path" in table.columns:
        rel_match = table[table["rel_path"].astype(str) == target.label]
        if not rel_match.empty:
            return rel_match
    return trial_match


def scalar(table: pd.DataFrame, column: str, default=""):
    if table.empty or column not in table.columns:
        return default
    return table[column].iloc[0]


def load_trial_metadata(output_root: Path) -> pd.DataFrame:
    table = read_csv(output_root / "00_trial_metadata" / "trial_metadata.csv")
    if table.empty:
        return table
    if "raw_trial_id" in table.columns and "trial_id" not in table.columns:
        table = table.rename(columns={"raw_trial_id": "trial_id"})
    return table


def trial_metadata_row(metadata: pd.DataFrame, target: TrialTarget) -> pd.DataFrame:
    return trial_row(metadata, target)


def numeric_value(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except Exception:
        return default


def truthy_count(value) -> bool:
    return numeric_value(value, 0.0) > 0


def table_has_numeric_angle(table: pd.DataFrame) -> bool:
    if table.empty:
        return False
    angle_columns = [
        column
        for column in table.columns
        if column.lower() in {"pol_angle", "aolp", "angle", "angle_deg", "polarization_angle"}
        or "pol_angle" in column.lower()
        or "aolp" in column.lower()
        or "angle" in column.lower()
    ]
    for column in angle_columns:
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().any():
            return True
        text_values = table[column].dropna().astype(str)
        for value in text_values:
            if re.search(r"[-+]?\d+(?:\.\d+)?", value):
                return True
    return False


def stim_count_from_map(stim_map: pd.DataFrame) -> int:
    if stim_map.empty:
        return 0
    if "number_of_stim" in stim_map.columns:
        count = pd.to_numeric(stim_map["number_of_stim"], errors="coerce").fillna(0).max()
        return int(max(count, 0))
    if "stim_type" in stim_map.columns:
        stim_type = stim_map["stim_type"].fillna("").astype(str).str.lower().str.strip()
        no_stim = stim_type.isin({"", "none", "no_stim", "nostim", "no stim", "baseline"})
        return int((~no_stim).sum())
    return 0


def global_trial_table(output_root: Path, filename: str, trial_id: str) -> pd.DataFrame:
    table = read_csv(output_root / "02_stim_map" / filename)
    if table.empty or "trialID" not in table.columns:
        return pd.DataFrame()
    return table[table["trialID"].astype(str) == str(trial_id)].copy()


def trial_stimulus_flags(output_root: Path, target: TrialTarget, qc_row: pd.DataFrame) -> dict:
    trial_id = target.trial_id
    dff_dir = step_dir(output_root, "06_dff", target)
    stim_dir = step_dir(output_root, "08_stim_response", target)
    angle_dir = step_dir(output_root, "09_angle_tuning", target)
    stim_summary = load_json(stim_dir / f"{trial_id}_stim_response_summary.json")
    angle_summary = load_json(angle_dir / f"{trial_id}_angle_tuning_summary.json")
    stim_map = pd.concat(
        [
            read_csv(dff_dir / f"{trial_id}_stim_map.csv"),
            global_trial_table(output_root, "stim_map.csv", trial_id),
            global_trial_table(output_root, "stim_protocol_matches.csv", trial_id),
        ],
        ignore_index=True,
        sort=False,
    )
    stim_events = pd.concat(
        [
            read_csv(dff_dir / f"{trial_id}_stim_events.csv"),
            global_trial_table(output_root, "stim_events.csv", trial_id),
        ],
        ignore_index=True,
        sort=False,
    )
    response_table = read_csv(stim_dir / f"{trial_id}_stim_response_table.csv")
    angle_table = read_csv(angle_dir / f"{trial_id}_angle_response_table.csv")

    n_stim_events = int(
        max(
            numeric_value(scalar(qc_row, "n_stim_events", 0), 0),
            numeric_value(stim_summary.get("n_stim_events", 0), 0),
            numeric_value(len(stim_events), 0),
            numeric_value(stim_count_from_map(stim_map), 0),
        )
    )
    n_response_rows = int(numeric_value(stim_summary.get("n_response_rows", len(response_table)), len(response_table)))
    n_angle_rows = int(numeric_value(angle_summary.get("n_angle_rows", len(angle_table)), len(angle_table)))
    has_stim = n_stim_events > 0 or n_response_rows > 0
    has_aolp = (
        n_angle_rows > 0
        or table_has_numeric_angle(stim_map)
        or table_has_numeric_angle(stim_events)
        or table_has_numeric_angle(response_table)
        or table_has_numeric_angle(angle_table)
    )
    return {
        "has_stim": bool(has_stim),
        "has_aolp": bool(has_aolp),
        "n_stim_events": n_stim_events,
        "n_response_rows": n_response_rows,
        "n_angle_rows": n_angle_rows,
    }


def corrected_qc_for_display(output_root: Path, qc: pd.DataFrame, targets: list[TrialTarget]) -> pd.DataFrame:
    if qc.empty or "trial_id" not in qc.columns:
        return qc
    out = qc.copy()
    if "n_stim_events" not in out.columns:
        out["n_stim_events"] = ""
    for target in targets:
        matched = out["trial_id"].astype(str) == target.trial_id
        if not matched.any():
            continue
        qc_row = out[matched].iloc[[0]]
        out.loc[matched, "n_stim_events"] = trial_stimulus_flags(output_root, target, qc_row)["n_stim_events"]
    return out


def row_by_roi(table: pd.DataFrame, roi_id: int) -> dict:
    if table.empty or "roi_id" not in table.columns:
        return {}
    matched = table[pd.to_numeric(table["roi_id"], errors="coerce") == roi_id]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def metric_line(label: str, value) -> str:
    return f'<span><strong>{html.escape(label)}</strong> {html.escape(format_value(value))}</span>'


def safe_load_npy(path: Path):
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return None


def manual_plane0_dir(output_root: Path, target: TrialTarget) -> Path:
    return output_root / "05e_roi_manual_curation" / target.rel_path / "suite2p_compatible" / "plane0"


def manual_roi_set_path(output_root: Path, target: TrialTarget) -> Path:
    return output_root / "05e_roi_manual_curation" / target.rel_path / f"{target.trial_id}_manual_roi_set.json"


def load_manual_roi_origin_map(output_root: Path, target: TrialTarget) -> dict[int, dict]:
    origin_map: dict[int, dict] = {}
    for roi in load_json_list(manual_roi_set_path(output_root, target)):
        final_id = int(numeric_value(roi.get("final_roi_id", roi.get("manual_roi_id")), 0))
        if final_id <= 0:
            continue
        roi_source = str(roi.get("roi_source", "") or "").strip()
        roi_type = str(roi.get("roi_type", "") or "").strip()
        origin = roi_source or roi_type
        if origin == "manual" and roi_type in {"cellpose", "suite2p"}:
            origin = roi_type
        origin_map[final_id] = {
            "origin": origin,
            "roi_source": roi_source,
            "roi_type": roi_type,
            "source_roi_id": roi.get("source_roi_id", ""),
            "cellpose_original_id": roi.get("cellpose_original_id", ""),
            "suite2p_original_id": roi.get("suite2p_original_id", ""),
        }
    return origin_map


def suite2p_plane0_dir(output_root: Path, target: TrialTarget) -> Path:
    return output_root / "05_suite2p_roi_detection" / target.rel_path / "suite2p" / "plane0"


def load_final_roi_stat(output_root: Path, target: TrialTarget) -> tuple[np.ndarray | None, str, Path | None]:
    path = manual_plane0_dir(output_root, target) / "stat.npy"
    stat = safe_load_npy(path)
    if stat is not None and len(stat) > 0:
        return stat, "manual final", path
    return None, "missing", None


def load_ops(output_root: Path, target: TrialTarget) -> dict:
    value = safe_load_npy(suite2p_plane0_dir(output_root, target) / "ops.npy")
    if value is None:
        return {}
    try:
        item = value.item()
        return item if isinstance(item, dict) else {}
    except Exception:
        return {}


def infer_spatial_shape(stat: np.ndarray | None, ops: dict) -> tuple[int, int]:
    ly = int(ops.get("Ly") or 0) if isinstance(ops, dict) else 0
    lx = int(ops.get("Lx") or 0) if isinstance(ops, dict) else 0
    if ly > 0 and lx > 0:
        return lx, ly
    max_x = 1
    max_y = 1
    if stat is not None:
        for roi in stat:
            if not isinstance(roi, dict):
                continue
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            if xpix.size:
                max_x = max(max_x, int(np.nanmax(xpix)) + 2)
            if ypix.size:
                max_y = max(max_y, int(np.nanmax(ypix)) + 2)
    return max_x, max_y


def write_mean_image_asset(ops: dict, asset_path: Path) -> Path | None:
    mean_img = ops.get("meanImg") if isinstance(ops, dict) else None
    if mean_img is None:
        return None
    image = np.asarray(mean_img, dtype=float)
    if image.ndim != 2 or image.size == 0:
        return None
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return None
    low, high = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        return None
    normalized = np.clip((image - low) / (high - low), 0, 1)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(asset_path, normalized, cmap="gray", vmin=0, vmax=1)
    except Exception as exc:
        LOGGER.warning("Could not write mean-image asset %s: %s", asset_path, exc)
        return None
    return asset_path


def roi_boundary_points(roi: dict, max_points: int = 260) -> list[tuple[float, float]]:
    ypix = np.asarray(roi.get("boundary_ypix", []), dtype=float)
    xpix = np.asarray(roi.get("boundary_xpix", []), dtype=float)
    if ypix.size < 3 or xpix.size < 3:
        ypix = np.asarray(roi.get("ypix", []), dtype=float)
        xpix = np.asarray(roi.get("xpix", []), dtype=float)
        if ypix.size >= 3 and xpix.size >= 3:
            cx = float(np.nanmean(xpix))
            cy = float(np.nanmean(ypix))
            order = np.argsort(np.arctan2(ypix - cy, xpix - cx))
            xpix = xpix[order]
            ypix = ypix[order]
    if ypix.size < 3 or xpix.size < 3:
        return []
    if len(xpix) > max_points:
        keep = np.linspace(0, len(xpix) - 1, max_points, dtype=int)
        xpix = xpix[keep]
        ypix = ypix[keep]
    return [(float(x), float(y)) for x, y in zip(xpix, ypix) if np.isfinite(x) and np.isfinite(y)]


def final_roi_shapes(output_root: Path, target: TrialTarget, report_root: Path) -> tuple[list[dict], int, int, Path | None, str]:
    stat, source_label, _source_path = load_final_roi_stat(output_root, target)
    ops = load_ops(output_root, target)
    width, height = infer_spatial_shape(stat, ops)
    asset_path = report_root / target.report_subdir / "_assets" / f"{target.trial_id}_mean_image.png"
    background = write_mean_image_asset(ops, asset_path)
    shapes = []
    if stat is not None:
        for index, roi in enumerate(stat, start=1):
            if not isinstance(roi, dict):
                continue
            points = roi_boundary_points(roi)
            if not points:
                continue
            xpix = np.asarray(roi.get("xpix", []), dtype=float)
            ypix = np.asarray(roi.get("ypix", []), dtype=float)
            shapes.append(
                {
                    "roi_id": index,
                    "points": points,
                    "x_mean": float(np.nanmean(xpix)) if xpix.size else "",
                    "y_mean": float(np.nanmean(ypix)) if ypix.size else "",
                    "npix": int(len(xpix)) if xpix.size else "",
                }
            )
    return shapes, width, height, background, source_label


def shape_points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def roi_explorer_html(
    roi_table: pd.DataFrame,
    response_table: pd.DataFrame,
    angle_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    trace_summary: pd.DataFrame,
    origin_map: dict[int, dict],
    trace_dir: Path,
    roi_shapes: list[dict],
    view_width: int,
    view_height: int,
    spatial_background: Path | None,
    roi_shape_source: str,
    out_path: Path,
) -> str:
    source = feature_table if not feature_table.empty else roi_table
    if source.empty or "roi_id" not in source.columns:
        return '<section class="panel" id="roi-explorer"><h2>ROI Explorer</h2><p>No ROI table available.</p></section>'

    background_image = ""
    if spatial_background is not None and spatial_background.exists():
        background_image = (
            f'<image class="mean-image" href="{html.escape(href_from(out_path, spatial_background))}" '
            f'x="0" y="0" width="{view_width}" height="{view_height}" preserveAspectRatio="none"></image>'
        )

    rows = []
    cards = []
    shape_by_roi = {int(shape["roi_id"]): shape for shape in roi_shapes}
    shape_elements = []
    for shape in roi_shapes:
        roi_id = int(shape["roi_id"])
        hue = (roi_id * 47) % 360
        shape_elements.append(
            f'<a class="roi-link" href="#roi-card-{roi_id}" data-roi="{roi_id}" aria-label="Show ROI {roi_id}">'
            f'<polygon class="roi-shape" data-roi="{roi_id}" role="button" tabindex="0" '
            f'style="--roi-hue:{hue}" '
            f'points="{html.escape(shape_points_attr(shape["points"]))}">'
            f'<title>ROI {roi_id}</title></polygon></a>'
        )

    for _, roi in source.iterrows():
        roi_id = int(numeric_value(roi.get("roi_id"), 0))
        if roi_id <= 0:
            continue
        response = row_by_roi(response_table, roi_id)
        angle = row_by_roi(angle_table, roi_id)
        trace = row_by_roi(trace_summary, roi_id)
        origin_info = origin_map.get(roi_id, {})
        roi_source = roi.get("roi_source", response.get("roi_source", angle.get("roi_source", "")))
        origin = origin_info.get("origin") or roi.get("roi_origin", "") or roi_source
        origin_type = origin_info.get("roi_type", "")
        source_roi_id = origin_info.get("source_roi_id", "")
        origin_filter = str(origin).lower()
        if "cellpose" in origin_filter:
            origin_filter = "cellpose"
        elif "suite2p" in origin_filter:
            origin_filter = "suite2p"
        else:
            origin_filter = "manual"
        response_type = response.get("response_type", "")
        max_z = response.get("max_zscore", roi.get("max_zscore", ""))
        osi = angle.get("OSI", roi.get("OSI", ""))
        pref_angle = angle.get("preferred_angle", "")
        angle_selective = str(angle.get("angle_selective", "")).lower() in {"true", "1", "yes"}
        responsive = numeric_value(max_z, 0.0) >= 3.0
        filters = " ".join(
            item
            for item, enabled in (
                ("responsive", responsive),
                ("angle", angle_selective),
                ("cellpose", origin_filter == "cellpose"),
                ("suite2p", origin_filter == "suite2p"),
                ("manual", origin_filter == "manual"),
            )
            if enabled
        )
        search_text = " ".join(
            str(value)
            for value in (roi_id, origin, origin_type, source_roi_id, roi_source, response_type, pref_angle, max_z, osi)
        )
        rows.append(
            '<tr class="roi-row" '
            f'data-roi="{roi_id}" data-filter="{html.escape(filters)}" data-search="{html.escape(search_text.lower())}">'
            f'<td><input type="checkbox" class="roi-check" value="{roi_id}"></td>'
            f'<td><a href="#roi-card-{roi_id}" class="roi-row-link">{roi_id}</a></td>'
            f'<td>{html.escape(format_value(origin))}</td>'
            f'<td>{html.escape(format_value(response_type))}</td>'
            f'<td>{html.escape(format_value(max_z))}</td>'
            f'<td>{html.escape(format_value(pref_angle))}</td>'
            f'<td>{html.escape(format_value(osi))}</td>'
            f'<td>{badge("angle", "ok") if angle_selective else ""} {badge("resp", "warn") if responsive else ""}</td>'
            '</tr>'
        )

        dff_img = trace_dir / "roi_traces" / f"ROI_{roi_id:04d}_trace.png"
        norm_img = trace_dir / "roi_traces_normalized" / f"ROI_{roi_id:04d}_trace_norm.png"
        image_parts = []
        for image_path, caption in ((dff_img, "dF/F trace"), (norm_img, "Normalized trace")):
            if image_path.exists():
                image_parts.append(
                    '<figure>'
                    f'<img src="{html.escape(href_from(out_path, image_path))}" alt="ROI {roi_id} {html.escape(caption)}">'
                    f'<figcaption>ROI {roi_id} {html.escape(caption)}</figcaption>'
                    '</figure>'
                )
        metrics = [
            metric_line("origin", origin),
            metric_line("origin type", origin_type),
            metric_line("source ROI", source_roi_id),
            metric_line("dF/F source", roi_source),
            metric_line("response", response_type),
            metric_line("max z", max_z),
            metric_line("mean resp", response.get("mean_response", "")),
            metric_line("event rate", response.get("mean_event_rate_response", roi.get("event_rate_hz", ""))),
            metric_line("pref angle", pref_angle),
            metric_line("OSI", osi),
            metric_line("reliability", angle.get("preferred_reliability", angle.get("reliability", ""))),
            metric_line("mean dF/F", roi.get("mean_dff", "")),
            metric_line("noise", roi.get("baseline_noise", "")),
            metric_line("trace peaks", trace.get("n_detected_stim_peaks", "")),
            metric_line("shape source", roi_shape_source if roi_id in shape_by_roi else ""),
        ]
        origin_metrics = [
            metric_line("origin", origin),
            metric_line("origin type", origin_type),
            metric_line("source ROI", source_roi_id),
            metric_line("cellpose id", origin_info.get("cellpose_original_id", "")),
            metric_line("suite2p id", origin_info.get("suite2p_original_id", "")),
            metric_line("dF/F source", roi_source),
            metric_line("shape source", roi_shape_source if roi_id in shape_by_roi else ""),
        ]
        cards.append(
            f'<article class="roi-card" id="roi-card-{roi_id}" data-roi="{roi_id}">'
            f'<h3>ROI {roi_id}</h3>'
            '<div class="roi-card-tabs" role="tablist" aria-label="ROI detail views">'
            f'<button class="roi-tab active" type="button" data-roi-tab="trace" aria-selected="true">Trace</button>'
            f'<button class="roi-tab" type="button" data-roi-tab="metrics" aria-selected="false">Metrics</button>'
            f'<button class="roi-tab" type="button" data-roi-tab="origin" aria-selected="false">Origin</button>'
            '</div>'
            f'<div class="roi-tab-panel active" data-roi-panel="trace"><div class="grid">{"".join(image_parts) if image_parts else "<p>No trace plot available.</p>"}</div></div>'
            f'<div class="roi-tab-panel" data-roi-panel="metrics"><div class="roi-metrics">{"".join(metrics)}</div></div>'
            f'<div class="roi-tab-panel" data-roi-panel="origin"><div class="roi-metrics">{"".join(origin_metrics)}</div></div>'
            '</article>'
        )

    return (
        '<section class="panel" id="roi-explorer"><h2>ROI Explorer</h2>'
        '<div class="controls roi-controls">'
        '<input class="search-input roi-search" type="search" placeholder="Search ROI, origin, response, angle">'
        '<button class="control-btn" type="button" data-roi-action="select-visible">Select visible</button>'
        '<button class="control-btn" type="button" data-roi-action="clear">Clear</button>'
        '<button class="control-btn" type="button" data-roi-filter="responsive">Responsive</button>'
        '<button class="control-btn" type="button" data-roi-filter="angle">Angle selective</button>'
        '<button class="control-btn" type="button" data-roi-filter="cellpose">Cellpose</button>'
        '<button class="control-btn" type="button" data-roi-filter="suite2p">Suite2p</button>'
        '<button class="control-btn" type="button" data-roi-filter="manual">Manual/drawn</button>'
        '<span class="muted"><span data-selected-count>0</span> selected</span>'
        '</div>'
        f'<p class="note">ROI shapes shown here come from {html.escape(roi_shape_source)} ROI stat data. Click shapes on the image to select ROI; the list below is only a secondary selector.</p>'
        '<div class="roi-layout">'
        '<div>'
        '<div class="spatial-view">'
        f'<svg class="roi-svg" viewBox="0 0 {view_width} {view_height}" role="img" aria-label="Final ROI spatial map">'
        f'{background_image}<rect class="roi-backdrop" x="0" y="0" width="{view_width}" height="{view_height}"></rect>'
        f'{"".join(shape_elements) if shape_elements else ""}'
        '</svg>'
        + ("" if shape_elements else "<p>No final ROI shapes available.</p>")
        + '</div>'
        '<div class="table-wrap roi-list"><table class="data-table sortable"><thead><tr>'
        '<th></th><th>ROI</th><th>Origin</th><th>Response</th><th>Max z</th><th>Pref angle</th><th>OSI</th><th>Flags</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table></div></div>'
        '<div class="roi-selected"><p class="muted roi-empty">Select one or more ROI to show trace plots and metrics.</p>'
        + "".join(cards)
        + '</div></div></section>'
    )


def css() -> str:
    return """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f4f6f8; }
main { max-width: 1240px; margin: 0 auto; padding: 28px; }
h1 { font-size: 28px; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 0 0 12px; }
h3 { font-size: 15px; margin: 14px 0 8px; }
.topbar { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 18px; }
.panel { background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; margin: 16px 0; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.metric { border: 1px solid #dfe4ea; border-radius: 8px; padding: 12px; background: #fbfcfd; }
.metric-label { color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
.metric-value { font-size: 24px; font-weight: 650; margin-top: 4px; }
.metric-detail { color: #667085; font-size: 12px; min-height: 16px; }
.badge { display: inline-block; border-radius: 999px; padding: 3px 9px; font-size: 12px; border: 1px solid #ccd4df; background: #f8fafc; color: #344054; }
.badge.ok { background: #ecfdf3; color: #067647; border-color: #abefc6; }
.badge.warn { background: #fffaeb; color: #b54708; border-color: #fedf89; }
.badge.bad { background: #fef3f2; color: #b42318; border-color: #fecdca; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0 14px; }
.search-input { min-width: 220px; flex: 1 1 260px; border: 1px solid #cfd6df; border-radius: 8px; padding: 8px 10px; font-size: 13px; background: white; }
.control-btn, .section-link { border: 1px solid #cfd6df; border-radius: 8px; background: #fff; color: #344054; padding: 7px 10px; font-size: 12px; cursor: pointer; }
.control-btn.active { background: #175cd3; border-color: #175cd3; color: white; }
.section-nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; position: sticky; top: 0; z-index: 5; padding: 8px 0; background: #f4f6f8; }
.day-group { border: 1px solid #d9dee7; border-radius: 8px; margin: 10px 0; overflow: hidden; background: #fff; }
.day-header { width: 100%; display: flex; justify-content: space-between; align-items: center; gap: 12px; border: 0; background: #f8fafc; padding: 10px 12px; cursor: pointer; font-weight: 650; color: #1f2933; }
.day-list { padding: 10px 18px 14px 28px; margin: 0; columns: 2; }
.trial-item.hidden, .day-group.hidden, tr.hidden, figure.hidden, section.hidden { display: none; }
.roi-layout { display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.4fr); gap: 16px; align-items: start; }
.roi-list { max-height: 640px; }
.roi-row { cursor: pointer; }
.roi-row.selected { background: #eff8ff; }
.spatial-view { position: relative; min-height: 360px; margin-bottom: 12px; border: 1px solid #d9dee7; border-radius: 8px; background: #101828; overflow: hidden; }
.spatial-view p { color: white; padding: 12px; }
.roi-svg { display: block; width: 100%; height: auto; max-height: 72vh; aspect-ratio: auto; }
.mean-image { opacity: 0.78; filter: contrast(1.18) brightness(0.9); }
.roi-backdrop { fill: transparent; pointer-events: none; }
.roi-link { outline: none; }
.roi-shape { fill: hsla(var(--roi-hue), 92%, 56%, 0.24); stroke: hsl(var(--roi-hue), 96%, 64%); stroke-width: 1.8; vector-effect: non-scaling-stroke; cursor: pointer; pointer-events: all; mix-blend-mode: screen; transition: fill 120ms ease, stroke 120ms ease, stroke-width 120ms ease; }
.roi-shape:hover { fill: hsla(var(--roi-hue), 96%, 62%, 0.5); stroke: hsl(var(--roi-hue), 100%, 78%); stroke-width: 2.8; }
.roi-shape.selected { fill: rgba(255, 214, 10, 0.62); stroke: #ff2d55; stroke-width: 3.4; mix-blend-mode: normal; }
.roi-shape.hidden { display: none; }
.roi-selected { max-height: 760px; overflow: auto; border: 1px solid #e1e6ee; border-radius: 8px; padding: 12px; background: #fbfcfd; }
.roi-selected.has-selection { border-color: #ff2d55; box-shadow: 0 0 0 3px rgba(255, 45, 85, 0.12); }
.roi-card { display: none; border: 1px solid #d9dee7; border-radius: 8px; background: white; padding: 12px; margin-bottom: 12px; }
.roi-card.selected-card, .roi-card:target { display: block; }
.roi-card:target { border-color: #ff2d55; box-shadow: 0 0 0 3px rgba(255, 45, 85, 0.12); }
.roi-selected:has(.roi-card:target) .roi-empty { display: none; }
.roi-card h3 { margin-top: 0; }
.roi-card-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0 12px; border-bottom: 1px solid #e4e7ec; padding-bottom: 8px; }
.roi-tab { border: 1px solid #cfd6df; border-radius: 8px; background: #fff; color: #344054; padding: 6px 10px; font-size: 12px; cursor: pointer; }
.roi-tab.active { background: #175cd3; border-color: #175cd3; color: white; }
.roi-tab-panel { display: none; }
.roi-tab-panel.active { display: block; }
.roi-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 6px 12px; font-size: 12px; margin-bottom: 10px; }
.roi-metrics span { background: #f8fafc; border: 1px solid #e4e7ec; border-radius: 6px; padding: 5px 7px; }
figure { margin: 0; background: #fff; border: 1px solid #e1e6ee; border-radius: 8px; padding: 10px; }
figure img { cursor: zoom-in; }
img { max-width: 100%; height: auto; display: block; }
figcaption { color: #667085; font-size: 13px; margin-top: 8px; }
.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
.data-table th, .data-table td { border: 1px solid #d9dee7; padding: 5px 7px; text-align: left; vertical-align: top; }
.data-table th { background: #eef2f6; position: sticky; top: 0; cursor: pointer; user-select: none; }
.table-wrap { max-height: 520px; overflow: auto; border: 1px solid #e1e6ee; border-radius: 8px; }
.table-wrap .data-table th, .table-wrap .data-table td { border-left: 0; border-right: 0; }
.table-filter { width: 100%; border: 1px solid #cfd6df; border-radius: 8px; padding: 8px 10px; margin: 4px 0 8px; font-size: 13px; }
.muted { color: #667085; }
.nav-list { columns: 2; padding-left: 18px; }
.note { color: #475467; font-size: 13px; line-height: 1.45; }
.lightbox { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.86); display: none; align-items: center; justify-content: center; z-index: 50; padding: 24px; }
.lightbox.open { display: flex; }
.lightbox img { max-width: 96vw; max-height: 88vh; background: white; border-radius: 8px; }
.lightbox button { position: fixed; top: 18px; right: 18px; border: 1px solid white; background: rgba(255,255,255,0.12); color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; }
a { color: #175cd3; text-decoration: none; }
@media (max-width: 900px) { .roi-layout { grid-template-columns: 1fr; } }
@media (max-width: 760px) { main { padding: 16px; } .topbar { display: block; } .nav-list, .day-list { columns: 1; } .section-nav { position: static; } }
"""


def script() -> str:
    return """
<script>
(() => {
  const text = el => (el.textContent || '').toLowerCase();

  document.querySelectorAll('.controls').forEach(control => {
    const input = control.querySelector('.search-input');
    const buttons = [...control.querySelectorAll('[data-filter]')];
    const targetSelector = control.dataset.target || '';
    let activeFilter = '';
    const apply = () => {
      const query = (input?.value || '').trim().toLowerCase();
      document.querySelectorAll(targetSelector).forEach(item => {
        const matchesSearch = !query || text(item).includes(query);
        const itemFilter = item.dataset.filter || '';
        const matchesFilter = !activeFilter || itemFilter.includes(activeFilter);
        item.classList.toggle('hidden', !(matchesSearch && matchesFilter));
      });
      document.querySelectorAll('.day-group').forEach(group => {
        const visible = group.querySelectorAll('.trial-item:not(.hidden)').length;
        group.classList.toggle('hidden', visible === 0);
        const count = group.querySelector('[data-visible-count]');
        if (count) count.textContent = visible;
      });
    };
    input?.addEventListener('input', apply);
    buttons.forEach(button => button.addEventListener('click', () => {
      activeFilter = activeFilter === button.dataset.filter ? '' : button.dataset.filter;
      buttons.forEach(btn => btn.classList.toggle('active', btn === button && activeFilter));
      apply();
    }));
    apply();
  });

  document.querySelectorAll('.day-header').forEach(header => {
    header.addEventListener('click', () => {
      const body = header.parentElement.querySelector('.day-list');
      if (body) body.hidden = !body.hidden;
    });
  });

  document.querySelectorAll('.table-wrap').forEach(wrap => {
    const table = wrap.querySelector('table');
    if (!table) return;
    const filter = document.createElement('input');
    filter.className = 'table-filter';
    filter.type = 'search';
    filter.placeholder = 'Filter table';
    wrap.before(filter);
    filter.addEventListener('input', () => {
      const query = filter.value.trim().toLowerCase();
      table.querySelectorAll('tbody tr').forEach(row => {
        row.classList.toggle('hidden', query && !text(row).includes(query));
      });
    });
    table.querySelectorAll('th').forEach((th, index) => {
      th.addEventListener('click', () => {
        const tbody = table.tBodies[0];
        const rows = [...tbody.rows];
        const dir = th.dataset.sortDir === 'asc' ? -1 : 1;
        table.querySelectorAll('th').forEach(cell => delete cell.dataset.sortDir);
        th.dataset.sortDir = dir === 1 ? 'asc' : 'desc';
        rows.sort((a, b) => {
          const av = a.cells[index]?.textContent?.trim() || '';
          const bv = b.cells[index]?.textContent?.trim() || '';
          const an = Number(av), bn = Number(bv);
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * dir;
          return av.localeCompare(bv, undefined, {numeric: true}) * dir;
        });
        rows.forEach(row => tbody.appendChild(row));
      });
    });
  });

  document.querySelectorAll('#roi-explorer').forEach(explorer => {
    const rows = [...explorer.querySelectorAll('.roi-row')];
    const checks = [...explorer.querySelectorAll('.roi-check')];
    const cards = [...explorer.querySelectorAll('.roi-card')];
    const shapes = [...explorer.querySelectorAll('.roi-shape')];
    const empty = explorer.querySelector('.roi-empty');
    const count = explorer.querySelector('[data-selected-count]');
    const search = explorer.querySelector('.roi-search');
    const svg = explorer.querySelector('.roi-svg');
    let activeFilter = '';

    const updateSelection = () => {
      const selected = new Set(checks.filter(check => check.checked).map(check => check.value));
      rows.forEach(row => row.classList.toggle('selected', selected.has(row.dataset.roi)));
      shapes.forEach(shape => shape.classList.toggle('selected', selected.has(shape.dataset.roi)));
      cards.forEach(card => card.classList.toggle('selected-card', selected.has(card.dataset.roi)));
      if (empty) empty.hidden = selected.size > 0;
      if (count) count.textContent = selected.size;
      const selectedPanel = explorer.querySelector('.roi-selected');
      if (selectedPanel) selectedPanel.classList.toggle('has-selection', selected.size > 0);
    };

    const toggleRoi = (roiId, scrollCard = false) => {
      const check = explorer.querySelector(`.roi-check[value="${roiId}"]`);
      const card = explorer.querySelector(`.roi-card[data-roi="${roiId}"]`);
      if (!check) return;
      check.checked = !check.checked;
      updateSelection();
      if (scrollCard && card && check.checked) {
        if (history.replaceState) history.replaceState(null, '', `#roi-card-${roiId}`);
        card.scrollIntoView({block: 'nearest'});
      }
    };

    const applyRoiFilter = () => {
      const query = (search?.value || '').trim().toLowerCase();
      rows.forEach(row => {
        const haystack = `${row.dataset.search || ''} ${text(row)}`;
        const matchesSearch = !query || haystack.includes(query);
        const matchesFilter = !activeFilter || (row.dataset.filter || '').includes(activeFilter);
        row.classList.toggle('hidden', !(matchesSearch && matchesFilter));
        const shape = explorer.querySelector(`.roi-shape[data-roi="${row.dataset.roi}"]`);
        if (shape) shape.classList.toggle('hidden', row.classList.contains('hidden'));
      });
    };

    rows.forEach(row => {
      row.addEventListener('click', event => {
        if (event.target.closest && event.target.closest('a, button')) {
          return;
        }
        if (event.target.tagName !== 'INPUT') {
          const check = row.querySelector('.roi-check');
          check.checked = !check.checked;
        }
        updateSelection();
      });
    });
    checks.forEach(check => check.addEventListener('change', updateSelection));
    shapes.forEach(shape => {
      shape.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          toggleRoi(shape.dataset.roi, true);
        }
      });
    });
    svg?.addEventListener('click', event => {
      const shape = event.target.closest ? event.target.closest('.roi-shape') : null;
      if (!shape || !explorer.contains(shape)) return;
      event.preventDefault();
      event.stopPropagation();
      toggleRoi(shape.dataset.roi, true);
    });
    search?.addEventListener('input', applyRoiFilter);
    explorer.querySelectorAll('[data-roi-filter]').forEach(button => {
      button.addEventListener('click', () => {
        activeFilter = activeFilter === button.dataset.roiFilter ? '' : button.dataset.roiFilter;
        explorer.querySelectorAll('[data-roi-filter]').forEach(btn => btn.classList.toggle('active', btn === button && activeFilter));
        applyRoiFilter();
      });
    });
    explorer.querySelectorAll('[data-roi-action]').forEach(button => {
      button.addEventListener('click', () => {
        if (button.dataset.roiAction === 'clear') {
          checks.forEach(check => check.checked = false);
        }
        if (button.dataset.roiAction === 'select-visible') {
          rows.forEach(row => {
            const check = row.querySelector('.roi-check');
            if (check && !row.classList.contains('hidden')) check.checked = true;
          });
        }
        updateSelection();
      });
    });
    applyRoiFilter();
    updateSelection();
  });

  document.querySelectorAll('.roi-card').forEach(card => {
    const tabs = [...card.querySelectorAll('.roi-tab')];
    const panels = [...card.querySelectorAll('.roi-tab-panel')];
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const view = tab.dataset.roiTab || '';
        tabs.forEach(item => {
          const active = item === tab;
          item.classList.toggle('active', active);
          item.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        panels.forEach(panel => panel.classList.toggle('active', panel.dataset.roiPanel === view));
      });
    });
  });

  const lightbox = document.createElement('div');
  lightbox.className = 'lightbox';
  lightbox.innerHTML = '<button type="button">Close</button><img alt="">';
  document.body.appendChild(lightbox);
  const lightboxImg = lightbox.querySelector('img');
  document.querySelectorAll('figure img').forEach(img => {
    img.addEventListener('click', () => {
      lightboxImg.src = img.src;
      lightboxImg.alt = img.alt;
      lightbox.classList.add('open');
    });
  });
  lightbox.addEventListener('click', event => {
    if (event.target === lightbox || event.target.tagName === 'BUTTON') lightbox.classList.remove('open');
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') lightbox.classList.remove('open');
  });
})();
</script>
"""


def write_trial_report(output_root: Path, report_root: Path, target: TrialTarget, qc: pd.DataFrame) -> Path:
    sections = []
    trial_id = target.trial_id
    out_path = report_root / target.report_subdir / target.report_name
    dff_dir = step_dir(output_root, "06_dff", target)
    event_dir = step_dir(output_root, "07_events", target)
    stim_dir = step_dir(output_root, "08_stim_response", target)
    angle_dir = step_dir(output_root, "09_angle_tuning", target)
    trace_dir = step_dir(output_root, "10_trace_plots", target)
    feature_dir = step_dir(output_root, "11_population_features", target)
    slice_dir = step_dir(output_root, "12_stimulus_slice_features", target)
    similarity_dir = step_dir(output_root, "13_population_similarity", target)
    cluster_dir = step_dir(output_root, "14_hierarchical_clustering", target)
    leiden_dir = step_dir(output_root, "15_leiden", target)
    embed_dir = step_dir(output_root, "16_dimensionality_reduction", target)
    roi_shapes, view_width, view_height, spatial_background, roi_shape_source = final_roi_shapes(output_root, target, report_root)
    origin_map = load_manual_roi_origin_map(output_root, target)

    summary = {
        "dff": load_json(dff_dir / f"{trial_id}_dff_summary.json"),
        "events": load_json(event_dir / f"{trial_id}_event_summary.json"),
        "stim_response": load_json(stim_dir / f"{trial_id}_stim_response_summary.json"),
        "angle": load_json(angle_dir / f"{trial_id}_angle_tuning_summary.json"),
        "features": load_json(feature_dir / f"{trial_id}_population_feature_summary.json"),
        "stimulus_slices": load_json(slice_dir / f"{trial_id}_stim_slice_feature_summary.json"),
        "similarity": load_json(similarity_dir / f"{trial_id}_population_similarity_summary.json"),
        "cluster": load_json(cluster_dir / f"{trial_id}_hierarchical_clustering_summary.json"),
        "leiden": load_json(leiden_dir / f"{trial_id}_leiden_summary.json"),
        "embedding": load_json(embed_dir / f"{trial_id}_embedding_summary.json"),
    }
    qc_row = trial_row(qc, target)
    metadata_row = trial_metadata_row(load_trial_metadata(output_root), target)
    stim_flags = trial_stimulus_flags(output_root, target, qc_row)
    warnings = str(scalar(qc_row, "qc_warnings", "") or "")
    status_badge = badge("Review QC", "warn") if warnings else badge("No QC warning", "ok")
    context_badges = context_badges_from_row(qc_row)
    canonical_trial_id = str(scalar(metadata_row, "canonical_trial_id_proposed", "") or "")
    canonical_group_id = str(scalar(metadata_row, "canonical_group_id", "") or "")

    sections.append(
        '<section class="panel" id="at-a-glance"><h2>At a Glance</h2>'
        + status_badge
        + (f'<div class="badge-row">{context_badges}</div>' if context_badges else "")
        + (
            f'<p class="note"><strong>Canonical trial ID:</strong> {html.escape(canonical_trial_id)}</p>'
            if canonical_trial_id
            else ""
        )
        + (
            f'<p class="note"><strong>Canonical group:</strong> {html.escape(canonical_group_id)}</p>'
            if canonical_group_id
            else ""
        )
        + metric_grid(
            [
                ("ROI", scalar(qc_row, "n_roi", summary["dff"].get("n_roi_selected", "")), "manual-selected or fallback"),
                ("Frames", scalar(qc_row, "n_frames", summary["dff"].get("n_frames", "")), "trace length"),
                ("Stim events", stim_flags["n_stim_events"], "from stim map"),
                ("Responsive ROI", scalar(qc_row, "n_responsive_roi", summary["stim_response"].get("n_responsive_roi", "")), "z threshold count"),
                ("Angle selective", scalar(qc_row, "n_angle_selective_roi", summary["angle"].get("n_angle_selective_roi", "")), "OSI/reliability filter"),
                ("Clusters", scalar(qc_row, "n_hierarchical_clusters", summary["cluster"].get("n_clusters", "")), "hierarchical"),
            ]
        )
        + (
            f'<p class="note">Planned condition: {html.escape(str(scalar(qc_row, "analysis_branch_key", "") or ""))}</p>'
            if str(scalar(qc_row, "analysis_branch_key", "") or "")
            else ""
        )
        + (
            f'<p class="note">Experiment note: {html.escape(str(scalar(qc_row, "note_text", "") or ""))}</p>'
            if str(scalar(qc_row, "note_text", "") or "")
            else ""
        )
        + (f'<p class="note">QC warnings: {html.escape(warnings)}</p>' if warnings else "")
        + "</section>"
    )

    summary_rows = []
    for name, values in summary.items():
        if values:
            keys = (
                "status",
                "n_roi",
                "n_roi_selected",
                "n_frames",
                "fps",
                "n_events",
                "n_stim_events",
                "n_response_rows",
                "n_angle_rows",
                "n_angle_selective_roi",
                "n_evoked_slices",
                "n_slice_features",
                "n_communities",
                "n_clusters",
                "cluster_source",
                "trace_source",
                "roi_source_used",
            )
            summary_rows.append({"section": name, **{k: v for k, v in values.items() if k in keys}})
    sections.append('<section class="panel" id="step-summary"><h2>Step Summary</h2><div class="table-wrap">' + table_html(pd.DataFrame(summary_rows), 30) + "</div></section>")

    figures = [
        image_tag(dff_dir / f"{trial_id}_dff_heatmap.png", out_path, "dF/F heatmap"),
        image_tag(dff_dir / f"{trial_id}_example_traces.png", out_path, "Example dF/F traces"),
        image_tag(event_dir / f"{trial_id}_event_raster.png", out_path, "Calcium event raster"),
        image_tag(stim_dir / f"{trial_id}_stim_response_heatmap.png", out_path, "Stimulus response heatmap"),
        image_tag(stim_dir / f"{trial_id}_psth_examples.png", out_path, "Stimulus-triggered response examples"),
        image_tag(angle_dir / f"{trial_id}_angle_heatmap.png", out_path, "Angle response heatmap"),
        image_tag(angle_dir / f"{trial_id}_osi_distribution.png", out_path, "OSI distribution"),
        image_tag(angle_dir / f"{trial_id}_polar_plots.png", out_path, "Example angle tuning curves"),
        image_tag(trace_dir / f"{trial_id}_trace_overview.png", out_path, "Trace overview"),
        image_tag(slice_dir / f"{trial_id}_stim_slice_response_heatmap.png", out_path, "Stimulus-slice response score"),
        image_tag(cluster_dir / f"{trial_id}_clustered_heatmap.png", out_path, "Clustered feature heatmap"),
        image_tag(cluster_dir / f"{trial_id}_cluster_mean_angle_tuning.png", out_path, "Cluster mean angle tuning"),
        image_tag(cluster_dir / f"{trial_id}_cluster_mean_traces.png", out_path, "Cluster mean traces"),
        image_tag(similarity_dir / f"{trial_id}_similarity_heatmap.png", out_path, "ROI similarity heatmap"),
        image_tag(leiden_dir / f"{trial_id}_leiden_roi_spatial_map.png", out_path, "Leiden spatial map"),
        image_tag(embed_dir / f"{trial_id}_pca_plot.png", out_path, "PCA embedding"),
        image_tag(embed_dir / f"{trial_id}_umap_plot.png", out_path, "UMAP embedding"),
    ]
    figure_html = "".join(fig for fig in figures if fig)
    if figure_html:
        sections.append('<section class="panel" id="figures"><h2>Evidence Figures</h2><div class="grid">' + figure_html + "</div></section>")

    roi_table = read_csv(dff_dir / f"{trial_id}_roi_table.csv")
    response_table = read_csv(stim_dir / f"{trial_id}_roi_response_summary.csv")
    angle_table = read_csv(angle_dir / f"{trial_id}_preferred_angle_by_roi.csv")
    feature_table = read_csv(feature_dir / f"{trial_id}_roi_feature_matrix.csv")
    slice_summary_table = read_csv(slice_dir / f"{trial_id}_stim_slice_roi_summary.csv")
    trace_summary = read_csv(trace_dir / f"{trial_id}_trace_plot_summary.csv")
    sections.append(
        roi_explorer_html(
            roi_table,
            response_table,
            angle_table,
            feature_table,
            trace_summary,
            origin_map,
            trace_dir,
            roi_shapes,
            view_width,
            view_height,
            spatial_background,
            roi_shape_source,
            out_path,
        )
    )
    response_cols = ["roi_id", "roi_source", "response_type", "mean_response", "max_response", "max_zscore", "mean_event_rate_response", "mean_latency_sec"]
    angle_cols = ["roi_id", "roi_source", "preferred_angle", "preferred_response", "preferred_reliability", "orthogonal_response", "OSI", "vector_strength", "angle_selective"]
    feature_cols = ["roi_id", "roi_source", "x_mean", "y_mean", "npix", "mean_dff", "std_dff", "max_dff", "baseline_noise", "event_rate_hz", "max_zscore", "OSI"]
    slice_cols = ["roi_id", "roi_source", "n_stimulus_evoked_slices", "fraction_stimulus_evoked", "max_slice_peak_z", "max_slice_score", "mean_onset_latency_sec", "mean_recovery_latency_sec", "slice_responsive_roi"]
    sections.append('<section class="panel" id="responses"><h2>ROI Response Preview</h2><div class="table-wrap">' + table_html(response_table, 25, response_cols) + "</div></section>")
    sections.append('<section class="panel" id="stimulus-slices"><h2>Stimulus-Slice Response Preview</h2><div class="table-wrap">' + table_html(slice_summary_table, 25, slice_cols) + "</div></section>")
    sections.append('<section class="panel" id="angle"><h2>Angle Tuning Preview</h2><div class="table-wrap">' + table_html(angle_table, 25, angle_cols) + "</div></section>")
    sections.append('<section class="panel" id="features"><h2>Feature Matrix Preview</h2><div class="table-wrap">' + table_html(feature_table, 25, feature_cols) + "</div></section>")

    exports = [
        dff_dir / f"{trial_id}_roi_table.csv",
        dff_dir / f"{trial_id}_dff_summary.csv",
        event_dir / f"{trial_id}_event_table.csv",
        stim_dir / f"{trial_id}_stim_response_table.csv",
        angle_dir / f"{trial_id}_angle_response_table.csv",
        feature_dir / f"{trial_id}_roi_feature_matrix.csv",
        slice_dir / f"{trial_id}_stim_slice_response_table.csv",
        slice_dir / f"{trial_id}_stim_slice_roi_summary.csv",
        slice_dir / f"{trial_id}_stim_slice_feature_columns.csv",
        cluster_dir / f"{trial_id}_hierarchical_cluster_labels.csv",
        embed_dir / f"{trial_id}_pca_embedding.csv",
    ]
    sections.append('<section class="panel" id="exports"><h2>Trial Exports</h2>' + file_links(exports, out_path) + "</section>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    back_href = href_from(out_path, report_root / "global_report.html")
    section_nav = (
        '<nav class="section-nav">'
        '<a class="section-link" href="#at-a-glance">At a Glance</a>'
        '<a class="section-link" href="#roi-explorer">ROI Explorer</a>'
        '<a class="section-link" href="#figures">Figures</a>'
        '<a class="section-link" href="#responses">Responses</a>'
        '<a class="section-link" href="#stimulus-slices">Slices</a>'
        '<a class="section-link" href="#angle">Angle</a>'
        '<a class="section-link" href="#features">Features</a>'
        '<a class="section-link" href="#exports">Exports</a>'
        '</nav>'
    )
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(target.label)} report</title><style>{css()}</style></head>
<body><main>
<div class="topbar"><div><h1>{html.escape(trial_id)}</h1><p class="muted">{html.escape(target.label)}</p>{f'<p class="muted">Canonical: {html.escape(canonical_trial_id)}</p>' if canonical_trial_id else ''}</div><p><a href="{html.escape(back_href)}">Back to global report</a></p></div>
{section_nav}
{''.join(sections)}
{script()}
</main></body></html>
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_global_report(output_root: Path, report_root: Path, trial_reports: list[tuple[TrialTarget, Path]]) -> Path:
    out_path = report_root / "global_report.html"
    qc = read_csv(output_root / "17_cross_trial_summary" / "cross_trial_qc_summary.csv")
    summary = load_json(output_root / "17_cross_trial_summary" / "cross_trial_summary.json")
    excluded_trial_ids = load_excluded_trial_ids(output_root)
    trial_metadata = load_trial_metadata(output_root)
    targets = [target for target, _ in trial_reports]
    qc_display = corrected_qc_for_display(output_root, qc, targets)
    if not trial_metadata.empty and not qc_display.empty and "trial_id" in qc_display.columns:
        qc_display = qc_display.merge(
            trial_metadata[["trial_id", "canonical_trial_id_proposed", "canonical_group_id", "canonical_name_status"]],
            on="trial_id",
            how="left",
        )
    top_responsive = read_csv(output_root / "17_cross_trial_summary" / "top_responsive_rois.csv")
    top_event = read_csv(output_root / "17_cross_trial_summary" / "top_event_rois.csv")
    top_angle = read_csv(output_root / "17_cross_trial_summary" / "top_angle_selective_rois.csv")
    stim_slice_preview = read_csv(output_root / "17_cross_trial_summary" / "all_trials_stim_slice_roi_summary.csv")
    trial_context = read_csv(output_root / "17_cross_trial_summary" / "trial_context_summary.csv")
    if excluded_trial_ids:
        top_responsive = filter_table_by_excluded_trials(top_responsive, excluded_trial_ids)
        top_event = filter_table_by_excluded_trials(top_event, excluded_trial_ids)
        top_angle = filter_table_by_excluded_trials(top_angle, excluded_trial_ids)
        stim_slice_preview = filter_table_by_excluded_trials(stim_slice_preview, excluded_trial_ids)
        trial_context = filter_table_by_excluded_trials(trial_context, excluded_trial_ids)
    if not trial_metadata.empty and not trial_context.empty and "trial_id" in trial_context.columns:
        trial_context = trial_context.merge(
            trial_metadata[["trial_id", "canonical_trial_id_proposed", "canonical_group_id", "canonical_name_status"]],
            on="trial_id",
            how="left",
        )
    warning_count = 0
    if not qc.empty and "qc_warnings" in qc.columns:
        warning_count = int(qc["qc_warnings"].fillna("").astype(str).str.len().gt(0).sum())
    grouped_reports: dict[str, list[tuple[TrialTarget, Path, str]]] = {}
    flag_rows = []
    for target, path in trial_reports:
        qc_row = trial_row(qc, target)
        warnings = str(scalar(qc_row, "qc_warnings", "") or "")
        stim_flags = trial_stimulus_flags(output_root, target, qc_row)
        filters = " ".join(
            item
            for item, enabled in (
                ("warning", bool(warnings)),
                ("withstim", stim_flags["has_stim"]),
                ("nostim", not stim_flags["has_stim"]),
                ("aolp", stim_flags["has_aolp"]),
                ("noaolp", not stim_flags["has_aolp"]),
                ("pulse", str(scalar(qc_row, "planned_stimulus_mode", "") or "") == "pulse"),
                ("sustain", str(scalar(qc_row, "planned_stimulus_mode", "") or "") == "sustain"),
                ("reducedcl", str(scalar(qc_row, "reduced_chloride", "") or "") == "yes"),
                ("stdcl", str(scalar(qc_row, "reduced_chloride", "") or "") == "no"),
            )
            if enabled
        )
        flag_rows.append(
            {
                "date": target.report_subdir.as_posix() if str(target.report_subdir) not in ("", ".") else "undated",
                "trial_id": target.trial_id,
                "canonical_trial_id": scalar(trial_metadata_row(trial_metadata, target), "canonical_trial_id_proposed", ""),
                "canonical_group_id": scalar(trial_metadata_row(trial_metadata, target), "canonical_group_id", ""),
                "has_stimulus": "yes" if stim_flags["has_stim"] else "no",
                "has_AoLP": "yes" if stim_flags["has_aolp"] else "no",
                "n_stim_events": stim_flags["n_stim_events"],
                "n_response_rows": stim_flags["n_response_rows"],
                "n_angle_rows": stim_flags["n_angle_rows"],
                "planned_stimulus_mode": scalar(qc_row, "planned_stimulus_mode", ""),
                "chloride_condition": scalar(qc_row, "chloride_condition", ""),
                "reduced_chloride": scalar(qc_row, "reduced_chloride", ""),
                "qc_warnings": warnings,
            }
        )
        group = target.report_subdir.as_posix() if str(target.report_subdir) not in ("", ".") else "undated"
        grouped_reports.setdefault(group, []).append((target, path, filters))
    links = ""
    flag_by_trial = {row["trial_id"]: row for row in flag_rows}
    for group, items in grouped_reports.items():
        rows = "".join(
            '<li class="trial-item" '
            f'data-filter="{html.escape(filters)}">'
            f'<a href="{html.escape(href_from(out_path, path))}">{html.escape(target.trial_id)}</a> '
            f'{badge("stim", "ok") if flag_by_trial[target.trial_id]["has_stimulus"] == "yes" else badge("no stim", "neutral")} '
            f'{badge("AoLP", "ok") if flag_by_trial[target.trial_id]["has_AoLP"] == "yes" else badge("no AoLP", "neutral")} '
            f'{badge(str(flag_by_trial[target.trial_id]["planned_stimulus_mode"]).replace("_", " "), "neutral") if str(flag_by_trial[target.trial_id]["planned_stimulus_mode"]) not in {"", "unknown"} else ""} '
            f'{badge(str(flag_by_trial[target.trial_id]["chloride_condition"]).replace("_", " "), "warn") if str(flag_by_trial[target.trial_id]["reduced_chloride"]) == "yes" else ""} '
            f'{canonical_label_html(flag_by_trial[target.trial_id]["canonical_trial_id"])} '
            f'<span class="muted">{html.escape(target.label)}</span>'
            '</li>'
            for target, path, filters in items
        )
        links += (
            '<div class="day-group">'
            f'<button class="day-header" type="button"><span>{html.escape(group)}</span>'
            f'<span><span data-visible-count>{len(items)}</span> / {len(items)}</span></button>'
            f'<ul class="day-list">{rows}</ul>'
            '</div>'
        )
    figures = [
        image_tag(output_root / "17_cross_trial_summary" / "cross_trial_roi_count.png", out_path, "ROI count by trial"),
        image_tag(output_root / "17_cross_trial_summary" / "cross_trial_responsive_fraction.png", out_path, "Responsive fraction by trial"),
        image_tag(output_root / "17_cross_trial_summary" / "cross_trial_angle_selective_fraction.png", out_path, "Angle selective fraction by trial"),
    ]
    export_root = output_root / "17_cross_trial_summary"
    exports = [
        export_root / "cross_trial_qc_summary.csv",
        export_root / "all_trials_roi_features.csv",
        export_root / "all_trials_event_table.csv",
        export_root / "all_trials_stim_response_table.csv",
        export_root / "all_trials_stim_slice_roi_summary.csv",
        export_root / "all_trials_stim_slice_response_table.csv",
        export_root / "all_trials_angle_response_table.csv",
        export_root / "all_trials_cluster_labels.csv",
        export_root / "all_trials_pca_embedding.csv",
        export_root / "trial_context_summary.csv",
        output_root / "00_trial_metadata" / "trial_metadata.csv",
        export_root / "top_responsive_rois.csv",
        export_root / "top_event_rois.csv",
        export_root / "top_angle_selective_rois.csv",
    ]
    summary_table = pd.DataFrame([summary]) if summary else pd.DataFrame()
    flag_table = pd.DataFrame(flag_rows)
    trials_with_stim = int((flag_table["has_stimulus"] == "yes").sum()) if not flag_table.empty else 0
    trials_with_aolp = int((flag_table["has_AoLP"] == "yes").sum()) if not flag_table.empty else 0
    trials_without_stim = int((flag_table["has_stimulus"] == "no").sum()) if not flag_table.empty else 0
    trials_without_aolp = int((flag_table["has_AoLP"] == "no").sum()) if not flag_table.empty else 0
    planned_pulse = int((flag_table.get("planned_stimulus_mode", pd.Series(dtype=str)).fillna("").astype(str) == "pulse").sum()) if not flag_table.empty else 0
    planned_sustain = int((flag_table.get("planned_stimulus_mode", pd.Series(dtype=str)).fillna("").astype(str) == "sustain").sum()) if not flag_table.empty else 0
    reduced_chloride = int((flag_table.get("reduced_chloride", pd.Series(dtype=str)).fillna("").astype(str) == "yes").sum()) if not flag_table.empty else 0
    overview_metrics = [
        ("Trials", summary.get("n_trials", len(trial_reports)), "with generated pages"),
        ("Total ROI", summary.get("n_roi_total", ""), "selected ROI across trials"),
        ("Trials with stim", trials_with_stim, f"{trials_without_stim} without stim"),
        ("Trials with AoLP", trials_with_aolp, f"{trials_without_aolp} without AoLP"),
        ("Planned pulse", planned_pulse, f"{planned_sustain} sustain"),
        ("Reduced chloride", reduced_chloride, "from notes / trial names"),
        ("Stim rows", summary.get("n_stim_response_rows", ""), "per-stim response rows"),
        ("Evoked slices", summary.get("n_evoked_slices", ""), "slice-level calls"),
        ("Angle rows", summary.get("n_angle_response_rows", ""), "AoLP response rows"),
        ("QC warnings", warning_count, "trials needing review"),
    ]
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Calcium imaging pipeline report</title><style>{css()}</style></head>
<body><main>
<div class="topbar"><div><h1>Calcium Imaging Pipeline Report</h1><p class="muted">Post-manual analysis overview for ROI, response, AoLP tuning, and population structure.</p></div>{badge("Review QC", "warn") if warning_count else badge("Ready for inspection", "ok")}</div>
<section class="panel"><h2>Overview</h2>{metric_grid(overview_metrics)}</section>
<section class="panel"><h2>Global Summary</h2><div class="table-wrap">{table_html(summary_table, 5)}</div></section>
<section class="panel"><h2>Trial Reports</h2>{control_bar("Search trial/date", ".trial-item", [("Has stim", "withstim"), ("No stim", "nostim"), ("Has AoLP", "aolp"), ("No AoLP", "noaolp"), ("Pulse", "pulse"), ("Sustain", "sustain"), ("Reduced Cl-", "reducedcl"), ("Std Cl", "stdcl"), ("QC warnings", "warning")])}{links}</section>
<section class="panel"><h2>Stimulus / AoLP Summary</h2><div class="table-wrap">{table_html(flag_table, 120, ["date", "trial_id", "canonical_trial_id", "planned_stimulus_mode", "has_stimulus", "has_AoLP", "chloride_condition", "reduced_chloride", "n_stim_events", "n_response_rows", "n_angle_rows", "qc_warnings"])}</div></section>
<section class="panel"><h2>Trial Context</h2><div class="table-wrap">{table_html(trial_context, 120, ["trial_id", "canonical_trial_id_proposed", "canonical_group_id", "note_date", "planned_stimulus_mode", "measured_stimulus_mode", "measured_has_stimulus", "measured_has_AoLP", "chloride_condition", "reduced_chloride", "note_text"])}</div></section>
<section class="panel"><h2>Cross-Trial QC</h2><div class="table-wrap">{table_html(qc_display, 80, ["trial_id", "canonical_trial_id_proposed", "planned_stimulus_mode", "measured_has_AoLP", "chloride_condition", "n_roi", "n_frames", "n_events", "mean_event_rate_hz", "n_stim_events", "n_responsive_roi", "fraction_responsive", "n_slice_responsive_roi", "n_evoked_slices", "fraction_slice_responsive", "n_angle_selective_roi", "fraction_angle_selective", "n_hierarchical_clusters", "n_leiden_communities", "qc_warnings"])}</div></section>
<section class="panel"><h2>Top Responsive ROI Preview</h2><div class="table-wrap">{table_html(top_responsive, 30, ["trial_id", "roi_id", "roi_source", "stim_index", "pol_angle", "zscore_response", "response_mean", "response_peak", "latency_to_peak_sec"])}</div></section>
<section class="panel"><h2>Stimulus-Slice ROI Preview</h2><div class="table-wrap">{table_html(stim_slice_preview, 30, ["trial_id", "roi_id", "roi_source", "n_stimulus_evoked_slices", "fraction_stimulus_evoked", "max_slice_peak_z", "max_slice_score", "mean_onset_latency_sec", "mean_recovery_latency_sec", "slice_responsive_roi"])}</div></section>
<section class="panel"><h2>Top Event ROI Preview</h2><div class="table-wrap">{table_html(top_event, 30, ["trial_id", "roi_id", "roi_source", "event_rate_hz", "n_events", "mean_amplitude", "mean_duration_sec", "mean_auc"])}</div></section>
<section class="panel"><h2>Top Angle ROI Preview</h2><div class="table-wrap">{table_html(top_angle, 30, ["trial_id", "roi_id", "roi_source", "pol_angle", "mean_response", "reliability", "n_trials"])}</div></section>
<section class="panel"><h2>Exported Tables</h2>{file_links(exports, out_path)}</section>
<section class="panel"><h2>Figures</h2><div class="grid">{''.join(fig for fig in figures if fig)}</div></section>
{script()}
</main></body></html>
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-trial and global HTML reports.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root).resolve()
    report_root = step_output_root(output_root)
    if report_root.exists() and args.action == "skip" and (report_root / "global_report.html").exists():
        LOGGER.info("[skip] report already exists: %s", report_root)
        return 0
    if args.dry_run:
        LOGGER.info("[dry-run] Would generate HTML reports under %s", report_root)
        return 0
    if args.action == "overwrite":
        clean_step_outputs(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    qc = read_csv(output_root / "17_cross_trial_summary" / "cross_trial_qc_summary.csv")
    reports = [(target, write_trial_report(output_root, report_root, target, qc)) for target in discover_trial_targets(output_root)]
    global_report = write_global_report(output_root, report_root, reports)
    LOGGER.info("Generated %d trial report(s).", len(reports))
    LOGGER.info("Global report: %s", global_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
