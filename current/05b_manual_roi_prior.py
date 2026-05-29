#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a manual ROI shape prior from ImageJ/Fiji RoiSet.zip files.

Default use:
  input : a folder containing historical manual ROI folders
  output: OUTPUT_ROOT/05b_manual_roi_prior/

This script is read-only with respect to the manual ROI source folder. It does
not modify RoiSet.zip files or TIFFs. The goal is to summarize what manually
accepted cell ROIs look like, so later suite2p ROIs can be filtered more
conservatively.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("manual_roi_prior")

STEP_NAME = "05b_manual_roi_prior"
STEP_OUTPUT_PATTERNS = (
    "manual_roi_pairs.csv",
    "manual_roi_table.csv",
    "manual_roi_shape_summary.csv",
    "manual_roi_prior.json",
    "manual_roi_area_distribution.png",
    "manual_roi_area_distribution.pdf",
    "manual_roi_diameter_distribution.png",
    "manual_roi_diameter_distribution.pdf",
)

ROI_TYPE_NAMES = {
    0: "polygon",
    1: "rect",
    2: "oval",
    3: "line",
    4: "freehand",
    5: "traced",
    6: "polyline",
    7: "freeline",
    8: "angle",
    9: "point",
    10: "no_roi",
}


@dataclass(frozen=True)
class ManualPair:
    pair_id: str
    rel_parent: Path
    folder: Path
    roi_zip: Path
    reference_tif: Path | None


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


def default_output_root(data_root: Path | None, manual_root: Path) -> Path:
    return data_root if data_root is not None else manual_root


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def clean_step_outputs(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    removed = 0
    for pattern in STEP_OUTPUT_PATTERNS:
        for path in out_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
                LOGGER.info("Removed old folder: %s", path)
            elif path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
                LOGGER.info("Removed old file: %s", path)
    return removed


def is_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def find_reference_tif(folder: Path) -> Path | None:
    preferred = sorted(
        path
        for path in folder.glob("*_P0_Reg_full.tif*")
        if path.is_file() and not is_appledouble(path)
    )
    if preferred:
        return preferred[0]

    candidates = sorted(
        path
        for path in folder.glob("*.tif*")
        if path.is_file() and not is_appledouble(path)
    )
    return candidates[0] if candidates else None


def discover_pairs(manual_root: Path) -> list[ManualPair]:
    pairs = []
    for roi_zip in sorted(manual_root.rglob("RoiSet.zip")):
        if is_appledouble(roi_zip):
            continue
        folder = roi_zip.parent
        rel_parent = folder.parent.relative_to(manual_root) if folder.parent != manual_root else Path(".")
        pairs.append(
            ManualPair(
                pair_id=folder.name,
                rel_parent=rel_parent,
                folder=folder,
                roi_zip=roi_zip,
                reference_tif=find_reference_tif(folder),
            )
        )
    return pairs


def read_tiff_shape(path: Path | None) -> tuple[int | None, int | None, tuple[int, ...] | None]:
    if path is None:
        return None, None, None
    try:
        import tifffile as tf

        with tf.TiffFile(path) as tif:
            shape = tuple(int(x) for x in tif.series[0].shape)
        if len(shape) >= 2:
            return int(shape[-2]), int(shape[-1]), shape
    except Exception as exc:
        LOGGER.warning("Could not read TIFF shape %s: %s", path, exc)
    return None, None, None


def read_short(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], byteorder="big", signed=True)


def parse_imagej_roi(data: bytes) -> dict:
    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("Not an ImageJ ROI file")

    roi_type = data[6]
    top = read_short(data, 8)
    left = read_short(data, 10)
    bottom = read_short(data, 12)
    right = read_short(data, 14)
    n_coords = max(0, read_short(data, 16))

    width = max(0.0, float(right - left))
    height = max(0.0, float(bottom - top))
    points = []
    if roi_type in {0, 4, 5, 6, 7, 8} and n_coords > 0 and len(data) >= 64 + 4 * n_coords:
        x0 = 64
        y0 = 64 + 2 * n_coords
        xs = [left + read_short(data, x0 + 2 * i) for i in range(n_coords)]
        ys = [top + read_short(data, y0 + 2 * i) for i in range(n_coords)]
        points = list(zip(xs, ys))

    if points:
        area = polygon_area(points)
        perimeter = polygon_perimeter(points, closed=roi_type in {0, 4, 5})
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        width = float(max(xs) - min(xs))
        height = float(max(ys) - min(ys))
        x_mean = float(np.mean(xs))
        y_mean = float(np.mean(ys))
    elif roi_type == 2:
        area = math.pi * (width / 2.0) * (height / 2.0)
        perimeter = ellipse_perimeter(width, height)
        x_mean = float(left + width / 2.0)
        y_mean = float(top + height / 2.0)
    else:
        area = width * height
        perimeter = 2.0 * (width + height) if width > 0 and height > 0 else np.nan
        x_mean = float(left + width / 2.0)
        y_mean = float(top + height / 2.0)

    equiv_diameter = 2.0 * math.sqrt(area / math.pi) if area > 0 else np.nan
    circularity = (4.0 * math.pi * area / (perimeter**2)) if area > 0 and perimeter and perimeter > 0 else np.nan
    aspect_ratio = max(width, height) / max(min(width, height), 1e-6) if width > 0 and height > 0 else np.nan

    return {
        "roi_type": int(roi_type),
        "roi_type_name": ROI_TYPE_NAMES.get(int(roi_type), f"type_{roi_type}"),
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "width_px": width,
        "height_px": height,
        "area_px": float(area),
        "perimeter_px": float(perimeter) if np.isfinite(perimeter) else np.nan,
        "equiv_diameter_px": float(equiv_diameter) if np.isfinite(equiv_diameter) else np.nan,
        "circularity": float(circularity) if np.isfinite(circularity) else np.nan,
        "aspect_ratio": float(aspect_ratio) if np.isfinite(aspect_ratio) else np.nan,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "n_coordinates": int(n_coords),
    }


def polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.asarray([p[0] for p in points], dtype=float)
    y = np.asarray([p[1] for p in points], dtype=float)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: list[tuple[int, int]], closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    values = []
    limit = len(points) if closed else len(points) - 1
    for i in range(limit):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        values.append(math.hypot(x1 - x0, y1 - y0))
    return float(np.sum(values))


def ellipse_perimeter(width: float, height: float) -> float:
    a = width / 2.0
    b = height / 2.0
    if a <= 0 or b <= 0:
        return np.nan
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return float(math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h))))


def parse_roi_zip(pair: ManualPair) -> tuple[list[dict], dict]:
    rows = []
    height, width, tiff_shape = read_tiff_shape(pair.reference_tif)
    with zipfile.ZipFile(pair.roi_zip) as archive:
        roi_names = [name for name in archive.namelist() if name.lower().endswith(".roi") and not Path(name).name.startswith("._")]
        for roi_index, name in enumerate(sorted(roi_names), start=1):
            try:
                parsed = parse_imagej_roi(archive.read(name))
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "rel_parent": str(pair.rel_parent),
                        "folder": str(pair.folder),
                        "roi_zip": str(pair.roi_zip),
                        "reference_tif": str(pair.reference_tif) if pair.reference_tif else None,
                        "tiff_height": height,
                        "tiff_width": width,
                        "tiff_shape": str(tiff_shape) if tiff_shape is not None else None,
                        "roi_file": name,
                        "roi_index": roi_index,
                        **parsed,
                    }
                )
            except Exception as exc:
                LOGGER.warning("Could not parse ROI %s in %s: %s", name, pair.roi_zip, exc)
    pair_row = {
        "pair_id": pair.pair_id,
        "rel_parent": str(pair.rel_parent),
        "folder": str(pair.folder),
        "roi_zip": str(pair.roi_zip),
        "reference_tif": str(pair.reference_tif) if pair.reference_tif else None,
        "tiff_height": height,
        "tiff_width": width,
        "tiff_shape": str(tiff_shape) if tiff_shape is not None else None,
        "n_roi": len(rows),
    }
    return rows, pair_row


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


def finite_values(rows: list[dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            value = np.nan
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def percentile_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def build_prior(rows: list[dict], pair_rows: list[dict], manual_root: Path) -> dict:
    metrics = {
        "area_px": percentile_summary(finite_values(rows, "area_px")),
        "equiv_diameter_px": percentile_summary(finite_values(rows, "equiv_diameter_px")),
        "width_px": percentile_summary(finite_values(rows, "width_px")),
        "height_px": percentile_summary(finite_values(rows, "height_px")),
        "circularity": percentile_summary(finite_values(rows, "circularity")),
        "aspect_ratio": percentile_summary(finite_values(rows, "aspect_ratio")),
    }
    return {
        "manual_root": str(manual_root),
        "n_pairs": len(pair_rows),
        "n_roi": len(rows),
        "metrics": metrics,
        "recommended_soft_rules": {
            "area_px_min": metrics["area_px"].get("p05"),
            "area_px_max": metrics["area_px"].get("p95"),
            "equiv_diameter_px_min": metrics["equiv_diameter_px"].get("p05"),
            "equiv_diameter_px_max": metrics["equiv_diameter_px"].get("p95"),
            "aspect_ratio_max": metrics["aspect_ratio"].get("p95"),
            "circularity_min": metrics["circularity"].get("p05"),
        },
    }


def save_summary_csv(path: Path, prior: dict) -> None:
    rows = []
    for metric, summary in prior["metrics"].items():
        row = {"metric": metric, **summary}
        rows.append(row)
    write_csv(path, rows)


def save_distribution_plot(values: np.ndarray, output_path: Path, title: str, xlabel: str, dpi: int) -> None:
    if values.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(30, max(5, int(math.sqrt(values.size)))), color="tab:blue", alpha=0.8)
    ax.axvline(np.median(values), color="black", lw=1.2, label="median")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize historical manually drawn ImageJ ROI sets.")
    parser.add_argument("--manual-root", type=Path, required=True, help="Folder containing RoiSet.zip files and matching TIFFs.")
    parser.add_argument("--data-root", type=Path, help="Optional pipeline data root used as default output root.")
    parser.add_argument("--output-root", type=Path, help="Output root. Default: DATA_ROOT if set, else MANUAL_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    manual_root = args.manual_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root, manual_root).resolve()
    out_root = step_output_root(output_root)

    if not manual_root.exists():
        LOGGER.error("Manual ROI root does not exist: %s", manual_root)
        return 1

    pairs = discover_pairs(manual_root)
    summary = RunSummary(found=len(pairs))
    LOGGER.info("Manual root: %s", manual_root)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d RoiSet.zip file(s).", len(pairs))

    if args.dry_run:
        for pair in pairs:
            LOGGER.info("[dry-run] %s | tif=%s | roi=%s", pair.folder, pair.reference_tif, pair.roi_zip)
        LOGGER.info("Summary:")
        LOGGER.info("  found: %d", summary.found)
        LOGGER.info("  processed or would process: %d", len(pairs))
        LOGGER.info("  skipped: 0")
        LOGGER.info("  failed: 0")
        return 0

    if out_root.exists() and args.action == "skip" and (out_root / "manual_roi_prior.json").exists():
        LOGGER.info("[skip] Existing manual ROI prior: %s", out_root / "manual_roi_prior.json")
        return 0
    if args.action == "overwrite":
        clean_step_outputs(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    roi_rows: list[dict] = []
    pair_rows: list[dict] = []
    for pair in pairs:
        try:
            rows, pair_row = parse_roi_zip(pair)
            roi_rows.extend(rows)
            pair_rows.append(pair_row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_roi=%d", pair.pair_id, len(rows))
        except Exception as exc:
            summary.failed += 1
            LOGGER.exception("[failed] %s: %s", pair.pair_id, exc)

    write_csv(out_root / "manual_roi_pairs.csv", pair_rows)
    write_csv(out_root / "manual_roi_table.csv", roi_rows)
    prior = build_prior(roi_rows, pair_rows, manual_root=manual_root)
    with (out_root / "manual_roi_prior.json").open("w", encoding="utf-8") as handle:
        json.dump(prior, handle, indent=2)
    save_summary_csv(out_root / "manual_roi_shape_summary.csv", prior)
    save_distribution_plot(
        finite_values(roi_rows, "area_px"),
        out_root / "manual_roi_area_distribution.png",
        title="Manual ROI area distribution",
        xlabel="Area (px)",
        dpi=args.dpi,
    )
    save_distribution_plot(
        finite_values(roi_rows, "equiv_diameter_px"),
        out_root / "manual_roi_diameter_distribution.png",
        title="Manual ROI equivalent diameter distribution",
        xlabel="Equivalent diameter (px)",
        dpi=args.dpi,
    )

    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    LOGGER.info("  manual ROI count: %d", len(roi_rows))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
