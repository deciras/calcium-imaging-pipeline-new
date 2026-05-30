#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate lightweight HTML reports from pipeline outputs."""

from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("report_generator")
STEP_NAME = "16_reports"
STEP_OUTPUT_PATTERNS = ("*.html",)


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
        for path in out_dir.glob(pattern):
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


def image_tag(path: Path, base: Path, caption: str) -> str:
    if not path.exists():
        return ""
    return (
        '<figure>'
        f'<img src="../{html.escape(rel(path, base))}" alt="{html.escape(caption)}">'
        f'<figcaption>{html.escape(caption)}</figcaption>'
        '</figure>'
    )


def table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "<p>No table available.</p>"
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, escape=True, classes="data-table")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def trial_ids(output_root: Path) -> list[str]:
    roots = [
        output_root / "06_dff",
        output_root / "08_stim_response",
        output_root / "10_population_features",
    ]
    ids = set()
    for root in roots:
        if root.exists():
            ids.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(ids)


def css() -> str:
    return """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #202124; background: #f7f8fa; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1, h2, h3 { margin: 0.8em 0 0.4em; }
.panel { background: white; border: 1px solid #d9dde3; border-radius: 8px; padding: 18px; margin: 16px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
figure { margin: 0; background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 10px; }
img { max-width: 100%; height: auto; display: block; }
figcaption { color: #5f6368; font-size: 13px; margin-top: 8px; }
.data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.data-table th, .data-table td { border: 1px solid #d9dde3; padding: 5px 7px; text-align: left; }
.data-table th { background: #eef1f5; }
.muted { color: #5f6368; }
a { color: #1457b8; text-decoration: none; }
"""


def write_trial_report(output_root: Path, report_root: Path, trial_id: str) -> Path:
    sections = []
    dff_dir = output_root / "06_dff" / trial_id
    event_dir = output_root / "07_events" / trial_id
    stim_dir = output_root / "08_stim_response" / trial_id
    angle_dir = output_root / "09_angle_tuning" / trial_id
    cluster_dir = output_root / "12_hierarchical_clustering" / trial_id
    leiden_dir = output_root / "13_leiden" / trial_id
    embed_dir = output_root / "14_dimensionality_reduction" / trial_id

    summary = {
        "dff": load_json(dff_dir / f"{trial_id}_dff_summary.json"),
        "stim_response": load_json(stim_dir / f"{trial_id}_stim_response_summary.json"),
        "angle": load_json(angle_dir / f"{trial_id}_angle_tuning_summary.json"),
        "cluster": load_json(cluster_dir / f"{trial_id}_hierarchical_clustering_summary.json"),
        "leiden": load_json(leiden_dir / f"{trial_id}_leiden_summary.json"),
        "embedding": load_json(embed_dir / f"{trial_id}_embedding_summary.json"),
    }

    summary_rows = []
    for name, values in summary.items():
        if values:
            summary_rows.append({"section": name, **{k: v for k, v in values.items() if k in ("status", "n_roi", "n_frames", "n_events", "n_stim_events", "n_communities", "n_clusters")}})
    sections.append('<section class="panel"><h2>Summary</h2>' + table_html(pd.DataFrame(summary_rows), 30) + "</section>")

    figures = [
        image_tag(dff_dir / f"{trial_id}_dff_heatmap.png", output_root, "dF/F heatmap"),
        image_tag(dff_dir / f"{trial_id}_example_traces.png", output_root, "Example dF/F traces"),
        image_tag(event_dir / f"{trial_id}_event_raster.png", output_root, "Calcium event raster"),
        image_tag(stim_dir / f"{trial_id}_stim_response_heatmap.png", output_root, "Stimulus response heatmap"),
        image_tag(angle_dir / f"{trial_id}_angle_heatmap.png", output_root, "Angle response heatmap"),
        image_tag(cluster_dir / f"{trial_id}_clustered_heatmap.png", output_root, "Clustered feature heatmap"),
        image_tag(leiden_dir / f"{trial_id}_leiden_roi_spatial_map.png", output_root, "Leiden spatial map"),
        image_tag(embed_dir / f"{trial_id}_pca_plot.png", output_root, "PCA embedding"),
    ]
    sections.append('<section class="panel"><h2>Figures</h2><div class="grid">' + "".join(fig for fig in figures if fig) + "</div></section>")

    response_table = read_csv(stim_dir / f"{trial_id}_roi_response_summary.csv")
    feature_table = read_csv(output_root / "10_population_features" / trial_id / f"{trial_id}_roi_feature_matrix.csv")
    sections.append('<section class="panel"><h2>ROI Response Preview</h2>' + table_html(response_table, 15) + "</section>")
    sections.append('<section class="panel"><h2>Feature Matrix Preview</h2>' + table_html(feature_table, 15) + "</section>")

    out_path = report_root / f"{trial_id}_report.html"
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(trial_id)} report</title><style>{css()}</style></head>
<body><main><p><a href="global_report.html">Back to global report</a></p><h1>{html.escape(trial_id)}</h1>{''.join(sections)}</main></body></html>
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_global_report(output_root: Path, report_root: Path, trial_report_paths: list[Path]) -> Path:
    qc = read_csv(output_root / "15_cross_trial_summary" / "cross_trial_qc_summary.csv")
    summary = load_json(output_root / "15_cross_trial_summary" / "cross_trial_summary.json")
    links = "".join(f'<li><a href="{html.escape(path.name)}">{html.escape(path.stem.replace("_report", ""))}</a></li>' for path in trial_report_paths)
    figures = [
        image_tag(output_root / "15_cross_trial_summary" / "cross_trial_roi_count.png", output_root, "ROI count by trial"),
        image_tag(output_root / "15_cross_trial_summary" / "cross_trial_responsive_fraction.png", output_root, "Responsive fraction by trial"),
        image_tag(output_root / "15_cross_trial_summary" / "cross_trial_angle_selective_fraction.png", output_root, "Angle selective fraction by trial"),
    ]
    summary_table = pd.DataFrame([summary]) if summary else pd.DataFrame()
    out_path = report_root / "global_report.html"
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Calcium imaging pipeline report</title><style>{css()}</style></head>
<body><main>
<h1>Calcium Imaging Pipeline Report</h1>
<section class="panel"><h2>Global Summary</h2>{table_html(summary_table, 5)}</section>
<section class="panel"><h2>Trial Reports</h2><ul>{links}</ul></section>
<section class="panel"><h2>Cross-Trial QC</h2>{table_html(qc, 50)}</section>
<section class="panel"><h2>Figures</h2><div class="grid">{''.join(fig for fig in figures if fig)}</div></section>
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
    reports = [write_trial_report(output_root, report_root, trial_id) for trial_id in trial_ids(output_root)]
    global_report = write_global_report(output_root, report_root, reports)
    LOGGER.info("Generated %d trial report(s).", len(reports))
    LOGGER.info("Global report: %s", global_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
