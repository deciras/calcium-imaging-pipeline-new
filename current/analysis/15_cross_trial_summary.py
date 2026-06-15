#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create cross-trial summary tables and simple QC plots."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("cross_trial_summary")
STEP_NAME = "15_cross_trial_summary"
STEP_OUTPUT_PATTERNS = (
    "all_trials_roi_summary.csv",
    "all_trials_roi_features.csv",
    "all_trials_event_summary.csv",
    "all_trials_event_table.csv",
    "all_trials_stim_response_summary.csv",
    "all_trials_stim_response_table.csv",
    "all_trials_angle_tuning_summary.csv",
    "all_trials_angle_response_table.csv",
    "all_trials_cluster_summary.csv",
    "all_trials_cluster_labels.csv",
    "all_trials_embedding_summary.csv",
    "all_trials_pca_embedding.csv",
    "top_responsive_rois.csv",
    "top_event_rois.csv",
    "top_angle_selective_rois.csv",
    "cross_trial_qc_summary.csv",
    "cross_trial_summary.json",
    "cross_trial_roi_count.png",
    "cross_trial_roi_count.pdf",
    "cross_trial_responsive_fraction.png",
    "cross_trial_responsive_fraction.pdf",
    "cross_trial_angle_selective_fraction.png",
    "cross_trial_angle_selective_fraction.pdf",
)


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
            if path.is_dir():
                shutil.rmtree(path)
            else:
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


def collect_trial_tables(root: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(f"*/{pattern}")):
        df = read_csv(path)
        if not df.empty:
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def build_qc_table(output_root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tables: dict[str, pd.DataFrame] = {}
    tables["dff"] = read_csv(output_root / "06_dff" / "dff_summary.csv")
    tables["events"] = read_csv(output_root / "07_events" / "event_summary.csv")
    tables["stim"] = read_csv(output_root / "08_stim_response" / "stim_response_summary.csv")
    tables["angle"] = read_csv(output_root / "09_angle_tuning" / "angle_tuning_summary.csv")
    tables["features"] = read_csv(output_root / "10_population_features" / "population_feature_summary.csv")
    tables["similarity"] = read_csv(output_root / "11_population_similarity" / "population_similarity_summary.csv")
    tables["clusters"] = read_csv(output_root / "12_hierarchical_clustering" / "hierarchical_clustering_summary.csv")
    tables["leiden"] = read_csv(output_root / "13_leiden" / "leiden_summary.csv")
    tables["embedding"] = read_csv(output_root / "14_dimensionality_reduction" / "embedding_summary.csv")
    tables["roi_features"] = collect_trial_tables(output_root / "10_population_features", "*_roi_feature_matrix.csv")
    tables["event_table"] = collect_trial_tables(output_root / "07_events", "*_event_table.csv")
    tables["stim_response_table"] = collect_trial_tables(output_root / "08_stim_response", "*_stim_response_table.csv")
    tables["angle_response_table"] = collect_trial_tables(output_root / "09_angle_tuning", "*_angle_response_table.csv")
    tables["cluster_labels"] = collect_trial_tables(output_root / "12_hierarchical_clustering", "*_hierarchical_cluster_labels.csv")
    tables["pca_embedding"] = collect_trial_tables(output_root / "14_dimensionality_reduction", "*_pca_embedding.csv")

    trial_ids = sorted(set().union(*(set(df["trial_id"].dropna().astype(str)) for df in tables.values() if "trial_id" in df.columns)))
    qc_rows = []
    for trial_id in trial_ids:
        row = {"trial_id": trial_id}
        dff = tables["dff"][tables["dff"]["trial_id"].astype(str) == trial_id] if "trial_id" in tables["dff"] else pd.DataFrame()
        events = tables["events"][tables["events"]["trial_id"].astype(str) == trial_id] if "trial_id" in tables["events"] else pd.DataFrame()
        stim = tables["stim"][tables["stim"]["trial_id"].astype(str) == trial_id] if "trial_id" in tables["stim"] else pd.DataFrame()
        angle = tables["angle"][tables["angle"]["trial_id"].astype(str) == trial_id] if "trial_id" in tables["angle"] else pd.DataFrame()
        clusters = tables["clusters"][tables["clusters"]["trial_id"].astype(str) == trial_id] if "trial_id" in tables["clusters"] else pd.DataFrame()
        leiden = tables["leiden"][tables["leiden"]["trial_id"].astype(str) == trial_id] if "trial_id" in tables["leiden"] else pd.DataFrame()

        row["n_roi"] = int(dff["n_roi_selected"].iloc[0]) if not dff.empty and "n_roi_selected" in dff else np.nan
        row["n_frames"] = int(dff["n_frames"].iloc[0]) if not dff.empty and "n_frames" in dff else np.nan
        row["n_events"] = int(events["n_events"].iloc[0]) if not events.empty and "n_events" in events else np.nan
        row["mean_event_rate_hz"] = float(events["mean_event_rate_hz"].iloc[0]) if not events.empty and "mean_event_rate_hz" in events else np.nan
        row["n_stim_events"] = int(stim["n_stim_events"].iloc[0]) if not stim.empty and "n_stim_events" in stim else np.nan
        row["n_responsive_roi"] = int(stim["n_responsive_roi"].iloc[0]) if not stim.empty and "n_responsive_roi" in stim else np.nan
        row["fraction_responsive"] = row["n_responsive_roi"] / row["n_roi"] if row.get("n_roi", 0) else np.nan
        row["n_angle_selective_roi"] = int(angle["n_angle_selective_roi"].iloc[0]) if not angle.empty and "n_angle_selective_roi" in angle else np.nan
        row["fraction_angle_selective"] = row["n_angle_selective_roi"] / row["n_roi"] if row.get("n_roi", 0) else np.nan
        row["n_hierarchical_clusters"] = int(clusters["n_clusters"].iloc[0]) if not clusters.empty and "n_clusters" in clusters else np.nan
        row["n_leiden_communities"] = int(leiden["n_communities"].iloc[0]) if not leiden.empty and "n_communities" in leiden else np.nan

        warnings = []
        if row.get("n_roi", 0) == 0:
            warnings.append("no_roi")
        if row.get("n_stim_events", np.nan) == 0:
            warnings.append("no_stim")
        if not leiden.empty and str(leiden.get("status", pd.Series([""])).iloc[0]) == "skipped":
            warnings.append("leiden_skipped")
        row["qc_warnings"] = ";".join(warnings)
        qc_rows.append(row)
    return pd.DataFrame(qc_rows), tables


def save_bar(table: pd.DataFrame, y: str, out_path: Path, title: str, ylabel: str, dpi: int) -> None:
    if table.empty or y not in table:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.65 * len(table)), 4))
    ax.bar(table["trial_id"].astype(str), pd.to_numeric(table[y], errors="coerce").fillna(0))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def write_top_roi_tables(tables: dict[str, pd.DataFrame], out_root: Path, top_n: int) -> None:
    roi_features = tables.get("roi_features", pd.DataFrame())
    stim = tables.get("stim_response_table", pd.DataFrame())
    angle = tables.get("angle_response_table", pd.DataFrame())

    if not stim.empty and "zscore_response" in stim.columns:
        top_responsive = (
            stim.sort_values("zscore_response", ascending=False)
            .groupby("trial_id", as_index=False)
            .head(top_n)
        )
        top_responsive.to_csv(out_root / "top_responsive_rois.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_root / "top_responsive_rois.csv", index=False)

    if not roi_features.empty and "event_rate_hz" in roi_features.columns:
        top_event = (
            roi_features.sort_values("event_rate_hz", ascending=False)
            .groupby("trial_id", as_index=False)
            .head(top_n)
        )
        top_event.to_csv(out_root / "top_event_rois.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_root / "top_event_rois.csv", index=False)

    if not angle.empty and "reliability" in angle.columns:
        sort_cols = [col for col in ("reliability", "mean_response") if col in angle.columns]
        top_angle = (
            angle.sort_values(sort_cols, ascending=False)
            .groupby("trial_id", as_index=False)
            .head(top_n)
        )
        top_angle.to_csv(out_root / "top_angle_selective_rois.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_root / "top_angle_selective_rois.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create cross-trial summaries.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--top-n-roi", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else default_output_root(data_root).resolve()
    out_root = step_output_root(output_root)
    if out_root.exists() and args.action == "skip" and (out_root / "cross_trial_summary.json").exists():
        LOGGER.info("[skip] cross-trial summary already exists: %s", out_root)
        return 0
    if args.dry_run:
        LOGGER.info("[dry-run] Would summarize pipeline outputs under %s -> %s", output_root, out_root)
        return 0
    if args.action == "overwrite":
        clean_step_outputs(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    qc, tables = build_qc_table(output_root)
    tables["dff"].to_csv(out_root / "all_trials_roi_summary.csv", index=False)
    tables["roi_features"].to_csv(out_root / "all_trials_roi_features.csv", index=False)
    tables["events"].to_csv(out_root / "all_trials_event_summary.csv", index=False)
    tables["event_table"].to_csv(out_root / "all_trials_event_table.csv", index=False)
    tables["stim"].to_csv(out_root / "all_trials_stim_response_summary.csv", index=False)
    tables["stim_response_table"].to_csv(out_root / "all_trials_stim_response_table.csv", index=False)
    tables["angle"].to_csv(out_root / "all_trials_angle_tuning_summary.csv", index=False)
    tables["angle_response_table"].to_csv(out_root / "all_trials_angle_response_table.csv", index=False)
    tables["clusters"].to_csv(out_root / "all_trials_cluster_summary.csv", index=False)
    tables["cluster_labels"].to_csv(out_root / "all_trials_cluster_labels.csv", index=False)
    tables["embedding"].to_csv(out_root / "all_trials_embedding_summary.csv", index=False)
    tables["pca_embedding"].to_csv(out_root / "all_trials_pca_embedding.csv", index=False)
    qc.to_csv(out_root / "cross_trial_qc_summary.csv", index=False)
    write_top_roi_tables(tables, out_root, top_n=args.top_n_roi)

    save_bar(qc, "n_roi", out_root / "cross_trial_roi_count.png", "ROI count by trial", "ROI count", args.dpi)
    save_bar(qc, "fraction_responsive", out_root / "cross_trial_responsive_fraction.png", "Responsive fraction by trial", "fraction", args.dpi)
    save_bar(qc, "fraction_angle_selective", out_root / "cross_trial_angle_selective_fraction.png", "Angle selective fraction by trial", "fraction", args.dpi)

    summary = {
        "status": "ok",
        "n_trials": int(len(qc)),
        "n_roi_total": int(pd.to_numeric(qc.get("n_roi", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "n_trials_with_stim": int((pd.to_numeric(qc.get("n_stim_events", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
        "n_event_rows": int(len(tables["event_table"])),
        "n_stim_response_rows": int(len(tables["stim_response_table"])),
        "n_angle_response_rows": int(len(tables["angle_response_table"])),
        "top_n_roi": int(args.top_n_roi),
        "output_root": str(out_root),
    }
    (out_root / "cross_trial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Summary:")
    LOGGER.info("  trials: %d", summary["n_trials"])
    LOGGER.info("  total ROI: %d", summary["n_roi_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
