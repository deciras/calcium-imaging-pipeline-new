#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run hierarchical clustering on ROI feature matrices."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("hierarchical_clustering")
STEP_NAME = "12_hierarchical_clustering"
STEP_OUTPUT_PATTERNS = (
    "*_hierarchical_cluster_labels.csv",
    "*_hierarchical_cluster_summary.csv",
    "*_dendrogram.png",
    "*_dendrogram.pdf",
    "*_clustered_heatmap.png",
    "*_clustered_heatmap.pdf",
    "*_cluster_mean_traces.png",
    "*_cluster_mean_traces.pdf",
    "*_cluster_mean_angle_tuning.png",
    "*_cluster_mean_angle_tuning.pdf",
    "*_hierarchical_clustering_summary.json",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    feature_dir: Path
    feature_matrix_path: Path
    trace_matrix_path: Path | None
    angle_response_matrix_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_input_root(output_root: Path) -> Path:
    return output_root / "10_population_features"


def step_output_root(output_root: Path) -> Path:
    return output_root / STEP_NAME


def trial_output_dir(step_root: Path, trial: TrialInput) -> Path:
    return step_root / trial.rel_parent / trial.trial_id if str(trial.rel_parent) not in ("", ".") else step_root / trial.trial_id


def find_first_existing(folder: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_trials(input_root: Path) -> list[TrialInput]:
    trials = []
    for feature_path in sorted(input_root.rglob("*_roi_feature_matrix_zscored.csv")):
        feature_dir = feature_path.parent
        trial_id = feature_dir.name
        rel_parent = feature_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                feature_dir=feature_dir,
                feature_matrix_path=feature_path,
                trace_matrix_path=find_first_existing(feature_dir, ("*_trace_matrix.npy",)),
                angle_response_matrix_path=find_first_existing(feature_dir, ("*_angle_response_matrix.npy",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_hierarchical_cluster_labels.csv",
        out_dir / f"{trial_id}_hierarchical_cluster_summary.csv",
        out_dir / f"{trial_id}_hierarchical_clustering_summary.json",
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
            else:
                path.unlink()
            removed += 1
    return removed


def numeric_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    skip = {"trial_id", "roi_id", "suite2p_original_id", "response_type"}
    cols = [c for c in df.columns if c not in skip]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.to_numpy(dtype=np.float32), list(numeric.columns)


def save_dendrogram(linkage_matrix: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(f"Hierarchical clustering - {trial_id}")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_clustered_heatmap(matrix: np.ndarray, order: np.ndarray, labels: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if matrix.size == 0:
        return
    import matplotlib.pyplot as plt

    ordered = matrix[order]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(ordered, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(f"Clustered feature heatmap - {trial_id}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("ROI ordered by cluster")
    fig.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_cluster_mean_traces(trace_matrix: np.ndarray, labels: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if trace_matrix.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster in sorted(set(labels)):
        mean_trace = np.nanmean(trace_matrix[labels == cluster], axis=0)
        ax.plot(mean_trace, lw=1, label=f"C{cluster}")
    ax.set_title(f"Cluster mean traces - {trial_id}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean dF/F")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_cluster_mean_angle(angle_matrix: np.ndarray, labels: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if angle_matrix.size == 0 or angle_matrix.shape[0] != len(labels):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for cluster in sorted(set(labels)):
        ax.plot(np.nanmean(angle_matrix[labels == cluster], axis=0), marker="o", lw=1, label=f"C{cluster}")
    ax.set_title(f"Cluster mean angle response - {trial_id}")
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Mean response")
    ax.legend(fontsize=8)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    from scipy.cluster.hierarchy import fcluster, linkage, leaves_list

    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(trial.feature_matrix_path)
    matrix, feature_names = numeric_feature_matrix(features)
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 ROI for hierarchical clustering")
    method = args.linkage_method
    metric = args.distance_metric
    if method == "ward" and metric != "euclidean":
        LOGGER.warning("Ward linkage requires euclidean distance; using euclidean.")
        metric = "euclidean"
    linkage_matrix = linkage(matrix, method=method, metric=metric)
    labels = fcluster(linkage_matrix, t=args.n_clusters, criterion="maxclust").astype(int)
    order = leaves_list(linkage_matrix)

    labels_df = features[["trial_id", "roi_id", "suite2p_original_id"]].copy()
    labels_df["hierarchical_cluster"] = labels
    labels_df.to_csv(out_dir / f"{trial.trial_id}_hierarchical_cluster_labels.csv", index=False)
    cluster_summary = labels_df.groupby("hierarchical_cluster", as_index=False).agg(n_roi=("roi_id", "count"))
    cluster_summary.to_csv(out_dir / f"{trial.trial_id}_hierarchical_cluster_summary.csv", index=False)

    save_dendrogram(linkage_matrix, out_dir / f"{trial.trial_id}_dendrogram.png", trial.trial_id, args.dpi)
    save_clustered_heatmap(matrix, order, labels, out_dir / f"{trial.trial_id}_clustered_heatmap.png", trial.trial_id, args.dpi)
    if trial.trace_matrix_path and trial.trace_matrix_path.exists():
        save_cluster_mean_traces(np.load(trial.trace_matrix_path, allow_pickle=True), labels, out_dir / f"{trial.trial_id}_cluster_mean_traces.png", trial.trial_id, args.dpi)
    if trial.angle_response_matrix_path and trial.angle_response_matrix_path.exists():
        save_cluster_mean_angle(np.load(trial.angle_response_matrix_path, allow_pickle=True), labels, out_dir / f"{trial.trial_id}_cluster_mean_angle_tuning.png", trial.trial_id, args.dpi)

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_clusters": int(args.n_clusters),
        "linkage_method": method,
        "distance_metric": metric,
        "feature_names": feature_names,
    }
    (out_dir / f"{trial.trial_id}_hierarchical_clustering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hierarchical clustering on ROI features.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-10 root. Default: OUTPUT_ROOT/10_population_features.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--linkage-method", choices=("ward", "average", "complete"), default="ward")
    parser.add_argument("--distance-metric", choices=("euclidean", "correlation", "cosine"), default="euclidean")
    parser.add_argument("--n-clusters", type=int, default=6)
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
    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1
    trials = discover_trials(input_root)
    summary = RunSummary(found=len(trials))
    rows = []
    LOGGER.info("Input root : %s", input_root)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Found %d trial(s).", len(trials))
    for trial in trials:
        out_dir = trial_output_dir(out_root, trial)
        if required_outputs_done(out_dir, trial.trial_id) and args.action == "skip":
            summary.skipped += 1
            LOGGER.info("[skip] %s", trial.trial_id)
            continue
        if args.dry_run:
            summary.processed += 1
            LOGGER.info("[dry-run] Would cluster %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            _, row = process_trial(trial, out_dir, args)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_clusters=%s", trial.trial_id, row.get("n_clusters"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "hierarchical_clustering_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
