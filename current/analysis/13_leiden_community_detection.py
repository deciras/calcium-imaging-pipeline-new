#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run optional Leiden community detection on ROI similarity graphs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("leiden")
STEP_NAME = "13_leiden"
STEP_OUTPUT_PATTERNS = (
    "*_leiden_labels.csv",
    "*_graph_edges.csv",
    "*_leiden_community_summary.csv",
    "*_leiden_embedding.png",
    "*_leiden_embedding.pdf",
    "*_leiden_roi_spatial_map.png",
    "*_leiden_roi_spatial_map.pdf",
    "*_community_mean_traces.png",
    "*_community_mean_traces.pdf",
    "*_leiden_summary.json",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    similarity_dir: Path
    feature_dir: Path
    edge_path: Path | None
    feature_matrix_path: Path | None
    trace_matrix_path: Path | None


@dataclass
class RunSummary:
    found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_input_root(output_root: Path) -> Path:
    return output_root / "11_population_similarity"


def default_feature_root(output_root: Path) -> Path:
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


def discover_trials(similarity_root: Path, feature_root: Path) -> list[TrialInput]:
    trials = []
    for summary_path in sorted(similarity_root.rglob("*_population_similarity_summary.json")):
        similarity_dir = summary_path.parent
        trial_id = similarity_dir.name
        rel_parent = similarity_dir.parent.relative_to(similarity_root)
        feature_dir = feature_root / rel_parent / trial_id
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                similarity_dir=similarity_dir,
                feature_dir=feature_dir,
                edge_path=find_first_existing(similarity_dir, ("*_similarity_edges.csv",)),
                feature_matrix_path=find_first_existing(feature_dir, ("*_roi_feature_matrix.csv",)),
                trace_matrix_path=find_first_existing(feature_dir, ("*_trace_matrix.npy",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_leiden_labels.csv",
        out_dir / f"{trial_id}_graph_edges.csv",
        out_dir / f"{trial_id}_leiden_summary.json",
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


def load_optional_dependencies():
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except Exception as exc:
        return None, None, str(exc)
    return ig, leidenalg, None


def write_skipped_outputs(trial: TrialInput, out_dir: Path, reason: str) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(trial.feature_matrix_path) if trial.feature_matrix_path and trial.feature_matrix_path.exists() else pd.DataFrame()
    label_cols = [
        c
        for c in (
            "trial_id",
            "roi_id",
            "source_roi_id",
            "roi_source",
            "roi_type",
            "manual_roi_id",
            "suite2p_original_id",
            "previous_suite2p_original_id",
            "stat_index",
        )
        if c in features.columns
    ]
    labels = features[label_cols].copy() if label_cols else pd.DataFrame(columns=["trial_id", "roi_id"])
    labels["leiden_community"] = pd.Series(dtype="float")
    labels.to_csv(out_dir / f"{trial.trial_id}_leiden_labels.csv", index=False)
    pd.DataFrame(columns=["source_roi_id", "target_roi_id", "weight"]).to_csv(out_dir / f"{trial.trial_id}_graph_edges.csv", index=False)
    pd.DataFrame(columns=["leiden_community", "n_roi"]).to_csv(out_dir / f"{trial.trial_id}_leiden_community_summary.csv", index=False)
    summary = {
        "trial_id": trial.trial_id,
        "status": "skipped",
        "reason": reason,
        "n_roi": int(len(features)),
        "n_communities": 0,
    }
    (out_dir / f"{trial.trial_id}_leiden_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_graph(edges: pd.DataFrame, roi_ids: np.ndarray, ig):
    id_to_vertex = {int(roi_id): index for index, roi_id in enumerate(roi_ids)}
    graph_edges = []
    weights = []
    for _, row in edges.iterrows():
        source = int(row.get("source_roi_id"))
        target = int(row.get("target_roi_id"))
        if source not in id_to_vertex or target not in id_to_vertex:
            continue
        graph_edges.append((id_to_vertex[source], id_to_vertex[target]))
        weights.append(float(row.get("similarity", row.get("weight", 1.0))))
    graph = ig.Graph(n=len(roi_ids), edges=graph_edges, directed=False)
    if weights:
        graph.es["weight"] = weights
    return graph, graph_edges, weights


def save_embedding(features: pd.DataFrame, labels: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if features.empty or "x_mean" not in features.columns or "y_mean" not in features.columns:
        return
    import matplotlib.pyplot as plt

    merged = features.merge(labels[["roi_id", "leiden_community"]], on="roi_id", how="left")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(merged["x_mean"], merged["y_mean"], c=merged["leiden_community"], s=12, cmap="tab20", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Leiden spatial map - {trial_id}")
    fig.colorbar(sc, ax=ax, label="community")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def save_mean_traces(trace_matrix: np.ndarray, labels: pd.DataFrame, out_path: Path, trial_id: str, dpi: int) -> None:
    if trace_matrix.size == 0 or labels.empty or "leiden_community" not in labels.columns:
        return
    import matplotlib.pyplot as plt

    communities = pd.to_numeric(labels["leiden_community"], errors="coerce").to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for community in sorted(set(int(c) for c in communities if np.isfinite(c))):
        mask = communities == community
        if mask.any():
            ax.plot(np.nanmean(trace_matrix[mask], axis=0), lw=1, label=f"C{community}")
    ax.set_title(f"Leiden community mean traces - {trial_id}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean dF/F")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace, ig, leidenalg) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if trial.edge_path is None or not trial.edge_path.exists():
        return write_skipped_outputs(trial, out_dir, "missing similarity edge table")
    if trial.feature_matrix_path is None or not trial.feature_matrix_path.exists():
        return write_skipped_outputs(trial, out_dir, "missing feature matrix")

    features = pd.read_csv(trial.feature_matrix_path)
    edges = pd.read_csv(trial.edge_path)
    if features.empty or edges.empty:
        return write_skipped_outputs(trial, out_dir, "empty feature matrix or graph")
    roi_ids = features["roi_id"].astype(int).to_numpy()
    graph, graph_edges, weights = build_graph(edges, roi_ids, ig)
    if graph.ecount() == 0:
        return write_skipped_outputs(trial, out_dir, "graph has no edges")

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"] if weights else None,
        resolution_parameter=args.resolution,
        seed=args.random_seed,
    )
    communities = np.asarray(partition.membership, dtype=int)
    label_cols = [
        col
        for col in (
            "trial_id",
            "roi_id",
            "source_roi_id",
            "roi_source",
            "roi_type",
            "manual_roi_id",
            "suite2p_original_id",
            "previous_suite2p_original_id",
            "stat_index",
        )
        if col in features.columns
    ]
    labels = features[label_cols].copy()
    labels["leiden_community"] = communities
    labels.to_csv(out_dir / f"{trial.trial_id}_leiden_labels.csv", index=False)
    graph_edges_df = edges.rename(columns={"similarity": "weight"}).copy()
    graph_edges_df.to_csv(out_dir / f"{trial.trial_id}_graph_edges.csv", index=False)
    community_summary = labels.groupby("leiden_community", as_index=False).agg(n_roi=("roi_id", "count"))
    community_summary.to_csv(out_dir / f"{trial.trial_id}_leiden_community_summary.csv", index=False)

    save_embedding(features, labels, out_dir / f"{trial.trial_id}_leiden_roi_spatial_map.png", trial.trial_id, args.dpi)
    save_embedding(features, labels, out_dir / f"{trial.trial_id}_leiden_embedding.png", trial.trial_id, args.dpi)
    if trial.trace_matrix_path and trial.trace_matrix_path.exists():
        save_mean_traces(np.load(trial.trace_matrix_path, allow_pickle=True), labels, out_dir / f"{trial.trial_id}_community_mean_traces.png", trial.trial_id, args.dpi)

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(len(features)),
        "n_edges": int(graph.ecount()),
        "n_communities": int(len(set(communities.tolist()))),
        "resolution": float(args.resolution),
    }
    (out_dir / f"{trial.trial_id}_leiden_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run optional Leiden community detection.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-11 root. Default: OUTPUT_ROOT/11_population_similarity.")
    parser.add_argument("--feature-root", type=Path, help="Step-10 root. Default: OUTPUT_ROOT/10_population_features.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=0)
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
    feature_root = args.feature_root.expanduser().resolve() if args.feature_root else default_feature_root(output_root).resolve()
    out_root = step_output_root(output_root)
    if not input_root.exists():
        LOGGER.error("Input root does not exist: %s", input_root)
        return 1

    ig, leidenalg, missing_reason = load_optional_dependencies()
    if missing_reason:
        LOGGER.warning("Optional Leiden dependency missing: %s", missing_reason)

    trials = discover_trials(input_root, feature_root)
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
            LOGGER.info("[dry-run] Would run Leiden for %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            if missing_reason:
                _, row = write_skipped_outputs(trial, out_dir, f"optional dependency missing: {missing_reason}")
            else:
                _, row = process_trial(trial, out_dir, args, ig, leidenalg)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[%s] %s: n_communities=%s", row.get("status"), trial.trial_id, row.get("n_communities"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "leiden_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
