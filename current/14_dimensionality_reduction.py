#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run lightweight dimensionality reduction on ROI feature matrices."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("dimensionality_reduction")
STEP_NAME = "14_dimensionality_reduction"
STEP_OUTPUT_PATTERNS = (
    "*_pca_embedding.csv",
    "*_pca_variance.csv",
    "*_umap_embedding.csv",
    "*_tsne_embedding.csv",
    "*_embedding_summary.json",
    "*_pca_plot.png",
    "*_pca_plot.pdf",
    "*_umap_plot.png",
    "*_umap_plot.pdf",
    "*_tsne_plot.png",
    "*_tsne_plot.pdf",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    feature_matrix_path: Path
    cluster_labels_path: Path | None


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


def discover_trials(output_root: Path, input_root: Path) -> list[TrialInput]:
    cluster_root = output_root / "12_hierarchical_clustering"
    trials = []
    for feature_path in sorted(input_root.rglob("*_roi_feature_matrix_zscored.csv")):
        input_dir = feature_path.parent
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                feature_matrix_path=feature_path,
                cluster_labels_path=find_first_existing(cluster_root / rel_parent / trial_id, ("*_hierarchical_cluster_labels.csv",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_pca_embedding.csv",
        out_dir / f"{trial_id}_pca_variance.csv",
        out_dir / f"{trial_id}_embedding_summary.json",
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


def numeric_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    skip = {"trial_id", "roi_id", "suite2p_original_id", "response_type"}
    cols = [c for c in df.columns if c not in skip]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.to_numpy(dtype=np.float32), list(numeric.columns)


def attach_labels(embedding: pd.DataFrame, feature_df: pd.DataFrame, cluster_path: Path | None) -> pd.DataFrame:
    for col in ("trial_id", "roi_id", "suite2p_original_id", "preferred_angle", "max_response", "event_rate_hz", "response_type", "x_mean", "y_mean"):
        if col in feature_df.columns:
            embedding[col] = feature_df[col].values
    if cluster_path and cluster_path.exists():
        clusters = pd.read_csv(cluster_path)
        if "roi_id" in clusters.columns:
            embedding = embedding.merge(clusters[["roi_id", "hierarchical_cluster"]], on="roi_id", how="left")
    return embedding


def save_scatter(table: pd.DataFrame, out_path: Path, trial_id: str, x: str, y: str, color_col: str | None, dpi: int) -> None:
    if table.empty or x not in table or y not in table:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    color = pd.to_numeric(table[color_col], errors="coerce") if color_col and color_col in table else None
    scatter = ax.scatter(table[x], table[y], c=color, s=12, cmap="viridis", alpha=0.85)
    if color is not None:
        fig.colorbar(scatter, ax=ax, label=color_col)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"PCA embedding - {trial_id}")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def run_optional_umap(matrix: np.ndarray, n_neighbors: int, min_dist: float) -> tuple[np.ndarray | None, str | None]:
    try:
        import umap  # type: ignore
    except Exception as exc:
        return None, f"umap skipped: {exc}"
    reducer = umap.UMAP(n_neighbors=min(n_neighbors, max(2, matrix.shape[0] - 1)), min_dist=min_dist, random_state=0)
    return reducer.fit_transform(matrix), None


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    from sklearn.decomposition import PCA

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df = pd.read_csv(trial.feature_matrix_path)
    matrix, feature_names = numeric_features(feature_df)
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 ROI for dimensionality reduction")

    n_components = min(args.pca_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    embedding = pca.fit_transform(matrix)
    pca_table = pd.DataFrame({f"PC{i + 1}": embedding[:, i] for i in range(n_components)})
    pca_table = attach_labels(pca_table, feature_df, trial.cluster_labels_path)
    pca_table.to_csv(out_dir / f"{trial.trial_id}_pca_embedding.csv", index=False)
    pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(out_dir / f"{trial.trial_id}_pca_variance.csv", index=False)
    save_scatter(pca_table, out_dir / f"{trial.trial_id}_pca_plot.png", trial.trial_id, "PC1", "PC2" if n_components > 1 else "PC1", "hierarchical_cluster", args.dpi)

    umap_status = "not_requested"
    if args.run_umap:
        umap_embedding, warning = run_optional_umap(matrix, args.umap_neighbors, args.umap_min_dist)
        if umap_embedding is None:
            umap_status = warning or "umap skipped"
            LOGGER.warning("%s: %s", trial.trial_id, umap_status)
        else:
            umap_table = pd.DataFrame({"UMAP1": umap_embedding[:, 0], "UMAP2": umap_embedding[:, 1]})
            umap_table = attach_labels(umap_table, feature_df, trial.cluster_labels_path)
            umap_table.to_csv(out_dir / f"{trial.trial_id}_umap_embedding.csv", index=False)
            save_scatter(umap_table, out_dir / f"{trial.trial_id}_umap_plot.png", trial.trial_id, "UMAP1", "UMAP2", "hierarchical_cluster", args.dpi)
            umap_status = "ok"

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "pca_components": int(n_components),
        "umap_status": umap_status,
        "feature_names": feature_names,
    }
    (out_dir / f"{trial.trial_id}_embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PCA and optional UMAP on ROI features.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-10 root. Default: OUTPUT_ROOT/10_population_features.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--run-umap", action="store_true")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
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
    trials = discover_trials(output_root, input_root)
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
            LOGGER.info("[dry-run] Would embed %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            _, row = process_trial(trial, out_dir, args)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_roi=%s", trial.trial_id, row.get("n_roi"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "embedding_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
