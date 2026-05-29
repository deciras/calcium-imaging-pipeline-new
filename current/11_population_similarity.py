#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute ROI-ROI similarity matrices from population feature outputs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("population_similarity")
STEP_NAME = "11_population_similarity"
STEP_OUTPUT_PATTERNS = (
    "*_trace_correlation_matrix.npy",
    "*_response_correlation_matrix.npy",
    "*_feature_similarity_matrix.npy",
    "*_distance_matrix.npy",
    "*_similarity_edges.csv",
    "*_similarity_heatmap.png",
    "*_similarity_heatmap.pdf",
    "*_population_similarity_summary.json",
)


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    rel_parent: Path
    input_dir: Path
    feature_matrix_path: Path
    z_feature_matrix_path: Path
    trace_matrix_path: Path | None
    response_matrix_path: Path | None


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
    for feature_path in sorted(input_root.rglob("*_roi_feature_matrix.csv")):
        input_dir = feature_path.parent
        trial_id = input_dir.name
        rel_parent = input_dir.parent.relative_to(input_root)
        trials.append(
            TrialInput(
                trial_id=trial_id,
                rel_parent=rel_parent,
                input_dir=input_dir,
                feature_matrix_path=feature_path,
                z_feature_matrix_path=input_dir / f"{trial_id}_roi_feature_matrix_zscored.csv",
                trace_matrix_path=find_first_existing(input_dir, ("*_trace_matrix.npy",)),
                response_matrix_path=find_first_existing(input_dir, ("*_response_matrix.npy",)),
            )
        )
    return trials


def required_outputs_done(out_dir: Path, trial_id: str) -> bool:
    required = (
        out_dir / f"{trial_id}_feature_similarity_matrix.npy",
        out_dir / f"{trial_id}_distance_matrix.npy",
        out_dir / f"{trial_id}_similarity_edges.csv",
        out_dir / f"{trial_id}_population_similarity_summary.json",
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


def numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    skip = {"trial_id", "roi_id", "suite2p_original_id", "response_type"}
    cols = [c for c in df.columns if c not in skip]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


def safe_corr(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    if matrix.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    std = np.nanstd(matrix, axis=1)
    keep = std > 0
    corr = np.eye(matrix.shape[0], dtype=np.float32)
    if keep.sum() >= 2:
        corr_sub = np.corrcoef(matrix[keep])
        corr[np.ix_(keep, keep)] = np.nan_to_num(corr_sub, nan=0.0).astype(np.float32)
    return corr


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = matrix / norm
    return np.clip(unit @ unit.T, -1.0, 1.0).astype(np.float32)


def edges_from_similarity(feature_df: pd.DataFrame, similarity: np.ndarray, min_corr: float, knn: int) -> pd.DataFrame:
    if similarity.size == 0:
        return pd.DataFrame(columns=["source_roi_id", "target_roi_id", "similarity"])
    roi_ids = feature_df["roi_id"].astype(int).to_numpy() if "roi_id" in feature_df else np.arange(similarity.shape[0]) + 1
    rows = []
    for i in range(similarity.shape[0]):
        order = np.argsort(similarity[i])[::-1]
        kept = 0
        for j in order:
            if i == j or j < i:
                continue
            value = float(similarity[i, j])
            if value < min_corr:
                continue
            rows.append({"source_roi_id": int(roi_ids[i]), "target_roi_id": int(roi_ids[j]), "similarity": value})
            kept += 1
            if knn > 0 and kept >= knn:
                break
    return pd.DataFrame(rows)


def save_heatmap(matrix: np.ndarray, out_path: Path, trial_id: str, dpi: int) -> None:
    if matrix.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"ROI similarity - {trial_id}")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, label="similarity")
    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)


def process_trial(trial: TrialInput, out_dir: Path, args: argparse.Namespace) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df = pd.read_csv(trial.feature_matrix_path)
    z_df = pd.read_csv(trial.z_feature_matrix_path) if trial.z_feature_matrix_path.exists() else feature_df
    feature_matrix = numeric_matrix(z_df)
    feature_similarity = cosine_similarity(feature_matrix)
    distance = (1.0 - feature_similarity).astype(np.float32)
    np.save(out_dir / f"{trial.trial_id}_feature_similarity_matrix.npy", feature_similarity)
    np.save(out_dir / f"{trial.trial_id}_distance_matrix.npy", distance)

    trace_corr = np.empty((0, 0), dtype=np.float32)
    if trial.trace_matrix_path and trial.trace_matrix_path.exists():
        trace_corr = safe_corr(np.load(trial.trace_matrix_path, allow_pickle=True))
    response_corr = np.empty((0, 0), dtype=np.float32)
    if trial.response_matrix_path and trial.response_matrix_path.exists():
        response = np.load(trial.response_matrix_path, allow_pickle=True)
        if response.ndim == 2 and response.shape[1] > 0:
            response_corr = safe_corr(response)
    np.save(out_dir / f"{trial.trial_id}_trace_correlation_matrix.npy", trace_corr)
    np.save(out_dir / f"{trial.trial_id}_response_correlation_matrix.npy", response_corr)

    source = args.similarity_source
    selected = {"traces": trace_corr, "responses": response_corr, "features": feature_similarity}.get(source, feature_similarity)
    if selected.size == 0:
        selected = feature_similarity
        source = "features"
    edges = edges_from_similarity(feature_df, selected, min_corr=args.min_corr, knn=args.knn)
    edges.to_csv(out_dir / f"{trial.trial_id}_similarity_edges.csv", index=False)
    save_heatmap(selected, out_dir / f"{trial.trial_id}_similarity_heatmap.png", trial.trial_id, args.dpi)

    summary = {
        "trial_id": trial.trial_id,
        "status": "ok",
        "n_roi": int(len(feature_df)),
        "n_features": int(feature_matrix.shape[1]),
        "similarity_source": source,
        "n_edges": int(len(edges)),
        "min_corr": float(args.min_corr),
        "knn": int(args.knn),
    }
    (out_dir / f"{trial.trial_id}_population_similarity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return "processed", summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ROI-ROI similarity matrices.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Step-10 root. Default: OUTPUT_ROOT/10_population_features.")
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--similarity-source", choices=("traces", "responses", "features"), default="features")
    parser.add_argument("--min-corr", type=float, default=0.3)
    parser.add_argument("--knn", type=int, default=10)
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
            LOGGER.info("[dry-run] Would compute similarity for %s -> %s", trial.trial_id, out_dir)
            continue
        if args.action == "overwrite":
            clean_step_outputs(out_dir)
        try:
            _, row = process_trial(trial, out_dir, args)
            rows.append(row)
            summary.processed += 1
            LOGGER.info("[ok] %s: n_edges=%s", trial.trial_id, row.get("n_edges"))
        except Exception as exc:
            summary.failed += 1
            rows.append({"trial_id": trial.trial_id, "status": "failed", "message": str(exc)})
            LOGGER.exception("[failed] %s: %s", trial.trial_id, exc)
    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "population_similarity_summary.csv", index=False)
    LOGGER.info("Summary:")
    LOGGER.info("  found: %d", summary.found)
    LOGGER.info("  processed or would process: %d", summary.processed)
    LOGGER.info("  skipped: %d", summary.skipped)
    LOGGER.info("  failed: %d", summary.failed)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
