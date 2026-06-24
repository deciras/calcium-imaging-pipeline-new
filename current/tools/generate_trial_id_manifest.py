#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a proposed canonical trial-id manifest without renaming files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from trial_context_utils import build_trial_context_table  # noqa: E402


LOGGER = logging.getLogger("trial_id_manifest")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def discover_trial_ids(output_root: Path) -> list[str]:
    trial_ids: set[str] = set()
    dff_root = output_root / "06_dff"
    if dff_root.exists():
        for path in dff_root.rglob("*_dff_summary.json"):
            trial_ids.add(path.parent.name)
    summary_root = output_root / "17_cross_trial_summary"
    qc = read_csv(summary_root / "cross_trial_qc_summary.csv")
    if not qc.empty and "trial_id" in qc.columns:
        trial_ids.update(qc["trial_id"].dropna().astype(str).tolist())
    return sorted(trial_ids)


def collect_tables(output_root: Path) -> dict[str, pd.DataFrame]:
    summary_root = output_root / "17_cross_trial_summary"
    qc = read_csv(summary_root / "cross_trial_qc_summary.csv")
    if qc.empty:
        qc = pd.DataFrame({"trial_id": discover_trial_ids(output_root)})
    return {
        "qc": qc,
        "stim_response_table": read_csv(summary_root / "all_trials_stim_response_table.csv"),
        "angle_response_table": read_csv(summary_root / "all_trials_angle_response_table.csv"),
    }


def build_manifest(output_root: Path) -> pd.DataFrame:
    tables = collect_tables(output_root)
    qc = tables.pop("qc")
    context = build_trial_context_table(output_root, qc, tables)
    if context.empty:
        return pd.DataFrame()
    manifest = context.rename(columns={"trial_id": "raw_trial_id"}).copy()
    manifest["canonical_group_id"] = manifest["canonical_trial_id_proposed"].astype(str).str.replace(r"_rep\d{4}$", "", regex=True)
    manifest["rename_ready"] = manifest["experiment_note_available"].eq("yes").map({True: "yes", False: "review"})
    manifest["raw_trial_id_changed"] = manifest["raw_trial_id"].astype(str) != manifest["canonical_trial_id_proposed"].astype(str)
    cols = [
        "raw_trial_id",
        "canonical_trial_id_proposed",
        "canonical_group_id",
        "rename_ready",
        "raw_trial_id_changed",
        "note_date",
        "planned_stimulus_mode",
        "measured_stimulus_mode",
        "measured_has_stimulus",
        "measured_has_AoLP",
        "chloride_condition",
        "reduced_chloride",
        "analysis_branch_key",
        "note_file",
        "note_text",
    ]
    keep = [col for col in cols if col in manifest.columns]
    return manifest[keep].sort_values(["note_date", "raw_trial_id"], kind="stable").reset_index(drop=True)


def split_manifest_scope(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest.empty:
        return manifest.copy(), manifest.copy()
    raw_id = manifest.get("raw_trial_id", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    canonical = manifest.get("canonical_trial_id_proposed", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    exclude_mask = raw_id.str.contains("wb_cells") | canonical.str.contains("_cells_")
    included = manifest.loc[~exclude_mask].copy().reset_index(drop=True)
    excluded = manifest.loc[exclude_mask].copy().reset_index(drop=True)
    if not excluded.empty:
        excluded["exclude_reason"] = "excluded_nonretina_series"
    return included, excluded


def build_summary(manifest: pd.DataFrame, excluded: pd.DataFrame, out_dir: Path) -> dict:
    collisions = (
        manifest.groupby("canonical_trial_id_proposed")["raw_trial_id"].nunique().reset_index(name="n_raw")
        if not manifest.empty
        else pd.DataFrame(columns=["canonical_trial_id_proposed", "n_raw"])
    )
    collision_rows = collisions[collisions["n_raw"] > 1].copy()
    collision_rows.to_csv(out_dir / "trial_id_manifest_collisions.csv", index=False)
    excluded.to_csv(out_dir / "trial_id_manifest_excluded.csv", index=False)
    summary = {
        "status": "ok",
        "n_trials": int(len(manifest)),
        "n_excluded_trials": int(len(excluded)),
        "n_note_backed_trials": int((manifest.get("rename_ready", pd.Series(dtype=str)) == "yes").sum()),
        "n_changed_names": int(pd.to_numeric(manifest.get("raw_trial_id_changed", pd.Series(dtype=bool)), errors="coerce").fillna(False).sum()),
        "n_collisions": int(len(collision_rows)),
        "manifest_csv": str(out_dir / "trial_id_manifest.csv"),
        "collision_csv": str(out_dir / "trial_id_manifest_collisions.csv"),
        "excluded_csv": str(out_dir / "trial_id_manifest_excluded.csv"),
    }
    (out_dir / "trial_id_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a proposed canonical trial-id manifest.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="Pipeline output root. Default: DATA_ROOT.")
    parser.add_argument("--manifest-dir", type=Path, help="Manifest output directory. Default: OUTPUT_ROOT/00_trial_metadata.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    output_root = args.output_root.expanduser().resolve() if args.output_root else args.data_root.expanduser().resolve()
    out_dir = args.manifest_dir.expanduser().resolve() if args.manifest_dir else output_root / "00_trial_metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(output_root)
    if manifest.empty:
        LOGGER.error("No trials were found for manifest generation.")
        return 1
    manifest, excluded = split_manifest_scope(manifest)
    manifest.to_csv(out_dir / "trial_id_manifest.csv", index=False)
    summary = build_summary(manifest, excluded, out_dir)
    LOGGER.info("Wrote manifest: %s", out_dir / "trial_id_manifest.csv")
    LOGGER.info(
        "Trials: %d | excluded: %d | changed: %d | collisions: %d",
        summary["n_trials"],
        summary["n_excluded_trials"],
        summary["n_changed_names"],
        summary["n_collisions"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
