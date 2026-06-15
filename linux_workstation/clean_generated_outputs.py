#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove pipeline-generated output folders from a Linux workstation data root.

The cleaner is intentionally conservative:
- dry-run by default
- never removes raw-looking date/trial folders
- never removes 00_original_files unless explicitly requested
- only removes known pipeline output folder names
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PREMANUAL_DIRS = (
    "01_oir_to_tif",
    "02_stim_map",
    "03_motion_correct",
    "04_spatial_highpass",
    "05_suite2p_roi_detection",
    "05_suite2p_param_tests",
)

LEGACY_PREMANUAL_GLOBS = (
    "Processed_TIF_*",
)

MANUAL_DIRS = (
    "05e_roi_manual_curation",
)

POSTMANUAL_DIRS = (
    "06_dff",
    "07_events",
    "08_stim_response",
    "09_angle_tuning",
    "10_population_features",
    "11_population_similarity",
    "12_hierarchical_clustering",
    "13_leiden",
    "14_dimensionality_reduction",
    "15_cross_trial_summary",
    "16_reports",
)

LOG_DIRS = (
    "pipeline_logs",
    "pipeline_outputs",
)

RAW_DIRS = (
    "00_original_files",
)


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean generated calcium-imaging pipeline outputs while preserving raw data."
        )
    )
    parser.add_argument("data_root", type=Path, help="Experiment data root on the workstation.")
    parser.add_argument(
        "--scope",
        choices=("premanual", "all-generated"),
        default="premanual",
        help=(
            "premanual removes 01-05 outputs for a fresh overnight run. "
            "all-generated also removes manual, 06-16, and pipeline log folders."
        ),
    )
    parser.add_argument(
        "--include-organized-raw",
        action="store_true",
        help=(
            "Also remove 00_original_files. Use only when raw files still exist elsewhere; "
            "this may delete the only organized copy of the raw OIR files."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints a dry-run plan.",
    )
    return parser.parse_args(argv)


def scope_names(scope: str, include_organized_raw: bool) -> dict[str, str]:
    names: dict[str, str] = {name: "premanual output" for name in PREMANUAL_DIRS}
    if scope == "all-generated":
        names.update({name: "manual GUI output" for name in MANUAL_DIRS})
        names.update({name: "postmanual analysis output" for name in POSTMANUAL_DIRS})
        names.update({name: "pipeline log/output folder" for name in LOG_DIRS})
    if include_organized_raw:
        names.update({name: "organized raw folder" for name in RAW_DIRS})
    return names


def collect_candidates(data_root: Path, names: dict[str, str], legacy_globs: tuple[str, ...]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for name, reason in names.items():
        path = data_root / name
        if path.exists():
            candidates.append(Candidate(path=path, reason=reason))
    for pattern in legacy_globs:
        for path in sorted(data_root.glob(pattern)):
            if path.exists():
                candidates.append(Candidate(path=path, reason="legacy premanual output"))
    return candidates


def refuse_unsafe_root(data_root: Path) -> None:
    resolved = data_root.resolve()
    unsafe = {Path("/").resolve(), Path.home().resolve()}
    if resolved in unsafe:
        raise ValueError(f"Refusing to clean unsafe data root: {resolved}")
    if not resolved.exists():
        raise ValueError(f"Data root does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Data root is not a directory: {resolved}")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()

    try:
        refuse_unsafe_root(data_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    names = scope_names(args.scope, args.include_organized_raw)
    legacy_globs = LEGACY_PREMANUAL_GLOBS if args.scope in {"premanual", "all-generated"} else ()
    candidates = collect_candidates(data_root, names, legacy_globs)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode      : {mode}")
    print(f"Data root : {data_root}")
    print(f"Scope     : {args.scope}")
    print()

    if not candidates:
        print("No matching generated output folders found.")
        return 0

    print("Will remove:" if args.execute else "Would remove:")
    for candidate in candidates:
        print(f"  - {candidate.path}  ({candidate.reason})")

    if not args.execute:
        print()
        print("Dry-run only. Re-run with --execute to delete these folders.")
        return 0

    print()
    for candidate in candidates:
        remove_path(candidate.path)
        print(f"Removed: {candidate.path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
