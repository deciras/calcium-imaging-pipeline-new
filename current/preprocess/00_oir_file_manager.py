#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize Olympus .oir files into per-trial folders.

This step is intentionally conservative:
- no hardcoded data path
- no recursive file moves
- default conflict behavior is "skip"
- dry-run mode prints the plan without changing files
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


LOGGER = logging.getLogger("oir_file_manager")

METADATA_HEADERS = [
    "Trial_ID",
    "Time_Per_Frame",
    "Num_Flashes",
    "Stim_Start_s",
    "Stim_Duration_s",
    "Stim_Interval_s",
    "Polarization_Angles",
]

DEFAULT_ORIGINAL_ROOT_NAME = "00_original_files"


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    trial_id: str


@dataclass
class FolderSummary:
    folder: Path
    planned_moves: int = 0
    moved: int = 0
    skipped: int = 0
    failed: int = 0
    metadata_rows_added: int = 0
    trial_ids: set[str] = field(default_factory=set)


@dataclass
class RunSummary:
    folders_seen: int = 0
    planned_moves: int = 0
    moved: int = 0
    skipped: int = 0
    failed: int = 0
    metadata_rows_added: int = 0


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_trial_id(filename: str) -> str:
    """Remove the extension and spaces so folder names are stable."""
    return Path(filename).stem.replace(" ", "")


def file_belongs_to_trial(filename: str, trial_id: str) -> bool:
    """Match the .oir file itself and its chunk/sidecar files."""
    clean_name = filename.replace(" ", "")
    return clean_name == f"{trial_id}.oir" or clean_name.startswith(f"{trial_id}_")


def list_direct_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file())


def folder_has_direct_oir(folder: Path) -> bool:
    return any(path.is_file() and path.suffix.lower() == ".oir" for path in folder.iterdir())


def folder_has_own_oir(folder: Path) -> bool:
    """Return True when a trial folder contains its matching .oir file."""
    folder_id = folder.name.replace(" ", "")
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != ".oir":
            continue
        if normalize_trial_id(path.name) == folder_id:
            return True
    return False


def existing_trial_dirs(folder: Path) -> list[Path]:
    """Return direct child folders that already look like organized trials."""
    trial_dirs: list[Path] = []
    for child in sorted(path for path in folder.iterdir() if path.is_dir()):
        try:
            if folder_has_own_oir(child):
                trial_dirs.append(child)
        except OSError as exc:
            LOGGER.warning("Could not inspect folder %s: %s", child, exc)
    return trial_dirs


def is_pipeline_step_dir(path: Path) -> bool:
    return bool(re.match(r"^\d{2}_", path.name)) and path.name != DEFAULT_ORIGINAL_ROOT_NAME


def should_move_to_original_root(path: Path) -> bool:
    if path.name in {DEFAULT_ORIGINAL_ROOT_NAME, "pipeline_outputs"}:
        return False
    if is_pipeline_step_dir(path):
        return False
    if path.name == "tree.txt":
        return False
    if path.is_dir():
        try:
            return folder_has_own_oir(path) or bool(existing_trial_dirs(path))
        except OSError:
            return False
    if path.is_file():
        return path.suffix.lower() == ".oir" or path.name == "metadata.csv"
    return False


def prepare_original_root(data_root: Path, original_root_name: str, action: str, dry_run: bool) -> Path:
    """
    Move raw trial folders into DATA_ROOT/00_original_files.

    This keeps raw files beside processing step folders without mixing them
    with generated outputs.
    """
    if data_root.name == original_root_name:
        return data_root

    original_root = data_root / original_root_name
    candidates = [path for path in sorted(data_root.iterdir()) if should_move_to_original_root(path)]

    if not candidates:
        if not original_root.exists() and not dry_run:
            original_root.mkdir(parents=True, exist_ok=True)
        return original_root

    if dry_run:
        LOGGER.info("[dry-run] Would ensure raw data root exists: %s", original_root)
    else:
        original_root.mkdir(parents=True, exist_ok=True)

    for source in candidates:
        destination = original_root / source.name
        if destination.exists():
            if action == "skip":
                LOGGER.info("Skipping existing raw destination path: %s", destination)
                continue
            LOGGER.error("Refusing to overwrite existing raw item: %s", destination)
            continue

        if dry_run:
            LOGGER.info("[dry-run] Would move raw item: %s -> %s", source, destination)
        else:
            shutil.move(str(source), str(destination))
            LOGGER.info("Moved raw item: %s -> %s", source, destination)

    if dry_run and not original_root.exists():
        return data_root
    return original_root


def discover_work_folders(data_root: Path, layout: str) -> list[Path]:
    """
    Find folders to organize.

    direct:
        Treat data_root itself as the folder containing .oir files.
    children:
        Treat direct child folders as date/session folders.
    auto:
        If data_root directly contains .oir files, process data_root.
        Otherwise process direct child folders.
    """
    if layout == "direct":
        return [data_root]

    if layout == "children":
        return sorted(path for path in data_root.iterdir() if path.is_dir())

    if folder_has_direct_oir(data_root):
        return [data_root]

    if existing_trial_dirs(data_root):
        return [data_root]

    return sorted(path for path in data_root.iterdir() if path.is_dir())


def build_move_plan(folder: Path) -> tuple[list[MovePlan], set[str]]:
    """Plan moves for direct .oir files and their sidecar/chunk files."""
    files = list_direct_files(folder)
    filenames = [path.name for path in files]
    oir_files = [path for path in files if path.suffix.lower() == ".oir"]

    trial_ids = {normalize_trial_id(path.name) for path in oir_files}
    for trial_dir in existing_trial_dirs(folder):
        trial_ids.add(trial_dir.name.replace(" ", ""))

    plans: list[MovePlan] = []
    processed_names: set[str] = set()

    tid_file_pairs = sorted(
        ((normalize_trial_id(path.name), path.name) for path in oir_files),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for trial_id, _oir_name in tid_file_pairs:
        if folder.name.replace(" ", "") == trial_id:
            continue

        target_folder = folder / trial_id

        for filename in filenames:
            if filename in processed_names:
                continue
            if not file_belongs_to_trial(filename, trial_id):
                continue

            source = folder / filename
            destination = target_folder / filename.replace(" ", "")
            if source.resolve() == destination.resolve():
                processed_names.add(filename)
                continue

            plans.append(MovePlan(source=source, destination=destination, trial_id=trial_id))
            processed_names.add(filename)

    return plans, trial_ids


def read_existing_metadata_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    try:
        with csv_path.open(mode="r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return {row["Trial_ID"] for row in reader if row.get("Trial_ID")}
    except (OSError, csv.Error, KeyError) as exc:
        raise RuntimeError(f"Could not read existing metadata CSV: {csv_path}: {exc}") from exc


def update_metadata_csv(
    folder: Path,
    trial_ids: set[str],
    dry_run: bool,
) -> int:
    """Create or append metadata.csv rows for newly discovered trial IDs."""
    csv_path = folder / "metadata.csv"
    existing_ids = read_existing_metadata_ids(csv_path)
    new_ids = [trial_id for trial_id in sorted(trial_ids) if trial_id not in existing_ids]

    if not new_ids:
        LOGGER.info("metadata.csv is already up to date: %s", csv_path)
        return 0

    if dry_run:
        LOGGER.info(
            "[dry-run] Would add %d row(s) to %s: %s",
            len(new_ids),
            csv_path,
            ", ".join(new_ids),
        )
        return len(new_ids)

    csv_exists = csv_path.exists()
    with csv_path.open(mode="a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS)
        if not csv_exists:
            writer.writeheader()
        for trial_id in new_ids:
            writer.writerow({"Trial_ID": trial_id})

    LOGGER.info("Added %d metadata row(s): %s", len(new_ids), csv_path)
    return len(new_ids)


def apply_move_plan(
    plans: list[MovePlan],
    action: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Run planned moves. Returns moved, skipped, failed."""
    moved = 0
    skipped = 0
    failed = 0

    for plan in plans:
        if plan.destination.exists() and action == "skip":
            LOGGER.info("Skipping existing destination path: %s", plan.destination)
            skipped += 1
            continue

        if dry_run:
            if plan.destination.exists() and action == "overwrite":
                LOGGER.info("[dry-run] Would replace file: %s", plan.destination)
            LOGGER.info("[dry-run] Would move: %s -> %s", plan.source, plan.destination)
            moved += 1
            continue

        try:
            plan.destination.parent.mkdir(parents=True, exist_ok=True)

            if plan.destination.exists():
                if action != "overwrite":
                    LOGGER.info("Skipping existing destination path: %s", plan.destination)
                    skipped += 1
                    continue
                if plan.destination.is_dir():
                    LOGGER.error("Refusing to overwrite directory: %s", plan.destination)
                    failed += 1
                    continue
                plan.destination.unlink()

            shutil.move(str(plan.source), str(plan.destination))
            LOGGER.info("Moved: %s -> %s", plan.source, plan.destination)
            moved += 1
        except OSError as exc:
            LOGGER.error("Move failed %s -> %s: %s", plan.source, plan.destination, exc)
            failed += 1

    return moved, skipped, failed


def organize_folder(folder: Path, action: str, dry_run: bool) -> FolderSummary:
    folder = folder.resolve()
    summary = FolderSummary(folder=folder)

    if not folder.exists():
        LOGGER.error("Folder does not exist: %s", folder)
        summary.failed += 1
        return summary
    if not folder.is_dir():
        LOGGER.error("Path is not a folder: %s", folder)
        summary.failed += 1
        return summary

    LOGGER.info("Inspecting folder: %s", folder)
    plans, trial_ids = build_move_plan(folder)
    summary.trial_ids = trial_ids
    summary.planned_moves = len(plans)

    if not plans and not trial_ids:
        LOGGER.info("No direct .oir files or organized trial folders found: %s", folder)
        return summary

    LOGGER.info(
        "Found %d trial(s), planned %d move(s).",
        len(trial_ids),
        len(plans),
    )

    moved, skipped, failed = apply_move_plan(plans, action=action, dry_run=dry_run)
    summary.moved = moved
    summary.skipped = skipped
    summary.failed = failed

    try:
        summary.metadata_rows_added = update_metadata_csv(
            folder=folder,
            trial_ids=trial_ids,
            dry_run=dry_run,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        summary.failed += 1

    return summary


def env_default_data_root() -> Path | None:
    raw = os.environ.get("CALCIUM_DATA_ROOT")
    if not raw:
        return None
    return Path(raw).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Organize Olympus .oir files into trial folders and update metadata.csv."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=env_default_data_root(),
        help=(
            "Root data folder. If omitted, CALCIUM_DATA_ROOT is used. "
            "This argument is intentionally explicit to avoid touching the wrong folder."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "direct", "children"),
        default="auto",
        help=(
            "auto: infer structure; direct: data-root itself contains .oir files; "
            "children: direct child folders contain .oir files."
        ),
    )
    parser.add_argument(
        "--action",
        choices=("skip", "overwrite"),
        default="skip",
        help="What to do if a destination file already exists. Default: skip.",
    )
    parser.add_argument(
        "--original-root-name",
        default=DEFAULT_ORIGINAL_ROOT_NAME,
        help="Folder name used for raw organized files. Default: 00_original_files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves and metadata updates without changing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra details.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.data_root is None:
        LOGGER.error(
            "No data root was provided. Use --data-root /path/to/data, "
            "or set CALCIUM_DATA_ROOT."
        )
        return 2

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        LOGGER.error("Data root does not exist: %s", data_root)
        return 1
    if not data_root.is_dir():
        LOGGER.error("Data root is not a folder: %s", data_root)
        return 1

    original_root = prepare_original_root(
        data_root=data_root,
        original_root_name=args.original_root_name,
        action=args.action,
        dry_run=args.dry_run,
    )

    LOGGER.info("Data root: %s", data_root)
    LOGGER.info("Raw data root: %s", original_root)
    LOGGER.info("Layout mode: %s", args.layout)
    LOGGER.info("Conflict policy: %s", args.action)
    LOGGER.info("Run mode: %s", "dry-run" if args.dry_run else "apply changes")

    try:
        work_folders = discover_work_folders(original_root, args.layout)
    except OSError as exc:
        LOGGER.error("Could not inspect data root %s: %s", data_root, exc)
        return 1

    if not work_folders:
        LOGGER.warning("No processable folders were found under: %s", data_root)
        return 0

    LOGGER.info("Selected %d folder(s) for processing.", len(work_folders))

    run_summary = RunSummary(folders_seen=len(work_folders))
    for folder in work_folders:
        folder_summary = organize_folder(
            folder=folder,
            action=args.action,
            dry_run=args.dry_run,
        )
        run_summary.planned_moves += folder_summary.planned_moves
        run_summary.moved += folder_summary.moved
        run_summary.skipped += folder_summary.skipped
        run_summary.failed += folder_summary.failed
        run_summary.metadata_rows_added += folder_summary.metadata_rows_added

    LOGGER.info("Summary:")
    LOGGER.info("  folders seen: %d", run_summary.folders_seen)
    LOGGER.info("  planned moves: %d", run_summary.planned_moves)
    LOGGER.info("  moved or would move: %d", run_summary.moved)
    LOGGER.info("  skipped: %d", run_summary.skipped)
    LOGGER.info("  failed: %d", run_summary.failed)
    LOGGER.info("  metadata rows added or planned: %d", run_summary.metadata_rows_added)

    return 1 if run_summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
