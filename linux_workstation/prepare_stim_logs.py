#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect stimulus controller logs into DATA_ROOT/stim_logs.

Step 02 looks for timestamp_log_*.csv files in one stimulus log folder. The raw
workstation export may instead contain many date-specific *_motor_rotation
folders. This helper creates a flat stim_logs folder using symlinks by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


LOG_PATTERNS = (
    "timestamp_log_*.csv",
    "stim_map_*.csv",
    "experiment_config_*.json",
    "mcu_config_*.json",
    "*_angle_list.txt",
)


@dataclass(frozen=True)
class LinkPlan:
    source: Path
    destination: Path
    action: str


@dataclass(frozen=True)
class SynthConfigPlan:
    source: Path
    destination: Path
    run_id: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a flat stim_logs folder from *_motor_rotation folders."
    )
    parser.add_argument("data_root", type=Path, help="Experiment data root.")
    parser.add_argument("--target-name", default="stim_logs", help="Output folder name under data root.")
    parser.add_argument("--source-glob", default="*_motor_rotation", help="Direct child folders to collect from.")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks on Linux by default; use copy if the filesystem does not preserve links.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in stim_logs.")
    parser.add_argument("--execute", action="store_true", help="Actually create links/files. Default is dry-run.")
    return parser.parse_args(argv)


def parse_run_id(path: Path, prefix: str, suffix: str) -> str | None:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def collect_source_files(data_root: Path, source_glob: str) -> list[Path]:
    files: list[Path] = []
    for folder in sorted(data_root.glob(source_glob)):
        if not folder.is_dir():
            continue
        for pattern in LOG_PATTERNS:
            files.extend(sorted(path for path in folder.glob(pattern) if path.is_file()))
    return sorted(set(files))


def build_link_plans(files: list[Path], target_root: Path, overwrite: bool) -> list[LinkPlan]:
    plans: list[LinkPlan] = []
    for source in files:
        destination = target_root / source.name
        if destination.exists() or destination.is_symlink():
            if overwrite:
                action = "replace"
            else:
                action = "skip-existing"
        else:
            action = "create"
        plans.append(LinkPlan(source=source, destination=destination, action=action))
    return plans


def build_synth_config_plans(files: list[Path], target_root: Path, overwrite: bool) -> list[SynthConfigPlan]:
    existing_experiment_ids: set[str] = set()
    mcu_files: list[Path] = []

    for source in files:
        exp_id = parse_run_id(source, "experiment_config_", ".json")
        if exp_id:
            existing_experiment_ids.add(exp_id)
        mcu_id = parse_run_id(source, "mcu_config_", ".json")
        if mcu_id:
            mcu_files.append(source)

    plans: list[SynthConfigPlan] = []
    for source in mcu_files:
        run_id = parse_run_id(source, "mcu_config_", ".json")
        if not run_id or run_id in existing_experiment_ids:
            continue
        destination = target_root / f"experiment_config_{run_id}.json"
        if destination.exists() and not overwrite:
            continue
        plans.append(SynthConfigPlan(source=source, destination=destination, run_id=run_id))
    return plans


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def apply_link_plan(plan: LinkPlan, mode: str) -> None:
    if plan.action == "skip-existing":
        return
    if plan.action == "replace" and (plan.destination.exists() or plan.destination.is_symlink()):
        remove_existing(plan.destination)

    if mode == "symlink":
        plan.destination.symlink_to(plan.source.resolve())
    else:
        shutil.copy2(plan.source, plan.destination)


def apply_synth_config(plan: SynthConfigPlan) -> None:
    if plan.destination.exists() or plan.destination.is_symlink():
        remove_existing(plan.destination)
    with plan.source.open("r", encoding="utf-8") as handle:
        mcu_config = json.load(handle)
    payload = {
        "run_id": plan.run_id,
        "mcu_config": mcu_config,
        "source_mcu_config": str(plan.source),
        "generated_by": "linux_workstation/prepare_stim_logs.py",
    }
    with plan.destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_manifest(target_root: Path, link_plans: list[LinkPlan], synth_plans: list[SynthConfigPlan]) -> None:
    manifest_path = target_root / "stim_logs_manifest.csv"
    rows: list[dict[str, str]] = []
    for plan in link_plans:
        rows.append(
            {
                "destination": plan.destination.name,
                "source": str(plan.source),
                "kind": "source_file",
                "action": plan.action,
            }
        )
    for plan in synth_plans:
        rows.append(
            {
                "destination": plan.destination.name,
                "source": str(plan.source),
                "kind": "synthesized_experiment_config",
                "action": "create",
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("destination", "source", "kind", "action"))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists() or not data_root.is_dir():
        print(f"ERROR: data root does not exist or is not a folder: {data_root}", file=sys.stderr)
        return 2

    target_root = data_root / args.target_name
    files = collect_source_files(data_root, args.source_glob)
    link_plans = build_link_plans(files, target_root, overwrite=args.overwrite)
    synth_plans = build_synth_config_plans(files, target_root, overwrite=args.overwrite)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode      : {mode}")
    print(f"Data root : {data_root}")
    print(f"Target    : {target_root}")
    print(f"Sources   : {args.source_glob}")
    print(f"Files     : {len(files)}")
    print()

    if not files:
        print("No stimulus log files found.")
        return 0

    for plan in link_plans:
        verb = {"create": "link/copy", "replace": "replace", "skip-existing": "skip"}[plan.action]
        print(f"{verb:11s} {plan.destination.name} <- {plan.source.parent.name}/{plan.source.name}")
    for plan in synth_plans:
        print(f"synthesize  {plan.destination.name} <- {plan.source.parent.name}/{plan.source.name}")

    if not args.execute:
        print()
        print("Dry-run only. Re-run with --execute to create stim_logs.")
        return 0

    target_root.mkdir(parents=True, exist_ok=True)
    for plan in link_plans:
        apply_link_plan(plan, args.mode)
    for plan in synth_plans:
        apply_synth_config(plan)
    write_manifest(target_root, link_plans, synth_plans)
    print()
    print(f"Prepared stimulus logs: {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
