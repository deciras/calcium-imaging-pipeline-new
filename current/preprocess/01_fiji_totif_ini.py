#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch Fiji/ImageJ headlessly to convert Olympus .oir files to TIFF.

The actual Bio-Formats work happens in 01_fiji_totif_worker.py, which runs
inside Fiji's Jython environment. This launcher stays normal Python so it can
handle paths, safety checks, and dry-run behavior.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("fiji_totif_launcher")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def default_worker_script() -> Path:
    return Path(__file__).resolve().with_name("01_fiji_totif_worker_wrapper.py")


def default_output_root(data_root: Path) -> Path:
    return data_root


def default_original_root(data_root: Path) -> Path:
    candidate = data_root / "00_original_files"
    return candidate if candidate.exists() else data_root


def candidate_fiji_bins() -> list[Path]:
    candidates: list[Path] = []
    home = Path.home()

    env_value = os.environ.get("FIJI_BIN") or os.environ.get("FIJI_PATH")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            Path("/Applications/Fiji.app"),
            Path("/Applications/Fiji.app/Fiji.app"),
            Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
            Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx-arm64"),
            Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx-x64"),
            Path("/Applications/ImageJ.app/Contents/MacOS/ImageJ-macosx"),
            home / "Fiji.app",
            home / "Fiji",
            home / "Applications" / "Fiji.app",
            Path("/opt/Fiji.app"),
            Path("/opt/Fiji"),
            Path("/usr/local/Fiji.app"),
            Path("/usr/local/Fiji"),
        ]
    )
    return candidates


def resolve_fiji_executable(path: Path) -> Path | None:
    path = path.expanduser()
    if not path.exists():
        return None
    if path.is_file():
        return path if os.access(path, os.X_OK) else None

    bundled_candidates = [
        path / "Contents" / "MacOS" / "ImageJ-macosx",
        path / "Contents" / "MacOS" / "ImageJ-macosx-arm64",
        path / "Contents" / "MacOS" / "ImageJ-macosx-x64",
        path / "Fiji.app" / "Contents" / "MacOS" / "fiji-macos-arm64",
        path / "Fiji.app" / "Contents" / "MacOS" / "fiji-macos",
        path / "Fiji.app" / "Contents" / "MacOS" / "fiji-macos-x64",
        path / "fiji",
        path / "ImageJ-linux64",
        path / "ImageJ-linux32",
        path / "fiji-linux-x64",
        path / "fiji-linux64",
    ]
    for candidate in bundled_candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_fiji_bin(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return resolve_fiji_executable(explicit_path)

    for candidate in candidate_fiji_bins():
        resolved = resolve_fiji_executable(candidate)
        if resolved is not None:
            return resolved
    return None


def quote_for_fiji(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def prefers_jaunch_cli(fiji_bin: Path) -> bool:
    name = fiji_bin.name.lower()
    return name == "fiji" or name.startswith("fiji-")


def prefers_direct_script_cli(fiji_bin: Path) -> bool:
    name = fiji_bin.name.lower()
    return name.startswith("imagej-")


def build_fiji_args(
    data_root: Path,
    output_root: Path,
    extension: str,
    threads: int,
    action: str,
    projection_mode: str,
    metadata_mode: str,
    stim_export_mode: str,
) -> str:
    return (
        'rootDir="{}",outputRoot="{}",ext="{}",threads={},'
        'existingMode="{}",projectionMode="{}",metadataMode="{}",stimExportMode="{}"'
    ).format(
        quote_for_fiji(data_root),
        quote_for_fiji(output_root),
        quote_for_fiji(extension),
        int(threads),
        quote_for_fiji(action),
        quote_for_fiji(projection_mode),
        quote_for_fiji(metadata_mode),
        quote_for_fiji(stim_export_mode),
    )


def build_fiji_command(
    fiji_bin: Path,
    worker_script: Path,
    worker_args: str,
    fiji_memory: str | None,
) -> list[str]:
    command = [str(fiji_bin), "--headless"]
    if fiji_memory:
        command.append(f"--mem={fiji_memory}")
    if prefers_direct_script_cli(fiji_bin):
        command.extend([str(worker_script), worker_args])
    elif prefers_jaunch_cli(fiji_bin):
        # Linux Fiji launcher "fiji" exits cleanly if given a script path
        # directly, but may never execute the worker. Use explicit --run.
        command.extend(["--run", str(worker_script), worker_args])
    else:
        command.extend(["--run", str(worker_script), worker_args])
    return command


def run_fiji_task(
    fiji_bin: Path,
    worker_script: Path,
    data_root: Path,
    output_root: Path,
    extension: str,
    threads: int,
    action: str,
    fiji_memory: str | None,
    projection_mode: str,
    metadata_mode: str,
    stim_export_mode: str,
) -> bool:
    args = build_fiji_args(
        data_root=data_root,
        output_root=output_root,
        extension=extension,
        threads=threads,
        action=action,
        projection_mode=projection_mode,
        metadata_mode=metadata_mode,
        stim_export_mode=stim_export_mode,
    )
    command = build_fiji_command(
        fiji_bin=fiji_bin,
        worker_script=worker_script,
        worker_args=args,
        fiji_memory=fiji_memory,
    )

    LOGGER.info("Launching Fiji")
    LOGGER.info("  Fiji executable: %s", fiji_bin)
    LOGGER.info("  Worker script: %s", worker_script)
    LOGGER.info("  Raw data root: %s", data_root)
    LOGGER.info("  Output root: %s", output_root)
    LOGGER.info("  Extension: %s", extension)
    LOGGER.info("  Conflict policy: %s", action)
    LOGGER.info("  Fiji memory: %s", fiji_memory or "Fiji default")
    LOGGER.info("  Projection mode: %s", projection_mode)
    LOGGER.info("  Metadata mode: %s", metadata_mode)
    LOGGER.info("  Stim export mode: %s", stim_export_mode)
    LOGGER.info("  Command: %s", subprocess.list2cmdline(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    saw_error = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if (
                "[ERROR]" in line
                or "OutOfMemoryError" in line
                or "Main loop error" in line
                or "Failed around file" in line
            ):
                saw_error = True
            sys.stdout.write(line)
            sys.stdout.flush()
        process.wait()
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user; terminating Fiji process.")
        process.terminate()
        process.wait()
        raise

    if process.returncode == 0 and not saw_error:
        LOGGER.info("Fiji task completed successfully.")
        return True

    if saw_error:
        LOGGER.error("Detected error markers in Fiji output; treating this step as failed.")
    else:
        LOGGER.error("Fiji task failed with exit code %s.", process.returncode)
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Olympus .oir files to TIFF through Fiji/Bio-Formats."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder containing organized .oir trial folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Pipeline output root. Default: DATA_ROOT.",
    )
    parser.add_argument(
        "--fiji-bin",
        type=Path,
        help="Path to Fiji executable. If omitted, FIJI_BIN/FIJI_PATH and common locations are checked.",
    )
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=default_worker_script(),
        help="Path to 01_fiji_totif_worker.py.",
    )
    parser.add_argument(
        "--action",
        choices=("skip", "overwrite", "keep"),
        default="skip",
        help="Existing-output behavior. Default: skip.",
    )
    parser.add_argument(
        "--extension",
        default=".oir",
        help="Input extension to process. Default: .oir.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Compatibility parameter passed to Fiji worker. Processing is sequential.",
    )
    parser.add_argument(
        "--fiji-memory",
        default=None,
        help="Optional Fiji Java heap size, for example 12g or 16g.",
    )
    parser.add_argument(
        "--projection-mode",
        choices=("auto", "safe", "fast"),
        default="auto",
        help=(
            "Projection strategy for step 01. auto uses fast for smaller movies "
            "and safe for larger ones. safe is slower but uses less memory."
        ),
    )
    parser.add_argument(
        "--metadata-mode",
        choices=("skip", "update-missing", "refresh"),
        default="update-missing",
        help=(
            "What to do with metadata when TIFF outputs already exist. "
            "skip: leave JSON untouched; update-missing: add missing fields such as "
            "acquisition.start_time; refresh: rewrite refreshable metadata fields."
        ),
    )
    parser.add_argument(
        "--stim-export-mode",
        choices=("projected", "raw", "both"),
        default="projected",
        help=(
            "How to export channel 2 when present. projected keeps the current "
            "memory-saving Z-projected Stim_Analog.tif; raw writes an unprojected "
            "plane-series folder for more precise stimulus timing; both writes both."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Fiji command without running Fiji.",
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

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        LOGGER.error("Data root does not exist: %s", data_root)
        return 1
    if not data_root.is_dir():
        LOGGER.error("Data root is not a folder: %s", data_root)
        return 1

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else default_output_root(data_root).resolve()
    )
    worker_script = args.worker_script.expanduser().resolve()
    if not worker_script.exists():
        LOGGER.error("Worker script not found: %s", worker_script)
        return 1

    fiji_bin = find_fiji_bin(args.fiji_bin)
    if fiji_bin is None:
        if args.dry_run:
            LOGGER.warning(
                "Fiji executable was not found, but dry-run will still show the planned arguments."
            )
            LOGGER.info("Data root: %s", data_root)
            LOGGER.info("Output root: %s", output_root)
            LOGGER.info("Worker script: %s", worker_script)
            LOGGER.info(
                "Fiji arguments: %s",
                build_fiji_args(
                    default_original_root(data_root),
                    output_root,
                    args.extension,
                    args.threads,
                    args.action,
                    args.projection_mode,
                    args.metadata_mode,
                    args.stim_export_mode,
                ),
            )
            return 0
        LOGGER.error(
            "Fiji executable was not found. Install Fiji, set FIJI_BIN, or pass --fiji-bin."
        )
        return 1

    fiji_bin = fiji_bin.resolve()
    command_preview = build_fiji_command(
        fiji_bin=fiji_bin,
        worker_script=worker_script,
        worker_args=build_fiji_args(
            default_original_root(data_root),
            output_root,
            args.extension,
            args.threads,
            args.action,
            args.projection_mode,
            args.metadata_mode,
            args.stim_export_mode,
        ),
        fiji_memory=args.fiji_memory,
    )

    if args.dry_run:
        LOGGER.info("Dry-run only; Fiji will not be launched.")
        LOGGER.info("Data root: %s", data_root)
        LOGGER.info("Raw data root: %s", default_original_root(data_root))
        LOGGER.info("Output root: %s", output_root)
        LOGGER.info("Command: %s", subprocess.list2cmdline(command_preview))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    ok = run_fiji_task(
        fiji_bin=fiji_bin,
        worker_script=worker_script,
        data_root=default_original_root(data_root),
        output_root=output_root,
        extension=args.extension,
        threads=args.threads,
        action=args.action,
        fiji_memory=args.fiji_memory,
        projection_mode=args.projection_mode,
        metadata_mode=args.metadata_mode,
        stim_export_mode=args.stim_export_mode,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
