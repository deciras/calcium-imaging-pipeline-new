#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path


DEFAULT_NOTES_ROOT = Path("/Users/dingyifei/EvernoteMigration/markdown/实验室/钙成像实验记录")
TRIAL_LINE_RE = re.compile(r"^\s*[-*]*\s*\*{0,2}(?P<trial>[A-Za-z0-9_]+)\*{0,2}\s*[:：]\s*(?P<detail>.+?)\s*$")
DATE_PREFIX_RE = re.compile(r"^(?P<date>\d{8})")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)s")
INITIAL_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)s\s*initial delay", re.IGNORECASE)
INTERVAL_RE = re.compile(r"(\d+(?:\.\d+)?)s\s*interval", re.IGNORECASE)
STIMS_RE = re.compile(r"(\d+)\s*stims?\b", re.IGNORECASE)
CHINESE_GLOBAL_INTERVAL_RE = re.compile(r"每\s*(\d+(?:\.\d+)?)\s*s.*?给一个刺激")
CHINESE_GLOBAL_DURATION_RE = re.compile(r"每次刺激\s*(\d+(?:\.\d+)?)\s*s")
CHINESE_GLOBAL_COUNT_RE = re.compile(r"转\s*(\d+)\s*次")
PULSE_CONFIG_RE = re.compile(
    r"onTime\s*=\s*(\d+(?:\.\d+)?)\s*,\s*offTime\s*=\s*(\d+(?:\.\d+)?)\s*,\s*blinkCount\s*=\s*(\d+)\s*,\s*intialDelayTime\s*=\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
COUNT_RE = re.compile(r"(\d+)\s*次")


def milliseconds_to_seconds(value_ms: str) -> float:
    return float(value_ms) / 1000.0


@dataclass(frozen=True)
class TrialBlock:
    trial_id: str
    detail: str
    extra_lines: list[str]
    note_date: str
    note_path: Path
    source_trial_id: str


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    source_trial_id: str
    note_date: str
    note_path: Path
    stim_mode: str
    stim_on_sec: float
    initial_delay_sec: float
    interval_sec: float | None
    pulse_count: int | None
    pulse_off_sec: float | None
    n_events: int
    angles: list[float]


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    note_date: str
    status: str
    reason: str
    output_dir: Path | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manual stim log bundles from experiment notes.")
    parser.add_argument("--notes-root", type=Path, default=DEFAULT_NOTES_ROOT, help="Root folder containing note markdown files.")
    parser.add_argument(
        "--note-files",
        nargs="*",
        help="Optional list of note filenames under notes-root. If omitted, all note files matching calcium-imaging dates are scanned.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Output staging root.")
    parser.add_argument(
        "--lookup-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional data root used to resolve note trial IDs against metadata.csv. "
            "Can be repeated. Accepts either a year data root like /.../2025_olympus "
            "or its parent folder."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated bundles.")
    parser.add_argument("--dry-run", action="store_true", help="Preview output without writing files.")
    return parser.parse_args(argv)


def iter_note_paths(notes_root: Path, note_files: list[str] | None) -> list[Path]:
    if note_files:
        return [notes_root / name for name in note_files]
    return sorted(path for path in notes_root.glob("*.md") if DATE_PREFIX_RE.match(path.name))


def note_date_from_path(path: Path) -> str | None:
    match = DATE_PREFIX_RE.match(path.name)
    return match.group("date") if match else None


def extract_trial_blocks(note_path: Path) -> list[TrialBlock]:
    note_date = note_date_from_path(note_path)
    if note_date is None:
        return []

    blocks: list[TrialBlock] = []
    current_trial_id: str | None = None
    current_detail = ""
    current_extra: list[str] = []

    lines = note_path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        match = TRIAL_LINE_RE.match(line)
        if match:
            if current_trial_id is not None:
                blocks.append(
                    TrialBlock(
                        trial_id=current_trial_id,
                        detail=current_detail,
                        extra_lines=current_extra,
                        note_date=note_date,
                        note_path=note_path,
                        source_trial_id=current_trial_id,
                    )
                )
            current_trial_id = match.group("trial")
            current_detail = match.group("detail")
            current_extra = []
            continue
        if current_trial_id is not None:
            current_extra.append(line)

    if current_trial_id is not None:
        blocks.append(
            TrialBlock(
                trial_id=current_trial_id,
                detail=current_detail,
                extra_lines=current_extra,
                note_date=note_date,
                note_path=note_path,
                source_trial_id=current_trial_id,
            )
        )
    return blocks


def metadata_dir_for_date(lookup_root: Path, note_date: str) -> Path | None:
    candidates = [
        lookup_root / "00_original_files" / note_date,
        lookup_root / year_bucket(note_date) / "00_original_files" / note_date,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_metadata_trial_ids(lookup_root: Path, note_date: str) -> list[str]:
    metadata_dir = metadata_dir_for_date(lookup_root, note_date)
    if metadata_dir is None:
        return []
    metadata_path = metadata_dir / "metadata.csv"
    if not metadata_path.exists():
        return []

    trial_ids: list[str] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trial_id = str(row.get("Trial_ID", "") or "").strip()
            if trial_id:
                trial_ids.append(trial_id)
    return trial_ids


def trial_id_tokens(trial_id: str) -> list[str]:
    return [token for token in trial_id.split("_") if token]


def is_token_subsequence(shorter: list[str], longer: list[str]) -> bool:
    if not shorter:
        return False
    long_index = 0
    for token in shorter:
        while long_index < len(longer) and longer[long_index] != token:
            long_index += 1
        if long_index >= len(longer):
            return False
        long_index += 1
    return True


def resolve_trial_id(note_trial_id: str, note_date: str, lookup_roots: list[Path]) -> str:
    if not lookup_roots:
        return note_trial_id

    suffix = note_trial_id.rsplit("_", 1)[-1] if "_" in note_trial_id else note_trial_id
    note_tokens = trial_id_tokens(note_trial_id)
    candidates: list[str] = []

    for lookup_root in lookup_roots:
        for trial_id in read_metadata_trial_ids(lookup_root, note_date):
            if trial_id not in candidates:
                candidates.append(trial_id)

    if not candidates:
        return note_trial_id
    if note_trial_id in candidates:
        return note_trial_id

    suffix_matches = [trial_id for trial_id in candidates if trial_id.endswith(f"_{suffix}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    subsequence_matches = [
        trial_id
        for trial_id in candidates
        if is_token_subsequence(note_tokens, trial_id_tokens(trial_id))
        or is_token_subsequence(trial_id_tokens(trial_id), note_tokens)
    ]
    if len(subsequence_matches) == 1:
        return subsequence_matches[0]

    return note_trial_id


def parse_number_list(line: str) -> list[float]:
    text = line.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return []
    values = [float(item) for item in NUMBER_RE.findall(text)]
    return values


def positive_angle_lists(extra_lines: list[str]) -> list[list[float]]:
    lists: list[list[float]] = []
    for line in extra_lines:
        values = parse_number_list(line)
        if not values:
            continue
        if all(0.0 <= value <= 180.0 for value in values):
            lists.append(values)
    return lists


def first_seconds_token(detail: str) -> float | None:
    for match in SECONDS_RE.finditer(detail):
        start = match.start()
        end = match.end()
        after = detail[end : end + 6].lower()
        if after.startswith("/frame"):
            continue
        before = detail[max(0, start - 8) : start].lower()
        if "delay" in after or "interval" in after:
            continue
        if "every" in before:
            continue
        return float(match.group(1))
    return None


def global_defaults(note_text: str) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {}

    interval_match = CHINESE_GLOBAL_INTERVAL_RE.search(note_text)
    if interval_match:
        defaults["interval_sec"] = float(interval_match.group(1))
        defaults.setdefault("initial_delay_sec", float(interval_match.group(1)))

    duration_match = CHINESE_GLOBAL_DURATION_RE.search(note_text)
    if duration_match:
        defaults["stim_on_sec"] = float(duration_match.group(1))

    count_match = CHINESE_GLOBAL_COUNT_RE.search(note_text)
    if count_match:
        defaults["n_events"] = int(count_match.group(1))

    return defaults


def parse_trial_spec(block: TrialBlock, defaults: dict[str, float | int]) -> tuple[TrialSpec | None, str]:
    detail_lower = block.detail.lower()
    if any(token in detail_lower for token in ("nostim", "no stim")) or "没有刺激" in block.detail:
        return None, "nostim"
    if "同上" in block.detail:
        return None, "same-as-previous"
    if "每60帧" in block.detail or "不小心" in block.detail:
        return None, "needs manual confirmation"

    pulse_match = PULSE_CONFIG_RE.search(block.detail)
    if pulse_match:
        stim_on_sec = milliseconds_to_seconds(pulse_match.group(1))
        pulse_off_sec = milliseconds_to_seconds(pulse_match.group(2))
        pulse_count = int(pulse_match.group(3))
        initial_delay_sec = milliseconds_to_seconds(pulse_match.group(4))
        return (
            TrialSpec(
                trial_id=block.trial_id,
                source_trial_id=block.source_trial_id,
                note_date=block.note_date,
                note_path=block.note_path,
                stim_mode="pulse",
                stim_on_sec=stim_on_sec,
                initial_delay_sec=initial_delay_sec,
                interval_sec=None,
                pulse_count=pulse_count,
                pulse_off_sec=pulse_off_sec,
                n_events=1,
                angles=[],
            ),
            "ok",
        )

    angle_groups = positive_angle_lists(block.extra_lines)
    angles = [value for group in angle_groups for value in group]

    initial_delay_match = INITIAL_DELAY_RE.search(block.detail)
    interval_match = INTERVAL_RE.search(block.detail)
    n_stims_match = STIMS_RE.search(block.detail)
    count_match = COUNT_RE.search(block.detail)

    stim_on_sec = first_seconds_token(block.detail)
    if stim_on_sec is None:
        stim_on_sec = float(defaults["stim_on_sec"]) if "stim_on_sec" in defaults else None
    initial_delay_sec = (
        float(initial_delay_match.group(1))
        if initial_delay_match
        else float(defaults["initial_delay_sec"]) if "initial_delay_sec" in defaults else None
    )
    interval_sec = (
        float(interval_match.group(1))
        if interval_match
        else float(defaults["interval_sec"]) if "interval_sec" in defaults else None
    )

    n_events = len(angles)
    if n_events == 0:
        if n_stims_match:
            n_events = int(n_stims_match.group(1))
        elif count_match and any(token in detail_lower for token in ("sustain", "pulse")):
            n_events = int(count_match.group(1))
        elif "n_events" in defaults and any(token in detail_lower for token in ("sustain", "pulse", "half", "full")):
            n_events = int(defaults["n_events"])

    if angles and n_events != len(angles):
        n_events = len(angles)

    if stim_on_sec is None or initial_delay_sec is None or interval_sec is None or n_events <= 0:
        return None, "insufficient timing information"

    if "50-100 stim" in detail_lower and not angles:
        return None, "ambiguous free-form stim note"

    return (
            TrialSpec(
                trial_id=block.trial_id,
                source_trial_id=block.source_trial_id,
                note_date=block.note_date,
                note_path=block.note_path,
                stim_mode="sustain",
            stim_on_sec=float(stim_on_sec),
            initial_delay_sec=float(initial_delay_sec),
            interval_sec=float(interval_sec),
            pulse_count=None,
            pulse_off_sec=None,
            n_events=int(n_events),
            angles=[float(value) for value in angles],
        ),
        "ok",
    )


def year_bucket(note_date: str) -> str:
    return f"{note_date[:4]}_olympus"


def output_dir_for_trial(output_root: Path, note_date: str, trial_id: str) -> Path:
    return output_root / year_bucket(note_date) / "00_stim_logs_raw" / "manual_notes" / note_date / trial_id


def timestamp_rows(spec: TrialSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if spec.stim_mode == "pulse":
        rows.append(
            {
                "event": "STIM_1",
                "mcu_offset_ms": int(round(spec.initial_delay_sec * 1000.0)),
                "host_timestamp": "",
                "mcu_stim_index": 1,
                "angle": "",
            }
        )
        return rows

    assert spec.interval_sec is not None
    for index in range(spec.n_events):
        onset_sec = spec.initial_delay_sec + index * spec.interval_sec
        angle = spec.angles[index] if index < len(spec.angles) else ""
        rows.append(
            {
                "event": f"STIM_{index + 1}",
                "mcu_offset_ms": int(round(onset_sec * 1000.0)),
                "host_timestamp": "",
                "mcu_stim_index": index + 1,
                "angle": angle,
            }
        )
    return rows


def stim_map_rows(spec: TrialSpec) -> list[dict[str, object]]:
    if not spec.angles:
        return []
    return [{"stim_id": index + 1, "angle": angle} for index, angle in enumerate(spec.angles)]


def experiment_config(spec: TrialSpec) -> dict[str, object]:
    config: dict[str, object] = {
        "run_id": spec.trial_id,
        "target_trial_id": spec.trial_id,
        "source_note_path": str(spec.note_path),
        "source_note_trial_id": spec.source_trial_id,
        "generated_by": "current/tools/generate_manual_stim_logs_from_notes.py",
    }
    if spec.stim_mode == "pulse":
        pulse_count = int(spec.pulse_count or 0)
        pulse_off_ms = int(round((spec.pulse_off_sec or 0.0) * 1000.0))
        config["mcu_config"] = {
            "stim_mode": "pulse",
            "initial_delay_ms": int(round(spec.initial_delay_sec * 1000.0)),
            "pulse_count": pulse_count,
            "pulse_on_ms": int(round(spec.stim_on_sec * 1000.0)),
            "pulse_off_ms": pulse_off_ms,
        }
        config["computed_stim_total_ms"] = pulse_count * int(round(spec.stim_on_sec * 1000.0)) + max(
            0,
            pulse_count - 1,
        ) * pulse_off_ms
        return config

    assert spec.interval_sec is not None
    config["mcu_config"] = {
        "stim_mode": "sustain",
        "initial_delay_ms": int(round(spec.initial_delay_sec * 1000.0)),
        "interval_ms": int(round(spec.interval_sec * 1000.0)),
        "stim_on_ms": int(round(spec.stim_on_sec * 1000.0)),
        "stim_count": spec.n_events,
    }
    config["computed_stim_total_ms"] = int(round(spec.stim_on_sec * 1000.0))
    return config


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bundle(spec: TrialSpec, output_dir: Path, overwrite: bool, dry_run: bool) -> TrialResult:
    timestamp_path = output_dir / f"timestamp_log_{spec.trial_id}.csv"
    stim_map_path = output_dir / f"stim_map_{spec.trial_id}.csv"
    config_path = output_dir / f"experiment_config_{spec.trial_id}.json"

    if not overwrite and any(path.exists() for path in (timestamp_path, stim_map_path, config_path)):
        return TrialResult(spec.trial_id, spec.note_date, "skipped", "existing bundle", output_dir)

    if dry_run:
        return TrialResult(spec.trial_id, spec.note_date, "planned", "dry-run", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        timestamp_path,
        ["event", "mcu_offset_ms", "host_timestamp", "mcu_stim_index", "angle"],
        timestamp_rows(spec),
    )
    write_csv(stim_map_path, ["stim_id", "angle"], stim_map_rows(spec))
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(experiment_config(spec), handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return TrialResult(spec.trial_id, spec.note_date, "written", "ok", output_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    notes_root = args.notes_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    lookup_roots = [path.expanduser().resolve() for path in args.lookup_root]
    note_paths = iter_note_paths(notes_root, args.note_files)

    results: list[TrialResult] = []
    for note_path in note_paths:
        if not note_path.exists():
            results.append(TrialResult(note_path.name, "unknown", "failed", "note file missing", None))
            continue

        note_text = note_path.read_text(encoding="utf-8")
        defaults = global_defaults(note_text)
        last_spec_for_note: TrialSpec | None = None
        for block in extract_trial_blocks(note_path):
            resolved_trial_id = resolve_trial_id(block.trial_id, block.note_date, lookup_roots)
            effective_block = replace(block, trial_id=resolved_trial_id)
            spec, status = parse_trial_spec(effective_block, defaults)
            if status == "same-as-previous" and last_spec_for_note is not None:
                spec = replace(
                    last_spec_for_note,
                    trial_id=effective_block.trial_id,
                    source_trial_id=effective_block.source_trial_id,
                    note_date=effective_block.note_date,
                    note_path=effective_block.note_path,
                )
                status = "ok"
            if spec is None:
                results.append(
                    TrialResult(
                        trial_id=effective_block.trial_id,
                        note_date=effective_block.note_date,
                        status="skipped",
                        reason=status,
                        output_dir=None,
                    )
                )
                continue
            last_spec_for_note = spec
            result = write_bundle(
                spec=spec,
                output_dir=output_dir_for_trial(output_root, spec.note_date, spec.trial_id),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            results.append(result)

    written = [item for item in results if item.status in {"written", "planned"}]
    skipped = [item for item in results if item.status == "skipped"]
    failed = [item for item in results if item.status == "failed"]

    print(f"Notes root : {notes_root}")
    print(f"Output root: {output_root}")
    print(f"Dry-run    : {args.dry_run}")
    print(f"Generated  : {len(written)}")
    print(f"Skipped    : {len(skipped)}")
    print(f"Failed     : {len(failed)}")
    print()

    for item in written:
        print(f"[{item.status}] {item.note_date} {item.trial_id} -> {item.output_dir}")
    if skipped:
        print()
        for item in skipped:
            print(f"[skip] {item.note_date} {item.trial_id}: {item.reason}")
    if failed:
        print()
        for item in failed:
            print(f"[failed] {item.trial_id}: {item.reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
