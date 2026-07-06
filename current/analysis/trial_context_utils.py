#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for trial-condition context derived from pipeline outputs and notes."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TRIAL_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?(?P<trial>[A-Za-z0-9_.-]+)(?:\*\*)?\s*:\s*(?P<desc>.+?)\s*$")
TITLE_DATE_RE = re.compile(r"^\s*#\s*(?P<date>\d{8})")
ANGLE_COLUMNS = ("pol_angle", "aolp", "angle", "angle_deg", "polarization_angle")
NO_STIM_TOKENS = ("nostim", "no stim", "no_stim", "baseline")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_excluded_trial_ids(output_root: Path) -> set[str]:
    manifest_dir = output_root / "00_trial_metadata"
    excluded = read_csv_if_exists(manifest_dir / "trial_id_manifest_excluded.csv")
    if excluded.empty:
        return set()
    for column in ("raw_trial_id", "trial_id"):
        if column in excluded.columns:
            return set(excluded[column].dropna().astype(str))
    return set()


def filter_table_by_excluded_trials(df: pd.DataFrame, excluded_trial_ids: set[str]) -> pd.DataFrame:
    if df.empty or not excluded_trial_ids:
        return df
    for column in ("trial_id", "raw_trial_id"):
        if column in df.columns:
            keep = ~df[column].astype(str).isin(excluded_trial_ids)
            return df.loc[keep].reset_index(drop=True)
    return df


def notes_root_candidates(output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    import os

    env_value = os.environ.get("CALCIUM_EXPERIMENT_NOTES_ROOT", "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    home = Path.home()
    candidates.extend(
        [
            output_root.parent / "notes_backup" / "钙成像实验记录",
            output_root.parent / "notes_backup" / "calcium_imaging_experiment_notes",
            home / "Documents" / "实验记录" / "钙成像实验记录",
            home / "Documents" / "calcium_imaging_experiment_notes",
        ]
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def find_notes_root(output_root: Path) -> Path | None:
    for candidate in notes_root_candidates(output_root):
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def canonical_stimulus_mode(text: str) -> str:
    lower = str(text or "").lower()
    if any(token in lower for token in NO_STIM_TOKENS):
        return "no_stim"
    has_pulse = "pulse" in lower or "100 ms" in lower or "100ms" in lower or "flash" in lower
    has_sustain = "sustain" in lower or "5000 ms" in lower or "5000ms" in lower or "steady" in lower or "continuous" in lower
    if has_pulse and has_sustain:
        return "mixed"
    if has_pulse:
        return "pulse"
    if has_sustain:
        return "sustain"
    if "stim" in lower:
        return "other_stim"
    return "unknown"


def normalized_trial_key(trial_id: str) -> str:
    text = str(trial_id or "").strip().lower()
    text = text.replace("eupyrmna", "euprymna")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def canonical_stimulus_tag(mode: str) -> str:
    return {
        "no_stim": "nostim",
        "pulse": "pulse",
        "sustain": "sustain",
        "mixed": "mixstim",
        "other_stim": "otherstim",
        "unknown": "unkstim",
    }.get(str(mode or "").strip(), "unkstim")


def chloride_condition(trial_id: str, note_text: str) -> tuple[str, str]:
    blob = f"{trial_id} {note_text}".lower()
    if "50glu" in blob:
        return "chloride_50pct_replacement", "yes"
    if "cl50" in blob:
        return "chloride_50pct_replacement", "yes"
    if "100cl" in blob:
        return "chloride_control", "no"
    return "unknown", "unknown"


def canonical_chloride_tag(condition: str) -> str:
    return {
        "chloride_50pct_replacement": "cl50rep",
        "chloride_control": "stdcl",
        "unknown": "unkcl",
    }.get(str(condition or "").strip(), "unkcl")


def canonical_aolp_tag(measured_has_aolp: str, note_polarization_setup: str) -> str:
    measured = str(measured_has_aolp or "").strip().lower()
    noted = str(note_polarization_setup or "").strip().lower()
    if measured == "yes" or noted == "with_aolp":
        return "aolp"
    if measured == "no" or noted == "no_aolp":
        return "noaolp"
    return "unkaolp"


def infer_species(trial_id: str, note_text: str = "") -> str:
    blob = f"{trial_id} {note_text}".lower()
    if "euprymna" in blob:
        return "euprymna"
    if "sepia" in blob:
        return "sepia"
    if "gold" in blob:
        return "gold"
    return "unknownsp"


def infer_preparation(trial_id: str, note_text: str = "") -> str:
    blob = f"{trial_id} {note_text}".lower()
    if "retina2" in blob:
        return "retina2"
    if "retina" in blob:
        return "retina"
    if "cell" in blob or "cells" in blob:
        return "cells"
    if "ol" in blob or "optic_lobe" in blob:
        return "ol"
    return "unkprep"


def infer_magnification(trial_id: str, note_text: str = "") -> str:
    blob = f"{trial_id} {note_text}".lower()
    for token in ("25x", "20x", "10x", "5x", "1x"):
        if token in blob:
            return token
    if "_xx" in blob or " xx" in blob:
        return "unkmag"
    return "unkmag"


def infer_replicate_tag(trial_id: str) -> str:
    match = re.search(r"_(\d{1,4})$", str(trial_id))
    if match:
        return f"rep{int(match.group(1)):04d}"
    return "rep0000"


def infer_date_tag(trial_id: str, note_date: str) -> str:
    note_value = str(note_date or "").strip()
    if re.fullmatch(r"\d{8}", note_value):
        return note_value
    trial_match = re.match(r"(\d{8})", str(trial_id or ""))
    if trial_match:
        return trial_match.group(1)
    return "undated"


def raw_trial_date(trial_id: str) -> str:
    trial_match = re.match(r"(\d{8})", str(trial_id or ""))
    return trial_match.group(1) if trial_match else ""


def propose_canonical_trial_id(
    *,
    trial_id: str,
    note_date: str,
    note_text: str,
    planned_stimulus_mode: str,
    measured_has_aolp: str,
    note_polarization_setup: str,
    chloride_condition_value: str,
) -> str:
    date_tag = infer_date_tag(trial_id, note_date)
    species = infer_species(trial_id, note_text)
    prep = infer_preparation(trial_id, note_text)
    mag = infer_magnification(trial_id, note_text)
    chloride_tag = canonical_chloride_tag(chloride_condition_value)
    stim_tag = canonical_stimulus_tag(planned_stimulus_mode)
    aolp_tag = canonical_aolp_tag(measured_has_aolp, note_polarization_setup)
    rep_tag = infer_replicate_tag(trial_id)
    parts = [date_tag, species, prep, mag, chloride_tag, stim_tag, aolp_tag, rep_tag]
    source_date = raw_trial_date(trial_id)
    if re.fullmatch(r"\d{8}", source_date) and source_date != date_tag:
        parts.append(f"src{source_date[2:]}")
    return "_".join(parts)


def table_has_numeric_angle(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    for column in df.columns:
        if column.lower() not in ANGLE_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            return True
    return False


def normalize_stim_type_values(values: list[str]) -> str:
    cleaned = {str(value).strip().lower() for value in values if str(value).strip()}
    cleaned = {value for value in cleaned if value not in {"nan", "none"}}
    if not cleaned:
        return "unknown"
    has_no_stim = any(any(token == value or token in value for token in NO_STIM_TOKENS) for value in cleaned)
    has_pulse = any("pulse" in value or "flash" in value for value in cleaned)
    has_sustain = any("sustain" in value or value in {"steady", "continuous"} for value in cleaned)
    if has_no_stim and len(cleaned) == 1:
        return "no_stim"
    modes = int(has_pulse) + int(has_sustain)
    if modes > 1:
        return "mixed"
    if has_pulse:
        return "pulse"
    if has_sustain:
        return "sustain"
    if has_no_stim and len(cleaned) == 0:
        return "no_stim"
    if has_no_stim and len(cleaned) == 1:
        return "no_stim"
    return "other_stim"


def parse_experiment_notes(notes_root: Path | None) -> dict[str, dict]:
    if notes_root is None or not notes_root.exists():
        return {}
    entries: dict[str, dict] = {}
    for path in sorted(notes_root.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        file_text_lower = text.lower()
        title_date = ""
        for line in text.splitlines():
            title_match = TITLE_DATE_RE.match(line)
            if title_match:
                title_date = title_match.group("date")
                break
        filename_date_match = re.match(r"(\d{8})", path.stem)
        note_date = title_date or (filename_date_match.group(1) if filename_date_match else "")
        file_no_polarizer = "没有偏振片" in text or "没接刺激光路" in text
        for line in text.splitlines():
            match = TRIAL_LINE_RE.match(line)
            if not match:
                continue
            trial_id = match.group("trial").strip()
            desc = match.group("desc").strip()
            payload = {
                "trial_id": trial_id,
                "trial_id_normalized": normalized_trial_key(trial_id),
                "note_date": note_date,
                "note_file": path.name,
                "note_text": desc,
                "planned_stimulus_mode": canonical_stimulus_mode(desc),
                "note_polarization_setup": "no_AoLP" if file_no_polarizer else ("with_AoLP" if "偏振" in file_text_lower else "unknown"),
            }
            entries[trial_id] = payload
            entries.setdefault(normalized_trial_key(trial_id), payload)
    return entries


def _trial_rows(table: pd.DataFrame, trial_id: str) -> pd.DataFrame:
    if table.empty or "trial_id" not in table.columns:
        return pd.DataFrame()
    return table[table["trial_id"].astype(str) == str(trial_id)].copy()


def build_trial_context_table(output_root: Path, qc: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    note_map = parse_experiment_notes(find_notes_root(output_root))
    trial_ids = sorted(set(qc["trial_id"].astype(str))) if not qc.empty and "trial_id" in qc.columns else sorted(note_map)
    rows: list[dict] = []
    for trial_id in trial_ids:
        note = note_map.get(trial_id, note_map.get(normalized_trial_key(trial_id), {}))
        stim_rows = _trial_rows(tables.get("stim_response_table", pd.DataFrame()), trial_id)
        angle_rows = _trial_rows(tables.get("angle_response_table", pd.DataFrame()), trial_id)
        qc_row = _trial_rows(qc, trial_id)
        n_stim_events = int(pd.to_numeric(qc_row.get("n_stim_events", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not qc_row.empty else 0
        measured_stimulus_mode = normalize_stim_type_values(stim_rows.get("stim_type", pd.Series(dtype=str)).astype(str).tolist())
        measured_has_stimulus = "yes" if (n_stim_events > 0 or measured_stimulus_mode not in {"unknown", "no_stim"}) else "no"
        measured_has_aolp = "yes" if (not angle_rows.empty or table_has_numeric_angle(stim_rows) or table_has_numeric_angle(angle_rows)) else "no"
        chlor_condition, reduced_chloride = chloride_condition(trial_id, str(note.get("note_text", "")))
        planned_mode = str(note.get("planned_stimulus_mode", "unknown") or "unknown")
        rows.append(
            {
                "trial_id": trial_id,
                "experiment_note_available": "yes" if note else "no",
                "note_date": note.get("note_date", ""),
                "note_file": note.get("note_file", ""),
                "note_text": note.get("note_text", ""),
                "planned_stimulus_mode": planned_mode,
                "planned_has_stimulus": "yes" if planned_mode not in {"unknown", "no_stim"} else ("no" if planned_mode == "no_stim" else "unknown"),
                "measured_stimulus_mode": measured_stimulus_mode,
                "measured_has_stimulus": measured_has_stimulus,
                "measured_has_AoLP": measured_has_aolp,
                "note_polarization_setup": note.get("note_polarization_setup", "unknown"),
                "chloride_condition": chlor_condition,
                "reduced_chloride": reduced_chloride,
                "analysis_branch_key": f"stim:{measured_stimulus_mode}|aolp:{measured_has_aolp}|chloride:{chlor_condition}",
                "canonical_trial_id_proposed": propose_canonical_trial_id(
                    trial_id=trial_id,
                    note_date=str(note.get("note_date", "")),
                    note_text=str(note.get("note_text", "")),
                    planned_stimulus_mode=planned_mode,
                    measured_has_aolp=measured_has_aolp,
                    note_polarization_setup=str(note.get("note_polarization_setup", "unknown")),
                    chloride_condition_value=chlor_condition,
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_context_warnings(qc: pd.DataFrame) -> pd.DataFrame:
    if qc.empty:
        return qc
    out = qc.copy()

    def combine(row: pd.Series) -> str:
        warnings = [item for item in str(row.get("qc_warnings", "") or "").split(";") if item]
        planned = str(row.get("planned_stimulus_mode", "") or "")
        measured_has_stim = str(row.get("measured_has_stimulus", "") or "")
        note_aolp = str(row.get("note_polarization_setup", "") or "")
        measured_aolp = str(row.get("measured_has_AoLP", "") or "")
        if planned in {"pulse", "sustain", "mixed", "other_stim"} and measured_has_stim == "no":
            warnings.append("planned_stim_not_detected")
        if note_aolp == "no_AoLP" and measured_aolp == "yes":
            warnings.append("note_aolp_mismatch")
        seen: set[str] = set()
        ordered: list[str] = []
        for warning in warnings:
            if warning not in seen:
                seen.add(warning)
                ordered.append(warning)
        return ";".join(ordered)

    out["qc_warnings"] = out.apply(combine, axis=1)
    return out
