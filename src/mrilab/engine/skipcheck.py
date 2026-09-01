"""Payload-validating skip checks (never bare file existence).

The incident this guards against: a partially-failed stage once wrote
well-formed JSON with an EMPTY payload (blimp == {}), which then poisoned every
existence-based skip downstream. Rule: a job may be skipped only if its
expected output file exists AND `row_is_valid(instrument, path)` says the
payload carries the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _valid_vitals(doc: dict) -> bool:
    if not isinstance(doc.get("loss"), (int, float)):
        return False
    # sens block present unless explicitly skipped at run time
    has_sens = isinstance(doc.get("dL_abs"), (int, float)) and \
        isinstance(doc.get("S_wt"), (int, float))
    has_r = isinstance(doc.get("R"), dict) and \
        isinstance(doc["R"].get("gain_unseen"), (int, float))
    return has_sens or has_r


def _valid_erosion(doc: dict) -> bool:
    return all(isinstance(doc.get(k), (int, float))
               for k in ("damage", "gain", "c4_base", "target_base")) \
        and bool(doc.get("history"))


def _valid_readiness_row(doc: dict) -> bool:
    by_k = doc.get("by_k")
    if not isinstance(by_k, dict) or not by_k:
        return False
    return all(isinstance(v, dict) and isinstance(v.get("gain_unseen"), (int, float))
               for v in by_k.values())


def _valid_zter(doc: dict) -> bool:
    return all(isinstance(doc.get(k), (int, float))
               for k in ("base_c4", "int8", "int4"))


def _valid_m1(doc: dict) -> bool:
    if not isinstance(doc.get("induction_score"), (int, float)):
        return False
    # THE poison case: a "blimp" key whose dict is empty means every paradigm
    # skipped in-container — existence is not validity.
    if "blimp" in doc and not doc["blimp"]:
        return False
    return True


def _valid_nonempty(doc: dict) -> bool:
    return bool(doc)


VALIDATORS = {
    "vitals": _valid_vitals,
    "sensitivity_ladder": _valid_vitals,  # same row shape (loss + abs/rel blocks)
    "erosion_arms": _valid_erosion,
    "repetition_arms": _valid_erosion,
    "kd_arms": _valid_nonempty,
    "l8_arms": _valid_nonempty,
    "icl_readiness": _valid_readiness_row,
    "icl_ladder": _valid_nonempty,
    "z_ter": _valid_zter,
    "m1_signals": _valid_m1,
    "induction_ladder": _valid_nonempty,
    "dp_profile": _valid_nonempty,
    "s_cross": _valid_nonempty,
    "u1_run": _valid_nonempty,
    "match_capability": _valid_nonempty,
}


def row_is_valid(instrument: str, path: Path) -> bool:
    doc = _load(path)
    if doc is None:
        return False
    return VALIDATORS.get(instrument, _valid_nonempty)(doc)


def existing_valid_rows(instrument: str, output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return [p for p in sorted(output_dir.rglob("*.json"))
            if row_is_valid(instrument, p)]
