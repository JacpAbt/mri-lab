"""Typed loaders over the committed reference-surface bundle.

The bundle (src/mrilab/surfaces/bundle/) holds the research program's stored
result rows verbatim (minus stripped direction vectors); manifest.json carries
per-file sha256s. These loaders normalize the recurring shapes:

  - config+rows surfaces (icl_readiness.json / sensitivity_ladder.json)
  - standalone row files (row_*.json — one element of the parent's rows[])
  - flat erosion records (erosion2|W|ZB, eq_out, u2*_out, dp*_out crash rows)
  - verdict aggregates (vitals_*.json, zter rows)

Conventions preserved from upstream: stringified numeric keys ("0.005", "16")
are kept as strings at the storage level and exposed through typed accessors;
`revision: null` means HF `main` and is semantically meaningful.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BUNDLE = Path(__file__).parent / "bundle"

# family tag per readiness/sens directory suffix
_FAMILY_DIRS = {
    "70m": ("EleutherAI/pythia-70m", "pythia-70m"),
    "160m": ("EleutherAI/pythia-160m", "pythia-160m"),
    "410m": ("EleutherAI/pythia-410m", "pythia-410m"),
    "14b": ("EleutherAI/pythia-1.4b", "pythia-1.4b"),
    "olmo1b": ("allenai/OLMo-2-0425-1B", "OLMo-2-1B"),
}


def _read(rel: str) -> dict:
    return json.loads((BUNDLE / rel).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def manifest() -> dict:
    return _read("manifest.json")


def verify_manifest() -> list[str]:
    """Recompute bundled sha256s; return mismatch/missing descriptions."""
    bad: list[str] = []
    for e in manifest()["files"]:
        p = BUNDLE / e["path"]
        if not p.is_file():
            bad.append(f"missing: {e['path']}")
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != e["bundled_sha256"]:
            bad.append(f"sha mismatch: {e['path']}")
    return bad


# --- readiness surface (blueprint §38: 22 points, 5 sizes, 2 families) --------

@dataclass(frozen=True)
class ReadinessPoint:
    family: str
    repo: str
    revision: str
    loss: float
    gain_unseen: float  # PRIMARY per §38b (nats at k=16)
    gain_seen: float
    g_icl: float
    gain_unseen_ci: tuple[float, float]
    source: str


@lru_cache(maxsize=1)
def readiness_surface(uniform: bool = True) -> list[ReadinessPoint]:
    pts: list[ReadinessPoint] = []
    prefix = "icl_readiness_uniform_" if uniform else "icl_readiness_"
    for suffix, (repo, family) in _FAMILY_DIRS.items():
        d = BUNDLE / f"{prefix}{suffix}"
        if not d.is_dir():
            continue
        for rowf in sorted(d.glob("row_*.json")):
            row = json.loads(rowf.read_text(encoding="utf-8"))
            k16 = row["by_k"].get("16")
            if k16 is None:
                continue
            pts.append(ReadinessPoint(
                family=family, repo=repo,
                revision=row["revision"],
                loss=row["loss"],
                gain_unseen=k16["gain_unseen"],
                gain_seen=k16["gain_seen"],
                g_icl=k16["G_icl"],
                gain_unseen_ci=tuple(k16["gain_unseen_ci"]),
                source=str(rowf.relative_to(BUNDLE)),
            ))
    return sorted(pts, key=lambda p: p.loss)


# --- sensitivity ladder rows (Cell L; dL_abs / S_wt faces) --------------------

@dataclass(frozen=True)
class SensRow:
    family: str
    repo: str
    revision: str
    loss: float
    dl_abs: float           # ABS gamma=5e-3 mean (tr(H) face)
    dl_abs_sd: float
    s_wt: float             # REL rho=0.10 mean (tr(H)*||W||^2/N face)
    s_wt_sd: float
    source: str


@lru_cache(maxsize=1)
def sens_rows() -> list[SensRow]:
    rows: list[SensRow] = []
    for d in sorted(BUNDLE.glob("sens_ladder_*")):
        suffix = d.name.removeprefix("sens_ladder_").removesuffix("_cpu")
        base = suffix.removesuffix("_stage2")
        repo, family = _FAMILY_DIRS.get(base, ("?", suffix))
        for rowf in sorted(d.glob("row_*.json")):
            row = json.loads(rowf.read_text(encoding="utf-8"))
            a = row.get("abs", {}).get("0.005")
            r = row.get("rel", {}).get("0.1")
            if a is None or r is None:
                continue
            rows.append(SensRow(
                family=family + ("_stage2" if "_stage2" in d.name else ""),
                repo=repo, revision=row.get("revision"),
                loss=row.get("loss"),
                dl_abs=a["mean"], dl_abs_sd=a["sd"],
                s_wt=r["mean"], s_wt_sd=r["sd"],
                source=str(rowf.relative_to(BUNDLE)),
            ))
    return sorted(rows, key=lambda x: (x.family, x.loss if x.loss is not None else 0))


# --- crash-test / erosion records ---------------------------------------------

@dataclass(frozen=True)
class ErosionRow:
    repo: str
    revision: str
    target: str
    lr: float
    steps: int
    damage: float
    gain: float
    c4_base: float
    target_base: float
    source: str


def _erosion_from(doc: dict, source: str) -> ErosionRow | None:
    cfg = doc.get("config", {})
    if "damage" not in doc or "gain" not in doc:
        return None
    return ErosionRow(
        repo=cfg.get("repo"), revision=cfg.get("revision"),
        target=cfg.get("target"), lr=cfg.get("lr"), steps=cfg.get("steps"),
        damage=doc["damage"], gain=doc["gain"],
        c4_base=doc.get("c4_base"), target_base=doc.get("target_base"),
        source=source,
    )


@lru_cache(maxsize=1)
def erosion_rows() -> list[ErosionRow]:
    rows: list[ErosionRow] = []
    for pattern in ("erosion2/*/erosion.json", "erosionW/*/erosion.json",
                    "erosionZB/*/erosion.json", "eq_out/*/erosion.json",
                    "u2_out/*/erosion.json", "u2bis_out/*/erosion.json",
                    "u2ter_out/*/erosion.json", "dp_out/*/erosion.json",
                    "dp11_out/*/erosion.json"):
        for f in sorted(BUNDLE.glob(pattern)):
            row = _erosion_from(json.loads(f.read_text(encoding="utf-8")),
                                str(f.relative_to(BUNDLE)))
            if row is not None:
                rows.append(row)
    return rows


# --- int4 pricing pairs (Z-ter / Z-ter-b: dL_abs -> RTN int4 damage, 6/6) -----

@dataclass(frozen=True)
class Int4Pair:
    label: str
    family: str
    dl_abs: float
    int4_damage: float
    int8_damage: float
    base_c4: float


# dL_abs sources for the six artifacts: pythia from vitals rows (exact-seed
# panel), OLMo from the stored sens rows named in the Z-TER-B prereg block.
_ZTER_DLABS_SOURCES = {
    "160m_step40000": ("vitals_out/vitals_pythia-160m@step40000.json", "vitals"),
    "160m_step143000": ("vitals_out/vitals_pythia-160m@step143000.json", "vitals"),
    "160m_rejuv": ("vitals_out/vitals_rejuv_artifact_3e-5.json", "vitals"),
    "olmo_210B": ("sens_ladder_olmo1b/row_stage1-step100000-tokens210B.json", "sens"),
    "olmo_4001B": ("sens_ladder_olmo1b/row_stage1-step1907359-tokens4001B.json", "sens"),
    "olmo_stage2_51B": ("sens_ladder_olmo1b_stage2/row_stage2-ingredient1-step23852-tokens51B.json", "sens"),
}


@lru_cache(maxsize=1)
def int4_pricing_pairs() -> list[Int4Pair]:
    pairs: list[Int4Pair] = []
    for zdir, family in (("zter_out", "pythia-160m"), ("zterb_out_olmo", "OLMo-2-1B")):
        d = BUNDLE / zdir
        if not d.is_dir():
            continue
        for rowf in sorted(d.glob("row_*.json")):
            row = json.loads(rowf.read_text(encoding="utf-8"))
            label = row["label"]
            src = _ZTER_DLABS_SOURCES.get(label)
            if src is None:
                continue
            rel, kind = src
            if not (BUNDLE / rel).is_file():
                continue
            doc = _read(rel)
            dl = doc["dL_abs"] if kind == "vitals" else doc["abs"]["0.005"]["mean"]
            pairs.append(Int4Pair(label=label, family=family, dl_abs=dl,
                                  int4_damage=row["int4"], int8_damage=row["int8"],
                                  base_c4=row["base_c4"]))
    return sorted(pairs, key=lambda p: p.dl_abs)


# --- vitals rows ---------------------------------------------------------------

def vitals_row(label: str) -> dict:
    return _read(f"vitals_out/vitals_{label}.json")


def vitals_labels() -> list[str]:
    return sorted(p.stem.removeprefix("vitals_")
                  for p in (BUNDLE / "vitals_out").glob("vitals_*.json"))
