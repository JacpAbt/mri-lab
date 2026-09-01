"""Claims registry: load, validate, and render.

The registry YAML is hand-curated (the prereg->claims join is free text; the
curation IS the program speaking and is user-reviewed). This module is the ONLY
path to printing a claim: render() takes ClaimRefs, refuses unknown ids and
WITHDRAWN/STRUCK entries, and always emits class + n + bounds.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

from mrilab.claims.schema import UNQUOTABLE, Claim, ClaimRef, ClaimStatus

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


class UnknownClaim(KeyError):
    pass


class UnquotableClaim(ValueError):
    pass


@lru_cache(maxsize=1)
def load() -> dict[str, Claim]:
    docs = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    claims = [Claim.model_validate(d) for d in docs]
    by_id: dict[str, Claim] = {}
    for c in claims:
        if c.id in by_id:
            raise ValueError(f"duplicate claim id: {c.id}")
        by_id[c.id] = c
    # referential integrity of supersession edges
    for c in claims:
        for sid in c.supersedes:
            if sid not in by_id:
                raise ValueError(f"{c.id} supersedes unknown claim {sid}")
        if c.superseded_by is not None and c.superseded_by not in by_id:
            raise ValueError(f"{c.id} superseded_by unknown claim {c.superseded_by}")
    return by_id


def registry_version() -> str:
    return hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()[:16]


def get(claim_id: str) -> Claim:
    try:
        return load()[claim_id]
    except KeyError as exc:
        raise UnknownClaim(claim_id) from exc


def render(ref: ClaimRef) -> dict:
    """Resolve a ClaimRef into the block payload faces embed in reports.

    WITHDRAWN/STRUCK claims are refused as positive content — they exist in the
    registry precisely so nothing can quote them; a face that wants to SHOW a
    retraction renders it via render_retraction().
    """
    c = get(ref.claim_id)
    if c.status in UNQUOTABLE:
        raise UnquotableClaim(
            f"claim {c.id} is {c.status}; it cannot be quoted as content"
        )
    return {
        "claim_id": c.id,
        "statement": c.statement,
        "evidence_class": c.evidence_class.value,
        "status": c.status.value,
        "n": c.n,
        "bounds": c.bounds,
        "source": c.source,
        "scope_guards": list(c.scope_guards),
        "superseded_by": c.superseded_by,
        "note": ref.note,
        "verbatim_verified": c.verbatim_verified,
    }


def render_retraction(claim_id: str) -> dict:
    """Explicitly render a WITHDRAWN/STRUCK/KILLED entry AS a retraction
    (misses shown with the same prominence as hits — house rule)."""
    c = get(claim_id)
    if c.status not in UNQUOTABLE and c.status != ClaimStatus.KILLED:
        raise ValueError(f"claim {c.id} is {c.status}, not a retraction")
    return {
        "claim_id": c.id,
        "retracted_statement": c.statement,
        "status": c.status.value,
        "n": c.n,
        "bounds": c.bounds,
        "source": c.source,
        "superseded_by": c.superseded_by,
    }
