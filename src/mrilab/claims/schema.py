"""Claim registry schema.

A Claim is a quotable statement from the research program with its evidence
class, its n, and its bounds — the bounds text is carried VERBATIM from the
source document, never paraphrased or reformatted through floats. Faces may
print claims only through ClaimRefs resolved by the registry (claims/registry.py);
free-text claims cannot enter a report.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EvidenceClass(StrEnum):
    MEASURED = "measured"      # ours, against a measured null
    LIT = "lit"                # established in the literature — cited, not re-proven
    CONJECTURE = "conjecture"  # designed/hypothesized, not scored


class ClaimStatus(StrEnum):
    LIVE = "LIVE"              # scored and standing (or synthesis over scored cells)
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    RETIRED = "RETIRED"        # replaced/bounded by later cells; do not feature
    WITHDRAWN = "WITHDRAWN"    # explicitly retracted — renderer refuses
    STRUCK = "STRUCK"          # numbers struck before scoring — renderer refuses
    KILLED = "KILLED"          # hypothesis killed; quotable ONLY as a negative result


#: statuses the renderer refuses to feature as positive claims
UNQUOTABLE = {ClaimStatus.WITHDRAWN, ClaimStatus.STRUCK}


class Claim(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,60}$")
    statement: str
    evidence_class: EvidenceClass
    status: ClaimStatus
    n: str  # explicit: families / checkpoints / arms / seeds — part of the claim
    bounds: str  # verbatim-faithful scope text; REQUIRED, never empty
    source: str  # doc + section/cell heading
    scope_guards: list[str] = []
    supersedes: list[str] = []
    superseded_by: str | None = None
    # bounds text verified against the source document verbatim (user review
    # upgrades this; entries sourced from memory summaries start False)
    verbatim_verified: bool = False
    tags: list[str] = []


class ClaimRef(BaseModel):
    """The only object faces may hand the renderer."""
    claim_id: str
    note: str | None = None  # optional reading-specific annotation (not a claim)
