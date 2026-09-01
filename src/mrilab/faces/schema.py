"""Report document model (the frontend's wire contract).

Commitments (plan P6): discriminated block union on `kind` (unknown kinds
render as title + body_md — additive evolution); every block carries
Provenance; `body_md` is presentation while `data` is machine truth; ONE
renderer (render_md.py) walks this JSON; schema_version evolves additively.
`data` payloads are dicts in v1 — their inner shapes are produced exclusively
by faces/common.py constructors, which is where typing tightens next.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class Provenance(BaseModel):
    instrument: str | None = None
    row_paths: list[str] = []
    row_hashes: list[str] = []
    surface_ids: list[str] = []
    claim_ids: list[str] = []
    env_fingerprint_id: str | None = None
    dtype: str | None = None


class BlockBase(BaseModel):
    id: str
    title: str
    body_md: str = ""
    severity: Literal["info", "note", "warn", "block"] = "info"
    provenance: Provenance = Provenance()


class Reading(BlockBase):
    kind: Literal["reading"] = "reading"
    data: dict


class SurfacePlacement(BlockBase):
    kind: Literal["surface_placement"] = "surface_placement"
    data: dict


class ClaimVerdict(BlockBase):
    kind: Literal["claim_verdict"] = "claim_verdict"
    data: dict  # exactly claims.registry.render() output


class Retraction(BlockBase):
    kind: Literal["retraction"] = "retraction"
    data: dict  # exactly claims.registry.render_retraction() output


class Advisory(BlockBase):
    kind: Literal["advisory"] = "advisory"
    data: dict  # claim-gated: must carry claim_ids in provenance


class Gap(BlockBase):
    kind: Literal["gap"] = "gap"
    data: dict  # {"why": ..., "unlocked_by": ...}


class TrapWarningBlock(BlockBase):
    kind: Literal["trap_warning"] = "trap_warning"
    data: dict


Block = Annotated[
    Union[Reading, SurfacePlacement, ClaimVerdict, Retraction, Advisory, Gap,
          TrapWarningBlock],
    Field(discriminator="kind"),
]


class Section(BaseModel):
    id: str
    title: str
    blocks: list[Block] = []


class SubjectRef(BaseModel):
    repo: str | None = None
    revision: str | None = None
    path: str | None = None
    label: str
    resolved_sha: str | None = None


class Report(BaseModel):
    schema_version: str = SCHEMA_VERSION
    face: Literal["model_report", "preflight", "training_recorder",
                  "release_cert", "memorization_audit", "arena"]
    report_id: str
    created_at: dt.datetime
    subject: SubjectRef
    registry_version: str
    engine_env: dict | None = None
    verdict_summary: dict = {}
    sections: list[Section] = []
