"""The report kit: the ONLY constructors faces use to make blocks.

Claims enter reports exclusively as ClaimRefs resolved through the registry
(refusing WITHDRAWN/STRUCK); denied scope guards become Gap blocks — a refusal
is rendered, never silent.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path

from mrilab.claims import registry
from mrilab.claims.schema import ClaimRef
from mrilab.claims.scope_guards import GuardResult
from mrilab.faces.schema import (Advisory, ClaimVerdict, Gap, Provenance,
                                 Reading, Report, Retraction, Section,
                                 SubjectRef, SurfacePlacement,
                                 TrapWarningBlock)
from mrilab.guards.traps import TrapWarning


def _bid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def reading(title: str, data: dict, body_md: str = "",
            provenance: Provenance | None = None) -> Reading:
    return Reading(id=_bid("rd"), title=title, body_md=body_md, data=data,
                   provenance=provenance or Provenance())


def placement(title: str, data: dict, body_md: str = "",
              surface_ids: list[str] | None = None,
              claim_ids: list[str] | None = None) -> SurfacePlacement:
    return SurfacePlacement(
        id=_bid("pl"), title=title, body_md=body_md, data=data,
        provenance=Provenance(surface_ids=surface_ids or [],
                              claim_ids=claim_ids or []))


def claim(claim_id: str, note: str | None = None) -> ClaimVerdict:
    data = registry.render(ClaimRef(claim_id=claim_id, note=note))
    return ClaimVerdict(
        id=_bid("cl"), title=data["statement"].split(":")[0][:80],
        body_md=data["statement"], data=data,
        severity="note" if data["evidence_class"] != "measured" else "info",
        provenance=Provenance(claim_ids=[claim_id]))


def retraction(claim_id: str) -> Retraction:
    data = registry.render_retraction(claim_id)
    return Retraction(id=_bid("rt"), title=f"RETRACTED: {claim_id}",
                      body_md=data["retracted_statement"], data=data,
                      severity="warn",
                      provenance=Provenance(claim_ids=[claim_id]))


def advisory(title: str, body_md: str, claim_ids: list[str],
             data: dict | None = None) -> Advisory:
    if not claim_ids:
        raise ValueError("advisories must be claim-gated (no free-text advice)")
    for cid in claim_ids:
        registry.render(ClaimRef(claim_id=cid))  # raises if unquotable
    return Advisory(id=_bid("ad"), title=title, body_md=body_md,
                    data=data or {}, severity="note",
                    provenance=Provenance(claim_ids=claim_ids))


def gap(title: str, why: str, unlocked_by: str | None = None) -> Gap:
    return Gap(id=_bid("gp"), title=title, body_md=why,
               data={"why": why, "unlocked_by": unlocked_by}, severity="note")


def guard_gap(result: GuardResult, context: str) -> Gap:
    """A DENIED scope guard rendered as an explicit refusal."""
    return Gap(id=_bid("gg"), title=f"Not licensed: {context}",
               body_md=result.reason,
               data={"guard": result.guard, "reason": result.reason},
               severity="warn")


def trap_block(w: TrapWarning) -> TrapWarningBlock:
    return TrapWarningBlock(id=_bid("tw"), title=f"Instrument trap: {w.trap}",
                            body_md=w.message, severity=w.severity,
                            data={"trap": w.trap, "severity": w.severity,
                                  "message": w.message})


def row_provenance(instrument: str, paths: list[Path],
                   dtype: str | None = None) -> Provenance:
    hashes = []
    for p in paths:
        try:
            hashes.append(hashlib.sha256(p.read_bytes()).hexdigest()[:16])
        except OSError:
            hashes.append("unreadable")
    return Provenance(instrument=instrument, row_paths=[str(p) for p in paths],
                      row_hashes=hashes, dtype=dtype)


def new_report(face: str, subject: SubjectRef,
               sections: list[Section]) -> Report:
    counts: dict[str, int] = {}
    for sec in sections:
        for b in sec.blocks:
            counts[b.kind] = counts.get(b.kind, 0) + 1
    return Report(
        face=face,  # type: ignore[arg-type]
        report_id=f"rep_{uuid.uuid4().hex[:12]}",
        created_at=dt.datetime.now(dt.timezone.utc),
        subject=subject,
        registry_version=registry.registry_version(),
        verdict_summary={"block_counts": counts},
        sections=sections,
    )


def save_report(report: Report, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    p = reports_dir / f"{report.report_id}.json"
    p.write_text(report.model_dump_json(indent=1), encoding="utf-8")
    return p


def load_report(reports_dir: Path, report_id: str) -> Report:
    p = reports_dir / f"{report_id}.json"
    return Report.model_validate_json(p.read_text(encoding="utf-8"))
