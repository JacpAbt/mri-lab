from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import select

from mrilab.claims import registry
from mrilab.claims.schema import Claim
from mrilab.engine.db import session
from mrilab.engine.models import RowIndex
from mrilab.settings import get_settings
from mrilab.surfaces import loaders, placement

router = APIRouter(tags=["browse"])


@router.get("/rows")
def list_rows(instrument: str | None = None, subject: str | None = None,
              limit: int = 200) -> list[dict]:
    with session(get_settings()) as db:
        q = select(RowIndex).order_by(RowIndex.created_at.desc()).limit(limit)
        if instrument:
            q = q.where(RowIndex.instrument == instrument)
        if subject:
            q = q.where(RowIndex.subject == subject)
        return [r.model_dump() for r in db.exec(q).all()]


@router.get("/models")
def list_models() -> dict:
    with session(get_settings()) as db:
        rows = db.exec(select(RowIndex)).all()
    measured = sorted({r.subject for r in rows})
    return {"workspace_subjects": measured,
            "bundle_vitals": loaders.vitals_labels()}


@router.get("/surfaces")
def list_surfaces() -> dict:
    m = loaders.manifest()
    dirs: dict[str, int] = {}
    for e in m["files"]:
        dirs[e["path"].split("/")[0]] = dirs.get(e["path"].split("/")[0], 0) + 1
    return {"bundled": m["bundled"], "n_files": m["n_files"],
            "total_bytes": m["total_bytes"], "directories": dirs}


@router.get("/surfaces/readiness")
def readiness_surface() -> list[dict]:
    return [p.__dict__ for p in loaders.readiness_surface()]


class PlaceRequest(BaseModel):
    loss: float


@router.post("/surfaces/readiness/place")
def place_readiness(req: PlaceRequest) -> dict:
    pl = placement.readiness_placement(req.loss)
    return {"loss": pl.loss, "verdict": pl.verdict,
            "expected_write_fate": pl.expected_sign,
            "neighbors": [n.__dict__ for n in pl.neighbors],
            "band_note": pl.band_note}


@router.get("/claims")
def list_claims(status: str | None = None, tag: str | None = None) -> list[Claim]:
    claims = registry.load().values()
    return [c for c in claims
            if (status is None or c.status == status)
            and (tag is None or tag in c.tags)]


@router.get("/claims/{claim_id}")
def get_claim(claim_id: str) -> Claim:
    try:
        return registry.get(claim_id)
    except registry.UnknownClaim:
        raise HTTPException(404, f"unknown claim {claim_id}")
