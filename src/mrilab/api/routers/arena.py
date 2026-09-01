from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import select

from mrilab.claims import registry
from mrilab.engine.db import session
from mrilab.engine.models import ArenaPrediction
from mrilab.settings import get_settings

router = APIRouter(tags=["arena"])


class RegisterRequest(BaseModel):
    claim_id: str
    title: str
    notes: str | None = None


class ResolveRequest(BaseModel):
    status: str  # challenged | falsified | upheld
    notes: str | None = None


@router.post("/arena/predictions")
def register(req: RegisterRequest) -> dict:
    registry.get(req.claim_id)  # must exist
    with session(get_settings()) as db:
        row = ArenaPrediction(claim_id=req.claim_id, title=req.title,
                              notes=req.notes)
        db.add(row)
        db.flush()
        db.refresh(row)
        return row.model_dump()


@router.get("/arena/predictions")
def list_predictions() -> list[dict]:
    with session(get_settings()) as db:
        return [r.model_dump() for r in db.exec(select(ArenaPrediction)).all()]


@router.post("/arena/predictions/{pred_id}/resolve")
def resolve(pred_id: int, req: ResolveRequest) -> dict:
    if req.status not in ("challenged", "falsified", "upheld"):
        raise HTTPException(422, "status must be challenged|falsified|upheld")
    with session(get_settings()) as db:
        row = db.get(ArenaPrediction, pred_id)
        if row is None:
            raise HTTPException(404, "unknown prediction")
        row.status = req.status
        row.notes = req.notes or row.notes
        row.resolved_at = dt.datetime.now(dt.timezone.utc)
        db.add(row)
        db.flush()
        db.refresh(row)
        return row.model_dump()


@router.get("/arena/scoreboard")
def scoreboard() -> dict:
    with session(get_settings()) as db:
        rows = db.exec(select(ArenaPrediction)).all()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {"counts": counts, "n": len(rows),
            "note": "a falsification localizes which law was wrong — that is "
                    "the theory earning its keep, not an embarrassment"}
