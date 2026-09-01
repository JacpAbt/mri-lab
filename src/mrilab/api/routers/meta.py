from __future__ import annotations

import json

from fastapi import APIRouter
from sqlmodel import select

from mrilab.claims import registry
from mrilab.engine import fingerprint as fp
from mrilab.engine.db import session
from mrilab.engine.executor import gpu_temperature_c
from mrilab.engine.models import Job, JobState
from mrilab.settings import get_settings

router = APIRouter(tags=["meta"])


@router.get("/meta/health")
def health() -> dict:
    s = get_settings()
    return {
        "ok": True,
        "workspace": str(s.workspace),
        "registry_version": registry.registry_version(),
        "gpu_temperature_c": gpu_temperature_c(),
    }


@router.get("/meta/env")
def env() -> dict:
    s = get_settings()
    lock = json.loads(fp.lock_path(s).read_text(encoding="utf-8"))
    try:
        live = fp.capture(s)
        bad = fp.mismatches(live, lock)
        return {"lock": lock, "live": live, "matches_lock": not bad,
                "mismatches": bad}
    except Exception as exc:
        return {"lock": lock, "live": None, "matches_lock": False,
                "mismatches": [f"capture failed: {exc}"]}


@router.get("/queue")
def queue_state() -> dict:
    s = get_settings()
    with session(s) as db:
        running = db.exec(select(Job).where(Job.state == JobState.RUNNING)).all()
        queued = db.exec(select(Job).where(Job.state == JobState.QUEUED)).all()
    return {
        "gpu_temperature_c": gpu_temperature_c(),
        "running": [j.id for j in running],
        "queued": [j.id for j in queued],
        "lanes": {"gpu": {"workers": 1}, "cpu": {"workers": 2}},
    }
