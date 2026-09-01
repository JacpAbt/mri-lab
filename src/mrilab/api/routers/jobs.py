from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from sqlmodel import select

from mrilab.engine.db import session
from mrilab.engine.jobs import JobSpec
from mrilab.engine.models import Job, JobState
from mrilab.engine.queue import submit
from mrilab.settings import get_settings

router = APIRouter(tags=["jobs"])


@router.post("/jobs")
def create_job(spec: JobSpec) -> dict:
    return submit(get_settings(), spec).model_dump()


@router.get("/jobs")
def list_jobs(state: JobState | None = None, instrument: str | None = None,
              limit: int = 100) -> list[dict]:
    with session(get_settings()) as db:
        q = select(Job).order_by(Job.created_at.desc()).limit(limit)
        if state is not None:
            q = q.where(Job.state == state)
        if instrument is not None:
            q = q.where(Job.instrument == instrument)
        return [j.model_dump() for j in db.exec(q).all()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with session(get_settings()) as db:
        job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    return job.model_dump()


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, tail: int = 200) -> str:
    with session(get_settings()) as db:
        job = db.get(Job, job_id)
    if job is None or not job.log_path or not Path(job.log_path).is_file():
        raise HTTPException(404, "no log for this job")
    lines = Path(job.log_path).read_text(encoding="utf-8",
                                         errors="replace").splitlines()
    return "\n".join(lines[-tail:])


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    with session(get_settings()) as db:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        if job.state in (JobState.QUEUED, JobState.WAITING_DEPS):
            job.state = JobState.CANCELLED
            db.add(job)
        elif job.state == JobState.RUNNING:
            raise HTTPException(409, "job already running; not interruptible in v1")
        return job.model_dump()
