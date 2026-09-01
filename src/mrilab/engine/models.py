"""SQLModel tables. The DB is a REBUILDABLE INDEX — result-row JSON files on
disk are the source of truth (the research repo's discipline, preserved);
`mrilab rescan` re-derives everything here from the filesystem. Huey handles
dispatch/retry only; THIS jobs table is the domain truth about what happened.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from sqlmodel import Field, SQLModel


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class JobState(StrEnum):
    QUEUED = "QUEUED"
    WAITING_DEPS = "WAITING_DEPS"
    RUNNING = "RUNNING"
    DONE = "DONE"
    SKIPPED = "SKIPPED"      # payload-valid output already existed
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)  # JobSpec content hash
    instrument: str
    lane: str  # "gpu" | "cpu"
    dtype: str  # "fp32" | "bf16" | "mixed" — REQUIRED, "auto" rejected upstream
    subject: str  # "<repo>@<revision>" or local path label
    params_json: str  # canonical JSON of the JobSpec params
    state: JobState = JobState.QUEUED
    rc: int | None = None
    log_path: str | None = None
    output_dir: str | None = None
    fingerprint_id: str | None = None
    thermal_warning: str | None = None
    trap_warnings_json: str | None = None  # lint results recorded on the job
    error: str | None = None
    created_at: dt.datetime = Field(default_factory=_utcnow)
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    duration_s: float | None = None


class EnvFingerprint(SQLModel, table=True):
    id: str = Field(primary_key=True)  # sha256 of the canonical payload
    payload_json: str
    matches_lock: bool
    mismatches: str | None = None
    created_at: dt.datetime = Field(default_factory=_utcnow)


class RowIndex(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    instrument: str
    subject: str
    repo: str | None = None
    revision: str | None = None
    resolved_sha: str | None = None  # HF commit sha at run time (the repo's own gap, closed)
    path: str  # absolute path of the row JSON on disk
    payload_sha256: str
    dtype: str | None = None
    fingerprint_id: str | None = None
    job_id: str | None = None
    created_at: dt.datetime = Field(default_factory=_utcnow)


class ReportIndex(SQLModel, table=True):
    id: str = Field(primary_key=True)
    face: str
    subject: str
    path: str
    registry_version: str
    created_at: dt.datetime = Field(default_factory=_utcnow)


class ArenaPrediction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    claim_id: str
    title: str
    status: str = "standing"  # standing | challenged | falsified | upheld
    notes: str | None = None
    registered_at: dt.datetime = Field(default_factory=_utcnow)
    resolved_at: dt.datetime | None = None
