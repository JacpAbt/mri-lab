"""Huey lanes + the job wrapper.

Division of labor (the design's main risk, stated): Huey is DISPATCH ONLY —
persistence of pending tasks, retries, revocation. Domain truth (what happened)
lives in the Job table; nothing ever queries Huey state for a domain decision.
Consumer-restart re-delivery is safe because idempotency comes from the
payload-validating skip-checks, not from queue semantics.

Two SqliteHuey instances on separate files under the workspace:
  gpu lane -> 1 worker (serial GPU discipline), cpu lane -> 2 workers.
Run consumers via `mrilab worker gpu|cpu`.

Wrapper sequence per job: mark RUNNING -> payload skip-check -> trap lint
(blocks refuse) -> thermal gate (gpu lane; proceed-with-warning is STAMPED)
-> subprocess via the executor -> validate output payload -> fingerprint ->
index rows -> mark DONE/FAILED.
"""

from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache

from huey import SqliteHuey

from mrilab.engine import rowstore
from mrilab.engine.db import session
from mrilab.engine.executor import run_instrument, wait_for_cool
from mrilab.engine.jobs import JobSpec
from mrilab.engine.models import Job, JobState
from mrilab.engine.skipcheck import existing_valid_rows
from mrilab.guards.traps import lint_job
from mrilab.settings import Settings, get_settings


@lru_cache(maxsize=1)
def queues() -> dict[str, SqliteHuey]:
    s = get_settings()
    s.ensure_workspace()
    return {
        "gpu": SqliteHuey("mrilab-gpu", filename=str(s.queue_dir / "gpu.db")),
        "cpu": SqliteHuey("mrilab-cpu", filename=str(s.queue_dir / "cpu.db")),
    }


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def submit(settings: Settings, spec: JobSpec, enqueue: bool = True) -> Job:
    """Create (or reuse) the Job row and enqueue it on its lane. Resubmitting a
    FAILED/CANCELLED job re-queues the SAME row (content hash = identity).
    enqueue=False creates/queues the row without dispatching to Huey — the CLI
    --sync path uses it, then calls _run in-process."""
    job_id = spec.content_hash()
    warnings = lint_job({"kind": _job_kind(spec), **spec.params})
    blocked = [w for w in warnings if w.severity == "block"]
    with session(settings) as s:
        job = s.get(Job, job_id)
        if job is not None and job.state in (JobState.QUEUED, JobState.RUNNING,
                                             JobState.DONE, JobState.SKIPPED):
            return job
        if job is None:
            job = Job(
                id=job_id, instrument=spec.instrument, lane=spec.lane,
                dtype=spec.dtype, subject=spec.subject.label,
                params_json=json.dumps(spec.params, sort_keys=True),
                output_dir=str(spec.output_dir(settings)),
            )
        job.trap_warnings_json = json.dumps(
            [w.__dict__ for w in warnings]) if warnings else None
        if blocked:
            job.state = JobState.FAILED
            job.error = "trap linter blocked: " + "; ".join(w.message for w in blocked)
        else:
            job.state = JobState.QUEUED
            job.error = None
        s.add(job)
        state = job.state
    if state == JobState.QUEUED and enqueue:
        _enqueue(spec.lane, job_id, spec)
    return job


def _job_kind(spec: JobSpec) -> str:
    """Lint kind. The frozen panel instruments (vitals/sensitivity_ladder etc.)
    declare their conventions and grids INTERNALLY (exact Cell L seeds), so the
    perturbation-design traps do not apply to them — 'training' jobs get the
    epoch/LR-grid/pool traps; everything else lints as inference."""
    training = {"erosion_arms", "repetition_arms", "kd_arms", "l8_arms", "u1_run"}
    if spec.instrument in training:
        return "training"
    return "inference"


def _enqueue(lane: str, job_id: str, spec: JobSpec) -> None:
    payload = spec.model_dump_json()
    if lane == "gpu":
        run_gpu_job(job_id, payload)
    else:
        run_cpu_job(job_id, payload)


def _run(job_id: str, spec_json: str) -> str:
    settings = get_settings()
    spec = JobSpec.model_validate_json(spec_json)
    out_dir = spec.output_dir(settings)

    with session(settings) as s:
        job = s.get(Job, job_id)
        if job is None or job.state == JobState.CANCELLED:
            return "cancelled"
        job.state = JobState.RUNNING
        job.started_at = _utcnow()
        s.add(job)

    # payload-validating skip (never bare existence)
    if existing_valid_rows(spec.instrument, out_dir):
        with session(settings) as s:
            job = s.get(Job, job_id)
            job.state = JobState.SKIPPED
            job.finished_at = _utcnow()
            s.add(job)
        return "skipped"

    thermal_note = None
    if spec.lane == "gpu":
        ok, temp = wait_for_cool(settings)
        if not ok:
            thermal_note = (f"proceeded at {temp}C after "
                            f"{settings.thermal_retries} waits (limit "
                            f"{settings.thermal_limit_c}C)")

    res = run_instrument(settings, spec.instrument, spec.cli_args(settings),
                         log_name=job_id)

    valid = existing_valid_rows(spec.instrument, out_dir)
    fid = None
    try:
        fid = rowstore.store_fingerprint(settings)
    except Exception:
        pass  # fingerprint capture failure is recorded as absence, never fatal
    if res.ok and valid:
        rowstore.index_rows(settings, spec.instrument, out_dir,
                            subject=spec.subject.label,
                            repo=spec.subject.repo,
                            revision=spec.subject.revision,
                            dtype=spec.dtype, fingerprint_id=fid, job_id=job_id)

    with session(settings) as s:
        job = s.get(Job, job_id)
        job.rc = res.rc
        job.log_path = str(res.log_path)
        job.duration_s = res.duration_s
        job.fingerprint_id = fid
        job.thermal_warning = thermal_note
        job.finished_at = _utcnow()
        if res.ok and valid:
            job.state = JobState.DONE
        else:
            job.state = JobState.FAILED
            job.error = ("subprocess failed" if not res.ok
                         else "no payload-valid row produced (empty-output poison guard)")
        s.add(job)
    return "done" if res.ok and valid else "failed"


# Huey task registration (module import time; consumers pick their lane)
run_gpu_job = queues()["gpu"].task(retries=0)(_run)
run_cpu_job = queues()["cpu"].task(retries=0)(_run)
