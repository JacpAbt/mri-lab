"""Row index over the workspace's result JSONs (rebuildable; files are truth).

Every indexed row carries: payload sha256, the env fingerprint it was measured
under, and — what the research repo's own ENVIRONMENT.md flags as a gap — the
RESOLVED HF commit sha of the checkpoint at run time (an upstream re-upload
would otherwise change inputs silently).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlmodel import select

from mrilab.engine import fingerprint as fp
from mrilab.engine.db import session
from mrilab.engine.models import EnvFingerprint, RowIndex
from mrilab.engine.skipcheck import existing_valid_rows
from mrilab.instruments._registry import INSTRUMENTS
from mrilab.settings import Settings


def resolve_hf_sha(repo: str | None, revision: str | None) -> str | None:
    if not repo:
        return None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, revision=revision or "main")
        return info.sha
    except Exception:
        return None  # offline is fine: recorded as unresolved, never guessed


def store_fingerprint(settings: Settings) -> str:
    """Capture the live research-venv fingerprint, upsert it, return its id."""
    payload = fp.capture(settings)
    canon = json.dumps(payload, sort_keys=True)
    fid = "env_" + hashlib.sha256(canon.encode()).hexdigest()[:16]
    lock = json.loads(fp.lock_path(settings).read_text(encoding="utf-8"))
    bad = fp.mismatches(payload, lock)
    with session(settings) as s:
        if s.get(EnvFingerprint, fid) is None:
            s.add(EnvFingerprint(id=fid, payload_json=canon,
                                 matches_lock=not bad,
                                 mismatches="; ".join(bad) or None))
    return fid


def index_rows(settings: Settings, instrument: str, output_dir: Path,
               subject: str, repo: str | None, revision: str | None,
               dtype: str | None, fingerprint_id: str | None,
               job_id: str | None) -> int:
    """Index every payload-valid row under output_dir. Idempotent on
    (path, payload sha)."""
    resolved = resolve_hf_sha(repo, revision)
    n = 0
    with session(settings) as s:
        for p in existing_valid_rows(instrument, output_dir):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            already = s.exec(
                select(RowIndex).where(RowIndex.path == str(p),
                                       RowIndex.payload_sha256 == sha)
            ).first()
            if already:
                continue
            s.add(RowIndex(instrument=instrument, subject=subject, repo=repo,
                           revision=revision, resolved_sha=resolved,
                           path=str(p), payload_sha256=sha, dtype=dtype,
                           fingerprint_id=fingerprint_id, job_id=job_id))
            n += 1
    return n


def rescan(settings: Settings) -> int:
    """Rebuild the index from disk (the DB is disposable)."""
    total = 0
    for instrument in INSTRUMENTS:
        root = settings.rows_dir / instrument
        if not root.is_dir():
            continue
        for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            total += index_rows(settings, instrument, subject_dir,
                                subject=subject_dir.name, repo=None,
                                revision=None, dtype=None,
                                fingerprint_id=None, job_id=None)
    return total
