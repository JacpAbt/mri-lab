from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlmodel import select

from mrilab.engine.db import session
from mrilab.engine.jobs import Subject
from mrilab.engine.models import ReportIndex
from mrilab.faces import (arena, certificate, common as kit, gaudit,
                          mri_report, preflight, recorder, render_md)
from mrilab.faces.gaudit import FamilySplitSpec
from mrilab.faces.recorder import CIGates
from mrilab.faces.schema import Report
from mrilab.settings import get_settings

router = APIRouter(tags=["reports"])


class ReportRequest(BaseModel):
    # model_report / release_cert
    label: str | None = None
    repo: str | None = None
    revision: str | None = None
    # preflight
    target: str | None = None
    horizon_steps: int | None = None
    persistence: str = "durable"
    pool_chunks: int | None = None
    loss: float | None = None
    # memorization_audit
    split: FamilySplitSpec | None = None
    # training_recorder
    run_label: str | None = None
    labels: list[str] | None = None
    gates: CIGates | None = None


def _subject_label(req: ReportRequest) -> str:
    if req.label:
        return req.label
    if req.repo:
        return f"{req.repo.split('/')[-1]}@{req.revision or 'main'}"
    raise HTTPException(422, "need label or repo")


def _build(face: str, req: ReportRequest) -> Report:
    settings = get_settings()
    if face == "model_report":
        try:
            return mri_report.build_from_stored(settings, _subject_label(req),
                                                repo=req.repo,
                                                revision=req.revision)
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc))
    if face == "preflight":
        if not (req.repo and req.target and req.horizon_steps):
            raise HTTPException(422, "preflight needs repo, target, horizon_steps")
        return preflight.build(
            settings,
            Subject(repo=req.repo, revision=req.revision),
            target=req.target, horizon_steps=req.horizon_steps,
            persistence=req.persistence, pool_chunks=req.pool_chunks,
            loss=req.loss)
    if face == "release_cert":
        return certificate.build(settings, _subject_label(req))
    if face == "memorization_audit":
        if req.split is None:
            raise HTTPException(422, "memorization_audit needs split")
        return gaudit.build(req.split)
    if face == "training_recorder":
        if not (req.run_label and req.labels and req.gates):
            raise HTTPException(422, "training_recorder needs run_label, labels, gates")
        return recorder.build(settings, req.run_label, req.labels, req.gates)
    if face == "arena":
        return arena.build()
    raise HTTPException(404, f"unknown face {face}")


@router.post("/reports/{face}")
def create_report(face: str, req: ReportRequest) -> Report:
    settings = get_settings()
    report = _build(face, req)
    path = kit.save_report(report, settings.reports_dir)
    with session(settings) as db:
        db.add(ReportIndex(id=report.report_id, face=report.face,
                           subject=report.subject.label, path=str(path),
                           registry_version=report.registry_version))
    return report


@router.get("/reports")
def list_reports(face: str | None = None, limit: int = 100) -> list[dict]:
    with session(get_settings()) as db:
        q = select(ReportIndex).order_by(ReportIndex.created_at.desc()).limit(limit)
        if face:
            q = q.where(ReportIndex.face == face)
        return [r.model_dump() for r in db.exec(q).all()]


def _load(report_id: str) -> Report:
    settings = get_settings()
    try:
        return kit.load_report(settings.reports_dir, report_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown report {report_id}")


@router.get("/reports/{report_id}.md", response_class=PlainTextResponse)
def get_report_md(report_id: str) -> str:
    # registered BEFORE the plain route: starlette matches in order and
    # {report_id} would otherwise swallow the ".md" suffix
    return render_md.render(_load(report_id))


@router.get("/reports/{report_id}")
def get_report(report_id: str) -> Report:
    return _load(report_id)
