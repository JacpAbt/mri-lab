"""Server-rendered rooms (constitution §2: three doors, rooms not dashboards).

Pages are built IN-PROCESS from the same face builders the API uses — no HTTP
round-trip, one source of truth. Templates receive prerendered fragments from
ui.render (block chrome) plus small typed contexts; islands.js adds the three
progressive enhancements. Every page works with JS disabled.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from mrilab.claims import registry
from mrilab.claims.schema import ClaimRef, ClaimStatus
from mrilab.engine import queue as jobqueue
from mrilab.engine.db import session
from mrilab.engine.jobs import Subject
from mrilab.engine.models import Job, ReportIndex
from mrilab.faces import common as kit, mri_report, preflight
from mrilab.settings import get_settings
from mrilab.surfaces import loaders, placement
from mrilab.ui import chart, portrait, render

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

TARGETS = ("legal", "code", "math", "finance_news", "wikitext", "c4")
WALL_STATUSES = {ClaimStatus.WITHDRAWN, ClaimStatus.STRUCK, ClaimStatus.KILLED}


def _base_ctx(request: Request, **extra) -> dict:
    ctx = {"request": request, "registry_version": registry.registry_version()[:8]}
    ctx.update(extra)
    return ctx


def _row_or_404(label: str) -> tuple[dict, Path]:
    found = mri_report.find_vitals_row(get_settings(), label)
    if found is None:
        known = ", ".join(loaders.vitals_labels()[:8])
        raise HTTPException(404, f"no stored vitals row for '{label}' — run a "
                                 f"vitals job first, or open one of: {known}")
    return found


def _subject_bar(label: str, row: dict, row_path: Path) -> dict:
    r = row.get("R") or {}
    face = portrait.mini_face_svg(row.get("loss"), r.get("gain_unseen"),
                                  row.get("dL_abs"), row.get("S_wt"))
    sha = hashlib.sha256(row_path.read_bytes()).hexdigest()[:8]
    who = row.get("repo") or label
    if row.get("revision"):
        who += f" @ {row['revision']}"
    return {"subject_label": label, "face_svg": face, "who": who,
            "sha": f"row {sha} · fp32"}


def _q(label: str) -> str:
    return urllib.parse.quote(label, safe="@._-")


# ---------------------------------------------------------------- load door --

@router.get("/", response_class=HTMLResponse)
def load_page(request: Request):
    settings = get_settings()
    recents = []
    labels = list(loaders.vitals_labels())
    try:  # workspace-measured subjects join the bundle's (DB may be empty)
        from mrilab.engine.models import RowIndex
        with session(settings) as db:
            ws = db.exec(select(RowIndex).where(
                RowIndex.instrument == "vitals")).all()
        labels += [r.subject for r in ws]
    except Exception:
        pass
    for label in sorted(set(labels)):
        found = mri_report.find_vitals_row(settings, label)
        if not found:
            continue
        row, _ = found
        r = row.get("R") or {}
        recents.append({
            "label": label,
            "face": portrait.mini_face_svg(row.get("loss"), r.get("gain_unseen"),
                                           row.get("dL_abs"), row.get("S_wt")),
            "loss": f"{row['loss']:.3f}" if row.get("loss") is not None else "—",
            "gain": (f"{r['gain_unseen']:+.2f}"
                     if r.get("gain_unseen") is not None else None),
        })
    m = loaders.manifest()
    return templates.TemplateResponse(request, "load.html", _base_ctx(
        request, room="load", recents=recents,
        known_labels=[e["label"] for e in recents],
        bundle_note=f"surfaces: {m['n_files']} stored rows bundled"))


@router.get("/go")
def go(label: str):
    return RedirectResponse(f"/m/{_q(label.strip())}", status_code=303)


# ----------------------------------------------------------------- overview --

def _verdict(loss: float, dl_abs: float | None, n_crash: int) -> tuple[str, list[dict]]:
    pl = placement.readiness_placement(loss)
    chips: list[dict] = []
    if pl.verdict == "below-crossing":
        text = "A rule-forming substrate — taught families install as rules here."
        chips.append({"cls": "c-ok", "label": "RULES", "title": ""})
    elif pl.verdict == "above-crossing":
        text = ("An item-forming substrate — writes memorize, with measured "
                "anti-generalization risk.")
        chips.append({"cls": "c-block", "label": "ITEMS RISK", "title": ""})
    else:
        text = "Inside the crossing band — no CI-clear write-fate verdict."
        chips.append({"cls": "c-warn", "label": "NO VERDICT", "title": ""})
    if loss < chart.SAFETY_CROSSING_NATS:
        chips.append({"cls": "c-ok", "label": "PAST SAFETY CROSSING",
                      "title": "safety-readiness-gated: crossing ≈3.46, "
                               "n=1 payload class, in-context channel"})
    elif pl.verdict == "below-crossing":
        chips.append({"cls": "c-warn", "label": "BEFORE SAFETY CROSSING",
                      "title": "rules install but the safety-payload crossing "
                               "(≈3.46, n=1 class) is not yet passed"})
    if dl_abs is not None:
        rank = placement.int4_rank(dl_abs)
        cls = "c-warn" if rank.rank_of >= rank.n_pairs - 1 else ""
        chips.append({"cls": cls, "label": f"INT4 RANK {rank.rank_of}/{rank.n_pairs}",
                      "title": rank.note})
        if rank.rank_of == rank.n_pairs:
            text += " The most quantization-fragile artifact in the measured set."
    if n_crash:
        chips.append({"cls": "c-ok", "label": f"CRASH ×{n_crash}",
                      "title": f"{n_crash} crash arm(s) measured for this subject"})
    return text, chips


def _gauges(row: dict) -> list[dict]:
    r = row.get("R") or {}
    loss, gain = row.get("loss"), r.get("gain_unseen")
    dl, swt = row.get("dL_abs"), row.get("S_wt")
    out = [{
        "ch": "ch-dots", "sym": "◦", "symcls": "", "name": "tremor · held-out loss",
        "sub": "frozen 32-doc C4 · fp32",
        "val": f"{loss:.4f}" if loss is not None else "—", "valcls": "",
        "link": {"href": "/registry#claim-readiness-crossing",
                 "text": "readiness-crossing"}, "small": "",
    }]
    if gain is not None:
        ci = r.get("gain_unseen_ci") or (None, None)
        sub = f"k={r.get('k')} · CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" \
            if ci[0] is not None else f"k={r.get('k')}"
        out.append({"ch": "ch-spokes", "sym": "╱", "symcls": "good",
                    "name": "spokes · unseen-rule gain", "sub": sub,
                    "val": f"{gain:+.2f}",
                    "valcls": "good" if gain > 0 else "bad",
                    "link": {"href": "/registry#claim-unseen-gain-primary",
                             "text": "unseen-gain-primary"}, "small": ""})
    else:
        out.append({"ch": "ch-spokes", "sym": "╱", "symcls": "",
                    "name": "spokes · unseen-rule gain",
                    "sub": "not measured", "val": "—", "valcls": "",
                    "link": None, "small": "run icl_readiness"})
    if dl is not None:
        pct = placement.dl_abs_percentile(dl)
        out.append({"ch": "ch-hatch", "sym": "⌇", "symcls": "hot",
                    "name": "hatch · isotropic curvature",
                    "sub": f"γ=5e-3 · p{pct.percentile:.0f} of {pct.n_rows} stored rows",
                    "val": f"{dl:.3f}",
                    "valcls": "hot" if pct.percentile > 66 else "",
                    "link": {"href": "/registry#claim-dl-abs-prices-int4",
                             "text": "dl-abs-prices-int4"}, "small": ""})
    else:
        out.append({"ch": "ch-hatch", "sym": "⌇", "symcls": "",
                    "name": "hatch · isotropic curvature", "sub": "not measured",
                    "val": "—", "valcls": "", "link": None, "small": "run vitals"})
    if swt is not None:
        pct = placement.s_wt_percentile(swt)
        out.append({"ch": "ch-ring", "sym": "◌", "symcls": "",
                    "name": "tolerance ring · ‖W‖²/N",
                    "sub": f"norm clock · p{pct.percentile:.0f} of {pct.n_rows} stored rows",
                    "val": f"{swt:.3f}", "valcls": "",
                    "link": {"href": "/registry#claim-norm-clock",
                             "text": "norm-clock"}, "small": ""})
    else:
        out.append({"ch": "ch-ring", "sym": "◌", "symcls": "",
                    "name": "tolerance ring · ‖W‖²/N", "sub": "not measured",
                    "val": "—", "valcls": "", "link": None, "small": "run vitals"})
    return out


@router.get("/m/{label}", response_class=HTMLResponse)
def overview(request: Request, label: str):
    settings = get_settings()
    row, row_path = _row_or_404(label)
    r = row.get("R") or {}
    loss, gain = row["loss"], r.get("gain_unseen")
    dl, swt = row.get("dL_abs"), row.get("S_wt")
    from mrilab.engine.skipcheck import existing_valid_rows
    crash_rows = existing_valid_rows(
        "erosion_arms", settings.rows_dir / "erosion_arms" / label)
    verdict_text, chips = _verdict(loss, dl, len(crash_rows))
    pl = placement.readiness_placement(loss)

    rank_n = "—"
    frag_desc = "int4 pricing needs dL_abs (run vitals)"
    if dl is not None:
        rk = placement.int4_rank(dl)
        rank_n = f"{rk.rank_of}/{rk.n_pairs}"
        frag_desc = (f"dL_abs p{placement.dl_abs_percentile(dl).percentile:.0f} "
                     f"of stored rows · crash "
                     + (f"{len(crash_rows)} arm(s) measured" if crash_rows
                        else "not measured"))
    rooms = [
        {"slug": "readiness", "title": "READINESS",
         "n": f"{gain:+.2f}" if gain is not None else "—",
         "ncls": "good" if (gain or 0) > 0 else "",
         "desc": f"{pl.verdict} · {len(loaders.readiness_surface())} stored "
                 f"points · 2 families"},
        {"slug": "fragility", "title": "FRAGILITY", "n": rank_n,
         "ncls": "hot" if rank_n.split('/')[0] not in ('—', '0') else "",
         "desc": frag_desc},
        {"slug": "economics", "title": "ECONOMICS", "n": "≈10⁴×", "ncls": "",
         "desc": "channel prices · exchange table · attacker/defender asymmetry"},
        {"slug": "lineage", "title": "LINEAGE", "n": "—", "ncls": "",
         "desc": "post-training fingerprint · stage ladder is a measured gap"},
    ]
    gaps = [kit.gap("Stage ladder (SFT/DPO/Instruct fingerprint) not measured",
                    "Base-checkpoint view. Unlock: vitals + readiness jobs on "
                    "the family's stage checkpoints (~20 min each).",
                    unlocked_by="LINEAGE room")]
    if not crash_rows:
        gaps.append(kit.gap(
            "Fine-tuning fragility not measured for this subject",
            "The crash test is measured, never predicted from static gauges.",
            unlocked_by="PREFLIGHT room → queue 3 probes"))
    return templates.TemplateResponse(request, "overview.html", _base_ctx(
        request, room="overview",
        **_subject_bar(label, row, row_path),
        verdict_text=verdict_text, chips=chips,
        portrait=portrait.portrait_svg(loss, gain, dl, swt),
        gauges=_gauges(row), rooms=rooms,
        gaps_html="".join(render.block_html(g) for g in gaps),
        source_note=f"vitals row: {row_path.name}"))


# ---------------------------------------------------------------- preflight --
# NB: registered BEFORE /m/{label}/{room} — starlette matches in order and the
# {room} wildcard would otherwise swallow "preflight".

@router.get("/m/{label}/preflight", response_class=HTMLResponse)
def preflight_page(request: Request, label: str, target: str | None = None,
                   horizon: int | None = None, persistence: str = "durable",
                   queued: int | None = None):
    settings = get_settings()
    row, row_path = _row_or_404(label)
    ctx = _base_ctx(request, room="preflight",
                    **_subject_bar(label, row, row_path),
                    targets=TARGETS, target=target, horizon=horizon,
                    persistence=persistence, queued=queued,
                    probe_lrs=list(preflight.PROBE_LRS),
                    sections_html=None, can_queue_probes=False)
    if target and horizon:
        if target not in TARGETS:
            raise HTTPException(422, f"unknown target '{target}'")
        subject = Subject(repo=row.get("repo"), revision=row.get("revision"),
                          path=row.get("path"))
        report = preflight.build(settings, subject, target=target,
                                 horizon_steps=horizon, persistence=persistence)
        probes_missing = any(
            b.kind == "gap" and b.title == "Probes not yet run"
            for s in report.sections for b in s.blocks)
        ctx.update(sections_html=render.sections_html(report),
                   can_queue_probes=probes_missing and bool(row.get("repo")
                                                            or row.get("path")))
    return templates.TemplateResponse(request, "preflight.html", ctx)


@router.post("/m/{label}/preflight/probes")
def queue_probes(label: str, target: str = Form(...), horizon: int = Form(...),
                 persistence: str = Form("durable")):
    settings = get_settings()
    row, _ = _row_or_404(label)
    if not (row.get("repo") or row.get("path")):
        raise HTTPException(409, "stored row carries no repo/path to probe")
    subject = Subject(repo=row.get("repo"), revision=row.get("revision"),
                      path=row.get("path"))
    n = 0
    for spec in preflight.probe_specs(subject, target, horizon):
        jobqueue.submit(settings, spec)
        n += 1
    url = (f"/m/{_q(label)}/preflight?target={target}&horizon={horizon}"
           f"&persistence={persistence}&queued={n}")
    return RedirectResponse(url, status_code=303)


# -------------------------------------------------------------------- rooms --

ROOMS = {
    "readiness": {
        "ids": ["readiness"], "title": "Readiness",
        "question": "What does teaching become here?",
    },
    "fragility": {
        "ids": ["quantization", "fine-tuning"], "title": "Fragility",
        "question": "What breaks first, and at what price?",
    },
    "economics": {
        "ids": ["economics"], "title": "Economics",
        "question": "What does a behaviour cost in each channel?",
    },
    "lineage": {
        "ids": ["post-training", "objects"], "title": "Lineage",
        "question": "What did post-training do to this substrate?",
    },
}


@router.get("/m/{label}/{room}", response_class=HTMLResponse)
def room_page(request: Request, label: str, room: str, anno: str = "on"):
    spec = ROOMS.get(room)
    if spec is None:
        raise HTTPException(404, f"no room '{room}'")
    settings = get_settings()
    row, row_path = _row_or_404(label)
    report = mri_report.build_from_stored(settings, label)
    ctx = _base_ctx(request, room=room, **_subject_bar(label, row, row_path),
                    room_title=spec["title"], question=spec["question"],
                    sections_html=render.sections_html(report, spec["ids"]),
                    source_note=f"vitals row: {row_path.name}",
                    anno=(anno != "off"))
    if room == "readiness":
        r = row.get("R") or {}
        gain = r.get("gain_unseen")
        pl = placement.readiness_placement(row["loss"])
        fate = {"below-crossing": "<b>Rules.</b>",
                "above-crossing": "<b>Items.</b>",
                "inside-crossing-band": "<b>No verdict.</b>"}[pl.verdict]
        ctx.update(
            answer_html=f"{fate} This checkpoint sits {pl.verdict.replace('-', ' ')} "
                        f"(4.7–5.5); expected write fate: {pl.expected_sign}.",
            chart_html=chart.readiness_chart_svg(
                subject_loss=row["loss"], subject_gain=gain,
                subject_label=label),
            chart_title="The readiness surface",
            chart_src=f"{len(loaders.readiness_surface())} measured points · "
                      f"2 families · no fit drawn",
            chart_caption=("subject ringed among the measured points"
                           if gain is not None else
                           "unseen gain not measured — subject not placeable "
                           "on the y-axis"))
    return templates.TemplateResponse(request, "room.html", ctx)


# --------------------------------------------------- registry / queue / etc --

@router.get("/registry", response_class=HTMLResponse)
def registry_page(request: Request):
    claims = registry.load()
    groups = {"measured": [], "lit": [], "conjecture": []}
    wall = []
    for cid in sorted(claims):
        c = claims[cid]
        if c.status in WALL_STATUSES:
            wall.append(render.retraction_card(registry.render_retraction(cid)))
        else:
            groups[c.evidence_class.value].append(
                render.claim_card(registry.render(ClaimRef(claim_id=cid))))
    n_live = sum(len(v) for v in groups.values())
    return templates.TemplateResponse(request, "registry.html", _base_ctx(
        request, room="registry", n_live=n_live,
        groups=[{"title": f"Evidence class: {k}", "cards": v}
                for k, v in groups.items() if v],
        wall=wall))


@router.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request):
    with session(get_settings()) as db:
        jobs = db.exec(select(Job).order_by(Job.created_at.desc())
                       .limit(100)).all()
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.state] = counts.get(j.state, 0) + 1
    return templates.TemplateResponse(request, "queue.html", _base_ctx(
        request, room="queue", jobs=jobs, counts=counts))


@router.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request):
    with session(get_settings()) as db:
        reports = db.exec(select(ReportIndex)
                          .order_by(ReportIndex.created_at.desc())
                          .limit(100)).all()
    return templates.TemplateResponse(request, "reports_list.html", _base_ctx(
        request, room="reports", reports=reports))


@router.get("/reports/{report_id}", response_class=HTMLResponse)
def report_page(request: Request, report_id: str):
    settings = get_settings()
    try:
        report = kit.load_report(settings.reports_dir, report_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown report {report_id}")
    label = report.subject.label
    bar = {}
    found = mri_report.find_vitals_row(settings, label)
    if found:
        bar = _subject_bar(label, *found)
    bar["subject_label"] = label
    return templates.TemplateResponse(request, "report.html", _base_ctx(
        request, room="reports", **bar, face=report.face,
        report_id=report.report_id, created_at=report.created_at,
        report_registry=report.registry_version[:8],
        sections_html=render.sections_html(report)))


@router.get("/recipe", response_class=HTMLResponse)
def recipe_page(request: Request):
    by_target: dict[str, list[dict]] = {}
    for e in loaders.erosion_rows():
        if e.target is None or e.lr is None:
            continue
        by_target.setdefault(e.target, []).append({
            "label": f"{(e.repo or '?').split('/')[-1]}@{e.revision}",
            "lr": f"{e.lr:g}", "steps": e.steps,
            "damage": e.damage, "gain": e.gain,
            "source": e.source.split("/")[0],
        })
    profiles = [{"target": t, "rows": sorted(rows, key=lambda r: (r["lr"], r["steps"]))}
                for t, rows in sorted(by_target.items())]
    return templates.TemplateResponse(request, "recipe.html", _base_ctx(
        request, room="recipe", profiles=profiles,
        retraction_html=render.retraction_card(
            registry.render_retraction("dp-predictive-column"))))
