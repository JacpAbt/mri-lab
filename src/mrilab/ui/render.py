"""Block -> HTML fragments: the web view of the Report JSON.

Same contract as faces/render_md.py (the ONE-renderer principle, §3 of the
constitution): walk sections[].blocks[] by kind; unknown kinds degrade to
title + body. Evidence chrome is structural — claim cards always carry class +
n + bounds; refusals and retractions render as first-class furniture, never
disappear. All text is escaped; block data is machine truth shown honestly.
"""

from __future__ import annotations

import html
import json

from mrilab.faces.schema import Report, Section


def esc(v: object) -> str:
    return html.escape(str(v))


def fmt_value(v: object) -> str:
    if v is None:
        return "&#8212;"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        # tiny magnitudes keep significant digits (the lr=+0.0000 lesson)
        return f"{v:.3g}" if 0 < abs(v) < 0.005 else f"{v:.4f}".rstrip("0").rstrip(".")
    return esc(v)


def _details_json(v: object, summary: str = "raw") -> str:
    return (f'<details class="raw"><summary>{esc(summary)}</summary>'
            f'<pre>{esc(json.dumps(v, indent=1, default=str))}</pre></details>')


def _rows_table(rows: list[dict]) -> str:
    """A list of homogeneous dicts (neighbors, matched arms) -> table."""
    if not rows:
        return '<p class="dim">none</p>'
    keys = list(rows[0].keys())
    head = "".join(f"<th>{esc(k)}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{fmt_value(r.get(k))}</td>" for k in keys) + "</tr>"
        for r in rows)
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def _kv_table(data: dict) -> str:
    rows = []
    for k, v in data.items():
        if isinstance(v, list) and v and all(isinstance(e, dict) for e in v):
            rows.append(f'<tr><th>{esc(k)}</th><td>{_rows_table(v)}</td></tr>')
        elif isinstance(v, (list, dict)):
            rows.append(f'<tr><th>{esc(k)}</th><td>{_details_json(v)}</td></tr>')
        else:
            rows.append(f'<tr><th>{esc(k)}</th><td class="v">{fmt_value(v)}</td></tr>')
    return f'<table class="kv">{"".join(rows)}</table>'


def _body(md: str) -> str:
    if not md:
        return ""
    paras = "".join(f"<p>{esc(p)}</p>" for p in md.split("\n\n") if p.strip())
    return f'<div class="body">{paras}</div>'


def _provenance(block) -> str:
    p = block.provenance
    bits: list[str] = []
    if p.instrument:
        bits.append(esc(p.instrument))
    if p.dtype:
        bits.append(esc(p.dtype))
    if p.row_paths:
        tail = esc(p.row_paths[0].replace("\\", "/").rsplit("/", 1)[-1])
        more = f" +{len(p.row_paths) - 1}" if len(p.row_paths) > 1 else ""
        bits.append(f"row {tail}{more}")
    if p.surface_ids:
        bits.append("surfaces " + esc(", ".join(p.surface_ids)))
    links = "".join(
        f' <a class="claimlink" href="/registry#claim-{esc(cid)}">{esc(cid)}</a>'
        for cid in p.claim_ids)
    if not bits and not links:
        return ""
    return f'<div class="prov">{" &#183; ".join(bits)}{links}</div>'


_EVIDENCE_CHIP = {"measured": "c-ok", "lit": "c-lit", "conjecture": "c-warn"}


def claim_card(d: dict) -> str:
    chips = [f'<span class="chip {_EVIDENCE_CHIP.get(d["evidence_class"], "")}">'
             f'{esc(d["evidence_class"]).upper()}</span>',
             f'<span class="chip">n: {esc(d["n"])}</span>']
    if d.get("status") not in (None, "LIVE"):
        chips.append(f'<span class="chip c-warn">{esc(d["status"])}</span>')
    if not d.get("verbatim_verified", False):
        chips.append('<span class="chip c-warn" title="bounds text pending '
                     'verbatim verification against the source document">'
                     'UNVERIFIED BOUNDS</span>')
    note = f'<p class="note">{esc(d["note"])}</p>' if d.get("note") else ""
    return (
        f'<div class="claim" id="claim-{esc(d["claim_id"])}">'
        f'<div class="claimtop"><span class="id">{esc(d["claim_id"])}</span>'
        f'<span class="chips">{"".join(chips)}</span></div>'
        f'<p class="statement">{esc(d["statement"])}</p>'
        f'<p class="bounds">{esc(d["bounds"])}</p>{note}'
        f'<div class="prov">{esc(d["source"])}</div></div>')


def retraction_card(d: dict) -> str:
    return (f'<div class="refusal" id="claim-{esc(d["claim_id"])}">'
            f'<b>{esc(d["status"])}: {esc(d["claim_id"])}</b>'
            f'<p class="statement">{esc(d["retracted_statement"])}</p>'
            f'<p class="bounds">{esc(d["bounds"])}</p>'
            f'<div class="prov">{esc(d["source"])} &#183; n: {esc(d["n"])}'
            + (f' &#183; superseded by <a class="claimlink" '
               f'href="#claim-{esc(d["superseded_by"])}">'
               f'{esc(d["superseded_by"])}</a>' if d.get("superseded_by") else "")
            + "</div></div>")


def block_html(b) -> str:
    kind = getattr(b, "kind", "?")
    if kind == "claim_verdict":
        return claim_card(b.data)
    if kind == "retraction":
        return retraction_card(b.data)
    if kind == "gap":
        unlock = b.data.get("unlocked_by")
        return (f'<div class="gapblk"><b>'
                f'{"NOT LICENSED" if b.severity == "warn" else "NOT MEASURED"}:'
                f'</b> {esc(b.title)}{_body(b.body_md)}'
                + (f'<div class="unlock">unlock: {esc(unlock)}</div>' if unlock else "")
                + "</div>")
    if kind == "trap_warning":
        return (f'<div class="refusal"><b>INSTRUMENT TRAP &#183; '
                f'{esc(b.data.get("trap", "?"))}</b>{_body(b.body_md)}</div>')
    if kind == "advisory":
        return (f'<div class="advisory"><h3>{esc(b.title)}</h3>{_body(b.body_md)}'
                f'{_kv_table(b.data) if b.data else ""}{_provenance(b)}</div>')
    # reading / surface_placement / unknown kinds
    data = getattr(b, "data", None) or {}
    return (f'<div class="blk"><h3>{esc(b.title)}</h3>{_body(b.body_md)}'
            f'{_kv_table(data) if data else ""}{_provenance(b)}</div>')


def section_html(sec: Section, extra_html: str = "") -> str:
    blocks = "".join(block_html(b) for b in sec.blocks)
    return (f'<section id="{esc(sec.id)}"><h2>{esc(sec.title)}</h2>'
            f'{extra_html}{blocks}</section>')


def sections_html(report: Report, section_ids: list[str] | None = None,
                  extra_by_id: dict[str, str] | None = None) -> str:
    extra = extra_by_id or {}
    out = []
    for sec in report.sections:
        if section_ids is not None and sec.id not in section_ids:
            continue
        out.append(section_html(sec, extra.get(sec.id, "")))
    return "".join(out)
