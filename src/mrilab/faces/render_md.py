"""The ONE renderer: walks the report JSON and emits markdown. The future
frontend re-renders from the same `data` payloads — there is no parallel
builder path (plan P6 commitment c)."""

from __future__ import annotations

from mrilab.faces.schema import Report

_KIND_BADGE = {
    "reading": "READING",
    "surface_placement": "PLACEMENT",
    "claim_verdict": "CLAIM",
    "retraction": "RETRACTED",
    "advisory": "ADVISORY",
    "gap": "GAP",
    "trap_warning": "TRAP",
}


def _fmt_value(v) -> str:
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.005:
            return f"{v:g}"  # small magnitudes (LRs, floors) keep their form
        return f"{v:+.4f}" if abs(v) < 100 else f"{v:.4g}"
    return str(v)


def render(report: Report) -> str:
    lines: list[str] = []
    s = report.subject
    lines.append(f"# {report.face.replace('_', ' ').title()} — {s.label}")
    lines.append("")
    lines.append(f"- report `{report.report_id}` · schema v{report.schema_version} "
                 f"· registry `{report.registry_version}`")
    if s.resolved_sha:
        lines.append(f"- checkpoint resolved sha `{s.resolved_sha}`")
    lines.append(f"- created {report.created_at.isoformat(timespec='seconds')}")
    counts = report.verdict_summary.get("block_counts", {})
    if counts:
        lines.append("- blocks: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines.append("")

    for sec in report.sections:
        lines.append(f"## {sec.title}")
        lines.append("")
        for b in sec.blocks:
            badge = _KIND_BADGE.get(b.kind, b.kind.upper())
            sev = "" if b.severity == "info" else f" [{b.severity.upper()}]"
            lines.append(f"### [{badge}]{sev} {b.title}")
            if b.body_md:
                lines.append(b.body_md.strip())
            if b.kind == "reading" and b.data:
                lines.append("")
                for k, v in b.data.items():
                    if isinstance(v, (int, float, str)):
                        lines.append(f"- {k}: {_fmt_value(v)}")
            if b.kind == "claim_verdict":
                d = b.data
                lines.append("")
                lines.append(f"- evidence: **{d['evidence_class']}** · status "
                             f"{d['status']} · n: {d['n']}")
                lines.append(f"- bounds: {d['bounds']}")
                lines.append(f"- source: {d['source']}")
                if not d.get("verbatim_verified", True):
                    lines.append("- note: bounds transcription pending verbatim "
                                 "verification against the prereg scored block")
            if b.kind == "retraction":
                d = b.data
                lines.append("")
                lines.append(f"- status {d['status']} · n: {d['n']}")
                lines.append(f"- bounds: {d['bounds']}")
                if d.get("superseded_by"):
                    lines.append(f"- superseded by: `{d['superseded_by']}`")
            if b.kind == "gap" and b.data.get("unlocked_by"):
                lines.append(f"- unlocked by: {b.data['unlocked_by']}")
            prov = b.provenance
            bits = []
            if prov.instrument:
                bits.append(f"instrument {prov.instrument}")
            if prov.dtype:
                bits.append(f"dtype {prov.dtype}")
            if prov.row_paths:
                bits.append(f"{len(prov.row_paths)} source row(s)")
            if bits:
                lines.append(f"- provenance: {'; '.join(bits)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
