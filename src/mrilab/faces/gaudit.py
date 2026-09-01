"""Face 5 — memorization audit (v1): validates a user family-split spec and
renders the protocol + the claims that license it. Custom-payload execution
(compiling a user split into icl_ladder rungs) is post-v1 and is declared as
a Gap, not faked."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from mrilab.faces import common as kit
from mrilab.faces.schema import Report, Section, SubjectRef


class FamilySplitSpec(BaseModel):
    name: str
    seen: list[str]
    unseen: list[str]
    description: str = ""

    @field_validator("unseen")
    @classmethod
    def _disjoint(cls, v, info):
        seen = set(info.data.get("seen", []))
        overlap = seen & set(v)
        if overlap:
            raise ValueError(f"seen/unseen overlap: {sorted(overlap)[:5]}")
        if len(v) < 4:
            raise ValueError("need >=4 unseen members for a CI worth reading")
        return v


def build(spec: FamilySplitSpec, subject_label: str = "benchmark") -> Report:
    subject = SubjectRef(label=f"{subject_label}:{spec.name}")
    sections = [
        Section(id="protocol", title="Protocol (measured instrument)", blocks=[
            kit.reading("Family split (validated)", {
                "name": spec.name, "n_seen": len(spec.seen),
                "n_unseen": len(spec.unseen)}),
            kit.claim("readiness-by-inference"),
            kit.claim("unseen-gain-primary"),
            kit.claim("readiness-null-calibration"),
            kit.claim("format-share-controls"),
        ]),
        Section(id="execution", title="Execution", blocks=[
            kit.gap("Custom-payload execution is post-v1",
                    "Compiling an arbitrary family split into the payload-"
                    "ladder machinery (item frames, format twin, G1 gates) is "
                    "the next face increment; the protocol above is what will "
                    "run, unchanged.",
                    unlocked_by="payload compiler (faces/gaudit v2)"),
        ]),
    ]
    return kit.new_report("memorization_audit", subject, sections)
