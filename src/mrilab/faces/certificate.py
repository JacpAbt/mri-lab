"""Face 4 — release certificate (v1): the two-column audit rendered from
stored rows + claims. Readiness-cost x tamper channels, quench state,
rejuvenation advisory — every number carries its n; quantization wording is
RTN-simulated until Z-ter-b v2."""

from __future__ import annotations

from mrilab.faces import common as kit
from mrilab.faces.mri_report import (_quantization_section, find_vitals_row)
from mrilab.faces.schema import Report, Section, SubjectRef
from mrilab.settings import Settings


def build(settings: Settings, label: str) -> Report:
    found = find_vitals_row(settings, label)
    row = found[0] if found else {}
    subject = SubjectRef(repo=row.get("repo"), revision=row.get("revision"),
                         path=row.get("path"), label=label)
    sections = [
        Section(id="columns", title="Two-column audit", blocks=[
            kit.claim("post-training-readiness-cost"),
            kit.claim("sft-double-ratchet"),
            kit.claim("sycophancy-sft-signature"),
        ]),
        _quantization_section(row.get("dL_abs")),
        Section(id="tamper", title="Tamper channels", blocks=[
            kit.claim("no-static-gauge-predicts-training"),
            kit.claim("crash-test-ranks"),
            kit.claim("rising-susceptibility"),
            kit.claim("exchange-table"),
            kit.claim("attacker-defender-asymmetry"),
        ]),
        Section(id="quench", title="Quench state", blocks=[
            kit.claim("anneal-freezes-coordinates-not-curvature"),
            kit.claim("quenched-endpoint"),
            kit.claim("anneal-costs-int4-fragility"),
            kit.gap("Ship-the-pair advisory (untested)",
                    "Releasing the pre-anneal checkpoint alongside the annealed "
                    "endpoint is an UNTESTED recommendation (IDEAS-24) and is "
                    "listed as such, not advised.",
                    unlocked_by="IDEAS-24 cell (fork-release fine-tune pair)"),
        ]),
        Section(id="integrity", title="Retractions carried on this certificate",
                blocks=[
                    kit.retraction("iso-damage-lr-rule"),
                    kit.retraction("dp-predictive-column"),
                    kit.claim("transport-calculus"),
                ]),
    ]
    return kit.new_report("release_cert", subject, sections)
