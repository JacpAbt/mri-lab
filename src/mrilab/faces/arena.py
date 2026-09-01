"""Face 6 — the standing-predictions arena (v1): §42d's forbidden list rendered
from the registry, with the ground-truth protocol. Register/resolve lives in
the API; a resolution that falsifies a prediction localizes which law was
wrong — that is the theory earning its keep, rendered as such."""

from __future__ import annotations

from mrilab.faces import common as kit
from mrilab.faces.schema import Report, Section, SubjectRef

STANDING = [
    "forbid-static-gauge-prediction",
    "forbid-posttraining-structure",
    "forbid-arch-endpoint",
    "forbid-coordinate-transport",
]


def build() -> Report:
    subject = SubjectRef(label="standing-predictions")
    sections = [
        Section(id="predictions", title="Standing predictions (§42d)", blocks=[
            *[kit.claim(cid) for cid in STANDING],
        ]),
        Section(id="protocol", title="Ground truth & scoring", blocks=[
            kit.claim("crash-test-ranks"),
            kit.reading("Challenge protocol", {
                "ground_truth": "crash test at the deployment horizon "
                                "(damage, gain, target_base at declared "
                                "reference LR/steps)",
                "stored_rows": "surfaces bundle: erosion2/W/ZB, eq_out, u2*",
                "submission": "a static gauge g(weights) -> predicted damage "
                              "rank; scored by Spearman against measured rank "
                              "on held-out arms",
                "falsification": "a gauge beating the crash test's own "
                                 "100-vs-200-step rank agreement (.976) "
                                 "falsifies claim forbid-static-gauge-"
                                 "prediction and localizes L4",
            }),
            kit.claim("transport-calculus"),
        ]),
    ]
    return kit.new_report("arena", subject, sections)
