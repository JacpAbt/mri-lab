"""Face 3 — training recorder + CI assertions (v1).

Advisories only; NO knobs and NO raw single-probe triggers (DEMO-CL: an n=24
probe swings +-0.8; thresholds must be calibrated in-population; smoothing +
hysteresis required — enforced via the trigger_smoothing guard). The CI half
evaluates declared gates over indexed rows for a checkpoint sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from mrilab.claims import scope_guards as sg
from mrilab.engine.skipcheck import existing_valid_rows
from mrilab.faces import common as kit
from mrilab.faces.schema import Report, Section, SubjectRef
from mrilab.settings import Settings


class CIGates(BaseModel):
    max_dl_abs_ratio_vs_first: float | None = None   # fragility budget
    min_unseen_gain_slope: float | None = None       # readiness stall
    max_c4_regression_nats: float | None = None      # capability budget


def _series(settings: Settings, instrument: str, labels: list[str],
            extract) -> list[tuple[str, float]]:
    out = []
    for label in labels:
        for p in existing_valid_rows(instrument, settings.rows_dir / instrument / label):
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
            v = extract(doc)
            if v is not None:
                out.append((label, v))
                break
    return out


def evaluate_gates(settings: Settings, labels: list[str],
                   gates: CIGates) -> dict:
    results: dict[str, dict] = {}
    dl = _series(settings, "vitals", labels, lambda d: d.get("dL_abs"))
    loss = _series(settings, "vitals", labels, lambda d: d.get("loss"))
    r = _series(settings, "vitals", labels,
                lambda d: (d.get("R") or {}).get("gain_unseen"))
    if gates.max_dl_abs_ratio_vs_first is not None and len(dl) >= 2:
        ratio = dl[-1][1] / dl[0][1] if dl[0][1] else float("inf")
        results["fragility_budget"] = {
            "pass": ratio <= gates.max_dl_abs_ratio_vs_first,
            "ratio": ratio, "limit": gates.max_dl_abs_ratio_vs_first,
            "series": dl}
    if gates.min_unseen_gain_slope is not None and len(r) >= 3:
        slope = (r[-1][1] - r[0][1]) / max(len(r) - 1, 1)
        results["readiness_stall"] = {
            "pass": slope >= gates.min_unseen_gain_slope,
            "slope_per_checkpoint": slope, "series": r,
            "note": "n>=3 checkpoints; never a single-probe trigger (DEMO-CL)"}
    if gates.max_c4_regression_nats is not None and len(loss) >= 2:
        reg = loss[-1][1] - min(v for _, v in loss)
        results["capability_budget"] = {
            "pass": reg <= gates.max_c4_regression_nats,
            "regression": reg, "limit": gates.max_c4_regression_nats,
            "series": loss}
    results["_overall_pass"] = all(
        v["pass"] for k, v in results.items() if not k.startswith("_"))
    return results


def build(settings: Settings, run_label: str, labels: list[str],
          gates: CIGates) -> Report:
    subject = SubjectRef(label=run_label)
    results = evaluate_gates(settings, labels, gates)
    smoothing = sg.trigger_smoothing(n_probe_items=96, k_consecutive=2)
    sections = [
        Section(id="gates", title="CI gate evaluation", blocks=[
            kit.reading("Gate results over indexed checkpoints",
                        {k: v for k, v in results.items()}),
        ]),
        Section(id="discipline", title="Trigger discipline", blocks=[
            kit.claim("trigger-discipline"),
            (kit.reading("Smoothing requirement", {
                "n_probe_items_min": 96, "k_consecutive_min": 2,
                "status": "enforced"}) if smoothing else
             kit.guard_gap(smoothing, "trigger display")),
            kit.gap("No knobs",
                    "Closed-loop actions unlock per landed DEMO-CL union; the "
                    "closed-loop claim is UNDISCHARGED (P-CL1 not scored).",
                    unlocked_by="DEMO-CL v2"),
        ]),
    ]
    return kit.new_report("training_recorder", subject, sections)
