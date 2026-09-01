"""Face 2 — the fine-tune preflight ("MRI before surgery").

Input: subject + intent (target domain, training horizon in steps, required
persistence). Output: readiness verdict, the 3-probe plan AT THE DECLARED
HORIZON (probes license only their own horizon — U2-ter), the LR pick when
probe rows exist (2-D rule: damage cap + gain rank), collateral comparisons,
and claim-gated channel/epoch/KD advisories. Trap lint runs over the intended
fine-tune parameters.
"""

from __future__ import annotations

import json

from mrilab.claims import scope_guards as sg
from mrilab.engine.jobs import JobSpec, Subject
from mrilab.engine.skipcheck import existing_valid_rows
from mrilab.faces import common as kit
from mrilab.faces.mri_report import find_vitals_row
from mrilab.faces.schema import Report, Section, SubjectRef
from mrilab.guards.traps import lint_job
from mrilab.settings import Settings
from mrilab.surfaces import placement

PROBE_LRS = (1e-5, 3e-5, 1e-4)  # the U2 probe triplet
DEFAULT_DAMAGE_CAP = 0.05       # U2's declared probe damage bound (held-out nats)


def probe_specs(subject: Subject, target: str, horizon_steps: int,
                lrs: tuple[float, ...] = PROBE_LRS) -> list[JobSpec]:
    """The three crash probes AT THE DEPLOYMENT HORIZON."""
    return [
        JobSpec(instrument="erosion_arms", subject=subject, dtype="fp32",
                params={"target": target, "lr": lr, "steps": horizon_steps,
                        "warmup": max(1, horizon_steps // 5),
                        "eval_every": max(1, horizon_steps // 4)})
        for lr in lrs
    ]


def _existing_probe_rows(settings: Settings, subject: Subject, target: str,
                         horizon_steps: int) -> list[dict]:
    out = []
    for p in existing_valid_rows("erosion_arms",
                                 settings.rows_dir / "erosion_arms" / subject.label):
        doc = json.loads(p.read_text(encoding="utf-8"))
        cfg = doc.get("config", {})
        if cfg.get("target") == target and cfg.get("steps") == horizon_steps:
            doc["_path"] = str(p)
            out.append(doc)
    return out


def select_lr(probe_rows: list[dict], damage_cap: float = DEFAULT_DAMAGE_CAP
              ) -> dict | None:
    """The 2-D rule: among probes with damage <= cap, pick max gain; if none
    clears the cap, pick min damage and say so. Requires matched target_base
    (trap 7) — enforced by the caller's trap block."""
    if not probe_rows:
        return None
    ok = [r for r in probe_rows if r["damage"] <= damage_cap]
    if ok:
        best = max(ok, key=lambda r: r["gain"])
        rule = f"max gain subject to damage <= {damage_cap}"
    else:
        best = min(probe_rows, key=lambda r: r["damage"])
        rule = f"no probe cleared damage <= {damage_cap}: min-damage fallback"
    return {"lr": best["config"]["lr"], "rule": rule,
            "damage": best["damage"], "gain": best["gain"],
            "n_probes": len(probe_rows), "source": best.get("_path")}


def build(settings: Settings, subject: Subject, target: str,
          horizon_steps: int, persistence: str = "durable",
          pool_chunks: int | None = None,
          loss: float | None = None) -> Report:
    ref = SubjectRef(repo=subject.repo, revision=subject.revision,
                     path=subject.path, label=subject.label)

    # --- readiness -------------------------------------------------------------
    if loss is None:
        found = find_vitals_row(settings, subject.label)
        loss = found[0]["loss"] if found else None
    readiness_blocks: list = []
    if loss is not None:
        pl = placement.readiness_placement(loss)
        readiness_blocks += [
            kit.placement(
                "Will your data become a rule or items here?",
                {"loss": loss, "verdict": pl.verdict,
                 "expected_write_fate": pl.expected_sign,
                 "neighbors": [n.__dict__ for n in pl.neighbors]},
                body_md=pl.band_note, surface_ids=["icl_readiness_uniform_*"],
                claim_ids=["readiness-crossing"]),
            kit.claim("readiness-crossing"),
        ]
        if pl.verdict == "above-crossing":
            readiness_blocks.append(kit.claim("early-writes-anti-generalize"))
    else:
        readiness_blocks.append(kit.gap(
            "No capability reading for this checkpoint",
            "Readiness placement needs held-out loss; run a vitals job first.",
            unlocked_by="POST /api/v1/jobs {instrument: vitals}"))
    sections = [Section(id="readiness", title="Readiness verdict",
                        blocks=readiness_blocks)]

    # --- LR probes at the declared horizon --------------------------------------
    hguard = sg.horizon_match(horizon_steps, horizon_steps)
    assert hguard
    specs = probe_specs(subject, target, horizon_steps)
    probe_rows = _existing_probe_rows(settings, subject, target, horizon_steps)
    lr_blocks: list = [
        kit.claim("probe-at-your-horizon"),
        kit.claim("no-static-gauge-predicts-training"),
    ]
    if probe_rows:
        from mrilab.guards.traps import check_target_base_match
        tb = check_target_base_match(probe_rows)
        if tb:
            lr_blocks.append(kit.trap_block(tb))
        pick = select_lr(probe_rows)
        lr_blocks.append(kit.reading(
            "LR selection (2-D rule: damage cap + gain rank)",
            pick, body_md="Selected from measured probes at YOUR horizon; the "
                          "grid-parity claim carries the recorded circularity "
                          "caveat until U2-quater's new-domain cell lands.",
            provenance=kit.row_provenance(
                "erosion_arms",
                [settings.rows_dir / "erosion_arms" / subject.label],
                dtype="fp32")))
        comp = placement.crash_comparison(pick["damage"], pick["gain"],
                                          lr=pick["lr"], steps=horizon_steps,
                                          target=target)
        lr_blocks.append(kit.placement(
            "Your probe among the stored arms (rank only)",
            {"matched_arms": comp.matched_arms}, body_md=comp.note,
            surface_ids=["erosion2", "u2_out"]))
    else:
        lr_blocks.append(kit.gap(
            "Probes not yet run",
            f"Plan: 3 crash probes at lr={list(PROBE_LRS)} for "
            f"{horizon_steps} steps on target '{target}' "
            f"(~4 min/arm at 160m, ~13 min/arm at 1.4b — measured timings).",
            unlocked_by="submit the probe JobSpecs below"))
        lr_blocks.append(kit.reading(
            "Probe JobSpecs (submit via POST /api/v1/jobs)",
            {"specs": [s.model_dump() for s in specs]},
        ))
    sections.append(Section(id="lr", title="Learning-rate preflight", blocks=lr_blocks))

    # --- collateral --------------------------------------------------------------
    sections.append(Section(id="collateral", title="Collateral", blocks=[
        kit.claim("redistribution-hidden"),
        kit.claim("curvature-provenance"),
        kit.advisory(
            "Measure the redistribution vector, not a scalar",
            "Evaluate held-out loss over the stored domain loaders "
            "(c4/code/legal/math) before and after — a single-domain gate hid "
            "+0.44 nats once.",
            claim_ids=["redistribution-hidden"]),
    ]))

    # --- recipe advisories (all claim-gated) --------------------------------------
    advisory_blocks = [
        kit.claim("epoch-safety"),
        kit.advisory(
            "Small pool? Consider pure KD (n=1 config)",
            "At matched everything, pure KD resisted small-pool memorization "
            "collapse 15-35x vs CE — but was net harmful with a ~0.4-nat "
            "teacher lead. Single configuration; labeled accordingly.",
            claim_ids=["kd-regulariser"]),
    ]
    if persistence in ("session", "per-session"):
        advisory_blocks += [
            kit.claim("exchange-table"),
            kit.advisory(
                "Session-only persistence: you may be overpaying ~10^4 per token",
                "Behaviours needed only per-session are purchasable in the "
                "context channel at ~0.23 nats/token vs training's ~1.4e-5 "
                "(order-of-magnitude claim, one disposition, one family).",
                claim_ids=["exchange-table", "attacker-defender-asymmetry"]),
        ]
    # trap lint over the intended fine-tune
    lint_params = {"kind": "training", "lr": None, "steps": horizon_steps,
                   "batch_size": 4, "accum": 8}
    if pool_chunks:
        lint_params["pool_chunks"] = pool_chunks
    for w in lint_job(lint_params):
        advisory_blocks.append(kit.trap_block(w))
    sections.append(Section(id="recipe", title="Recipe advisories", blocks=advisory_blocks))

    return kit.new_report("preflight", ref, sections)
