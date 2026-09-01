"""Face 1 — the Model MRI report.

v1 builds FROM STORED ROWS (bundle first, then workspace rows): the M0-1
acceptance is exactly "reproduce the known verdicts from stored rows without
re-measuring". Live measurement arrives through the API's job graph: missing
rows surface as the JobSpecs that would produce them.
"""

from __future__ import annotations

import json
from pathlib import Path

from mrilab.claims import scope_guards as sg
from mrilab.engine.skipcheck import existing_valid_rows
from mrilab.faces import common as kit
from mrilab.faces.schema import Report, Section, SubjectRef
from mrilab.settings import Settings
from mrilab.surfaces import loaders, placement


def find_vitals_row(settings: Settings, label: str) -> tuple[dict, Path] | None:
    """Workspace rows first (fresh measurements), bundle second (the program's
    own stored panel rows)."""
    ws = settings.rows_dir / "vitals" / label
    for p in existing_valid_rows("vitals", ws):
        return json.loads(p.read_text(encoding="utf-8")), p
    bundle_p = loaders.BUNDLE / "vitals_out" / f"vitals_{label}.json"
    if bundle_p.is_file():
        return json.loads(bundle_p.read_text(encoding="utf-8")), bundle_p
    return None


def _vitals_section(row: dict, row_path: Path) -> Section:
    prov = kit.row_provenance("vitals", [row_path], dtype="fp32")
    blocks = [kit.reading(
        "Vitals panel (frozen instruments, exact Cell L seeds)",
        {
            "C_heldout_c4_nats": row["loss"],
            "batch_hash": row.get("batch_hash"),
            "dL_abs (ABS gamma=5e-3, curvature face)": row.get("dL_abs"),
            "S_wt (REL rho=0.10, curvature x weight-scale)": row.get("S_wt"),
            "n_matrices": row.get("n_matrices"),
            "n_params": row.get("n_params"),
        },
        body_md="fp32; the perturbation grid is calibrated against a ~2e-4-nat "
                "forward-pass noise floor.",
        provenance=prov,
    )]
    if "R" in row:
        r = row["R"]
        g_guard = sg.g_ratio_display(r["gain_seen"],
                                     seen_ci_low=r.get("gain_seen_ci", [None])[0])
        data = {
            "R_unseen_gain_nats (PRIMARY)": r["gain_unseen"],
            "unseen_ci": str(r.get("gain_unseen_ci")),
            "seen_gain_nats": r["gain_seen"],
            "k": r.get("k"), "n_items": r.get("n"),
        }
        if g_guard:
            data["G_icl (secondary)"] = r.get("G_icl")
        blocks.append(kit.reading("Readiness gauge (dialect unseen gain @k)",
                                  data, provenance=prov))
        if not g_guard:
            blocks.append(kit.guard_gap(g_guard, "G ratio display"))
    if "S_grad" in row:
        blocks.append(kit.reading(
            "S_grad (gradient face) — regime-flagged",
            {"curvature": row["S_grad"].get("curvature"),
             "first_order": row["S_grad"].get("first_order"),
             "rho": row["S_grad"].get("rho")},
            body_md="Read at rho=0.10: directional probes need ~300x smaller "
                    "rho (see claim sgrad-regime-flag) — treat as saturated "
                    "unless re-measured at rho_dir=3e-4.",
            provenance=prov))
        blocks.append(kit.claim("sgrad-regime-flag"))
    return Section(id="vitals", title="Vitals", blocks=blocks)


def _readiness_section(loss: float) -> Section:
    pl = placement.readiness_placement(loss)
    naive = sg.readiness_naive_substrate(payload_trained=False)
    blocks = [
        kit.placement(
            "Position on the readiness surface",
            {"loss": pl.loss, "verdict": pl.verdict,
             "expected_write_fate": pl.expected_sign,
             "neighbors": [n.__dict__ for n in pl.neighbors]},
            body_md=pl.band_note,
            surface_ids=["icl_readiness_uniform_*"],
            claim_ids=["readiness-crossing"]),
        kit.claim("readiness-crossing"),
        kit.claim("unseen-gain-primary"),
        kit.claim("maturity-residual"),
    ]
    if not naive:
        blocks.insert(0, kit.guard_gap(naive, "readiness gauge"))
    blocks.append(kit.gap(
        "Semantic depth (D4) and Choice payloads (Cell Q)",
        "The readiness law rests on five payload classes plus a null that "
        "never fires; SEMANTIC depth is the one open axis, and whether Choice "
        "(values/safety) payloads obey the law is untested.",
        unlocked_by="semantic-depth axis cell; CELL Q (Modal)"))
    return Section(id="readiness", title="Readiness (what a write becomes here)",
                   blocks=blocks)


def _quantization_section(dl_abs: float | None) -> Section:
    blocks: list = []
    guard = sg.dl_abs_channel_guard("quantization")
    assert guard  # licensed by construction here
    if dl_abs is not None:
        pct = placement.dl_abs_percentile(dl_abs)
        rank = placement.int4_rank(dl_abs)
        blocks.append(kit.placement(
            "dL_abs among the stored sensitivity rows",
            {"dl_abs": pct.value, "percentile": round(pct.percentile, 1),
             "n_rows": pct.n_rows,
             "below": pct.below.__dict__ if pct.below else None,
             "above": pct.above.__dict__ if pct.above else None},
            surface_ids=["sens_ladder_*"]))
        blocks.append(kit.placement(
            "Rank among the measured dL_abs -> int4 pricing pairs",
            {"rank_of": rank.rank_of, "n_pairs": rank.n_pairs,
             "below": rank.below.__dict__ if rank.below else None,
             "above": rank.above.__dict__ if rank.above else None},
            body_md=rank.note,
            surface_ids=["zter_out", "zterb_out_olmo"],
            claim_ids=["dl-abs-prices-int4"]))
    blocks += [
        kit.claim("dl-abs-prices-int4"),
        kit.claim("anneal-costs-int4-fragility"),
        kit.advisory(
            "Rejuvenate before you quantize (n=1 pair; RTN-simulated)",
            "A brief fresh-data re-warm halved int4 RTN damage on an "
            "over-annealed release at ~0.003% pretraining cost. Wording stays "
            "RTN-simulated until Z-ter-b v2 (nf4/GPTQ) lands.",
            claim_ids=["rejuvenate-before-quantize", "quenched-endpoint"]),
    ]
    return Section(id="quantization",
                   title="Quantization channel (the one dL_abs prices)",
                   blocks=blocks)


def _finetuning_section(settings: Settings, label: str) -> Section:
    denied = sg.dl_abs_channel_guard("fine-tuning")
    blocks = [
        kit.guard_gap(denied, "static-gauge commentary on fine-tuning"),
        kit.claim("no-static-gauge-predicts-training"),
        kit.claim("crash-test-ranks"),
        kit.claim("curvature-provenance"),
        kit.claim("local-geometry-vs-equilibrium"),
    ]
    crash_dir = settings.rows_dir / "erosion_arms" / label
    rows = existing_valid_rows("erosion_arms", crash_dir)
    if rows:
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in rows]
        for doc, p in zip(docs, rows):
            cfg = doc["config"]
            comp = placement.crash_comparison(doc["damage"], doc["gain"],
                                              lr=cfg["lr"], steps=cfg["steps"],
                                              target=cfg.get("target"))
            blocks.append(kit.placement(
                f"Crash test @ lr={cfg['lr']}, {cfg['steps']} steps, "
                f"target={cfg.get('target')}",
                {"damage": doc["damage"], "gain": doc["gain"],
                 "target_base": doc["target_base"],
                 "matched_stored_arms": comp.matched_arms},
                body_md=comp.note,
                surface_ids=["erosion2", "erosionW"]))
    else:
        blocks.append(kit.gap(
            "No crash test measured for this checkpoint",
            "Fine-tuning fragility is measured, never predicted: run the "
            "preflight face (3 probes at your horizon, ~4-13 min/arm at "
            "160m-1.4b scale).",
            unlocked_by="POST /api/v1/reports/preflight"))
    return Section(id="fine-tuning",
                   title="Fine-tuning channel (crash test only)", blocks=blocks)


def _posttraining_section() -> Section:
    return Section(id="post-training", title="Post-training fingerprint", blocks=[
        kit.claim("sft-double-ratchet"),
        kit.claim("post-training-readiness-cost"),
        kit.claim("sycophancy-sft-signature"),
        kit.gap("Stage ladder not measured for this subject",
                "Stiffness/curvature/readiness-cost per release stage requires "
                "the family's stage checkpoints (base/SFT/DPO/Instruct).",
                unlocked_by="vitals + readiness jobs on the stage ladder"),
    ])


def _honesty_section() -> Section:
    return Section(id="objects", title="Objects: truth, honesty, batteries", blocks=[
        kit.claim("truth-decomposition"),
        kit.claim("format-share-controls"),
        kit.claim("gemma-format-boundary"),
    ])


def _integrity_section() -> Section:
    return Section(id="integrity", title="Retractions and standing predictions", blocks=[
        kit.retraction("semantic-transfer-headline"),
        kit.retraction("iso-damage-lr-rule"),
        kit.retraction("dp-predictive-column"),
        kit.claim("transport-calculus"),
        kit.claim("forbid-static-gauge-prediction"),
        kit.claim("forbid-coordinate-transport"),
    ])


def build_from_stored(settings: Settings, label: str,
                      repo: str | None = None,
                      revision: str | None = None) -> Report:
    found = find_vitals_row(settings, label)
    if found is None:
        raise FileNotFoundError(
            f"no stored vitals row for '{label}' — submit a vitals job first "
            f"(POST /api/v1/jobs) or pick from: {loaders.vitals_labels()}")
    row, row_path = found
    subject = SubjectRef(repo=repo or row.get("repo"),
                         revision=revision or row.get("revision"),
                         path=row.get("path"), label=label)
    sections = [
        _vitals_section(row, row_path),
        _readiness_section(row["loss"]),
        _quantization_section(row.get("dL_abs")),
        _finetuning_section(settings, label),
        _posttraining_section(),
        _honesty_section(),
        _integrity_section(),
    ]
    return kit.new_report("model_report", subject, sections)
