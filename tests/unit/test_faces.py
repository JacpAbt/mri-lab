from __future__ import annotations

import pytest

from mrilab.engine.jobs import Subject
from mrilab.faces import arena, common as kit, mri_report, preflight, render_md
from mrilab.faces.gaudit import FamilySplitSpec, build as build_gaudit
from mrilab.settings import Settings


@pytest.fixture()
def settings(tmp_path) -> Settings:
    # empty tmp workspace: builders find no workspace rows and fall back to the
    # committed bundle; nothing here touches a GPU or writes outside tmp
    return Settings(workspace=str(tmp_path / "ws"))


def test_model_report_from_bundle(settings) -> None:
    rep = mri_report.build_from_stored(settings, "pythia-160m@step143000")
    ids = [s.id for s in rep.sections]
    assert ids == ["vitals", "readiness", "quantization", "fine-tuning",
                   "post-training", "objects", "integrity"]
    # every claim block carries class + n + bounds (the discipline, structurally)
    n_claims = 0
    for sec in rep.sections:
        for b in sec.blocks:
            if b.kind == "claim_verdict":
                n_claims += 1
                assert b.data["evidence_class"] in ("measured", "lit", "conjecture")
                assert b.data["n"] and b.data["bounds"] and b.data["source"]
    assert n_claims >= 12
    # the readiness verdict for this checkpoint is the known one
    readiness = next(s for s in rep.sections if s.id == "readiness")
    pl = next(b for b in readiness.blocks if b.kind == "surface_placement")
    assert pl.data["verdict"] == "below-crossing"
    # retractions are rendered (misses as prominent as hits)
    integrity = next(s for s in rep.sections if s.id == "integrity")
    assert any(b.kind == "retraction" for b in integrity.blocks)
    # the fine-tuning section must contain the guard REFUSAL, not a forecast
    ft = next(s for s in rep.sections if s.id == "fine-tuning")
    assert any(b.kind == "gap" and "crash test" in b.body_md for b in ft.blocks)


def test_model_report_unknown_label_names_the_fix(settings) -> None:
    with pytest.raises(FileNotFoundError, match="vitals job"):
        mri_report.build_from_stored(settings, "no-such-model@main")


def test_preflight_without_probes_emits_plan(settings) -> None:
    rep = preflight.build(settings,
                          Subject(repo="EleutherAI/pythia-410m", revision="step12000"),
                          target="legal", horizon_steps=200,
                          persistence="session", loss=3.363)
    lr = next(s for s in rep.sections if s.id == "lr")
    specs_block = next(b for b in lr.blocks if b.kind == "reading"
                       and "JobSpecs" in b.title)
    specs = specs_block.data["specs"]
    assert len(specs) == 3
    assert all(s["params"]["steps"] == 200 for s in specs)  # horizon-matched
    assert {s["params"]["lr"] for s in specs} == {1e-5, 3e-5, 1e-4}
    # session persistence -> the channel-economics advisory is present
    recipe = next(s for s in rep.sections if s.id == "recipe")
    assert any("10^4" in b.title or "overpaying" in b.title for b in recipe.blocks
               if b.kind == "advisory")
    # readiness verdict from the provided loss
    readiness = next(s for s in rep.sections if s.id == "readiness")
    pl = next(b for b in readiness.blocks if b.kind == "surface_placement")
    assert pl.data["verdict"] == "below-crossing"


def test_preflight_lr_selection_2d_rule() -> None:
    rows = [
        {"config": {"lr": 1e-5}, "damage": 0.01, "gain": 0.05, "target_base": 1.0},
        {"config": {"lr": 3e-5}, "damage": 0.04, "gain": 0.09, "target_base": 1.0},
        {"config": {"lr": 1e-4}, "damage": 0.30, "gain": 0.12, "target_base": 1.0},
    ]
    pick = preflight.select_lr(rows, cap=0.05)
    assert pick["lr"] == 3e-5  # max gain under the damage cap, NOT max gain overall
    # nothing clears the cap -> min damage, and the rule says so
    rows_hot = [{"config": {"lr": lr}, "damage": d, "gain": g, "target_base": 1.0}
                for lr, d, g in ((1e-5, 0.08, 0.02), (3e-5, 0.2, 0.1))]
    pick2 = preflight.select_lr(rows_hot, cap=0.05)
    assert pick2["lr"] == 1e-5 and "fallback" in pick2["rule"]
    # the cap scales with the declared horizon (0.05 per 100 steps)
    assert preflight.damage_cap(100) == 0.05
    assert preflight.damage_cap(200) == 0.10


def test_probe_specs_pin_the_pool() -> None:
    # trap #3 regression: the instrument default (512 chunks) is 12.5 epochs at
    # 200x4x8 draws — probe specs must pin n_train to the stored-grid protocol
    specs = preflight.probe_specs(
        Subject(repo="EleutherAI/pythia-410m", revision="step12000"),
        target="legal", horizon_steps=200)
    for s in specs:
        assert s.params["n_train"] == preflight.PROBE_N_TRAIN
        draws = s.params["steps"] * 4 * 8
        assert draws / s.params["n_train"] < 2.0  # X-bis: ~2 epochs safe


def test_lint_sees_effective_params() -> None:
    # an erosion job WITHOUT n_train must still trip the epoch trap via the
    # instrument's own default (512), not sail through on an undeclared pool
    from mrilab.engine.jobs import JobSpec
    from mrilab.engine.queue import _lint_params
    from mrilab.guards.traps import lint_job
    spec = JobSpec(instrument="erosion_arms",
                   subject=Subject(repo="x/y", revision="z"), dtype="fp32",
                   params={"target": "legal", "lr": 1e-5, "steps": 200})
    warnings = lint_job(_lint_params(spec))
    assert any(w.trap == "epoch_trap" and w.severity == "block" for w in warnings)


def test_advisories_must_be_claim_gated() -> None:
    with pytest.raises(ValueError, match="claim-gated"):
        kit.advisory("free advice", "just trust me", claim_ids=[])
    with pytest.raises(Exception):
        kit.advisory("bad claim", "x", claim_ids=["semantic-transfer-headline"])


def test_gaudit_spec_validation() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FamilySplitSpec(name="x", seen=["a", "b"], unseen=["b", "c", "d", "e"])
    with pytest.raises(ValueError, match=">=4"):
        FamilySplitSpec(name="x", seen=["a"], unseen=["b", "c"])
    rep = build_gaudit(FamilySplitSpec(name="units", seen=["m", "kg"],
                                       unseen=["s", "A", "K", "mol"]))
    assert rep.face == "memorization_audit"


def test_arena_and_markdown_render(settings) -> None:
    rep = arena.build()
    md = render_md.render(rep)
    assert "Standing predictions" in md
    assert "conjecture" in md  # evidence class rendered
    rep2 = mri_report.build_from_stored(settings, "pythia-160m@step143000")
    md2 = render_md.render(rep2)
    assert "bounds:" in md2 and "RETRACTED" in md2
    assert "3.3796" in md2 or "3.380" in md2  # the loss reading is in the text
