"""Golden-row reproduction tests (the correctness cornerstone, plan P2).

Per-field tolerance policy (declared here, tightened only with measurement):
- fp32 forward losses on identical data: |delta| <= 5e-4 (GPU nondeterministic
  reductions; the instrument's own noise floor is ~2e-4 on CPU fp32).
- seeded perturbation blocks (exact Cell L seeds): each val |delta| <= 5e-3,
  mean within 3x the stored seed-sd — the scored P-PAN1 criterion.
- readiness gains (n=96 margins): |delta| <= 0.05 and inside the stored 95% CI.
- 200-step trained damage/gain: nondeterminism COMPOUNDS through the optimizer;
  first calibration band = max(30% relative, 2e-3 absolute). Tighten after the
  first rerun measurement is recorded here.
- z_ter (CPU, deterministic given fp32): |delta| <= 2e-4 (= the noise floor).
- induction at high score: |delta| <= 5e-3 (bf16; the instrument is
  dtype-sensitive only near its floor — this row reads 0.9658).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import assert_ci_overlap, assert_close, load_expected

from mrilab.engine.executor import run_instrument

pytestmark = pytest.mark.golden

VITALS_ROWS = ["vitals_pythia-160m@step143000.json", "vitals_pythia-160m@step40000.json"]


@pytest.mark.gpu
@pytest.mark.parametrize("row_name", VITALS_ROWS)
def test_vitals_golden(settings, research_env, tmp_path: Path, row_name: str) -> None:
    want = load_expected(row_name)
    res = run_instrument(
        settings, "vitals",
        ["--repo", want["repo"], "--revision", want["revision"],
         "--device", "cuda", "--output", str(tmp_path)],
        log_name=f"golden_vitals_{want['revision']}", timeout_s=2400,
    )
    assert res.ok, f"vitals failed rc={res.rc}, log: {res.log_path}"
    got = json.loads((tmp_path / f"vitals_{want['label']}.json").read_text())

    assert got["batch_hash"] == want["batch_hash"], "C4 eval batch changed upstream"
    assert got["n_matrices"] == want["n_matrices"]
    assert got["n_params"] == want["n_params"]
    assert_close("loss", got["loss"], want["loss"], 5e-4)

    for conv, mag in (("abs", "0.005"), ("rel", "0.1")):
        w, g = want[conv][mag], got[conv][mag]
        assert_close(f"{conv}.mean", g["mean"], w["mean"], 3 * w["sd"])
        for i, (gv, wv) in enumerate(zip(g["vals"], w["vals"])):
            assert_close(f"{conv}.vals[{i}]", gv, wv, 5e-3)

    for k in ("gain_seen", "gain_unseen"):
        assert_close(f"R.{k}", got["R"][k], want["R"][k], 0.05)
        assert_ci_overlap(f"R.{k}", got["R"][k], want["R"][f"{k}_ci"], pad=0.02)

    # S_grad: gradient-direction response, larger magnitudes -> relative band
    assert_close("S_grad.curvature", got["S_grad"]["curvature"],
                 want["S_grad"]["curvature"], 0.02 * abs(want["S_grad"]["curvature"]))


@pytest.mark.gpu
def test_crash_golden(settings, research_env, seeded_pools, tmp_path: Path) -> None:
    if not seeded_pools["code_pool"]:
        pytest.skip("code pool not seeded (run scripts/seed_workspace.sh)")
    want = load_expected("erosion_step40000_lr1e-5.json")
    cfg = want["config"]
    out = tmp_path / "erosion"
    res = run_instrument(
        settings, "erosion_arms",
        ["--repo", cfg["repo"], "--revision", cfg["revision"],
         "--target", cfg["target"], "--lr", str(cfg["lr"]),
         "--steps", str(cfg["steps"]), "--warmup", str(cfg["warmup"]),
         "--eval-every", str(cfg["eval_every"]),
         "--batch-size", str(cfg["batch_size"]), "--accum", str(cfg["accum"]),
         "--n-train", str(cfg["n_train"]), "--seed", str(cfg["seed"]),
         "--output", str(out)],
        log_name="golden_crash_step40000_lr1e-5", timeout_s=3600,
    )
    assert res.ok, f"erosion_arms failed rc={res.rc}, log: {res.log_path}"
    got = json.loads((out / "erosion.json").read_text())

    # identical pool + identical eval batch -> bases must match tightly
    assert_close("c4_base", got["c4_base"], want["c4_base"], 5e-4)
    assert_close("target_base", got["target_base"], want["target_base"], 5e-4)
    # trained outcomes: declared first-calibration band (see module docstring)
    for k in ("damage", "gain"):
        band = max(0.30 * abs(want[k]), 2e-3)
        assert_close(k, got[k], want[k], band)


def test_zter_golden(settings, research_env, seeded_pools, tmp_path: Path) -> None:
    """CPU lane (no gpu marker) but heavy (~25 min) and research-venv-bound,
    hence golden-marked. Fresh measurement: clear our workspace zter_out first
    (never touches the research tree's originals)."""
    zter_out = settings.pools_dir / "zter_out"
    if zter_out.exists():
        for f in zter_out.glob("row_*.json"):
            f.unlink()
    res = run_instrument(settings, "z_ter", ["--set", "pythia"],
                         log_name="golden_zter", timeout_s=5400)
    assert res.ok, f"z_ter failed rc={res.rc}, log: {res.log_path}"

    labels = ["160m_step40000", "160m_step143000"]
    if seeded_pools["rejuv"]:
        labels.append("160m_rejuv")
    for label in labels:
        exp_name = f"zter_row_{label}.json"
        if not (Path(__file__).parent / "expected" / exp_name).exists():
            continue
        want = load_expected(exp_name)
        got = json.loads((zter_out / f"row_{label}.json").read_text())
        assert_close(f"{label}.base_c4", got["base_c4"], want["base_c4"], 2e-4)
        for tag in ("int8", "int4"):
            assert_close(f"{label}.{tag}", got[tag], want[tag], 2e-4)


def test_readiness_golden(settings, research_env, tmp_path: Path) -> None:
    """CPU on purpose: the stored CELL J rows are CPU-fp32; matching the
    instrument exactly keeps the tolerance tight."""
    want = load_expected("icl_readiness_row_step1000.json")
    res = run_instrument(
        settings, "icl_readiness",
        ["--repo", "EleutherAI/pythia-160m", "--revisions", "step1000",
         "--device", "cpu", "--output", str(tmp_path)],
        log_name="golden_readiness_step1000", timeout_s=3600,
    )
    assert res.ok, f"icl_readiness failed rc={res.rc}, log: {res.log_path}"
    got = json.loads((tmp_path / "row_step1000.json").read_text())

    assert_close("loss", got["loss"], want["loss"], 2e-4)
    assert got["ctx_tokens"] == want["ctx_tokens"], "context prefixes changed"
    for k in ("1", "4", "16"):
        for field in ("gain_seen", "gain_unseen"):
            assert_close(f"k{k}.{field}", got["by_k"][k][field],
                         want["by_k"][k][field], 0.02)
            assert_ci_overlap(f"k{k}.{field}", got["by_k"][k][field],
                              want["by_k"][k][f"{field}_ci"], pad=0.01)


@pytest.mark.gpu
def test_induction_golden(settings, research_env, tmp_path: Path) -> None:
    want = load_expected("induction_pythia-1.4b_step71000.json")
    res = run_instrument(
        settings, "induction_ladder",
        ["--repo", "EleutherAI/pythia-1.4b", "--revisions", "step71000",
         "--output", str(tmp_path)],
        log_name="golden_induction_14b", timeout_s=1800,
    )
    assert res.ok, f"induction_ladder failed rc={res.rc}, log: {res.log_path}"

    scores: list[float] = []

    def collect(node) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if key.startswith("induction") and isinstance(val, (int, float)):
                    scores.append(float(val))
                else:
                    collect(val)
        elif isinstance(node, list):
            for val in node:
                collect(val)

    for f in tmp_path.rglob("*.json"):
        collect(json.loads(f.read_text()))
    assert scores, f"no induction score found in {tmp_path}"
    assert_close("induction_score", scores[0], want["induction_score"], 5e-3)
