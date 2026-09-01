from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrilab.engine.jobs import JobSpec, Subject
from mrilab.engine.skipcheck import row_is_valid


def _spec(**over) -> JobSpec:
    base = dict(instrument="vitals",
                subject=Subject(repo="EleutherAI/pythia-160m", revision="step143000"),
                dtype="fp32", params={})
    base.update(over)
    return JobSpec(**base)


def test_jobspec_hash_is_content_stable() -> None:
    a = _spec(params={"n": 96, "k": 16})
    b = _spec(params={"k": 16, "n": 96})  # key order must not matter
    assert a.content_hash() == b.content_hash()
    c = _spec(params={"n": 96, "k": 8})
    assert a.content_hash() != c.content_hash()


def test_jobspec_rejects_auto_dtype_and_unknown_instrument() -> None:
    with pytest.raises(ValueError, match="auto"):
        _spec(dtype="auto")
    with pytest.raises(ValueError, match="unknown instrument"):
        _spec(instrument="nonexistent")


def test_cli_args_shape(tmp_path) -> None:
    from mrilab.settings import Settings
    s = Settings(workspace="/home/jacopo/mri-lab-data")
    spec = _spec(params={"batch_size": 4, "skip_grad": True})
    args = spec.cli_args(s)
    assert args[:4] == ["--repo", "EleutherAI/pythia-160m", "--revision", "step143000"]
    assert "--batch-size" in args and "4" in args
    assert "--skip-grad" in args  # bare flag for True
    assert "--output" in args
    # plural-revision instruments
    r = _spec(instrument="icl_readiness", dtype="fp32").cli_args(s)
    assert "--revisions" in r and "--revision" not in " ".join(r).replace("--revisions", "")


def test_subject_label() -> None:
    assert Subject(repo="EleutherAI/pythia-160m", revision="step2").label == "pythia-160m@step2"
    assert Subject(repo="org/model").label == "model@main"
    assert Subject(path="/x/rejuv_artifact_3e-5").label == "rejuv_artifact_3e-5"


# --- skip-check adversarial suite (the empty-payload poison class) -------------

def _write(tmp_path: Path, doc) -> Path:
    p = tmp_path / "row.json"
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    return p


def test_skipcheck_refuses_truncated_json(tmp_path) -> None:
    p = tmp_path / "row.json"
    p.write_text('{"loss": 3.37, "dL_abs":')  # truncated
    assert not row_is_valid("vitals", p)


def test_skipcheck_refuses_wellformed_but_empty(tmp_path) -> None:
    # THE documented incident: well-formed JSON, empty payload
    assert not row_is_valid("m1_signals", _write(tmp_path, {
        "pretrained": "x", "revision": "y", "induction_score": 0.5, "blimp": {}}))
    assert not row_is_valid("vitals", _write(tmp_path, {}))


def test_skipcheck_refuses_wrong_schema(tmp_path) -> None:
    assert not row_is_valid("erosion_arms", _write(tmp_path, {
        "config": {}, "damage": "not-a-number", "gain": 0.1,
        "c4_base": 3.3, "target_base": 1.8, "history": [{}]}))
    assert not row_is_valid("icl_readiness", _write(tmp_path, {"loss": 4.5, "by_k": {}}))


def test_skipcheck_accepts_valid_rows(tmp_path) -> None:
    assert row_is_valid("vitals", _write(tmp_path, {
        "loss": 3.37, "dL_abs": 2.02, "S_wt": 2.16}))
    assert row_is_valid("erosion_arms", _write(tmp_path, {
        "damage": 0.005, "gain": 0.052, "c4_base": 3.376, "target_base": 1.867,
        "history": [{"step": 0}]}))
    assert row_is_valid("m1_signals", _write(tmp_path, {
        "induction_score": 0.96, "blimp": {"passive_1": 0.9}}))
    assert row_is_valid("z_ter", _write(tmp_path, {
        "base_c4": 3.379, "int8": 0.0025, "int4": 0.83}))
