"""API contract tests (TestClient; tmp workspace via env set BEFORE app import)."""

from __future__ import annotations

import os
import tempfile

_WS = tempfile.mkdtemp(prefix="mrilab-test-ws-")
os.environ["MRILAB_WORKSPACE"] = _WS

from fastapi.testclient import TestClient  # noqa: E402

from mrilab.api.app import app  # noqa: E402

client = TestClient(app)
API = "/api/v1"


def test_health() -> None:
    r = client.get(f"{API}/meta/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["registry_version"]


def test_claims_browse() -> None:
    r = client.get(f"{API}/claims")
    assert r.status_code == 200 and len(r.json()) >= 35
    one = client.get(f"{API}/claims/readiness-crossing").json()
    assert one["evidence_class"] == "measured" and one["bounds"]
    assert client.get(f"{API}/claims/nope").status_code == 404


def test_surfaces_and_placement() -> None:
    r = client.get(f"{API}/surfaces")
    assert r.status_code == 200 and r.json()["n_files"] >= 250
    p = client.post(f"{API}/surfaces/readiness/place", json={"loss": 3.38}).json()
    assert p["verdict"] == "below-crossing"
    assert len(p["neighbors"]) == 4


def test_report_roundtrip_from_bundle() -> None:
    r = client.post(f"{API}/reports/model_report",
                    json={"label": "pythia-160m@step143000"})
    assert r.status_code == 200, r.text
    rep = r.json()
    rid = rep["report_id"]
    assert rep["registry_version"]
    got = client.get(f"{API}/reports/{rid}")
    assert got.status_code == 200
    md = client.get(f"{API}/reports/{rid}.md")
    assert md.status_code == 200 and "bounds:" in md.text
    listing = client.get(f"{API}/reports").json()
    assert any(e["id"] == rid for e in listing)


def test_report_409_names_missing_row() -> None:
    r = client.post(f"{API}/reports/model_report", json={"label": "ghost@main"})
    assert r.status_code == 409
    assert "vitals job" in r.json()["detail"]


def test_preflight_via_api() -> None:
    r = client.post(f"{API}/reports/preflight", json={
        "repo": "EleutherAI/pythia-410m", "revision": "step12000",
        "target": "legal", "horizon_steps": 200, "loss": 3.363})
    assert r.status_code == 200, r.text
    faces = [s["id"] for s in r.json()["sections"]]
    assert faces == ["readiness", "lr", "collateral", "recipe"]


def test_job_submit_and_cancel() -> None:
    spec = {"instrument": "vitals",
            "subject": {"repo": "EleutherAI/pythia-160m", "revision": "step143000"},
            "dtype": "fp32", "params": {}}
    r = client.post(f"{API}/jobs", json=spec)
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["state"] in ("QUEUED", "SKIPPED")
    # dtype auto must be rejected at the boundary
    bad = dict(spec, dtype="auto")
    assert client.post(f"{API}/jobs", json=bad).status_code == 422
    got = client.get(f"{API}/jobs/{job['id']}")
    assert got.status_code == 200


def test_arena_register_resolve() -> None:
    r = client.post(f"{API}/arena/predictions", json={
        "claim_id": "forbid-static-gauge-prediction",
        "title": "no static gauge predicts fine-tuning outcomes"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    res = client.post(f"{API}/arena/predictions/{pid}/resolve",
                      json={"status": "upheld", "notes": "no challenger yet"})
    assert res.status_code == 200 and res.json()["status"] == "upheld"
    board = client.get(f"{API}/arena/scoreboard").json()
    assert board["counts"].get("upheld", 0) >= 1
