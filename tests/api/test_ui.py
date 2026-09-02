"""UI page tests (TestClient over the same app; tmp workspace via env set
BEFORE app import — the test_api.py pattern). All pages must build from the
committed bundle alone: no GPU, no network, no workspace rows."""

from __future__ import annotations

import os
import tempfile

_WS = tempfile.mkdtemp(prefix="mrilab-test-ws-ui-")
os.environ.setdefault("MRILAB_WORKSPACE", _WS)

from fastapi.testclient import TestClient  # noqa: E402

from mrilab.api.app import app  # noqa: E402

client = TestClient(app)
LABEL = "pythia-160m@step143000"


def test_load_page() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "CHECKPOINT" in r.text and "RECIPE" in r.text
    assert LABEL in r.text  # bundle subjects surface as recents
    assert "/ui/static/mrilab.css" in r.text


def test_go_redirect() -> None:
    r = client.get("/go", params={"label": LABEL}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/m/{LABEL}"


def test_static_served() -> None:
    assert client.get("/ui/static/mrilab.css").status_code == 200
    assert client.get("/ui/static/islands.js").status_code == 200


def test_overview() -> None:
    r = client.get(f"/m/{LABEL}")
    assert r.status_code == 200
    # verdict + portrait + gauges + rooms
    assert "rule-forming substrate" in r.text
    assert 'class="portrait"' in r.text and "ch-hatch" in r.text
    assert 'data-ch="ch-spokes"' in r.text
    assert "INT4 RANK" in r.text
    assert f"/m/{LABEL}/readiness" in r.text
    # gaps rendered, not hidden
    assert "NOT MEASURED" in r.text


def test_overview_unknown_subject_404() -> None:
    r = client.get("/m/ghost@main")
    assert r.status_code == 404
    assert "vitals job" in r.json()["detail"]


def test_room_readiness() -> None:
    r = client.get(f"/m/{LABEL}/readiness")
    assert r.status_code == 200
    assert "What does teaching become here?" in r.text
    assert 'class="bandrect"' in r.text  # the chart with the crossing band
    assert "hollow" in r.text            # CI discipline visible
    assert "readiness-crossing" in r.text
    assert "safety" in r.text


def test_room_fragility() -> None:
    r = client.get(f"/m/{LABEL}/fragility")
    assert r.status_code == 200
    assert "dl-abs-prices-int4" in r.text
    assert "NOT LICENSED" in r.text  # the fine-tuning guard refusal is rendered


def test_room_economics() -> None:
    r = client.get(f"/m/{LABEL}/economics")
    assert r.status_code == 200
    assert "exchange-table" in r.text and "DEMO-CL v2" in r.text


def test_room_lineage() -> None:
    r = client.get(f"/m/{LABEL}/lineage")
    assert r.status_code == 200
    assert "sft-double-ratchet" in r.text


def test_room_unknown_404() -> None:
    assert client.get(f"/m/{LABEL}/attic").status_code == 404


def test_registry_page() -> None:
    r = client.get("/registry")
    assert r.status_code == 200
    assert "The retraction wall" in r.text
    assert "semantic-transfer-headline" in r.text  # the wall shows the misses
    assert "UNVERIFIED BOUNDS" in r.text  # pending-review chip is honest


def test_queue_page_empty_ok() -> None:
    r = client.get("/queue")
    assert r.status_code == 200


def test_recipe_page() -> None:
    r = client.get("/recipe")
    assert r.status_code == 200
    assert "Descriptive only" in r.text
    assert "dp-predictive-column" in r.text  # the kill is on the page


def test_preflight_page_plan() -> None:
    r = client.get(f"/m/{LABEL}/preflight",
                   params={"target": "legal", "horizon": 200})
    assert r.status_code == 200
    assert "Probes not yet run" in r.text  # tmp workspace has no probe rows
    assert "probe-at-your-horizon" in r.text


def test_report_html_view() -> None:
    made = client.post("/api/v1/reports/model_report", json={"label": LABEL})
    assert made.status_code == 200
    rid = made.json()["report_id"]
    lst = client.get("/reports")
    assert lst.status_code == 200 and rid in lst.text
    page = client.get(f"/reports/{rid}")
    assert page.status_code == 200
    assert "model_report" in page.text and "Retractions" in page.text
