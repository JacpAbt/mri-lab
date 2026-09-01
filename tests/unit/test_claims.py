from __future__ import annotations

import pytest

from mrilab.claims import registry
from mrilab.claims.schema import ClaimRef, ClaimStatus, EvidenceClass
from mrilab.claims.scope_guards import GUARDS


def test_registry_loads_and_lints() -> None:
    claims = registry.load()
    assert len(claims) >= 35
    for c in claims.values():
        assert c.bounds.strip(), f"{c.id}: empty bounds"
        assert c.n.strip(), f"{c.id}: empty n"
        assert c.source.strip(), f"{c.id}: empty source"
        for g in c.scope_guards:
            assert g in GUARDS, f"{c.id}: unknown scope guard {g}"


def test_registry_has_both_hits_and_retractions() -> None:
    claims = registry.load()
    statuses = {c.status for c in claims.values()}
    # misses/retractions carried with the same prominence as hits — house rule
    assert ClaimStatus.LIVE in statuses
    assert ClaimStatus.WITHDRAWN in statuses
    assert ClaimStatus.KILLED in statuses
    classes = {c.evidence_class for c in claims.values()}
    assert classes == {EvidenceClass.MEASURED, EvidenceClass.LIT, EvidenceClass.CONJECTURE}


def test_render_carries_class_n_bounds() -> None:
    block = registry.render(ClaimRef(claim_id="readiness-crossing"))
    assert block["evidence_class"] == "measured"
    assert "5.5" in block["statement"] and "4.7" in block["statement"]
    assert block["n"] and block["bounds"] and block["source"]


def test_render_refuses_unknown_and_withdrawn() -> None:
    with pytest.raises(registry.UnknownClaim):
        registry.render(ClaimRef(claim_id="does-not-exist"))
    with pytest.raises(registry.UnquotableClaim):
        registry.render(ClaimRef(claim_id="semantic-transfer-headline"))
    with pytest.raises(registry.UnquotableClaim):
        registry.render(ClaimRef(claim_id="panel-anisotropy-trajectory"))


def test_retraction_rendering_is_explicit() -> None:
    r = registry.render_retraction("semantic-transfer-headline")
    assert r["status"] == "WITHDRAWN"
    with pytest.raises(ValueError):
        registry.render_retraction("readiness-crossing")  # not a retraction


def test_struck_claim_points_to_successor() -> None:
    c = registry.get("panel-anisotropy-trajectory")
    assert c.superseded_by == "sgrad-regime-flag"
    assert registry.get("sgrad-regime-flag").status == ClaimStatus.LIVE


def test_registry_version_is_stable_hash() -> None:
    assert registry.registry_version() == registry.registry_version()
    assert len(registry.registry_version()) == 16
