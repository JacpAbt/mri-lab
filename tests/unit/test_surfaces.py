from __future__ import annotations

from mrilab.surfaces import loaders


def test_manifest_verifies() -> None:
    assert loaders.verify_manifest() == []


def test_readiness_surface_shape() -> None:
    pts = loaders.readiness_surface()
    # blueprint §38: 22 points, 5 sizes, 2 families; the bundle may carry a
    # few extra points measured after §38 was written — never fewer.
    assert len(pts) >= 22
    assert {p.family for p in pts} >= {"pythia-70m", "pythia-160m", "pythia-410m",
                                       "pythia-1.4b", "OLMo-2-1B"}
    losses = [p.loss for p in pts]
    assert losses == sorted(losses)
    # the crossing (§38a-i) as SCORED: verdicts only where the CI clears zero.
    # The known exception below the crossing is the 70m corner (§38a-iv,
    # sharpened by L-ter to -0.308 with CI spanning zero at 300B tokens):
    # maturity cannot substitute for capability — a no-verdict point, not a
    # counterexample. Placement code must inherit exactly this gate.
    for p in pts:
        lo, hi = p.gain_unseen_ci
        ci_clear = lo > 0 or hi < 0
        if p.loss > 5.5 and ci_clear:
            assert p.gain_unseen < 0, p
        if p.loss < 4.7 and ci_clear:
            assert p.gain_unseen > 0 or p.family == "pythia-70m", p


def test_sens_rows_present() -> None:
    rows = loaders.sens_rows()
    assert len(rows) >= 28  # the Cell L 28-row store + stage2 block
    for r in rows:
        assert r.dl_abs > 0 and r.s_wt > 0


def test_erosion_rows_present() -> None:
    rows = loaders.erosion_rows()
    assert len(rows) >= 80  # erosion2/W/ZB + eq + u2 grids + dp crash rows
    lrs = {r.lr for r in rows}
    assert 1e-05 in lrs


def test_int4_pricing_pairs_six_over_two_families() -> None:
    pairs = loaders.int4_pricing_pairs()
    assert len(pairs) == 6
    by_family: dict[str, list] = {}
    for p in pairs:
        by_family.setdefault(p.family, []).append(p)
    # the 6/6 monotone pricing (Z-ter + Z-ter-b scored): int4 damage ordered
    # by dL_abs WITHIN each family
    for family, fp in by_family.items():
        fp.sort(key=lambda x: x.dl_abs)
        damages = [x.int4_damage for x in fp]
        assert damages == sorted(damages), f"{family}: {damages}"


def test_vitals_rows_load() -> None:
    labels = loaders.vitals_labels()
    assert "pythia-160m@step143000" in labels
    row = loaders.vitals_row("pythia-160m@step143000")
    assert abs(row["loss"] - 3.3796) < 1e-3
