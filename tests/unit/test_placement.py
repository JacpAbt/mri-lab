from __future__ import annotations

from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from mrilab.surfaces import placement


def test_readiness_verdicts_at_known_points() -> None:
    # pythia-160m@step143000: loss 3.3796 — well below the crossing
    p = placement.readiness_placement(3.3796)
    assert p.verdict == "below-crossing"
    assert p.expected_sign == "rules"
    assert len(p.neighbors) == 4
    # nearest neighbors must actually be near
    assert all(abs(n.loss - 3.3796) < 0.4 for n in p.neighbors)

    # step512-era substrate: loss 5.54 — above the crossing
    assert placement.readiness_placement(5.54).verdict == "above-crossing"
    # inside the band: no verdict, honestly
    mid = placement.readiness_placement(5.1)
    assert mid.verdict == "inside-crossing-band"
    assert mid.expected_sign == "no verdict"


@given(st.floats(min_value=2.0, max_value=8.0))
@hyp_settings(max_examples=50, deadline=None)
def test_readiness_verdict_boundaries_are_exact(loss: float) -> None:
    v = placement.readiness_placement(loss).verdict
    if loss > placement.CROSSING_HI:
        assert v == "above-crossing"
    elif loss < placement.CROSSING_LO:
        assert v == "below-crossing"
    else:
        assert v == "inside-crossing-band"


@given(st.floats(min_value=1e-4, max_value=50.0),
       st.floats(min_value=1e-4, max_value=50.0))
@hyp_settings(max_examples=50, deadline=None)
def test_dl_abs_percentile_monotone(a: float, b: float) -> None:
    pa = placement.dl_abs_percentile(a)
    pb = placement.dl_abs_percentile(b)
    if a <= b:
        assert pa.percentile <= pb.percentile
    assert 0.0 <= pa.percentile <= 100.0


def test_int4_rank_is_rank_language_only() -> None:
    r = placement.int4_rank(0.30)  # between olmo_210B (0.0296) and olmo_4001B (0.3706)
    assert r.n_pairs == 6
    assert 0 < r.rank_of < 6
    assert r.below is not None and r.above is not None
    assert r.below.value <= 0.30 <= r.above.value
    assert "not a damage prediction" in r.note
    # extremes clamp to rank 0 / n without inventing neighbors
    assert placement.int4_rank(1e-9).rank_of == 0
    assert placement.int4_rank(1e9).rank_of == 6


def test_crash_comparison_matches_horizon_and_lr() -> None:
    c = placement.crash_comparison(damage=0.006, gain=0.05, lr=1e-05, steps=200)
    assert c.matched_arms, "stored erosion2 arms at lr=1e-5/200 steps must match"
    assert all(a["source"] for a in c.matched_arms)
    assert "never extrapolates" in c.note
    # a horizon nobody measured: empty, not interpolated
    assert placement.crash_comparison(0.006, 0.05, lr=1e-05, steps=137).matched_arms == []
