"""Unit tests: portrait determinism + canon, chart contract, block chrome."""

from __future__ import annotations

from mrilab.faces import common as kit
from mrilab.surfaces import loaders
from mrilab.ui import chart, portrait, render


def test_portrait_deterministic() -> None:
    a = portrait.portrait_svg(3.0, 4.0, 1.0, 2.0)
    b = portrait.portrait_svg(3.0, 4.0, 1.0, 2.0)
    assert a == b
    c = portrait.portrait_svg(6.5, 4.0, 1.0, 2.0)
    assert c != a  # tremor channel moved


def test_portrait_channels_present_and_absent() -> None:
    full = portrait.portrait_svg(3.0, 4.0, 1.0, 2.0)
    for cls in ("ch-dots", "ch-spokes", "ch-hatch", "ch-ring"):
        assert cls in full
    # unmeasured channels draw nothing (absence is honest)
    bare = portrait.portrait_svg(3.0, None, None, None)
    assert "ch-dots" in bare
    for cls in ("ch-spokes", "ch-hatch", "ch-ring"):
        assert cls not in bare


def test_portrait_anti_generalization_shards() -> None:
    neg = portrait.portrait_svg(6.0, -0.5, 1.0, 2.0)
    assert 'stroke="var(--block)"' in neg  # red shards, not teal spokes


def test_mini_face_none_safe() -> None:
    svg = portrait.mini_face_svg(None, None, None, None)
    assert svg.startswith("<svg") and "circle" in svg


def test_chart_draws_every_stored_point() -> None:
    svg = chart.readiness_chart_svg()
    n = len(loaders.readiness_surface())
    assert svg.count('<circle class="pt') == n
    assert 'class="bandrect"' in svg and 'class="safety"' in svg
    assert 'class="anno"' in svg
    assert 'class="ring"' not in svg  # no subject -> no ring


def test_chart_subject_ring() -> None:
    svg = chart.readiness_chart_svg(subject_loss=3.38, subject_gain=4.29,
                                    subject_label="me@here")
    assert 'class="ring"' in svg and "me@here" in svg


def test_fmt_value() -> None:
    assert render.fmt_value(None) == "&#8212;"
    assert render.fmt_value(1e-05) == "1e-05"  # the lr=+0.0000 lesson
    assert render.fmt_value(0.05) == "0.05"
    assert render.fmt_value(True) == "yes"


def test_block_chrome() -> None:
    g = kit.gap("thing not measured", "why text", unlocked_by="a job")
    html = render.block_html(g)
    assert "NOT MEASURED" in html and "a job" in html
    c = kit.claim("readiness-crossing")
    html = render.block_html(c)
    assert "MEASURED" in html and "claim-readiness-crossing" in html
    assert "bounds" in html
