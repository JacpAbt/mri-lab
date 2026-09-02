"""The readiness surface as a figure-as-instrument (FRONTEND_PRINCIPLES §4):
every measured point drawn, CI-hollow discipline visible, crossing band and
safety crossing marked, HOW-TO-READ annotation layer togglable, no fit drawn.

Rendered server-side from loaders.readiness_surface() — the 22 stored points
ARE the chart; a subject checkpoint is ringed among them, never interpolated.
"""

from __future__ import annotations

import html
import math

from mrilab.surfaces import loaders, placement

# Cell Q (2026-09-01, claim `safety-readiness-gated`): the safety-payload
# crossing sits ~1.1 nats after the neutral crossing, at ~3.46 held-out nats
# (1 payload class, in-context channel). Drawn as a marker, quoted via the claim.
SAFETY_CROSSING_NATS = 3.46

# viewBox geometry
W, H = 640, 372
PL, PR, PT, PB = 44, 624, 14, 326


def _fmt(v: float) -> str:
    return f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}"


def readiness_chart_svg(subject_loss: float | None = None,
                        subject_gain: float | None = None,
                        subject_label: str = "this checkpoint",
                        annotated: bool = True) -> str:
    pts = loaders.readiness_surface()
    losses = [p.loss for p in pts]
    gains = [g for p in pts for g in (p.gain_unseen, *p.gain_unseen_ci)]
    if subject_loss is not None:
        losses.append(subject_loss)
    if subject_gain is not None:
        gains.append(subject_gain)
    lo_x, hi_x = min(losses) - 0.25, max(losses) + 0.25
    lo_x = min(lo_x, SAFETY_CROSSING_NATS - 0.3)
    lo_y, hi_y = min(gains) - 0.4, max(gains) + 0.4

    def x(loss: float) -> float:
        return PL + (loss - lo_x) / (hi_x - lo_x) * (PR - PL)

    def y(gain: float) -> float:
        return PB - (gain - lo_y) / (hi_y - lo_y) * (PB - PT)

    parts: list[str] = []

    # the crossing band (4.7-5.5): between these losses no CI-clear verdict
    bx0, bx1 = x(placement.CROSSING_LO), x(placement.CROSSING_HI)
    parts.append(f'<rect class="bandrect" x="{bx0:.1f}" y="{PT}" '
                 f'width="{bx1 - bx0:.1f}" height="{PB - PT}"/>')

    # gain gridlines at integers; zero is the axis
    grid = []
    for g in range(math.ceil(lo_y), math.floor(hi_y) + 1):
        if g == 0:
            continue
        grid.append(f'<line x1="{PL}" y1="{y(g):.1f}" x2="{PR}" y2="{y(g):.1f}"/>')
        parts.append(f'<text x="{PL - 6}" y="{y(g) + 4:.1f}" '
                     f'text-anchor="end">{g:+d}</text>')
    parts.insert(0, f'<g class="grid">{"".join(grid)}</g>')
    parts.append(f'<line class="ax" x1="{PL}" y1="{y(0):.1f}" '
                 f'x2="{PR}" y2="{y(0):.1f}"/>')
    parts.append(f'<text x="{PL - 6}" y="{y(0) + 4:.1f}" text-anchor="end">0</text>')

    # loss ticks
    for lv in range(math.ceil(lo_x), math.floor(hi_x) + 1):
        parts.append(f'<text x="{x(lv):.1f}" y="{PB + 16}" '
                     f'text-anchor="middle">{lv}</text>')
    parts.append(f'<text x="{(PL + PR) / 2:.0f}" y="{PB + 34}" '
                 f'text-anchor="middle">held-out C4 loss (nats) '
                 f'&#8212; training moves leftward</text>')

    # the safety crossing (Cell Q)
    sx = x(SAFETY_CROSSING_NATS)
    parts.append(f'<line class="safety" x1="{sx:.1f}" y1="{PT}" '
                 f'x2="{sx:.1f}" y2="{PB}"/>')

    # measured points (hollow when the CI spans zero: no verdict)
    for fam_prefix, cls in (("pythia", "g-pythia"), ("OLMo", "g-olmo")):
        dots = []
        for p in pts:
            if not p.family.startswith(fam_prefix):
                continue
            lo_ci, hi_ci = p.gain_unseen_ci
            hollow = ' class="pt hollow"' if lo_ci <= 0 <= hi_ci else ' class="pt"'
            tip = (f"{p.family}@{p.revision} &#183; loss {p.loss:.3f} &#183; "
                   f"unseen {_fmt(p.gain_unseen)} nats "
                   f"[CI {_fmt(lo_ci)}, {_fmt(hi_ci)}]")
            dots.append(f'<circle{hollow} cx="{x(p.loss):.1f}" '
                        f'cy="{y(p.gain_unseen):.1f}" r="4.5">'
                        f'<title>{tip}</title></circle>')
        parts.append(f'<g class="{cls}">{"".join(dots)}</g>')

    # the subject, ringed among the measured points (never a fitted position)
    if subject_loss is not None and subject_gain is not None:
        cx, cy = x(subject_loss), y(subject_gain)
        parts.append(f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="9"/>')
        anchor = "end" if cx > (PL + PR) / 2 else "start"
        lx = cx - 14 if anchor == "end" else cx + 14
        parts.append(f'<text class="lab" x="{lx:.1f}" y="{cy - 10:.1f}" '
                     f'text-anchor="{anchor}">{html.escape(subject_label)}</text>')

    # HOW-TO-READ overlay (state controlled by the figpanel's data-anno attr).
    # Three explainers on three separate lines — they must never collide.
    anno: list[str] = []
    anno.append(f'<text x="{PL + 4}" y="{PT + 12}">each point = one measured '
                f'checkpoint &#183; no fit drawn</text>')
    anno.append(f'<text x="{(bx0 + bx1) / 2:.0f}" y="{PT + 28}" '
                f'text-anchor="middle">the crossing 4.7&#8211;5.5: '
                f'writes flip items&#8594;rules</text>')
    anno.append(f'<text x="{sx + 6:.1f}" y="{PB - 6}">safety crossing '
                f'&#8776;3.46 (n=1 payload class)</text>')
    hollows = [p for p in pts if p.gain_unseen_ci[0] <= 0 <= p.gain_unseen_ci[1]]
    if hollows:
        hp = max(hollows, key=lambda p: p.loss)
        hx, hy = x(hp.loss), y(hp.gain_unseen)
        left = hx > (PL + PR) / 2  # keep the callout inside the frame
        ex = hx - 40 if left else hx + 40
        anno.append(f'<line x1="{hx - 8 if left else hx + 8:.1f}" '
                    f'y1="{hy - 8:.1f}" x2="{ex:.1f}" y2="{hy - 30:.1f}"/>')
        anno.append(f'<text x="{ex - 4 if left else ex + 4:.1f}" '
                    f'y="{hy - 34:.1f}" text-anchor='
                    f'"{"end" if left else "start"}">hollow = CI spans zero '
                    f'&#8594; no verdict</text>')
    parts.append(f'<g class="anno">{"".join(anno)}</g>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="unseen-rule '
            f'gain vs held-out loss; every stored point with the crossing band '
            f'and the safety crossing marked">{"".join(parts)}</svg>')
