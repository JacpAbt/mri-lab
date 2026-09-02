"""The state portrait — the glyph canon (FRONTEND_PRINCIPLES §5, FROZEN)
rendered server-side from measured values.

Canon: dot tremor/displacement = held-out loss (frozen dots = coordinates
annealed); teal spokes = unseen-rule gain (red shards when negative); warm
hatch = isotropic curvature (dL_abs); dashed tolerance ring radius = ||W||^2/N
(norm clock); center crosshair = registration. Every mark is a measured
quantity; a channel with no reading draws NOTHING (absence is honest).

Normalization anchors are the stored surfaces themselves (readiness losses,
sens-ladder dL_abs/S_wt ranges), so the portrait is itself a placement among
the measured points — never an aesthetic free parameter.

Determinism: layout pseudo-randomness is sha256 over (channel, index) so the
same readings always draw the same face (no RNG, no time).
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from mrilab.surfaces import loaders

CENTER = 170  # 340-box center; the mini face scales by viewBox


def _h01(*parts: object) -> float:
    """Deterministic hash -> [0, 1)."""
    b = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(b[:8], "big") / 2**64


@lru_cache(maxsize=1)
def anchors() -> dict[str, float]:
    pts = loaders.readiness_surface()
    sens = loaders.sens_rows()
    losses = [p.loss for p in pts]
    gains = [p.gain_unseen for p in pts]
    dls = [r.dl_abs for r in sens if r.dl_abs and r.dl_abs > 0]
    swts = [r.s_wt for r in sens if r.s_wt and r.s_wt > 0]
    return {
        "loss_lo": min(losses), "loss_hi": max(losses),
        "gain_hi": max(gains),
        "dl_lo": min(dls), "dl_hi": max(dls),
        "swt_lo": min(swts), "swt_hi": max(swts),
    }


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _lin(v: float, lo: float, hi: float) -> float:
    return _clamp01((v - lo) / (hi - lo)) if hi > lo else 0.5


def _log(v: float, lo: float, hi: float) -> float:
    if v <= 0 or lo <= 0 or hi <= lo:
        return 0.0
    return _clamp01(math.log(v / lo) / math.log(hi / lo))


def channels(loss: float | None, gain_unseen: float | None,
             dl_abs: float | None, s_wt: float | None) -> dict[str, float | None]:
    """Measured readings -> normalized channel intensities (None = not measured)."""
    a = anchors()
    return {
        # tremor: 0 at the most-annealed stored loss, 1 at the earliest
        "jit": None if loss is None else _lin(loss, a["loss_lo"], a["loss_hi"]),
        "ready": None if gain_unseen is None else _clamp01(gain_unseen / a["gain_hi"]),
        "anti": None if gain_unseen is None else _clamp01(-gain_unseen),
        "heat": None if dl_abs is None else _log(dl_abs, a["dl_lo"], a["dl_hi"]),
        "halo": None if s_wt is None else _log(s_wt, a["swt_lo"], a["swt_hi"]),
    }


def _pt(r: float, theta: float) -> tuple[float, float]:
    return (CENTER + r * math.cos(theta), CENTER + r * math.sin(theta))


def _line(x1: float, y1: float, x2: float, y2: float) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}"/>')


def portrait_svg(loss: float | None, gain_unseen: float | None,
                 dl_abs: float | None, s_wt: float | None,
                 css_class: str = "portrait") -> str:
    ch = channels(loss, gain_unseen, dl_abs, s_wt)
    parts: list[str] = []

    # tolerance ring (norm clock) — drawn only when ||W||^2/N is measured
    if ch["halo"] is not None:
        r_ring = 138 + 26 * ch["halo"]
        parts.append(
            f'<circle class="ch-ring" cx="{CENTER}" cy="{CENTER}" '
            f'r="{r_ring:.1f}" fill="none" stroke="var(--dim)" '
            f'stroke-dasharray="5 4"/>')

    # spokes (readiness) or shards (anti-generalization)
    if ch["ready"] is not None and gain_unseen is not None:
        if gain_unseen >= 0:
            n = 4 + round(31 * ch["ready"])
            lines = []
            for i in range(n):
                th = 2 * math.pi * _h01("spoke", i)
                r0 = 26
                r1 = r0 + (58 + 56 * ch["ready"]) * (0.55 + 0.45 * _h01("slen", i))
                x1, y1 = _pt(r0, th)
                x2, y2 = _pt(r1, th)
                lines.append(_line(x1, y1, x2, y2))
            parts.append(f'<g class="ch-spokes" stroke="var(--teal)" '
                         f'stroke-width="1">{"".join(lines)}</g>')
        else:
            n = 3 + round(9 * ch["anti"])
            lines = []
            for i in range(n):
                th = 2 * math.pi * _h01("shard", i)
                x1, y1 = _pt(148, th)
                x2, y2 = _pt(148 + 8 + 8 * _h01("shlen", i), th)
                lines.append(_line(x1, y1, x2, y2))
            parts.append(f'<g class="ch-spokes" stroke="var(--block)" '
                         f'stroke-width="1.3">{"".join(lines)}</g>')

    # guide rings (always: the substrate exists even before any reading)
    radii = [26 + 7.6 * i for i in range(12)]
    for i, r in enumerate(radii):
        op = 0.45 + 0.04 * i
        sw = 1.3 if i == len(radii) - 1 else 0.8
        parts.append(f'<circle cx="{CENTER}" cy="{CENTER}" r="{r:.1f}" '
                     f'fill="none" stroke="var(--ink)" stroke-width="{sw}" '
                     f'opacity="{op:.2f}"/>')

    # hatch (isotropic curvature): tangent ticks scattered on the outer rings
    if ch["heat"] is not None and ch["heat"] > 0:
        n = round(150 * ch["heat"])
        ticks = []
        for i in range(n):
            r = 52 + (110 - 52) * _h01("hatchr", i)
            th = 2 * math.pi * _h01("hatcha", i)
            x, y = _pt(r, th)
            # tick perpendicular-ish to the radius (tangent direction)
            tth = th + math.pi / 2 + 0.5 * (_h01("hatcht", i) - 0.5)
            dx, dy = 2.6 * math.cos(tth), 2.6 * math.sin(tth)
            ticks.append(_line(x - dx, y - dy, x + dx, y + dy))
        parts.append(f'<g class="ch-hatch" stroke="var(--warn)" '
                     f'stroke-width="1.1">{"".join(ticks)}</g>')

    # dots (tremor): 12 dots per guide ring, displaced by the loss channel
    jit = ch["jit"] if ch["jit"] is not None else 0.0
    dots = []
    for ri, r in enumerate(radii):
        off = 2 * math.pi * _h01("ringoff", ri)
        for k in range(12):
            th = off + 2 * math.pi * k / 12
            dr = jit * 7.0 * (2 * _h01("dr", ri, k) - 1)
            dth = jit * 0.12 * (2 * _h01("dth", ri, k) - 1)
            x, y = _pt(r + dr, th + dth)
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6"/>')
    parts.append(f'<g class="ch-dots" fill="var(--ink)">{"".join(dots)}</g>')

    # registration crosshair
    parts.append(f'<g stroke="var(--dim)" opacity=".8">'
                 f'{_line(CENTER - 7, CENTER, CENTER + 7, CENTER)}'
                 f'{_line(CENTER, CENTER - 7, CENTER, CENTER + 7)}</g>')

    return (f'<svg class="{css_class}" viewBox="0 0 340 340" role="img" '
            f'aria-label="state portrait: dot tremor=held-out loss, '
            f'spokes=unseen-rule gain, hatch=isotropic curvature, '
            f'dashed ring=norm clock">{"".join(parts)}</svg>')


def mini_face_svg(loss: float | None, gain_unseen: float | None,
                  dl_abs: float | None, s_wt: float | None,
                  css_class: str = "face") -> str:
    """The 30px identity mark (subject bars, recents, brand)."""
    ch = channels(loss, gain_unseen, dl_abs, s_wt)
    c = 17
    parts = ['<g fill="none" stroke="var(--ink)" opacity=".85">'
             '<circle cx="17" cy="17" r="5"/><circle cx="17" cy="17" r="9"/>'
             '<circle cx="17" cy="17" r="13"/></g>']
    if ch["ready"] is not None and gain_unseen is not None and gain_unseen >= 0:
        n = 2 + round(5 * ch["ready"])
        lines = []
        for i in range(n):
            th = 2 * math.pi * _h01("mspoke", i)
            r1 = 12 + 4 * _h01("mslen", i)
            lines.append(_line(c, c, c + r1 * math.cos(th), c + r1 * math.sin(th)))
        parts.append(f'<g stroke="var(--teal)" opacity=".9">{"".join(lines)}</g>')
    elif ch["anti"] is not None and ch["anti"] > 0:
        lines = []
        for i in range(2 + round(3 * ch["anti"])):
            th = 2 * math.pi * _h01("mshard", i)
            lines.append(_line(c + 13 * math.cos(th), c + 13 * math.sin(th),
                               c + 16 * math.cos(th), c + 16 * math.sin(th)))
        parts.append(f'<g stroke="var(--block)">{"".join(lines)}</g>')
    if ch["heat"] is not None and ch["heat"] > 0:
        ticks = []
        for i in range(round(6 * ch["heat"])):
            r = 11 + 4 * _h01("mhr", i)
            th = 2 * math.pi * _h01("mha", i)
            x, y = c + r * math.cos(th), c + r * math.sin(th)
            tth = th + math.pi / 2
            ticks.append(_line(x - 1.8 * math.cos(tth), y - 1.8 * math.sin(tth),
                               x + 1.8 * math.cos(tth), y + 1.8 * math.sin(tth)))
        parts.append(f'<g stroke="var(--warn)">{"".join(ticks)}</g>')
    parts.append('<circle cx="17" cy="17" r="16" fill="none" '
                 'stroke="var(--dim)" stroke-dasharray="3 2.5"/>')
    return (f'<svg class="{css_class}" viewBox="0 0 34 34" '
            f'aria-hidden="true">{"".join(parts)}</svg>')
