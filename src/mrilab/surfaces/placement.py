"""Placement math: put a user checkpoint's readings ON the stored surfaces.

Transport calculus enforced structurally (claim `transport-calculus`): every
function returns RANKS, CROSSING VERDICTS, NEIGHBORS, and BANDS — never an
extrapolated magnitude for an out-of-family point. Verdicts respect the
CI-clear discipline (points whose CI spans zero yield no verdict; the 70m
corner is the documented boundary case).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrilab.surfaces import loaders

# the measured crossing band (§38a-i): every CI-clear point above 5.5 negative,
# below 4.7 positive, across 2 families / 5 sizes
CROSSING_HI = 5.5
CROSSING_LO = 4.7
MATCHED_LOSS_BAND = 2.6  # x-band of readiness at matched loss (§38a-ii; 2.1-2.6x)


@dataclass(frozen=True)
class Neighbor:
    label: str
    family: str
    loss: float
    value: float
    source: str


@dataclass(frozen=True)
class ReadinessPlacement:
    loss: float
    verdict: str            # "above-crossing" | "below-crossing" | "inside-crossing-band"
    expected_sign: str      # "items (anti-generalization risk)" | "rules" | "no verdict"
    neighbors: list[Neighbor] = field(default_factory=list)
    band_note: str = ""


def readiness_placement(loss: float, k_neighbors: int = 4) -> ReadinessPlacement:
    pts = loaders.readiness_surface()
    if loss > CROSSING_HI:
        verdict, sign = "above-crossing", "items (anti-generalization risk)"
    elif loss < CROSSING_LO:
        verdict, sign = "below-crossing", "rules"
    else:
        verdict, sign = "inside-crossing-band", "no verdict"
    near = sorted(pts, key=lambda p: abs(p.loss - loss))[:k_neighbors]
    neighbors = [Neighbor(label=f"{p.family}@{p.revision}", family=p.family,
                          loss=p.loss, value=p.gain_unseen, source=p.source)
                 for p in near]
    return ReadinessPlacement(
        loss=loss, verdict=verdict, expected_sign=sign, neighbors=neighbors,
        band_note=(f"readiness at matched loss spans a ~{MATCHED_LOSS_BAND}x band "
                   "(within-family the residual tracks maturity); neighbors are "
                   "the measured points, not a prediction"),
    )


@dataclass(frozen=True)
class Percentile:
    value: float
    percentile: float  # 0-100 among stored rows
    n_rows: int
    below: Neighbor | None
    above: Neighbor | None


def _percentile_among(value: float, rows: list[tuple[str, str, float, float, str]]
                      ) -> Percentile:
    """rows: (label, family, loss, metric, source), sorted by metric."""
    rows = sorted(rows, key=lambda r: r[3])
    n = len(rows)
    n_below = sum(1 for r in rows if r[3] <= value)
    below = rows[n_below - 1] if n_below > 0 else None
    above = rows[n_below] if n_below < n else None

    def nb(r):
        return None if r is None else Neighbor(label=r[0], family=r[1],
                                               loss=r[2], value=r[3], source=r[4])

    return Percentile(value=value, percentile=100.0 * n_below / max(n, 1),
                      n_rows=n, below=nb(below), above=nb(above))


def dl_abs_percentile(dl_abs: float, family: str | None = None) -> Percentile:
    rows = [(f"{r.family}@{r.revision}", r.family, r.loss, r.dl_abs, r.source)
            for r in loaders.sens_rows()
            if family is None or r.family.startswith(family)]
    return _percentile_among(dl_abs, rows)


@dataclass(frozen=True)
class Int4Rank:
    dl_abs: float
    rank_of: int          # how many measured pairs sit at or below this dL_abs
    n_pairs: int
    below: Neighbor | None
    above: Neighbor | None
    note: str


def int4_rank(dl_abs: float) -> Int4Rank:
    """RANK the checkpoint among the six measured dL_abs -> int4 pricing pairs.
    Never a damage forecast: the pricing is 6/6 monotone WITHIN families
    (RTN-simulated); a rank between neighbors is what the evidence licenses."""
    pairs = loaders.int4_pricing_pairs()
    rows = [(p.label, p.family, p.base_c4, p.dl_abs, f"int4 +{p.int4_damage:.3f}")
            for p in pairs]
    rows.sort(key=lambda r: r[3])
    n_below = sum(1 for r in rows if r[3] <= dl_abs)
    below = rows[n_below - 1] if n_below else None
    above = rows[n_below] if n_below < len(rows) else None

    def nb(r):
        return None if r is None else Neighbor(label=r[0], family=r[1],
                                               loss=r[2], value=r[3], source=r[4])

    return Int4Rank(
        dl_abs=dl_abs, rank_of=n_below, n_pairs=len(rows),
        below=nb(below), above=nb(above),
        note=("int4 RTN damage is ordered by dL_abs 6/6 within 2 families; "
              "this is a rank among measured pairs, not a damage prediction "
              "(deployment-grade nf4/GPTQ validation queued: Z-ter-b v2)"),
    )


@dataclass(frozen=True)
class CrashComparison:
    damage: float
    gain: float
    matched_arms: list[dict]
    note: str


def crash_comparison(damage: float, gain: float, lr: float, steps: int,
                     target: str | None = None) -> CrashComparison:
    """Compare a fresh crash-test reading against stored arms at the SAME lr,
    horizon, and (optionally) target — probes license only their own horizon."""
    arms = [r for r in loaders.erosion_rows()
            if r.lr == lr and r.steps == steps
            and (target is None or r.target == target)]
    matched = sorted(
        ({"label": f"{(a.repo or '?').split('/')[-1]}@{a.revision}",
          "target": a.target, "damage": a.damage, "gain": a.gain,
          "damage_rank_vs_yours": "higher" if a.damage > damage else "lower-or-equal",
          "gain_rank_vs_yours": "higher" if a.gain > gain else "lower-or-equal",
          "source": a.source}
         for a in arms),
        key=lambda d: d["damage"],
    )
    return CrashComparison(
        damage=damage, gain=gain, matched_arms=matched,
        note=(f"comparison restricted to stored arms at lr={lr}, steps={steps}"
              + (f", target={target}" if target else "")
              + " — the crash test ranks and never extrapolates"),
    )
