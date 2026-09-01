"""Gauge scope guards AS CODE (TRACK_C §4: 'gauge scope guards are code, not
prose'). Each predicate takes the relevant context and returns a GuardResult;
faces consult the guard named by a claim's `scope_guards` before rendering any
commentary that leans on that gauge. A denied guard renders as a Gap block
('not licensed here: <reason>'), never as silence.

Every threshold cites its scored origin.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    guard: str
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


def dl_abs_channel_guard(channel: str) -> GuardResult:
    """dL_abs prices the weight-grid/random channel (quantization) ONLY:
    ordered int4 damage 6/6 in 2 families (Z-ter, Z-ter-b). It does NOT price
    the gradient channel — killed four independent ways (W, Z-bis, EQ, DP v1.1);
    the crash test owns fine-tuning damage."""
    ok = channel == "quantization"
    return GuardResult(
        ok, "dl_abs_channel_guard",
        "dL_abs commentary licensed for the quantization channel only"
        if ok else
        f"dL_abs may not comment on channel '{channel}': static gauges do not "
        "price training trajectories (W/Z-bis/EQ/DP kills); use the crash test",
    )


def readiness_naive_substrate(payload_trained: bool) -> GuardResult:
    """R (dialect unseen gain) is licensed on NAIVE substrates only: context
    re-statement of a TRAINED rule degrades it (-1.24 at dose .03, -3.49 at
    dose .30, scaling with trained dose — Cell N'/N' interference, n=2 doses)."""
    ok = not payload_trained
    return GuardResult(
        ok, "readiness_naive_substrate",
        "substrate is payload-naive" if ok else
        "R is not licensed: the substrate carries the probe's payload in its "
        "weights (N' interference: context re-statement degrades a trained rule)",
    )


def g_ratio_display(gain_seen: float, seen_ci_low: float | None = None) -> GuardResult:
    """G = unseen/seen is licensed only when the denominator clears the G1
    floor (an arm that does not learn cannot be said to have generalized;
    denominator blowups produced a meaningless -5.9x once — G4). Threshold:
    gain_seen > 1.0 nats with CI clear of zero (the scored G1 gate)."""
    ok = gain_seen > 1.0 and (seen_ci_low is None or seen_ci_low > 0)
    return GuardResult(
        ok, "g_ratio_display",
        "G displayable (denominator clears the G1 floor)" if ok else
        "G suppressed: seen gain does not clear the G1 floor (report unseen "
        "gain in nats instead — §38b: unseen-gain primary)",
    )


def family_quarantine(top5_norm_mass: float | None,
                      min_split_half_ceiling: float | None) -> GuardResult:
    """Outlier-channel pathology (Gemma root cause, §27): top-5-of-hidden norm
    mass > 0.5 or a NEGATIVE split-half ceiling means the estimator is unstable
    (corrected cosines beyond +-1 were observed); the cell quarantines
    automatically rather than scoring either way."""
    problems = []
    if top5_norm_mass is not None and top5_norm_mass > 0.5:
        problems.append(f"top-5 norm mass {top5_norm_mass:.2f} > 0.5")
    if min_split_half_ceiling is not None and min_split_half_ceiling < 0:
        problems.append(f"negative split-half ceiling {min_split_half_ceiling:.2f}")
    ok = not problems
    return GuardResult(
        ok, "family_quarantine",
        "battery estimator stable" if ok else
        "QUARANTINED (instrument failure, not a finding): " + "; ".join(problems),
    )


def dtype_disclosure(dtype: str, induction_score: float | None = None) -> GuardResult:
    """The random-token induction probe is dtype-sensitive near its floor
    (0.1496 bf16 vs 0.2722 fp32 on the same checkpoint; identical to 4 decimals
    where the score is high). Any low reading must carry its dtype."""
    if induction_score is not None and induction_score < 0.5 and dtype != "fp32":
        return GuardResult(
            False, "dtype_disclosure",
            f"induction {induction_score:.3f} read in {dtype}: near-floor values "
            "are dtype-sensitive — render only with the dtype caveat attached",
        )
    return GuardResult(True, "dtype_disclosure", f"dtype {dtype} disclosed")


def three_b_seed_rule(n_params: int, n_seeds: int) -> GuardResult:
    """SmolLM3-3B seed-SD measured ~10x OLMo's: any 3B-class clause needs
    n>=20 seeds (Cell AA consequence)."""
    is_3b = 2_000_000_000 <= n_params < 5_000_000_000
    ok = (not is_3b) or n_seeds >= 20
    return GuardResult(
        ok, "three_b_seed_rule",
        "seed count sufficient" if ok else
        f"3B-class substrate with n_seeds={n_seeds} < 20: seed-SD at this scale "
        "is ~10x OLMo's — the clause is not quotable",
    )


def horizon_match(probe_steps: int, deploy_steps: int) -> GuardResult:
    """Probes license only their own horizon: gain rank FLIPS between 100 and
    200 steps at 1.4b (U2-ter KILL); horizon-matched selection had regret
    0.0000 at both scales. Damage accumulates; gain peaks."""
    ok = probe_steps == deploy_steps
    return GuardResult(
        ok, "horizon_match",
        "probe horizon matches the deployment horizon" if ok else
        f"probe at {probe_steps} steps does not license conclusions at "
        f"{deploy_steps} steps (U2-ter: gain rank flips across horizons)",
    )


def trigger_smoothing(n_probe_items: int, k_consecutive: int) -> GuardResult:
    """DEMO-CL instrument findings: an n=24 live R probe swings +-0.8 on noise
    (would have fired spuriously); triggers need n>=96 and k-consecutive
    smoothing/hysteresis. The interface must never expose a raw single-probe
    trigger."""
    ok = n_probe_items >= 96 and k_consecutive >= 2
    return GuardResult(
        ok, "trigger_smoothing",
        "trigger reading is smoothed" if ok else
        f"raw/under-smoothed trigger (n={n_probe_items}, k={k_consecutive}): "
        "requires n>=96 and k>=2 consecutive readings (DEMO-CL)",
    )


GUARDS = {
    f.__name__: f
    for f in (
        dl_abs_channel_guard,
        readiness_naive_substrate,
        g_ratio_display,
        family_quarantine,
        dtype_disclosure,
        three_b_seed_rule,
        horizon_match,
        trigger_smoothing,
    )
}
