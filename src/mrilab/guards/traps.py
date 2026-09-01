"""The seven instrument traps as pre-run checks (plan P4).

Each voided or nearly voided a cell in the research program's 2026-08 battery
(instrument-battery-traps ledger). Here they run BEFORE compute is spent —
over a JobSpec-like dict — and AFTER, over produced rows where the trap is
only visible in results. Findings render as TrapWarning blocks; severity
"block" means the engine refuses the job without an explicit override.
"""

from __future__ import annotations

from dataclasses import dataclass

FP32_NOISE_FLOOR_NATS = 2e-4  # measured fp32 forward-pass floor (Cell L smoke)


@dataclass(frozen=True)
class TrapWarning:
    trap: str
    severity: str  # "warn" | "block"
    message: str


def check_norm_convention(params: dict) -> TrapWarning | None:
    """Trap 1: a perturbation at fixed ELEMENTWISE scale and one at fixed
    RELATIVE norm measure different physics (tr(H) vs tr(H)*||W||^2/N) and can
    move in opposite directions over a run. Any perturbation job must declare
    its convention."""
    if params.get("kind") != "perturbation":
        return None
    conv = params.get("convention")
    if conv not in ("abs", "rel"):
        return TrapWarning(
            "norm_convention", "block",
            "perturbation job without a declared norm convention (abs|rel): "
            "ABS and REL measure different physics and can disagree in sign",
        )
    return None


def check_grid_floor(params: dict) -> TrapWarning | None:
    """Trap 2: a grid set by intuition lands in the noise floor (first REL grid
    read 2e-4..2e-3 against a ~2e-4 fp32 floor). Warn when any requested
    magnitude sits below 10x the floor-calibrated smallest validated magnitude."""
    if params.get("kind") != "perturbation":
        return None
    mags = params.get("magnitudes") or []
    tiny = [m for m in mags if m < 1e-3]  # smallest validated grid points: 1e-3
    if tiny:
        return TrapWarning(
            "grid_noise_floor", "warn",
            f"magnitudes {tiny} sit below the validated grid; smoke the grid "
            f"and report the fp32 noise floor ({FP32_NOISE_FLOOR_NATS} nats) "
            "before interpreting deltas",
        )
    return None


def check_epochs(params: dict) -> TrapWarning | None:
    """Trap 3: a tiny training pool is an epoch trap (512 chunks x 3200 draws
    = 6+ epochs memorized the pool; held-out loss ROSE). Compute draws/chunks
    BEFORE launching: warn > 2 epochs, block > 6 without override."""
    if params.get("kind") != "training":
        return None
    chunks = params.get("pool_chunks")
    steps = params.get("steps")
    batch = params.get("batch_size", 1)
    accum = params.get("accum", 1)
    if not chunks or not steps:
        return None
    epochs = steps * batch * accum / chunks
    if epochs > 6 and not params.get("override_epochs"):
        return TrapWarning(
            "epoch_trap", "block",
            f"{epochs:.1f} epochs over a {chunks}-chunk pool: the 512-chunk/6+"
            "-epoch trap detonated a cell (held-out loss rose; KD v2 CE arms "
            "+1.3..+1.8 nats). Enlarge the pool or set override_epochs",
        )
    if epochs > 2:
        return TrapWarning(
            "epoch_trap", "warn",
            f"{epochs:.1f} epochs over a {chunks}-chunk pool (X-bis: ~2 epochs "
            "safe on both axes at small scale; 4 epochs already visible in loss)",
        )
    return None


def check_lr_grid_per_substrate(params: dict) -> TrapWarning | None:
    """Trap 4: two substrates do not share a usable LR (a 16x curvature gap on
    one grid voided 3 of 4 arms). Multi-substrate comparisons must sweep LR or
    declare per-substrate grids."""
    if params.get("kind") != "training":
        return None
    subs = params.get("substrates") or []
    lrs = params.get("lrs") or ([params["lr"]] if params.get("lr") else [])
    if len(subs) > 1 and len(lrs) == 1:
        return TrapWarning(
            "lr_grid_per_substrate", "warn",
            f"{len(subs)} substrates on a single LR ({lrs[0]}): if their "
            "sensitivities differ, the comparison is between one arm inside "
            "its regime and one outside — sweep the LR per substrate",
        )
    return None


def check_pool_caching(params: dict) -> TrapWarning | None:
    """Trap 5: dataset loaders are the memory risk (12288 streamed chunks hit
    64GB RSS, GPU at 1%) — and per-arm re-streaming breaks matched-data
    designs. Pools must be disk-cached (the vendored instruments do this;
    flag any job that opts out)."""
    if params.get("kind") != "training":
        return None
    if params.get("no_pool_cache"):
        return TrapWarning(
            "pool_caching", "block",
            "pool caching disabled: arms would re-stream (memory risk) and "
            "would NOT see identical data — matched-arm designs require the "
            "cached pool",
        )
    return None


def check_saturation_margin(params: dict) -> TrapWarning | None:
    """Trap 6: a saturated arm has no headroom (Cell N: +24-nat trained margins;
    demonstrations pushed the preference off its operating point, -3.49 seen).
    Any job layering a second channel on stored content must declare the
    stored margin and it must sit below the channel ceiling."""
    if not params.get("layered_channel"):
        return None
    margin = params.get("stored_margin_nats")
    ceiling = params.get("channel_ceiling_nats", 12.0)
    if margin is None:
        return TrapWarning(
            "saturation_margin", "warn",
            "layered-channel job without a declared stored margin: check the "
            "stored margins sit below the second channel's ceiling first",
        )
    if margin >= ceiling:
        return TrapWarning(
            "saturation_margin", "block",
            f"stored margin {margin:.1f} nats >= channel ceiling {ceiling:.1f}: "
            "the arm is saturated and the cell voids on its own gate (Cell N)",
        )
    return None


def check_target_base_match(rows: list[dict], tol: float = 0.05) -> TrapWarning | None:
    """Trap 7: cross-checkpoint gain/damage comparisons need matched
    target_base (Z-bis: a +0.44-nat hidden offset made ~97% of printed gains
    recovery; two HITs died under this one check). Report-level: any
    comparison group must match target_base within tol or the comparison voids."""
    bases = [r.get("target_base") for r in rows if r.get("target_base") is not None]
    if len(bases) < 2:
        return None
    spread = max(bases) - min(bases)
    if spread > tol:
        return TrapWarning(
            "target_base_match", "block",
            f"target_base spread {spread:.3f} nats > {tol}: cross-checkpoint "
            "gain/damage comparison voids (gains may be recovery, not learning; "
            "capability gates must name their measurement domain)",
        )
    return None


JOB_CHECKS = (
    check_norm_convention,
    check_grid_floor,
    check_epochs,
    check_lr_grid_per_substrate,
    check_pool_caching,
    check_saturation_margin,
)


def lint_job(params: dict) -> list[TrapWarning]:
    out = []
    for check in JOB_CHECKS:
        w = check(params)
        if w is not None:
            out.append(w)
    return out
