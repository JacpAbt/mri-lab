from __future__ import annotations

from mrilab.guards import traps


def test_norm_convention_required_for_perturbation() -> None:
    w = traps.check_norm_convention({"kind": "perturbation"})
    assert w and w.severity == "block"
    assert traps.check_norm_convention({"kind": "perturbation", "convention": "abs"}) is None
    assert traps.check_norm_convention({"kind": "training"}) is None


def test_grid_floor_warns_on_tiny_magnitudes() -> None:
    w = traps.check_grid_floor({"kind": "perturbation", "convention": "rel",
                                "magnitudes": [2e-4, 0.05]})
    assert w and w.severity == "warn" and "floor" in w.message


def test_epoch_trap_reproduces_the_512_chunk_incident() -> None:
    # the documented incident: 512 chunks x 3200 draws = 6.25 epochs -> memorised
    w = traps.check_epochs({"kind": "training", "pool_chunks": 512,
                            "steps": 3200, "batch_size": 1, "accum": 1})
    assert w and w.severity == "block"
    # ~2 epochs: safe per X-bis, no block; >2 warns
    w2 = traps.check_epochs({"kind": "training", "pool_chunks": 8192,
                             "steps": 200, "batch_size": 4, "accum": 8})
    assert w2 is None or w2.severity == "warn"
    # explicit override converts the block into user responsibility
    w3 = traps.check_epochs({"kind": "training", "pool_chunks": 512,
                             "steps": 3200, "batch_size": 1, "accum": 1,
                             "override_epochs": True})
    assert w3 is None or w3.severity == "warn"


def test_lr_grid_per_substrate() -> None:
    w = traps.check_lr_grid_per_substrate({
        "kind": "training", "lr": 3e-5,
        "substrates": ["pythia-160m@step1000", "pythia-160m@step12000"],
    })
    assert w and "sweep the LR" in w.message
    assert traps.check_lr_grid_per_substrate(
        {"kind": "training", "lr": 3e-5, "substrates": ["one"]}) is None


def test_pool_caching_block() -> None:
    w = traps.check_pool_caching({"kind": "training", "no_pool_cache": True})
    assert w and w.severity == "block"


def test_saturation_margin() -> None:
    # Cell N: +24-nat trained margins under a ~12-nat context ceiling -> void
    w = traps.check_saturation_margin({"layered_channel": True,
                                       "stored_margin_nats": 24.0})
    assert w and w.severity == "block"
    assert traps.check_saturation_margin({"layered_channel": True,
                                          "stored_margin_nats": 3.0}) is None
    w2 = traps.check_saturation_margin({"layered_channel": True})
    assert w2 and w2.severity == "warn"


def test_target_base_match_kills_zbis_style_comparisons() -> None:
    # Z-bis: a +0.44-nat hidden target_base offset made printed gains recovery
    rows = [{"target_base": 1.867}, {"target_base": 2.307}]
    w = traps.check_target_base_match(rows)
    assert w and w.severity == "block"
    assert traps.check_target_base_match(
        [{"target_base": 1.867}, {"target_base": 1.885}]) is None  # 0.018: U-prime-safe


def test_lint_job_aggregates() -> None:
    warnings = traps.lint_job({"kind": "training", "pool_chunks": 512,
                               "steps": 3200, "batch_size": 1, "accum": 1,
                               "no_pool_cache": True})
    kinds = {w.trap for w in warnings}
    assert {"epoch_trap", "pool_caching"} <= kinds
