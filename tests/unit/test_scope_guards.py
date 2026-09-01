from __future__ import annotations

from mrilab.claims import scope_guards as sg


def test_dl_abs_only_quantization() -> None:
    assert sg.dl_abs_channel_guard("quantization")
    denied = sg.dl_abs_channel_guard("fine-tuning")
    assert not denied and "crash test" in denied.reason


def test_readiness_needs_naive_substrate() -> None:
    assert sg.readiness_naive_substrate(payload_trained=False)
    assert not sg.readiness_naive_substrate(payload_trained=True)


def test_g_ratio_floor() -> None:
    assert sg.g_ratio_display(gain_seen=5.0, seen_ci_low=4.1)
    assert not sg.g_ratio_display(gain_seen=0.4)  # below G1 floor
    assert not sg.g_ratio_display(gain_seen=1.5, seen_ci_low=-0.1)  # CI not clear


def test_family_quarantine_on_stored_gemma_pathology() -> None:
    # the stored Gemma numbers: top-5 mass up to 0.95; ceilings as low as -0.806
    q = sg.family_quarantine(top5_norm_mass=0.95, min_split_half_ceiling=-0.806)
    assert not q and "QUARANTINED" in q.reason
    assert sg.family_quarantine(top5_norm_mass=0.12, min_split_half_ceiling=0.47)
    # each pathology alone suffices
    assert not sg.family_quarantine(top5_norm_mass=0.7, min_split_half_ceiling=0.9)
    assert not sg.family_quarantine(top5_norm_mass=0.1, min_split_half_ceiling=-0.1)


def test_dtype_disclosure_near_floor() -> None:
    # 0.1496 bf16 vs 0.2722 fp32 was the measured discrepancy
    assert not sg.dtype_disclosure("bf16", induction_score=0.15)
    assert sg.dtype_disclosure("fp32", induction_score=0.15)
    assert sg.dtype_disclosure("bf16", induction_score=0.97)  # high score: fine


def test_three_b_seed_rule() -> None:
    assert not sg.three_b_seed_rule(n_params=3_000_000_000, n_seeds=5)
    assert sg.three_b_seed_rule(n_params=3_000_000_000, n_seeds=20)
    assert sg.three_b_seed_rule(n_params=160_000_000, n_seeds=5)


def test_horizon_match() -> None:
    assert sg.horizon_match(probe_steps=200, deploy_steps=200)
    assert not sg.horizon_match(probe_steps=100, deploy_steps=200)


def test_trigger_smoothing() -> None:
    assert sg.trigger_smoothing(n_probe_items=96, k_consecutive=3)
    assert not sg.trigger_smoothing(n_probe_items=24, k_consecutive=1)  # DEMO-CL
