# HANDOFF_VALIDATION — the tests still to do before the instrument is trusted

State at 2026-09-01 (backend v1 code complete; P0–P6 committed; 64 CPU tests
green; golden lane WRITTEN but NOT RUN — it needs a cleared machine window,
and another session currently owns the GPU).

## 1. Software gates (block shipping; all runnable today given a machine window)

- [x] **Golden lane GREEN 2026-09-01** — 6/6 in 12:39 on the WSL twin
      (fingerprint gate passed; vitals ×2 + crash + readiness + z_ter ×3 +
      induction all reproduced stored rows within their declared bands).
      Note for the next rerun: tee the produced row values into the log so the
      crash band can be tightened from measurement (this run recorded
      pass/fail only). Original checklist:
  - [ ] vitals ×2 (pythia-160m@step143000, @step40000): loss to 4 decimals,
        dL_abs/S_wt within 3× stored seed-sd, R CI overlap (P-PAN1/2 criteria).
        ~5–10 min each [gpu].
  - [ ] crash ×1 (step40000, lr 1e-5, 200 steps, code target): bases ≤5e-4;
        damage/gain within the declared first-calibration band
        (max(30% rel, 2e-3)) — **tighten the band from this first rerun and
        record it in tests/golden/test_golden_rows.py**. ~4 min [gpu; needs
        `scripts/seed_workspace.sh` first].
  - [ ] readiness ×1 (step1000, CPU to match the stored instrument): loss
        ≤2e-4, gains ≤0.02 + CI overlap. ~10 min [cpu].
  - [ ] z_ter ×3 (pythia set incl. rejuv artifact after seeding): all values
        ≤2e-4 (the fp32 noise floor). ~25 min [cpu-heavy].
  - [ ] induction ×1 (pythia-1.4b@step71000): score ±5e-3 vs 0.9658 [gpu].
- [ ] **M0 acceptance**:
  1. [ ] `mrilab report model_report --label pythia-160m@step143000` renders
         the known verdicts purely from bundle + registry (no re-measurement).
         *(Already exercised by tests/unit/test_faces.py + the API smoke —
         re-run once as a CLI acceptance and record the output.)*
  2. [ ] Full report for one never-measured ≤2B model end-to-end on the 3090
         in <1 h (vitals job → model_report). Candidate: Qwen3-1.7B or
         SmolLM2-1.7B.
  3. [ ] `preflight` on pythia-410m@step12000, target legal, horizon 200:
         probe jobs → LR pick reproduces the stored U2-ter selection
         (regret 0.0000 vs the stored grid rows in the bundle).
- [ ] **Ops checks**: 300 W power cap re-applied after every reboot
      (nvidia-smi -pl 300; observed 80 °C during v1 build — likely uncapped);
      `mrilab fingerprint` clean; `scripts/backup_workspace.sh` scheduled.

## 2. Registry review (content gate)

- [ ] USER pass over `src/mrilab/claims/registry.yaml` (40 entries): confirm
      statements, and upgrade `verbatim_verified: false` entries by checking
      their `bounds` strings against the PANEL_PREREG/INSTRUMENT_PREREG scored
      blocks verbatim. Entries currently false: norm-clock, item-channel-decays,
      readiness-null-calibration, quenched-endpoint, rejuvenate-before-quantize,
      dl-abs-prices-int4, anneal-costs-int4-fragility, crash-test-ranks,
      probe-at-your-horizon, curvature-provenance, local-geometry-vs-equilibrium,
      sgrad-regime-flag, zero-sum-onset-curvature-driven, redistribution-hidden,
      epoch-safety, kd-regulariser, sft-double-ratchet,
      post-training-readiness-cost, trigger-discipline, induction-dtype-floor,
      iso-damage-lr-rule, dp-predictive-column, panel-anisotropy-trajectory.

## 3. Open research cells that gate specific REPORT CLAIMS
(cost | kill condition | the wording each changes)

- [ ] **U2-quater on a new domain** (~2–3 GPU-h | regret >0.020 | preflight
      LR-pick wording: drops the recorded circularity caveat → "validated on
      an unseen domain").
- [ ] **Z-ter-b v2 (nf4/GPTQ, bitsandbytes)** (GPU | nf4 ordering breaks the
      6/6 dL_abs pricing | certificate/quantization wording: "RTN-simulated"
      → "deployment-grade").
- [ ] **DP v1.1 crash re-run at 4096 chunks** (GPU | — | corpus crash-profiles
      become externally quotable; currently descriptive-only after the v1
      withdrawal).
- [ ] **n≥20 seeds for any 3B clause** (cheap per clause | — | lifts the
      three_b_seed_rule refusal on 3B-class numbers).
- [x] **Cell Q core LANDED 2026-09-01** (21 checkpoints, 70M-9B: P-Q1 HIT,
      safety crossing ~1.1 nats later than dialect, cross-construct transfer
      3/3+3/3 at maturity) → registry claim `safety-readiness-gated` added;
      readiness-face wording updated. Still gated on it:
  - [ ] Q stage 2 (32B frontier, detached on Modal — confirm and bank).
  - [ ] **Cell Q-bis** (gradient-channel safety install | channels may differ:
        Cell C voided a 160M gradient install | gates any wording beyond
        "in-context channel").
  - [ ] **O-double-prime** (graded semantic-depth ladder, frozen 2026-09-01,
        chained behind L8 | flat result kills the depth reading of Q's gap |
        decides whether the crossing-location axis is pragmatic or
        representational depth).
- [ ] **DEMO-CL v2** (calibrated + smoothed trigger | <0.7× best fixed |
      unlocks any recorder TRIGGER feature beyond advisory; until then the
      trigger_smoothing guard keeps raw probes un-renderable).
- [ ] **KD v3a/b + KD-REG prereg** (GPU | per prereg | KD advisory sheds its
      "n=1 config" label).

## 4. Known v1 limitations (deliberate, not bugs)

- gaudit executes no custom payloads yet (spec validation + protocol only);
  the payload compiler is the next face increment.
- recorder has no live run-watcher; CI gates evaluate indexed rows on demand.
- No Modal executor (pattern documented: modal_32b.py image/volume/spawn).
- No websockets/SSE; log tail by polling.
- Faces 1–2 are the complete ones; 3–6 are v1-thin by plan.
