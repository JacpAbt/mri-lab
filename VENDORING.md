# VENDORING policy

The modules under `src/mrilab/instruments/`, `src/mrilab/datasets/`, and
`src/mrilab/inspector/` are **frozen copies** of the research repo's validated
instruments (`experiments_mri_llm`, vendored 2026-09-01; per-file provenance
headers carry the source path and source sha256; the machine-readable record is
`scripts/VENDOR_MANIFEST.json`).

## Rules

1. **Behavioral edits are forbidden.** These files reproduce stored golden rows
   (exact seed formulas, fp32 noise floor ~2e-4 nats, `FDGC_*` env semantics).
   The only changes ever applied at vendor time are recorded in
   `scripts/vendor.py`: mechanical dotted-import rewrites, the inlined
   `_shuffle` in `m1_signals.py`, and CPU guards on two
   `torch.cuda.empty_cache()` calls in `s_cross.py`. Slash paths like
   `experiments/regimes/_pool_*.pt` are deliberately untouched — the executor
   provides that subtree under its runtime cwd.
2. **Re-vendoring** (pulling upstream changes) is done ONLY via
   `py -3.11 scripts/vendor.py`, followed by the **full golden lane**
   (`pytest -m golden` on the WSL twin) before merge. Log the event below.
3. `scripts/vendor.py --check` (surfaced as `mrilab vendor-diff`) detects
   upstream drift against the manifest. Drift is information, not an
   instruction — the frozen copy stays authoritative until a deliberate
   re-vendor + golden revalidation.
4. Known upstream quirks are preserved on purpose (do not "fix"):
   `icl_readiness.items_for` mutates `prefs._DIALECT` (finally-restored);
   `payload_ladder.register_for_battery()` patches `prefs.SOURCES`;
   fp32 vs bf16 split (see `instruments/_registry.py`); four near-duplicate
   `loss_on`/`eval_loss` implementations.

## Vendor log

- 2026-09-01: initial vendor of 21 files + manual trim of
  `experiments/fdgc/datasets.py` (1,266 → ~350 lines; kept symbols listed in
  the file header). Golden lane: PENDING (P2).
