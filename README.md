# mri-lab

Backend of the **Model MRI**: the regimes program's frozen measurement instruments (vitals panel,
crash test, readiness/payload ladders, formation clocks, RTN quantization damage), its stored
reference surfaces, and a claims registry with evidence classes — behind one engine with six
report faces:

1. **Model MRI report** — load a checkpoint, get its measured biography.
2. **Fine-tune preflight** — readiness verdict, probe-based LR pick at your horizon, collateral.
3. **Training recorder + CI** — panel on a cadence, gate assertions, trap linter. No knobs.
4. **Release certificate** — two-column audit (readiness-cost × tamper channels), quench state.
5. **Memorization audit** — seen/unseen family splits on benchmark-shaped data.
6. **Arena** — the program's standing predictions (§42d) with their kill conditions.

Backend only (core library + FastAPI). The frontend is a separate later phase.

## Two trees, two venvs

- **Windows** `...\Programmazione\mri-lab` — edit side (this repo).
- **WSL** `/home/jacopo/mri-lab` — run side (sync with `scripts/sync_to_wsl.sh`, run inside WSL).
- **Workspace** `/home/jacopo/mri-lab-data` — all runtime state (rows, pools, reports, logs, DB,
  HF cache). ext4 only, never under `/mnt/*`.
- **Service venv** (`/home/jacopo/mri-lab/.venv-svc`): fastapi/huey/sqlmodel — no torch.
- **Research venv** (`/home/jacopo/experiments/.venv`): the environment of record that produced
  every golden row (`runtime_lock.json`). GPU jobs run as subprocesses under it with
  `PYTHONPATH=/home/jacopo/mri-lab/src`. Nothing is ever pip-installed into it.

## Discipline (inherited from the research program, non-negotiable)

- Every rendered claim carries evidence class + n + bounds, quoted verbatim from the scored
  blocks via `claims/registry.yaml`. Free-text claims cannot enter a report.
- Gauge scope guards are code (`claims/scope_guards.py`), not prose.
- The crash test is the only fine-tuning damage claim; static gauges are predictors with measured
  failure modes. Reports speak in ranks, crossings, and dissociations — never cross-family
  magnitude forecasts.
- Vendored instruments under `src/mrilab/instruments|datasets|inspector` are **frozen**:
  behavioral edits forbidden; any re-vendor requires the full golden lane
  (`pytest -m golden`). See `VENDORING.md`.

## Dev quickstart (inside WSL)

```bash
cd /home/jacopo/mri-lab
python3 -m venv .venv-svc && .venv-svc/bin/pip install -e '.[dev]'
.venv-svc/bin/pytest            # CPU lane (gpu/network excluded by default)
```

Plan of record: `~/.claude/plans/i-feel-like-all-tingly-muffin.md` (backend build plan).
Validation state: `HANDOFF_VALIDATION.md` (written at P7).
