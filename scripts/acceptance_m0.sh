#!/usr/bin/env bash
# M0 acceptance runs 2 + 3 (HANDOFF_VALIDATION §1), through the REAL engine
# path (mrilab run --sync: skip-check -> trap lint -> thermal gate ->
# research-venv subprocess -> payload validation -> fingerprint -> row index).
# Run INSIDE WSL, detached. Everything is idempotent (payload skip-checks).
set -u
cd /home/jacopo/mri-lab
M=".venv-svc/bin/mrilab"
LOG=/home/jacopo/mri-lab-data/logs/acceptance_m0.log
exec > "$LOG" 2>&1

echo "=== M0 acceptance start $(date -Is) ==="

echo "--- [3a] preflight probes: pythia-410m@step12000, legal, horizon 200, 3 LRs ---"
for lr in 1e-05 3e-05 0.0001; do
  echo ">> probe lr=$lr $(date +%H:%M:%S)"
  $M run erosion_arms --repo EleutherAI/pythia-410m --revision step12000 \
     --dtype fp32 --sync \
     --params "{\"target\":\"legal\",\"lr\":$lr,\"steps\":200,\"warmup\":40,\"eval_every\":50}"
done

echo "--- [3b] preflight report (should now select from the fresh probes) ---"
$M report preflight --repo EleutherAI/pythia-410m --revision step12000 \
   --target legal --horizon-steps 200 --loss 3.363 --md

echo "--- [2a] never-measured model: Qwen/Qwen3-1.7B vitals (fp32 panel) ---"
echo ">> vitals $(date +%H:%M:%S)"
$M run vitals --repo Qwen/Qwen3-1.7B --dtype fp32 --sync --params '{}'

echo "--- [2b] model report for the never-measured model ---"
$M report model_report --repo Qwen/Qwen3-1.7B --label "Qwen3-1.7B@main" --md

echo "=== M0 acceptance end $(date -Is) ==="
