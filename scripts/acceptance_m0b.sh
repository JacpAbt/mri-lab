#!/usr/bin/env bash
# M0 acceptance #3, TAKE 2 — corrected probe protocol (n_train pinned to 4096,
# warmup steps/4, horizon-scaled cap). Waits for the first acceptance script to
# release the GPU, reruns the three probes (new content hashes -> fresh rows;
# the 12.5-epoch rows stay on disk as the recorded mistake), rebuilds the
# preflight report, and prints the pick vs the stored-grid optimum.
set -u
cd /home/jacopo/mri-lab
M=".venv-svc/bin/mrilab"
LOG=/home/jacopo/mri-lab-data/logs/acceptance_m0b.log
exec > "$LOG" 2>&1

echo "=== M0b start $(date -Is) — waiting for acceptance_m0 to finish ==="
for i in $(seq 1 240); do
  grep -q "acceptance end" /home/jacopo/mri-lab-data/logs/acceptance_m0.log && break
  sleep 15
done

for lr in 1e-05 3e-05 0.0001; do
  echo ">> corrected probe lr=$lr $(date +%H:%M:%S)"
  $M run erosion_arms --repo EleutherAI/pythia-410m --revision step12000 \
     --dtype fp32 --sync \
     --params "{\"target\":\"legal\",\"lr\":$lr,\"steps\":200,\"n_train\":4096,\"warmup\":50,\"eval_every\":50}"
done

echo "--- preflight report, corrected probes ---"
$M report preflight --repo EleutherAI/pythia-410m --revision step12000 \
   --target legal --horizon-steps 200 --loss 3.363 --no-md | tail -2

echo "--- pick vs stored grid optimum ---"
python3 - <<PYEOF
import json, glob
fresh = []
for f in glob.glob("/home/jacopo/mri-lab-data/rows/erosion_arms/pythia-410m@step12000/*/erosion.json"):
    d = json.load(open(f)); c = d["config"]
    if c.get("n_train") == 4096:
        fresh.append((c["lr"], d["damage"], d["gain"], d["target_base"]))
print("fresh corrected probes:")
for lr, dm, g, tb in sorted(fresh):
    print("  lr=%g damage=%+.4f gain=%+.4f target_base=%.4f" % (lr, dm, g, tb))
CAP = 0.10  # 0.05 per 100 steps at horizon 200
ok = [r for r in fresh if r[1] <= CAP]
pick = max(ok, key=lambda r: r[2]) if ok else min(fresh, key=lambda r: r[1])
grid = []
for f in glob.glob("/home/jacopo/mri-lab/src/mrilab/surfaces/bundle/u2ter_out/grid_pythia-410m_step12000_lr*/erosion.json"):
    d = json.load(open(f)); grid.append((d["config"]["lr"], d["damage"], d["gain"]))
gok = [r for r in grid if r[1] <= CAP]
gbest = max(gok, key=lambda r: r[2]) if gok else min(grid, key=lambda r: r[1])
regret = gbest[2] - dict((lr, g) for lr, dm, g in grid).get(pick[0], float("nan"))
print("pick: lr=%g   stored-grid optimum: lr=%g   regret vs grid: %+.4f nats" %
      (pick[0], gbest[0], regret))
PYEOF
echo "=== M0b end $(date -Is) ==="
