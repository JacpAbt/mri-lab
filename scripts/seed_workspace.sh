#!/usr/bin/env bash
# One-time workspace seeding (run INSIDE WSL): copy the tokenized pools and the
# rejuvenated local artifact from the research run tree into the mri-lab
# workspace runtime, so (a) golden crash/SX runs see BIT-IDENTICAL training
# data ("matched data across arms" is load-bearing, not a convenience), and
# (b) z_ter's pythia set — which includes the local rejuv checkpoint — runs 3/3.
# Copies only; the research tree is never written.
set -eu

SRC=/home/jacopo/mri_compressor/experiments/regimes
WS="${MRILAB_WORKSPACE:-/home/jacopo/mri-lab-data}"
DST="$WS/runtime/experiments/regimes"

mkdir -p "$DST"
n=0
for f in "$SRC"/_pool_*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  if [ ! -f "$DST/$b" ]; then cp "$f" "$DST/$b"; echo "seeded pool $b"; n=$((n+1));
  else echo "have   pool $b"; fi
done

if [ -d "$SRC/rejuv_artifact_3e-5" ] && [ ! -d "$DST/rejuv_artifact_3e-5" ]; then
  cp -r "$SRC/rejuv_artifact_3e-5" "$DST/rejuv_artifact_3e-5"
  echo "seeded rejuv_artifact_3e-5"
  n=$((n+1))
fi

echo "seeded $n item(s) -> $DST"
du -sh "$DST" 2>/dev/null || true
