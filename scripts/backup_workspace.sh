#!/usr/bin/env bash
# Mirror workspace RESULT files (JSON rows, reports, the SQLite index) from WSL
# back to the Windows tree, so no measurement ever exists only inside a vhdx.
# (The research repo learned this on 2026-08-24 when the vhdx failed to mount.)
# Run INSIDE WSL on a schedule or after big runs. Results only — never weights,
# never pools, never the HF cache.
set -u

SRC="${MRILAB_WORKSPACE:-/home/jacopo/mri-lab-data}"
DST="/mnt/c/Users/Jacopo Abate/Desktop/Mine/Programmazione/mri-lab/WORKSPACE_BACKUP"

mkdir -p "$DST"
n=0
while IFS= read -r f; do
  rel="${f#$SRC/}"
  mkdir -p "$DST/$(dirname "$rel")"
  cp "$f" "$DST/$rel" && n=$((n+1))
done < <(find "$SRC" \( -name '*.json' -o -name '*.md' \) -size -20M \
          -not -path '*/hfcache/*' \
          -not -path '*/runtime/*' \
          -not -path '*/queues/*' \
          -not -path '*/logs/*' \
          -not -name 'config.json' -not -name 'generation_config.json' \
          -not -name 'tokenizer*.json' -not -name 'special_tokens_map.json')

# the DB is small and precious: copy a consistent snapshot via sqlite3 if present
if [ -f "$SRC/mrilab.db" ] && command -v sqlite3 >/dev/null 2>&1; then
  sqlite3 "$SRC/mrilab.db" ".backup '$DST/mrilab.db'" && n=$((n+1))
fi

echo "backed up $n files"
du -sh "$DST" 2>/dev/null || true
