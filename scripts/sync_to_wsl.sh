#!/usr/bin/env bash
# Sync the Windows edit tree into the WSL run twin. Run INSIDE WSL:
#   bash /mnt/c/Users/Jacopo\ Abate/Desktop/Mine/Programmazione/mri-lab/scripts/sync_to_wsl.sh
# Code flows one way (Windows -> WSL). State never lives in either tree
# (it lives in the workspace, /home/jacopo/mri-lab-data); results are mirrored
# back separately by backup_workspace.sh.
set -eu

W="/mnt/c/Users/Jacopo Abate/Desktop/Mine/Programmazione/mri-lab"
D="/home/jacopo/mri-lab"

mkdir -p "$D"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'WORKSPACE_BACKUP/' \
  --exclude '.env' \
  "$W/" "$D/"

# Windows breaks two things in shell scripts: CRLF and the executable bit.
find "$D" -name '*.sh' -print0 | while IFS= read -r -d '' f; do
  sed -i 's/\r$//' "$f"
  chmod +x "$f"
done

echo "synced -> $D"
