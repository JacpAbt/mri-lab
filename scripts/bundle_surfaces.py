"""Bundle the research program's stored result JSONs into mrilab's committed
reference surfaces.

Run INSIDE WSL with any python3 (stdlib only), from the repo's WSL twin or the
Windows mount:

    python3 scripts/bundle_surfaces.py \
        --research /home/jacopo/mri_compressor/experiments/regimes \
        --dest "/mnt/c/Users/Jacopo Abate/Desktop/Mine/Programmazione/mri-lab/src/mrilab/surfaces/bundle"

Rules (plan P3):
- JSONs copied VERBATIM except for one documented strip pass: embedded
  direction vectors (`dirs`, and any key ending in `_dir`) are replaced by
  {"_stripped": "<n> floats"} — they are 99% of the bytes and no surface
  placement uses them. Every stripped file is flagged in the manifest.
- HF-machinery JSONs (config/tokenizer/generation) and checkpoint dirs
  (snap_*, final_hf, *_artifact_*) are excluded.
- Internal "experiments/regimes/..." strings are join keys: preserved.
- manifest.json records, per file: source path relative to the research
  regimes dir, source sha256 (pre-strip), bundled sha256, size, mtime,
  stripped keys count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# result dirs to bundle (globs relative to the research regimes dir)
INCLUDE = [
    "ladder_out", "oprime_out",
    "icl_readiness_*",
    "sens_ladder_*",
    "erosion2", "erosionW", "erosionZB", "eq_out",
    "zter_out", "zterb_out_*",
    "dp_out", "dp11_out",
    "u2_out", "u2bis_out", "u2ter_out",
    "vitals_out",
    "m1_signals",
    "induction_control",
    "sx_out", "sgsurface_out", "sgsur_out",
    "rung_shares",
]
EXCLUDE_NAMES = {"config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "vocab.json"}
EXCLUDE_DIR_PARTS = {"snap_", "final_hf", "hfcache"}
MAX_BYTES = 20 * 1024 * 1024


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def strip_vectors(node, stripped: list) -> object:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if (k == "dirs" or k.endswith("_dir")) and isinstance(v, (list, dict)):
                n = len(v)
                stripped.append(k)
                out[k] = {"_stripped": f"{n} float(s) removed at bundle time"}
            else:
                out[k] = strip_vectors(v, stripped)
        return out
    if isinstance(node, list):
        return [strip_vectors(v, stripped) for v in node]
    return node


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", type=Path,
                    default=Path("/home/jacopo/mri_compressor/experiments/regimes"))
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()

    entries = []
    n_files = 0
    total = 0
    for pattern in INCLUDE:
        for d in sorted(args.research.glob(pattern)):
            if not d.is_dir():
                continue
            for f in sorted(d.rglob("*.json")):
                rel = f.relative_to(args.research)
                if f.name in EXCLUDE_NAMES:
                    continue
                if any(part.startswith(p) for part in rel.parts for p in EXCLUDE_DIR_PARTS):
                    continue
                raw = f.read_bytes()
                if len(raw) > MAX_BYTES:
                    print(f"skip >20MB: {rel}")
                    continue
                src_sha = sha256_bytes(raw)
                stripped: list = []
                try:
                    doc = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"skip invalid JSON: {rel}")
                    continue
                doc = strip_vectors(doc, stripped)
                out_bytes = json.dumps(doc, indent=1).encode()
                dst = args.dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(out_bytes)
                entries.append({
                    "path": str(rel).replace(os.sep, "/"),
                    "src_sha256": src_sha,
                    "bundled_sha256": sha256_bytes(out_bytes),
                    "bytes": len(out_bytes),
                    "src_mtime": datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                    "stripped_keys": len(stripped),
                })
                n_files += 1
                total += len(out_bytes)
    manifest = {
        "bundled": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "research_root": str(args.research),
        "n_files": n_files,
        "total_bytes": total,
        "files": entries,
    }
    (args.dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"bundled {n_files} files, {total/1e6:.2f} MB -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
