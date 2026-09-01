"""Vendor the frozen instruments from the research repo into mrilab.

Deterministic and auditable: copy each source file, apply the MECHANICAL import
rewrites (dotted module paths only — slash paths like "experiments/regimes/..."
are relative-cwd pool/output paths and are deliberately left alone), apply the
few exact-match patches listed below, prepend a provenance header carrying the
SOURCE file's sha256, and record everything in scripts/VENDOR_MANIFEST.json.

    py -3.11 scripts/vendor.py                 # (re)vendor everything
    py -3.11 scripts/vendor.py --check         # drift check against the manifest
                                               # (the basis of `mrilab vendor-diff`)

Policy (VENDORING.md): vendored files are FROZEN. Behavioral edits are forbidden;
any re-vendor requires the full golden lane (pytest -m golden) before merge.

NOTE: fdgc_datasets.py (the trimmed loader module) and the inlined _shuffle in
m1_signals are the only non-copy transformations; the m1_signals one is an exact
patch below, the datasets trim is hand-written with its own provenance header
and is NOT managed by this script (listed in the manifest as "manual").
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH = REPO.parent / "experiments_mri_llm"
MANIFEST = REPO / "scripts" / "VENDOR_MANIFEST.json"

INSTRUMENTS = [
    "sensitivity_ladder", "vitals", "icl_readiness", "icl_ladder",
    "payload_ladder", "honesty_battery", "prefs", "pref_battery",
    "m1_signals", "induction_ladder", "erosion_arms", "repetition_arms",
    "kd_arms", "l8_arms", "s_cross", "dp_profile", "z_ter", "u1_run",
    "match_capability",
]

# (src relative to research root, dst relative to repo root)
FILES = [(f"experiments/regimes/{m}.py", f"src/mrilab/instruments/{m}.py")
         for m in INSTRUMENTS] + [
    ("mri_compressor/model_utils.py", "src/mrilab/inspector/model_utils.py"),
    ("mri_compressor/data_utils.py", "src/mrilab/datasets/data_utils.py"),
]

# Ordered, dotted-path-only rewrites. Slash paths are intentionally untouched.
REWRITES = [
    (re.compile(r"from experiments\.regimes import "), "from mrilab.instruments import "),
    (re.compile(r"experiments\.regimes\."), "mrilab.instruments."),
    (re.compile(r"from experiments\.fdgc import datasets"), "from mrilab.datasets import fdgc_datasets"),
    (re.compile(r"experiments\.fdgc\.datasets"), "mrilab.datasets.fdgc_datasets"),
    (re.compile(r"mri_compressor\.model_utils"), "mrilab.inspector.model_utils"),
    (re.compile(r"mri_compressor\.data_utils"), "mrilab.datasets.data_utils"),
]

_SHUFFLE_INLINE = '''# vendored inline from experiments/regimes/subspace_null.py (the only symbol
# used there): per-row token permutation for the shuffled-C4 fluency contrast.
# Inlining avoids dragging subspace_null's module-level dataset-dispatch imports.
def _shuffle(ids, gen):
    out = ids.clone()
    for r in range(out.shape[0]):
        out[r] = out[r][torch.randperm(out.shape[1], generator=gen)]
    return out'''

# Exact-match patches, applied AFTER the rewrites. Each must match exactly once.
PATCHES = {
    "src/mrilab/instruments/m1_signals.py": [
        ("from mrilab.instruments.subspace_null import _shuffle",
         _SHUFFLE_INLINE),
    ],
    "src/mrilab/instruments/s_cross.py": [
        ("            del gmap\n            torch.cuda.empty_cache()",
         "            del gmap\n            if args.device.startswith(\"cuda\"):\n"
         "                torch.cuda.empty_cache()"),
        ("        del model, pmap, orig\n        torch.cuda.empty_cache()",
         "        del model, pmap, orig\n        if args.device.startswith(\"cuda\"):\n"
         "            torch.cuda.empty_cache()"),
    ],
}

HEADER = """\
# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: {src}
# source sha256: {sha}
# vendored: {date} by scripts/vendor.py ({n_patches} patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""


def sha256_text(text: str) -> str:
    # hash normalized to LF so Windows/WSL checkouts agree
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def vendor(research_root: Path) -> int:
    entries = []
    for src_rel, dst_rel in FILES:
        src = research_root / src_rel
        dst = REPO / dst_rel
        text = src.read_text(encoding="utf-8").replace("\r\n", "\n")
        src_sha = sha256_text(text)
        for rx, repl in REWRITES:
            text = rx.sub(repl, text)
        patches = PATCHES.get(dst_rel, [])
        for old, new in patches:
            n = text.count(old)
            if n != 1:
                print(f"FATAL: patch target found {n} times in {dst_rel}:\n{old}")
                return 1
            text = text.replace(old, new)
        leftover = re.findall(r"experiments\.\w|mri_compressor", text)
        if leftover:
            print(f"FATAL: unrewritten internal imports remain in {dst_rel}: {leftover}")
            return 1
        header = HEADER.format(src=src_rel, sha=src_sha,
                               date=_dt.date.today().isoformat(),
                               n_patches=len(patches))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(header + text, encoding="utf-8", newline="\n")
        entries.append({"src": src_rel, "dst": dst_rel, "src_sha256": src_sha,
                        "patches": len(patches)})
        print(f"vendored {src_rel} -> {dst_rel}")
    entries.append({
        "src": "experiments/fdgc/datasets.py",
        "dst": "src/mrilab/datasets/fdgc_datasets.py",
        "src_sha256": sha256_text(
            (research_root / "experiments/fdgc/datasets.py").read_text(encoding="utf-8")),
        "patches": "manual-trim (see file header for kept symbols)",
    })
    MANIFEST.write_text(json.dumps({"vendored": _dt.date.today().isoformat(),
                                    "research_root": str(research_root),
                                    "files": entries}, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST}")
    return 0


def check(research_root: Path) -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = 0
    for e in man["files"]:
        src = research_root / e["src"]
        if not src.is_file():
            print(f"MISSING upstream: {e['src']}")
            drift += 1
            continue
        sha = sha256_text(src.read_text(encoding="utf-8"))
        if sha != e["src_sha256"]:
            print(f"DRIFT: {e['src']} changed upstream since vendoring "
                  f"({e['src_sha256'][:12]} -> {sha[:12]})")
            drift += 1
    print("clean: no upstream drift" if drift == 0 else
          f"{drift} file(s) drifted — re-vendor requires the full golden lane")
    return 1 if drift else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.research_root.is_dir():
        print(f"research root not found: {args.research_root}")
        return 1
    return check(args.research_root) if args.check else vendor(args.research_root)


if __name__ == "__main__":
    sys.exit(main())
