# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/induction_ladder.py
# source sha256: 153ad4d755c17b17cdbe2fb30ef1aa2717fbd2f46c3e663ca5de11a8bfd96e7d
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL S (INSTRUMENT_PREREG_2026_08.md addendum): IS READINESS WHAT INDUCTION
BUYS?

Section 4's surviving mechanistic lead is that induction LEADS the formation
cascade by one interval and is the context-use enabler. Sections 37/38 measure
READINESS through the context channel and find it ordered by held-out loss at
Spearman -0.88, with a 2.57x residual band at matched loss that nothing
explains. If induction is the enabler, that band should be induction variance.

This runs the induction instrument — the SAME `_induction_score` used for
section 30's Hubble cells and for Cell H's exoneration control (which read
0.9658 on pythia-1.4b) — on the exact checkpoints of the section-38 readiness
surface, so R, loss, S_wt and induction all sit on one row.

Uses ModelInspector, as every prior induction measurement in this program did,
so the numbers are comparable to sections 4/19/20/30. FDGC_ATTN_IMPL must be
set (sdpa) — flash_attn is installed and becomes the silent default.

    python -m mrilab.instruments.induction_ladder \
        --repo EleutherAI/pythia-160m \
        --revisions step512 step1000 step2000 step8000 step40000 step143000 \
        --output experiments/regimes/induction_ladder_160m
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from mrilab.instruments.m1_signals import _induction_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from mrilab.inspector.model_utils import ModelInspector

    os.makedirs(args.output, exist_ok=True)
    rows: List[Dict] = []
    for rev in args.revisions:
        row_p = os.path.join(args.output, f"row_{rev}.json")
        if os.path.isfile(row_p):
            rows.append(json.load(open(row_p)))
            print(f"SKIP {rev}", flush=True)
            continue
        insp = ModelInspector(args.repo, device=args.device, revision=rev)
        model = insp.model
        dev = next(model.parameters()).device
        with torch.no_grad():
            score = _induction_score(model, insp.vocab_size, dev,
                                     seed=args.seed)
        row = {"repo": args.repo, "revision": rev, "induction": float(score),
               "vocab": int(insp.vocab_size)}
        print(f"{rev}: induction {score:+.4f}", flush=True)
        with open(row_p, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)
        assert os.path.isfile(row_p), f"row not written: {row_p}"
        rows.append(row)
        del model, insp
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_p = os.path.join(args.output, "induction_ladder.json")
    with open(out_p, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "rows": rows}, fh, indent=2)
    assert os.path.isfile(out_p)
    print(f"\nwrote {out_p}")
    for r in rows:
        print("%-32s %+8.4f" % (r["revision"], r["induction"]))


if __name__ == "__main__":
    main()
