# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/match_capability.py
# source sha256: 3bb9c8b86993cda4a3ba59bc28bed448fcbe7b8cfc4f0532360f07412be99bbc
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL F step 1: pick the pythia-410m checkpoint whose held-out C4 loss
matches pythia-160m@step40000 (the mature fork of sections 29/31), so a
capability-matched / maturity-mismatched injection pair can be built.

    python -m mrilab.instruments.match_capability \
        --output experiments/regimes/steward/capmatch
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def heldout_loss(repo: str, rev: str, ids: torch.Tensor, dev: str,
                 bs: int) -> float:
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=torch.float32).to(dev).eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], bs):
            b = ids[i:i + bs].to(dev)
            tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
            n += b.shape[0]
    del model
    if dev != "cpu":
        torch.cuda.empty_cache()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-repo", default="EleutherAI/pythia-160m")
    ap.add_argument("--target-rev", default="step40000")
    ap.add_argument("--cand-repo", default="EleutherAI/pythia-410m")
    ap.add_argument("--cand-revs", nargs="*",
                    default=["step512", "step1000", "step2000", "step4000",
                             "step8000"])
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.target_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from mrilab.datasets.fdgc_datasets import load_generic
    ids = load_generic("c4", tok, args.seq_len, args.n_docs).input_ids

    target = heldout_loss(args.target_repo, args.target_rev, ids,
                          args.device, args.batch_size)
    print(f"TARGET {args.target_repo}@{args.target_rev}: loss {target:.4f}",
          flush=True)
    rows = []
    for rev in args.cand_revs:
        l = heldout_loss(args.cand_repo, rev, ids, args.device,
                         args.batch_size)
        rows.append({"revision": rev, "loss": l, "abs_err": abs(l - target)})
        print(f"  {args.cand_repo}@{rev}: loss {l:.4f} "
              f"(|err| {abs(l - target):.4f})", flush=True)
    best = min(rows, key=lambda r: r["abs_err"])
    out = {"target": {"repo": args.target_repo, "revision": args.target_rev,
                      "loss": target},
           "candidates": rows, "best": best}
    with open(os.path.join(args.output, "capmatch.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nBEST MATCH: {args.cand_repo}@{best['revision']} "
          f"(loss {best['loss']:.4f} vs target {target:.4f}, "
          f"err {best['abs_err']:.4f})")
    print(f"wrote {args.output}/capmatch.json")


if __name__ == "__main__":
    main()
