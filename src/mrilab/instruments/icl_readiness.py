# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/icl_readiness.py
# source sha256: ac46433ee9db78f02b3db49e1ca30755a46bb7eb2424f13db2035a58fbacbe03
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL J (NEXT_CELLS_PREREG.md): readiness measured through the CONTEXT
channel, CPU-only. No training runs — put k examples of the dialect rule in
the prompt (drawn ONLY from pairs[0:16]) and test generalization to
pairs[16:24], which never appear in the prompt. Mirrors CELL A's design so
the numbers are directly comparable to the training-channel G.

Also reports held-out C4 loss per checkpoint with the same instrument CELL F
used, so capability is a shared axis across channels.

    python -m mrilab.instruments.icl_readiness \
        --repo EleutherAI/pythia-160m \
        --revisions step512 step2000 step8000 step40000 step143000 \
        --output experiments/regimes/icl_readiness_160m
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments import prefs
from mrilab.instruments.honesty_battery import _pair_margins

N_BOOT = 1000


def items_for(which: str, n: int, seed: int):
    """dialect items restricted to the seen (0:16) or unseen (16:24) slice."""
    saved = prefs._DIALECT
    prefs._DIALECT = saved[:16] if which == "seen" else saved[16:]
    try:
        return prefs._dialect(n, seed)
    finally:
        prefs._DIALECT = saved


def ctx_prefix(k: int, seed: int) -> str:
    """k demonstrations of the rule, built ONLY from the seen pairs."""
    if k == 0:
        return ""
    demos = items_for("seen", k, seed)
    return "\n".join(p + g for p, g, _ in demos) + "\n"


def heldout_loss(model, tok, dev, n_docs, seq_len, bs) -> float:
    from mrilab.datasets.fdgc_datasets import load_generic
    ids = load_generic("c4", tok, seq_len, n_docs).input_ids
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], bs):
            b = ids[i:i + bs].to(dev)
            tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
            n += b.shape[0]
    return tot / max(n, 1)


def ci(t: torch.Tensor, gen: torch.Generator) -> List[float]:
    idx = torch.randint(0, t.numel(), (N_BOOT, t.numel()), generator=gen)
    q = torch.quantile(t[idx].mean(dim=1), torch.tensor([0.025, 0.975]))
    return [float(q[0]), float(q[1])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="EleutherAI/pythia-160m")
    ap.add_argument("--revisions", nargs="*",
                    default=["step512", "step2000", "step8000", "step40000",
                             "step143000"])
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 4, 16])
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--ctx-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--loss-docs", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ev = {s: items_for(s, args.n, args.eval_seed) for s in ("seen", "unseen")}
    prefixes = {k: ctx_prefix(k, args.ctx_seed) for k in [0] + list(args.ks)}
    gen = torch.Generator().manual_seed(0)

    rows: List[Dict] = []
    for rev in args.revisions:
        row_p = os.path.join(args.output, f"row_{rev}.json")
        if os.path.isfile(row_p):
            rows.append(json.load(open(row_p)))
            print(f"SKIP {rev}")
            continue
        model = AutoModelForCausalLM.from_pretrained(
            args.repo, revision=rev, dtype=torch.float32).to(args.device).eval()
        loss = heldout_loss(model, tok, args.device, args.loss_docs, 512,
                            args.batch_size)
        marg: Dict = {}
        for k, pre in prefixes.items():
            for s, items in ev.items():
                mod = [(pre + p, g, b) for p, g, b in items]
                marg[(k, s)] = _pair_margins(model, tok, mod, args.device,
                                             args.batch_size, args.max_len)
        row: Dict = {"revision": rev, "loss": loss,
                     "ctx_tokens": {str(k): len(tok(p)["input_ids"])
                                    for k, p in prefixes.items()},
                     "margin_k0_seen": float(marg[(0, "seen")].mean()),
                     "margin_k0_unseen": float(marg[(0, "unseen")].mean()),
                     "by_k": {}}
        for k in args.ks:
            gs = marg[(k, "seen")] - marg[(0, "seen")]
            gu = marg[(k, "unseen")] - marg[(0, "unseen")]
            g_icl = (float(gu.mean()) / float(gs.mean())
                     if abs(float(gs.mean())) > 1e-9 else float("nan"))
            row["by_k"][str(k)] = {
                "gain_seen": float(gs.mean()), "gain_seen_ci": ci(gs, gen),
                "gain_unseen": float(gu.mean()), "gain_unseen_ci": ci(gu, gen),
                "G_icl": g_icl}
            print(f"  {rev} loss {loss:.3f} k={k:2d}: "
                  f"seen {float(gs.mean()):+7.3f} unseen {float(gu.mean()):+7.3f}"
                  f"  G_icl {g_icl:+.4f}", flush=True)
        with open(row_p, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)
        rows.append(row)
        del model

    out = {"config": vars(args), "rows": rows}
    with open(os.path.join(args.output, "icl_readiness.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.output}/icl_readiness.json")
    print("\nreadiness curve (k=16):")
    print("%-14s %8s %10s" % ("revision", "loss", "G_icl"))
    for r in rows:
        g = r["by_k"].get(str(args.ks[-1]), {}).get("G_icl")
        print("%-14s %8.3f %10s" % (r["revision"], r["loss"],
                                    f"{g:+.4f}" if g is not None else "-"))


if __name__ == "__main__":
    main()
