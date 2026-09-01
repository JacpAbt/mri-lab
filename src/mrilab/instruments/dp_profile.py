# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/dp_profile.py
# source sha256: ea37cb0c08507fcd8a9c3a9abcfa6d3c9854ee3f42eff181886a26b90cb5c285
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL DP v1 (PANEL_PREREG PART VI): the dataset profile — CPU columns.

Per (corpus x reference checkpoint): tokenizer fit, duplication, and the
redistribution-risk column (C4 response along the corpus gradient at the
validated rho_dir = 3e-4). The GPU crash-on-corpus column runs separately.

    python -m mrilab.instruments.dp_profile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import (
    LOSS_DOCS, SEQ_LEN, target_params, loss_on)
from mrilab.instruments.vitals import grad_on

CORPORA = ["code", "legal", "math", "finance", "wikitext"]
# 2026-08-28: references upgraded to the two largest cached pythia points
# (user preference for >160M substrates). The 160m rows from v1 remain on
# disk and are still reported by analyze_dp --variant v1.
CKPTS = [("EleutherAI/pythia-410m", "step12000"),
         ("EleutherAI/pythia-1.4b", "step71000")]
RHO_DIR = 3e-4
N_STATS = 1024
OUT = "experiments/regimes/dp_out"


def corpus_chunks(name: str, tok, n: int) -> torch.Tensor:
    from mrilab.datasets import fdgc_datasets as D
    pure = {"code": D.load_code_dataset, "legal": D.load_legal_dataset,
            "math": D.load_math_dataset,
            "finance": D.load_finance_news_dataset,
            "wikitext": lambda tk, sl, m: D.load_wikitext_data(
                tk, split="train", max_seq_len=sl, num_samples=m)}
    cache = os.path.join("experiments/regimes",
                         f"_dp_pool_{name}_{n}_{SEQ_LEN}.pt")
    if os.path.isfile(cache):
        return torch.load(cache)
    ids = pure[name](tok, SEQ_LEN, n).input_ids
    torch.save(ids, cache)
    return ids


def stats_columns(name: str, tok, ids: torch.Tensor) -> Dict:
    n = ids.shape[0]
    # duplication: exact chunk dup + repeated 8-gram fraction
    hashes = [hashlib.md5(bytes(str(tuple(r.tolist())),
                                "utf8")).hexdigest() for r in ids]
    dup_rate = 1.0 - len(set(hashes)) / n
    grams = Counter()
    for r in ids[: min(n, 512)]:
        t = r.tolist()
        for i in range(0, len(t) - 8, 8):
            grams[tuple(t[i:i + 8])] += 1
    rep8 = (sum(c for c in grams.values() if c > 1)
            / max(sum(grams.values()), 1))
    # tokenizer fit: tokens/byte + unigram entropy
    sample = ids[: min(n, 256)]
    n_bytes = sum(len(tok.decode(r).encode("utf8")) for r in sample)
    tpb = float(sample.numel()) / max(n_bytes, 1)
    cnt = Counter(sample.flatten().tolist())
    tot = sum(cnt.values())
    ent = -sum((c / tot) * math.log2(c / tot) for c in cnt.values())
    return {"n_chunks": n, "dup_rate": dup_rate, "rep8_fraction": rep8,
            "tokens_per_byte": tpb, "token_entropy_bits": ent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for repo, rev in CKPTS:
        short = repo.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(repo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = None
        from mrilab.datasets.fdgc_datasets import load_generic
        c4 = load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
        for name in CORPORA:
            row_p = os.path.join(OUT, f"row_{name}_{short}.json")
            if os.path.isfile(row_p):
                print(f"SKIP {name} x {short}")
                continue
            ids = corpus_chunks(name, tok, N_STATS)
            if ids.shape[0] < 512:
                json.dump({"corpus": name, "ckpt": short,
                           "void": f"only {ids.shape[0]} chunks"},
                          open(row_p, "w"))
                print(f"{name} x {short}: G-DP1 VOID")
                continue
            row = {"corpus": name, "repo": repo, "revision": rev,
                   **stats_columns(name, tok, ids)}
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    repo, revision=rev,
                    dtype=torch.float32).to(args.device).eval()
                names = target_params(model)
                pmap = {k: p for k, p in model.named_parameters()
                        if k in names}
                orig = {k: p.detach().clone() for k, p in pmap.items()}
                base_c4 = loss_on(model, c4, args.device, 4)
            gbatch = ids[:32]
            grad_on(model, gbatch, args.device, 4)
            gmap = {k: p.grad.detach().clone() for k, p in pmap.items()
                    if p.grad is not None}
            model.zero_grad(set_to_none=True)
            deltas = {}
            for sign in (+1.0, -1.0):
                for k, p in pmap.items():
                    g = gmap[k]
                    d = g * (sign * RHO_DIR * orig[k].norm()
                             / (g.norm() + 1e-12))
                    p.data.copy_(orig[k] + d)
                deltas[sign] = loss_on(model, c4, args.device, 4) - base_c4
            for k, p in pmap.items():
                p.data.copy_(orig[k])
            del gmap
            row["sx_first_order_c4"] = -(deltas[1.0] - deltas[-1.0]) / 2
            row["sx_curvature_c4"] = (deltas[1.0] + deltas[-1.0]) / 2
            print(f"{name:8s} x {short}: dup {row['dup_rate']:.3f}  rep8 "
                  f"{row['rep8_fraction']:.3f}  t/B "
                  f"{row['tokens_per_byte']:.3f}  H "
                  f"{row['token_entropy_bits']:.1f}  |  S_x first "
                  f"{row['sx_first_order_c4']:+.4f}  curv "
                  f"{row['sx_curvature_c4']:+.4f}", flush=True)
            with open(row_p, "w", encoding="utf-8") as fh:
                json.dump(row, fh, indent=2)
        del model
    print("DP CPU columns done")


if __name__ == "__main__":
    main()
