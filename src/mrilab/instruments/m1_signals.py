# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/m1_signals.py
# source sha256: 1754c225811ebb043da69e993c36ef09a3471a60a7a47708a59af3d9d56c4d31
# vendored: 2026-09-01 by scripts/vendor.py (1 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""M1 observational signals, one revision per invocation.

Three clocks measured with one model load, to test whether they share the
1B-21B window that already hosts the fluency lock-in and the depth-schedule
assembly (M2):

  induction    verbatim-retrieval score on doubled random-token sequences:
               1 - loss(second copy)/loss(first copy). Content-free, the
               classic phase-change quantity (Olsson lineage; timing law
               2511.16893).
  directions   fluency direction = unit mean-shift real-vs-shuffled C4 at a
               mid layer; grammaticality direction = unit mean-shift BLiMP
               good-vs-bad. Saved raw so the across-revision analysis can
               compute cosine-to-final and pre/post-window orthogonality —
               the basis-formation metrics adopted from Evolution of
               Concepts (2509.17196).
  BLiMP        per-paradigm accuracy (sum log-prob comparison) on a spread of
               paradigms, for per-phenomenon snap times ("Fifty shapes"
               showed the average hides them).

Paradigms that fail to load are skipped and logged — BLiMP config names have
churned before, and a missing paradigm must not kill the revision.

    python -m mrilab.instruments.m1_signals \
        --pretrained allenai/OLMo-2-0425-1B --revision stage1-step0-tokens0B \
        --output experiments/regimes/m1_signals/stage1-step0-tokens0B
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from mrilab.inspector.model_utils import ModelInspector
from mrilab.datasets.fdgc_datasets import load_generic
# vendored inline from experiments/regimes/subspace_null.py (the only symbol
# used there): per-row token permutation for the shuffled-C4 fluency contrast.
# Inlining avoids dragging subspace_null's module-level dataset-dispatch imports.
def _shuffle(ids, gen):
    out = ids.clone()
    for r in range(out.shape[0]):
        out[r] = out[r][torch.randperm(out.shape[1], generator=gen)]
    return out

PARADIGMS = [
    "anaphor_number_agreement",
    "determiner_noun_agreement_1",
    "regular_plural_subject_verb_agreement_1",
    "wh_questions_object_gap",
    "passive_1",
    "principle_A_domain_1",
    "npi_present_1",
    "existential_there_quantifiers_1",
    "superlative_quantifiers_1",
    "ellipsis_n_bar_1",
    "coordinate_structure_constraint_object_extraction",
    "irregular_past_participle_adjectives",
]


@torch.no_grad()
def _seq_logprobs(model, tok, sents: List[str], device, bs: int) -> torch.Tensor:
    out = []
    for i in range(0, len(sents), bs):
        batch = tok(sents[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=96).to(device)
        logits = model(**batch).logits[:, :-1]
        targets = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits.float(), -1)
        tokp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out.append((tokp * mask).sum(-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def _induction_score(model, vocab: int, device, n_seq=32, half=128,
                     seed=0) -> float:
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, vocab - 1000, (n_seq, half), generator=gen)
    ids = torch.cat([ids, ids], 1).to(device)
    logits = model(input_ids=ids).logits[:, :-1]
    targets = ids[:, 1:]
    lp = torch.log_softmax(logits.float(), -1)
    tokp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    first = -tokp[:, : half - 1].mean().item()
    second = -tokp[:, half:].mean().item()
    return 1.0 - second / max(first, 1e-9)


@torch.no_grad()
def _mean_at_layer(model, block, tok_batches: List[torch.Tensor], device):
    acc, n = None, 0
    store = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        store["flat"] = h.reshape(-1, h.shape[-1]).detach().double().cpu()

    handle = block.register_forward_hook(hook)
    try:
        for ids in tok_batches:
            model(input_ids=ids.to(device))
            f = store["flat"]
            acc = f.sum(0) if acc is None else acc + f.sum(0)
            n += f.shape[0]
    finally:
        handle.remove()
    return acc / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--quant", default=None, choices=("8bit", "4bit"))
    ap.add_argument("--blimp-n", type=int, default=200)
    ap.add_argument("--dir-n", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    insp = ModelInspector(args.pretrained, device=args.device,
                          revision=args.revision, quant=args.quant)
    model, tok, blocks = insp.model, insp.tokenizer, insp.get_layers()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    block = blocks[args.layer]
    res: Dict = {"pretrained": args.pretrained, "revision": args.revision,
                 "layer": args.layer}

    res["induction_score"] = _induction_score(model, insp.vocab_size, device)
    print(f"induction score: {res['induction_score']:.4f}")

    ids = load_generic("c4", tok, args.seq_len, args.dir_n).input_ids
    sids = _shuffle(ids, gen)
    bs = 4
    real = _mean_at_layer(model, block,
                          [ids[i:i + bs] for i in range(0, len(ids), bs)], device)
    shuf = _mean_at_layer(model, block,
                          [sids[i:i + bs] for i in range(0, len(sids), bs)], device)
    d = (real - shuf)
    res["fluency_dir"] = (d / d.norm().clamp_min(1e-9)).tolist()
    res["fluency_shift_norm"] = float(d.norm())
    print(f"fluency mean-shift norm: {res['fluency_shift_norm']:.4f}")

    from datasets import load_dataset
    blimp, good_all, bad_all = {}, [], []
    for p in PARADIGMS:
        # hub-native parquet repo first (works on datasets 4.x AND 5.x, no
        # script loader); bare legacy name as fallback. The 32b_fix re-spawn
        # produced empty blimp dicts because every paradigm skipped silently
        # in the container — hence the loud skip and the hard abort below.
        try:
            try:
                ds = load_dataset("nyu-mll/blimp", p, split="train")
            except Exception:                                    # noqa: BLE001
                ds = load_dataset("blimp", p, split="train")
        except Exception as exc:                                 # noqa: BLE001
            import datasets as _ds
            import huggingface_hub as _hub
            print(f"  blimp {p}: SKIP [datasets {_ds.__version__}, "
                  f"hub {_hub.__version__}] {exc!r}")
            continue
        rows = ds.select(range(min(args.blimp_n, len(ds))))
        good = [r["sentence_good"] for r in rows]
        bad = [r["sentence_bad"] for r in rows]
        lg = _seq_logprobs(model, tok, good, device, args.batch_size)
        lb = _seq_logprobs(model, tok, bad, device, args.batch_size)
        acc = float((lg > lb).float().mean())
        blimp[p] = acc
        good_all.extend(good[:32])
        bad_all.extend(bad[:32])
        print(f"  blimp {p}: {acc:.3f}")
    if PARADIGMS and not blimp:
        raise SystemExit("blimp: every paradigm skipped — refusing to write "
                         "an empty result (fix the datasets loader first)")
    res["blimp"] = blimp

    if good_all:
        gb = tok(good_all, return_tensors="pt", padding=True, truncation=True,
                 max_length=48)["input_ids"]
        bb = tok(bad_all, return_tensors="pt", padding=True, truncation=True,
                 max_length=48)["input_ids"]
        mg = _mean_at_layer(model, block,
                            [gb[i:i + 16] for i in range(0, len(gb), 16)], device)
        mb = _mean_at_layer(model, block,
                            [bb[i:i + 16] for i in range(0, len(bb), 16)], device)
        d = mg - mb
        res["grammar_dir"] = (d / d.norm().clamp_min(1e-9)).tolist()
        res["grammar_shift_norm"] = float(d.norm())
        print(f"grammar mean-shift norm: {res['grammar_shift_norm']:.4f}")

    path = os.path.join(args.output, "m1_signals.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
