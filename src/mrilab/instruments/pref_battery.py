# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/pref_battery.py
# source sha256: f6221eee366583b9291e8e13d658f3b5ac13251e1bf028f151e3e03fe388fa65
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""Preference-direction battery: is a preference category real, or a dataset?

For every source in prefs.CATEGORIES, fit the preference mean-shift direction
at one layer: mean activation over CONTINUATION positions of the preferred
side minus the dispreferred side (prompts are identical across sides, so the
contrast is carried by the continuations). Same construction as the fluency
and grammar directions in m1_signals, so all of them live in one comparable
space.

Readouts:
  within-category cosine   same category, different datasets. High = the
                           category is a real object; low = dataset artifact.
  between-category cosine  the geometry of the "decision area" itself.
  vs fluency/grammar dirs  the values <-> frame bridge: does a preference
                           direction live inside the text-type structure that
                           forms in the same window it locks in?

Split-half is built in: each source is fitted on two disjoint halves, and the
half-vs-half cosine is that source's own reliability ceiling — a
within-category cosine is only interpretable against the two sources'
ceilings (M5's lesson, applied here from the start).

    python -m mrilab.instruments.pref_battery \
        --pretrained allenai/OLMo-2-0425-1B --layer 8 \
        --compare-dirs experiments/regimes/m1_signals/stage1-step1907359-tokens4001B/m1_signals.json \
        --output experiments/regimes/pref_battery_olmo1b
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch

from mrilab.inspector.model_utils import ModelInspector
from mrilab.instruments import prefs


@torch.no_grad()
def _side_mean(model, block, tok, items: List, side: int, layer_H: int,
               device, bs: int, max_len: int):
    """Mean activation over continuation positions, pooled across items."""
    acc = torch.zeros(layer_H, dtype=torch.float64)
    npos = 0
    store = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        store["h"] = h.detach().float()

    handle = block.register_forward_hook(hook)
    try:
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            texts = [p + (g if side == 0 else b) for p, g, b in chunk]
            prompts = [p for p, _, _ in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            plens = [len(tok(p, add_special_tokens=True)["input_ids"])
                     if p else 1 for p in prompts]
            model(**enc)
            h = store["h"]                                    # (B,T,H)
            for j in range(h.shape[0]):
                T = int(enc["attention_mask"][j].sum())
                lo = min(plens[j], T - 1)
                seg = h[j, lo:T]
                acc += seg.sum(0).double().cpu()
                npos += seg.shape[0]
    finally:
        handle.remove()
    return acc / max(npos, 1)


def _cos(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    n = float(a @ b)
    d = float(a.norm() * b.norm())
    return n / d if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="default: every source in prefs.CATEGORIES")
    ap.add_argument("--compare-dirs", default=None,
                    help="m1_signals.json with fluency_dir/grammar_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    srcs = args.sources or [s for ss in prefs.CATEGORIES.values() for s in ss]

    insp = ModelInspector(args.pretrained, device=args.device,
                          revision=args.revision)
    model, tok = insp.model, insp.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    block = insp.get_layers()[args.layer]
    H = insp.hidden_size

    out: Dict[str, Dict] = {}
    for s in srcs:
        try:
            items = prefs.load_items(s, args.n, args.seed)
        except Exception as exc:                                # noqa: BLE001
            print(f"{s}: LOAD FAIL ({str(exc)[:90]})")
            continue
        if len(items) < 32:
            print(f"{s}: only {len(items)} items, skipping")
            continue
        halves = (items[::2], items[1::2])
        dirs = []
        for half in (items,) + halves:
            mg = _side_mean(model, block, tok, half, 0, H, device,
                            args.batch_size, args.max_len)
            mb = _side_mean(model, block, tok, half, 1, H, device,
                            args.batch_size, args.max_len)
            d = (mg - mb).float()
            dirs.append(d / d.norm().clamp_min(1e-9))
        rel = _cos(dirs[1], dirs[2])
        # Lexical control: refit on word-shuffled continuations. Shuffling
        # keeps each continuation's bag of words and destroys its structure,
        # so cos(dir, dir_shuffled) is the share of the preference direction
        # that mere vocabulary buys. High = the "preference" is a lexicon;
        # low = it needs structure/meaning. The confound this addresses is
        # the battery's own: found datasets differ lexically between sides.
        rng = __import__("random").Random(args.seed)

        def wshuf(t):
            w = t.split()
            rng.shuffle(w)
            return " " + " ".join(w)
        sh_items = [(p, wshuf(g), wshuf(b)) for p, g, b in items]
        mgs = _side_mean(model, block, tok, sh_items, 0, H, device,
                         args.batch_size, args.max_len)
        mbs = _side_mean(model, block, tok, sh_items, 1, H, device,
                         args.batch_size, args.max_len)
        ds = (mgs - mbs).float()
        ds = ds / ds.norm().clamp_min(1e-9)
        lex = _cos(dirs[0], ds)
        out[s] = {"dir": dirs[0].tolist(), "split_half_cos": rel,
                  "lexical_share_cos": lex, "n": len(items)}
        print(f"{s:22s} n={len(items):4d}  split-half ceiling {rel:+.3f}"
              f"  lexical share {lex:+.3f}")

    print("\n=== within-category (vs the two sources' ceilings) ===")
    for cat, ss in prefs.CATEGORIES.items():
        have = [s for s in ss if s in out]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = have[i], have[j]
                c = _cos(torch.tensor(out[a]["dir"]),
                         torch.tensor(out[b]["dir"]))
                # A ceiling at or below ~0.2 means the source does not even
                # agree with itself across split halves; correcting by it
                # manufactures |values| > 1. Refuse to report those rather
                # than emit an impossible number (Gemma-3 taught us this).
                ca, cb = out[a]["split_half_cos"], out[b]["split_half_cos"]
                if min(ca, cb) < 0.2:
                    print(f"  {cat:12s} {a} | {b}: cos {c:+.3f}"
                          f"  UNRELIABLE (ceiling {min(ca, cb):+.2f} < 0.2)")
                    continue
                ceil = math.sqrt(max(ca, 0) * max(cb, 0))
                print(f"  {cat:12s} {a} | {b}: cos {c:+.3f}"
                      f"  (ceiling {ceil:.3f}, corrected"
                      f" {c / ceil if ceil > 0.05 else float('nan'):+.3f})")

    print("\n=== between-category (category means) ===")
    cat_dir = {}
    for cat, ss in prefs.CATEGORIES.items():
        vs = [torch.tensor(out[s]["dir"]) for s in ss if s in out]
        if vs:
            m = torch.stack(vs).mean(0)
            cat_dir[cat] = m / m.norm().clamp_min(1e-9)
    cats = sorted(cat_dir)
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            print(f"  {cats[i]:12s} | {cats[j]:12s}: "
                  f"{_cos(cat_dir[cats[i]], cat_dir[cats[j]]):+.3f}")

    if args.compare_dirs and os.path.isfile(args.compare_dirs):
        ref = json.load(open(args.compare_dirs, encoding="utf-8"))
        print("\n=== values <-> frame bridge ===")
        for name in ("fluency_dir", "grammar_dir"):
            if name not in ref:
                continue
            rd = torch.tensor(ref[name])
            for cat, d in cat_dir.items():
                print(f"  {name:12s} | {cat:12s}: {_cos(rd, d):+.3f}")

    with open(os.path.join(args.output, "pref_battery.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"config": vars(args), "sources": out}, fh)
    print(f"\nwrote {args.output}/pref_battery.json")


if __name__ == "__main__":
    main()
