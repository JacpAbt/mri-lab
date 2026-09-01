# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/icl_ladder.py
# source sha256: b4470d1b7d0273f736c86b40186f7d4bd5aaf96cd71ec49bb10e8f62c624d54f
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL O sweep: the payload ladder through the context channel.

For each (checkpoint x rung): margins at k=0 and k=16 on every eval set the
rung ships (seen / unseen / format twin / content twin). Row JSON per
checkpoint, skip-checked. Small ladder runs fp32; big models bf16 (the Cell
K convention), dtype recorded per row.

    python -m mrilab.instruments.icl_ladder \
        --repo EleutherAI/pythia-160m --revisions step8000 step40000 \
        --output experiments/regimes/ladder_out
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.icl_readiness import heldout_loss, ci
from mrilab.instruments.honesty_battery import _pair_margins
from mrilab.instruments.payload_ladder import RUNGS

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
K = 16
N_EVAL = 48


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revisions", nargs="*", default=[None])
    ap.add_argument("--rungs", nargs="*", default=sorted(RUNGS))
    ap.add_argument("--n", type=int, default=N_EVAL)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--loss-docs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    # Guarded 4-bit path (CELL Q frontier, 32B on a 40GB card). Off by
    # default; when set, bitsandbytes places the model and we do NOT call
    # .to(device). No other behaviour changes.
    ap.add_argument("--quant", default="", choices=["", "4bit"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    gen = torch.Generator().manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for rev in args.revisions:
        tag = f"{args.repo.split('/')[-1]}@{rev or 'main'}"
        row_p = os.path.join(args.output, f"row_{tag}.json")
        if os.path.isfile(row_p):
            print(f"SKIP {tag}", flush=True)
            continue
        kw = {"revision": rev} if rev else {}
        if args.quant == "4bit":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DTYPES[args.dtype],
                bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.repo, quantization_config=bnb, device_map=args.device,
                **kw).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.repo, dtype=DTYPES[args.dtype],
                **kw).to(args.device).eval()
        loss = heldout_loss(model, tok, args.device, args.loss_docs, 512, 4)
        row: Dict = {"repo": args.repo, "revision": rev, "loss": loss,
                     "dtype": args.dtype, "n": args.n, "k": args.k,
                     "rungs": {}}
        print(f"{tag}: loss {loss:.4f}", flush=True)

        for rung in args.rungs:
            spec = RUNGS[rung]
            demos = spec["demos"](args.k, 0)
            prefix = "\n".join(p + g for p, g, _ in demos) + "\n"
            ml = spec["max_len"]
            r: Dict = {"prefix_tokens": len(tok(prefix)["input_ids"]),
                       "sets": {}}
            for name, build in spec["evals"].items():
                items = build(args.n)
                if len(items) < 8:
                    r["sets"][name] = {"error": f"only {len(items)} items"}
                    continue
                m0 = _pair_margins(model, tok, items, args.device,
                                   args.batch_size, ml)
                mk = _pair_margins(
                    model, tok, [(prefix + p, g, b) for p, g, b in items],
                    args.device, args.batch_size, ml)
                gain = mk - m0
                r["sets"][name] = {
                    "n": len(items),
                    "m0_mean": float(m0.mean()), "m0_sd": float(m0.std()),
                    "gain": float(gain.mean()), "gain_ci": ci(gain, gen),
                }
                print(f"  {rung:<12} {name:<13} gain {float(gain.mean()):+8.3f}"
                      f"  [{r['sets'][name]['gain_ci'][0]:+.3f},"
                      f"{r['sets'][name]['gain_ci'][1]:+.3f}]", flush=True)
            row["rungs"][rung] = r
        with open(row_p, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)
        assert os.path.isfile(row_p)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("ladder sweep done")


if __name__ == "__main__":
    main()
