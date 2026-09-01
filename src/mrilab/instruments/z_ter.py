# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/z_ter.py
# source sha256: f9155a665ed1c6f2528e5f2df9c8305d6e173d56bf5801115b02375a4077e5f7
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL Z-TER (PANEL_PREREG PART VI): the quantization payoff, CPU.

Simulated round-to-nearest per-output-channel symmetric quantization on the
sensitivity gauges' parameter set; delta-C4 on the frozen instrument.

    python -m mrilab.instruments.z_ter
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import (
    LOSS_DOCS, SEQ_LEN, target_params, loss_on)

SETS = {
    "pythia": [
        ("160m_step40000", "EleutherAI/pythia-160m", "step40000", None),
        ("160m_step143000", "EleutherAI/pythia-160m", "step143000", None),
        ("160m_rejuv", None, None,
         "experiments/regimes/rejuv_artifact_3e-5"),
    ],
    # Z-TER-B v1: the pricing claim on a second family (dL_abs stored:
    # 0.0296 / 0.3706 / 0.5437 — an 18x spread).
    "olmo": [
        ("olmo_210B", "allenai/OLMo-2-0425-1B",
         "stage1-step100000-tokens210B", None),
        ("olmo_4001B", "allenai/OLMo-2-0425-1B",
         "stage1-step1907359-tokens4001B", None),
        ("olmo_stage2_51B", "allenai/OLMo-2-0425-1B",
         "stage2-ingredient1-step23852-tokens51B", None),
    ],
}
ARTIFACTS = SETS["pythia"]
BITS = {"int8": 127, "int4": 7}
OUT = "experiments/regimes/zter_out"


def rtn(w: torch.Tensor, qmax: int) -> torch.Tensor:
    """Per-output-channel (dim 0) symmetric round-to-nearest."""
    scale = w.abs().amax(dim=tuple(range(1, w.ndim)), keepdim=True) / qmax
    scale = scale.clamp_min(1e-12)
    return (w / scale).round().clamp(-qmax, qmax) * scale


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="pythia", choices=sorted(SETS))
    args = ap.parse_args()
    global ARTIFACTS, OUT
    ARTIFACTS = SETS[args.set]
    if args.set != "pythia":
        OUT = f"experiments/regimes/zterb_out_{args.set}"
    os.makedirs(OUT, exist_ok=True)
    for label, repo, rev, path in ARTIFACTS:
        row_p = os.path.join(OUT, f"row_{label}.json")
        if os.path.isfile(row_p):
            print(f"SKIP {label}")
            continue
        src = path or repo
        tok = AutoTokenizer.from_pretrained(src)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = {"revision": rev} if rev else {}
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.float32, **kw).eval()
        from mrilab.datasets.fdgc_datasets import load_generic
        ids = load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
        names = target_params(model)
        pmap = {n: p for n, p in model.named_parameters() if n in names}
        orig = {n: p.detach().clone() for n, p in pmap.items()}
        base = loss_on(model, ids, "cpu", 4)
        row = {"label": label, "base_c4": base}
        for tag, qmax in BITS.items():
            for n, p in pmap.items():
                p.data.copy_(rtn(orig[n], qmax))
            row[tag] = loss_on(model, ids, "cpu", 4) - base
            for n, p in pmap.items():
                p.data.copy_(orig[n])
            print(f"{label} {tag}: dC4 {row[tag]:+.4f} (base {base:.4f})",
                  flush=True)
        with open(row_p, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)
        del model, pmap, orig

    rows = {}
    for label, *_ in ARTIFACTS:
        p = os.path.join(OUT, f"row_{label}.json")
        if os.path.isfile(p):
            rows[label] = json.load(open(p))
    if len(rows) == 3:
        print("\n" + "=" * 70)
        print(f"CELL Z-TER scorecard (set={args.set})")
        print("=" * 70)
        for l, r in rows.items():
            print(f"  {l:<16} base {r['base_c4']:.4f}  int8 {r['int8']:+.4f}"
                  f"  int4 {r['int4']:+.4f}")
        if args.set == "pythia":
            par = rows["160m_step143000"]["int4"]
            rej = rows["160m_rejuv"]["int4"]
            ctl = rows["160m_step40000"]["int4"]
            ratio = rej / par if par > 0 else float("nan")
            print(f"\n-- P-ZT1: int4(rejuv)/int4(parent) = {ratio:.2f} "
                  f"(HIT <= 0.60, KILL >= 0.83) -> "
                  f"{'HIT' if ratio <= 0.60 else ('KILL — flatness prices nothing practical' if ratio >= 0.83 else 'PARTIAL')}")
            print(f"-- P-ZT2 ordering parent > rejuv > step40000: "
                  f"{par:.4f} > {rej:.4f} > {ctl:.4f} : "
                  f"{'HIT' if par > rej > ctl else 'MISS'}")
        elif args.set == "olmo":
            a = rows["olmo_210B"]["int4"]
            b = rows["olmo_4001B"]["int4"]
            c = rows["olmo_stage2_51B"]["int4"]
            print(f"\n-- P-ZTB1 ordering (dL_abs 0.0296 < 0.3706 < 0.5437):"
                  f" int4 {a:.4f} < {b:.4f} < {c:.4f} : "
                  f"{'HIT' if a < b < c else 'MISS'}")
            if a >= c:
                print("-- KILL: the 18x-separated extremes invert — the "
                      "pricing is pythia-specific")
        print("=" * 70)


if __name__ == "__main__":
    main()
