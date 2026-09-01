# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/l8_arms.py
# source sha256: f68955da51ec0dffbd54b5540277bb2e49cdd6c736d00b6670f09a96e8b07294
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL L8: the cooling-rate law. Three cooling schedules to one
destination; fragility of the endpoints compared.

    python -m mrilab.instruments.l8_arms --arm slow
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import LOSS_DOCS, SEQ_LEN

REPO = "EleutherAI/pythia-410m"
REV = "step8000"
LR_HI, LR_LO = 3e-5, 1e-6
STEPS = 2000
POOL_CHUNKS = 4096
# steps spent at constant LR_HI before the linear ramp to LR_LO begins
RAMP_START = {"fast": 1800, "medium": 1000, "slow": 0}


def lr_at(step: int, arm: str) -> float:
    s0 = RAMP_START[arm]
    if step <= s0:
        return LR_HI
    frac = (step - s0) / (STEPS - s0)
    return LR_HI + (LR_LO - LR_HI) * frac


def eval_loss(model, ids, dev, bs=4) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], bs):
            b = ids[i:i + bs].to(dev)
            tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
            n += b.shape[0]
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(RAMP_START), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    done = os.path.join(args.output, "l8.json")
    wpath = os.path.join(args.output, "final_weights.pt")
    if os.path.isfile(done):
        print("SKIP (exists): " + done)
        return

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Reuse kd_arms' pool builder: pool_for(n) returns n+LOSS_DOCS chunks,
    # cached under the name n, and BUILDS it if missing. This is the
    # existing _pool_kd_c4_4096_512.pt (which already holds 4096+32 chunks).
    from mrilab.instruments.kd_arms import pool_for
    allids = pool_for(tok, POOL_CHUNKS)
    heldout = allids[:LOSS_DOCS]
    pool = allids[LOSS_DOCS:LOSS_DOCS + POOL_CHUNKS]
    epochs = STEPS * args.accum * args.batch_size / pool.shape[0]
    print(f"arm {args.arm}: pool {tuple(pool.shape)} ({epochs:.1f} epochs)",
          flush=True)
    assert epochs < 16, "epoch trap"

    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REV, dtype=torch.float32).to(args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_HI, weight_decay=0.1)
    base_c4 = eval_loss(model, heldout, args.device)
    print(f"base C4 {base_c4:.4f}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    n = pool.shape[0]
    hist = []
    for step in range(1, STEPS + 1):
        lr = lr_at(step, args.arm)
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(args.accum):
            idx = torch.randint(0, n, (args.batch_size,), generator=g)
            b = pool[idx].to(args.device)
            loss = model(input_ids=b, labels=b).loss / args.accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            c4 = eval_loss(model, heldout, args.device)
            hist.append({"step": step, "lr": lr, "c4": c4})
            print(f"  step {step}: lr {lr:.2e}  c4 {c4:.4f}", flush=True)

    final_c4 = eval_loss(model, heldout, args.device)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, wpath)
    out = {"config": vars(args), "arm": args.arm, "repo": REPO, "rev": REV,
           "lr_hi": LR_HI, "lr_lo": LR_LO, "steps": STEPS,
           "ramp_start": RAMP_START[args.arm], "epochs": epochs,
           "base_c4": base_c4, "final_c4": final_c4, "history": hist,
           "weights": wpath}
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"{args.arm}: c4 {base_c4:.4f} -> {final_c4:.4f}; weights saved "
          f"for the fragility readout")


if __name__ == "__main__":
    main()
