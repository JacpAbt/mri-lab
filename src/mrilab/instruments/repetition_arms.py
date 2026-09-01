# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/repetition_arms.py
# source sha256: dba4dda4db3cccf6f90678e4149bca9c617d8157188962f2c46854f5b4ab9f92
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL X (PANEL_PREREG_2026_08.md PART IV): REPETITION vs FRESH.

Does repeated data sharpen the basin where fresh data does not, at matched
tokens, steps, LR and schedule? Three arms on pythia-160m@step40000:

    fresh   pool 16,000 chunks  = exactly  1 epoch
    rep8    pool  2,000 chunks  = exactly  8 epochs
    rep64   pool    250 chunks  = exactly 64 epochs

All at constant LR 3e-5 (50-step warmup), 500 optimizer steps, accum 8,
batch 4, seq 512 = 8.19M tokens/arm. Pools are NESTED PREFIXES of one master
C4 pool; the master's first 32 chunks are asserted equal to the frozen eval
instrument and excluded from training. Sampling is EPOCH-EXACT (permutation
per epoch, no replacement) so the fresh arm is exactly 1.0 epochs.

End-of-run: dL_abs (gamma 5e-3) and S_wt (rho 0.10), 5 seeds each, THE EXACT
CELL L SEED INDICES, measured in-process. Baseline = Cell L's stored
step40000 row.

    python -m mrilab.instruments.repetition_arms --build-only   # pool, CPU ok
    python -m mrilab.instruments.repetition_arms --arm fresh \
        --output experiments/regimes/repetition_out
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import (
    target_params, loss_on, perturb, restore)
from mrilab.instruments.vitals import SEED_ABS, SEED_REL, PRIMARY_GAMMA
from mrilab.instruments.sensitivity_ladder import PRIMARY_RHO

LOSS_DOCS = 32
SEQ_LEN = 512
N_EVAL_EXCLUDE = 32          # frozen-instrument chunks, excluded from pools
N_TRAIN_MASTER = 16000       # fresh arm = the whole master train pool
POOLS = {"fresh": 16000, "rep2": 8000, "rep4": 4000, "rep8": 2000,
         "rep64": 250}
PROBE_N = 32                 # shared memorization probe = train[0:32]
MASTER_CACHE = os.path.join("experiments/regimes",
                            f"_pool_c4rep_{N_EVAL_EXCLUDE + N_TRAIN_MASTER}"
                            f"_{SEQ_LEN}.pt")


def build_master(tok) -> torch.Tensor:
    if os.path.isfile(MASTER_CACHE):
        return torch.load(MASTER_CACHE)
    from mrilab.datasets.fdgc_datasets import load_generic
    n = N_EVAL_EXCLUDE + N_TRAIN_MASTER
    print(f"building master pool ({n} chunks) — one-time, cached", flush=True)
    master = load_generic("c4", tok, SEQ_LEN, n).input_ids
    torch.save(master, MASTER_CACHE)
    print(f"cached -> {MASTER_CACHE}", flush=True)
    return master


def eval_loss(model, ids, dev, bs):
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(POOLS), default=None)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--repo", default="EleutherAI/pythia-160m")
    ap.add_argument("--revision", default="step40000")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="experiments/regimes/repetition_out")
    # CELL Y / Z-bis additions (2026-08-27): an optional linear LR decay
    # over the tail of the run, and an optional weight snapshot at the end.
    # Defaults preserve the original constant-LR, no-save behaviour exactly.
    ap.add_argument("--decay-start", type=int, default=None,
                    help="step at which linear decay begins (default: none)")
    ap.add_argument("--final-lr-frac", type=float, default=0.0,
                    help="LR at the last step, as a fraction of --lr")
    ap.add_argument("--save-dir", default=None,
                    help="save model+tokenizer here at run end")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    master = build_master(tok)
    if args.build_only:
        print(f"master pool ready: {tuple(master.shape)}")
        return
    if args.arm is None:
        raise SystemExit("need --arm (or --build-only)")

    os.makedirs(args.output, exist_ok=True)
    done = os.path.join(args.output, f"row_{args.arm}.json")
    if os.path.isfile(done):
        print(f"SKIP (exists): {done}")
        return

    # The eval-leak check is code, not assumption (PART IV): master[:32] must
    # BE the frozen instrument, and is then excluded from every pool.
    from mrilab.datasets.fdgc_datasets import load_generic
    c4 = load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
    assert torch.equal(master[:N_EVAL_EXCLUDE], c4), \
        "master pool prefix != frozen eval instrument; pool build not " \
        "prefix-stable — Cell X void, fix the loader"
    train = master[N_EVAL_EXCLUDE:]
    pool = train[:POOLS[args.arm]]
    probe = train[:PROBE_N]
    epochs_planned = args.steps * args.accum * args.batch_size / pool.shape[0]
    print(f"arm {args.arm}: pool {pool.shape[0]} chunks, "
          f"{epochs_planned:.1f} epochs planned", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.repo, revision=args.revision,
        dtype=torch.float32).to(args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(args.seed)

    hist: List[Dict] = []

    def record(step: int, epoch: float) -> None:
        row = {"step": step, "epoch": round(epoch, 2),
               "c4": eval_loss(model, c4, args.device, args.batch_size),
               "probe": eval_loss(model, probe, args.device,
                                  args.batch_size)}
        hist.append(row)
        print(f"  step {step:4d} (ep {epoch:5.1f}): c4 {row['c4']:.4f}  "
              f"probe {row['probe']:.4f}", flush=True)

    record(0, 0.0)
    # Epoch-exact sampling: permutation per epoch, sequential micro-batches.
    perm = torch.randperm(pool.shape[0], generator=g)
    cursor = 0
    seen = 0
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        if args.decay_start is not None and step > args.decay_start:
            frac = ((step - args.decay_start)
                    / max(args.steps - args.decay_start, 1))
            lr = args.lr * (1.0 - (1.0 - args.final_lr_frac) * frac)
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(args.accum):
            if cursor + args.batch_size > pool.shape[0]:
                perm = torch.randperm(pool.shape[0], generator=g)
                cursor = 0
            idx = perm[cursor:cursor + args.batch_size]
            cursor += args.batch_size
            seen += args.batch_size
            b = pool[idx].to(args.device)
            loss = model(input_ids=b, labels=b).loss / args.accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} — abort "
                                 f"(G0 gate lesson)")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.eval_every == 0:
            record(step, seen / pool.shape[0])

    # End-of-run sensitivity, exact Cell L seeds, in-process.
    model.eval()
    names = target_params(model)
    pmap = {n: p for n, p in model.named_parameters() if n in names}
    orig = {n: p.detach().clone() for n, p in pmap.items()}
    base = loss_on(model, c4, args.device, args.batch_size)
    sens: Dict = {}
    for key, conv, mag, sbase in (("dL_abs", "abs", PRIMARY_GAMMA, SEED_ABS),
                                  ("S_wt", "rel", PRIMARY_RHO, SEED_REL)):
        vals = []
        for s in range(args.seeds):
            perturb(pmap, orig, conv, mag, seed=sbase + s, dev=args.device)
            vals.append(loss_on(model, c4, args.device,
                                args.batch_size) - base)
        restore(pmap, orig)
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        sens[key] = {"mean": m, "sd": sd, "vals": vals}
        print(f"  end {key}: {m:+.4f} +- {sd:.4f}", flush=True)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"saved weights -> {args.save_dir}", flush=True)

    out = {"config": vars(args), "pool_chunks": int(pool.shape[0]),
           "epochs": round(seen / pool.shape[0], 2), "history": hist,
           "c4_base": hist[0]["c4"], "c4_final": hist[-1]["c4"],
           "probe_base": hist[0]["probe"], "probe_final": hist[-1]["probe"],
           "end_loss": base, "dL_abs": sens["dL_abs"], "S_wt": sens["S_wt"]}
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    assert os.path.isfile(done)
    print(f"wrote {done}")


if __name__ == "__main__":
    main()
