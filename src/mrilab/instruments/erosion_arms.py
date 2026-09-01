# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/erosion_arms.py
# source sha256: 3f514ab962a67ccaf0de77a92fbd8107b501b01365bdeeefb26f986af996eb51
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL U (INSTRUMENT_PREREG_2026_08.md addendum): DOES S_wt PREDICT REAL
FINE-TUNING DAMAGE?

The gauge is only worth something if it predicts what a fine-tune actually
does. A natural experiment is already on disk:

    pythia-160m step40000   held-out loss 3.3763   S_wt 0.1362
    pythia-160m step143000  held-out loss 3.3796   S_wt 2.1588

Capability matched to 0.003 nats — below the loss instrument's own noise floor
— with a 16x gap in measured curvature. Fine-tune both identically on an
off-distribution corpus and watch two things at once:

    DAMAGE  held-out C4 loss rises (forgetting), measured on the FROZEN
            32-document instrument so it joins the CELL L axis exactly
    GAIN    target-domain held-out loss falls (learning)

If S_wt is a genuine gain parameter rather than a fragility curiosity, it must
predict BOTH — the same weight step should buy more learning AND more
forgetting in the high-S substrate. That symmetry is the susceptibility
account's whole content, and this is the first test of it against a real
optimizer step rather than random noise (the limitation CELL M had to carry).

Harness lessons inherited from the steward G0 gate (blueprint section 24):
explicit float32 load, linear warmup before constant LR, hard abort on
non-finite loss.

    python -m mrilab.instruments.erosion_arms \
        --repo EleutherAI/pythia-160m --revision step40000 --lr 3e-5 \
        --output experiments/regimes/erosion/s40000_lr3e-5
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Frozen to the section-38 / CELL L instrument so the axis is shared.
LOSS_DOCS = 32
SEQ_LEN = 512


def eval_loss(model, ids: torch.Tensor, dev: str, bs: int) -> float:
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
    ap.add_argument("--repo", default="EleutherAI/pythia-160m")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--target", default="code")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    # Gradient accumulation, added 2026-08-25 after the first Cell U run failed
    # its own sanity gate on all four arms. A 4x512 batch is ~2048 tokens
    # against Pythia's pretraining batch of ~2M — a 1000x reduction — so the
    # updates were gradient NOISE rather than gradient SIGNAL, and the models
    # degraded on the target they were being trained on.
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-target-eval", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    # Big-substrate variant (2026-08-28): 8-bit Adam state so >=1B models
    # fit the 24GB card. Forward/backward stay fp32, so only the OPTIMIZER
    # STATE precision differs from the frozen crash protocol. DECLARED
    # VARIANT: every comparison using it must use it for ALL of its arms,
    # and the row records the flag.
    ap.add_argument("--adam8bit", action="store_true")
    # CELL EQ addition (2026-08-27): measure dL_abs / S_wt on the final
    # weights with the exact Cell L seeds (the equilibrium-sharpness gauge).
    # Default off; the frozen erosion behaviour is unchanged without it.
    ap.add_argument("--end-sens", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    done = os.path.join(args.output, "erosion.json")
    if os.path.isfile(done):
        print(f"SKIP (exists): {done}")
        return

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from mrilab.datasets import fdgc_datasets as D
    c4 = D.load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
    # The target must be OFF the pretraining distribution. `codeheavy` is NOT:
    # the 2026-08-25 smoke showed it is a MIXTURE (1523 code + 653 c4 docs), so
    # fine-tuning on it IMPROVED held-out C4 (-0.142 nats in 50 steps) and
    # "damage" came out negative. Pure single-domain loaders only.
    PURE = {"code": D.load_code_dataset, "legal": D.load_legal_dataset,
            "math": D.load_math_dataset,
            "finance": D.load_finance_news_dataset,
            # wikitext added 2026-08-28 for DP v1.1's fifth corpus. It
            # qualifies under the original gate's REASON (single-source,
            # contains no C4 documents) even though it is stylistically
            # NEAR C4 — which is exactly why it was chosen: it extends the
            # profile's range at the low-disruption end and breaks v1's
            # near-tie middle pairs. Expect small damage by construction.
            "wikitext": lambda tok, sl, n: D.load_wikitext_data(
                tok, split="train", max_seq_len=sl, num_samples=n)}
    if args.target not in PURE:
        raise SystemExit(f"--target must be one of {sorted(PURE)}; mixed "
                         f"generics contain C4 and confound damage with gain")
    # Cache the tokenised pool. Two reasons, and the second is the important
    # one: (1) streaming 12k code chunks took ~20 min per arm and left the GPU
    # at 8%; (2) each arm re-streamed, so the arms were NOT guaranteed to see
    # identical data — and "matched data across arms" is a load-bearing part of
    # this design, not a convenience.
    n_need = args.n_train + args.n_target_eval
    cache = os.path.join("experiments/regimes",
                         f"_pool_{args.target}_{n_need}_{SEQ_LEN}.pt")
    if os.path.isfile(cache):
        pool = torch.load(cache)
        print(f"pool from cache {cache}", flush=True)
    else:
        pool = PURE[args.target](tok, SEQ_LEN, n_need).input_ids
        torch.save(pool, cache)
        print(f"pool cached -> {cache}", flush=True)
    tgt_eval = pool[:args.n_target_eval]
    train = pool[args.n_target_eval:]
    print(f"c4 eval {tuple(c4.shape)}  target train {tuple(train.shape)}  "
          f"target eval {tuple(tgt_eval.shape)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.repo, revision=args.revision, dtype=torch.float32).to(args.device)
    model.train()
    if args.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                  weight_decay=0.1)
        print("optimizer: bnb AdamW8bit (declared variant)", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.1)

    hist: List[Dict] = []

    def record(step: int) -> None:
        row = {"step": step,
               "c4": eval_loss(model, c4, args.device, args.batch_size),
               "target": eval_loss(model, tgt_eval, args.device,
                                   args.batch_size)}
        hist.append(row)
        print(f"  step {step:4d}: c4 {row['c4']:.4f}  "
              f"target {row['target']:.4f}", flush=True)

    record(0)
    g = torch.Generator().manual_seed(args.seed)
    n = train.shape[0]
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(args.accum):
            idx = torch.randint(0, n, (args.batch_size,), generator=g)
            b = train[idx].to(args.device)
            loss = model(input_ids=b, labels=b).loss / args.accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} — aborting "
                                 f"(steward G0 gate lesson, section 24)")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.eval_every == 0:
            record(step)

    base = hist[0]
    final = hist[-1]
    out = {"config": vars(args), "history": hist,
           "c4_base": base["c4"], "c4_final": final["c4"],
           "damage": final["c4"] - base["c4"],
           "target_base": base["target"], "target_final": final["target"],
           "gain": base["target"] - final["target"]}

    if args.end_sens:
        from mrilab.instruments.sensitivity_ladder import (
            target_params, loss_on, perturb, restore)
        from mrilab.instruments.vitals import (
            SEED_ABS, SEED_REL, PRIMARY_GAMMA)
        from mrilab.instruments.sensitivity_ladder import PRIMARY_RHO
        model.eval()
        names = target_params(model)
        pmap = {n: p for n, p in model.named_parameters() if n in names}
        orig = {n: p.detach().clone() for n, p in pmap.items()}
        sens_base = eval_loss(model, c4, args.device, args.batch_size)
        model.eval()
        for key, conv, mag, sbase in (
                ("end_dL_abs", "abs", PRIMARY_GAMMA, SEED_ABS),
                ("end_S_wt", "rel", PRIMARY_RHO, SEED_REL)):
            vals = []
            for s in range(5):
                perturb(pmap, orig, conv, mag, seed=sbase + s,
                        dev=args.device)
                model.eval()
                vals.append(eval_loss(model, c4, args.device,
                                      args.batch_size) - sens_base)
            restore(pmap, orig)
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / 4) ** 0.5
            out[key] = {"mean": m, "sd": sd, "vals": vals}
            print(f"  {key}: {m:+.4f} +- {sd:.4f}", flush=True)
    out["damage_per_gain"] = (out["damage"] / out["gain"]
                              if abs(out["gain"]) > 1e-9 else float("nan"))
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    assert os.path.isfile(done)
    print(f"\n{args.revision} lr={args.lr:g}:  damage {out['damage']:+.4f}  "
          f"gain {out['gain']:+.4f}  damage/gain {out['damage_per_gain']:+.4f}")
    print(f"wrote {done}")


if __name__ == "__main__":
    main()
