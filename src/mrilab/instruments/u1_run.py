# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/u1_run.py
# source sha256: dc9b75de90eff763a489408b3707dc5c47ad0c155bc212fd83cf2d694f78d0e0
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""DEMO-CL / union U1 (PANEL_PREREG PART VI): the readiness-triggered rule
phase inside one compressed-traversal run.

    python -m mrilab.instruments.u1_run --build-only        # pools, CPU ok
    python -m mrilab.instruments.u1_run --arm trigger
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.icl_readiness import items_for, ctx_prefix, ci
from mrilab.instruments.honesty_battery import _pair_margins

SEQ_LEN = 512
POOL_N = 40960
RULE_STEPS = 50          # rule batches, injected every 10th step
RULE_CADENCE = 10
STEPS = 2500
WARMUP = 100
LR = 3e-5
PROBE_EVERY_TRIGGER = 100
PROBE_EVERY_RECORD = 250
TRIGGER_NATS = 1.0
FALLBACK_STEP = 2000
SWITCH_AT = {"control": None, "fixed20": 500, "fixed50": 1250,
             "trigger": "probe"}
POOL_CACHE = f"experiments/regimes/_pool_u1_{POOL_N}_{SEQ_LEN}.pt"
RULE_CACHE = f"experiments/regimes/_pool_u1_rule_{SEQ_LEN}.pt"


def build_pools(tok):
    if os.path.isfile(POOL_CACHE):
        pool = torch.load(POOL_CACHE)
    else:
        from mrilab.datasets.fdgc_datasets import load_generic
        pool = load_generic("c4", tok, SEQ_LEN, POOL_N + 32).input_ids[32:]
        torch.save(pool, POOL_CACHE)
    if os.path.isfile(RULE_CACHE):
        rule = torch.load(RULE_CACHE)
    else:
        items = items_for("seen", 512, 0)
        text = "\n".join(p + g for p, g, _ in items)
        ids = tok(text, return_tensors="pt")["input_ids"][0]
        n = ids.shape[0] // SEQ_LEN
        rule = ids[: n * SEQ_LEN].reshape(n, SEQ_LEN)
        torch.save(rule, RULE_CACHE)
    return pool, rule


def probe_R(model, tok, dev, n=24, k=16) -> float:
    """Unseen-member gain at k demos, the live trigger sensor."""
    pre = ctx_prefix(k, 0)
    out = {}
    for s in ("seen", "unseen"):
        items = items_for(s, n, 1234)
        m0 = _pair_margins(model, tok, items, dev, 8, 640)
        mk = _pair_margins(model, tok,
                           [(pre + p, g, b) for p, g, b in items],
                           dev, 8, 640)
        out[s] = float((mk - m0).mean())
    return out["unseen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(SWITCH_AT), default=None)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="experiments/regimes/u1_out")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pool, rule = build_pools(tok)
    if args.build_only:
        print(f"pools ready: c4 {tuple(pool.shape)} rule {tuple(rule.shape)}")
        return
    if args.arm is None:
        raise SystemExit("need --arm")
    os.makedirs(args.output, exist_ok=True)
    done = os.path.join(args.output, f"row_{args.arm}.json")
    if os.path.isfile(done):
        print(f"SKIP {args.arm}")
        return

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m", revision="step512",
        dtype=torch.float32).to(args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(pool.shape[0], generator=g)
    cursor = 0

    switch_step = SWITCH_AT[args.arm]
    switched_at = None
    rule_used = 0
    probes: List[Dict] = []

    def next_pool_batch():
        nonlocal cursor, perm
        if cursor + 4 > pool.shape[0]:
            perm = torch.randperm(pool.shape[0], generator=g)
            cursor = 0
        idx = perm[cursor:cursor + 4]
        cursor += 4
        return pool[idx]

    for step in range(1, STEPS + 1):
        lr = LR * min(1.0, step / WARMUP)
        for pg in opt.param_groups:
            pg["lr"] = lr
        # probe: trigger arm every 100 until fired; all arms every 250.
        need_probe = (step % PROBE_EVERY_RECORD == 0) or (
            args.arm == "trigger" and switched_at is None
            and step % PROBE_EVERY_TRIGGER == 0)
        if need_probe:
            model.eval()
            r = probe_R(model, tok, args.device)
            model.train()
            probes.append({"step": step, "R_unseen": r})
            print(f"  probe step {step}: R_unseen {r:+.3f}", flush=True)
            if (args.arm == "trigger" and switched_at is None
                    and r > TRIGGER_NATS):
                switched_at = step
                print(f"  TRIGGER FIRED at step {step}", flush=True)
        if args.arm == "trigger" and switched_at is None \
                and step >= FALLBACK_STEP:
            switched_at = step
            print(f"  FALLBACK switch at {step}", flush=True)
        if isinstance(switch_step, int) and switched_at is None \
                and step >= switch_step:
            switched_at = step

        in_rule_phase = (switched_at is not None and rule_used < RULE_STEPS
                         and (step - switched_at) % RULE_CADENCE == 0)
        for _ in range(8):
            if in_rule_phase:
                idx = torch.randint(0, rule.shape[0], (4,), generator=g)
                b = rule[idx].to(args.device)
            else:
                b = next_pool_batch().to(args.device)
            loss = model(input_ids=b, labels=b).loss / 8
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            loss.backward()
        if in_rule_phase:
            rule_used += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"  step {step} (rule_used {rule_used}, "
                  f"switched_at {switched_at})", flush=True)

    # end readout: k=0 margins on held-out sentences, seen + unseen pairs
    model.eval()
    gen = torch.Generator().manual_seed(0)
    out: Dict = {"arm": args.arm, "switched_at": switched_at,
                 "rule_used": rule_used, "probes": probes}
    for s in ("seen", "unseen"):
        items = items_for(s, 96, 1234)
        m = _pair_margins(model, tok, items, args.device, 8, 640)
        out[f"margin_{s}"] = float(m.mean())
        out[f"margin_{s}_ci"] = ci(m, gen)
        out[f"margin_{s}_vals"] = m.tolist()
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"{args.arm}: switched_at {switched_at}  rule_used {rule_used}  "
          f"seen {out['margin_seen']:+.3f}  unseen {out['margin_unseen']:+.3f}")


if __name__ == "__main__":
    main()
