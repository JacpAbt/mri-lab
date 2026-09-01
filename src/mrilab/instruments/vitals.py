# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/vitals.py
# source sha256: 6efecde1839346bb8eeea1ad64edd97c2ca7d53adfdc75b0cd88a47a30c8cc64
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""THE VITALS PANEL (PANEL_PREREG_2026_08.md PART II).

One command, one checkpoint, one JSON row, one printed panel. Every gauge is
the frozen instrument from the cell that built it, imported from that cell's
module rather than re-implemented, so a panel row REPRODUCES a ladder row:

    C        held-out C4 loss (32 docs, seq 512, batch 4, fp32)   [38 / L]
    R        readiness: dialect unseen gain at k=16, n=96         [37 / 38]
    dL_abs   ABS gamma=5e-3, 5 seeds — the tr(H) face             [L]
    S_wt     REL rho=0.10,   5 seeds — the tr(H)*||W||^2/N face   [L]
    S_grad   EXPLORATORY: response along the loss gradient at
             matched per-matrix relative norm; first-order and
             directional-curvature components                     [new]

Seed discipline: the ABS/REL draws use THE EXACT CELL L SEED INDICES
(abs gamma-5e-3 -> 102000+s, rel rho-0.10 -> 201000+s), so on any Cell L
checkpoint the panel must reproduce the stored row within GPU nondeterminism
(P-PAN1). --full-grids runs the entire ladder grid with the ladder's own
seed formula (identical rows by construction).

    python -m mrilab.instruments.vitals \
        --repo EleutherAI/pythia-160m --revision step40000 \
        --device cuda --output experiments/regimes/vitals_out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import (
    GAMMAS, RHOS, PRIMARY_RHO, LOSS_DOCS, SEQ_LEN,
    target_params, loss_on, perturb, restore, loglog_slope)
from mrilab.instruments.icl_readiness import items_for, ctx_prefix, ci
from mrilab.instruments.honesty_battery import _pair_margins

PRIMARY_GAMMA = 5e-3
# Exact Cell L seed bases: seed = 100000*(ci+1) + 1000*mi + s, where ci is the
# convention index (abs=0, rel=1) and mi the magnitude's index in the ladder's
# default grid. gamma=5e-3 is GAMMAS[2]; rho=0.10 is RHOS[1].
SEED_ABS = 100000 * 1 + 1000 * GAMMAS.index(PRIMARY_GAMMA)
SEED_REL = 100000 * 2 + 1000 * RHOS.index(PRIMARY_RHO)
N_SEEDS = 5
ADVISORY = ("WITHDRAWN 2026-08-27 (CELL W MISS, ratio 1.79x vs band "
            "[4.0,36.1]): dL_abs does NOT set the usable-LR scale across "
            "provenances — a warmup-sharp checkpoint tolerated ~the same LR "
            "as a 12x-flatter mid-run one. The U-prime iso-damage relation "
            "held anneal-sharp vs mid only (1 pair). Candidate restatement, "
            "UNTESTED: the rule applies only at low anisotropy "
            "(S_grad/S_wt small, i.e. annealed endpoints).")


def grad_on(model, ids: torch.Tensor, dev: str, bs: int) -> None:
    """Accumulate d(mean loss)/dW into .grad, batching like loss_on."""
    model.zero_grad(set_to_none=True)
    n_total = ids.shape[0]
    for i in range(0, n_total, bs):
        b = ids[i:i + bs].to(dev)
        loss = model(input_ids=b, labels=b).loss * (b.shape[0] / n_total)
        loss.backward()


# Panel v1.1 (2026-08-27, dated in PANEL_PREREG PART II): --offload-orig
# keeps the pristine-weights copy on CPU so >2B-parameter models fit the
# 24GB card. BIT-COMPATIBLE BY CONSTRUCTION with sensitivity_ladder.perturb:
# same per-call generator seeding, same iteration order, noise created on
# the same device with the same shape/dtype, fp32 CPU<->GPU roundtrip exact.
# Verified against a stored row before first use (run_after_chain.sh).
def perturb_offload(pmap, orig_cpu, conv, mag, seed, dev) -> None:
    g = torch.Generator(device=dev).manual_seed(seed)
    for name, p in pmap.items():
        w = orig_cpu[name].to(dev)
        noise = torch.randn(w.shape, generator=g, dtype=torch.float32,
                            device=dev)
        if conv == "abs":
            noise.mul_(mag)
        else:
            noise.mul_(mag * w.norm() / (noise.norm() + 1e-12))
        p.data.copy_(w + noise)


def restore_offload(pmap, orig_cpu, dev) -> None:
    for name, p in pmap.items():
        p.data.copy_(orig_cpu[name].to(dev))


def sens_block(model, pmap, orig, ids, dev: str, bs: int, base: float,
               conv: str, mag: float, seed_base: int,
               n_seeds: int, offload: bool = False) -> Dict:
    pert = perturb_offload if offload else perturb
    vals = []
    for s in range(n_seeds):
        pert(pmap, orig, conv, mag, seed=seed_base + s, dev=dev)
        vals.append(loss_on(model, ids, dev, bs) - base)
    if offload:
        restore_offload(pmap, orig, dev)
    else:
        restore(pmap, orig)
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "sd": sd, "vals": vals}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="HF repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--path", default=None,
                    help="local snapshot dir (alternative to --repo)")
    ap.add_argument("--tokenizer", default=None,
                    help="load the tokenizer from this repo instead (for "
                         "checkpoint branches that ship without one; use "
                         "the family's own tokenizer)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n", type=int, default=96, help="readiness items")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--skip-readiness", action="store_true")
    ap.add_argument("--skip-sens", action="store_true")
    ap.add_argument("--skip-grad", action="store_true")
    ap.add_argument("--offload-orig", action="store_true",
                    help="pristine weights kept on CPU (>2B models on a "
                         "24GB card); bit-compatible, see panel v1.1 note")
    ap.add_argument("--full-grids", action="store_true",
                    help="run the entire Cell L grid (beta/c_rel)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not args.repo and not args.path:
        raise SystemExit("need --repo or --path")

    src = args.path or args.repo
    label = (os.path.basename(src.rstrip("/")) if args.path
             else f"{args.repo.split('/')[-1]}@{args.revision or 'main'}")
    os.makedirs(args.output, exist_ok=True)
    row_p = os.path.join(args.output, f"vitals_{label}.json")
    if os.path.isfile(row_p):
        print(f"SKIP (exists): {row_p}")
        return

    tok = AutoTokenizer.from_pretrained(args.tokenizer or src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"revision": args.revision} if args.repo and args.revision else {}
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.float32, **kw).to(args.device).eval()

    from mrilab.datasets.fdgc_datasets import load_generic
    ids = load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
    batch_hash = hashlib.sha256(
        ids.cpu().numpy().tobytes()).hexdigest()[:16]

    row: Dict = {"label": label, "repo": args.repo, "path": args.path,
                 "revision": args.revision, "device": args.device,
                 "batch_hash": batch_hash}

    # ---- C ----------------------------------------------------------------
    base = loss_on(model, ids, args.device, args.batch_size)
    row["loss"] = base
    print(f"[C]      held-out loss {base:.4f}   (batch {batch_hash})",
          flush=True)

    # ---- R ----------------------------------------------------------------
    if not args.skip_readiness:
        ev = {s: items_for(s, args.n, 1234) for s in ("seen", "unseen")}
        gen = torch.Generator().manual_seed(0)
        marg = {}
        for k in (0, args.k):
            pre = ctx_prefix(k, 0)
            for s, items in ev.items():
                mod = [(pre + p, g, b) for p, g, b in items]
                marg[(k, s)] = _pair_margins(model, tok, mod, args.device,
                                             max(args.batch_size, 8), 640)
        gs = marg[(args.k, "seen")] - marg[(0, "seen")]
        gu = marg[(args.k, "unseen")] - marg[(0, "unseen")]
        row["R"] = {"k": args.k, "n": args.n,
                    "gain_seen": float(gs.mean()),
                    "gain_seen_ci": ci(gs, gen),
                    "gain_unseen": float(gu.mean()),
                    "gain_unseen_ci": ci(gu, gen),
                    "G_icl": (float(gu.mean()) / float(gs.mean())
                              if abs(float(gs.mean())) > 1e-9
                              else float("nan"))}
        print(f"[R]      unseen gain {row['R']['gain_unseen']:+.3f} nats "
              f"(seen {row['R']['gain_seen']:+.3f}, "
              f"G_icl {row['R']['G_icl']:+.3f})", flush=True)

    # ---- sensitivity -------------------------------------------------------
    names = target_params(model)
    pmap = {n: p for n, p in model.named_parameters() if n in names}
    orig = {n: (p.detach().clone().cpu() if args.offload_orig
                else p.detach().clone()) for n, p in pmap.items()}
    row["n_matrices"] = len(names)
    row["n_params"] = int(sum(w.numel() for w in orig.values()))
    row["total_fro"] = float(sum(float(w.norm()) ** 2
                                 for w in orig.values()) ** 0.5)
    row["w2n"] = row["total_fro"] ** 2 / row["n_params"]
    row["offload_orig"] = bool(args.offload_orig)

    if not args.skip_sens:
        if args.full_grids:
            for ci_, (conv, mags) in enumerate((("abs", GAMMAS),
                                                ("rel", RHOS))):
                row[conv] = {}
                for mi, mag in enumerate(mags):
                    row[conv][f"{mag:g}"] = sens_block(
                        model, pmap, orig, ids, args.device,
                        args.batch_size, base, conv, mag,
                        100000 * (ci_ + 1) + 1000 * mi, args.seeds,
                        offload=args.offload_orig)
            row["beta_rel"] = loglog_slope(
                RHOS, [row["rel"][f"{r:g}"]["mean"] for r in RHOS])
            row["beta_abs"] = loglog_slope(
                GAMMAS, [row["abs"][f"{g:g}"]["mean"] for g in GAMMAS])
            row["c_rel"] = sum(row["rel"][f"{r:g}"]["mean"] / (r * r)
                               for r in RHOS) / len(RHOS)
            row["dL_abs"] = row["abs"][f"{PRIMARY_GAMMA:g}"]["mean"]
            row["S_wt"] = row["rel"][f"{PRIMARY_RHO:g}"]["mean"]
        else:
            a = sens_block(model, pmap, orig, ids, args.device,
                           args.batch_size, base, "abs", PRIMARY_GAMMA,
                           SEED_ABS, args.seeds, offload=args.offload_orig)
            r = sens_block(model, pmap, orig, ids, args.device,
                           args.batch_size, base, "rel", PRIMARY_RHO,
                           SEED_REL, args.seeds, offload=args.offload_orig)
            row["abs"] = {f"{PRIMARY_GAMMA:g}": a}
            row["rel"] = {f"{PRIMARY_RHO:g}": r}
            row["dL_abs"] = a["mean"]
            row["S_wt"] = r["mean"]
        print(f"[dL_abs] {row['dL_abs']:+.4f}   [S_wt] {row['S_wt']:+.4f}",
              flush=True)

    # ---- S_grad (exploratory) ---------------------------------------------
    if not args.skip_grad:
        grad_on(model, ids, args.device, args.batch_size)
        gmap = {n: p.grad.detach().clone() for n, p in pmap.items()
                if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.eval()
        deltas = []
        for sign in (+1.0, -1.0):
            for n, p in pmap.items():
                g = gmap[n]
                w = (orig[n].to(args.device) if args.offload_orig
                     else orig[n])
                d = g * (sign * PRIMARY_RHO * w.norm() / (g.norm() + 1e-12))
                p.data.copy_(w + d)
            deltas.append(loss_on(model, ids, args.device,
                                  args.batch_size) - base)
        if args.offload_orig:
            restore_offload(pmap, orig, args.device)
        else:
            restore(pmap, orig)
        row["S_grad"] = {"rho": PRIMARY_RHO,
                         "dL_plus": deltas[0], "dL_minus": deltas[1],
                         "first_order": (deltas[0] - deltas[1]) / 2,
                         "curvature": (deltas[0] + deltas[1]) / 2}
        print(f"[S_grad] first-order {row['S_grad']['first_order']:+.4f}  "
              f"curvature {row['S_grad']['curvature']:+.4f}  "
              f"(vs S_wt {row.get('S_wt', float('nan')):+.4f})", flush=True)

    # ---- derived + panel ---------------------------------------------------
    if "R" in row and "S_wt" in row and row["S_wt"] > 0:
        row["R_over_S"] = row["R"]["gain_unseen"] / row["S_wt"]
    row["advisory"] = ADVISORY

    with open(row_p, "w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)
    assert os.path.isfile(row_p)

    print("\n" + "=" * 72)
    print(f"VITALS  {label}   (device {args.device})")
    print("=" * 72)
    print(f"  C        {row['loss']:.4f} nats  (held-out C4, frozen 32-doc)")
    if "R" in row:
        print(f"  R        {row['R']['gain_unseen']:+.4f} nats unseen gain "
              f"@k={args.k}  (G_icl {row['R']['G_icl']:+.3f})")
    if "dL_abs" in row:
        print(f"  dL_abs   {row['dL_abs']:+.4f}  (curvature face)")
        print(f"  S_wt     {row['S_wt']:+.4f}  (curvature x weight-scale)")
    if "S_grad" in row:
        print(f"  S_grad   {row['S_grad']['curvature']:+.4f} curv / "
              f"{row['S_grad']['first_order']:+.4f} 1st-order  [exploratory]")
    if "R_over_S" in row:
        print(f"  R/S_wt   {row['R_over_S']:+.2f}  (frontier coordinate)")
    print(f"  ADVISORY {ADVISORY}")
    print("=" * 72)
    print(f"wrote {row_p}")


if __name__ == "__main__":
    main()
