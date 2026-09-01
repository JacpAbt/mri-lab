# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/sensitivity_ladder.py
# source sha256: f896230054701eaaf27f9e5a3c0c6368c26b05516a7ec06d339358f2dde705bf
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL L (INSTRUMENT_PREREG_2026_08.md): the SENSITIVITY LADDER.

Measures S_wt(t) = held-out-loss response to a fixed-size weight perturbation
across the same checkpoints as the section-38 readiness surface, so the two
gauges join on one capability axis.

Two conventions, both reported:
  ABS  delta_W ~ N(0, gamma^2) elementwise  (the published convention;
       comparable WITHIN one (family, size) only)
  REL  delta_W scaled per matrix to ||delta_W||_F / ||W||_F = rho
       (comparable ACROSS sizes; this is the normalization section 26's E1
       decomposition says matters, since weight norms grow with maturity)

The held-out loss instrument is FROZEN to the section-38 settings
(c4, 32 docs, seq_len 512, batch 4, float32) so every point joins the
readiness surface exactly. delta_L is a PAIRED difference computed on the
identical token batches.

    python -m mrilab.instruments.sensitivity_ladder \
        --repo EleutherAI/pythia-160m \
        --revisions step512 step1000 step2000 step8000 step40000 step143000 \
        --device cuda --output experiments/regimes/sens_ladder_160m
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Frozen to match icl_readiness.heldout_loss exactly (section 38's axis).
LOSS_DOCS = 32
SEQ_LEN = 512
GAMMAS = [1e-3, 2e-3, 5e-3, 1e-2]
# Calibrated by the 2026-08-24 smoke: the first REL grid (2.5e-3..2e-2) landed
# at dL 2e-4..2e-3, i.e. inside the loss instrument's noise floor (~2e-4 nats
# of fp32 forward-pass nondeterminism). A fixed elementwise gamma is a MUCH
# larger relative perturbation than it looks (gamma=1e-2 is ~33% of a matrix's
# Frobenius norm at this width), which is exactly why the two conventions
# cannot be compared to each other. Grid moved up one decade.
RHOS = [0.05, 0.10, 0.20, 0.40]
N_SEEDS = 5
PRIMARY_RHO = 0.10


def target_params(model) -> List[str]:
    """Every ndim>=2 parameter that is not an embedding/unembedding.

    Embeddings are excluded because the embedding anchor is this program's
    stable reference object (measured 1.000 at every checkpoint of every
    family) and because vocab size differs across families, which would make
    the REL convention incomparable across the ladder.
    """
    skip = ("embed", "wte", "lm_head", "unembed")
    out = []
    for name, p in model.named_parameters():
        low = name.lower()
        if p.ndim >= 2 and not any(s in low for s in skip):
            out.append(name)
    return out


def loss_on(model, ids: torch.Tensor, dev: str, bs: int) -> float:
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], bs):
            b = ids[i:i + bs].to(dev)
            tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
            n += b.shape[0]
    return tot / max(n, 1)


def perturb(params: Dict[str, torch.Tensor], orig: Dict[str, torch.Tensor],
            conv: str, mag: float, seed: int, dev: str) -> None:
    """Write orig + noise into params, in place. Seeds are index-derived, not
    hashed, so a re-run reproduces the same draws exactly."""
    g = torch.Generator(device=dev).manual_seed(seed)
    for name, p in params.items():
        w = orig[name]
        noise = torch.randn(w.shape, generator=g, dtype=torch.float32,
                            device=dev)
        if conv == "abs":
            noise.mul_(mag)
        else:
            noise.mul_(mag * w.norm() / (noise.norm() + 1e-12))
        p.data.copy_(w + noise)


def restore(params: Dict[str, torch.Tensor],
            orig: Dict[str, torch.Tensor]) -> None:
    for name, p in params.items():
        p.data.copy_(orig[name])


def loglog_slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of log(y) on log(x); nan if any y<=0."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--device", default="cuda")
    # Reduced-arm options, added 2026-08-24 so the stage2 anneal block (P-L4)
    # could run on CPU while the GPU was unavailable. S_wt is DEFINED as dL at
    # rho=PRIMARY_RHO, so a REL-only run at that rho yields the identical
    # primary scalar; what is lost is beta_rel and c_rel for those points.
    ap.add_argument("--rhos", type=float, nargs="*", default=RHOS)
    ap.add_argument("--gammas", type=float, nargs="*", default=GAMMAS)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if PRIMARY_RHO not in args.rhos:
        raise SystemExit(f"--rhos must contain the primary rho {PRIMARY_RHO}; "
                         f"S_wt is defined at that magnitude")

    os.makedirs(args.output, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from mrilab.datasets.fdgc_datasets import load_generic
    ids = load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
    print(f"eval batch: {tuple(ids.shape)}", flush=True)

    rows: List[Dict] = []
    for rev in args.revisions:
        row_p = os.path.join(args.output, f"row_{rev}.json")
        if os.path.isfile(row_p):
            rows.append(json.load(open(row_p)))
            print(f"SKIP {rev}", flush=True)
            continue

        model = AutoModelForCausalLM.from_pretrained(
            args.repo, revision=rev, dtype=torch.float32).to(args.device).eval()
        names = target_params(model)
        pmap = {n: p for n, p in model.named_parameters() if n in names}
        orig = {n: p.detach().clone() for n, p in pmap.items()}
        fro = [float(w.norm()) for w in orig.values()]
        fro_sorted = sorted(fro)
        med = fro_sorted[len(fro_sorted) // 2]

        base = loss_on(model, ids, args.device, args.batch_size)
        row: Dict = {
            "repo": args.repo, "revision": rev, "loss": base,
            "n_matrices": len(names),
            "n_params": int(sum(w.numel() for w in orig.values())),
            "total_fro": float(sum(f * f for f in fro) ** 0.5),
            "fro_max_over_median": max(fro) / med if med > 0 else float("nan"),
            "abs": {}, "rel": {},
        }
        print(f"{rev}: base loss {base:.4f}  ({len(names)} matrices, "
              f"{row['n_params']/1e6:.1f}M params perturbed)", flush=True)

        for ci, (conv, mags) in enumerate((("abs", args.gammas),
                                           ("rel", args.rhos))):
            for mi, mag in enumerate(mags):
                vals = []
                for s in range(args.seeds):
                    perturb(pmap, orig, conv, mag,
                            seed=100000 * (ci + 1) + 1000 * mi + s,
                            dev=args.device)
                    vals.append(loss_on(model, ids, args.device,
                                        args.batch_size) - base)
                restore(pmap, orig)
                m = sum(vals) / len(vals)
                sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                row[conv][f"{mag:g}"] = {"mean": m, "sd": sd, "vals": vals}
                print(f"    {conv} {mag:g}: dL {m:+.4f} +- {sd:.4f}", flush=True)

        row["S_wt"] = row["rel"][f"{PRIMARY_RHO:g}"]["mean"]
        row["beta_rel"] = loglog_slope(
            args.rhos, [row["rel"][f"{r:g}"]["mean"] for r in args.rhos])
        row["beta_abs"] = loglog_slope(
            args.gammas, [row["abs"][f"{g:g}"]["mean"] for g in args.gammas])
        # Scale-free curvature: for a random perturbation dL ~ (1/2) d^T H d,
        # so dL/rho^2 is constant inside the quadratic regime and is the
        # cleaner sensitivity scalar. Reported alongside S_wt; if beta departs
        # far from 2 the response is not quadratic and c_rel is not meaningful.
        row["c_rel"] = sum(row["rel"][f"{r:g}"]["mean"] / (r * r)
                           for r in args.rhos) / len(args.rhos)
        print(f"  -> S_wt {row['S_wt']:+.4f}  c_rel {row['c_rel']:.3f}  "
              f"beta_rel {row['beta_rel']:.2f}", flush=True)

        with open(row_p, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)
        assert os.path.isfile(row_p), f"row not written: {row_p}"
        rows.append(row)
        del model, pmap, orig
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_p = os.path.join(args.output, "sensitivity_ladder.json")
    with open(out_p, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "rows": rows}, fh, indent=2)
    assert os.path.isfile(out_p), f"summary not written: {out_p}"
    print(f"\nwrote {out_p}")
    print("%-32s %8s %10s %9s %6s" % ("revision", "loss", "S_wt", "c_rel",
                                      "beta"))
    for r in rows:
        print("%-32s %8.4f %+10.4f %9.3f %6.2f"
              % (r["revision"], r["loss"], r["S_wt"], r["c_rel"],
                 r["beta_rel"]))


if __name__ == "__main__":
    main()
