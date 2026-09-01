# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/s_cross.py
# source sha256: c6560aba8c9b0ca94169b5673d6a83367ab8307a035b52260429d3a8f34a4cb9
# vendored: 2026-09-01 by scripts/vendor.py (2 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL SX (PANEL_PREREG_2026_08.md PART VI): the cross-domain damage face.

Perturb along the CODE-batch gradient at matched per-matrix relative norm
(rho = 0.10, the REL convention), read the C4 response — first-order and
curvature — plus the code-side response (sanity) and uniform C4-S_grad.

    python -m mrilab.instruments.s_cross --output experiments/regimes/sx_out
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import (
    PRIMARY_RHO, LOSS_DOCS, SEQ_LEN, target_params, loss_on)
from mrilab.instruments.vitals import grad_on

CODE_POOL = "experiments/regimes/_pool_code_8224_512.pt"
ARTIFACTS = [
    ("160m_step40000", "EleutherAI/pythia-160m", "step40000", None),
    ("160m_step143000", "EleutherAI/pythia-160m", "step143000", None),
    ("160m_rejuv", None, None, "experiments/regimes/rejuv_artifact_3e-5"),
    ("410m_step1000", "EleutherAI/pythia-410m", "step1000", None),
    ("410m_step12000", "EleutherAI/pythia-410m", "step12000", None),
]


def directional(model, pmap, orig, gmap, ids_pairs, dev, bs, bases):
    """Apply +/- the per-matrix REL-scaled direction gmap; return the loss
    deltas for every eval set in ids_pairs (dict name->ids)."""
    out: Dict[str, Dict[str, float]] = {n: {} for n in ids_pairs}
    for sign in (+1.0, -1.0):
        for n, p in pmap.items():
            g = gmap[n]
            d = g * (sign * PRIMARY_RHO * orig[n].norm()
                     / (g.norm() + 1e-12))
            p.data.copy_(orig[n] + d)
        for name, ids in ids_pairs.items():
            out[name]["plus" if sign > 0 else "minus"] = (
                loss_on(model, ids, dev, bs) - bases[name])
    for n, p in pmap.items():
        p.data.copy_(orig[n])
    for name in out:
        out[name]["first_order"] = (out[name]["plus"]
                                    - out[name]["minus"]) / 2
        out[name]["curvature"] = (out[name]["plus"]
                                  + out[name]["minus"]) / 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    # Amendment 1 (2026-08-27, after the first rho=0.10 row and before any
    # other): directional probes along a GRADIENT need a far smaller
    # magnitude than random ones — at rho=0.10 the first artifact returned
    # 20+ nat responses and a NEGATIVE code-side first-order along +g
    # (G-SX1's exact failure mode). rho_dir = 0.01 puts the directional
    # response in the quadratic regime. All predictions are rank/sign
    # statements and are unchanged.
    ap.add_argument("--rho-dir", type=float, default=0.01)
    ap.add_argument("--only", default=None,
                    help="run a single artifact label (for magnitude smokes)")
    # SG-SURFACE sweep mode: --repo + --revisions overrides the hardcoded
    # calibration-artifact list.
    ap.add_argument("--repo", default=None)
    ap.add_argument("--revisions", nargs="*", default=None)
    ap.add_argument("--output", default="experiments/regimes/sx_out")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    global PRIMARY_RHO, ARTIFACTS
    PRIMARY_RHO = args.rho_dir
    if args.repo:
        short = args.repo.split("/")[-1]
        ARTIFACTS = [(f"{short}@{rev}", args.repo, rev, None)
                     for rev in (args.revisions or ["main"])]
    if args.only:
        ARTIFACTS = [a for a in ARTIFACTS if a[0] == args.only]
        if not ARTIFACTS:
            raise SystemExit(f"unknown label {args.only}")

    code = torch.load(CODE_POOL)[:32]
    print(f"code batch {tuple(code.shape)} (= every erosion cell's target "
          f"eval)", flush=True)

    for label, repo, rev, path in ARTIFACTS:
        row_p = os.path.join(args.output, f"row_{label}.json")
        if os.path.isfile(row_p):
            print(f"SKIP {label}")
            continue
        src = path or repo
        tok = AutoTokenizer.from_pretrained(src)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = {"revision": rev} if rev else {}
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.float32, **kw).to(args.device).eval()

        from mrilab.datasets.fdgc_datasets import load_generic
        c4 = load_generic("c4", tok, SEQ_LEN, LOSS_DOCS).input_ids
        names = target_params(model)
        pmap = {n: p for n, p in model.named_parameters() if n in names}
        orig = {n: p.detach().clone() for n, p in pmap.items()}
        bases = {"c4": loss_on(model, c4, args.device, args.batch_size),
                 "code": loss_on(model, code, args.device, args.batch_size)}
        ids_pairs = {"c4": c4, "code": code}

        row: Dict = {"label": label, "repo": repo, "revision": rev,
                     "path": path, "rho": PRIMARY_RHO,
                     "base_c4": bases["c4"], "base_code": bases["code"]}
        for gname, gids in (("gcode", code), ("gc4", c4)):
            grad_on(model, gids, args.device, args.batch_size)
            gmap = {n: p.grad.detach().clone() for n, p in pmap.items()
                    if p.grad is not None}
            model.zero_grad(set_to_none=True)
            row[gname] = directional(model, pmap, orig, gmap, ids_pairs,
                                     args.device, args.batch_size, bases)
            del gmap
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        # practical first-order: the response along -g_code (a training step)
        row["practical_first_order_c4"] = -row["gcode"]["c4"]["first_order"]
        row["curvature_x"] = row["gcode"]["c4"]["curvature"]
        row["dmg_hat"] = (row["practical_first_order_c4"]
                          + row["curvature_x"])
        print(f"{label}: base c4 {bases['c4']:.4f} code {bases['code']:.4f}"
              f"  | S_x first(-g) {row['practical_first_order_c4']:+.4f}"
              f"  curv {row['curvature_x']:+.4f}"
              f"  dmg_hat {row['dmg_hat']:+.4f}"
              f"  | code-side first(+g) "
              f"{row['gcode']['code']['first_order']:+.4f}"
              f"  | S_grad(c4) curv {row['gc4']['c4']['curvature']:+.4f}",
              flush=True)
        with open(row_p, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2)
        del model, pmap, orig
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("SX rows done")


if __name__ == "__main__":
    main()
