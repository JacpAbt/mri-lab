# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/kd_arms.py
# source sha256: 4798062f2ecdb5f2503fc062cde8793d42455580f7fefe2a999d46a7ad99b56b
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL KD (PANEL_PREREG PART VI): is DISTILLATION readiness-gated?

One student fork, one teacher, two arms that differ ONLY in the loss:

    ce   plain cross-entropy on the pool          (the control)
    kd   pure forward-KL to the teacher's logits  (the treatment)

Matched data, order, seed, steps, LR and budget, so the KD ADVANTAGE
(held-out loss under CE minus under KD) is a within-fork quantity. Running
it at several forks of KNOWN readiness turns "when should you distill" into
a measurement instead of folklore.

Second readout, the trace-memorization signature: teacher-student agreement
on the DISTILLATION pool (seen) versus on held-out text (unseen). A student
that mimics without generalizing shows a large seen-minus-unseen gap - the
distillation analogue of this program's rule-vs-item split.

    python -m mrilab.instruments.kd_arms --student-rev step12000 --loss kd
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrilab.instruments.sensitivity_ladder import LOSS_DOCS, SEQ_LEN

TEACHER = "EleutherAI/pythia-1.4b"
TEACHER_REV = "step71000"
STUDENT = "EleutherAI/pythia-410m"
POOL_CHUNKS = 4096          # v1 default; v2 passes --pool-chunks 256
UNSEEN_AGREE_CHUNKS = 32    # provenance-matched, never trained
AGREE_CHUNKS = 32          # from the pool = "seen" agreement
TEMP = 1.0


def pool_for(tok, n: int) -> torch.Tensor:
    cache = f"experiments/regimes/_pool_kd_c4_{n}_{SEQ_LEN}.pt"
    if os.path.isfile(cache):
        return torch.load(cache)
    from mrilab.datasets.fdgc_datasets import load_generic
    ids = load_generic("c4", tok, SEQ_LEN, n + LOSS_DOCS).input_ids
    torch.save(ids, cache)
    return ids


@torch.no_grad()
def eval_ce(model, ids, dev, bs) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], bs):
        b = ids[i:i + bs].to(dev)
        tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def eval_agreement(student, teacher, ids, dev, bs) -> Dict[str, float]:
    """Forward KL(teacher||student) per token and top-1 agreement rate."""
    student.eval()
    kl_tot, top1_tot, n_tok = 0.0, 0.0, 0
    for i in range(0, ids.shape[0], bs):
        b = ids[i:i + bs].to(dev)
        ls = student(input_ids=b).logits.float()
        lt = teacher(input_ids=b).logits.float()
        pt = F.softmax(lt, dim=-1)
        kl = (pt * (F.log_softmax(lt, dim=-1)
                    - F.log_softmax(ls, dim=-1))).sum(-1)
        kl_tot += float(kl.sum())
        top1_tot += float((ls.argmax(-1) == lt.argmax(-1)).sum())
        n_tok += kl.numel()
    student.train()
    return {"kl_teacher_student": kl_tot / max(n_tok, 1),
            "top1_agreement": top1_tot / max(n_tok, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-rev", required=True)
    # KD v2: gap-matched teachers. Must share the student tokenizer, so
    # pythia-family only (OLMo checkpoints at ideal losses are excluded by
    # vocabulary mismatch).
    ap.add_argument("--teacher-repo", default=TEACHER)
    ap.add_argument("--teacher-rev", default=TEACHER_REV)
    ap.add_argument("--pool-chunks", type=int, default=POOL_CHUNKS,
                    help="v2 uses 256 so the student can actually "
                         "memorise the distillation set")
    ap.add_argument("--loss", choices=["ce", "kd"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    done = os.path.join(args.output, "kd.json")
    if os.path.isfile(done):
        print("SKIP (exists): " + done)
        return

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_need = args.pool_chunks + UNSEEN_AGREE_CHUNKS
    allids = pool_for(tok, n_need)
    heldout = allids[:LOSS_DOCS]          # the frozen instrument's chunks
    pool = allids[LOSS_DOCS:LOSS_DOCS + args.pool_chunks]
    # PROVENANCE-MATCHED agreement sets: both drawn from the same C4 stream
    # by the same packing procedure, one inside the training pool and one
    # never trained. v1 compared pool chunks against the frozen instrument's
    # chunks, so its seen-minus-unseen gap measured set composition (the CE
    # arms showed the same gaps) rather than behaviour.
    seen_agree = pool[:AGREE_CHUNKS]
    unseen_agree = allids[LOSS_DOCS + args.pool_chunks:]
    epochs = (args.steps * args.accum * args.batch_size
              / max(pool.shape[0], 1))
    print(f"pool {tuple(pool.shape)} ({epochs:.1f} epochs)  "
          f"heldout {tuple(heldout.shape)}  "
          f"unseen_agree {tuple(unseen_agree.shape)}", flush=True)

    if args.teacher_repo.split("/")[-1].split("-")[0] != "pythia":
        raise SystemExit("teacher must share the student tokenizer "
                         "(pythia family); got " + args.teacher_repo)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_repo, revision=args.teacher_rev,
        dtype=torch.bfloat16).to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=args.student_rev,
        dtype=torch.float32).to(args.device)
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=0.1)

    t_ce = eval_ce(teacher, heldout, args.device, args.batch_size)
    print("teacher held-out CE: %.4f" % t_ce, flush=True)

    hist: List[Dict] = []

    def record(step: int) -> None:
        row = {"step": step,
               "heldout_ce": eval_ce(student, heldout, args.device,
                                     args.batch_size)}
        row.update({f"seen_{k}": v for k, v in eval_agreement(
            student, teacher, seen_agree, args.device, 2).items()})
        row.update({f"unseen_{k}": v for k, v in eval_agreement(
            student, teacher, unseen_agree, args.device, 2).items()})
        hist.append(row)
        print(f"  step {step:4d}: heldout_ce {row['heldout_ce']:.4f}  "
              f"seen_kl {row['seen_kl_teacher_student']:.4f}  "
              f"unseen_kl {row['unseen_kl_teacher_student']:.4f}  "
              f"seen_top1 {row['seen_top1_agreement']:.4f}  "
              f"unseen_top1 {row['unseen_top1_agreement']:.4f}", flush=True)

    record(0)
    g = torch.Generator().manual_seed(args.seed)
    n = pool.shape[0]
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(args.accum):
            idx = torch.randint(0, n, (args.batch_size,), generator=g)
            b = pool[idx].to(args.device)
            if args.loss == "ce":
                loss = student(input_ids=b, labels=b).loss / args.accum
            else:
                with torch.no_grad():
                    lt = teacher(input_ids=b).logits.float()
                ls = student(input_ids=b).logits.float()
                # forward KL(teacher || student) on next-token positions
                pt = F.softmax(lt[:, :-1] / TEMP, dim=-1)
                logq = F.log_softmax(ls[:, :-1] / TEMP, dim=-1)
                logp = F.log_softmax(lt[:, :-1] / TEMP, dim=-1)
                loss = ((pt * (logp - logq)).sum(-1).mean()
                        * (TEMP ** 2) / args.accum)
                del lt, ls, pt, logq, logp
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.eval_every == 0:
            record(step)

    base, final = hist[0], hist[-1]
    out = {"config": vars(args),
           "teacher": f"{args.teacher_repo}@{args.teacher_rev}",
           "pool_chunks": int(pool.shape[0]), "epochs": epochs,
           "teacher_heldout_ce": t_ce,
           "history": hist,
           "heldout_ce_base": base["heldout_ce"],
           "heldout_ce_final": final["heldout_ce"],
           "heldout_ce_delta": final["heldout_ce"] - base["heldout_ce"],
           "seen_kl_final": final["seen_kl_teacher_student"],
           "unseen_kl_final": final["unseen_kl_teacher_student"],
           "seen_top1_final": final["seen_top1_agreement"],
           "unseen_top1_final": final["unseen_top1_agreement"],
           "agreement_gap_top1": (final["seen_top1_agreement"]
                                  - final["unseen_top1_agreement"]),
           "agreement_gap_kl": (final["unseen_kl_teacher_student"]
                                - final["seen_kl_teacher_student"])}
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{args.student_rev} {args.loss}: heldout_ce "
          f"{out['heldout_ce_base']:.4f} -> {out['heldout_ce_final']:.4f} "
          f"(delta {out['heldout_ce_delta']:+.4f})  "
          f"top1 seen {out['seen_top1_final']:.4f} / unseen "
          f"{out['unseen_top1_final']:.4f}")
    print("wrote " + done)


if __name__ == "__main__":
    main()
