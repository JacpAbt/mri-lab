from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from mrilab.faces.recorder import CIGates, evaluate_gates
from mrilab.settings import get_settings

router = APIRouter(tags=["ci"])


class AssertRequest(BaseModel):
    labels: list[str]  # checkpoint subject labels, in training order
    gates: CIGates


@router.post("/ci/assert")
def ci_assert(req: AssertRequest) -> dict:
    return evaluate_gates(get_settings(), req.labels, req.gates)


@router.get("/ci/gates")
def ci_gates() -> dict:
    return {
        "max_dl_abs_ratio_vs_first": "fragility budget: dL_abs(last)/dL_abs(first) ceiling",
        "min_unseen_gain_slope": "readiness stall: minimum unseen-gain slope per checkpoint (n>=3; never single-probe)",
        "max_c4_regression_nats": "capability budget: held-out C4 regression vs best",
    }
