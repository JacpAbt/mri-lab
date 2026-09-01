"""Golden-lane fixtures.

The golden lane re-runs vendored instruments through the real executor
(research venv subprocess, whitelist env, runtime cwd) and compares against
stored rows from the research program. It is the vendored code's license to
exist and the executor's integration test at once.

Run explicitly on the WSL twin:  pytest -m golden
(gpu-marked members need the 3090 free — coordinate with any other session.)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mrilab.engine import fingerprint
from mrilab.settings import Settings

EXPECTED = Path(__file__).parent / "expected"
RESEARCH_REGIMES = Path("/home/jacopo/mri_compressor/experiments/regimes")


def load_expected(name: str) -> dict:
    return json.loads((EXPECTED / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def settings() -> Settings:
    s = Settings()
    if not s.gpu_python.is_file():
        pytest.skip(f"research interpreter not found: {s.gpu_python} "
                    "(golden lane runs on the WSL twin)")
    s.ensure_workspace()
    return s


@pytest.fixture(scope="session")
def research_env(settings: Settings) -> dict:
    """Refuse the whole lane on environment drift: a golden mismatch on a
    drifted venv is uninterpretable (env vs vendoring)."""
    bad = fingerprint.check(settings)
    if bad:
        pytest.skip("research venv does not match runtime_lock.json: " + "; ".join(bad))
    return {"ok": True}


def _seed(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


@pytest.fixture(scope="session")
def seeded_pools(settings: Settings) -> dict[str, bool]:
    """Copy the research tree's tokenized pools + the local rejuv artifact into
    the workspace runtime (bit-identical training data is load-bearing for the
    crash golden; z_ter's pythia set includes the rejuv checkpoint)."""
    out: dict[str, bool] = {}
    pools = settings.pools_dir
    pools.mkdir(parents=True, exist_ok=True)
    out["code_pool"] = _seed(RESEARCH_REGIMES / "_pool_code_8224_512.pt",
                             pools / "_pool_code_8224_512.pt")
    out["rejuv"] = _seed(RESEARCH_REGIMES / "rejuv_artifact_3e-5",
                         pools / "rejuv_artifact_3e-5")
    return out


# --- shared assertion helpers -------------------------------------------------

def assert_close(name: str, got: float, want: float, tol: float) -> None:
    assert abs(got - want) <= tol, f"{name}: got {got!r}, want {want!r} (tol {tol})"


def assert_ci_overlap(name: str, got: float, ci: list[float], pad: float = 0.0) -> None:
    lo, hi = ci
    assert lo - pad <= got <= hi + pad, f"{name}: {got!r} outside stored CI [{lo}, {hi}]"
