"""Subprocess executor for the frozen instruments (the engine's core).

Runs `python -m mrilab.instruments.<module>` under the RESEARCH venv with a
WHITELIST environment (never os.environ — this is what makes the
attention-backend trap structurally impossible), cwd = the workspace runtime
dir (which carries the experiments/regimes/ pool subtree the vendored code's
relative paths expect), one log file per run.

The queue wrapper (P5) builds on this; the golden tests (tests/golden/) use it
directly, so the executor itself is exercised by the golden lane.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mrilab.settings import Settings

# env the vendored instruments READ (names preserved verbatim for bit-compat)
_INSTRUMENT_ENV = {
    "FDGC_ATTN_IMPL": "sdpa",  # flash_attn is installed upstream; never let it default
    "FDGC_EVAL_DOC_SKIP": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_XET": "1",
}


@dataclass
class RunResult:
    rc: int
    cmd: list[str]
    log_path: Path
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.rc == 0


def build_env(settings: Settings, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Whitelist env for instrument subprocesses. Inherit nothing implicitly."""
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONPATH": str(settings.code_root / "src"),
        "PYTHONUNBUFFERED": "1",
        **_INSTRUMENT_ENV,
    }
    # hf_cache unset -> inherit the user default (~/.cache/huggingface), which is
    # the research repo's cache: golden runs reuse already-downloaded models.
    if settings.hf_cache is not None:
        env["HF_HUB_CACHE"] = str(settings.hf_cache)
    if overrides:
        env.update(overrides)
    return env


def run_instrument(
    settings: Settings,
    module: str,
    args: list[str],
    log_name: str,
    timeout_s: float | None = None,
    env_overrides: dict[str, str] | None = None,
) -> RunResult:
    """Run one instrument module to completion; stdout+stderr -> one log file."""
    settings.ensure_workspace()
    cmd = [str(settings.gpu_python), "-m", f"mrilab.instruments.{module}", *args]
    log_path = settings.logs_dir / f"{log_name}.log"
    t0 = time.monotonic()
    with open(log_path, "ab") as fh:
        fh.write(
            f"=== {module} start {_dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"=== cmd: {' '.join(cmd)}\n".encode()
        )
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=settings.runtime_cwd,
            env=build_env(settings, env_overrides),
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        dur = time.monotonic() - t0
        fh.write(f"=== {module} exit {proc.returncode} after {dur:.1f}s\n".encode())
    return RunResult(rc=proc.returncode, cmd=cmd, log_path=log_path, duration_s=dur)


# --- GPU thermal gate (measured practice: <78C, 3 tries, 90s waits, 300W cap) ---

def gpu_temperature_c() -> int | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_cool(settings: Settings, sleeper=time.sleep) -> tuple[bool, int | None]:
    """True if under the limit (or unmeasurable); after the retries, proceed
    anyway — the caller stamps the warning on the job (never silently)."""
    t = gpu_temperature_c()
    for _ in range(settings.thermal_retries):
        if t is None or t < settings.thermal_limit_c:
            return True, t
        sleeper(settings.thermal_wait_s)
        t = gpu_temperature_c()
    return False, t
