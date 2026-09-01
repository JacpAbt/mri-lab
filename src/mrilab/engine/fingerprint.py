"""Environment-fingerprint contract: the research venv vs runtime_lock.json.

The stored golden rows were produced by one specific environment; ENVIRONMENT.md
documents that "no single environment description covers every result" as a trap.
This module turns that prose into a machine check: capture the live fingerprint
by running a snippet UNDER the research interpreter, compare against the
committed lock, and let callers refuse (golden lane) or stamp (ordinary rows)
on mismatch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mrilab.settings import Settings

# Keep in sync with scripts/capture_runtime_lock.py (that script writes the lock;
# this snippet is the runtime probe — same fields).
_SNIPPET = r"""
import json, platform, sys
def v(m):
    try:
        mod = __import__(m)
    except Exception:
        return None
    return getattr(mod, "__version__", "present")
fp = {"python": platform.python_version(),
      "packages": {m: v(m) for m in ("torch","transformers","datasets","numpy",
        "accelerate","flash_attn","bitsandbytes","mamba_ssm","huggingface_hub",
        "tokenizers","safetensors")}}
try:
    import torch
    fp["torch_cuda"] = torch.version.cuda
    fp["cuda_available"] = torch.cuda.is_available()
    fp["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    fp.update(torch_cuda=None, cuda_available=False, gpu_name=None, torch_error=repr(e))
print(json.dumps(fp))
"""


def capture(settings: Settings, timeout_s: float = 120) -> dict:
    out = subprocess.run(
        [str(settings.gpu_python), "-c", _SNIPPET],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if out.returncode != 0:
        raise RuntimeError(f"fingerprint capture failed: {out.stderr[-500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def lock_path(settings: Settings) -> Path:
    return settings.code_root / "runtime_lock.json"


def mismatches(live: dict, lock: dict, require_gpu: bool = False) -> list[str]:
    """Field-by-field comparison; returns human-readable mismatch strings."""
    out: list[str] = []
    if live.get("python") != lock.get("python"):
        out.append(f"python {live.get('python')} != lock {lock.get('python')}")
    for name, want in (lock.get("packages") or {}).items():
        got = (live.get("packages") or {}).get(name)
        if got != want:
            out.append(f"{name} {got} != lock {want}")
    if lock.get("gpu_name") and live.get("gpu_name") != lock.get("gpu_name"):
        out.append(f"gpu {live.get('gpu_name')} != lock {lock.get('gpu_name')}")
    if require_gpu and not live.get("cuda_available"):
        out.append("cuda not available")
    return out


def check(settings: Settings, require_gpu: bool = False) -> list[str]:
    lock = json.loads(lock_path(settings).read_text(encoding="utf-8"))
    return mismatches(capture(settings), lock, require_gpu=require_gpu)
