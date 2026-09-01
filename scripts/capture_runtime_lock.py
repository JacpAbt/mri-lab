"""Capture the research venv's environment fingerprint into runtime_lock.json.

Run under the RESEARCH interpreter (the environment of record for every golden row):

    /home/jacopo/experiments/.venv/bin/python scripts/capture_runtime_lock.py > runtime_lock.json

The same snippet logic is reused at service startup and before golden-comparable
GPU jobs (engine/fingerprint.py) to refuse or flag runs on a drifted environment.
Stdlib + the packages being fingerprinted only — no mrilab imports (this runs in
the research venv, which never gets mrilab installed).
"""

from __future__ import annotations

import json
import platform
import sys


def _ver(mod_name: str) -> str | None:
    try:
        mod = __import__(mod_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "present")


def main() -> None:
    fp: dict = {
        "python": platform.python_version(),
        "packages": {
            name: _ver(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "numpy",
                "accelerate",
                "flash_attn",
                "bitsandbytes",
                "mamba_ssm",
                "huggingface_hub",
                "tokenizers",
                "safetensors",
            )
        },
    }
    try:
        import torch

        fp["torch_cuda"] = torch.version.cuda
        fp["cuda_available"] = torch.cuda.is_available()
        fp["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # capture must not die on a CPU-only box
        fp["torch_cuda"] = None
        fp["cuda_available"] = False
        fp["gpu_name"] = None
        fp["torch_error"] = repr(exc)

    json.dump(fp, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
