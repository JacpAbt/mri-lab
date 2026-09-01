"""Repo-root conftest: make `mrilab` importable when the package is NOT
installed (e.g. running pytest on a Windows checkout without the venv),
so tests keep importing `mrilab.*`.

Deliberately NOT solved by importing `src.mrilab.*`: that loads every module
a second time under a second name (src is an implicit namespace package),
which splits lru_caches, Settings identities, and Huey registrations between
the two copies — the classic dual-import trap.
"""

import sys
from pathlib import Path

try:
    import mrilab  # noqa: F401  — editable install present (the WSL twin)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
