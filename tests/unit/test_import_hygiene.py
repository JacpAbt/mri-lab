"""AST import-lint for the torch-side packages.

The GPU subprocess imports mrilab.instruments.* inside the RESEARCH venv, which has
torch but none of the service deps. Any fastapi/pydantic/matplotlib import reaching
that path would crash every measurement — and importing mrilab.instruments.X also
executes mrilab/__init__.py, so the root must stay import-free too.

AST, not import hooks: no side effects, runs in the CPU lane.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mrilab"
TORCH_SIDE_DIRS = ("instruments", "datasets", "inspector")

# import roots the research venv is known to provide (extend only when vendoring demands)
ALLOWED_ROOTS = {
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "einops",
    "bitsandbytes",
    # intra-package (torch-side only)
    "mrilab",
}
ALLOWED_MRILAB_SUBPACKAGES = {"instruments", "datasets", "inspector"}
STDLIB = set(sys.stdlib_module_names)


def _torch_side_files() -> list[Path]:
    files: list[Path] = []
    for d in TORCH_SIDE_DIRS:
        files.extend((SRC / d).rglob("*.py"))
    return files


def _imports_of(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):  # module-level AND function-local imports
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                found.append((".", node.lineno))  # bare relative import
            elif node.module is not None:
                prefix = "." * node.level
                found.append((prefix + node.module, node.lineno))
    return found


def test_torch_side_imports_are_research_venv_safe() -> None:
    violations: list[str] = []
    for f in _torch_side_files():
        for name, lineno in _imports_of(f):
            if name.startswith("."):
                # relative imports stay inside the torch-side package: fine
                continue
            root = name.split(".")[0]
            if root in STDLIB:
                continue
            if root not in ALLOWED_ROOTS:
                violations.append(f"{f.relative_to(SRC)}:{lineno} imports {name}")
                continue
            if root == "mrilab":
                parts = name.split(".")
                if len(parts) >= 2 and parts[1] not in ALLOWED_MRILAB_SUBPACKAGES:
                    violations.append(
                        f"{f.relative_to(SRC)}:{lineno} imports service-side {name}"
                    )
    assert not violations, "torch-side import violations:\n" + "\n".join(violations)


def test_init_files_on_torch_path_are_import_free() -> None:
    inits = [SRC / "__init__.py"] + [SRC / d / "__init__.py" for d in TORCH_SIDE_DIRS]
    for init in inits:
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
        bad = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        assert not bad, f"{init} must stay import-free (imports at lines {bad})"
