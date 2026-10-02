"""The scan behind `tests/contracts/test_sources_are_routed.py`: who imports `nflreadpy`.

Top-level here because pyrefly cannot resolve an import between two test modules in
`tests/contracts`, and the scan wants to be called on a planted tree as well as the real one.
"""
from __future__ import annotations

import ast
from pathlib import Path

LIBRARY = "nflreadpy"

# The one module allowed to import it: `hub.fetch.nflverse`, whose `Network` adapter is the
# first adapter at the nflverse seam. Relative to the tree scanned.
LOADER = "fetch/nflverse.py"


def _imports(tree: ast.AST) -> list[int]:
    """The lines at which a tree imports the library, in any of the forms an import takes."""
    lines: list[int] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] == LIBRARY for a in n.names):
            lines.append(n.lineno)
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == LIBRARY:
            lines.append(n.lineno)
        elif (isinstance(n, ast.Call)
              and (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
              in ("import_module", "__import__")
              and any(isinstance(a, ast.Constant) and a.value == LIBRARY for a in n.args)):
            lines.append(n.lineno)
    return sorted(lines)


def direct_importers(root: Path) -> dict[str, list[int]]:
    """Every `.py` under `root` that imports the library, as `{relative path: lines}`."""
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        lines = _imports(ast.parse(path.read_text()))
        if lines:
            found[path.relative_to(root).as_posix()] = lines
    return found
