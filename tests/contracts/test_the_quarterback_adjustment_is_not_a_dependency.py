"""`hub.models.quarterback` is read by the harness that gates it and by nothing the product ships.

#299 pulled the quarterback adjustment from the published path. Two facts decided it: the
nfeloqb file names a new starter only *after* his first game (#291), and after #297 the
adjustment fires only on a stale poll -- so when it fires it adds the current `qb_adj` to a
line that already priced that same quarterback, and the one case it was built for, a change
the frozen line predates, cannot reach it from this source. The module stays, for the
reason `hub.exhibits` stays under ADR-0007: #291's gate is committed, tested and
re-runnable, and `hub.models.starter_change` is the harness that re-runs it. What this file
holds is that the harness is the *only* reader -- a product path that reaches the module,
by any import, has put an ungated adjustment back into a published number without the gate
clearing first.

Modelled on `test_the_exhibit_is_not_a_dependency.py` (#257): read off the AST, never off
the text, transitively, with function-local imports and literal `import_module` strings
counted, so the module cannot come back one hop away or inside a function body. The walker
is that file's, copied rather than shared so each contract stands on its own.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"

# The one production reader the adjustment may have: #291's harness, the gate and its oracle
# arm. Everything else under `src/` that imports the module is wiring it back into the
# product, which #299 undid and which the gate clearing is the only way to redo.
HARNESS = "models/starter_change.py"
MODULE = "hub.models.quarterback"

# What ships and what prices it: the weekly writer and the survivor grid, which #218 wired
# the adjustment into and #299 unwired; the schedule they both price from, which until #299
# declared the live-price cut in the quarterback module; the publisher; and the pool CLI,
# which plans through the survivor solver.
PRODUCT = ("hub.models.ratings", "hub.season.survivor", "hub.schedule", "hub.publish",
           "hub.season.pool")


def _path_of(module: str) -> pathlib.Path | None:
    """The file a `hub.*` module name resolves to, or None for a name that is not a module."""
    stem = SRC.parent / pathlib.Path(*module.split("."))
    if (stem / "__init__.py").exists():
        return stem / "__init__.py"
    if stem.with_suffix(".py").exists():
        return stem.with_suffix(".py")
    return None


def _literal_import(node: ast.Call) -> str | None:
    """The module a literal-string `import_module("hub...")` or `__import__("hub...")`
    names, or None; a name computed at runtime is the one gap this walker has."""
    f = node.func
    called = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    if called not in ("import_module", "__import__") or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str) \
            and first.value.startswith("hub"):
        return first.value
    return None


def _runtime_imports_of(path: pathlib.Path) -> set[str]:
    """Every `hub.*` module one file imports at runtime -- at module level and inside
    functions -- including a literal string handed to `import_module`. An
    `if TYPE_CHECKING:` block is skipped: nothing under it loads."""
    tree = ast.parse(path.read_text())
    typing_only: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            typing_only |= {id(n) for n in ast.walk(node)}
    out: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in typing_only:
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hub"):
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names
                    if _path_of(f"{node.module}.{a.name}")}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("hub")}
        elif isinstance(node, ast.Call) and (named := _literal_import(node)):
            out.add(named)
    return out


def _reach(module: str) -> dict[str, str | None]:
    """Every `hub.*` module `module` can load by importing, with the module that first
    reached each -- so a failure names the path and not only the endpoint."""
    parent: dict[str, str | None] = {module: None}
    todo = [module]
    while todo:
        here = todo.pop()
        path = _path_of(here)
        if path is None:
            continue
        for dep in sorted(_runtime_imports_of(path)):
            if dep not in parent:
                parent[dep] = here
                todo.append(dep)
    return parent


def test_the_only_production_reader_of_the_adjustment_is_the_harness():
    """A direct import anywhere under `src/` but the harness is the adjustment wired back
    into something the product runs, without #291's gate clearing first."""
    readers = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        if rel == HARNESS or rel == "models/quarterback.py":
            continue
        if MODULE in _runtime_imports_of(path):
            readers[rel] = MODULE
    assert not readers, (
        f"a module other than the harness imports `{MODULE}`: {sorted(readers)}. #299 pulled "
        f"the adjustment from the published path; the way back is a starter source that is "
        f"timely before kickoff and #291's gate clearing, not an import.")


def test_the_harness_does_read_the_adjustment():
    """The rule above is only a rule if the harness is the exception to it. A harness that
    stopped importing the module would leave #291's gate with nothing to re-run, which is
    the ADR-0007 failure the module is kept to prevent."""
    assert MODULE in _runtime_imports_of(SRC / HARNESS), (
        f"{HARNESS} no longer reaches `{MODULE}`, so nothing re-runs #291's gate")


@pytest.mark.parametrize("target", PRODUCT)
def test_nothing_that_ships_can_load_the_adjustment_by_any_import(target):
    """Transitively, off the AST, with function-local imports counted: not only a direct
    import, but nothing the target imports may import the module either, or the pull is
    undone one hop away."""
    reach = _reach(target)
    chain = []
    if MODULE in reach:
        step: str | None = MODULE
        while step is not None:
            chain.append(step)
            step = reach[step]
    assert MODULE not in reach, (
        f"{target} can load `{MODULE}` through {' <- '.join(chain)}: the adjustment is on a "
        f"path the product runs on. #299 pulled it; #291's gate clearing is the way back.")


@pytest.mark.parametrize("target", PRODUCT)
def test_a_fresh_interpreter_importing_the_product_does_not_load_the_adjustment(target):
    """The same question asked of a fresh interpreter, for a module-level import the AST
    walk might mis-resolve. A subprocess, for the reason #257 gives: evicting the module
    from this process's `sys.modules` leaves later tests holding two copies of it."""
    import subprocess
    import sys

    code = (f"import sys; import {target}; "
            f"print(sorted(m for m in sys.modules if m == {MODULE!r}))")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, check=True)
    assert run.stdout.strip() == "[]", (
        f"importing {target} loaded `{MODULE}`: the adjustment is on a product path")
