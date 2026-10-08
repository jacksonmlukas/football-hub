"""A held-out constant is held out everywhere it is read (#320).

`hub.holdout.applied` rebinds the attribute on the *declaring* module for the duration of a
season and puts the shipped value back after. That reaches exactly the readers that look the
attribute up when they run. A reader that took a copy -- `from hub.models.predict import
WEEKLY_K`, a copy of that copy re-exported through another module, a signature default, an
array computed at import -- holds whichever value was bound when it loaded, and the hold-out
cannot move it. Meanwhile the digest does move, because coverage reads the declaring module,
so the backtest would stamp "played under the hold-out constants" on rows a stale reader
played under the shipped ones.

Two shapes are forbidden anywhere under `src/hub`:

1. **A direct import of a held-out name** -- `from <hub module> import NAME`, at module level
   or inside a function. The import-time form is a stale copy; the in-function form is read
   fresh and happens to work, but it is one edit from the other, so the rule is the import and
   not the scope. Read the constant as `module.NAME`.
2. **A read of a held-out name that runs at import** -- the right-hand side of a module-level
   assignment, a class body, a decorator or a default argument (`def f(k=WEEKLY_K_POOLED)` is
   evaluated once, at `def` time, and no rebind can ever move it). A `lambda` body and a
   `def` body run later and are not import-time.

The declaring modules themselves are scanned too, minus the line that defines the name: a
constant derived from another at import (`B = A * 2`) is a copy of `A` the rebind of `A` does
not carry.

The held-out set is `hub.holdout.HELD_OUT`, so a twelfth constant is covered the day it is
added. Scripts under `scripts/` are not scanned: they run outside any `applied` block and
print the shipped value on purpose.

**The positive control is the last group of tests**: each forbidden shape is planted in a
temporary source and the scan must name it, and the permitted shapes must pass.
"""
from __future__ import annotations

import ast
from pathlib import Path

from hub import holdout
from hub.declare import module_level

SRC = Path(__file__).resolve().parents[2] / "src" / "hub"
NAMES = frozenset(k.split(".", 1)[1] for k in holdout.HELD_OUT)


def _loaded(node: ast.AST) -> list[tuple[int, str]]:
    """Held-out names read (Load) anywhere inside `node`, not descending into a lambda."""
    out: list[tuple[int, str]] = []
    todo = [node]
    while todo:
        n = todo.pop()
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in NAMES:
            out.append((n.lineno, n.id))
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and n.attr in NAMES:
            out.append((n.lineno, n.attr))
        todo.extend(ast.iter_child_nodes(n))
    return out


def _import_time(body: list[ast.stmt]) -> list[tuple[int, str]]:
    """Held-out reads in what runs when the module (or a class body) is imported."""
    out: list[tuple[int, str]] = []
    for node in module_level(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            exprs = [*a.defaults, *(d for d in a.kw_defaults if d is not None),
                     *node.decorator_list]
            for e in exprs:
                out += _loaded(e)
        elif isinstance(node, ast.ClassDef):
            for e in [*node.bases, *(k.value for k in node.keywords), *node.decorator_list]:
                out += _loaded(e)
            out += _import_time(node.body)
        elif not isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                                   ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)):
            out += _loaded(node)
        else:
            # A compound statement's own header (test, iterable, context managers) runs at
            # import; its bodies are yielded separately by `module_level`.
            for field in ("test", "iter", "subject"):
                if (e := getattr(node, field, None)) is not None:
                    out += _loaded(e)
            for item in getattr(node, "items", ()):
                out += _loaded(item.context_expr)
    return out


def violations(source: str) -> list[str]:
    """Every way `source` holds a held-out name as a copy, one line each."""
    tree = ast.parse(source)
    found: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and (
                n.level != 0 or n.module == "hub" or n.module.startswith("hub.")):
            found += [f"line {n.lineno}: imports {a.name} directly from {n.module}"
                      for a in n.names if a.name in NAMES]
    found += [f"line {line}: reads {name} at import (a copy, or a default no rebind moves)"
              for line, name in _import_time(tree.body)]
    return sorted(set(found))


def test_no_module_holds_a_copy_of_a_held_out_constant():
    bad = {}
    for path in sorted(SRC.rglob("*.py")):
        got = violations(path.read_text())
        if got:
            bad[path.relative_to(SRC).as_posix()] = got
    assert not bad, (
        "a held-out constant (hub.holdout.HELD_OUT) is read through a copy that "
        "`holdout.applied` cannot rebind; read it as `module.NAME` where it is used:\n"
        + "\n".join(f"  {f}: {v}" for f, vs in bad.items() for v in vs))


# --- the positive control: plant each shape and confirm the scan names it ---------------------

def test_the_scan_names_a_direct_import_at_module_level_and_inside_a_function():
    assert violations("from hub.models.predict import WEEKLY_K\n")
    assert violations("def f():\n    from hub.models.predict import IMPUTE_CV\n    return IMPUTE_CV\n")
    assert violations("from hub.draft.season import WEEKLY_K_POOLED as k\n")


def test_the_scan_names_a_default_argument_and_an_import_time_array():
    assert violations("from hub.models import predict\n"
                      "def f(k=predict.WEEKLY_K_POOLED):\n    return k\n")
    assert violations("import numpy as np\nfrom hub.models import predict\n"
                      "SD = np.array([predict.WEEKLY_K.get(p) for p in 'QB'])\n")
    assert violations("from hub.models import predict\nclass C:\n    k = predict.WEEKLY_K\n")
    assert violations("from hub.models import predict\nif True:\n    k = predict.TALENT_CV\n")


def test_the_scan_names_a_constant_derived_from_another_in_its_own_module():
    assert violations("WEEKLY_K_POOLED = fitted(2.04)\nBIG = WEEKLY_K_POOLED * 2\n")


def test_the_scan_passes_the_shapes_that_do_follow_a_rebind():
    assert not violations("WEEKLY_K_POOLED = fitted(2.04)\n")
    assert not violations("from hub.models import predict\n"
                          "def f():\n    return predict.WEEKLY_K_POOLED\n")
    assert not violations("from hub.models import predict\n"
                          "g = lambda: predict.WEEKLY_K_POOLED\n")
    assert not violations("def f(k=None):\n    return k\n")
    assert not violations("from other import WEEKLY_K\n")   # not a hub module
