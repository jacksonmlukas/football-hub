"""The scan behind `tests/contracts/test_store_reads_are_named.py`: who calls `store.sql`.

Top-level for the reason `tests/nflverse_routing.py` is: pyrefly cannot resolve an import
between two test modules in `tests/contracts`, and the scan wants a planted tree as well as the
real one.

A call is `store.sql(...)` -- or `sql(...)` where the module imported that name from `hub.store`
-- and the query it carries is resolved as far as it can be: a string literal, the literal parts
of an f-string, or a name bound to either in the same module. What cannot be resolved is
reported as `UNRESOLVED`, because a scan that skipped what it could not read would let the
reader it could not read through.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

STORE = "store.py"
UNRESOLVED = "<unresolved>"


def _literal(node: ast.AST, bound: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append(_literal(v.value, bound) or "")
        return "".join(parts)
    if isinstance(node, ast.Name):
        return bound.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left, bound), _literal(node.right, bound)
        return None if left is None or right is None else left + right
    return None


def _sql_names(tree: ast.AST) -> set[str]:
    """The local names `sql` goes by: imported from `hub.store`, or defined here."""
    names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "hub.store":
            names |= {a.asname or a.name for a in n.names if a.name == "sql"}
        elif isinstance(n, ast.FunctionDef) and n.name == "sql":
            names.add("sql")                # the store itself, where it is a local function
    return names


def sql_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Every `store.sql` call in a tree, as `(line, the query text it can be resolved to)`."""
    bound: dict[str, str] = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)):
            got = _literal(n.value, bound)
            if got is not None:
                bound[n.targets[0].id] = got
    local = _sql_names(tree)
    out: list[tuple[int, str]] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        f = n.func
        via_module = (isinstance(f, ast.Attribute) and f.attr == "sql"
                      and isinstance(f.value, ast.Name) and f.value.id in ("store", "hub_store"))
        via_name = isinstance(f, ast.Name) and f.id in local
        if via_module or via_name:
            out.append((n.lineno, _literal(n.args[0], bound) or UNRESOLVED))
    return out


def touches(query: str, table: str | None) -> bool:
    """Whether a query reads `table` as a whole word (`prop_lines` is not `lines`); with no
    table named, every query counts -- and so does one that could not be resolved."""
    if table is None or query == UNRESOLVED:
        return True
    return re.search(rf"\b{re.escape(table)}\b", query) is not None


def sql_callers(root: Path, table: str | None = None) -> dict[str, list[int]]:
    """Every module under `root` other than the store that calls `sql`, on `table` if named."""
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == STORE:
            continue
        lines = [ln for ln, q in sql_calls(ast.parse(path.read_text())) if touches(q, table)]
        if lines:
            found[rel] = lines
    return found
