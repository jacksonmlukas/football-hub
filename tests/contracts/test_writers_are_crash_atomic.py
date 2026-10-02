"""Every write in `src/hub` goes through `hub.atomic` (#258, #392, #393).

`CLAUDE.md` promises last-good state when a fetch fails. A writer that truncates its final
path before it has the bytes turns a mid-write kill into a file the next read refuses or
parses as garbage -- the fallback destroyed by the outage it exists to survive. `hub.atomic`
writes to a scratch name and renames over the target in one call; this file holds the whole
tree to it.

**The whole tree, not a named list.** This began (#258) as eleven named modules, on the
argument that a CLI's `--out` scratch table is written for an operator and read by nobody
unattended. The list drifted past itself within a month: `roster`, the draft picks file and
the ADP archive were all last-good writers it did not name (#392), and `backtest`'s
`--board` snapshot looked like scratch and is read back as the pinned input of every later
run. An atomic write costs nothing for a scratch table, so the rule is now the simple one
and a new module is covered the day it is written.

**The one way out is a named entry**, `ALLOWED["module.function"] = reason`, the reason at
least `hub.declare.MIN_REASON_WORDS` words -- the bar `not_an_input` holds an exclusion to,
for the same reason: a marker with no argument is the silence the list exists to prevent. An
entry whose function no longer writes directly is refused as stale, so the list only ever
holds exceptions that are still true. It is empty, and it should stay that way.

The AST, not a text search: the call this forbids is named in this docstring and in every
module's own comments.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeGuard

from hub.declare import MIN_REASON_WORDS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"
HELPER = "atomic"
DIRECT = frozenset({"write_text", "write_bytes", "write_parquet"})

# `module.function -> reason`, module being the dotted path under `hub` (`draft.backtest`).
# Empty by decision (#393): the twelve writes that were not routed were all routed.
ALLOWED: dict[str, str] = {}


def _is_direct(node: ast.AST) -> TypeGuard[ast.Call]:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in DIRECT
            and not (isinstance(node.func.value, ast.Name) and node.func.value.id == HELPER))


def _direct_writes(source: str) -> list[int]:
    """Lines calling `.write_text`, `.write_bytes` or `.write_parquet` on anything but the
    helper module itself (`atomic.write_text(...)` is the routed spelling)."""
    return [n.lineno for n in ast.walk(ast.parse(source)) if _is_direct(n)]


def _by_function(source: str, module: str) -> dict[str, list[int]]:
    """`module.function` (`module.Class.method`; `module.<module>` at top level) -> lines of
    its direct writes. A write in a nested function is named for the outermost function
    around it, which is what an allowlist author can read off the file."""
    found: dict[str, list[int]] = {}

    def walk(node: ast.AST, scope: tuple[str, ...], in_fn: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, (*scope, child.name), in_fn)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, scope if in_fn else (*scope, child.name), True)
            else:
                if _is_direct(child):
                    name = ".".join(scope) or "<module>"
                    found.setdefault(f"{module}.{name}", []).append(child.lineno)
                walk(child, scope, in_fn)

    walk(ast.parse(source), (), False)
    return found


def scan(root: Path) -> dict[str, list[int]]:
    """Every direct write under `root`, keyed `module.function`. `atomic.py` is the one
    module that makes the calls on purpose and is checked on its own below."""
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        if module == HELPER:
            continue
        found.update(_by_function(path.read_text(), module))
    return found


def violations(found: dict[str, list[int]], allowed: dict[str, str]) -> list[str]:
    """What is wrong with `found` against `allowed`: an unlisted writer, a reason too short
    to be an argument, or an entry that no longer names a direct write."""
    problems = [
        f"{name} writes a final path directly at line(s) {lines}. A process killed mid-write "
        f"leaves a truncated file there, and the next read serves garbage or refuses the "
        f"last-good state it exists to serve. Go through `hub.atomic`."
        for name, lines in found.items() if name not in allowed]
    problems += [
        f"{name} is allowed with a reason of {len(why.split())} words; an exception is a "
        f"decision on the record and needs at least {MIN_REASON_WORDS}."
        for name, why in allowed.items() if len(why.split()) < MIN_REASON_WORDS]
    problems += [
        f"{name} is allowed but no longer writes directly; delete the entry."
        for name in allowed if name not in found]
    return problems


def test_every_write_in_the_tree_is_atomic():
    problems = violations(scan(SRC), ALLOWED)
    assert not problems, "\n".join(problems)


def test_the_helper_is_the_one_place_that_writes():
    """The helper itself is where the three calls are allowed, and it has to make them --
    a helper that stopped writing would leave every module green and inert."""
    found = _direct_writes((SRC / f"{HELPER}.py").read_text())
    assert len(found) == 3, f"hub.{HELPER} should hold exactly the three writes, found {found}"


def test_the_scan_sees_a_direct_write():
    """The scan against a planted call, so a green run is evidence of routing and not of a
    scanner that matches nothing (the failure `test_guards_are_load_bearing.py` catalogues)."""
    assert _direct_writes("p.write_text(s)\nq.write_parquet(t)\n") == [1, 2]
    assert _direct_writes("atomic.write_text(p, s)\natomic.write_parquet(df, p)\n") == []


def test_a_planted_direct_write_in_a_new_module_is_found(tmp_path):
    """Rule 18 for the whole-tree claim: a module nobody listed anywhere, with a direct write
    in a method and another at top level, is found and named `module.function`."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "fresh.py").write_text(
        "def dump(df, out):\n    df.write_parquet(out)\n\n"
        "class K:\n    def go(self, p):\n        def inner():\n            p.write_text('x')\n"
        "        inner()\n")
    (tmp_path / "clean.py").write_text("from hub import atomic\natomic.write_text(p, 's')\n")
    found = scan(tmp_path)
    assert found == {"sub.fresh.dump": [2], "sub.fresh.K.go": [7]}
    assert len(violations(found, {})) == 2


def test_a_planted_allowlist_entry_with_a_short_reason_fails():
    found = {"mod.fn": [3]}
    assert violations(found, {"mod.fn": "scratch"}) != []
    assert any("1 words" in p for p in violations(found, {"mod.fn": "scratch"}))
    long_enough = " ".join(["because"] * MIN_REASON_WORDS)
    assert violations(found, {"mod.fn": long_enough}) == []


def test_a_planted_allowlist_entry_for_a_function_that_does_not_write_is_stale():
    long_enough = " ".join(["because"] * MIN_REASON_WORDS)
    assert any("no longer writes" in p for p in violations({}, {"mod.gone": long_enough}))


def test_the_committed_allowlist_argues_for_every_entry():
    short = {k: v for k, v in ALLOWED.items() if len(v.split()) < MIN_REASON_WORDS}
    assert not short, short
