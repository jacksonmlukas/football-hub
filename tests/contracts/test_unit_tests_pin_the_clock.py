"""No unit test may read the wall clock through a *dated* fixture and get away with it.

#356: six tests pinned kickoffs on a fixed date and then read `now`/`at` from whatever day
the suite happened to run on -- green on Saturday, red on Sunday, exactly the leak method
rule 2 is about: the outcome window moved under the predictor. Two fixture builders in this
tree hand a caller a grid with real `kickoff`/`result` columns: `_clocked`
(`test_pool_weekly.py`) and `_mid_season_grid` (`test_publish.py`). A grid with no
`kickoff`/`result` columns at all is read as entirely ahead regardless of the clock --
`schedule.forecastable`'s own docstring says so -- so the several dozen calls against
`FLAT`/`HOARD`/`DOUBLE`/etc. elsewhere in `test_pool_weekly.py` are not this file's
business. Only a call against a *dated* grid is the shape that broke on a live date, and
only that shape is refused here.

Two mechanisms, because the two modules disagree about which lever pins the clock.
`pool.auto_pick`/`.weekly`/`.leverage` (and `_weekly`, the local wrapper that forwards its
kwargs straight to `pool.weekly`) take a `now` keyword directly, so pinning it is passing
the keyword with a value that is not the literal `None`. `publish.survivor` threads no such
keyword down to `schedule.forecastable` -- #356's acceptance criterion is that nothing under
`src/` grows one just to give a test a seam -- so a test that wants a fixed answer out of a
dated grid instead freezes `hub.schedule`'s own `datetime.now` through the
`_pin_schedule_clock` helper `test_publish.py` carries for exactly this. Both are "the
class" #356 asks to be refused rather than fixed one test at a time.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "tests" / "unit"

# Calls that read the clock through a `now` keyword: `pool.auto_pick`/`.weekly`/`.leverage`
# themselves, and `_weekly`, `test_pool_weekly.py`'s wrapper that forwards straight to
# `pool.weekly`.
_NOW_KEYWORD_CALLS = {"_weekly", "pool.weekly", "pool.auto_pick", "pool.leverage"}

# The fixture builder that hands a `now`-keyword call a *dated* grid (kickoffs on the
# Thursday of week 1, some already resulted) -- its presence in a test is what makes an
# omitted `now` a live bug rather than dead weight against an undated grid.
_DATED_GRID_BUILDER = "_clocked"

# The fixture builder that hands `publish.survivor` a dated grid indirectly, through a
# monkeypatched `grid_from_schedule`, in test_publish.py.
_DATED_SCHEDULE_BUILDER = "_mid_season_grid"

# The helper `test_publish.py` calls to freeze `hub.schedule`'s wall-clock read.
_CLOCK_FREEZE_CALL = "_pin_schedule_clock"


def _call_name(node: ast.Call) -> str | None:
    """`pool.weekly` for an attribute call, `_weekly` for a bare one, else `None`."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f"{f.value.id}.{f.attr}"
    return None


def _calls_named(node: ast.AST, names: set[str]) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and _call_name(n) in names]


def _now_is_pinned(call: ast.Call) -> bool:
    """A `now=` keyword is present and is not the literal `None` -- an explicit `now=None`
    is the same wall-clock read as omitting it, `forecastable`'s own default."""
    for kw in call.keywords:
        if kw.arg == "now":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
    return False


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")]


@pytest.mark.parametrize("path", sorted(UNIT.glob("*.py")), ids=lambda p: p.name)
def test_no_test_reads_a_dated_grids_clock_unpinned(path):
    """The AST, not a text search -- a call spelled across a wrapped line, or one whose
    `now=` a docstring happens to mention, is exactly the drift a substring check would
    either miss or misfire on."""
    tree = ast.parse(path.read_text())
    violations = []
    for fn in _test_functions(tree):
        if _calls_named(fn, {_DATED_GRID_BUILDER}):
            for call in _calls_named(fn, _NOW_KEYWORD_CALLS):
                if not _now_is_pinned(call):
                    violations.append(
                        f"{path.name}:{call.lineno} {fn.name}: {_call_name(call)}() reads a "
                        f"grid built by `{_DATED_GRID_BUILDER}` without a fixed `now=`")
        if _calls_named(fn, {_DATED_SCHEDULE_BUILDER}) and not _calls_named(
                fn, {_CLOCK_FREEZE_CALL}):
            violations.append(
                f"{path.name}:{fn.lineno} {fn.name}: builds a dated grid with "
                f"`{_DATED_SCHEDULE_BUILDER}` but never calls `{_CLOCK_FREEZE_CALL}` to pin "
                f"the wall clock `publish.survivor` reads through it")
    assert not violations, (
        "clock-reading call(s) against a dated grid with no fixed `now` -- #356 is exactly "
        "this class, found the hard way (green on Saturday, red on Sunday):\n"
        + "\n".join(violations))


def test_the_builders_this_file_names_still_exist():
    """So a rename that leaves this file's names stale fails loudly rather than leaving the
    guard above checking for a fixture nothing builds any more."""
    names = {_DATED_GRID_BUILDER, _DATED_SCHEDULE_BUILDER, _CLOCK_FREEZE_CALL, "_weekly"}
    found: set[str] = set()
    for path in UNIT.glob("*.py"):
        tree = ast.parse(path.read_text())
        found |= {n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name in names}
    missing = names - found
    assert not missing, f"named but no longer defined anywhere in tests/unit: {missing}"
