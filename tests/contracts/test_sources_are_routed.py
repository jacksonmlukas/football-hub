"""The modules whose numbers a gate publishes reach nflverse through the fetch layer.

A published gate number that rests on an unpinned fetch cannot say what data produced it, and
#71 made that concrete: the backtest now stamps a `data_digest` folded from what the run
actually loaded. A call site that goes straight to `nflreadpy` contributes nothing to that
digest, so the stamp would quietly describe less than the run read -- provenance that looks
present and is partial, which is worse than none.

**Named modules rather than all of `src/hub`.** Twelve modules still import `nflreadpy`
directly, and routing them is #35 and the rest of U2. A list that claimed the whole tree would
have to be a list of exceptions instead, and an exception list is the thing that quietly grows.
This one only shrinks: a module joins it when it is routed and never leaves.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Routed by #34 and #36. Each of these feeds a gate, a board, or a fit whose output is
# published, which is why these went first.
ROUTED = (
    "draft/board.py",
    "draft/tune.py",
    "draft/availability.py",
    "season/weekly_gate_data.py",
    "models/experiment.py",
    "draft/durability.py",
    "draft/regression.py",
    "schedule.py",
)


@pytest.mark.parametrize("module", ROUTED)
def test_a_routed_module_does_not_fetch_directly(module):
    """`import nflreadpy` anywhere in one of these is a call site that left the pinned path.

    The AST, not a text search: this file names the import it forbids, and a substring check
    would match its own prose the way `test_every_optional_stage_goes_through_the_helper`
    found `report.adp = True` inside a docstring explaining why it no longer exists.
    """
    tree = ast.parse((ROOT / "src" / "hub" / module).read_text())
    direct = [
        n.lineno for n in ast.walk(tree)
        if (isinstance(n, ast.Import) and any(a.name == "nflreadpy" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module == "nflreadpy")
    ]
    assert not direct, (
        f"src/hub/{module} imports nflreadpy at line(s) {direct}. A gate-feeding read that "
        f"bypasses `hub.fetch.nflverse` contributes nothing to the run's data digest, so the "
        f"published stamp describes less data than the run actually read.")


def test_the_routed_list_only_grows():
    """Every module named here still exists, so the list cannot be quietly emptied by a
    rename -- which would leave the test green and the guard covering nothing."""
    missing = [m for m in ROUTED if not (ROOT / "src" / "hub" / m).exists()]
    assert not missing, f"the routed list names modules that are gone: {missing}"
