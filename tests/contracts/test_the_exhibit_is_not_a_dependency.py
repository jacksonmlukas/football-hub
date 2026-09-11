"""`hub.exhibits` is read by the harness that re-runs it and by nothing the product ships.

ADR-0009 removed championship equity from the draft-night output; ADR-0007 keeps its code
re-runnable. Until #198 the two were reconciled by leaving a thousand lines of removed arm
inside `hub.draft`, where a reader could not tell them from the code the product reads
without a call-site census -- and the census is what this file replaces. It holds three
things: that the removed arm lives under `hub.exhibits` and not under `hub.draft`; that the
only production importer of an exhibit is `hub.draft.backtest`, the harness ADR-0009 names
as the way to reopen the question; and that no exhibit is imported by the two draft-night
tools or the publisher, which is the exact wiring REMOVE undid.

Read off the AST, never off the text, for the reason `test_avoided_terms.py` gives: a grep
fires on the docstring explaining the rule, and a gate that fires on its own explanation is
one somebody deletes.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"

# The one production reader an exhibit may have. `hub.draft.backtest` is the P0b harness --
# ADR-0007's "committed, tested and re-runnable" -- and ADR-0009's *"reopening this means
# re-running `hub.draft.backtest`"*. Everything else in `src/` that wants championship
# equity is wiring it back into the product, which is the decision ADR-0009 made and a
# re-run has to unmake first.
HARNESS = "draft/backtest.py"

# The removed arm, by name. Each was reachable only from the harness on the 2026-09-06
# census, and each now lives in `hub.exhibits.championship_equity`. `tag_for` is not here:
# it labelled the equity column the board no longer prints, had no caller and no
# measurement to re-run, and was deleted rather than exhibited.
REMOVED_ARM = ("win_probability", "champion_probability", "rank_tiers", "_lift_frame")


def _imports_of(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every `hub.exhibits...` import in one file, as (line, module)."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("hub.exhibits"):
            out.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names
                    if a.name.startswith("hub.exhibits")]
    return out


def test_the_only_production_reader_of_an_exhibit_is_the_harness():
    """Anything else importing `hub.exhibits` has put a removed measurement back on a path
    the product reads, without the re-run ADR-0009 requires first."""
    readers = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        if rel.startswith("exhibits/") or rel == HARNESS:
            continue
        if found := _imports_of(path):
            readers[rel] = found
    assert not readers, (
        f"a module outside `hub.exhibits` and the harness imports an exhibit: {readers}. "
        f"That is championship equity wired back into the product; ADR-0009 says a re-run "
        f"of `hub.draft.backtest` comes first, and its verdict decides.")


def test_the_harness_does_read_the_exhibit():
    """The rule above is only a rule if the harness is the exception to it. A harness that
    stopped importing the exhibit would leave the measurement with no re-runner, which is
    the ADR-0007 failure this package exists to prevent."""
    found = _imports_of(SRC / HARNESS)
    assert found, (
        "hub.draft.backtest no longer reaches championship equity through the exhibit, so "
        "nothing re-runs the measurement ADR-0009 rests on")


@pytest.mark.parametrize("name", REMOVED_ARM)
def test_the_removed_arm_is_defined_in_the_exhibit_and_nowhere_in_the_draft_package(name):
    """A reader of `hub.draft` can tell the exhibit from the product without a census only
    if the removed arm is not in `hub.draft` at all -- a copy left behind, or a re-definition
    under the old name, puts the census back."""
    homes = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                homes.append(str(path.relative_to(SRC)))
    assert "exhibits/championship_equity.py" in homes, f"`{name}` is not in the exhibit"
    # Scoped to `hub.draft` rather than "nowhere else": `hub.season.lineup.win_probability`
    # is a different, live function -- P(this lineup outscores that one on Sunday) -- and
    # sharing a name with the removed arm is not being it.
    left = [h for h in homes if h.startswith("draft/")]
    assert not left, (
        f"`{name}` is still defined in the draft package at {left}; the removed arm is "
        f"defined in the exhibit and a copy left behind puts the census back")


@pytest.mark.parametrize("target", ["hub.draft.board", "hub.draft.live", "hub.publish"])
def test_the_draft_night_tools_and_the_publisher_do_not_reach_the_exhibit_transitively(target):
    """Not only a direct import: `hub.draft.board`, `hub.draft.live` and `hub.publish` must
    not import anything that imports an exhibit, or REMOVE is undone one hop away.

    Asked of a fresh interpreter rather than of this one. The first version evicted
    `hub.exhibits.*` from `sys.modules` and re-imported the target in-process, which left every
    later test holding two copies of the exhibit -- `backtest` bound to the old one, a
    monkeypatch landing on the new one -- and failed three seeding tests that had nothing to
    do with it, but only when the contracts ran first. A subprocess is the same question with
    no side on this process.
    """
    import subprocess
    import sys

    code = (f"import sys; import {target}; "
            f"print(sorted(m for m in sys.modules if m.startswith('hub.exhibits')))")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, check=True)
    assert run.stdout.strip() == "[]", (
        f"importing {target} loaded {run.stdout.strip()}: the exhibit is on a product path")
