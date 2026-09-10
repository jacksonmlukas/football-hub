"""Every place in `hub.draft` that asks a frame what a build stage left, and why it may.

`BuildReport` exists because consumers used to infer what had happened by sniffing for
columns, and issue #131 gave that question one owner. But the report could not *travel*:
every function downstream of `build` took a bare `pl.DataFrame`, so eleven sites went on
answering locally. Issue #199 gave the report a way to ride along -- `board.report_for`, and
a `report` parameter on the functions that needed one -- and moved those sites onto it.

**The count is the contract, and it is why this file exists rather than a note in a
docstring.** #131 was re-litigated file by file across #143-#148 and #164, each ticket
finding another site the last one did not own, and each closing with the same criterion:
*the column-sniffing this replaces is removed at the sites the report now answers for, or
each survivor carries why*. A criterion that is checked by reading is one that decays. This
one fails a run.

**What counts as a site.** A membership test against `.columns` that names a column some
optional build stage leaves -- `board.STAGE_COLUMNS`, which is the declaration of exactly
that, so this scan widens on its own when a stage does. Naming it by string literal or
through a collection literal bound in the same function; `backtest.correction_report`'s
five-column `need` set is the second form and the reason the scan bothers with it.

**What deliberately does not count**, so that a survivor's reason is never "the scanner
could not see it":

* a column named by a *parameter* rather than by this module -- `correct_projection(board,
  column="proj_blend")` is checking the caller's column, and its own frame reads are listed
  below on their merits;
* a read of a column no optional stage leaves -- `ecr_sd`, `team`, `page_type` -- which is
  not a question about what a stage did and never had an owner to move to;
* `BuildReport.of_served` itself, which is the one derivation this whole arrangement routes
  through, and which reads `STAGE_COLUMN` rather than any literal of its own.

**Adding a site is allowed. Adding it silently is not.** A new entry here has to say which
of the two acts it is -- reading a column for its *arithmetic*, or asking it for
*provenance* -- and the word `provenance` has to appear in the enclosing function, because
that is the distinction #164 wrote down and the one a reader needs to check the claim.
"""
import ast
from pathlib import Path

import pytest

from hub.draft.board import STAGE_COLUMNS

DRAFT = Path(__file__).resolve().parents[2] / "src" / "hub" / "draft"

STAGE_COLUMN_NAMES = frozenset(c for cols in STAGE_COLUMNS.values() for c in cols)

# Collection literals only. A name bound to an arbitrary expression that happens to mention
# a stage column -- `pool = board.filter(pl.col("adp") ...)` -- is not a set of column names,
# and treating it as one made this scan report the frame itself as a sniff site.
COLLECTION = (ast.Set, ast.List, ast.Tuple, ast.Dict)

# The census. `(module, function)` -> why that site reads the frame instead of the report.
#
# All six are answered. The two that were not were `report.injuries`, which held a
# `BuildReport`, gated on it, and re-derived two stages from the frame anyway; issue #146
# closed them by reading `report.adp` and `report.durability` instead, and the count below
# fell from eight to six because it did. #199 left them here on purpose rather than landing
# them without the rule test #146's acceptance criteria asked for.
SURVIVORS: dict[tuple[str, str], str] = {
    ("backtest.py", "correction_report"):
        "arithmetic: selects five columns and subtracts two, and `correction_tripwire` "
        "reaches it with hand-built frames that have no report behind them (#164)",
    ("optimize.py", "corrected_adp"):
        "arithmetic, and a producer besides -- `board._attach_market` calls it during the "
        "build, before the stage whose flag it would read has been marked",
    ("durability.py", "correct_projection"):
        "producer: runs inside `board._stage`, which is what sets the flag, so no report "
        "describing this frame exists yet",
    # `("regression.py", "correct_projection")` was the fourth survivor -- "producer, same as
    # durability's" -- until #48 deleted the function. Its site did not move to the report; it
    # stopped existing, because #186 found the correction it guarded did not earn its place.
    # Recorded here rather than dropped silently, since the count below is the criterion and a
    # number that fell needs a reason as much as one that rose.
}

# Two sites per function at two of them, one at the third. Stated as a number because that is
# the acceptance criterion: a new sniff has to move a figure a person wrote down. Eight until
# #146, when `report.injuries` stopped being two of them; six until #48 removed
# `regression.correct_projection`, which was the fifth.
EXPECTED_SITES = 5


def _stage_literals(node: ast.AST) -> set[str]:
    return {c.value for c in ast.walk(node)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and c.value in STAGE_COLUMN_NAMES}


def _sites(path: Path) -> list[tuple[str, int, frozenset[str]]]:
    """Every stage-column membership test in one module, as (function, line, columns)."""
    src = path.read_text()
    tree = ast.parse(src)
    found = []
    for fn in (n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        bound = {t.id: _stage_literals(n.value)
                 for n in ast.walk(fn)
                 if isinstance(n, ast.Assign) and isinstance(n.value, COLLECTION)
                 and _stage_literals(n.value)
                 for t in n.targets if isinstance(t, ast.Name)}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            if ".columns" not in (ast.get_source_segment(src, node) or ""):
                continue
            cols = _stage_literals(node)
            for ref in ast.walk(node):
                if isinstance(ref, ast.Name) and ref.id in bound:
                    cols |= bound[ref.id]
            if cols:
                found.append((fn.name, node.lineno, frozenset(cols)))
    return found


def _census() -> list[tuple[str, str, int, frozenset[str]]]:
    return [(p.name, fn, line, cols)
            for p in sorted(DRAFT.glob("*.py"))
            for fn, line, cols in _sites(p)]


def test_every_site_that_asks_a_frame_what_a_stage_left_is_one_this_file_names():
    """The contract. A twelfth site cannot appear without this test going red.

    Keyed on `(module, function)` and not on a line number on purpose: issue #199 opened
    with eleven line numbers that were all wrong by the time anyone read them, because the
    file had moved underneath the ticket. A function name survives an edit above it.
    """
    census = _census()
    got = {(mod, fn) for mod, fn, _, _ in census}
    missing = sorted(SURVIVORS.keys() - got)
    added = sorted(got - SURVIVORS.keys())
    assert not added, (
        f"new site(s) asking a frame what a build stage left: {added}. Either ask the "
        f"`BuildReport` -- `board.report_for(frame, report)` resolves one from a frame and "
        f"an optional report -- or add the site here with the reason it may not.")
    assert not missing, (
        f"{missing} no longer reads a stage column; remove it from SURVIVORS so the count "
        f"keeps meaning something.")
    assert len(census) == EXPECTED_SITES, (
        f"{len(census)} sites, expected {EXPECTED_SITES}: {sorted(census)}")


@pytest.mark.parametrize(("module", "function"), sorted(SURVIVORS))
def test_each_survivor_says_which_of_the_two_acts_it_is(module: str, function: str):
    """#199's fourth criterion, enforced rather than reviewed.

    Reading a column for its arithmetic and asking it for provenance look identical on the
    line and are opposite acts, so a survivor has to name the distinction. `provenance` is
    the word #164 used for it, and every survivor's reason above turns on it.
    """
    src = (DRAFT / module).read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == function)
    body = "\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno])
    assert "provenance" in body, (
        f"{module}:{function} reads a stage column off the frame but never says whether it "
        f"is doing arithmetic on it or guessing at what `build` did. SURVIVORS records: "
        f"{SURVIVORS[module, function]}")


def test_the_scan_can_see_a_sniff_written_the_ordinary_way():
    """The harness, proved on source it is handed rather than on the tree.

    A census that silently matches nothing is the failure mode `tests/contracts/
    test_guards_are_load_bearing.py` was written after eight of, and it passes exactly as
    loudly as a census that is right. So: a plain sniff, a sniff hidden behind a set, and
    two reads that must NOT count.
    """
    src = '''
def consumer(board, report):
    if "td_luck" not in board.columns:
        return None
    need = {"adp", "proj_blend"}
    if not need <= set(board.columns):
        return None
    if "ecr_sd" in board.columns:          # not a stage column
        pass
    pool = board.filter(pl.col("adp") > 1)  # not a membership test
    return pool
'''
    path = DRAFT / "__scan_probe__.py"
    path.write_text(src)
    try:
        got = _sites(path)
    finally:
        path.unlink()
    assert [(fn, cols) for fn, _, cols in got] == [
        ("consumer", frozenset({"td_luck"})),
        ("consumer", frozenset({"adp", "proj_blend"})),
    ], got
