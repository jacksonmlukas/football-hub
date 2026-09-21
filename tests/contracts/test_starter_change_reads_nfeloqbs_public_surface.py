"""`hub.models.starter_change` reads `hub.fetch.nfeloqb` through its public surface only (#339).

Before this ticket the event construction reached past the fetch module's public surface six
times -- `nfeloqb.ABBREVIATIONS` three times over, `nfeloqb._blank` once -- to rebuild, by
hand, the per-team-game transform `hub.fetch.nfeloqb` already did internally for its own
state (`_long`). Three of an architecture review's findings (#301, #328, #304's mutants)
lived in that hand-built copy. #339 published the transform once, as `nfeloqb.team_games`
and `nfeloqb.schedule`, and this module now reads those -- plus `nfeloqb.state`,
`nfeloqb.read_rows`, `nfeloqb.GAME_DAY_ZONE` -- and nothing underscore-prefixed, and never
`ABBREVIATIONS` itself: 538's raw spelling table, and exactly the kind of schema knowledge a
reader reaching past the module would be reconstructing.

Read off the AST, never off the text, in the style of `test_the_weekly_model_is_market_free.py`
and `test_the_quarterback_adjustment_is_not_a_dependency.py`: a grep fires on this file's own
docstring, which names `_schedule`, `_blank` and `ABBREVIATIONS` in the history above, and a
scan that fires on its own explanation is a scan nobody trusts.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "hub"
MODULE_PATH = SRC / "models" / "starter_change.py"

# The fetch module `hub.models.starter_change` is asked to read through its public surface
# only, and the one column name -- 538's own spelling table -- reaching it directly would be
# exactly the reconstruction #339 closed, whether or not the name happens to start with `_`.
FETCH_MODULE = "hub.fetch.nfeloqb"
FORBIDDEN_NAME = "ABBREVIATIONS"


def _violations(text: str) -> list[str]:
    """Every reach past `hub.fetch.nfeloqb`'s public surface in `text`: an import of an
    underscore-prefixed name (or `ABBREVIATIONS`) from the module, or an attribute access on
    a name bound to it (`nfeloqb.<attr>`) whose attribute starts with `_` or is
    `ABBREVIATIONS`. Reported by line and by what was found, never as a bare count."""
    lines = text.splitlines()
    found: set[tuple[int, str]] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module == FETCH_MODULE:
            for alias in node.names:
                if alias.name.startswith("_") or alias.name == FORBIDDEN_NAME:
                    found.add((node.lineno, f"import of {alias.name!r} from {FETCH_MODULE}"))
        elif isinstance(node, ast.Attribute):
            base = node.value
            if (isinstance(base, ast.Name) and base.id == "nfeloqb"
                    and (node.attr.startswith("_") or node.attr == FORBIDDEN_NAME)):
                found.add((node.lineno, f"nfeloqb.{node.attr}"))
    return [f"line {n}: {what} -- {lines[n - 1].strip()}" for n, what in sorted(found)]


def test_starter_change_imports_no_underscore_name_from_nfeloqb_and_reads_no_abbreviations():
    hits = _violations(MODULE_PATH.read_text())
    assert hits == [], (
        "hub.models.starter_change reaches past hub.fetch.nfeloqb's public surface:\n  "
        + "\n  ".join(hits) + "\n"
        "hub.fetch.nfeloqb owns the source's raw schema -- the two-sided row, the blank "
        "convention, the abbreviation map -- and a reach past its public names here is "
        "exactly the hand-rebuilt copy #339 published once, as nfeloqb.team_games and "
        "nfeloqb.schedule, to close. Read the public function instead of rebuilding its "
        "transform.")


def test_the_scan_still_catches_the_shape_it_was_written_against():
    """The positive control this file's own premise needs: a scan that matches nothing is
    only as trustworthy as its last proof it can match something
    (`tests/contracts/test_guards_are_load_bearing.py`'s own reason for existing). This is
    the exact reach #339 removed, run through the same scanner as a probe rather than as an
    edit to the real module."""
    src = '''
from hub.fetch import nfeloqb
from hub.fetch.nfeloqb import _blank as also_blank

def _schedule(rows):
    home = pl.col("team1").replace(nfeloqb.ABBREVIATIONS)
    return rows.filter(~nfeloqb._blank("1"))
'''
    hits = _violations(src)
    assert any("ABBREVIATIONS" in h for h in hits), hits
    assert any("nfeloqb._blank" in h for h in hits), hits
    assert any("_blank" in h and "import" in h for h in hits), hits


def test_the_scan_does_not_fire_on_the_ordinary_public_reads_this_module_makes():
    """The negative control: `nfeloqb.team_games`, `nfeloqb.schedule`, `nfeloqb.state`,
    `nfeloqb.read_rows` and `nfeloqb.GAME_DAY_ZONE` -- the public surface this module reads
    today -- must never themselves be findings, or the assertion above is vacuous the day
    the module's own reads happen to be exactly this list."""
    src = '''
from hub.fetch import nfeloqb

def team_games(rows):
    return nfeloqb.team_games(rows)

def _schedule(rows):
    return nfeloqb.schedule(rows)

def _poll_day():
    return nfeloqb.GAME_DAY_ZONE

def main():
    rows = nfeloqb.read_rows(None)
    return nfeloqb.state(rows)
'''
    assert _violations(src) == []


def test_the_module_still_imports_nfeloqb_at_all():
    """The denominator the two tests above assume: if `starter_change.py` stopped importing
    `hub.fetch.nfeloqb` altogether, both would pass over a file that reads nothing from it,
    which is not the property #339 established."""
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "hub.fetch" in imported, (
        "hub/models/starter_change.py no longer imports from hub.fetch at all -- the tests "
        "above are vacuous over a module that reads nothing from hub.fetch.nfeloqb")
