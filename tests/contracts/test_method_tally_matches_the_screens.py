"""`docs/method.md` rule 14 carries a tally of how many tests each screen runs, and the tally
is read off the code -- #37.

The count it replaces lived in a docstring and moved from nine to eight (#170) with nothing on
any page depending on it, which is how a multiple-comparison family comes to understate
itself: a feature enters or leaves `FEATURES` and the prose keeps the old number. So the
family sizes the write-up states are held to what the screens report, and the one family
that has no code -- the six preseason hypotheses, run by hand -- is held to the count the
index page states, the way `test_signal_screens_index_counts_its_table.py` derives it.

What this holds is arithmetic: the bold integer in a tally row against the length of the
tuple that row names. Which features belong in a family is the screen module's decision and
is tested there.
"""
import re
from pathlib import Path

import pytest

from hub.models import panel
from hub.models import weekly_screen as ws

DOCS = Path(__file__).resolve().parents[2] / "docs"
METHOD = DOCS / "method.md"
INDEX = DOCS / "signal-screens.md"
WEEKLY = DOCS / "weekly-screen.md"

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15}


def _tally() -> dict[str, str]:
    """The tally table's rows, keyed by their `screen` cell, valued by the `tests` cell."""
    lines = METHOD.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("| screen | tests in the family |"))
    rows: dict[str, str] = {}
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows[cells[0]] = cells[1]
    return rows


def _bold_int(cell: str) -> int:
    found = re.findall(r"\*\*(\d+)\*\*", cell)
    assert len(found) == 1, f"one bold count expected in {cell!r}"
    return int(found[0])


@pytest.fixture(scope="module")
def tally() -> dict[str, str]:
    rows = _tally()
    assert len(rows) == 6, f"six families are tallied; the table has {len(rows)} rows"
    return rows


def _row(tally: dict[str, str], key: str) -> str:
    matches = [v for k, v in tally.items() if key in k]
    assert len(matches) == 1, f"{key!r} names one tally row, not {len(matches)}"
    return matches[0]


def test_the_alone_family_is_the_feature_tuple(tally):
    assert _bold_int(_row(tally, "alone (`--run`)")) == len(ws.FEATURES)
    assert "snap_trend" in [f.name for f in ws.FEATURES], (
        "the row says the trend is counted, so it has to be in the tuple it counts")


def test_the_routes_family_adds_one(tally):
    assert _bold_int(_row(tally, "`--routes`")) == len(ws.FEATURES) + 1


def test_the_scheme_family_adds_the_scheme_trends(tally):
    assert _bold_int(_row(tally, "`--scheme`")) == len(ws.FEATURES) + len(ws.SCHEME_TRENDS)


def test_the_usage_family_is_the_count_tuple_per_feature(tally):
    assert _bold_int(_row(tally, "Usage")) == len(panel.USAGE)


def test_the_joint_family_is_not_a_fixed_number(tally):
    """The joint screen's family is that anchor's survivors, which is a run's output and not
    a constant; a bold integer in that row would be a number nothing re-derives."""
    assert not re.search(r"\*\*\d+\*\*", _row(tally, "joint"))


def test_the_preseason_family_is_the_count_the_index_states(tally):
    """The six hypotheses have no code; the index page's opening paragraph is the record,
    and its count is itself read off the table there by its own contract test."""
    opening = " ".join(INDEX.read_text().split("\n\n")[1].split())
    m = re.search(r"the (\w+) hypotheses screened before the season", opening)
    assert m, "signal-screens.md no longer states the preseason hypothesis count"
    assert _bold_int(_row(tally, "preseason")) == WORDS[m.group(1)]


def test_the_screen_page_states_the_same_alone_family():
    """`docs/weekly-screen.md` repeats the alone family in words, beside the description of
    what a run prints; the words and the tuple have to agree for the same reason."""
    text = WEEKLY.read_text()
    m = re.search(r"is \*\*(\w+) tests\*\* at each anchor", text)
    assert m, "weekly-screen.md no longer states the alone family size"
    assert WORDS[m.group(1)] == len(ws.FEATURES)


def test_the_rate_the_pages_quote_is_the_declared_one():
    """q = 0.10 is quoted on three pages and declared once; a change to `FDR_Q` that left
    the pages behind would be the docstring-count incident with a different number."""
    quoted = f"q = {ws.FDR_Q:.2f}"
    for page in (METHOD, INDEX, WEEKLY):
        assert quoted in page.read_text(), f"{page.name} does not quote {quoted}"
