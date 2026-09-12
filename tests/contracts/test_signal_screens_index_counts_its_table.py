"""`docs/signal-screens.md`'s opening paragraph counts the table beneath it.

The paragraph said *"Six hypotheses ... Five null, one not"* above a table that had grown to
twenty rows with five marked positive (issue #52, re-audit finding of 2026-09-07). Every one
of those rows was added by someone who had read the paragraph; the counts were true when
written and nothing re-derived them. So the four numbers the paragraph states -- rows in
the table, rows withdrawn, rows screened before the season, rows marked positive -- are read
off the table here, and a row added without the paragraph moving fails the build.

What this holds is arithmetic, not judgement: the preseason rows are the ones above the
first `next week` horizon, a hypothesis is a distinct (signal, horizon) pair among them with
the horizon read up to its first comma (the snap-share trend's "before week 8" row is a split
of one hypothesis, the depth-chart climb's two horizons are two), and positive is the result
cell opening with `**POSITIVE`. A row added
in the wrong place lands on the wrong side of the split, which is a change this test would
surface rather than absorb.
"""
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs" / "signal-screens.md"

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20}


def _opening(text: str) -> str:
    """The first paragraph after the title -- the one that states the counts."""
    body = text.split("\n\n")
    return " ".join(body[1].split())


def _rows(text: str) -> list[list[str]]:
    """The index table's body rows, split into cells."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| signal | horizon |"))
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows


def _number(word: str) -> int:
    return WORDS[word.lower()]


@pytest.fixture(scope="module")
def counted():
    text = DOC.read_text()
    rows = _rows(text)
    first_weekly = next(i for i, r in enumerate(rows) if "next week" in r[1])
    preseason = rows[:first_weekly]
    return {
        "opening": _opening(text),
        "rows": len(rows),
        "withdrawn": sum("withdrawn" in r[2] for r in rows),
        "preseason_rows": len(preseason),
        # A hypothesis is a signal at a horizon; what follows the comma in the horizon cell
        # ("before week 8") is a split of the same hypothesis, so it is not a second one.
        "preseason_hypotheses": len({(r[0].strip("*"), r[1].split(",")[0]) for r in preseason}),
        "weekly_rows": len(rows) - len(preseason),
        "positive": sum(r[2].startswith("**POSITIVE") for r in rows),
    }


def test_the_table_is_where_the_paragraph_says_it_is(counted):
    assert counted["rows"] >= 6, "the index table was not found or has lost its rows"


def test_the_opening_counts_the_rows(counted):
    m = re.match(r"(\w+) rows for beating consensus, (\w+) since withdrawn\.",
                 counted["opening"])
    assert m, "the opening no longer states the row count and the withdrawn count"
    assert _number(m.group(1)) == counted["rows"], (
        f"the paragraph says {m.group(1)} rows; the table has {counted['rows']}")
    assert _number(m.group(2)) == counted["withdrawn"], (
        f"the paragraph says {m.group(2)} withdrawn; the table marks {counted['withdrawn']}")


def test_the_opening_splits_preseason_from_in_season(counted):
    m = re.search(r"The first (\w+) rows are the (\w+) hypotheses screened before the "
                  r"season: (\w+) were null and \*\*one not\*\*", counted["opening"])
    assert m, "the opening no longer states the preseason split"
    assert _number(m.group(1)) == counted["preseason_rows"], (
        f"the paragraph says {m.group(1)} preseason rows; the table has "
        f"{counted['preseason_rows']} above the first next-week horizon")
    assert _number(m.group(2)) == counted["preseason_hypotheses"], (
        f"the paragraph says {m.group(2)} preseason hypotheses; the preseason rows name "
        f"{counted['preseason_hypotheses']} distinct signals")
    assert _number(m.group(3)) == counted["preseason_hypotheses"] - 1, (
        "the preseason nulls are the preseason hypotheses less the one survivor")
    m = re.search(r"added (\w+) rows at the next-week horizon", counted["opening"])
    assert m, "the opening no longer states how many rows the weekly screen added"
    assert _number(m.group(1)) == counted["weekly_rows"], (
        f"the paragraph says {m.group(1)} weekly rows; the table has {counted['weekly_rows']}")


def test_the_opening_counts_the_positive_rows(counted):
    m = re.search(r"\*\*(\w+) rows positive\*\*", counted["opening"])
    assert m, "the opening no longer states the positive count"
    assert _number(m.group(1)) == counted["positive"], (
        f"the paragraph says {m.group(1)} rows positive; the table marks {counted['positive']}")
