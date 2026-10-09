"""The count of measurements reads the same everywhere it is stated as current -- #323.

`docs/method.md`'s "The record" table is where the count lives. For 2026-09-11 to 2026-10-08 it
read fifteen in `method.md` and in `detectable-effects.md`, and fourteen in three other pages
(`architecture.md`, `weekly-projection-plan.md`, ADR-0015) that had been right on their own
dates and said so only sometimes: #282 closed with the split named as remaining. A count that
was true on the day a page was written is a record and is kept; what is not allowed is a count
that disagrees with the table and carries nothing saying which day it was true on.

What this decides: every `<number word> measurements` / `<number word> things have been
measured` in `docs/` and `README.md` either equals the table's last row number, or sits in a
paragraph that dates itself, or is in a file whose whole text is a dated record
(`DATED_RECORDS`, each with its reason). What it does not decide is whether a row belongs in
the table: `method.md` itself says some measurements since 2026-08-29 are not yet rowed.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
METHOD = DOCS / "method.md"

WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}

# Files whose whole text is a dated record and which say so at the top, so a count in them is
# the count of their date by construction. The reason is the file's own header.
DATED_RECORDS = {
    "where-to-look-next.md": "'Written 2026-08-25 ... the table in method.md is the current "
                             "count' is its second line, and its title keeps the original count",
}

# A paragraph that dates its count: "on that date", "this plan's date", "since 2026-09-11",
# "when this was written", "on this ADR's date", or an ISO date beside the number.
DATED = re.compile(
    r"(on (that|this \w+'s|its|the) date|when (this|it) was written|since 20\d\d-\d\d-\d\d"
    r"|20\d\d-\d\d-\d\d)", re.I)

COUNT = re.compile(
    r"\b(" + "|".join(w for w in WORDS if w != "zero") + r")\s+(?:prior\s+)?measurements\b"
    r"|\b(" + "|".join(w for w in WORDS if w != "zero")
    + r")\s+things\s+(?:have|had)\s+been\s+measured\b", re.I)


def _table_count() -> int:
    """The last row number of "The record" table: `15`, or the upper end of a `1-2` range (an en dash in the file)."""
    text = METHOD.read_text()
    block = text.split("## The record", 1)[1].split("\n---", 1)[0]
    last = 0
    for ln in block.splitlines():
        m = re.match(r"\|\s*(\d+)(?:\s*[\u2013-]\s*(\d+))?\s*\|", ln)
        if m:
            last = int(m.group(2) or m.group(1))
    return last


def _claims(text: str) -> list[tuple[int, str, bool]]:
    """(stated number, the paragraph it sits in, whether the paragraph dates itself)."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        flat = " ".join(block.split())
        for m in COUNT.finditer(flat):
            word = (m.group(1) or m.group(2)).lower()
            out.append((WORDS[word], flat, bool(DATED.search(flat))))
    return out


def _pages() -> list[Path]:
    return sorted([*DOCS.glob("*.md"), *DOCS.glob("adr/*.md"), ROOT / "README.md"])


# "two measurements" is a sentence about something else; a stale count of *the table* is in its
# neighbourhood, which is where every drift so far has been (fourteen for fifteen).
NEAR = 3


def _disagreements(count: int, pages: dict[str, str]) -> list[str]:
    bad = []
    for name, text in pages.items():
        if Path(name).name in DATED_RECORDS:
            continue
        for stated, para, dated in _claims(text):
            if 0 < abs(stated - count) <= NEAR and not dated:
                bad.append(f"{name}: says {stated}, the table says {count}: {para[:110]}")
    return bad


def test_the_table_has_the_count_its_own_sentence_states() -> None:
    m = re.search(r"\b(\w+) things have been measured properly", METHOD.read_text())
    assert m is not None and m.group(1).lower() in WORDS
    assert WORDS[m.group(1).lower()] == _table_count()


def test_no_page_states_a_different_count_without_dating_it() -> None:
    pages = {str(p.relative_to(ROOT)): p.read_text() for p in _pages()}
    assert _disagreements(_table_count(), pages) == []


def test_the_scan_sees_a_stale_undated_count_and_passes_a_dated_one() -> None:
    """Planted: the same sentence undated fails, dated passes, and a matching count passes."""
    n = _table_count()
    stale = "The record is fourteen measurements and the five nulls lost."
    dated = "The record is fourteen measurements (the count on this plan's date) and so on."
    right = f"The record is {next(w for w, i in WORDS.items() if i == n)} measurements."
    assert n != 14, "the plant needs a count that is not fourteen"
    assert _disagreements(n, {"x.md": stale}), "an undated stale count must be seen"
    assert _disagreements(n, {"x.md": dated}) == []
    assert _disagreements(n, {"x.md": right}) == []


def test_the_dated_records_are_files_that_exist_and_date_themselves() -> None:
    for name in DATED_RECORDS:
        text = (DOCS / name).read_text()
        assert DATED.search(text[:600]), f"{name} no longer dates itself at the top"
