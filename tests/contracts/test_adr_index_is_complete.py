"""`docs/architecture.md` must list every decision record, and no document may state a count
that contradicts the directory.

On 2026-09-07 three places in this repo gave three different answers to "how many decision
records are there", and the directory agreed with none of them:

    README.md          "index of twelve decision records"
    docs/method.md     "Fourteen decisions", linking docs/adr/
    docs/architecture.md   listed 0001-0014
    docs/adr/          23 files

Nine ADRs were absent from the index, one of them ADR-0019 -- the decision that changed the
adoption rule for every gate in the repo. So the index had been silent about the rule the rest
of it is scored under for three days.

**Why prose lost, and why a reminder would lose again.** Nothing connected the three counts to
the directory, so each was written once, was true once, and was never re-derived. That is the
same shape as `test_every_contract_is_applied.py` -- one instance of a defect found and fixed,
the others surviving for want of anything connecting them -- and the same shape as the ratchet
in `test_avoided_terms.py`, whose own preamble outlived its subject twice.

The coincidence is the part worth writing down. `docs/method.md` separately reports that
*"Fourteen things have been measured properly"*, a count of measurements and a genuinely
different fourteen that is still correct. For eleven days in August the two numbers were equal,
sitting in one file agreeing with each other while both descriptions of `docs/adr/` went wrong.
A reader checking one against the other would have been reassured. So this file checks counts
against the **directory**, never against another document.

**What this can and cannot decide.** It decides membership -- an ADR file with no row, or a row
with no file -- because that is mechanical and total. It decides a stated count only where the
count is stated *about* `docs/adr/` or about the index, which is why `_counts_about_the_adrs`
requires a link to one of them on the same line rather than scanning for number words. The
measurement fourteen above sits in a line that links neither, and must keep passing.

It does not decide whether a row describes its ADR correctly, whether a status is current, or
whether a re-scoring note is right. Those are prose and a regex cannot read them.

**Non-vacuity.** A scan matching nothing would pass every assertion here while proving nothing,
which is what `test_guards_are_load_bearing.py` exists about. So the scan is required to find
ADRs at all, to find rows at all, and to find at least one stated count -- an index that stopped
naming its own size would fail rather than quietly satisfying every other assertion.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = ROOT / "docs" / "adr"
INDEX = ROOT / "docs" / "architecture.md"
README = ROOT / "README.md"
METHOD = ROOT / "docs" / "method.md"

# `0007-measurements-...md` -> the four-digit number that names the decision.
ADR_FILE = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")

# A row linking one ADR: `[0007](adr/0007-....md)` in the index.
ROW_LINK = re.compile(r"\[(\d{4})\]\((?:\./)?adr/(\d{4})-[a-z0-9-]+\.md\)")

# Number words this repo actually writes counts in, plus digits.
WORDS = {
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
}

# "twenty-three decision records", "Fourteen decisions", "23 files". The noun is required:
# a bare number near the word `adr/` is not a claim about how many there are.
COUNT = re.compile(
    r"\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r"|\d{1,3})\b"
    r"[\s*_]*(?:decision records?|decisions|ADRs?|files)\b",
    re.I,
)

# A line only states a count *about the ADRs* if it also points at them.
POINTS_AT_THE_ADRS = re.compile(r"docs/adr/|\(adr/\)|architecture\.md|\bdocs/architecture\b")

# A count inside quotation marks is a *quotation of what some document said*, not a claim this
# repo is making now. `docs/architecture.md` records the three wrong counts of 2026-09-07
# verbatim, and a history that could not name the number it corrects would be useless. So
# quoted spans are removed before a line is read for claims -- which is also the rule that
# keeps this scan from forbidding the write-up of its own defect.
QUOTED = re.compile(r"\*?\"[^\"]*\"\*?|\*?“[^”]*”\*?")


def _numbers_on_disk() -> set[str]:
    """Every decision record in `docs/adr/`, by its four-digit number."""
    found = set()
    for path in sorted(ADR_DIR.glob("*.md")):
        match = ADR_FILE.match(path.name)
        assert match, f"{path.name} does not match the ADR filename shape NNNN-slug.md"
        found.add(match.group(1))
    return found


def _numbers_in_the_index() -> set[str]:
    """Every ADR the index links, and a link's text must agree with the file it points at."""
    found = set()
    for label, target in ROW_LINK.findall(INDEX.read_text(encoding="utf-8")):
        assert label == target, (
            f"the index links [{label}] to file {target}-*.md; a row that names one ADR and "
            f"points at another is worse than a missing row"
        )
        found.add(label)
    return found


def _counts_about_the_adrs(path: Path) -> list[tuple[str, int]]:
    """Stated counts of decision records in one file, as (the line, the number claimed)."""
    out: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not POINTS_AT_THE_ADRS.search(line):
            continue
        claimed_now = QUOTED.sub(" ", line)
        for raw, in (m.groups()[:1] for m in COUNT.finditer(claimed_now)):
            key = raw.lower()
            out.append((line.strip(), WORDS[key] if key in WORDS else int(raw)))
    return out


def test_the_scan_finds_decision_records_at_all() -> None:
    """Non-vacuity: the assertions below prove nothing against an empty directory."""
    on_disk = _numbers_on_disk()
    assert len(on_disk) >= 20, f"expected the repo's decision records, found {len(on_disk)}"


def test_the_index_lists_every_decision_record() -> None:
    missing = sorted(_numbers_on_disk() - _numbers_in_the_index())
    assert not missing, (
        f"{len(missing)} decision record(s) exist and are absent from docs/architecture.md: "
        f"{', '.join(missing)}. This is the defect issue #51 was filed about -- nine were "
        f"missing, one of them ADR-0019, the rule every gate in the repo is scored under."
    )


def test_the_index_lists_nothing_that_does_not_exist() -> None:
    dangling = sorted(_numbers_in_the_index() - _numbers_on_disk())
    assert not dangling, (
        f"docs/architecture.md links decision record(s) with no file: {', '.join(dangling)}"
    )


def test_no_document_states_a_count_that_contradicts_the_directory() -> None:
    """A count written about `docs/adr/` must be the number of files in `docs/adr/`.

    Checked against the directory and never against another document, because the two
    fourteens in `docs/method.md` agreed with each other for eleven days while both
    descriptions of the ADRs were wrong.
    """
    actual = len(_numbers_on_disk())
    wrong: list[str] = []
    for path in (README, METHOD, INDEX):
        for line, claimed in _counts_about_the_adrs(path):
            if claimed != actual:
                wrong.append(f"{path.relative_to(ROOT)}: claims {claimed}, there are {actual}\n    {line}")
    assert not wrong, "a stated count contradicts docs/adr/:\n  " + "\n  ".join(wrong)


def test_a_count_is_stated_somewhere() -> None:
    """Non-vacuity for the assertion above: an index that stopped naming its own size would
    satisfy it trivially, and the drift this file exists about began with a count nobody
    re-derived rather than with a count nobody wrote."""
    stated = _counts_about_the_adrs(INDEX)
    assert stated, "docs/architecture.md states no count of the decision records it indexes"


def test_a_quoted_historical_count_is_not_read_as_a_claim() -> None:
    """The `QUOTED` rule, held by the case it exists for.

    `docs/architecture.md` records what `README.md` and `docs/method.md` used to say, verbatim
    and in quotation marks. Those are quotations of a wrong number, not assertions of it, and a
    scan that could not tell the difference would forbid the write-up of the very defect it
    guards -- the failure mode `test_avoided_terms.py` names, where narrowing until nothing is
    seen is the easy way to a green build.
    """
    history = '| `README.md` | *"index of twelve decision records"* | No, see docs/adr/ |'
    assert COUNT.search(history), "the quoted count must be findable before it is discounted"
    assert not COUNT.search(QUOTED.sub(" ", history)), (
        "a count inside quotation marks is a quotation, not a claim"
    )

    claim = "There are twenty-three decision records in docs/adr/."
    assert COUNT.search(QUOTED.sub(" ", claim)), (
        "an unquoted count on a line pointing at the ADRs is a claim and must still be read"
    )


@pytest.mark.parametrize("path", [README, METHOD, INDEX])
def test_the_documents_that_carried_the_wrong_counts_still_exist(path: Path) -> None:
    """The three files that disagreed on 2026-09-07. A rename that dropped one of them would
    otherwise silently narrow this scan to the two that remain."""
    assert path.is_file(), f"{path.relative_to(ROOT)} is gone; this scan no longer covers it"
