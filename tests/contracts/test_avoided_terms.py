"""`CONTEXT.md`'s `_Avoid_` lists, enforced against the source tree and against `CONTEXT.md`.

Prose has failed to hold this rule five times. A frame was named `eligible` two lines after
the comment citing the entry whose `_Avoid_` names "eligible"; a docstring said "the sample"
against an `_Avoid_` naming "sample"; and issue #53 found eight more at once, one of them a
*new glossary entry* using "the market" bare -- the entry three other entries forbid. Every
one of those was written by someone who had read the glossary. Asking harder is not a plan.

**What this file can and cannot decide.** An `_Avoid_` list is prose, and most of it is not
mechanical. "roster" is forbidden *for the League's shape* and is the right word for your
players -- the entry says so itself. "volume" is forbidden *for Usage* and is the name of a
measured model. "backtest" is forbidden *for a screen or a gate* and is a module. A regex
cannot tell a use from the use the entry contrasts it with, and a scan that pretended it
could would be turned off inside a week.

So every term is classified, and the classification is the honest part:

* `CHECKED` -- the term has one meaning in this repo, so any occurrence is the forbidden
  one. Each carries the rule used and why that rule decides it.
* `UNCHECKED` -- no regex can decide it, and the entry's own words say why.

`test_every_avoided_term_in_the_glossary_is_classified` requires every term parsed out of
`CONTEXT.md` to sit in exactly one of them. A new `_Avoid_` term therefore forces a decision
rather than arriving unscanned, and a term struck from the glossary forces a cleanup. That
premise, and not the scan, is what stops this file quietly narrowing to nothing.

**Outstanding sites are listed, not hidden.** `OUTSTANDING` names every (file, term) pair
that violates a checked rule today, with the reason it is still there. It is a ratchet: an
unlisted pair fails, a listed pair that grows fails, and a listed pair that has gone to zero
fails as stale. Narrowing the scan until it sees nothing would have been the easier way to
get a green build and is the exact defect issue #53 was filed about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "CONTEXT.md"

# The source tree and the glossary, which is what the acceptance criterion names. Tests are
# deliberately outside it: `tests/` restates the vocabulary constantly when describing what
# it is testing, and folding it in would bury the source's own violations under hundreds of
# test-name hits. `tests/unit/test_preflight.py` is the site that costs us for that -- see
# OUTSTANDING's note on it.
SCAN_ROOTS: tuple[Path, ...] = (
    *sorted((ROOT / "src" / "hub").rglob("*.py")),
    *sorted((ROOT / "scripts").glob("*.sh")),
    ROOT / "site" / "index.html",
    CONTEXT,
)


# --- reading the glossary -----------------------------------------------------

# `**Term**:` or `**Scarcity** / **Value**:` at the start of a line.
_HEADWORD = re.compile(r"^\*\*(.+?)\*\*(?:\s*/\s*\*\*(.+?)\*\*)?:\s*$", re.M)
# `_Avoid_: ...` through to the next blank line -- the list wraps.
_AVOID = re.compile(r"^_Avoid_:(?P<body>.*?)(?=\n\s*\n|\Z)", re.M | re.S)
# Terms hide after the em dash too: "Also avoid active (...) and eligible."
_ALSO = re.compile(r"\balso avoid\b(?P<body>[^.]*)", re.I)


@dataclass(frozen=True)
class Entry:
    """One glossary entry: its headwords, the terms it forbids, and where its list sits."""

    headwords: tuple[str, ...]
    avoided: tuple[str, ...]
    span: tuple[int, int] = (0, 0)                  # line range of the `_Avoid_` paragraph

    def __repr__(self) -> str:                      # the parametrize id
        return "/".join(self.headwords)


def _terms(body: str) -> list[str]:
    """The terms one `_Avoid_` paragraph names.

    Parenthetical asides go first, because they carry commas ("roster (which is *your*
    players, not the league's shape), settings") and an em dash ("volume (which names only
    half of it -- a carry and a target ...)"). `P(win)` survives: only an aside preceded by
    whitespace is stripped, and that one is glued to its term.

    The prose after the em dash is the reason, not more terms -- except where it says "also
    avoid", which is how "active" and "eligible" reach the list at all. Losing those two
    would be the narrowing this file exists to refuse.
    """
    body = re.sub(r"\s\([^)]*\)", "", " ".join(body.split()))
    # Em dash, en dash or `--`: `CONTEXT.md` uses the first, source comments the last, and
    # the second is spelled as an escape because a bare one is indistinguishable from a hyphen.
    clauses = [re.split("\\s+[\u2014\u2013]\\s+|\\s+--\\s+", body)[0]]
    clauses += [m.group("body") for m in _ALSO.finditer(body)]
    out = []
    for clause in clauses:
        for raw in re.split(r",|\band\b", clause):
            term = raw.strip().strip(".").strip("*` ").lower()
            if term:
                out.append(term)
    return out


def glossary(text: str | None = None) -> list[Entry]:
    """Every entry in `CONTEXT.md` that forbids something, in the order they appear."""
    text = CONTEXT.read_text() if text is None else text
    heads = [(m.start(), tuple(g for g in (m.group(1), m.group(2)) if g))
             for m in _HEADWORD.finditer(text)]
    out = []
    for m in _AVOID.finditer(text):
        owner = max((h for h in heads if h[0] < m.start()), default=(0, ("?",)))
        lo = text.count("\n", 0, m.start()) + 1
        out.append(Entry(owner[1], tuple(_terms(m.group("body"))),
                         (lo, lo + m.group(0).count("\n"))))
    return out


def headwords(text: str | None = None) -> set[str]:
    """Every term `CONTEXT.md` defines, lower-cased."""
    text = CONTEXT.read_text() if text is None else text
    return {g.lower() for m in _HEADWORD.finditer(text)
            for g in (m.group(1), m.group(2)) if g}


# --- what can and cannot be decided by a scan ---------------------------------

@dataclass(frozen=True)
class Rule:
    """How one term is checked, and why that rule is allowed to decide it.

    `owners` is for `kind="name"` only: the path prefixes whose modules may use the word in
    a name, because the glossary entry itself says which thing owns it.
    """

    kind: str                                       # "phrase" | "name"
    why: str
    owners: tuple[str, ...] = ()
    pattern: str = ""                               # overrides the term, for a phrase

    def regex(self, term: str) -> re.Pattern[str]:
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(self.pattern or term)
                          + r"(?![A-Za-z0-9_])", re.I)


CHECKED: dict[str, Rule] = {
    # Three entries forbid this one, and the section above them says why in bold: the word
    # has meant all three markets in one paragraph. There is no fourth, legitimate market,
    # so every bare occurrence is the ambiguity -- which is what makes it scannable at all.
    "the market": Rule("phrase", "no unqualified market exists in this repo; the glossary "
                                 "names three and forbids the bare form in all three entries"),
    # "the plan unqualified", read as the bare phrase. A qualified plan -- "the remaining
    # plan", "this plan", "a survivor plan" -- does not match, so the pattern is exactly the
    # thing the entry forbids and nothing else.
    "the plan unqualified": Rule("phrase", "the entry forbids the bare phrase and names the "
                                           "qualified forms that are fine",
                                 pattern="the plan"),
    "the closing line": Rule("phrase", "names the close, which a snapshot rarely is; the "
                                       "phrase has no other referent here"),
    "line source": Rule("phrase", "the superseded spelling of Price source, and nothing "
                                  "else in the repo is called a line source"),
    "adjusted adp": Rule("phrase", "one spelling of Corrected ADP, with no other meaning"),
    "our adp": Rule("phrase", "one spelling of Corrected ADP, with no other meaning"),
    "normalised name": Rule("phrase", "the pre-2026-08-27 spelling of Player key"),
    "expected points": Rule("phrase", "xFP is expected fantasy *points per game*; the bare "
                                      "phrase names no other quantity in this repo"),
    "projected points": Rule("phrase", "same: no quantity here is called projected points"),
    "training data": Rule("phrase", "the Panel is the unit of measurement; nothing here "
                                    "trains on anything else"),
    "pick position": Rule("phrase", "a Slot, spelled the way the entry forbids"),
    "the book": Rule("phrase", "sportsbook prices are the Betting market; no book is read "
                               "anywhere in this repo"),
    "vegas": Rule("phrase", "a proper noun for the Betting market and nothing else"),
    "hunch": Rule("phrase", "the entry's point is that a Provisional rule is pre-registered "
                            "and logged; nothing here is a hunch on purpose"),
    "p(win) as a standalone term": Rule(
        "phrase", "the entry reserves `p_win` for the column and forbids the standalone "
                  "term, which is spelled only this way", pattern="p(win)"),
    # `provenance alone` is the one term here that is a *name* rather than a phrase. The
    # word is fine in prose -- "next to their provenance" is not a claim about Price source
    # -- and is a claim the moment it names something. `hub.schedule.provenance` is the
    # allowed one: it is price-source obtainability, and it is what `docs/track-record.md`
    # requires under the artifact key `provenance`.
    "provenance alone": Rule("name", "the word names a published artifact key; anything "
                                     "else it names is a second meaning for one name",
                             owners=("src/hub/schedule.py",), pattern="provenance"),
}

UNCHECKED: dict[str, str] = {
    "adp alone when the contrast with consensus matters": (
        "conditional on a contrast being drawn, which is a judgement about the paragraph"),
    "rankings": "ECR is a ranking; the entry forbids the plural for Consensus and the repo "
                "legitimately ranks many other things",
    "startable": "`hub.draft` uses the word for a different idea and the entry says so; "
                 "Replacement level's own definition uses it correctly",
    "active": "collides with ESPN's `ACTIVE` designation, which the fetch layer must name",
    "eligible": "flex eligibility and Slot eligibility are real and correctly named; the "
                "entry forbids it only for Can start",
    "the line": "`spread_line`, `close_spread` and every prose line of output are lines; "
                "the entry forbids it only for a Snapshot",
    "roster": "the entry's own words: it is *your* players, which is what the repo means "
              "everywhere it says roster. Only the League-shape sense is forbidden",
    "settings": "same entry, same problem -- Hydra settings are settings",
    "sample": "forbidden for a Cohort; a games sample and a synthetic sample are neither",
    "population": "forbidden for a Cohort; the word is used nowhere else, but deciding "
                  "which sense a use is in needs the sentence",
    "seat": "forbidden for a Slot, and `cohort` legitimately says seat for the human idea",
    "heuristic": "forbidden for a Provisional rule; a solver heuristic is not one",
    "validate": "forbidden for a screen or a gate; `validate_predictions` and every "
                "contract validation are the other sense",
    "backtest": "forbidden for a screen or a gate; `hub.draft.backtest` is a module and "
                "ADR-0002 names the harness",
    "opportunity": "forbidden for Usage; nflverse's `ff_opportunity` is the source's name",
    "volume": "forbidden for Usage; `hub.models.volume` is a measured model with its own "
              "ADR, and the entry contrasts the two rather than banning the word",
    "dataset": "forbidden for the Panel; `hub.inspect` and `hub.store` describe stored "
               "datasets that are not panels",
    "projection unqualified": "the entry asks for weekly projection or xFP where the "
                              "contrast matters, which is a judgement about the sentence",
    "edge": "forbidden for Lift; **Edge** is itself a glossary entry, so most uses are right",
    "stale": "the manifest says stale about all three states, so the word is unavoidable "
             "in the module the entry is about",
    "skipped": "a skipped test, a skipped workflow and a skipped file are all this word",
    "failed": "a failed fetch and a failed gate are both correct uses",
    "_norm": "a private helper that no longer exists; the term is a historical spelling",
    # The **As of** entry arrived from another change while issue #53 was in flight, which is
    # this file's premise check doing its job: three terms turned up and had to be decided.
    "cutoff": "forbidden for an As of date; `difflib.get_close_matches(cutoff=0.85)` in "
              "`hub.draft.state` is a similarity threshold, so an occurrence is not "
              "decidably the date sense. Worth checking once the two in `hub.draft.board` "
              "are settled -- they are the forbidden sense and they are the only two",
    "up to": "ordinary English, several times a page ('up to five rows'); the entry forbids "
             "it only where it bounds an as-of",
    "stating an as-of anywhere but at the fetch boundary": (
        "an architectural rule about where a comparison lives, not a word to grep for"),
}

# Headwords are a different class from `_Avoid_` terms: using **Board** for the draft board
# is *correct*, so a scan cannot flag the word wholesale. These two are checkable anyway,
# and only these two, because their own entries say what owns them -- "the frame every
# draft-night decision reads", and the measurement decision. A name carrying the word
# outside the owning package is a second meaning for a reserved one, which is exactly the
# `board`-for-scoreboard drift issue #53 found. Opt-in and short on purpose: this is not
# derived from the glossary the way `_Avoid_` terms are, so it is written down.
RESERVED: dict[str, Rule] = {
    "board": Rule("name", "the entry reserves it for the frame draft-night decisions read, "
                          "which lives in `hub.draft`; a `board` named anywhere else is a "
                          "second Board (`scoreboard` and `overlay` are both already in use)",
                  owners=("src/hub/draft/",)),
    "gate": Rule("name", "the entry reserves it for the measurement decision -- does this "
                         "beat what it replaces -- which only the modelling packages make",
                 owners=("src/hub/models/", "src/hub/season/", "src/hub/draft/")),
}


# --- the scan -----------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """One occurrence of a term that a rule forbids, and enough to go and look at it."""

    path: str
    line: int
    term: str
    rule: Rule = field(compare=False, repr=False, default=Rule("phrase", ""))
    text: str = field(compare=False, repr=False, default="")

    @property
    def site(self) -> tuple[str, str]:
        """The inventory key: a file and a term, not a line number, which drifts."""
        return (self.path, self.term)


# A mention is not a use. `# CONTEXT.md avoids "the plan" unqualified` cites the rule; the
# glossary's own `_Avoid_` line lists the term. Both are quoted or backticked *and* sit on a
# line that names the rule, which is as narrow as this exemption gets.
_CITES = re.compile(r"\bavoid|CONTEXT\.md|glossary\b", re.I)


def _is_mention(line: str, start: int, end: int) -> bool:
    before, after = line[:start], line[end:]
    quoted = ((before.count('"') % 2 or before.count("'") % 2 or before.count("`") % 2)
              and ('"' in after or "'" in after or "`" in after))
    return bool(quoted and _CITES.search(line))


# `def f`, `class C`, `NAME = ...`, and the guard markers, whose names are declarations too.
_PY_NAME = re.compile(r"^\s*(?:def|class)\s+(?P<n>[A-Za-z_][A-Za-z0-9_]*)"
                      r"|^(?P<c>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?="
                      r"|^\s*#\s*(?:GUARD|UNPROVED)\s+(?P<g>[a-z0-9][a-z0-9-]*)", re.M)
_SH_NAME = re.compile(r"^(?P<c>[A-Za-z_][A-Za-z0-9_]*)="
                      r"|^(?P<n>[A-Za-z_][A-Za-z0-9_]*)\s*\(\)"
                      r"|^\s*#\s*(?:GUARD|UNPROVED)\s+(?P<g>[a-z0-9][a-z0-9-]*)", re.M)


def _words(name: str) -> set[str]:
    return {w.lower() for w in re.split(r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])", name) if w}


def _skip_lines(path: Path) -> set[int]:
    """Lines a scan must not read as uses: the glossary's own `_Avoid_` paragraphs."""
    if path != CONTEXT:
        return set()
    return {n for e in glossary() for n in range(e.span[0], e.span[1] + 1)}


def scan(paths: tuple[Path, ...] = SCAN_ROOTS,
         rules: dict[str, Rule] | None = None) -> list[Hit]:
    """Every occurrence in `paths` that a checked rule forbids."""
    rules = {**CHECKED, **RESERVED} if rules is None else rules
    phrases = {t: r for t, r in rules.items() if r.kind == "phrase"}
    names = {t: r for t, r in rules.items() if r.kind == "name"}
    out: list[Hit] = []
    for path in paths:
        if not path.exists():
            continue
        # Relative for the inventory keys; absolute for a planted file under `tmp_path`,
        # which is how the premise tests point the same scan at something outside the repo.
        rel = (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT)
               else path.as_posix())
        text = path.read_text()
        skip = _skip_lines(path)
        for n, line in enumerate(text.splitlines(), 1):
            if n in skip:
                continue
            for term, rule in phrases.items():
                for m in rule.regex(term).finditer(line):
                    if not _is_mention(line, m.start(), m.end()):
                        out.append(Hit(rel, n, term, rule, line.strip()))
        if not names or path.suffix not in (".py", ".sh"):
            continue
        marks = _PY_NAME if path.suffix == ".py" else _SH_NAME
        for m in marks.finditer(text):
            declared = m.group("n") or m.group("c") or m.group("g") or ""
            words = _words(declared)
            for term, rule in names.items():
                word = rule.pattern or term
                if word in words and not rel.startswith(rule.owners):
                    out.append(Hit(rel, text.count("\n", 0, m.start()) + 1, term, rule,
                                   declared))
    return out


# --- what is outstanding, and why ---------------------------------------------
#
# Issue #53 fixed the sites it named and left the rest standing rather than rewriting sixty
# comments in one change. Every one of them is here, with the reason, and the ratchet below
# means the count can fall and never rise. Three of them are the second pass's, held back
# because three other changes were rewriting those files at the same time.

_BULK = "pre-existing prose, outside issue #53's scope; the word means the draft market here"
_BETTING = "pre-existing prose; the word means the betting market here"
_XFP = "pre-existing prose for xFP, which is expected fantasy points per game"

OUTSTANDING: dict[tuple[str, str], tuple[int, str]] = {
    # --- deferred to issue #53's second pass, because another change owns the file today --
    ("src/hub/config.py", "provenance alone"): (
        1, "DEFERRED: `config.provenance` returns a set of digests, which is not price "
           "source. `hub.schedule.provenance` already owns the word and the publisher "
           "imports it. The digest set should be named for what it is"),
    ("src/hub/fetch/espn.py", "board"): (
        1, "DEFERRED: the `board-resolved-nothing` guard names the ESPN scoreboard `board`, "
           "which is the reserved Board. `scoreboard` and `overlay` are both already used "
           "in this module; its tests carry three more of these"),
    # --- correct uses of a reserved word, in a module outside the owning package -----------
    ("src/hub/paths.py", "board"): (2, "the real draft board's paths, named outside "
                                       "`hub.draft` because paths are a leaf module"),
    ("src/hub/publish.py", "board"): (1, "`_board` reports whether the draft board's own "
                                         "file is there; the Board is what it means"),
    ("src/hub/contracts.py", "board"): (1, "the draft board's contract, named where every "
                                           "other contract is"),
    ("src/hub/contracts.py", "provenance alone"): (
        1, "`_PROVENANCE_NOTES` is the fetch layer's source provenance, which is a third "
           "thing again. Second pass, with the config module"),
    ("src/hub/models/ratings.py", "provenance alone"): (
        1, "`PROVENANCE_COLUMNS` is the same defect as the config one and predates it: it "
           "is a digest set, not a price source. Second pass, with the config module"),
    # --- "the market", the term three entries forbid --------------------------------------
    ("src/hub/draft/backtest.py", "the market"): (8, _BULK),
    ("src/hub/draft/board.py", "the market"): (7, _BULK),
    ("src/hub/draft/optimize.py", "the market"): (7, _BULK),
    ("src/hub/draft/report.py", "the market"): (5, _BULK),
    ("src/hub/models/volume.py", "the market"): (5, _BULK),
    ("src/hub/models/weekly.py", "the market"): (5, _BETTING),
    ("src/hub/draft/durability.py", "the market"): (4, _BULK),
    ("src/hub/draft/calibrate.py", "the market"): (3, _BULK),
    ("src/hub/models/predict.py", "the market"): (3, _BULK),
    ("src/hub/draft/adp_history.py", "the market"): (2, _BULK),
    ("src/hub/models/market.py", "the market"): (2, _BETTING),
    ("src/hub/models/ratings.py", "the market"): (2, _BETTING),
    ("src/hub/schedule.py", "the market"): (2, _BETTING),
    ("src/hub/season/weekly_gate_data.py", "the market"): (2, _BULK),
    ("src/hub/draft/cohort.py", "the market"): (1, _BULK),
    ("src/hub/draft/regression.py", "the market"): (1, _BULK),
    ("src/hub/draft/tune.py", "the market"): (1, _BULK),
    ("src/hub/models/base.py", "the market"): (1, _BETTING),
    ("src/hub/models/eval.py", "the market"): (1, _BETTING),
    ("src/hub/models/margin.py", "the market"): (1, _BETTING),
    ("src/hub/models/panel.py", "the market"): (1, _BETTING),
    ("src/hub/season/lineup_gate.py", "the market"): (1, _BULK),
    # --- "the plan" unqualified -----------------------------------------------------------
    ("src/hub/draft/live.py", "the plan unqualified"): (2, "the foundation plan document"),
    ("site/index.html", "the plan unqualified"): (2, "the survivor panel's own copy"),
    ("src/hub/models/eval.py", "the plan unqualified"): (1, "the foundation plan document"),
    ("src/hub/models/ratings.py", "the plan unqualified"): (1, "the foundation plan doc"),
    ("src/hub/models/weekly.py", "the plan unqualified"): (1, "the weekly projection plan"),
    ("src/hub/schedule.py", "the plan unqualified"): (1, "a survivor plan is meant"),
    # --- the rest -------------------------------------------------------------------------
    ("src/hub/draft/board.py", "expected points"): (4, _XFP),
    ("src/hub/draft/projection.py", "expected points"): (2, _XFP),
    ("src/hub/fetch/nflverse.py", "expected points"): (2, _XFP),
    ("src/hub/draft/regression.py", "expected points"): (1, _XFP),
    ("src/hub/draft/season.py", "expected points"): (1, _XFP),
    ("src/hub/models/predict.py", "expected points"): (1, _XFP),
    ("src/hub/season/lineup.py", "expected points"): (1, _XFP),
    ("src/hub/season/lineup_gate.py", "expected points"): (1, _XFP),
    ("src/hub/draft/board.py", "p(win) as a standalone term"): (
        2, "championship equity, discussed by the standalone name the entry reserves for "
           "the `p_win` column alone"),
    ("src/hub/draft/optimize.py", "p(win) as a standalone term"): (1, "the same"),
    ("src/hub/draft/regression.py", "p(win) as a standalone term"): (1, "the same"),
    ("src/hub/draft/season.py", "p(win) as a standalone term"): (1, "the same"),
    ("src/hub/draft/availability.py", "pick position"): (1, "a Slot, spelled as the entry "
                                                            "forbids"),
    ("src/hub/models/experiment.py", "pick position"): (1, "the same spelling"),
    ("src/hub/models/experiment.py", "training data"): (1, "the Panel, named as the entry "
                                                           "forbids"),
    ("src/hub/draft/state.py", "normalised name"): (1, "the pre-2026-08-27 spelling of "
                                                       "Player key, in the module it left"),
    ("src/hub/models/panel.py", "normalised name"): (1, "the same spelling"),
    ("src/hub/draft/optimize.py", "projected points"): (1, "xFP, spelled as forbidden"),
    ("src/hub/fetch/espn.py", "projected points"): (1, "ESPN's own projection field"),
    ("src/hub/models/market.py", "the closing line"): (1, "a snapshot, described as the "
                                                          "close it rarely is"),
    ("scripts/bootstrap_project.sh", "the closing line"): (1, "the same"),
}

# `tests/unit/test_preflight.py` names the secret scan a "gate" thirty times, including in
# test names. It is outside SCAN_ROOTS (tests are not the source tree) and outside issue
# #53's scope, and the script it tests no longer calls itself one. Recorded here in prose so
# the second pass has it written down rather than rediscovered.


def _outstanding_counts(hits: list[Hit]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for h in hits:
        out[h.site] = out.get(h.site, 0) + 1
    return out


# --- the premise: a scan that decides nothing must not pass --------------------
#
# Everything below the habit is vacuously true if the parse comes back empty, the term
# tables drift out of step with the glossary, or the scan stops matching. Issue #53's own
# `tests/contracts/test_guards_are_load_bearing.py` exists because eight guards passed while
# incapable of firing, so the checks in this section come first.

def test_the_glossary_parse_finds_the_entries_that_exist():
    entries = glossary()
    assert len(entries) >= 20, f"only parsed {[repr(e) for e in entries]}"
    forbidding_the_market = [e for e in entries if "the market" in e.avoided]
    assert len(forbidding_the_market) == 3, (
        f"three entries forbid 'the market' -- Draft market, Consensus and Betting market. "
        f"Parsed {[repr(e) for e in forbidding_the_market]}")
    assert "board" in headwords(), "the glossary defines Board; the headword parse has drifted"


def test_the_parse_keeps_the_terms_that_hide_after_the_em_dash():
    """`Can start` lists "startable" and then, after the reason, "Also avoid active ... and
    eligible". Cutting at the dash loses two terms -- one of them the `eligible` that this
    class of defect has already been found under."""
    can_start = next(e for e in glossary() if "can start" in [h.lower() for h in e.headwords])
    assert set(can_start.avoided) == {"startable", "active", "eligible"}, can_start.avoided


def test_the_parse_is_not_broken_by_a_comma_or_a_dash_inside_an_aside():
    got = _terms(" roster (which is *your* players, not the league's shape), settings.")
    assert got == ["roster", "settings"], got
    # `P(win)` keeps its parentheses: only an aside preceded by whitespace is an aside.
    assert _terms(" P(win) as a standalone term — say championship equity, and reserve "
                  "`p_win` for the column.") == ["p(win) as a standalone term"]


def test_every_avoided_term_in_the_glossary_is_classified():
    """The check that stops this file narrowing to nothing.

    A term parsed out of `CONTEXT.md` is either checked -- with the rule that decides it --
    or explicitly not checkable, with the reason. Adding an `_Avoid_` term to the glossary
    therefore fails this until someone has decided which it is.
    """
    parsed = {t for e in glossary() for t in e.avoided}
    classified = set(CHECKED) | set(UNCHECKED)
    assert not (parsed - classified), (
        f"unclassified `_Avoid_` terms: {sorted(parsed - classified)}. Add each to CHECKED "
        f"with the rule that decides it, or to UNCHECKED with why no regex can.")
    assert not (classified - parsed), (
        f"classified terms the glossary no longer forbids: {sorted(classified - parsed)}. "
        f"Strike them, or the tables describe a glossary that does not exist.")
    assert not (set(CHECKED) & set(UNCHECKED)), sorted(set(CHECKED) & set(UNCHECKED))


def test_every_reserved_term_is_a_headword_the_glossary_defines():
    defined = headwords()
    assert not (set(RESERVED) - defined), (
        f"RESERVED names terms `CONTEXT.md` does not define: "
        f"{sorted(set(RESERVED) - defined)}")


def test_the_scan_finds_the_terms_it_is_supposed_to():
    """The anti-vacuity check. A scan matching nothing passes every assertion below it."""
    hits = scan()
    assert hits, "the scan found nothing at all, which is not what this repo looks like"
    found = {h.term for h in hits}
    assert "the market" in found, (
        "the term three glossary entries forbid matched nowhere, so the phrase scan is dead")
    assert "board" in found, "the reserved-headword scan matched nothing"
    assert "provenance alone" in found, "the name scan matched nothing"
    assert len(hits) >= 50, f"only {len(hits)} hits; the scan has narrowed"


@pytest.mark.parametrize("path,term", [
    ("src/hub/config.py", "provenance alone"),
    ("src/hub/fetch/espn.py", "board"),
])
def test_the_scan_reports_the_sites_held_back_for_the_second_pass(path, term):
    """Pointed at the two files issue #53 deferred, this must report them.

    They are in OUTSTANDING and so do not fail the habit below -- but a scan that could not
    see them would be useless to the change that finally fixes them, which is the whole
    point of building the guard in the same ticket.
    """
    got = scan((ROOT / path,))
    assert any(h.term == term for h in got), (
        f"{path} violates `{term}` and the scan does not see it: {got}")


def test_a_fresh_violation_is_caught(tmp_path):
    """Plant one, and require the scan to name it, the term, and the line."""
    mod = tmp_path / "fresh.py"
    mod.write_text("def f():\n"
                   "    # The pick follows the market, whichever one that is.\n"
                   "    return 1\n")
    got = scan((mod,))
    assert [(h.term, h.line) for h in got] == [("the market", 2)], got


def test_a_citation_of_the_rule_is_not_a_violation(tmp_path):
    """The narrow exemption: a quoted mention on a line that names the rule.

    `CONTEXT.md`'s own `_Avoid_` lines and the comments citing them are talking *about* the
    term. Nothing wider than that is exempt -- the same words unquoted still fail."""
    mod = tmp_path / "cite.py"
    mod.write_text('# CONTEXT.md avoids "the market" unqualified, so this says which one.\n'
                   "# This one follows the market.\n")
    assert [h.line for h in scan((mod,))] == [2]


def test_the_outstanding_inventory_is_current():
    """A listed site that has gone to zero is a stale entry, and stale entries are how an
    inventory stops describing the repo."""
    counts = _outstanding_counts(scan())
    gone = [site for site in OUTSTANDING if site not in counts]
    assert not gone, (
        f"OUTSTANDING lists sites that no longer violate anything: {gone}. Delete them -- "
        f"an inventory nobody prunes is a list of exemptions nobody reads.")


# --- the habit ----------------------------------------------------------------

def test_no_source_file_or_glossary_entry_uses_a_term_its_glossary_forbids():
    """The whole point. Anything not in OUTSTANDING, or more of it than there was, is red."""
    counts = _outstanding_counts(scan())
    by_term = {t: e for e in glossary() for t in e.avoided}
    bad = []
    for hit in scan():
        allowed, _ = OUTSTANDING.get(hit.site, (0, ""))
        if counts[hit.site] > allowed:
            entry = by_term.get(hit.term)
            bad.append(f"  {hit.path}:{hit.line}  `{hit.term}` -- "
                       f"{repr(entry) if entry else 'reserved by **Board**/**Gate**'} "
                       f"forbids it ({hit.rule.why}).\n      {hit.text[:96]}")
    assert not bad, (
        "these use a term CONTEXT.md forbids:\n" + "\n".join(sorted(set(bad))) + "\n\n"
        "Say which market, which plan, which board. If the use is genuinely the entry's own "
        "sense, add it to OUTSTANDING with the reason -- and if the term is not decidable by "
        "a regex at all, move it to UNCHECKED and say why.")
