"""`CONTEXT.md`'s `_Avoid_` lists, enforced against the source tree, against `CONTEXT.md`, and
against the names `tests/` declares.

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

**The half a ratchet cannot reach is the prose around it.** Entries are policed; the sentences
introducing them are not, and they have now outlived their subject twice. `c8530e5` fixed two
that described violations already gone, and recorded the reason it had to be done by hand:
"prose is the half its own inventory test cannot check". The second was the inventory's own
preamble, which said "sixty comments" and "three of them are the second pass's" until
2026-09-06 -- by then the dict held 104 sites and not one deferred entry. Both figures were
true when written and neither was ever re-derived. So the preamble carries no number now, and
a test holds it to that: the only honest count is the one the dict computes.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "CONTEXT.md"

# The source tree and the glossary, read line by line.
SCAN_ROOTS: tuple[Path, ...] = (
    *sorted((ROOT / "src" / "hub").rglob("*.py")),
    *sorted((ROOT / "scripts").glob("*.sh")),
    ROOT / "site" / "index.html",
    CONTEXT,
)

# And `tests/`, read for the names it *declares* -- each file's own name, and every `def` and
# `class` in it -- and for nothing else. `docs/agents/domain.md` puts a test name in scope by
# name, and a test name is where someone learning what a thing is called meets the word first.
#
# Names and not prose, and that split is the decision rather than an implementation detail.
# This root used to be excluded outright, reasoning that `tests/` restates the vocabulary
# constantly while describing what it is testing, so folding it in would bury the source's own
# violations. That was argued and never measured, so both spans were measured on 2026-09-06,
# against the tree this change started from: all of `tests/`, prose included, was 297 hits --
# the burial the exclusion described, and still the right call. The declared names alone were
# 20, every one of them a real violation of an entry, and all 20 are renamed here. So the
# exclusion was right about prose and wrong about names, which is the half it never priced.
#
# Both figures are history, dated and left that way. The prose span already reads 306, because
# the paragraph you are reading restates the vocabulary exactly the way `tests/` always does --
# the exclusion's own argument, arriving on schedule.
#
# The evidence for widening was always about names, too. `cutoff` arrived in a *test name* in
# the same change that added the **As of** entry forbidding it, and the scan of the day could
# not see it; `test_the_name_that_prompted_this_widening_would_now_be_caught` holds that case
# by writing it out, since the name itself is gone.
NAME_ROOTS: tuple[Path, ...] = tuple(sorted((ROOT / "tests").rglob("*.py")))


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
    deny_trailing: str = ""                         # characters that also end a non-match

    def regex(self, term: str) -> re.Pattern[str]:
        """The term as a whole word, minus any trailing character `deny_trailing` rules out.

        Each character in `deny_trailing` is added to the word-boundary class, so it is a
        *set of characters* and never a sequence -- two characters deny two forms, not
        one two-character form. It exists so a term can be checked where one syntactic
        form of it is
        decidably a different thing -- `cutoff=` is a keyword argument and never an as-of
        date. Without it such a term goes to UNCHECKED whole, which is worse: UNCHECKED has
        no ratchet, so its known violations are not counted and a new one lands unseen. That
        is not hypothetical -- it is how `cutoff` gained a fresh violation in the same change
        that added the entry forbidding it.
        """
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(self.pattern or term)
                          + r"(?![A-Za-z0-9_" + re.escape(self.deny_trailing) + r"])", re.I)


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
    # The Ledger and the Decision journal were one word across five survivor tickets until
    # #80 was specified. The Ledger owns it by possession: `hub.season.pool` takes the
    # used-team set as `ledger`/`ledgers` and `buyback_restores_ledger` is a pool-config
    # field, so renaming that side would move `pool_digest` to settle a prose collision. A
    # bare `ledger` is therefore correct almost everywhere and unscannable; the two-word
    # phrase only ever names the journal, which is what makes this one decidable.
    "decision ledger": Rule("phrase", "the Ledger is the used-team set and shipped code "
                                      "names it that; the two-word phrase is always the "
                                      "journal wearing the taken word"),
    # Checked rather than parked. The one form that is decidably not an as-of date is the
    # keyword argument -- `difflib.get_close_matches(cutoff=0.85)` is a similarity threshold
    # -- and `_adp_saturation_cutoff` is already excluded by the word boundary. Everything
    # else the word can be here is the date sense the **As of** entry forbids.
    "cutoff": Rule("phrase", "the As of entry forbids it for a date; `cutoff=` is a keyword "
                             "argument and never a date, and a leading underscore makes it "
                             "part of another name", deny_trailing="="),
    # Checkable at zero. The helper it named is gone, so the whole-word form occurs nowhere
    # in the scanned roots and needs no inventory entry -- which is precisely what makes it
    # CHECKED rather than UNCHECKED. Its sibling on the same `_Avoid_` line is already
    # checked; parking this one said "no regex can decide it" about a term with no
    # occurrences, which is the shape this file caught for `cutoff` one commit ago.
    "_norm": Rule("phrase", "the helper it spelled is gone; the whole-word form occurs "
                            "nowhere, so any reappearance is the historical spelling"),
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
    # The Quarterback adjustment's two neighbours (#218). Both are proper names for other
    # quantities -- the source's Elo-point figure and the passer-rating statistic -- and
    # neither names anything this repo computes, so a bare use is the confusion the entry
    # names. `nfeloqb` and `qb_elos.csv` are not hits: the word boundary the rule requires
    # is absent inside a module or file name.
    "qb elo": Rule("phrase", "the source's quantity, in Elo points; the adjustment here is "
                             "in spread points and relative to the team"),
    "qb rating": Rule("phrase", "a passer rating is a different statistic, and nothing "
                                "here computes one"),
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
    "population": "forbidden for a Cohort. Three uses in the draft modules are the "
                  "statistical sense -- a population average, one population against "
                  "another, pick numbers over one population -- and which sense a use "
                  "is in needs the sentence, which no regex reads",
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
    # The **As of** entry arrived from another change while issue #53 was in flight, which is
    # this file's premise check doing its job: three terms turned up and had to be decided.
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
#
# The indent is `[ \t]*`, not `\s*`, and the difference is which line the hit names. `\s`
# matches a newline, so `^\s*def` anchored on the blank line above a top-level `def` matches
# from *there* and the hit is reported two lines early -- reproduced as line 2 for a `def` on
# line 4. The message is the whole product of this scan: a reader who follows it to a blank
# line, finds nothing, and stops trusting the tool is the failure this file exists to prevent,
# arriving through the tool itself. `_DECLARED` below was written with the correct class and
# the two halves disagreeing about one file is how this surfaced (issue #129).
_PY_NAME = re.compile(r"^[ \t]*(?:def|class)\s+(?P<n>[A-Za-z_][A-Za-z0-9_]*)"
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


# Only `def` and `class`. The module-level assignments and guard markers `_PY_NAME` also
# collects are read by the name-kind rules above, which do not run over `tests/`.
#
# The indent is `[ \t]*` and not `\s*`, which is the difference between naming the line and
# naming a blank one: `\s` matches a newline, so `^\s*def` anchored at the blank line above a
# top-level `def` matches from *there*, and the hit is reported two lines early. A message
# that sends a reader to a blank line is how a scan gets ignored.
_DECLARED = re.compile(r"^[ \t]*(?:def|class)\s+(?P<n>[A-Za-z_][A-Za-z0-9_]*)", re.M)


def declared_names(path: Path) -> list[tuple[int, str]]:
    """The file's own name, then every `def`/`class` in it, with the underscores taken out.

    Reading them back as prose is not a convenience, it is the whole mechanism. `cutoff` in
    `test_empty_series_has_no_cutoff` sits behind an underscore, and the word boundary that
    deliberately spares `_adp_saturation_cutoff` over in `src/` would spare this too -- so a
    scan widened to `tests/` but still reading identifiers as identifiers would have gone on
    missing the exact name that prompted widening it. Split on the underscores and the same
    string says "has no cutoff", which is a claim about a date and decidable as one.

    The file's own name is reported against line 1, because a reader told about a file name
    is being sent to the file rather than to a line in it.
    """
    text = path.read_text()
    out = [(1, path.stem.replace("_", " "))]
    out += [(text.count("\n", 0, m.start()) + 1, m.group("n").replace("_", " "))
            for m in _DECLARED.finditer(text)]
    return out


def scan_names(paths: tuple[Path, ...] = NAME_ROOTS,
               rules: dict[str, Rule] | None = None) -> list[Hit]:
    """Every declared name in `paths` that a checked *phrase* rule forbids.

    Phrase rules only, and that is a limit on what the widening claims rather than a shortcut.
    A name read as prose is prose, so the phrase rules are exactly the ones that can decide
    it. The `kind="name"` rules cannot come along: they say who owns a reserved word as a
    `src/` path prefix, and no file under `tests/` can match one, so `board` and `gate` would
    fire on all 138 declarations that exist to test the Board and the Gate (measured
    2026-09-06) -- correct uses every one, with no way for the rule to say so. Ownership is
    expressed in source paths, so reserved words stay policed where their owners live.

    There is no `_is_mention` escape either. A name cannot quote a term or sit on a line that
    cites the rule, so a test that genuinely needs to name a forbidden word says so in its
    docstring, which this half does not read.
    """
    rules = {**CHECKED, **RESERVED} if rules is None else rules
    phrases = {t: r for t, r in rules.items() if r.kind == "phrase"}
    out: list[Hit] = []
    for path in paths:
        if not path.exists():
            continue
        rel = (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT)
               else path.as_posix())
        for line, name in declared_names(path):
            for term, rule in phrases.items():
                for _ in rule.regex(term).finditer(name):
                    out.append(Hit(rel, line, term, rule, name))
    return out


def _all_hits() -> list[Hit]:
    """Both halves: the scanned files read whole, and `tests/` read for its names."""
    return scan() + scan_names()


# --- what is outstanding, and why ---------------------------------------------
#
# Issue #53 fixed the sites it named and left the rest standing rather than rewriting them
# all in a single change. All of them are here, each with its own reason, and the ratchet
# below means the count can fall and never rise.
#
# This paragraph deliberately states no count, and
# `test_the_inventorys_preamble_states_no_count_of_its_own` keeps it that way. The dict below
# is the count; a number restated up here is a second copy that nothing re-derives, and the
# module docstring records what that cost twice.

_BULK = "pre-existing prose, outside issue #53's scope; the word means the draft market here"
_BETTING = "pre-existing prose; the word means the betting market here"
_XFP = "pre-existing prose for xFP, which is expected fantasy points per game"

OUTSTANDING: dict[tuple[str, str], tuple[int, str]] = {
    # The two sites this file was written to hold are fixed: `config.provenance` is
    # `config.digests`, and the ESPN reader's guard is `scoreboard-resolved-nothing`. They
    # are out of the inventory rather than retained with a note, because an entry that
    # outlives its violation is the decay this whole file is about.
    # --- correct uses of a reserved word, in a module outside the owning package -----------
    ("src/hub/paths.py", "board"): (2, "the real draft board's paths, named outside "
                                       "`hub.draft` because paths are a leaf module"),
    ("src/hub/publish.py", "board"): (1, "`_board` reports whether the draft board's own "
                                         "file is there; the Board is what it means"),
    ("src/hub/contracts.py", "board"): (1, "the draft board's contract, named where every "
                                           "other contract is"),
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
    # `weekly.py` was here until #248: the sentence lived in `walk_forward`'s docstring and
    # went with the fitted arm it described.
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
    ("src/hub/exhibits/championship_equity.py", "p(win) as a standalone term"): (
        1, "the same, in the module #198 moved the sentence to"),
    # `regression.py` was here for the same reason until #48. The sentence carrying it lived
    # in `correct_projection`, and that function went with the correction #186 withdrew.
    ("src/hub/draft/season.py", "p(win) as a standalone term"): (1, "the same"),
    ("src/hub/models/experiment.py", "pick position"): (1, "the same spelling"),
    ("src/hub/models/experiment.py", "training data"): (1, "the Panel, named as the entry "
                                                           "forbids"),
    ("src/hub/draft/state.py", "normalised name"): (1, "the pre-2026-08-27 spelling of "
                                                       "Player key, in the module it left"),
    ("src/hub/models/panel.py", "normalised name"): (1, "the same spelling"),
    ("src/hub/exhibits/championship_equity.py", "projected points"): (
        1, "xFP, spelled as forbidden, in the module #198 moved the sentence to"),
    ("src/hub/fetch/espn.py", "projected points"): (1, "ESPN's own projection field"),
    ("src/hub/models/market.py", "the closing line"): (1, "a snapshot, described as the "
                                                          "close it rarely is"),
    ("scripts/bootstrap_project.sh", "the closing line"): (1, "the same"),
}

# `tests/unit/test_preflight.py` named the secret scan a "gate" throughout, including in
# its test names. Renamed in the second pass; the count is zero now, and this note is kept
# only because a sentence describing a violation outlived the violation once already here.
# Widening the scan into `tests/` does not pick that case up and never could: `gate` is a
# reserved *headword*, decided by ownership against `src/` prefixes, and `scan_names` says
# why ownership cannot be expressed for a test file. A reviewer caught it, as with `cutoff`.


# The inventory's own preamble: the comment block between the section rule and the first
# constant under it. Delimited rather than searched for by wording, so rephrasing the
# paragraph cannot slide it out of the test's view.
_PREAMBLE = re.compile(r"^# --- what is outstanding.*?$(.*?)^\s*$", re.M | re.S)

# Digits, and the number words a comment actually reaches for. `#53` is an identifier and is
# stripped before this runs -- an issue number names a ticket, it does not count anything.
_ISSUE_REF = re.compile(r"#\d+")
_A_COUNT = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|dozen)\b", re.I)


def test_the_harness_can_find_the_preamble_it_guards():
    """A renamed section rule would make the assertion below vacuously green."""
    assert _PREAMBLE.search(Path(__file__).read_text()), (
        "the OUTSTANDING preamble could not be located, so the next test guards nothing")


def test_the_inventorys_preamble_states_no_count_of_its_own():
    """The half a ratchet cannot reach, held by the one rule that is mechanical.

    `OUTSTANDING` is policed entry by entry. The paragraph introducing it is not, and it has
    outlived its subject twice -- most recently claiming "sixty comments" and "three of them
    are the second pass's" against a dict holding 104 sites and no deferred entry at all.
    Both were true when written. Neither was re-derived, because prose has nowhere to
    re-derive from.

    A count *here* is a second copy of something the dict below already holds, so the rule is
    that there is no count here at all. That is checkable where "keep it accurate" is not.
    The figures themselves are not lost: they sit in the module docstring, tied to the dates
    they were true on, which is what makes them history rather than a claim.

    **The rule is stricter than the defect, on purpose.** It has no way to tell a count from
    an idiom, and it caught "every *one* of them" in the very paragraph written to satisfy
    it. Loosening it to allow the idiomatic senses would put it back in the class this file's
    own docstring calls undecidable, where a regex cannot tell a use from the use it is
    contrasted with. Costing the preamble a few turns of phrase buys a rule that cannot be
    argued with, and the preamble is four sentences long.
    """
    body = _PREAMBLE.search(Path(__file__).read_text())
    assert body is not None
    prose = _ISSUE_REF.sub("", body.group(1))
    found = _A_COUNT.findall(prose)
    assert not found, (
        f"the OUTSTANDING preamble states {found!r}. The dict below is the count -- a number "
        "restated up here is a second copy that nothing re-derives, which is how this "
        "paragraph went stale twice. Put the figure in the module docstring against the date "
        "it was true, or leave it to the inventory.")


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
    assert len(hits) >= 50, f"only {len(hits)} hits; the scan has narrowed"


def test_the_name_scan_can_match_at_all(tmp_path):
    """The name-kind half of the anti-vacuity check, proved without a live violation.

    This used to assert that `provenance alone` was among the repo's hits, using a real
    violation as evidence that the name-kind scan was alive. That worked until the violations
    were fixed -- which was the goal -- and then the premise check failed for the one reason
    it should never fail: the repo got cleaner. A check that needs the codebase to stay dirty
    to prove itself is a check that argues against its own ticket.

    So the machinery is proved against a planted case instead, the way a fresh phrase
    violation already is. The repo-wide assertions above keep covering the kinds that still
    have real hits.
    """
    mod = tmp_path / "named.py"
    mod.write_text("PROVENANCE_COLUMNS = ('model', 'version')\n")
    got = scan((mod,))
    assert [(h.term, h.line) for h in got] == [("provenance alone", 1)], (
        f"the name-kind scan did not match a planted violation: {got}")


def test_the_name_scan_reads_the_test_roots_it_claims_to():
    """The widened half's anti-vacuity check, and it has to be written round the back.

    Every other root proves itself by what it finds. This one cannot: the change that widened
    the scan into `tests/` also renamed all 20 names it found, so there is nothing left for it
    to point at -- and `test_the_name_scan_can_match_at_all` already learned what happens to a
    premise check that needs the repo to stay dirty. So this asserts the machinery instead:
    the roots are populated, and a name comes back out of them as the prose the rules read.
    A glob that stops matching, or an extractor that stops splitting, fails here.
    """
    assert len(NAME_ROOTS) >= 30, (
        f"only {len(NAME_ROOTS)} files under `tests/`; the glob has drifted and the widened "
        f"half is scanning almost nothing")
    got = declared_names(Path(__file__))
    assert (1, "test avoided terms") in got, (
        f"this file's own name did not come back out of the extractor: {got[:3]}")
    assert any(n == "test the name scan reads the test roots it claims to" for _, n in got), (
        "this test's own name did not come back out of the extractor, so the `def` scan is "
        "dead and every name-kind assertion below it is vacuous")


def test_the_habit_test_reads_both_halves():
    """The one narrowing the checks above cannot see, closed by reading the call instead.

    Every name the widened half found is renamed, so `scan()` and `_all_hits()` return the
    same list today -- which means a change putting the habit test back on `scan()` alone is
    green, silently, until the next forbidden test name lands unreported. That is the
    narrowing this module's docstring calls the exact defect issue #53 was filed about, and no
    assertion about *results* can catch it while the results agree. So this one is about the
    call, in the same spirit as `test_the_harness_can_find_the_preamble_it_guards`.
    """
    src = inspect.getsource(test_nothing_scanned_uses_a_term_its_glossary_forbids)
    assert "_all_hits()" in src, (
        "the habit test no longer reads the names in `tests/`. Putting it back on one half of "
        "the scan is green today only because every name the other half found is already "
        "renamed, and stays green over the next one that is not.")


def test_a_forbidden_term_in_a_test_name_is_caught(tmp_path):
    """Plant one and require the scan to name the term and the line -- and only the name.

    The second line is the same violation in prose, and it is *supposed* to go unreported:
    test prose is the span this widening deliberately left out, priced at NAME_ROOTS, and a
    scan that quietly read it anyway would be a different decision than the one written there.
    """
    mod = tmp_path / "test_planted.py"
    mod.write_text("def test_a_pick_that_beats_the_market():\n"
                   "    # whichever way the market moves, this line is prose and unread\n"
                   "    assert True\n")
    got = scan_names((mod,))
    assert [(h.term, h.line, h.text) for h in got] == [
        ("the market", 1, "test a pick that beats the market")], got


def test_the_name_that_prompted_this_widening_would_now_be_caught(tmp_path):
    """The case this ticket was filed over, written out rather than described.

    `test_empty_series_has_no_cutoff` landed in `tests/unit/test_board_edge.py` in the same
    change that added the **As of** entry whose `_Avoid_` forbids the word, and the scan of
    the day could not see it twice over: `tests/` was outside the roots at all, and the
    underscore in front of the word would have spared it even inside them. A reviewer reading
    the diff caught it, which is the run of luck this file exists to replace.

    It is spelled out here because the name itself is gone -- it is
    `test_empty_series_has_no_saturation_point` now, which is what the helper it calls has
    always been about -- so this line is the only thing still holding the historical form.
    """
    mod = tmp_path / "test_board_edge.py"
    mod.write_text("def test_empty_series_has_no_cutoff():\n"
                   "    assert _adp_saturation_cutoff(EMPTY, teams=12) is None\n")
    got = scan_names((mod,))
    assert [(h.term, h.line) for h in got] == [("cutoff", 1)], (
        f"the name that prompted this widening is not caught: {got}")
    # And the file half still spares both, which is the complement that makes the two
    # necessary: `_adp_saturation_cutoff` is a threshold and not a date, and the identifier
    # in the `def` line is unreadable as prose until the underscores come out.
    assert not [h for h in scan((mod,)) if h.term == "cutoff"], (
        "the file half now flags the helper it was written to spare")


@pytest.mark.parametrize("path,term", [
    ("src/hub/config.py", "provenance alone"),
    ("src/hub/fetch/espn.py", "board"),
])
def test_the_sites_held_back_for_the_second_pass_are_fixed_and_stay_fixed(path, term):
    """The two files #53 deferred, now clean, and required to stay clean.

    This test was the inverse until the second pass landed: it pointed the scan at these two
    and demanded it *report* them, so the change that finally fixed them could not be made
    with a guard too narrow to see them. Both are fixed -- `config.provenance` is
    `config.digests`, and the ESPN reader's guard is `scoreboard-resolved-nothing` -- so it
    now asserts the other direction. Kept rather than deleted because these are the two sites
    the codebase has already drifted on once, and a term reintroduced here is the likeliest
    regression in the file.

    Note `src/hub/fetch/espn.py` still uses `board` correctly, for the draft Board it feeds
    with ADP and projections. Those are not violations and the scan does not count them;
    what was wrong was naming the ESPN scoreboard with the reserved word.
    """
    got = scan((ROOT / path,))
    assert not any(h.term == term for h in got), (
        f"{path} has regained a `{term}` violation: "
        f"{[(h.term, h.line) for h in got if h.term == term]}")


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
    counts = _outstanding_counts(_all_hits())
    gone = [site for site in OUTSTANDING if site not in counts]
    assert not gone, (
        f"OUTSTANDING lists sites that no longer violate anything: {gone}. Delete them -- "
        f"an inventory nobody prunes is a list of exemptions nobody reads.")


# --- the habit ----------------------------------------------------------------

def test_nothing_scanned_uses_a_term_its_glossary_forbids():
    """The whole point. Anything not in OUTSTANDING, or more of it than there was, is red.

    Both halves land in the same list, which is what makes the widening worth anything: a
    forbidden word in a test name is reported beside one in a module, in the same message,
    against the same inventory.
    """
    hits = _all_hits()
    counts = _outstanding_counts(hits)
    by_term = {t: e for e in glossary() for t in e.avoided}
    bad = []
    for hit in hits:
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
        "a regex at all, move it to UNCHECKED and say why. A name under `tests/` is a name "
        "you own outright, so rename it: an exemption there is a test that teaches the wrong "
        "word to the next reader, which is the whole reason names are scanned.")


# --- a hit names the line it is on (issue #129) -----------------------------

def _planted(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "planted.py"
    p.write_text(body)
    return p


def test_a_name_hit_reports_the_line_the_declaration_is_on(tmp_path):
    """`\\s` matches a newline, so `^\\s*def` anchored on the blank line above a top-level
    `def` matched from there and reported two lines early.

    The blank line is the fixture, not decoration: without it the anchor and the declaration
    are on the same line and the defect does not reproduce at all. That is why this went
    unnoticed -- every hit in a file whose declarations are not preceded by a blank line was
    reported correctly.
    """
    rule = Rule("name", "a planted rule", owners=())
    body = "import os\n\n\ndef the_market_thing():\n    pass\n"      # `def` is line 4
    got = scan((_planted(tmp_path, body),), {"market": rule})
    assert len(got) == 1, got
    assert got[0].line == 4, (
        f"reported line {got[0].line} for a declaration on line 4; a message that sends a "
        f"reader to a blank line is how a scan stops being read")


def test_a_phrase_hit_reports_its_line_too(tmp_path):
    """The other half, asserted here so the two paths cannot drift apart again -- which is
    exactly how this defect surfaced, with `_DECLARED` and `_PY_NAME` disagreeing."""
    rule = Rule("phrase", "a planted rule")
    body = "import os\n\n\n# a comment about the market here\n"      # comment is line 4
    got = scan((_planted(tmp_path, body),), {"the market": rule})
    assert len(got) == 1 and got[0].line == 4, got
