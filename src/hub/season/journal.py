"""Every decision, in the order it was made, with what it was chosen over.

A journal and not a Ledger. The accounting sense is the whole distinction and this repo needs
it in both senses at once: a Ledger here is the set of teams an entry has spent
(`hub.season.pool`), and a journal is entries in the order they happened. They were one word
until they collided.

**Two row kinds, never one mutable row.** `hub.store.write` writes a whole partition and
refuses to overwrite one whose contents differ -- deliberately, so a record cannot be quietly
rewritten. Which means a decision written now and rewritten later with its outcome attached is
exactly the case it raises on. So a decision row and an outcome row are written separately
under distinct partition names and joined on a key, and the original is never touched again.

**Absent is not zero, twice over.** A decision taken when the betting market had not posted
records that it had not, because a null price read as 0.0 is a claim nobody made. And the API
cost is a *difference* between two readings of the odds fetcher's credit balance, not a meter:
`credits_remaining` reports a last-known balance and reports nothing at all before the first
pull, so an unknown balance on either side records a null and a reason rather than a zero.

**What it is for.** Two questions nobody can currently answer. Whether picking manually beat
the free auto-pick, which needs the fallback recorded beside the choice every week -- a season
where they matched is a season the model earned nothing. And whether our numbers beat a closing
line, which `docs/track-record.md` argued for and nothing implemented; it costs a column here
rather than a research programme, and it is the instrument that would justify or retire pricing
against the betting market at all.

One column is an obligation rather than a nicety. `docs/decisions.md` registers a survivor
contrarian threshold as a provisional rule under ADR-0014, and that rule's own stated logging
duty is the week, the chalk pick, ours, and the probability cost accepted. A money layer that
recommends differentiation without recording what it gave up is breaking the rule it acts
under.

**The obligation is an invariant and not a column that may be filled.** It was checked by
asking whether a float was there, and three things walked past that: a departure written with
no free pick beside it, the zero the error message itself suggested, and a `kind` nothing
recognised. `_check_adr_0014` is where each of those is closed, and the shape of the fix is
that a cost is a *difference* -- so both figures it is the difference of are on the row, and
the cost is checked against them.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import store

# Two tables, not two schemas in one. The store builds a view per directory, and a
# directory holding partitions of differing shape has no single view to build -- a
# decision-only season would answer queries that name an outcome column with a missing
# column rather than a null.
TABLE = "journal"
OUTCOME_TABLE = "journal_outcome"
LEAGUE = "nfl"

# A closed set, not free text. `kind` decides which obligations a row carries -- ADR-0014's
# logging duty binds a pick, which has a free alternative every week -- so a kind nothing
# recognises is a row that owes nothing, and the way past the check was to invent one.
KINDS = ("pick", "buyback")

# The kinds that have a free alternative, and therefore owe one. Auto-pick assigns a team
# every week for nothing; there is no equivalent standing offer to re-enter, so a buyback is
# not measured against one.
FALLBACK_KINDS = ("pick",)

# Two survival figures compared as floats. Anything this close is the same figure arrived at
# by a different route, and anything further apart is two different claims.
_SAME = 1e-9

# What one entry carries. Nulls are meaningful in four of these and are never filled.
SCHEMA: dict[str, Any] = {
    "key": pl.Utf8,             # season, week, kind and the moment -- unique per decision
    "kind": pl.Utf8,            # one of KINDS
    "season": pl.Int64,
    "week": pl.Int64,
    "at": pl.Datetime,
    "chose": pl.Utf8,           # what we did
    "fallback": pl.Utf8,        # what auto-pick would have done for free
    "fallback_note": pl.Utf8,   # why there is none, when there is none
    "matched_fallback": pl.Boolean,
    "market_price": pl.Float64,  # the betting market's number when we decided; null if unposted
    "price_note": pl.Utf8,      # which source it came from, or why it is null
    "expected_dollars": pl.Float64,
    "chose_survives": pl.Float64,      # P(the season is survived, having taken what we took)
    "fallback_survives": pl.Float64,   # the same, for the free pick
    "survival_given_up": pl.Float64,   # their difference, per ADR-0014's logging duty
    "cost_credits": pl.Float64,        # a before-and-after difference, not a meter
    "cost_note": pl.Utf8,
}

OUTCOME_SCHEMA: dict[str, Any] = {
    "key": pl.Utf8,
    "settled_at": pl.Datetime,
    "survived": pl.Boolean,
    "dollars": pl.Float64,
    "note": pl.Utf8,
}


def key(season: int, week: int, kind: str, at: datetime) -> str:
    """One decision's name, and the partition it is written under.

    The moment is in it because a week holds more than one decision -- a pick and, if that
    pick loses, a buyback -- and two decisions that shared a name would be one partition and
    one of them would be lost.
    """
    return f"{season}-w{week:02d}-{kind}-{at:%Y%m%dT%H%M%S}"


_KEY = re.compile(r"^(?P<season>\d{4})-w(?P<week>\d{2})-(?P<kind>.+)-\d{8}T\d{6}$")


def parse_key(k: str) -> tuple[int, int, str]:
    """The season, week and kind a key already encodes.

    `settle` took the season and week as arguments beside the key, which made them free to
    disagree with it: an outcome filed under the wrong season went to a partition no read of
    the right season would ever join, and the decision stayed unsettled forever with nothing
    raised. They are read out of the key instead, which is the only place they can be wrong
    in exactly one way.
    """
    m = _KEY.match(k)
    if m is None:
        raise ValueError(
            f"{k!r} is not a journal key. Keys are what `journal.key` builds -- "
            "season, zero-padded week, kind, and the moment.")
    return int(m["season"]), int(m["week"]), m["kind"]


def _check_adr_0014(*, week: int, kind: str, chose: str, fallback: str | None,
                    fallback_note: str | None, chose_survives: float | None,
                    fallback_survives: float | None,
                    survival_given_up: float | None) -> None:
    """ADR-0014's logging duty, as an invariant rather than a column that may be filled.

    `docs/decisions.md` registers the survivor contrarian threshold under ADR-0014, and the
    rule's own logging duty is the week, the chalk pick, ours, and the probability cost
    accepted. The check tested that a float was present, which three things walked past.

    **Omitting the fallback.** A departure written with no free pick beside it claims nothing
    and so breaks nothing -- the duty was evaded by silence. A kind that has a free
    alternative every week has to say what it was, or say why there was none.

    **Passing the zero the error message suggested.** A cost is a difference between two
    survival figures, so both of them are named and the difference is checked against them.
    Zero remains a legitimate answer -- the two plans survive alike -- and is now an answer
    that had to be worked out to be written.

    **Using a kind nothing validates.** `kind` was free text, so a row could be filed under a
    name no rule had heard of. It is a closed set, checked before anything else here.
    """
    if kind not in KINDS:
        raise ValueError(
            f"{kind!r} is not a journal kind. One of {', '.join(map(repr, KINDS))} -- a kind "
            "decides which obligations a row carries, so an unrecognised one is a row that "
            "owes nothing, which is how ADR-0014's logging duty was walked past.")
    if kind in FALLBACK_KINDS and fallback is None and not fallback_note:
        raise ValueError(
            f"week {week}: a {kind} has a free alternative every week and this one names "
            "none. Pass `fallback` -- what auto-pick would have assigned -- or, where there "
            "genuinely was none, `fallback_note` saying so. Silence about the free pick is "
            "how ADR-0014's logging duty gets evaded rather than broken.")

    given = fallback is not None and chose != fallback
    if given and (chose_survives is None or fallback_survives is None
                  or survival_given_up is None):
        raise ValueError(
            f"week {week}: {chose!r} departs from the free pick {fallback!r} without "
            "recording what it gave up. ADR-0014's logging duty is the week, the chalk pick, "
            "ours, and the probability cost accepted -- and a cost is a difference, so it "
            "needs both survival figures it is the difference of. Pass chose_survives, "
            "fallback_survives and survival_given_up. Zero is an answer -- the two plans "
            "survive alike -- and it is an answer that has to be computed to be written.")

    for name, p in (("chose_survives", chose_survives),
                    ("fallback_survives", fallback_survives)):
        if p is not None and not 0.0 <= p <= 1.0:
            raise ValueError(f"{name}={p} is not a probability")
    if (chose_survives is not None and fallback_survives is not None
            and survival_given_up is not None
            and abs((fallback_survives - chose_survives) - survival_given_up) > _SAME):
        raise ValueError(
            f"week {week}: survival_given_up={survival_given_up} is not what the two "
            f"survival figures say. The free pick survives {fallback_survives} and ours "
            f"{chose_survives}, a cost of {fallback_survives - chose_survives}. A cost that "
            "does not follow from the figures it is a difference of was not computed from "
            "them, which is the thing ADR-0014 asks to be written down.")


def record(*, season: int, week: int, kind: str, chose: str,
           fallback: str | None = None, fallback_note: str | None = None,
           market_price: float | None = None,
           price_note: str | None = None, expected_dollars: float | None = None,
           chose_survives: float | None = None, fallback_survives: float | None = None,
           survival_given_up: float | None = None,
           credits_before: float | None = None, credits_after: float | None = None,
           at: datetime | None = None, base: Path | None = None) -> str:
    """Append one decision. Returns its key, which is how the outcome finds it later.

    `credits_before`/`credits_after` are the odds fetcher's balance either side of the work
    this decision needed. Either one unknown means the cost is unknown, which is recorded as
    such: a zero would claim the decision was free. Equal readings *are* a zero, and that is a
    different claim -- the decision was made and cost nothing. A balance that *rose* is
    neither: `credits_remaining` is a last-known balance and a quota top-up between the two
    readings makes the difference negative, which would be written into the one column whose
    whole point is that absent is not zero. Refused, so it is recorded as unknown instead.

    `price_note` stands whether or not a price does. It said why the price was null and was
    dropped the moment there was one, so the note that says *which* source a price came from,
    or what was odd about it, could not be written at all.

    The ADR-0014 duty is `_check_adr_0014`, which is where its three escapes are named.
    """
    _check_adr_0014(week=week, kind=kind, chose=chose, fallback=fallback,
                    fallback_note=fallback_note, chose_survives=chose_survives,
                    fallback_survives=fallback_survives,
                    survival_given_up=survival_given_up)
    cost = (credits_before - credits_after
            if credits_before is not None and credits_after is not None else None)
    if cost is not None and cost < 0:
        raise ValueError(
            f"week {week}: the credit balance rose from {credits_before} to {credits_after}, "
            "which is a top-up between the two readings rather than a decision that earned "
            "credits. Recording the difference would put a negative in `cost_credits`, whose "
            "whole purpose is that absent is not zero. Leave a reading out to record the "
            "cost as unknown, which is what it is.")
    at = at or datetime.now(UTC).replace(tzinfo=None)
    k = key(season, week, kind, at)
    row = pl.DataFrame({
        "key": [k], "kind": [kind], "season": [season], "week": [week], "at": [at],
        "chose": [chose], "fallback": [fallback], "fallback_note": [fallback_note],
        "matched_fallback": [None if fallback is None else chose == fallback],
        "market_price": [market_price],
        "price_note": [price_note],
        "expected_dollars": [expected_dollars],
        "chose_survives": [chose_survives], "fallback_survives": [fallback_survives],
        "survival_given_up": [survival_given_up],
        "cost_credits": [cost],
        "cost_note": [None if cost is not None
                      else "credit balance unknown on at least one side"],
    }, schema=SCHEMA)
    store.write(row, TABLE, LEAGUE, season, week, name=k, base=base)
    return k


def settle(k: str, *, survived: bool, season: int | None = None, week: int | None = None,
           dollars: float | None = None, note: str | None = None,
           at: datetime | None = None, base: Path | None = None) -> None:
    """Attach what happened, without touching what was written at the time.

    A separate partition rather than a rewrite. The decision row is the claim as it stood when
    nobody knew the answer, and a record that can be revised once the answer is in is not a
    record of a decision.

    **The season and week come off the key**, which already encodes them. They were arguments
    beside it and therefore free to disagree with it: an outcome filed under the wrong season
    went to a partition that no read of the right season joins, and the decision it belonged
    to stayed unsettled forever with nothing raised. They can still be passed, and are then a
    claim that is checked rather than a coordinate that is obeyed.

    **A key with no decision behind it is refused.** Settling one was writing an outcome that
    joins to nothing -- the same silence, reached by a typo instead of by a wrong season.
    """
    got_season, got_week, _ = parse_key(k)
    for name, given, real in (("season", season, got_season), ("week", week, got_week)):
        if given is not None and given != real:
            raise ValueError(
                f"{name}={given} disagrees with the key {k!r}, which says {real}. The key is "
                "the decision's name and carries its own season and week; an outcome filed "
                "under any others joins to nothing and leaves the decision unsettled.")
    if not store.partition(TABLE, LEAGUE, got_season, got_week, name=k, base=base).exists():
        raise KeyError(
            f"no decision named {k!r} in season {got_season}, week {got_week}. An outcome "
            "against a key nothing wrote joins to nothing, so it is refused rather than "
            "stored where no read will find it.")
    season, week = got_season, got_week
    at = at or datetime.now(UTC).replace(tzinfo=None)
    row = pl.DataFrame({"key": [k], "settled_at": [at], "survived": [survived],
                        "dollars": [dollars], "note": [note]}, schema=OUTCOME_SCHEMA)
    store.write(row, OUTCOME_TABLE, LEAGUE, season, week, name=k, base=base)


def read(season: int, week: int | None = None,
         base: Path | None = None) -> pl.DataFrame:
    """Decisions with their outcomes attached, or with nulls where none has arrived.

    A left join: a decision whose games have not been played is still a decision, and dropping
    it would make the journal read as though nothing had been decided this week.
    """
    empty = pl.DataFrame(schema={**SCHEMA, **{k: v for k, v in OUTCOME_SCHEMA.items()
                                              if k != "key"}})
    have = store.tables(base)
    if TABLE not in have:
        return empty
    params: list[object] = [LEAGUE, season]
    q = f"SELECT * FROM {TABLE} WHERE league = ? AND season = ?"
    if week is not None:
        q += " AND week = ?"
        params.append(week)
    decisions = store.sql(q, params=params, base=base)
    if decisions.is_empty():
        return empty
    # `season` and `week` arrive twice: once as written and once as the Hive partition the
    # store lays out, and the partition's string wins the name. Cast back, so a caller
    # filtering on week does not compare an int to "02".
    decisions = decisions.with_columns(
        pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)).select(list(SCHEMA))
    if OUTCOME_TABLE not in have:
        outcomes = pl.DataFrame(schema=OUTCOME_SCHEMA)
    else:
        outcomes = store.sql(
            f"SELECT * FROM {OUTCOME_TABLE} WHERE league = ? AND season = ?",
            params=[LEAGUE, season], base=base).select(list(OUTCOME_SCHEMA))
    return decisions.join(outcomes, on="key", how="left").sort("at")


def unmatched_weeks(season: int, base: Path | None = None) -> Sequence[int]:
    """Weeks where the pick was *not* the free one. The weeks the model was asked to earn.

    `docs/method.md` rule 5: gate against the simplest thing that already works. Auto-pick is
    that thing, and a season of weeks where we chose it anyway is a season the model did not
    need to exist for -- so the countable quantity is the weeks we departed from it, which is
    what this name says and, until now, the opposite of what it returned. Nothing called it,
    which is the only reason it was a landmine rather than a published figure that was the
    complement of itself.

    A week whose fallback was never recorded is neither matched nor unmatched, and is in
    neither list: `matched_fallback` is null there and a null is not a departure.
    """
    got = read(season, base=base)
    if got.is_empty():
        return []
    hit = got.filter(~pl.col("matched_fallback"))
    return sorted(int(w) for w in hit["week"].unique().to_list())
