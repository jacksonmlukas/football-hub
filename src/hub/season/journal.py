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
"""
from __future__ import annotations

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

# What one entry carries. Nulls are meaningful in three of these and are never filled.
SCHEMA: dict[str, Any] = {
    "key": pl.Utf8,             # season, week, kind and the moment -- unique per decision
    "kind": pl.Utf8,            # "pick", "buyback", or whatever else later decides money
    "season": pl.Int64,
    "week": pl.Int64,
    "at": pl.Datetime,
    "chose": pl.Utf8,           # what we did
    "fallback": pl.Utf8,        # what auto-pick would have done for free
    "matched_fallback": pl.Boolean,
    "market_price": pl.Float64,  # the betting market's number when we decided; null if unposted
    "price_note": pl.Utf8,      # why it is null, when it is
    "expected_dollars": pl.Float64,
    "survival_given_up": pl.Float64,   # against the fallback, per ADR-0014's logging duty
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


def record(*, season: int, week: int, kind: str, chose: str,
           fallback: str | None = None, market_price: float | None = None,
           price_note: str | None = None, expected_dollars: float | None = None,
           survival_given_up: float | None = None,
           credits_before: float | None = None, credits_after: float | None = None,
           at: datetime | None = None, base: Path | None = None) -> str:
    """Append one decision. Returns its key, which is how the outcome finds it later.

    `credits_before`/`credits_after` are the odds fetcher's balance either side of the work
    this decision needed. Either one unknown means the cost is unknown, which is recorded as
    such: a zero would claim the decision was free. Equal readings *are* a zero, and that is a
    different claim -- the decision was made and cost nothing.

    Refuses a pick that departs from the free one without saying what that cost. The column
    was optional and the obligation is not: `docs/decisions.md` registers the survivor
    contrarian threshold under ADR-0014, and the rule's own logging duty is the week, the
    chalk pick, ours, and the probability cost accepted. A journal that lets the entry be
    written anyway records a decision taken in breach of the rule it was taken under, and
    records it as though nothing were missing. `0.0` is a legitimate answer -- the two plans
    survive alike -- and is not the same claim as never having worked it out.
    """
    if fallback is not None and chose != fallback and survival_given_up is None:
        raise ValueError(
            f"week {week}: {chose!r} departs from the free pick {fallback!r} without "
            "recording what it gave up. docs/decisions.md registers the survivor contrarian "
            "threshold under ADR-0014, whose logging duty is the week, the chalk pick, ours, "
            "and the probability cost accepted. Pass survival_given_up -- 0.0 if the two "
            "plans survive alike, which is an answer and not an omission.")
    at = at or datetime.now(UTC).replace(tzinfo=None)
    k = key(season, week, kind, at)
    known = credits_before is not None and credits_after is not None
    row = pl.DataFrame({
        "key": [k], "kind": [kind], "season": [season], "week": [week], "at": [at],
        "chose": [chose], "fallback": [fallback],
        "matched_fallback": [None if fallback is None else chose == fallback],
        "market_price": [market_price],
        "price_note": [price_note if market_price is None else None],
        "expected_dollars": [expected_dollars],
        "survival_given_up": [survival_given_up],
        "cost_credits": [credits_before - credits_after if known else None],
        "cost_note": [None if known else "credit balance unknown on at least one side"],
    }, schema=SCHEMA)
    store.write(row, TABLE, LEAGUE, season, week, name=k, base=base)
    return k


def settle(k: str, *, season: int, week: int, survived: bool,
           dollars: float | None = None, note: str | None = None,
           at: datetime | None = None, base: Path | None = None) -> None:
    """Attach what happened, without touching what was written at the time.

    A separate partition rather than a rewrite. The decision row is the claim as it stood when
    nobody knew the answer, and a record that can be revised once the answer is in is not a
    record of a decision.
    """
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
    """Weeks where the pick was the free one. The number that says what the model earned.

    `docs/method.md` rule 5: gate against the simplest thing that already works. Auto-pick is
    that thing, and a season of weeks where we chose it anyway is a season the model did not
    need to exist for.
    """
    got = read(season, base=base)
    if got.is_empty():
        return []
    hit = got.filter(pl.col("matched_fallback"))
    return sorted(int(w) for w in hit["week"].unique().to_list())
