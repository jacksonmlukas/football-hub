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

**And an invariant with no caller is checked only against fixtures written to satisfy it.**
Those checks name `pool.Weekly`'s fields and nothing supplied one, so a units or
sign-convention disagreement between the two shapes could not have shown up. `record_weekly`
is the adapter that closes that, and the disagreement it found -- the column ADR-0014's rule
would be judged by was not the quantity ADR-0014's rule fires on -- is settled by #209: a
departure logs **both** costs under distinct names. `week_cost`, the free pick's win
probability minus ours, is the threshold quantity and the column that discharges the duty;
`survival_given_up` is the season figure and the thesis of our plan, and on the pinned grid
the two disagree in sign.

**A row carries what it takes to reproduce its own figure** (#162). The schema carried the
dollar figure and the survival given up and none of what produced them -- no pool digest, no
trial count, no seed, no board -- so a row could not be re-derived, and two rows written
under different pool rules were indistinguishable without reading something outside the
journal. `RERUN_COLUMNS` is what a re-run needs, and `test_journal` shows a re-run from a
row's own columns landing on its `expected_dollars` exactly. Rows written before those
columns existed read back with nulls in them and are never migrated to a value nobody
measured: a null there is the row saying it cannot be re-derived, which is true of it.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import store
from hub.season.pool import Weekly, plural

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
    "market_price": pl.Float64,  # the taken team's win probability when we decided, in
                                 # [0, 1]; null if unposted. One unit (#239): a caller
                                 # holding a moneyline converts it through
                                 # `hub.models.props.implied` before writing and says so
                                 # in `price_note`. Rows written before #239 carry
                                 # whichever unit their caller used and are not migrated.
    "price_note": pl.Utf8,      # which source it came from, or why it is null
    "expected_dollars": pl.Float64,
    "chose_survives": pl.Float64,      # P(the season is survived, having taken what we took)
    "fallback_survives": pl.Float64,   # the same, for the free pick
    "survival_given_up": pl.Float64,   # their difference: the thesis of our plan, not the
                                       # threshold quantity (#209)
    "fallback_price": pl.Float64,      # the free pick's own win probability this week
    "week_cost": pl.Float64,           # fallback_price - market_price: the win-probability
                                       # cost ADR-0014's threshold is stated in, and the
                                       # column that discharges its logging duty (#209)
    "cost_credits": pl.Float64,        # a before-and-after difference, not a meter
    "cost_note": pl.Utf8,
    # What it takes to run the figure again (#162): the rules, the board, the trials and
    # the seed they were reseeded to, and the two inputs the caller stated. Null on every
    # row written before these existed, and never filled in -- a null here says the row
    # cannot be re-derived, which is true of it, where a value nobody measured would say
    # it can. `RERUN_COLUMNS` names them so a reader can ask which rows carry it.
    "pool_digest": pl.Utf8,            # `hub.config.pool_digest` of the rules
    "grid_digest": pl.Utf8,            # `hub.season.pool.grid_digest` of the board
    "seed": pl.Int64,                  # what every candidate was reseeded to
    "trials": pl.Int64,                # trials each candidate ran; 0 for closed form
    "entries": pl.Int64,               # the field priced against, ours included
    "pot": pl.Float64,
    "outlay": pl.Float64,              # what `expected_dollars` is net of
    "plan_source": pl.Utf8,            # `Plan.source`: the optimiser, or the fallback
}

# The columns a row needs to be re-derived, in one place. `read` fills them with nulls on a
# store written before they existed rather than failing to select them, and a row whose
# `pool_digest` is null is one no re-run can be checked against.
RERUN_COLUMNS = ("pool_digest", "grid_digest", "seed", "trials", "entries", "pot",
                 "outlay", "plan_source")

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
                    fallback_survives: float | None, survival_given_up: float | None,
                    market_price: float | None = None, fallback_price: float | None = None,
                    week_cost: float | None = None) -> None:
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

    **And the price in the wrong unit** (#239). `market_price` is the taken team's win
    probability -- `week_cost` is `fallback_price - market_price` and is a cost only when
    both are one -- and it was range-checked as one only beside a `fallback_price`, so a
    matched pick could carry an American price there and nothing said so. It is checked
    on every row; a caller with a moneyline converts through `hub.models.props.implied`
    first and names the source in `price_note`.

    **And the cost in the wrong quantity** (#209). ADR-0014's threshold is a *week*
    win-probability cost -- "under ~8pp" -- and the column that discharged its logging duty
    was a season-survival difference, which disagrees with it in sign on the pinned grid.
    A departure now carries both, under distinct names: `week_cost` is the free pick's win
    probability minus ours and is the threshold quantity; `survival_given_up` stays as the
    thesis of our plan working. The same shape as the survival trio -- both prices on the
    row and the cost checked against them -- so a week cost nobody computed cannot be
    written either.
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

    if given and (market_price is None or fallback_price is None or week_cost is None):
        raise ValueError(
            f"week {week}: {chose!r} departs from the free pick {fallback!r} without "
            "recording the week's cost. ADR-0014's threshold is stated in win probability "
            "on the week -- under ~8pp -- and `survival_given_up` is not that quantity "
            "(#209). Pass market_price (ours), fallback_price (the free pick's) and "
            "week_cost, their difference, so the cost the rule fires on is on the row.")

    # `market_price` is a probability on every row, not only where a week cost is stated
    # in it (#239): checked only beside a `fallback_price`, a matched pick could carry 5.0
    # there and a moneyline row and a probability row were indistinguishable in a query.
    for name, p in (("chose_survives", chose_survives),
                    ("fallback_survives", fallback_survives),
                    ("fallback_price", fallback_price),
                    ("market_price", market_price)):
        if p is not None and not 0.0 <= p <= 1.0:
            raise ValueError(f"{name}={p} is not a probability")
    if (market_price is not None and fallback_price is not None and week_cost is not None
            and abs((fallback_price - market_price) - week_cost) > _SAME):
        raise ValueError(
            f"week {week}: week_cost={week_cost} is not what the two prices say. The free "
            f"pick wins at {fallback_price} and ours at {market_price}, a cost of "
            f"{fallback_price - market_price} on the week. ADR-0014's threshold is stated "
            "in this quantity, so a cost that was not computed from the prices is not "
            "the cost the rule fires on.")
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
           market_price: float | None = None, fallback_price: float | None = None,
           week_cost: float | None = None,
           price_note: str | None = None, expected_dollars: float | None = None,
           chose_survives: float | None = None, fallback_survives: float | None = None,
           survival_given_up: float | None = None,
           credits_before: float | None = None, credits_after: float | None = None,
           pool_digest: str | None = None, grid_digest: str | None = None,
           seed: int | None = None, trials: int | None = None, entries: int | None = None,
           pot: float | None = None, outlay: float | None = None,
           plan_source: str | None = None,
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

    The `RERUN_COLUMNS` columns are what a re-run needs, and they are optional here because a
    decision is a decision whether or not its figure can be reproduced -- a row that says
    "this was chosen" and cannot say under what is still the record of a choice. What is
    refused is *pretending*: a row carrying some of them and not others would read as
    reproducible to a query on any one column, so either every one of them is present or
    none is. `record_weekly` supplies all of them off `pool.Weekly`.
    """
    # `plan_source` labels the survivor plan and is no input to the re-run, and a buyback has no
    # plan; the seven that follow are what `pool.weekly` has to be handed to land on the row's
    # figure again.
    rerun = {"pool_digest": pool_digest, "grid_digest": grid_digest, "seed": seed,
             "trials": trials, "entries": entries, "pot": pot, "outlay": outlay}
    absent = tuple(k for k, v in rerun.items() if v is None)
    if absent and len(absent) != len(rerun):
        raise ValueError(
            f"week {week}: provenance has to come whole. Missing {absent} beside "
            f"{tuple(k for k in rerun if k not in absent)} -- a row naming the rules but "
            "not the seed, or the seed but not the board or the stakes, reads as "
            "reproducible to whichever column is queried and is not. Pass all of "
            f"{tuple(rerun)}, or none and let the row say it cannot be re-derived.")
    _check_adr_0014(week=week, kind=kind, chose=chose, fallback=fallback,
                    fallback_note=fallback_note, chose_survives=chose_survives,
                    fallback_survives=fallback_survives,
                    survival_given_up=survival_given_up, market_price=market_price,
                    fallback_price=fallback_price, week_cost=week_cost)
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
        "fallback_price": [fallback_price], "week_cost": [week_cost],
        "cost_credits": [cost],
        "cost_note": [None if cost is not None
                      else "credit balance unknown on at least one side"],
        "pool_digest": [pool_digest], "grid_digest": [grid_digest],
        "seed": [seed], "trials": [trials], "entries": [entries],
        "pot": [pot], "outlay": [outlay], "plan_source": [plan_source],
    }, schema=SCHEMA)
    store.write(row, TABLE, LEAGUE, season, week, name=k, base=base)
    return k


def record_weekly(w: Weekly, *, season: int, chose: str | None = None,
                  credits_before: float | None = None, credits_after: float | None = None,
                  at: datetime | None = None, base: Path | None = None) -> str:
    """Record a week `hub.season.pool.weekly` priced. The caller ADR-0014's duty was missing.

    `_check_adr_0014` was written against `pool.Weekly`'s field names and had no caller, so
    every figure it checks was only ever supplied by a hand-built fixture. This is the
    adapter, and it is the only place the two shapes meet -- which is where a units or
    sign-convention disagreement between them can be seen at all.

    **`chose` is what was entered, not what was recommended.** They are usually the same and
    the argument exists for when they are not: an operator may override, and a journal records
    the decision rather than the advice. `Weekly.given_up` cannot be forwarded in that case --
    it is `fallback.survives - recommend.survives` by construction, so against an overridden
    pick it is a cost from a comparison nobody made. The difference is recomputed from the
    candidate actually taken, and `_check_adr_0014` refuses the row if the two disagree.

    **The quantity mismatch, named rather than folded away.** ADR-0014 adopts the survivor
    contrarian threshold as "take a differentiation week when the win-probability *cost* is
    under ~8pp", and `docs/decisions.md` logs "the probability cost accepted". What is written
    to `survival_given_up` is a *season-survival* difference, which is a different quantity:
    on the grid in `tests/unit/test_journal.py` the chalk pick is 2.0pp better on the week and
    9.9pp *worse* over the season, so the two costs differ in sign as well as scale. That is
    not incidental -- the whole thesis of the survivor plan is that a lower win probability
    now can buy a higher survival later, so the two routinely disagree.

    Neither is relabelled as the other, and both are logged (#209). `market_price` carries
    the taken team's own win probability, `fallback_price` the free pick's, and `week_cost`
    their difference -- the quantity the ADR's threshold is stated in, and the column that
    discharges its logging duty. `survival_given_up` stays beside it as the thesis of the
    plan working: on the grid in `tests/unit/test_journal.py` they disagree in sign, 2.0pp
    better on the week and 27.8pp worse over the season, and dropping the season figure
    would lose the argument for our plan where logging it under the threshold's name was
    the error. ADR-0014 names which column is its threshold quantity, dated.

    **Who calls this.** An operator, at the point the week's pick is entered: `pool.weekly`
    prices the week and this writes down what was done about it. There is no scheduled job
    above either of them, and this does not add one -- `weekly` is a pure pricing function
    that the tests call hundreds of times, and a store write inside it would file a decision
    every time anybody asked what a week was worth.
    """
    by_team = {c.team: c for c in w.candidates}
    took = chose if chose is not None else w.recommend
    if took not in by_team:
        raise ValueError(
            f"week {w.week}: {took!r} is not one of the teams this week priced "
            f"({', '.join(sorted(by_team))}). A row whose survival figures came from a "
            "candidate nobody valued would satisfy ADR-0014's check and mean nothing.")
    fb = next((c for c in w.candidates if c.is_fallback), None)
    return record(
        season=season, week=w.week, kind="pick", chose=took,
        fallback=w.fallback,
        # `weekly_report` already refuses to stringify an absent auto-pick, and the note is
        # the same fact in the column that exists to carry it.
        fallback_note=(None if w.fallback is not None else
                       "auto-pick had no team left to assign, so there was nothing free"),
        market_price=by_team[took].win_prob,
        fallback_price=fb.win_prob if fb is not None else None,
        week_cost=(fb.win_prob - by_team[took].win_prob) if fb is not None else None,
        price_note="win probability off the board's grid at the moment of the decision",
        expected_dollars=by_team[took].expected_dollars,
        chose_survives=by_team[took].survives,
        fallback_survives=fb.survives if fb is not None else None,
        survival_given_up=(fb.survives - by_team[took].survives) if fb is not None else None,
        credits_before=credits_before, credits_after=credits_after,
        # Everything a re-run needs, off the shape that ran (#162). `plan_source` is the
        # taken candidate's, because the figure on this row is that candidate's figure.
        pool_digest=w.pool_digest, grid_digest=w.grid_digest, seed=w.seed,
        trials=w.trials, entries=w.entries, pot=w.pot, outlay=w.outlay,
        plan_source=by_team[took].plan_source or None,
        at=at, base=base)


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
    #
    # A column the schema has and the store does not is one every row predates: the store
    # unions partitions by name, so a mix of old and new rows already reads with nulls in
    # the old ones, and a store holding *only* old rows has no such column to union. Added
    # as null rather than failing the select, because those rows are readable decisions
    # that cannot be re-derived, and a null in `RERUN_COLUMNS` is exactly that claim (#162).
    absent = [pl.lit(None, dtype=t).alias(c) for c, t in SCHEMA.items()
              if c not in decisions.columns]
    decisions = decisions.with_columns(
        pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), *absent
    ).select(list(SCHEMA))
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


# --- the entry point (#163) ---------------------------------------------------------------
#
# The journal had writers and readers and no way to reach either from a terminal, which is
# the shape `tests/contracts/test_cli_surface.py` exists to refuse: a module with no `main`
# is exempt from the contract that every entry point answers absent input with a sentence.
# `hub.season.pool --record` is what writes a pick; this is what reads the season back and
# what settles a decision once the games it was about have been played.

def report(rows: pl.DataFrame) -> list[str]:
    """Decisions as lines rather than prints, so `hub.season.pool` can serve them as last-good.

    One line per row, and the columns a reader acts on: what was chosen, over what, at
    what price, and -- where the row can be re-derived -- the rules and the seed it can be
    re-derived under. A row from before `RERUN_COLUMNS` existed says so rather than printing
    a blank where a digest would go, because the blank would read as a missing digest and the
    fact is that nothing was recorded.
    """
    if rows.is_empty():
        return ["\n  no decisions recorded"]
    out = [f"\n  {'week':>4}  {'kind':<7}  {'chose':<10}  {'free':<5}  {'$':>8}  "
           f"{'week cost':>9}  {'given up':>9}  {'settled':<8}  rules"]
    for r in rows.sort("at").iter_rows(named=True):
        dollars = "" if r["expected_dollars"] is None else f"{r['expected_dollars']:+.2f}"
        # Both costs, under their own names (#209): the week's is the threshold quantity,
        # the season's is the thesis, and on a real grid they disagree in sign.
        week = "" if r["week_cost"] is None else f"{r['week_cost'] * 100:+.1f}pp"
        cost = ("" if r["survival_given_up"] is None else
                f"{r['survival_given_up'] * 100:+.1f}pp")
        settled = ("" if r.get("survived") is None else
                   "survived" if r["survived"] else "out")
        rules = (f"{r['pool_digest']} seed {r['seed']} x{r['trials']}"
                 if r.get("pool_digest") is not None else "not re-derivable")
        out.append(f"  {r['week']:>4}  {r['kind']:<7}  {r['chose']:<10}  "
                   f"{(r['fallback'] or '-'):<5}  {dollars:>8}  {week:>9}  {cost:>9}  "
                   f"{settled:<8}  {rules}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from hub.cli import unavailable
    from hub.config import SEASON_AHEAD

    ap = argparse.ArgumentParser(
        prog="hub.season.journal",
        description="Read the season's survivor decisions back, or settle one.")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--settle", default=None, metavar="KEY",
                    help="attach an outcome to the decision with this key")
    ap.add_argument("--survived", action="store_true", help="with --settle: the pick won")
    ap.add_argument("--dollars", type=float, default=None,
                    help="with --settle: what the decision paid, net")
    ap.add_argument("--note", default=None, help="with --settle: anything a replay cannot")
    ap.add_argument("--store", type=Path, default=None,
                    help="the processed store the journal is read from and written to")
    a = ap.parse_args(argv)

    if a.settle:
        try:
            settle(a.settle, survived=a.survived, dollars=a.dollars, note=a.note,
                   base=a.store)
        except (KeyError, ValueError) as e:
            return unavailable("hub.season.journal", f"the decision {a.settle!r}", e)
        print(f"  settled {a.settle}: {'survived' if a.survived else 'out'}")
        return 0

    try:
        rows = read(a.season, a.week, base=a.store)
    except Exception as e:
        return unavailable("hub.season.journal", f"the {a.season} decision journal", e)
    if rows.is_empty():
        where = a.store or store.DATA
        return unavailable(
            "hub.season.journal", f"the {a.season} decision journal",
            FileNotFoundError(f"no decision recorded for {a.season}"
                              + (f" week {a.week}" if a.week is not None else "")
                              + f" under {where}; `hub.season.pool --record` writes one"))
    for line in report(rows):
        print(line)
    left = unmatched_weeks(a.season, base=a.store)
    print(f"  {rows.height} decision(s); departed from auto-pick in "
          f"{plural(len(left), 'week')}" + (f": {', '.join(map(str, left))}" if left else ""))
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
