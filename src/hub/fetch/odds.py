"""The Odds API: two declared markets, one region, one pull.

The billing model is the whole design constraint. A request costs **markets x regions**,
so asking for spreads and totals across us and uk is four credits for one call.
`docs/decisions.md` records that this multiplier is why props are roadmap-only: a single
props pull would cost more than a month of the free tier.

**Two markets, and the guard that lets them through is stricter than the one that did not.**
It was one market until #211, and the reason for the second is structural rather than
incremental. A spread is a *difference* of two team scores and a total is a *sum*, so an
effect that is symmetric between the sides cancels in one and adds in the other: a defensive
downgrade raises the opponent's expected scoring, which subtracts from the margin and adds
to the total. Storing only spreads makes the repo blind to half the betting market, and the
half it cannot see is where the one untested non-QB hypothesis lives. The price of seeing it
is that one call returns the whole season, so a second market is one extra credit *per poll*
rather than per week -- of the order of twenty-five credits for the rest of the season.

"Cannot silently burn quota" therefore means three things, and this module does all three.
The multiplier cannot be triggered by accident -- a market or region this module has not
declared, or one asked for twice, is refused before a request is formed, which refuses every
props market by name rather than by counting commas. The balance is never a mystery: every
pull reads `x-requests-remaining` from the response, prints it, and stores it, and a stored
balance below the floor refuses the *next* pull before spending anything. And what a poll
actually cost is *measured* rather than asserted -- the balance before minus the balance
after, recorded beside the balance and compared against markets x regions, so the day the
billing model stops being what the guard assumes, the guard says so instead of the invoice.

What the snapshot is for matters as much as what it costs. `hub.store.AS_OF_LINES` joins
lines to predictions on nflverse `game_id`, and this is the only thing that will ever
write more than one line per game. Keyed on The Odds API's own event ids it would be an
island, so events are mapped back to nflverse games via team abbreviations and kickoff
date. Every snapshot appends, because a single closing line makes the as-of join
degenerate -- line *movement* is the thing it exists to resolve.

**A dated line is not the same as a live one, and since #210 the row says which.** A number
polled eight times over twelve days and identical at every poll is not a noisy estimate of
the close; it is a posted lookahead that does not respond to news, and the archive measured
on 2026-09-11 is mostly that -- 260 of 272 games never moved across the whole of it, and
every week from 2 out had at most two games move. So each row carries `polls_unmoved` and
`unmoved_since`: how many consecutive polls have returned this quote and when the run began.
`staleness` derives both from the archive, `_record` stamps them on the rows it writes, and
a consumer ranking a snapshot above the schedule's own field has the number it needs to stop
doing that for a quote nothing has touched in a fortnight.

    uv run python -m hub.fetch.odds --credits
    uv run python -m hub.fetch.odds --snapshot
    uv run python -m hub.fetch.odds --staleness
    uv run python -m hub.fetch.odds --noise-floor
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import store
from hub.cli import unavailable
from hub.config import SEASON_AHEAD
from hub.contracts import ODDS_SNAPSHOT, PROP_SNAPSHOT
from hub.names import player_key
from hub.paths import STATE_DIR

ROOT = Path(__file__).resolve().parents[3]
# Under `state/`, not `data/raw/`. The balance is what the floor below refuses on, and
# `.gitignore` excludes the whole of `data/` as redistributed third-party payloads -- so on
# an Actions runner `credits_remaining` always answered None, the floor never refused, and
# the header read back from each response was written to a file the next run could not see.
# A guard that cannot fire reads as a guard.
STATE = STATE_DIR / "odds.json"

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# What this module is allowed to ask for, and the whole of it. Cost is markets x regions,
# so this pair of tuples *is* the budget: two markets in one region is two credits a poll,
# and nothing outside them can be requested without editing these lines. Declared as
# allowlists rather than counted as commas because the expensive mistake is not "how many"
# but "which" -- `player_pass_tds` is a single market and runs about four credits an event,
# which a comma count waves through and a name check refuses.
MARKET = "spreads"
TOTALS_MARKET = "totals"
MARKETS = (MARKET, TOTALS_MARKET)

REGION = "us"
REGIONS = (REGION,)

# The sides the stored number belongs to. A price is meaningless without one: -110 on the
# home spread and -110 on the away spread are different facts about the same game, and a
# total priced from the Under is the mirror of one priced from the Over. Home for spreads
# because `close_spread` is the home line; Over for totals because that is the side the
# point is quoted from.
OVER = "Over"

# Refuse the next pull below this. Sized to leave room for a full week of Sunday-morning
# snapshots after the balance is noticed, rather than stopping dead at zero.
CREDIT_FLOOR = 50


class MultiplierRefused(Exception):
    """A market or region outside the declared budget: the request would cost more than it may.

    It caught "more than one" until #211, by counting commas. That is the wrong quantity --
    the second market this repo now wants costs one extra credit a poll and is affordable,
    while `player_pass_tds` is a single market with no comma in it and costs roughly four
    credits an *event*. So the refusal is on the name rather than the count, and the budget
    is `MARKETS` x `REGIONS` by construction rather than by arithmetic nobody re-derives.
    """


class QuotaFloor(Exception):
    """Stored balance is below the floor. Refused before spending a credit."""


class SnapshotIncomplete(Exception):
    """The betting market answered, the credit is spent, and what failed after is ours.

    A snapshot is a fetch followed by four things that are not one: recording the credit, a
    schedule join against a *different* provider, the contract check, and the write. Reporting
    any of those as "the betting market's prices unavailable" names the one source that
    worked, and sends an operator to retry -- which spends another credit from a metered
    monthly quota against a problem that is not there (issue #119).
    """


def _api_key() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("ODDS_API_KEY") or None


def _read_state(path: Path | None) -> dict[str, Any]:
    try:
        return json.loads(Path(path or STATE).read_text())
    except Exception:
        return {}


def credits_remaining(path: Path | None = None) -> int | None:
    """Last known balance, or None if we have never seen one.

    None is meaningfully different from zero: an unknown balance must not refuse the pull
    that would tell us what it is.
    """
    v = _read_state(path).get("remaining")
    return int(v) if v is not None else None


def _write_state(path: Path | None, remaining: int | None, when: datetime, *,
                 cost: int | None = None, declared: int | None = None,
                 asked: str | None = None) -> None:
    """The balance, and what the poll that reported it actually cost.

    `cost` is a measurement rather than a restatement of `declared`: the balance before the
    call minus the balance after it, both read off `x-requests-remaining`. The two are
    written side by side on purpose. `declared` is what markets x regions says the poll
    should have cost and is the number the guard budgets on; `cost` is what the account was
    charged. Recording only the first would make the budget unfalsifiable, which is the
    state #211 found this module in -- the multiplier was documented in three places and
    measured in none.

    `cost` is None on the first ever poll and after a monthly reset, and None is written
    rather than a guess. There is no prior balance to subtract from in the first case and
    the difference is meaningless in the second, and a plausible wrong number here would be
    read as evidence that the billing model is what we think it is.
    """
    p = Path(path or STATE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"remaining": remaining, "checked_at": when.isoformat(),
                             "last_cost": cost, "declared_cost": declared,
                             "asked_for": asked}, indent=2))


def _budgeted(value: str, allowed: tuple[str, ...], kind: str) -> tuple[str, ...]:
    """The parameter as this module is allowed to send it, or a refusal naming the cost.

    Two refusals, and both of them are about credits rather than tidiness.

    **Undeclared.** Cost is markets x regions, so anything not in `allowed` is a credit
    nobody budgeted for, on every poll for the rest of the season. Refusing by name rather
    than by counting separators is what makes this cover the case that actually matters:
    `docs/decisions.md` records a full week of props at roughly 64 credits, and every one of
    those market keys is a single token that a comma count is happy with.

    **Repeated.** `spreads,spreads` is two markets to a biller that counts what was asked
    for, and one to a reader skimming this file. Refusing it means the cost of a request is
    always `len(markets) * len(regions)` over *distinct* names, which is the arithmetic the
    measurement below is checked against.

    Whitespace is refused rather than stripped for the same reason a repeat is: a value this
    module rewrites before sending is a value whose cost is computed from something other
    than what was asked for.
    """
    asked = tuple(value.split(","))
    if not value or any(p != p.strip() or not p for p in asked):
        raise MultiplierRefused(
            f"{kind}={value!r}: empty or padded. Cost is markets x regions and is counted "
            f"off what is sent, so this module sends exactly what it was handed.")
    unknown = sorted({p for p in asked if p not in allowed})
    if unknown:
        raise MultiplierRefused(
            f"{kind}={value!r}: {unknown} outside the declared {kind} {list(allowed)}. "
            f"Cost is markets x regions, so an undeclared {kind} is an unbudgeted credit on "
            f"every poll -- a single props market runs about four credits an event.")
    if len(set(asked)) != len(asked):
        raise MultiplierRefused(
            f"{kind}={value!r} names one twice. Cost is markets x regions counted off what "
            f"was asked for, so a repeat is a second credit for a column already paid for.")
    return asked


def _http_get(params: Mapping[str, Any], key: str) -> tuple[Any, Mapping[str, str]]:
    import requests
    r = requests.get(f"{BASE}/sports/{SPORT}/odds", timeout=30,
                     params={**params, "apiKey": key})
    r.raise_for_status()
    return r.json(), r.headers


def _game_date(commence_time: str) -> str:
    """The date the game is *played on*, which is not the date its kickoff falls on in UTC.

    The Odds API stamps `commence_time` in UTC; nflverse's `gameday` is the local date in
    Eastern. Any kickoff at or after 20:00 ET is past midnight UTC, so slicing the raw string
    put it on the following day and it matched nothing.

    That is not a rounding error, it is the primetime slate. Measured against the 2026
    schedule: **55 of 272 games kick off at or after 20:00 ET** -- 17 Sunday, 17 Monday, 17
    Thursday -- and the 2026-09-04 snapshot lost 65 of 272 events, leaving the store with a
    line for 207 games and none for every Sunday, Monday and Thursday night game of the season.

    Converted rather than offset by a constant because the season crosses out of daylight
    saving in November: -4 through week 9 and -5 after it, and a fixed offset would fix the
    first half and break the second.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    raw = str(commence_time or "")
    try:
        utc = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw[:10]                # unparseable: fall back to the old behaviour, not a crash
    return utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _team_abbrs(valid: set[str]) -> dict[str, str]:
    """Full name to abbreviation, from nflverse rather than a hardcoded table.

    The Odds API says "Philadelphia Eagles"; nflverse says "PHI". A literal map would be
    32 lines that go stale the next time a team relocates.

    **`valid` is not optional in practice, and the reason is a silent collapse.**
    `nfl.load_teams()` lists "Los Angeles Rams" twice, once as `LA` and once as `LAR`, so
    building the map with a plain `dict(zip(...))` keeps whichever came last and throws the
    other away. It kept `LAR`. The 2026 schedule uses `LA`. Every Rams game therefore matched
    nothing -- 17 of them, which was the whole of the residual after the timezone fix, and it
    looked like a rounding error rather than one team's entire season.

    The `strict=True` on that zip gave false comfort: it checks that the two columns are the
    same length, which they were, and says nothing about duplicate keys.

    `valid` is required rather than defaulted for that reason: there is no principled way to
    pick between two abbreviations for one name without knowing which the season uses, and a
    default would just be choosing a different arbitrary winner. The caller has the schedule
    in hand, so it can say. A name whose abbreviations are all outside `valid` -- the historical
    entries, Oakland and San Diego and the rest -- keeps the first, and never appears in a
    payload for a season it did not play in.
    """
    import nflreadpy as nfl
    t = nfl.load_teams()
    out: dict[str, str] = {}
    for name, abbr in zip(t["team_name"].to_list(), t["team_abbr"].to_list(), strict=True):
        if name not in out or (abbr in valid and out[name] not in valid):
            out[name] = abbr
    return out


def _schedule(season: int) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_schedules().filter(pl.col("season") == season)


def _american(price: Any) -> float | None:
    """One outcome's price as American odds, or None if it cannot be American odds.

    American odds have a hole in them: a price is at most -100 or at least +100, and nothing
    lies between, because the two ways of quoting even money are the two edges of that gap.
    A number inside it is therefore not an American price -- decimal odds (1.91) and implied
    probabilities (0.52) both land there -- and neither is a missing or unparseable one.

    The request asks for `oddsFormat=american` and this is what notices the day that stops
    being honoured. It degrades rather than refuses: the point beside it is unaffected by
    how the price was quoted, so an unreadable price becomes a null in a nullable column and
    a line on the terminal. Refusing the frame instead would throw away a snapshot whose
    credit has already been spent, over the half of it that still parsed.
    """
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    return p if abs(p) >= 100.0 else None


def _median_price(prices: list[float]) -> float | None:
    """Median American price, taken in decimal space, which is not a detail.

    American odds are not a number line. They run ... -120, -110, -100 / +100, +110, +120
    ... with nothing between the two hundreds, so the ordinary median of an even number of
    quotes can land somewhere no book can quote: -110 and +110 median to **0**, which is not
    a price at all, and would sit unremarked inside any plausibility bound a contract could
    put on the column.

    Decimal odds are continuous and increase monotonically with the American price, so the
    median there is the median quote, and the conversion back cannot produce the hole it
    avoided -- a decimal below 2 returns at most -100 and one at or above 2 at least +100.
    The same pair comes back at +100.45, which is even money and a hair, which is what two
    books at -110 and +110 actually are.

    -100 round-trips to +100. They are the same price quoted from the two sides of even
    money, and +100 is the form this module stores.
    """
    if not prices:
        return None
    d = statistics.median([1.0 + p / 100.0 if p > 0 else 1.0 + 100.0 / -p for p in prices])
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def _median_quote(event: Mapping[str, Any], market: str,
                  side: str) -> tuple[float | None, float | None]:
    """The betting market's point and price for one side of one game, across books.

    Median rather than mean or first: one book hanging an outlier should not become the line
    of record, and books disagree by half points routinely.

    **The point and the price are medianed over the same books but not over the same list.**
    A book contributes a point when it posts one; it contributes a price as well when that
    price can be American odds. So a book quoting a point and nothing readable beside it
    still moves the point, exactly as it did before there was a price to read -- which is
    what keeps `close_spread` the number it has always been for any payload, rather than one
    that quietly shifts because a price was malformed somewhere.

    The side is matched case-insensitively because it is a display string. Team names come
    back the way `_team_abbrs` was built to read them, but "Over" is a word The Odds API
    chose the casing of, and this module has never seen a live totals response.
    """
    points: list[float] = []
    prices: list[float] = []
    for book in event.get("bookmakers") or []:
        for m in book.get("markets") or []:
            if m.get("key") != market:
                continue
            for outcome in m.get("outcomes") or []:
                if str(outcome.get("name") or "").lower() != side.lower():
                    continue
                if outcome.get("point") is None:
                    continue
                points.append(float(outcome["point"]))
                price = _american(outcome.get("price"))
                if price is not None:
                    prices.append(price)
    if not points:
        return None, None
    return statistics.median(points), _median_price(prices)


def _median_home_spread(event: Mapping[str, Any],
                        home: str) -> tuple[float | None, float | None]:
    """Median home-side line across books in nflverse's sign convention, and its price.

    The negation is the part to be careful about. The Odds API reports a *handicap*, so a
    home favourite is -8.5 -- they must win by more than 8.5. nflverse `spread_line` is the
    opposite: positive means the home team is favoured (2025_01_DAL_PHI is home PHI at
    +8.5, and PHI won). `store.AS_OF_LINES` and `store.verify` both already speak nflverse,
    so this converts once, here, rather than leaving two conventions loose in one table.
    Get it wrong and every backtest is confidently backwards while still looking calibrated.

    **The price is not negated with it, and that is the trap this pair creates.** The sign of
    a point is a convention two providers disagree about; the sign of an American price is
    part of the price -- -120 means risk 120 to win 100 and +120 means the reverse, and they
    are different numbers rather than one number seen from two sides. Negating the price
    alongside the point would turn every home favourite's juice into a plus number and read
    as a market offering value on the side it is charging most for.
    """
    point, price = _median_quote(event, MARKET, home)
    return (None if point is None else -point), price


def _median_game_total(event: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Median game total across books, and the Over's price.

    A total is a **sum** of the two team scores where a spread is a difference, which is the
    whole reason it is worth a second credit: an effect symmetric between the sides cancels
    in the difference and adds in the sum. It is also why nothing is negated here. There is
    no home-and-away convention to reconcile in a sum -- 47.5 is 47.5 to every provider --
    so the point comes back as it arrived, and a negation copied from the spread above would
    put every total outside any bound a contract could sanely declare.
    """
    return _median_quote(event, TOTALS_MARKET, OVER)


# --- staleness (#210) ---------------------------------------------------------

# The two columns a row carries about how long its spread quote has stood still. Declared
# here and read by `ODDS_SNAPSHOT`, so the writer and the contract cannot spell them apart.
STALENESS_COLUMNS: dict[str, Any] = {"polls_unmoved": pl.Int64, "unmoved_since": pl.Datetime}

# What `staleness` reads and the order it reads it in. `spread_price` is optional on the
# way in -- a partition written before #211 has no such column and the store hands it back
# as null -- and the derivation treats a null there as no evidence rather than as a move.
_QUOTE = ("game_id", "close_spread", "spread_price", "captured_at")


def _quote_moved(key: Sequence[str] = ("game_id",), point: str = "close_spread",
                 price: str = "spread_price") -> pl.Expr:
    """Whether this poll's spread quote differs from the previous poll of the same game.

    A quote is the pair -- the point and the price on it -- and the pair moved if either did.
    The point is the obvious half. The price is the half #211 made storable and the one this
    rule has to say something about, because a book that has a reason to move a line moves
    the juice first: -7 at -110 becoming -7 at -120 is the betting market leaning without crossing
    the key number, and a run of polls that is "unmoved" on the point alone would count that
    as nothing happening. A frozen lookahead is a number nobody has touched, and a shaded
    price is a touch.

    A price that is null on either side contributes nothing. Every snapshot before #211 has
    no price at all, and the poll that first records one is the archive gaining a column,
    not the betting market moving -- so the comparison is made only where both polls priced the
    point, and the point decides otherwise. Floats compare exactly because the medians on
    both sides are the same arithmetic over the same books' quotes: two polls at which no
    book changed anything produce identical bits, and a book joining or leaving the set is
    the betting market changing shape, which is honestly a move.

    The first poll of a game starts a run. There is no earlier quote to stand still against.

    **A prop's point can be null and a spread's cannot, and the rule reads the same either
    way.** `player_anytime_td` has no point at all -- it is a Yes price on "at least one" --
    so two polls that both carry a null point are compared on the price alone, and a null
    against a number is a move, since the shape of the quote changed. `ne_missing` is what
    makes null-against-null "not moved" rather than "unknown", which the plain `!=` would
    say, and it decides the spread case identically because a spread point is never null.
    """
    prev_point = pl.col(point).shift(1).over(*key)
    prev_price = pl.col(price).shift(1).over(*key)
    point_moved = pl.col(point).ne_missing(prev_point)
    price_moved = (pl.col(price).is_not_null() & prev_price.is_not_null()
                   & (pl.col(price) != prev_price))
    # The first poll: no previous *row*, which is not the same as a previous null point.
    first = pl.col("captured_at").shift(1).over(*key).is_null()
    return (first | point_moved | price_moved).alias("_moved")


def staleness(lines: pl.DataFrame, *, key: Sequence[str] = ("game_id",),
              point: str = "close_spread", price: str = "spread_price") -> pl.DataFrame:
    """Every poll, with how long its spread quote had stood still by then.

    `key`, `point` and `price` default to the spread's columns and are what the props archive
    passes differently: a prop is one quote per (game, player, market), its point is
    `point` and its price is the Over's. Same derivation, second caller -- #217 needs
    `polls_unmoved` and `unmoved_since` on a prop row for the reason #210 needed them on a
    spread row, and a second copy of this rule is how the two would come to disagree about
    what "unmoved" means.

    Two columns, and both are needed because they answer different questions. `polls_unmoved`
    is the count of consecutive polls, this one included, that returned the quote this row
    holds -- so a game polled once and a game polled eight times at one number read 1 and 8,
    which is the distinction #210 asks for. `unmoved_since` is `captured_at` of the first
    poll in that run: the *moment*, not the elapsed time, because how long ago that was
    depends on when the question is asked and a stored duration would be stale the moment it
    was written. A consumer subtracts it from its own `at`.

    Neither is a verdict. "Stale" is a threshold on these two numbers, and the threshold is
    the consumer's to declare -- `hub.schedule.priced_games` ranks a snapshot above the
    schedule field and is the one that has to say at what age it stops. This module measures.

    Derived from the archive rather than kept as running state, so it is the same function
    on the way in and on the way out: `_record` runs it over the prior polls plus the new
    rows and stamps the new rows, and a reader of the whole archive -- `hub.models.market`'s
    noise floor, `--staleness` below -- runs it over every partition and gets the same answer
    for the rows written before the columns existed. One derivation, two callers, nothing to
    drift.

    Rows are returned sorted by game and time, the two columns appended, and any column the
    caller passed beyond `_QUOTE` carried through untouched.
    """
    key = tuple(key)
    frame = lines if price in lines.columns else lines.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(price))
    ordered = frame.sort(*key, "captured_at")
    run = pl.col("_moved").cast(pl.Int64).cum_sum().over(*key).alias("_run")
    return (ordered.with_columns(_quote_moved(key, point, price))
                   .with_columns(run)
                   .with_columns(
                       pl.col("captured_at").cum_count().over(*key, "_run")
                         .cast(pl.Int64).alias("polls_unmoved"),
                       pl.col("captured_at").min().over(*key, "_run")
                         .alias("unmoved_since"))
                   .drop("_moved", "_run")
                   .pipe(lambda d: d if price in lines.columns else d.drop(price)))


_QUOTE_SCHEMA: dict[str, Any] = {"game_id": pl.Utf8, "close_spread": pl.Float64,
                                 "spread_price": pl.Float64, "captured_at": pl.Datetime}


def _archive(season: int, base: Path | None) -> pl.DataFrame:
    """Every poll already in the store for the season, in `_QUOTE` shape plus `week`.

    Empty on a fresh clone. Read with `SELECT *` rather than by naming the columns, because
    an archive written entirely before #211 -- which is the live one on 2026-09-11 -- has no
    `spread_price` column in any partition, and `union_by_name` unions what exists rather
    than what a contract now declares. A column the archive has never had is added here as
    nulls, which `_quote_moved` reads as no evidence.
    """
    if "lines" not in store.tables(base):
        return pl.DataFrame(schema={**_QUOTE_SCHEMA, "week": pl.Int64})
    got = store.sql("SELECT * FROM lines WHERE league = 'nfl' AND season = ?",
                    params=[season], base=base)
    absent = [pl.lit(None, dtype=t).alias(c) for c, t in _QUOTE_SCHEMA.items()
              if c not in got.columns]
    return got.with_columns(absent).select(*_QUOTE, pl.col("week").cast(pl.Int64))


def _with_staleness(new: pl.DataFrame, season: int, base: Path | None) -> pl.DataFrame:
    """The new rows, stamped with their staleness against everything already stored.

    The prior polls are read and never rewritten. Immutability of what is on disk is the
    whole as-of guarantee, so the archive's older rows keep the columns they were written
    with and get these two on read, from the same `staleness`, rather than by a backfill.
    """
    prior = _archive(season, base).select(_QUOTE)
    both = pl.concat([prior, new.select(_QUOTE)], how="vertical_relaxed")
    stamped = staleness(both).select("game_id", "captured_at", *STALENESS_COLUMNS)
    return new.join(stamped, on=["game_id", "captured_at"], how="left")


def staleness_report(season: int = SEASON_AHEAD, base: Path | None = None) -> int:
    """Per week: how much of the archive is a number that has never moved.

    Printed rather than returned because the reader is a person deciding whether the
    snapshot for a distant week is worth ranking above the schedule field. "Frozen" here is
    the whole-archive fact -- `polls_unmoved` equal to the game's poll count -- which is
    the one a consumer cannot see from a single row.
    """
    if "lines" not in store.tables(base):
        print("  odds staleness: no snapshot archive here")
        return 0
    got = _archive(season, base)
    if got.is_empty():
        print(f"  odds staleness: no {season} snapshots in the archive")
        return 0
    stale = staleness(got)
    latest = stale["captured_at"].max()
    per_game = (stale.group_by("game_id", "week")
                     .agg(polls=pl.len(),
                          unmoved=pl.col("polls_unmoved").last(),
                          since=pl.col("unmoved_since").last())
                     .with_columns((pl.col("polls") == pl.col("unmoved")).alias("frozen"),
                                   ((pl.lit(latest) - pl.col("since")).dt.total_hours()
                                    / 24.0).alias("days")))
    print(f"  odds staleness: {per_game.height} games, "
          f"{stale['captured_at'].n_unique()} polls, as of {latest}")
    print("  week  games  polls  frozen  median_days_unmoved")
    for r in (per_game.group_by("week")
                      .agg(games=pl.len(), polls=pl.col("polls").max(),
                           frozen=pl.col("frozen").sum(), days=pl.col("days").median())
                      .sort("week").iter_rows(named=True)):
        print(f"  {int(r['week']):>4}  {r['games']:>5}  {r['polls']:>5}  "
              f"{r['frozen']:>6}  {r['days']:>19.1f}")
    return 0


# --- the noise floor for line movement (#214) ----------------------------------
#
# How much a spread moves for no reason. **This is the ceiling computation for every
# line-movement question** (`docs/method.md` rule 8): #221 and whatever follows it claim
# that a line moved *on* something, and a move is only a move if it clears what the same
# line does between two polls with nothing to move on. So this is measured first, on the
# archive, on games whose starting quarterbacks did not change between the polls, and every
# later claim is compared against it rather than against zero.
#
# The cluster unit is the **game** (`docs/method.md` rule 3). Every interval between two
# polls of one game shares that game's teams, its week, its number and whatever the books
# think of it, so the intervals are not independent and the archive's eight polls of 272
# games are 272 observations of a process, not 1,900. Every standard error below is taken
# over games -- the per-day means by a cluster-robust sandwich, the floor's own interval by
# resampling games -- and the count of games is printed beside the count of intervals so a
# reader sees which one the precision came from.
#
# Polls on one Eastern date are one poll. The archive holds three polls within thirty-five
# minutes of each other on 2026-09-04 and two within an hour on 2026-08-27, all from fixes
# to the matcher rather than from any wish to sample the betting market at that cadence, and every
# one of those intervals is zero by construction. Counting them would report a floor near
# zero that is a fact about the polling and not about the line. So the last poll of each
# Eastern date stands for the date, and an interval is between consecutive poll *days*.

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# What one line-movement observation carries, in the order `line_moves` returns it.
MOVE_COLUMNS: dict[str, Any] = {
    "game_id": pl.Utf8, "week": pl.Int64, "from": pl.Datetime, "to": pl.Datetime,
    "days": pl.Float64, "day": pl.Utf8, "delta": pl.Float64, "frozen": pl.Boolean,
    "same_qb": pl.Boolean,
}

# How many game-resamples the floor's interval is drawn from.
NOISE_BOOTSTRAP = 2000


def _eastern(col: str) -> pl.Expr:
    """A naive-UTC moment as an Eastern one, which is the calendar the slate runs on."""
    return pl.col(col).dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")


def poll_days(polls: pl.DataFrame) -> pl.DataFrame:
    """One row per game per Eastern date: the last poll of that date, every column kept."""
    return (polls.sort("game_id", "captured_at")
                 .with_columns(_eastern("captured_at").dt.date().alias("poll_day"))
                 .group_by("game_id", "poll_day", maintain_order=True).last()
                 .drop("poll_day"))


def _starter_at(polls: pl.DataFrame, starters: pl.DataFrame, side: str) -> pl.DataFrame:
    """The quarterback listed first on `side`'s depth chart as of each poll, by as-of join."""
    chart = (starters.select(pl.col("dt"), pl.col("team").alias(side),
                             pl.col("qb").alias(f"{side}_qb"))
                     .sort("dt"))
    # Both sides are sorted on the line above and the one before it; the check polars
    # cannot run with `by` groups would only warn about what is already done.
    return (polls.sort("captured_at")
                 .join_asof(chart, left_on="captured_at", right_on="dt", by=side,
                            strategy="backward", check_sortedness=False)
                 .drop("dt"))


def line_moves(polls: pl.DataFrame, starters: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per consecutive pair of poll days of one game: what the spread did between.

    `polls` is the archive as `staleness` returns it -- `game_id`, `close_spread`,
    `captured_at`, `week`, `polls_unmoved` -- and `starters`, when given, is a depth chart
    reduced to `dt`, `team`, `qb`: who was listed first at quarterback for each team from
    each moment on. The teams are read off the nflverse game id, whose last two fields are
    the away and home abbreviations, so no schedule is fetched to measure the archive.

    Three columns are the study's and the rest are bookkeeping. `delta` is the change in
    the home spread from the earlier poll day to the later. `day` names the Eastern day of
    the week the later poll fell on, which is what "by day of week" means here: the move
    is observed at the later poll and attributed to it. `same_qb` is whether both teams'
    first-listed quarterback was the same at both polls -- null when no chart precedes a
    poll, which is not the same as a change and is reported apart -- and is null throughout
    when no chart was given.

    `frozen` is #210's whole-archive fact: the game's last poll had stood unmoved for every
    poll there was. It is a property of the game and rides on every one of its intervals,
    so a caller can exclude a frozen lookahead from the sample or report it separately
    without reading the staleness columns itself.
    """
    if polls.is_empty():
        return pl.DataFrame(schema=MOVE_COLUMNS)
    frozen = (polls.group_by("game_id")
                   .agg((pl.col("polls_unmoved").last() == pl.len()).alias("frozen")))
    days = (poll_days(polls)
            .with_columns(pl.col("game_id").str.split("_").list.get(2).alias("away"),
                          pl.col("game_id").str.split("_").list.get(3).alias("home")))
    if starters is not None:
        days = _starter_at(_starter_at(days, starters, "home"), starters, "away")
    else:
        days = days.with_columns(pl.lit(None, dtype=pl.Utf8).alias("home_qb"),
                                 pl.lit(None, dtype=pl.Utf8).alias("away_qb"))
    prev = {c: pl.col(c).shift(1).over("game_id") for c in
            ("captured_at", "close_spread", "home_qb", "away_qb")}
    same = ((pl.col("home_qb") == prev["home_qb"]) & (pl.col("away_qb") == prev["away_qb"]))
    # Every shifted column in one pass, *before* the first poll day of each game is dropped.
    # Shifting after the drop reads the second interval's earlier end as the first's, which
    # nulls the first interval of every game and misaligns the rest -- and it did, silently,
    # until the report's own "no chart before the poll" count named twelve intervals in an
    # archive every chart predates.
    return (days.sort("game_id", "captured_at")
                .with_columns(prev["captured_at"].alias("from"),
                              (pl.col("close_spread") - prev["close_spread"]).alias("delta"),
                              same.alias("same_qb"))
                .filter(pl.col("from").is_not_null())
                .with_columns(
                    pl.col("captured_at").alias("to"),
                    ((pl.col("captured_at") - pl.col("from")).dt.total_seconds()
                     / 86400.0).alias("days"),
                    _eastern("captured_at").dt.weekday().map_elements(
                        lambda d: DAYS[int(d) - 1], return_dtype=pl.Utf8).alias("day"))
                .join(frozen, on="game_id", how="left")
                .select(*(pl.col(c).cast(t) for c, t in MOVE_COLUMNS.items())))


def _num(v: Any) -> float:
    """A polars aggregate as a float; None -- an empty series -- as NaN rather than a crash."""
    return float(v) if v is not None else float("nan")


def _clustered_sd(delta: Any, game: Any, *, bootstrap: int,
                  seed: int) -> tuple[float, float, float]:
    """The pooled sd of `delta` and a percentile interval from resampling games.

    Per-game sufficient statistics rather than per-game arrays: a resample's sd is a
    function of the resampled count, sum and sum of squares, so two thousand draws over a
    few hundred games are a matrix product rather than a loop of concatenations.
    """
    import numpy as np
    frame = pl.DataFrame({"delta": delta, "game": game})
    per = (frame.group_by("game").agg(n=pl.len(), s1=pl.col("delta").sum(),
                                      s2=(pl.col("delta") ** 2).sum())
                .sort("game"))
    n, s1, s2 = (per[c].to_numpy().astype(float) for c in ("n", "s1", "s2"))
    total = n.sum()
    if total < 2:
        return float("nan"), float("nan"), float("nan")
    sd = math.sqrt(max((s2.sum() - s1.sum() ** 2 / total) / (total - 1.0), 0.0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(bootstrap, len(n)))
    bn, b1, b2 = n[idx].sum(axis=1), s1[idx].sum(axis=1), s2[idx].sum(axis=1)
    var = np.where(bn > 1, (b2 - b1 ** 2 / np.maximum(bn, 1)) / np.maximum(bn - 1, 1), 0.0)
    draws = np.sqrt(np.maximum(var, 0.0))
    return sd, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def noise_floor(moves: pl.DataFrame, *, bootstrap: int = NOISE_BOOTSTRAP,
                seed: int = 0) -> dict[str, Any]:
    """The floor over the intervals given, pooled and by day of week, clustered by game.

    The caller decides which intervals are in -- `noise_floor_report` drops frozen games
    and quarterback changes before calling this -- so that the same function measures the
    excluded sample too, and the two can be printed side by side.

    "By day of week" is the regression #214 asks for, written as what it is: the change in
    spread on a full set of day-of-week indicators and no intercept is the per-day mean,
    and its standard error under clustering by game is the cluster-robust sandwich for a
    means model -- the root of the summed squares of each game's summed residuals, over the
    day's interval count. Stated so nobody reaches for a regression package to reproduce
    seven averages.

    The pooled sd is the floor. Its interval resamples games, because the intervals of one
    game are not exchangeable with each other and are with another game's.
    """
    if moves.is_empty():
        return {"intervals": 0, "games": 0, "sd": float("nan"), "sd_lo": float("nan"),
                "sd_hi": float("nan"), "mean_abs": float("nan"), "moved": float("nan"),
                "sd_per_root_day": float("nan"), "by_day": []}
    sd, lo, hi = _clustered_sd(moves["delta"], moves["game_id"], bootstrap=bootstrap,
                               seed=seed)
    per_root_day, _, _ = _clustered_sd(moves["delta"] / moves["days"].sqrt(),
                                       moves["game_id"], bootstrap=1, seed=seed)
    by_day = []
    for name in DAYS:
        part = moves.filter(pl.col("day") == name)
        if part.is_empty():
            continue
        mean = _num(part["delta"].mean())
        summed = (part.with_columns((pl.col("delta") - mean).alias("r"))
                      .group_by("game_id").agg(pl.col("r").sum()))
        se = math.sqrt(_num((summed["r"] ** 2).sum())) / part.height
        by_day.append({"day": name, "intervals": part.height,
                       "games": part["game_id"].n_unique(),
                       "days_spanned": _num(part["days"].mean()),
                       "mean": mean, "se": se,
                       "sd": _num(part["delta"].std()) if part.height > 1 else float("nan"),
                       "moved": _num((part["delta"] != 0).mean()),
                       "max_abs": _num(part["delta"].abs().max())})
    return {"intervals": moves.height, "games": moves["game_id"].n_unique(),
            "sd": sd, "sd_lo": lo, "sd_hi": hi,
            "mean_abs": _num(moves["delta"].abs().mean()),
            "moved": _num((moves["delta"] != 0).mean()),
            "sd_per_root_day": per_root_day, "by_day": by_day}


def _qb_starters(season: int) -> pl.DataFrame:                  # pragma: no cover - network
    """Who is listed first at quarterback for each team, from each chart's moment on.

    nflverse's depth charts, reduced to the one position this study conditions on. `dt`
    is the chart's own timestamp, ISO 8601 in UTC, and comes back naive UTC to match
    `captured_at`. A team that changes its starter publishes a new chart, so the as-of join
    in `_starter_at` reads the change from the first chart that carries it.
    """
    import nflreadpy as nfl
    chart = nfl.load_depth_charts([season])
    return (chart.filter((pl.col("pos_abb") == "QB") & (pl.col("pos_rank") == 1))
                 .select(pl.col("dt").str.to_datetime("%Y-%m-%dT%H:%M:%SZ").alias("dt"),
                         pl.col("team"), pl.col("gsis_id").alias("qb"))
                 .sort("dt"))


def _print_floor(label: str, f: dict[str, Any]) -> None:
    print(f"  {label}: sd {f['sd']:.3f} [{f['sd_lo']:.3f}, {f['sd_hi']:.3f}] points "
          f"over {f['intervals']} intervals of {f['games']} games (cluster = game); "
          f"mean |move| {f['mean_abs']:.3f}, moved on {f['moved']:.1%} of intervals; "
          f"sd per root-day {f['sd_per_root_day']:.3f}")


def noise_floor_report(season: int = SEASON_AHEAD, base: Path | None = None, *,
                       starters: pl.DataFrame | None = None,
                       bootstrap: int = NOISE_BOOTSTRAP) -> int:
    """The numbers #214 asks for, printed: coverage, exclusions, floor, and by day of week.

    `starters` is injected by tests and fetched from nflverse otherwise. A chart that cannot
    be fetched does not take the floor down with it: the report says the same-quarterback
    condition could not be applied and measures the unconditioned sample under that label,
    which is the honest number a reader with no network can still get.
    """
    if "lines" not in store.tables(base):
        print("  line noise: no snapshot archive here")
        return 0
    polls = staleness(_archive(season, base))
    if polls.is_empty():
        print(f"  line noise: no {season} snapshots in the archive")
        return 0
    qb_note = ""
    if starters is None:
        try:
            starters = _qb_starters(season)
        except Exception as exc:                            # pragma: no cover - network
            qb_note = f" (same-quarterback condition NOT applied: {type(exc).__name__}: {exc})"
    moves = line_moves(polls, starters)
    days = poll_days(polls)
    print(f"  line noise: {polls['game_id'].n_unique()} games, "
          f"{polls['captured_at'].n_unique()} polls on {days['captured_at'].n_unique()} "
          f"days, weeks {polls['week'].min()}-{polls['week'].max()}, "
          f"{polls['captured_at'].min():%Y-%m-%d} to {polls['captured_at'].max():%Y-%m-%d}; "
          f"{moves.height} intervals between poll days{qb_note}")
    print("  This is the ceiling for every line-movement question (method rule 8): a move "
          "inside it is not a move.")

    frozen = moves.filter(pl.col("frozen"))
    live = moves.filter(~pl.col("frozen"))
    n_frozen = frozen["game_id"].n_unique()
    print(f"  frozen lookaheads excluded: {n_frozen} games whose number never moved across "
          f"the archive ({frozen.height} intervals); {live['game_id'].n_unique()} games kept. "
          f"Excluding a game for never moving conditions on the outcome, so the all-games "
          f"floor is printed beside it.")
    if starters is not None:
        changed = live.filter(pl.col("same_qb").fill_null(True).not_())
        unknown = live.filter(pl.col("same_qb").is_null())
        print(f"  quarterback changes excluded: {changed.height} intervals; "
              f"no chart before the poll: {unknown.height} intervals")
        live = live.filter(pl.col("same_qb"))
    _print_floor("floor", noise_floor(live, bootstrap=bootstrap))
    _print_floor("all games, frozen included", noise_floor(moves, bootstrap=bootstrap))
    print("  by day of week of the later poll (mean move +/- cluster-robust se by game):")
    print("  day  intervals  games  days_spanned     mean      se     sd   moved  max|move|")
    for d in noise_floor(live, bootstrap=1)["by_day"]:
        print(f"  {d['day']}  {d['intervals']:>9}  {d['games']:>5}  {d['days_spanned']:>12.1f}  "
              f"{d['mean']:>+7.3f}  {d['se']:>6.3f}  {d['sd']:>5.3f}  {d['moved']:>5.1%}  "
              f"{d['max_abs']:>9.1f}")
    absent = [n for n in DAYS if n not in set(live["day"].to_list())]
    if absent:
        print(f"  no poll closed on: {', '.join(absent)} -- the archive cannot speak to "
              f"those days yet")
    return 0


def snapshot(season: int = SEASON_AHEAD, *, markets: str = ",".join(MARKETS),
             regions: str = ",".join(REGIONS),
             state_path: Path | None = None, base: Path | None = None,
             floor: int = CREDIT_FLOOR, now: datetime | None = None) -> pl.DataFrame:
    """One pull, the declared markets, one region. Appends a dated line per matched game."""
    # GUARD only-budgeted-markets [unit/test_fetch_odds.py]: a market or region outside the
    # declared budget is refused before a request is formed, and so before a credit is spent
    asked_markets = _budgeted(markets, MARKETS, "markets")
    asked_regions = _budgeted(regions, REGIONS, "regions")
    # /GUARD
    # What markets x regions says this call costs. Carried down so the balance the response
    # reports can be checked against it, rather than the two being asserted to agree.
    declared_cost = len(asked_markets) * len(asked_regions)

    when = now or datetime.now(UTC).replace(tzinfo=None)
    have = credits_remaining(state_path)
    # GUARD credit-floor-refuses [unit/test_fetch_odds.py]: a low balance spends nothing
    if have is not None and have < floor:
        raise QuotaFloor(
            f"{have} credits left, floor is {floor}. Refusing before spending one.")
    # /GUARD

    key = _api_key()
    if not key:
        raise QuotaFloor("no ODDS_API_KEY set; cannot fetch")

    # The reach for the source, and the only part of this function that is one. Its errors
    # are the betting market's and are reported as such; everything below has answered.
    payload, headers = _http_get(
        {"markets": markets, "regions": regions, "oddsFormat": "american"}, key)

    try:
        return _record(payload, headers, season, when, state_path, base, floor,
                       before=have, declared_cost=declared_cost, asked=markets)
    except Exception as exc:
        raise SnapshotIncomplete(
            f"the betting market answered and a credit was spent, then "
            f"{type(exc).__name__}: {exc}. "
            f"Retrying spends another credit against a problem that is not the "
            f"betting market's."
        ) from exc


def _record(payload: Any, headers: Mapping[str, str], season: int, when: datetime,
            state_path: Path | None, base: Path | None, floor: int, *,
            before: int | None = None, declared_cost: int | None = None,
            asked: str | None = None) -> pl.DataFrame:
    """Everything after the betting market has answered: the credit, join, check and write.

    Split out so the guard above spans the reach for the source and nothing else. It is one
    function rather than four because the caller's question is binary -- did the betting
    market answer -- and "which of ours broke" is the exception it carries.
    """
    remaining_hdr = headers.get("x-requests-remaining")
    remaining = int(float(remaining_hdr)) if remaining_hdr is not None else None
    # What this poll cost, measured off the account rather than read off `declared_cost`.
    # A monthly reset makes the balance go *up*, and the difference is then not a cost, so
    # that case records nothing rather than a negative number.
    cost = (before - remaining
            if before is not None and remaining is not None and before >= remaining
            else None)
    _write_state(state_path, remaining, when, cost=cost, declared=declared_cost,
                 asked=asked)
    print(f"  odds snapshot: {len(payload or [])} events, "
          f"{remaining if remaining is not None else '?'} credits remaining, "
          f"this poll cost {cost if cost is not None else '?'} "
          f"(markets x regions says {declared_cost if declared_cost is not None else '?'})")
    # GUARD measured-cost-checked-against-declared [unit/test_fetch_odds.py]: the budget the
    # market guard rests on is compared with the account rather than assumed
    if cost is not None and declared_cost is not None and cost != declared_cost:
        print(f"  WARNING: the account was charged {cost} for a call markets x regions "
              f"budgets at {declared_cost}. The cost model `_budgeted` rests on is not the "
              f"one being billed -- check the account's pricing before the next poll.")
    # /GUARD
    if remaining is not None and remaining < floor:
        print(f"  WARNING: below the floor of {floor}; the next pull will refuse")

    index = _game_index(season)

    rows, no_game, no_line, no_total, no_price = [], 0, 0, 0, 0
    for ev in payload or []:
        home_name = ev.get("home_team", "")
        hit = _match_game(ev, index)
        spread, spread_price = _median_home_spread(ev, home_name)
        total, total_price = _median_game_total(ev)
        if not hit:
            no_game += 1
            continue
        # The spread still decides whether there is a row at all, and a game priced only in
        # the totals market is therefore still dropped and still counted below. That is the
        # pre-#211 boundary left where it was rather than widened by the side door: the
        # `lines` table is non-null on `close_spread` and every consumer in the repo reads
        # it, so a row with a total and no spread is a new shape for six modules to answer
        # for. Measured cost of keeping it: the events the count below reports, which for
        # the main markets of one book-covered slate is expected to be none, since a book
        # posting a total and no spread on an NFL game is not a thing books do.
        if spread is None:
            no_line += 1
            continue
        # Absent, never fabricated. A game with a spread and no posted total keeps a null
        # total rather than one derived from anything, because the two markets are separate
        # facts and a total inferred from a spread is the one number a totals hypothesis
        # must never be fitted on.
        if total is None:
            no_total += 1
        if spread_price is None or (total is not None and total_price is None):
            no_price += 1
        game_id, week = hit
        rows.append({"game_id": game_id, "close_spread": spread,
                     "spread_price": spread_price, "close_total": total,
                     "total_price": total_price,
                     "captured_at": when, "week": int(week)})

    # Counted apart. These used to share one tally reported as "no nflverse game for the
    # team/date pair", which asserted the first cause for both -- and the first cause was
    # the one that was actually broken, so the message was right by accident and would have
    # misdirected the next person the moment a bookmaker simply had not posted a line.
    if no_game:
        print(f"  {no_game} events with no nflverse game for the team/date pair")
    if no_line:
        print(f"  {no_line} events matched a game but had no posted spread")
    # Reported for the same reason as the two above, and it is the one a reader of a totals
    # result has to see: a column of nulls from a market nobody posted and one from a market
    # this module asked for wrongly are the same column, and only this line separates them.
    if no_total:
        print(f"  {no_total} games stored with a spread and no posted total")
    if no_price:
        print(f"  {no_price} games stored with a point and no American price beside it "
              f"-- prices outside [-100, +100] are what American odds are, so check that "
              f"oddsFormat=american is still being honoured")

    df = pl.DataFrame(rows, schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                    "spread_price": pl.Float64, "close_total": pl.Float64,
                                    "total_price": pl.Float64,
                                    "captured_at": pl.Datetime, "week": pl.Int64})
    # GUARD staleness-stamped-against-the-archive: a row says how long its quote has stood
    # still, measured against every poll already stored rather than asserted as new
    df = _with_staleness(df, season, base)
    # /GUARD
    # Every partition is checked before any is written. Interleaved, a contract failure on
    # week 3 left weeks 1 and 2 on disk carrying a fresh timestamp while the CLI reported
    # that nothing had been fetched -- a half-written snapshot the as-of join would read as
    # a whole one. Checking first costs one extra pass over a frame of at most a few hundred
    # rows and makes the write all-or-nothing (issue #119).
    parts = [(wk, df.filter(pl.col("week") == wk).drop("week"))
             for wk in sorted(set(df["week"].to_list()))]
    # Asserted on what is stored, which is where the boundary is. Validating the whole pull
    # instead would fail `min_rows` on a pull that matched nothing -- and a pull matching
    # nothing writes nothing, so there is no partition to be wrong about. The return is kept
    # rather than dropped so that a repair this contract may one day declare reaches the
    # partition actually written; today it declares none and this rebuilds the same frames.
    parts = [(wk, ODDS_SNAPSHOT.validate(part)) for wk, part in parts]
    for wk, part in parts:
        # Snapshots append. A fixed name would overwrite the morning's line with the
        # afternoon's and leave the as-of join nothing to resolve.
        store.write(part, "lines", "nfl", season, wk, base=base,
                    name=f"snap-{when:%Y%m%dT%H%M%S}")
    return df.drop("week")


_GameIndex = tuple[dict[str, str], dict[tuple[str | None, str | None, str], tuple[str, int]]]


def _game_index(season: int) -> _GameIndex:
    """The season's team abbreviations and its (home, away, date) -> (game_id, week) map.

    One reach for the schedule, shared by the spread recorder and the props recorder, so an
    event is mapped to a nflverse game the same way whichever market it carries.
    """
    sched = _schedule(season)
    # The season's own abbreviations resolve a team that nflverse lists under two of them.
    in_play = set(sched["home_team"].to_list()) | set(sched["away_team"].to_list())
    abbrs = _team_abbrs(in_play)
    lookup = {
        (r["home_team"], r["away_team"], str(r["gameday"])): (r["game_id"], int(r["week"]))
        for r in sched.iter_rows(named=True)
    }
    return abbrs, lookup


def _match_game(ev: Mapping[str, Any], index: _GameIndex) -> tuple[str, int] | None:
    abbrs, lookup = index
    home = abbrs.get(ev.get("home_team", ""))
    away = abbrs.get(ev.get("away_team", ""))
    return lookup.get((home, away, _game_date(ev.get("commence_time", ""))))


# --- player props (#217): the recording half, with no pull behind it -------------
#
# The markets a props payload may carry, by The Odds API's names, and the whole of what
# `prop_quotes` reads. **None of these is in `MARKETS` and none may be added there.** A
# props market is priced per *event* rather than per season -- `docs/decisions.md` measured
# about four credits an event and 64 a week for a full slate -- so `_budgeted` refuses every
# name below and this module holds no request that asks for one. What it holds is the half
# after the betting market has answered: a payload someone has already paid for, parsed to
# one quote per (game, player, market) and written to `prop_lines` with its staleness, so the
# scorecard in `hub.models.props` has an archive to read the day a pull is authorised.
#
# Two sides per quote. A yardage or count prop is an Over and an Under on one point; the
# anytime-touchdown market is a Yes and sometimes a No on no point. `over_price` carries the
# Over or the Yes, `under_price` the Under or the No, and `point` is null for the latter.
PROP_MARKETS = ("player_pass_yds", "player_rush_yds", "player_reception_yds",
                "player_receptions", "player_rush_attempts", "player_pass_tds",
                "player_anytime_td")
_OVER_SIDES = {"over", "yes"}
_UNDER_SIDES = {"under", "no"}
_POINTLESS = ("player_anytime_td",)

_PROP_KEY = ("game_id", "player_key", "market")
_PROP_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "player_key": pl.Utf8, "player": pl.Utf8, "market": pl.Utf8,
    "point": pl.Float64, "over_price": pl.Float64, "under_price": pl.Float64,
    "captured_at": pl.Datetime, "week": pl.Int64,
}


def prop_quotes(event: Mapping[str, Any], game_id: str, week: int,
                when: datetime) -> list[dict[str, Any]]:
    """Every player prop in one event's payload, one row per (player, market), across books.

    The shape is the one The Odds API documents for its per-event endpoint: each outcome
    carries the side in `name` ("Over"/"Under", or "Yes"/"No"), the player in `description`,
    the point in `point` and the price in `price`. Written from the documentation and never
    run against a live response, which `PROP_SNAPSHOT.verified_against_live` says.

    Books are combined the way the spread is -- the median point, and the median price on
    each side in decimal space -- and for the same reason: one book hanging an outlier must
    not become the quote of record. A book that posts a point and no readable price still
    moves the point. A market that needs a point and has none from any book is skipped
    rather than stored, since a price on nothing is not a quote; the anytime market is the
    one that legitimately has none and is stored with a null point.
    """
    points: dict[tuple[str, str], list[float]] = {}
    overs: dict[tuple[str, str], list[float]] = {}
    unders: dict[tuple[str, str], list[float]] = {}
    names: dict[str, str] = {}
    for book in event.get("bookmakers") or []:
        for m in book.get("markets") or []:
            market = str(m.get("key") or "")
            if market not in PROP_MARKETS:
                continue
            for outcome in m.get("outcomes") or []:
                name = str(outcome.get("description") or "")
                if not name:
                    continue
                key = (player_key(name), market)
                names.setdefault(key[0], name)
                side = str(outcome.get("name") or "").lower()
                if side not in _OVER_SIDES and side not in _UNDER_SIDES:
                    continue
                if outcome.get("point") is not None:
                    points.setdefault(key, []).append(float(outcome["point"]))
                price = _american(outcome.get("price"))
                if price is not None:
                    (overs if side in _OVER_SIDES else unders).setdefault(key, []).append(price)
    rows = []
    for key in sorted(set(points) | set(overs) | set(unders)):
        pk, market = key
        point = statistics.median(points[key]) if key in points else None
        if point is None and market not in _POINTLESS:
            continue
        over = _median_price(overs.get(key, []))
        if over is None:
            continue
        rows.append({"game_id": game_id, "player_key": pk, "player": names[pk],
                     "market": market, "point": point, "over_price": over,
                     "under_price": _median_price(unders.get(key, [])),
                     "captured_at": when, "week": week})
    return rows


def _prop_archive(season: int, base: Path | None) -> pl.DataFrame:
    """Every prop poll already stored for the season, in `_PROP_SCHEMA`. Empty on a fresh clone."""
    if "prop_lines" not in store.tables(base):
        return pl.DataFrame(schema=_PROP_SCHEMA)
    got = store.sql("SELECT * FROM prop_lines WHERE league = 'nfl' AND season = ?",
                    params=[season], base=base)
    return got.select(*[pl.col(c).cast(t) for c, t in _PROP_SCHEMA.items()])


def record_props(payload: Sequence[Mapping[str, Any]], season: int, when: datetime, *,
                 base: Path | None = None) -> pl.DataFrame:
    """Write the props in a payload already paid for to `prop_lines`, with staleness.

    `payload` is a list of per-event responses -- the shape the per-event endpoint returns,
    one object per game -- and this function is deliberately the whole of what this module
    does with props. There is no reach for the source in front of it: the only way a prop
    quote enters the archive is a payload handed in, which is what keeps the quota decision
    a person's rather than a poll's.

    Everything else is the spread recorder's discipline. Events are matched to nflverse games
    by the same index, staleness is stamped against every poll already stored by the same
    `staleness`, every partition is checked before any is written, and the write appends
    under a dated name so two polls of one prop are two rows.
    """
    index = _game_index(season)
    rows: list[dict[str, Any]] = []
    no_game = 0
    for ev in payload or []:
        hit = _match_game(ev, index)
        if not hit:
            no_game += 1
            continue
        game_id, week = hit
        rows += prop_quotes(ev, game_id, week, when)
    if no_game:
        print(f"  {no_game} events with no nflverse game for the team/date pair")
    new = pl.DataFrame(rows, schema=_PROP_SCHEMA)
    print(f"  props recorded: {new.height} quotes on "
          f"{new.select('game_id', 'player_key').n_unique() if new.height else 0} players")
    if new.is_empty():
        return new.drop("week")
    # GUARD prop-staleness-stamped-against-the-archive: a prop row says how long its quote
    # has stood still, measured against every poll of that prop already stored
    prior = _prop_archive(season, base).drop("week")
    both = pl.concat([prior, new.drop("week")], how="vertical_relaxed")
    stamped = (staleness(both, key=_PROP_KEY, point="point", price="over_price")
               .select(*_PROP_KEY, "captured_at", *STALENESS_COLUMNS))
    df = new.join(stamped, on=[*_PROP_KEY, "captured_at"], how="left")
    # /GUARD
    parts = [(wk, df.filter(pl.col("week") == wk).drop("week"))
             for wk in sorted(set(df["week"].to_list()))]
    parts = [(wk, PROP_SNAPSHOT.validate(part)) for wk, part in parts]
    for wk, part in parts:
        store.write(part, "prop_lines", "nfl", season, wk, base=base,
                    name=f"snap-{when:%Y%m%dT%H%M%S}")
    return df.drop("week")


def credits_report(path: Path | None = None) -> int:
    have = credits_remaining(path)
    state = _read_state(path)
    if have is None:
        print("  odds credits: unknown (no pull recorded yet)")
    else:
        print(f"  odds credits: {have:,} remaining, floor {CREDIT_FLOOR} "
              f"(as of {state.get('checked_at', '?')})")
    if not _api_key():
        print("  no ODDS_API_KEY set; fetching unavailable, accounting still works")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.odds",
        description=f"Markets {','.join(MARKETS)} in region {','.join(REGIONS)}, one pull. "
                    f"Cost is markets x regions, so {len(MARKETS) * len(REGIONS)} credits.")
    ap.add_argument("--snapshot", action="store_true", help="take one dated line snapshot")
    ap.add_argument("--credits", action="store_true", help="report the stored balance")
    ap.add_argument("--staleness", action="store_true",
                    help="per week, how much of the archive is a number that never moved; "
                         "reads the store and spends nothing")
    ap.add_argument("--noise-floor", action="store_true",
                    help="how much a spread moves between polls for no reason, by day of "
                         "week, on same-quarterback games; the ceiling every line-movement "
                         "claim is measured against. Reads the store and spends nothing")
    ap.add_argument("--record-props", metavar="PAYLOAD", default=None,
                    help="write a player-props payload already paid for (a JSON list of "
                         "per-event responses) to the prop_lines archive with its "
                         f"staleness. Reads a file and spends nothing; markets "
                         f"{','.join(PROP_MARKETS)}")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--state-path", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--base", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    spath = Path(a.state_path) if a.state_path else None
    base = Path(a.base) if a.base else None

    if a.record_props:
        try:
            payload = json.loads(Path(a.record_props).read_text())
            record_props(payload, a.season, datetime.now(UTC).replace(tzinfo=None), base=base)
        except Exception as e:
            return unavailable("hub.fetch.odds", f"the props payload {a.record_props}", e)
        return 0
    if a.staleness:
        return staleness_report(a.season, base=base)
    if a.noise_floor:
        return noise_floor_report(a.season, base=base)
    if a.credits or not a.snapshot:
        return credits_report(spath)

    if not _api_key():
        print("hub.fetch.odds: no ODDS_API_KEY set; add one to .env to fetch. "
              "Balance accounting still works via --credits.", file=sys.stderr)
        return 1
    try:
        snapshot(a.season, state_path=spath)
    except (QuotaFloor, MultiplierRefused, SnapshotIncomplete) as e:
        # The first two are this repo refusing to spend; the third is this repo failing
        # after it already has. None is the betting market being unreachable, and saying so
        # names the one source that worked -- then sends the operator to retry, which spends
        # another credit from a metered monthly quota (issue #119).
        print(f"hub.fetch.odds: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # What is left is the reach for the betting market itself, and a snapshot that
        # taken must not take the slate down with it -- `make slate` puts a leading `-` on
        # this line for the same reason.
        return unavailable("hub.fetch.odds", "the betting market's prices", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
