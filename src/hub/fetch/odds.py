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
on 2026-09-11 is mostly that -- 259 of 272 games never moved across the whole of it, and
every week from 3 out had at most one game move. So each row carries `polls_unmoved` and
`unmoved_since`: how many consecutive polls have returned this quote and when the run began.
`staleness` derives both from the archive, `_record` stamps them on the rows it writes, and
a consumer ranking a snapshot above the schedule's own field has the number it needs to stop
doing that for a quote nothing has touched in a fortnight.

    uv run python -m hub.fetch.odds --credits
    uv run python -m hub.fetch.odds --snapshot
    uv run python -m hub.fetch.odds --staleness
"""
from __future__ import annotations

import argparse
import json
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
from hub.contracts import ODDS_SNAPSHOT
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


def _quote_moved() -> pl.Expr:
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
    """
    prev_point = pl.col("close_spread").shift(1).over("game_id")
    prev_price = pl.col("spread_price").shift(1).over("game_id")
    point_moved = pl.col("close_spread") != prev_point
    price_moved = (pl.col("spread_price").is_not_null() & prev_price.is_not_null()
                   & (pl.col("spread_price") != prev_price))
    return (prev_point.is_null() | point_moved | price_moved).alias("_moved")


def staleness(lines: pl.DataFrame) -> pl.DataFrame:
    """Every poll, with how long its spread quote had stood still by then.

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
    frame = lines if "spread_price" in lines.columns else lines.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("spread_price"))
    ordered = frame.sort("game_id", "captured_at")
    run = pl.col("_moved").cast(pl.Int64).cum_sum().over("game_id").alias("_run")
    return (ordered.with_columns(_quote_moved())
                   .with_columns(run)
                   .with_columns(
                       pl.col("captured_at").cum_count().over("game_id", "_run")
                         .cast(pl.Int64).alias("polls_unmoved"),
                       pl.col("captured_at").min().over("game_id", "_run")
                         .alias("unmoved_since"))
                   .drop("_moved", "_run")
                   .pipe(lambda d: d if "spread_price" in lines.columns
                         else d.drop("spread_price")))


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

    sched = _schedule(season)
    # The season's own abbreviations resolve a team that nflverse lists under two of them.
    in_play = set(sched["home_team"].to_list()) | set(sched["away_team"].to_list())
    abbrs = _team_abbrs(in_play)
    lookup = {
        (r["home_team"], r["away_team"], str(r["gameday"])): (r["game_id"], r["week"])
        for r in sched.iter_rows(named=True)
    }

    rows, no_game, no_line, no_total, no_price = [], 0, 0, 0, 0
    for ev in payload or []:
        home_name = ev.get("home_team", "")
        home = abbrs.get(home_name)
        away = abbrs.get(ev.get("away_team", ""))
        hit = lookup.get((home, away, _game_date(ev.get("commence_time", ""))))
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
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--state-path", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--base", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    spath = Path(a.state_path) if a.state_path else None

    if a.staleness:
        return staleness_report(a.season, base=Path(a.base) if a.base else None)
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
