"""The season's games, with the market's number on each and where it came from.

One rule, with two readers. `hub.models.ratings` prices a weekly prediction from this and
`hub.season.survivor` prices a whole season of survivor picks, and the two must not be able
to disagree about the same game. They did, for a day: the weekly prediction moved onto the
dated snapshots (issue #6) and survivor was left reading nflverse's own field, so survivor
planned twelve of eighteen weeks and called the rest unpriced while the store held every
game of the season. `docs/next.md` names a second implementation of one idea as how they
drift; this is the first one.

**What a reader can follow is a separate question, and the answer today is neither source.**
A snapshot is dated and immutable and is *not published* -- `.gitignore` excludes the processed
store as redistributed third-party data. The moving field is published and has *moved*, so the
lookahead value a prediction was priced from cannot be fetched back. So `price_source` and
`priced_at` are a complete citation on the machine holding the store and an unfollowable one
anywhere else, which is what `PROVENANCE` below says out loud and what the weekly artifact
carries. The number used is published either way; what is missing is corroboration.

**Both columns quote the same quantity** -- the betting market's spread on the home team,
positive when the home team is favoured -- so choosing between them is provenance rather
than accuracy, and `tests/golden/test_line_agreement.py` carries the evidence that they
agree where both exist.

They are not interchangeable as a *record*, which is the point. `spread_line` is a lookahead
number that moves as the week runs and that upstream leaves empty for weeks that are far
away -- 16 of 16 populated for week 1 of this season and 0 of 16 for week 18. A prediction
priced from it cannot be shown afterwards, only asserted, because the field it came from has
since changed. `hub.fetch.odds` writes dated, immutable snapshots and `store.lines_as_of`
picks the one that was live at a given moment.

The moving field stays as the fallback. **Not for the reason issue #6 gave**: that ticket
expected far weeks to carry no snapshot when a fit first runs, and measured on 2026-09-04 the
opposite holds -- The Odds API is already posting week 18. What the fallback is actually for
is a store with no snapshot for a game: a fresh clone, a poller that has been down since the
last schedule refresh, or a game the pull did not return. Dropping those games would shrink
the slate rather than admit what priced it.

**And since #281, a snapshot outranks the moving field only while it is a live price.** #210
measured the archive: every week from 2 out had returned the identical quote at every poll
for the full twelve days it spanned. A quote no book has touched across a slate's worth of
news is a posted lookahead, and ranking it above a field upstream still refreshes labelled
it as though it were priced. So `price_source` is three-way -- `live`, `stale`, `schedule`
-- decided from the staleness field by the one cut `hub.models.quarterback.live_price`
declares: a live snapshot prices the game; a stale one yields to the moving field where
that field has a number and prices the game, labelled stale, where it does not.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import polars as pl

from hub import store
from hub.fetch import nflverse, odds
from hub.models import quarterback

ET = ZoneInfo("America/New_York")


def _kickoff() -> pl.Expr:
    """Kickoff as a naive UTC moment, from nflverse's Eastern date and time.

    Every other moment in this repo -- `captured_at`, `predicted_at`, the as-of `at` -- is
    naive UTC, and `gameday`/`gametime` are Eastern. Comparing the two directly is a four-hour
    error that looks like nothing, which this repo has already shipped once today: the first
    coverage measurement against the as-of join asked its question in local time and silently
    hid a morning's snapshots.

    Converted rather than offset by a constant, for the reason `hub.fetch.odds._game_date`
    records: the season crosses out of daylight saving in November, so a fixed -4 would be
    right through week 9 and wrong after it.
    """
    stamp = (pl.col("gameday").cast(pl.Utf8) + " " + pl.col("gametime").cast(pl.Utf8))
    return (stamp.str.to_datetime("%Y-%m-%d %H:%M", strict=False)
                 .dt.replace_time_zone("America/New_York", non_existent="null",
                                       ambiguous="earliest")
                 .dt.convert_time_zone("UTC")
                 .dt.replace_time_zone(None)
                 .alias("kickoff"))


class Slate(NamedTuple):
    """One league's scheduled games, and the league the rows actually came from.

    The name travels *with* the rows because it is a fact about the source that produced
    them and not about the argument that asked for them. Issue #174 is what the other
    arrangement costs: this module fetched nflverse's NFL season whatever league it was
    asked for and then stamped the asked-for name onto the result, so a caller asking for
    college was handed NFL games labelled `cfb`. Nothing downstream re-derives that label --
    `store.write` partitions on it and every reader treats it as true -- which makes a wrong
    one worse than a refusal.
    """
    games: pl.DataFrame
    league: str


class LeagueUnavailable(Exception):
    """Nothing here can produce this league's games, and the refusal names it.

    Named on purpose. A silent substitution is the defect; an exception carrying the league
    that was asked for is a caller's only way to tell "no college schedule" from "no games
    this week".
    """


def _nfl_slate(season: int, cache: Path | None) -> Slate:
    """The NFL season from nflverse, in this module's columns.

    The literal below belongs to this loader: it names where these rows came from, which is
    the one thing that can make the column true. `priced_games` never writes it, and the
    same string is what the frame is filed under, so the two cannot drift apart.
    """
    league = "nfl"
    sched = nflverse.load("schedules", seasons=[season], cache=cache)
    return Slate(sched.select(
        pl.col("game_id"),
        pl.lit(league).alias("league"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("home_team"),
        pl.col("away_team"),
        pl.col("spread_line").cast(pl.Float64).alias("schedule_spread"),
        pl.col("result"),
        # When it starts, which is when it stops being forecastable. A schedule without times
        # carries a null rather than a guess -- an invented kickoff would silently decide
        # whether a game may still be predicted.
        (_kickoff() if {"gameday", "gametime"} <= set(sched.columns)
         else pl.lit(None, dtype=pl.Datetime).alias("kickoff")),
    ), league)


# League to loader. A league is here when something in this repo can actually produce its
# games, and `cfb` is absent because nothing can. `hub.fetch.cfbd` pulls college games, but
# no price reaches them: `hub.fetch.odds` polls the NFL alone, so `store.lines_as_of` has
# never had a `league=cfb` row written for it to find, and CFBD's own lines endpoint is
# per-week -- a season of them is fifteen calls against a run ceiling of twelve. A key added
# here before those exist is exactly how the NFL season came to be served under a college
# name; the refusal below says which of them is missing.
SLATES: dict[str, Callable[[int, Path | None], Slate]] = {"nfl": _nfl_slate}


def priced_games(season: int, *, at: datetime | None = None, cache: Path | None = None,
                 base: Path | None = None, league: str = "nfl") -> pl.DataFrame:
    """Every scheduled game, priced from the dated snapshot where a live one exists.

    Carries `close_spread` -- the number to use -- alongside the two candidates it was
    chosen from, plus `price_source` naming which one won and `priced_at` naming the
    snapshot that did it, null where the moving field did. A game neither source prices
    keeps a null `close_spread` and a null source: absent from the plan rather than guessed
    at.

    `price_source` is `live` (a snapshot whose quote has stood within
    `hub.models.quarterback.STALE_AFTER_DAYS` at `at`), `schedule` (the moving field: no
    snapshot, or a snapshot gone stale), or `stale` (a stale snapshot and nothing else to
    price the game from -- used, and said so). The staleness columns `polls_unmoved` and
    `unmoved_since` describe the snapshot candidate whichever way it went, so a row the
    moving field priced over a frozen snapshot still shows how long that quote had stood.

    `at` defaults to now, in UTC and naive, which is how `hub.fetch.odds` stamps
    `captured_at`. Passing a *local* `datetime.now()` silently asks the as-of question hours
    in the past and hides the morning's snapshots -- it did, in the first measurement taken
    against this rule, and the coverage it reported looked entirely plausible.

    `league` selects a loader in `SLATES` and is never written onto the rows. A league no
    loader can produce raises `LeagueUnavailable` naming it, which is issue #174: this used
    to fetch the NFL season for every league and label it with whatever was asked for, so a
    college caller received NFL games under a college name and no reader could tell. The
    NFL path is the one it always was -- same source, same columns, same order.
    """
    build = SLATES.get(league)
    # GUARD unserved-league-is-refused: a league nothing can produce raises, saying which
    if build is None:
        raise LeagueUnavailable(
            f"no schedule here for league {league!r}; this module can produce "
            f"{sorted(SLATES)}. Refused rather than answered with the NFL season under a "
            f"{league!r} name -- which every reader would have believed, and which is the "
            f"whole of issue #174. For college: `hub.fetch.cfbd` fetches the games, and "
            f"nothing yet prices them.")
    # /GUARD
    slate = build(season, cache)
    # GUARD league-is-the-sources-own: rows filed under a name they disagree with are refused
    #
    # The mechanism rather than the promise. `_nfl_slate` writing its own name into the
    # column is what makes the column true; this is what keeps the *filing* honest, so a
    # future entry that pointed `cfb` at an NFL loader would be caught here instead of
    # reproducing #174 one registry line later.
    if slate.league != league:
        raise LeagueUnavailable(
            f"asked for league {league!r} and the loader filed under it returned "
            f"{slate.league!r} games. The league column names the source its rows came "
            f"from, so this frame cannot be served as {league!r}.")
    # /GUARD
    games = slate.games
    moment = at or datetime.now(UTC).replace(tzinfo=None)
    snaps = store.lines_as_of(moment, season, slate.league, base=base).rename(
        {"close_spread": "snapshot_spread", "captured_at": "priced_at"})
    # How long the snapshot's quote had stood still at `moment` (#210), carried onto the row
    # for the consumer that has to say at what age a quote stops being a live price --
    # `hub.models.quarterback` since #218. Joined on the capture that priced the row, so
    # the two columns describe that poll and not a later one. Null on a row the moving
    # field priced: nothing polls that field, so there is no run to measure.
    stale = odds.staleness_as_of(moment, season, base)
    snaps = snaps.join(stale, left_on=["game_id", "priced_at"],
                       right_on=["game_id", "captured_at"], how="left")
    # The cut is the quarterback layer's, applied here and read there (#281). The branch
    # order is the ranking: a live snapshot first, then the moving field, then a stale
    # snapshot only where the moving field has nothing -- which is what stops a frozen
    # lookahead outranking a field upstream still refreshes.
    snapshot = pl.col("snapshot_spread").is_not_null()
    moving = pl.col("schedule_spread").is_not_null()
    source = (pl.when(snapshot & quarterback.live_price(moment)).then(pl.lit("live"))
                .when(moving).then(pl.lit("schedule"))
                .when(snapshot).then(pl.lit("stale"))
                .otherwise(None))
    from_snapshot = pl.col("price_source").is_in(["live", "stale"])
    return (games.join(snaps, on="game_id", how="left")
                 .with_columns(source.alias("price_source"))
                 .with_columns(
                     pl.when(from_snapshot).then(pl.col("snapshot_spread"))
                       .when(pl.col("price_source") == "schedule")
                       .then(pl.col("schedule_spread"))
                       .otherwise(None).alias("close_spread"),
                     # The citation names the snapshot that priced the row and no other.
                     pl.when(from_snapshot).then(pl.col("priced_at")).otherwise(None)
                       .alias("priced_at")))


def forecastable(games: pl.DataFrame, at: datetime | None = None) -> pl.DataFrame:
    """The games still ahead of us: kickoff not yet passed, no result yet.

    Here rather than in one of its readers, because `kickoff` and `result` are this module's
    columns and the rule is the same rule for both of them. `hub.models.ratings` needs it for
    `docs/track-record.md` rule 1 -- a prediction counts only if it was committed before
    kickoff, and a Sunday-morning run would otherwise publish a prediction for Thursday
    night's finished game. `hub.season.survivor` needs it because a plan that spends teams on
    weeks already over is drawing every remaining pick from a pool degraded by picks that were
    never available. Two readers of one rule is what this module exists for, and the last time
    they each had their own copy they disagreed for a day.

    Both tests, because neither covers the other. A finished game has a result; a game *in
    progress* does not, and it is no more forecastable. A game whose kickoff is unknown is
    kept rather than dropped -- an invented time would silently decide this, and the result
    check still catches it once the game is over.
    """
    moment = at or datetime.now(UTC).replace(tzinfo=None)
    started = (pl.col("kickoff").is_not_null() & (pl.col("kickoff") <= moment)
               if "kickoff" in games.columns else pl.lit(False))
    done = pl.col("result").is_not_null() if "result" in games.columns else pl.lit(False)
    return games.filter(~(started | done))


def by_source(games: pl.DataFrame) -> dict[str, int]:
    """How many games each source priced.

    Not `coverage`, which `hub.season.survivor` already uses for a different question --
    which *weeks* the market has priced at all. The two sit side by side in survivor's own
    CLI, which is exactly where one word for two ideas would be read wrong.

    Reported by every caller rather than measured once. The fallback's share is what tells
    you whether the snapshot poller has been running, and a share that quietly climbs back
    to 100% is what a dead poller looks like from inside a model -- visible in the run's own
    output, not only in the watchdog.
    """
    src = games["price_source"]
    return {"live": int((src == "live").sum()),
            "stale": int((src == "stale").sum()),
            "schedule": int((src == "schedule").sum()),
            "unpriced": int(src.null_count())}


# --- what a reader can obtain -------------------------------------------------

class Provenance(NamedTuple):
    """Whether a third party can fetch the input a prediction was priced from, and why not.

    Not whether *we* can. `priced_at` names the snapshot on this machine and that citation is
    complete here; the question a public record has to answer is whether anyone else can
    follow it.
    """
    reader_can_obtain: bool
    why: str


# Neither source is obtainable by a reader today, and the reasons are opposites -- which is
# the useful part, because they have different futures. A snapshot is immutable and could be
# published in some derived form; the moving field is published and its value at our capture
# moment is simply gone.
#
# Stated here rather than in the site writer because this module owns `price_source`, and a
# classification that lives away from the thing it classifies is one that stops matching it.
_SNAPSHOT_WHY = ("the capture is dated and immutable, and it is not published: `.gitignore` "
                 "excludes the processed store as redistributed third-party data the repo "
                 "cannot publish. The number used is in the artifact; the source it came from "
                 "is not something a reader can open.")

PROVENANCE: dict[str, Provenance] = {
    "live": Provenance(reader_can_obtain=False, why=_SNAPSHOT_WHY),
    "stale": Provenance(
        reader_can_obtain=False,
        why=(_SNAPSHOT_WHY + " And this quote had stood unmoved past the cut that makes a "
             "snapshot a live price; it priced the game because the moving field had no "
             "number for it (#281).")),
    "schedule": Provenance(
        reader_can_obtain=False,
        why=("the source is public, but the value has moved. nflverse keeps one current "
             "`spread_line` per game and no history, so the lookahead number this was "
             "priced from cannot be fetched back -- which is why #6 stopped pricing from "
             "it where a snapshot exists.")),
    # The name every week published before #281 carries. `hub.publish` classifies every
    # published week on every run, and an old artifact must not take the page down.
    "snapshot": Provenance(
        reader_can_obtain=False,
        why=(_SNAPSHOT_WHY + " Written before #281 split the label into live and stale, so "
             "whether this quote was a live price at the time is not on the row.")),
}


def provenance(source: str) -> Provenance:
    """How obtainable the input behind a price source is. Unknown sources raise.

    Loudly, and on purpose. A default would let a fourth source ship carrying whichever
    answer was convenient, and the artifact would go back to claiming a verifiability it does
    not have -- which is the defect this exists to close.
    """
    # GUARD unclassified-source-raises [unit/test_provenance.py]: a fourth source is refused
    if source not in PROVENANCE:
        raise KeyError(
            f"{source!r} can price a prediction and is not classified. Say whether a reader "
            f"can obtain it, in `schedule.PROVENANCE`.")
    # /GUARD
    return PROVENANCE[source]
