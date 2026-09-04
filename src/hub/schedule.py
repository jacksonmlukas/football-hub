"""The season's games, with the market's number on each and where it came from.

One rule, with two readers. `hub.models.ratings` prices a weekly prediction from this and
`hub.season.survivor` prices a whole season of survivor picks, and the two must not be able
to disagree about the same game. They did, for a day: the weekly prediction moved onto the
dated snapshots (issue #6) and survivor was left reading nflverse's own field, so survivor
planned twelve of eighteen weeks and called the rest unpriced while the store held every
game of the season. `docs/next.md` names a second implementation of one idea as how they
drift; this is the first one.

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
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from hub import store
from hub.fetch import nflverse


def priced_games(season: int, *, at: datetime | None = None, cache: Path | None = None,
                 base: Path | None = None, league: str = "nfl") -> pl.DataFrame:
    """Every scheduled game, priced from the dated snapshot where one exists.

    Carries `close_spread` -- the number to use -- alongside the two candidates it was
    chosen from, plus `price_source` naming which one won and `priced_at` naming the
    snapshot that did it. A game neither source prices keeps a null `close_spread` and a
    null source: absent from the plan rather than guessed at.

    `at` defaults to now, in UTC and naive, which is how `hub.fetch.odds` stamps
    `captured_at`. Passing a *local* `datetime.now()` silently asks the as-of question hours
    in the past and hides the morning's snapshots -- it did, in the first measurement taken
    against this rule, and the coverage it reported looked entirely plausible.
    """
    sched = nflverse.load("schedules", seasons=[season], cache=cache)
    games = sched.select(
        pl.col("game_id"),
        pl.lit(league).alias("league"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("home_team"),
        pl.col("away_team"),
        pl.col("spread_line").cast(pl.Float64).alias("schedule_spread"),
        pl.col("result"),
    )
    moment = at or datetime.now(UTC).replace(tzinfo=None)
    snaps = store.lines_as_of(moment, season, league, base=base).rename(
        {"close_spread": "snapshot_spread", "captured_at": "priced_at"})
    return (games.join(snaps, on="game_id", how="left")
                 .with_columns(
                     pl.coalesce("snapshot_spread", "schedule_spread").alias("close_spread"),
                     pl.when(pl.col("snapshot_spread").is_not_null()).then(pl.lit("snapshot"))
                       .when(pl.col("schedule_spread").is_not_null()).then(pl.lit("schedule"))
                       .otherwise(None).alias("price_source")))


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
    return {"snapshot": int((src == "snapshot").sum()),
            "schedule": int((src == "schedule").sum()),
            "unpriced": int(src.null_count())}
