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
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

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
PROVENANCE: dict[str, Provenance] = {
    "snapshot": Provenance(
        reader_can_obtain=False,
        why=("the capture is dated and immutable, and it is not published: `.gitignore` "
             "excludes the processed store as redistributed third-party data the repo "
             "cannot publish. The number used is in the artifact; the source it came from "
             "is not something a reader can open.")),
    "schedule": Provenance(
        reader_can_obtain=False,
        why=("the source is public, but the value has moved. nflverse keeps one current "
             "`spread_line` per game and no history, so the lookahead number this was "
             "priced from cannot be fetched back -- which is why #6 stopped pricing from "
             "it where a snapshot exists.")),
}


def provenance(source: str) -> Provenance:
    """How obtainable the input behind a price source is. Unknown sources raise.

    Loudly, and on purpose. A default would let a fourth source ship carrying whichever
    answer was convenient, and the artifact would go back to claiming a verifiability it does
    not have -- which is the defect this exists to close.
    """
    if source not in PROVENANCE:
        raise KeyError(
            f"{source!r} can price a prediction and is not classified. Say whether a reader "
            f"can obtain it, in `schedule.PROVENANCE`.")
    return PROVENANCE[source]
