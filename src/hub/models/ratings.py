"""Ratings: currently a passthrough that returns the market prior unchanged.

This is a placeholder on purpose, and the plan says why: naive-but-real beats
sophisticated-but-absent. `Makefile:12` has always called `hub.models.ratings --fit`, and
until now that line dead-ended, which meant `make slate` could not run end to end and no
part of the weekly pipeline had ever been exercised together. A module that returns the
market prior makes the pipeline real, and makes every later improvement a diff against
something that works rather than construction against a gap.

**It has no edge and does not claim one.** Predictions written here carry
`model="market_baseline"`, so they cannot be mistaken in the track record for output from
a model that has learned something. Track A -- the Bayesian state-space ratings -- replaces
the middle of this function and nothing else.

    uv run python -m hub.models.ratings --fit
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from hub import store
from hub.config import SEASON_AHEAD, config_digest
from hub.fetch import nflverse
from hub.models.base import FitSpec, validate_predictions
from hub.models.market import MarketBaseline


def games_for(season: int, cache: Path | None = None, *, at: datetime | None = None,
              base: Path | None = None) -> pl.DataFrame:
    """Scheduled games priced from a dated snapshot where one exists, the moving field where
    not, and saying which.

    Both columns quote the same quantity -- the betting market's spread on the home team,
    positive when the home team is favoured -- so the choice between them is provenance
    rather than accuracy, and `tests/golden/test_line_agreement.py` carries the evidence
    that they agree where both exist.

    They are not interchangeable as a *record*, which is the whole point. `spread_line` is
    nflverse's own field: it moves as the week runs and is empty for weeks that are far
    away -- 16 of 16 populated for week 1 of this season and 0 of 16 for week 18. A
    prediction priced from it cannot be shown afterwards, only asserted, because the field
    it came from has since changed. `hub.fetch.odds` writes dated, immutable snapshots, and
    `store.lines_as_of` picks the one that was live at `at`.

    The moving field stays as the fallback permanently. **Not for the reason issue #6 gave**:
    that ticket expected far weeks to carry no snapshot when the fit first runs, and measured
    on 2026-09-04 the opposite holds -- The Odds API is already posting week 18, so all 272
    games price from a snapshot and the fallback currently fires on nothing. What it is
    actually for is a store that has no snapshot for a game: a fresh clone, a poller that has
    been down since the last schedule refresh, or a game the pull did not return. Dropping
    those games would shrink the slate rather than admit what priced it.

    `at` defaults to now, in UTC and naive, which is how `hub.fetch.odds` stamps
    `captured_at`; a test supplies its own so the as-of question has a fixed answer.
    """
    sched = nflverse.load("schedules", seasons=[season], cache=cache)
    games = sched.select(
        pl.col("game_id"),
        pl.lit("nfl").alias("league"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("spread_line").cast(pl.Float64).alias("schedule_spread"),
        pl.col("result"),
    )
    moment = at or datetime.now(UTC).replace(tzinfo=None)
    snaps = store.lines_as_of(moment, season, base=base).rename(
        {"close_spread": "snapshot_spread", "captured_at": "priced_at"})
    return (games.join(snaps, on="game_id", how="left")
                 .with_columns(
                     pl.coalesce("snapshot_spread", "schedule_spread").alias("close_spread"),
                     pl.when(pl.col("snapshot_spread").is_not_null()).then(pl.lit("snapshot"))
                       .when(pl.col("schedule_spread").is_not_null()).then(pl.lit("schedule"))
                       .otherwise(None).alias("price_source")))


def target_week(games: pl.DataFrame) -> int:
    """The first week not yet played, or the last week if the season is over.

    Predicting a week that already has results is not forecasting, and the leakage check
    in `validate_predictions` would reject it anyway -- better to pick the right week than
    to make the caller discover the tripwire.
    """
    unplayed = games.filter(pl.col("result").is_null() & pl.col("close_spread").is_not_null())
    if unplayed.height:
        return int(cast(int, unplayed["week"].min() or 1))
    return int(cast(int, games["week"].max() or 1))


# Stamped onto every prediction so a row can be traced to the exact model and configuration
# that produced it. `docs/foundation-plan.md` 3.5: two runs differing only in a
# hyperparameter must produce distinguishable rows, and the public claim is that a specific
# model made a specific prediction -- not "the Bayesian model" as a category.
PROVENANCE_COLUMNS = ("model", "version", "cfg_digest", "fit_digest")


def live_config():
    """The configuration this run is operating under.

    Its own function so a test can substitute one, and so the digest has a single source.
    """
    from hub.config import HubConfig
    return HubConfig()


def coverage(slate: pl.DataFrame) -> dict[str, int]:
    """How many games each source priced, over the slate about to be predicted.

    Reported every run rather than measured once. The fallback's share is the thing that
    tells you whether the snapshot poller has been running, and a share that quietly climbs
    back to 100% is what a dead poller looks like from here -- visible in the fit's own
    output, not only in the watchdog.
    """
    src = slate["price_source"]
    return {"snapshot": int((src == "snapshot").sum()),
            "schedule": int((src == "schedule").sum()),
            "unpriced": int(src.null_count())}


def fit(season: int = SEASON_AHEAD, week: int | None = None, *, cache: Path | None = None,
        base: Path | None = None, at: datetime | None = None) -> pl.DataFrame:
    """Fit through week-1, predict `week`, validate, write versioned predictions."""
    games = games_for(season, cache=cache, at=at, base=base)
    wk = week if week is not None else target_week(games)

    # Fit through the week before the one being predicted. The gap is the leakage
    # tripwire in validate_predictions, and it is load-bearing: leakage looks like
    # success rather than failure, so it has to be structural rather than remembered.
    # The config digest has to actually reach the spec. It defaulted to "default", so every
    # run under every configuration produced the same version string -- provenance that was
    # present in the schema and absent in the data.
    cfg = live_config()
    digest = config_digest(cfg)
    spec = FitSpec("nfl", season, wk - 1, cfg_digest=digest)
    slate = games.filter(pl.col("week") == wk).drop("result")

    model = MarketBaseline().fit(spec)
    preds = model.predict(slate)
    validate_predictions(preds, spec)
    preds = preds.with_columns(pl.lit(digest).alias("cfg_digest"),
                               pl.lit(spec.digest).alias("fit_digest"))
    # One partition per price source, because the version string now carries the source and
    # a partition should be homogeneous in what priced it. A mixed slate -- the normal case
    # in September, when near weeks have snapshots and far ones do not -- would otherwise be
    # filed under whichever source happened to sort first.
    #
    # The digest is in the filename as well as the rows: distinguishable rows are no use if
    # the second run lands on the first one's file. A *different* config, or a different
    # source, therefore writes a different partition and both survive.
    #
    # `replace=True` covers the other case, on purpose: the same config re-run against the
    # same source predicts the same games from the same lines, so only `predicted_at`
    # differs and replacing is what is wanted -- appending would put two rows per game into
    # `preds` and duplicate every game in the published artifact. The pre-registration that
    # `docs/track-record.md` rule 1 counts is the *commit*, not this timestamp, so replacing
    # it costs the record nothing.
    for src in sorted(set(preds["price_source"].to_list())):
        part = preds.filter(pl.col("price_source") == src)
        store.write(part, "preds", "nfl", season, wk, base=base,
                    name=f"{part['version'][0]}-{digest}", replace=True)

    cov = coverage(slate)
    print(f"  ratings (passthrough): season {season} week {wk}")
    print(f"    {preds.height} of {slate.height} games priced: "
          f"{cov['snapshot']} from a dated snapshot, {cov['schedule']} from the moving "
          f"field, {cov['unpriced']} unpriced")
    print(f"    model={MarketBaseline.name} version={model.version}"
          f" fit_through_week={spec.through_week}")
    return preds


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.ratings",
        description="Passthrough ratings: writes the market prior as versioned predictions.")
    ap.add_argument("--fit", action="store_true", help="fit and write predictions")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--week", type=int, default=None,
                    help="default: the first week not yet played")
    a = ap.parse_args(argv)
    if not a.fit:
        ap.print_help()
        return 0
    try:
        fit(a.season, a.week)
    except Exception as e:
        print(f"hub.models.ratings: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
