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
from pathlib import Path
from typing import Sequence, cast

import polars as pl

from hub import store
from hub.fetch import nflverse
from hub.models.base import FitSpec, validate_predictions
from hub.models.market import MarketBaseline


def games_for(season: int, cache: Path | None = None) -> pl.DataFrame:
    """Scheduled games with the market's number attached.

    `spread_line` is nflverse's closing spread, positive when the home team is favoured.
    Once `hub.fetch.odds` has been running there will be several snapshots per game and
    the as-of join picks the one that was live; this is the single-number fallback for a
    season nobody has been snapshotting.
    """
    sched = nflverse.load("schedules", seasons=[season], cache=cache)
    return sched.select(
        pl.col("game_id"),
        pl.lit("nfl").alias("league"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("spread_line").cast(pl.Float64).alias("close_spread"),
        pl.col("result"),
    )


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


def fit(season: int = 2026, week: int | None = None, *, cache: Path | None = None,
        base: Path | None = None) -> pl.DataFrame:
    """Fit through week-1, predict `week`, validate, write versioned predictions."""
    games = games_for(season, cache=cache)
    wk = week if week is not None else target_week(games)

    # Fit through the week before the one being predicted. The gap is the leakage
    # tripwire in validate_predictions, and it is load-bearing: leakage looks like
    # success rather than failure, so it has to be structural rather than remembered.
    spec = FitSpec("nfl", season, wk - 1)
    slate = games.filter(pl.col("week") == wk).drop("result")

    preds = MarketBaseline().fit(spec).predict(slate)
    validate_predictions(preds, spec)
    if preds.height:
        store.write(preds, "preds", "nfl", season, wk, base=base,
                    name=f"{preds['version'][0]}")

    print(f"  ratings (passthrough): season {season} week {wk}")
    print(f"    {preds.height} games priced from the market, "
          f"{slate.height - preds.height} without a line")
    print(f"    model={MarketBaseline.name} version={preds['version'][0] if preds.height else '-'}"
          f" fit_through_week={spec.through_week}")
    return preds


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.ratings",
        description="Passthrough ratings: writes the market prior as versioned predictions.")
    ap.add_argument("--fit", action="store_true", help="fit and write predictions")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None,
                    help="default: the first week not yet played")
    a = ap.parse_args(argv)
    if not a.fit:
        ap.print_help()
        return 0
    try:
        fit(a.season, a.week)
    except Exception as e:  # noqa: BLE001
        print(f"hub.models.ratings: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
