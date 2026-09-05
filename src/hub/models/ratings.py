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
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from hub import schedule, store
from hub.config import SEASON_AHEAD, config_digest
from hub.models.base import FitSpec, validate_predictions
from hub.models.market import MarketBaseline

# `forecastable` lives in `hub.schedule` now, which owns `kickoff` and `result` -- and which
# `hub.season.survivor` reads the same rule from. A survivor plan that spends teams on weeks
# already over has the same defect this rule fixes here, and the two must not be able to
# disagree about which games are still ahead.


def target_week(games: pl.DataFrame, at: datetime | None = None) -> int:
    """The first week still forecastable, or the last week if none is.

    Not "the first week unfinished": a run that fires late finds a week in progress, and
    predicting the rest of a slate whose first game is already over is the same backdating,
    one game at a time.
    """
    ahead = schedule.forecastable(games, at).filter(pl.col("close_spread").is_not_null())
    if ahead.height:
        return int(cast(int, ahead["week"].min() or 1))
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


def _with_committed(part: pl.DataFrame, season: int, week: int, name: str,
                    base: Path | None) -> pl.DataFrame:
    """This run's predictions, plus any already committed for a game it can no longer make.

    A second run in a week re-fits the games that have not started and rewrites the same
    partition. Without this the games that *have* started vanish from it -- so the record
    would quietly lose exactly the predictions reality has already tested, which is the
    direction a dishonest record trims in. `docs/track-record.md` rule 1 says the commit is
    the timestamp; a later commit must not un-say an earlier one.

    Only games absent from this slate are carried over. One still ahead of its kickoff is
    re-priced on purpose -- the commit still predates it, which is the whole test.
    """
    path = store.partition("preds", "nfl", season, week, name, base)
    if not path.exists():
        return part
    try:
        prior = pl.read_parquet(path)
    except Exception:                       # an unreadable partition is not a record to keep
        return part
    keep = prior.filter(~pl.col("game_id").is_in(part["game_id"].to_list()))
    return pl.concat([part, keep], how="diagonal_relaxed") if keep.height else part


def fit(season: int = SEASON_AHEAD, week: int | None = None, *, cache: Path | None = None,
        base: Path | None = None, at: datetime | None = None) -> pl.DataFrame:
    """Fit through week-1, predict `week`, validate, write versioned predictions."""
    games = schedule.priced_games(season, at=at, cache=cache, base=base)
    wk = week if week is not None else target_week(games, at)

    # Fit through the week before the one being predicted. The gap is the leakage
    # tripwire in validate_predictions, and it is load-bearing: leakage looks like
    # success rather than failure, so it has to be structural rather than remembered.
    # The config digest has to actually reach the spec. It defaulted to "default", so every
    # run under every configuration produced the same version string -- provenance that was
    # present in the schema and absent in the data.
    cfg = live_config()
    digest = config_digest(cfg)
    spec = FitSpec("nfl", season, wk - 1, cfg_digest=digest)
    # Rule 1, made structural rather than scheduled around.
    whole = games.filter(pl.col("week") == wk)
    slate = schedule.forecastable(whole, at).drop("result")
    under_way = whole.height - slate.height

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
        name = f"{part['version'][0]}-{digest}"
        store.write(_with_committed(part, season, wk, name, base), "preds", "nfl", season, wk,
                    base=base, name=name, replace=True)

    print(f"  ratings (passthrough): season {season} week {wk}")
    if under_way:
        # Said out loud: a reader seeing thirteen of sixteen games should learn that the fit
        # ran late, not conclude the week was light.
        print(f"    {under_way} already under way and not predicted -- a prediction counts "
              f"only if it was committed before kickoff")
    cov = schedule.by_source(slate)
    print(f"    {slate.height - cov['unpriced']} of {slate.height} games priced: "
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
