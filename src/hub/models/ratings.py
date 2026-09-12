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
what `forecaster()` returns and nothing else.

**One exception, since #218, and it is narrow on purpose.** Where the staleness field (#210)
marks a game as having no live price -- a snapshot quote that has stood untouched for more
than `hub.models.quarterback.STALE_AFTER_DAYS`, or the moving field that nothing polls -- the
number handed to the forecaster is quarterback-adjusted from `hub.fetch.nfeloqb`'s published
state, and the row says so: `adjusted_by` and `qb_adjustment` are filled, and the version
string gains `-qb`. Where a live price exists the row is what it always was, byte for byte.
`rated_games` is the seam, and it is shared with `hub.season.survivor` so the two cannot
disagree about a game they both rate. `docs/qb-adjustment.md` is the validation that licenses
this and no more.

**What this module is, and what it is not.** It is the *writer*: it decides which week may
be predicted, stamps provenance, and files partitions. It is not a model, and it no longer
does a model's job by hand -- fitting, predicting and checking are `hub.models.base.forecast`
against a `Forecaster`, which is the interface ADR-0002 says every track implements. This
used to name a concrete class and remember to call `validate_predictions` afterwards, so
the leakage tripwire hung off one untyped call and would have had to be remembered again by
whoever wrote Track A (issue #205).

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

from hub import schedule, store
from hub.config import SEASON_AHEAD, config_digest
from hub.contracts import ContractViolation
from hub.fetch import nfeloqb
from hub.models import quarterback
from hub.models.base import FitSpec, Forecaster, forecast
from hub.models.market import MarketBaseline


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
STAMP_COLUMNS = ("model", "version", "cfg_digest", "fit_digest")


def live_config():
    """The configuration this run is operating under.

    Its own function so a test can substitute one, and so the digest has a single source.

    It used to answer with a bare `HubConfig()`, which is the dataclass defaults and not what
    a run operates under -- the name said "live" and the body said "as shipped". They agree
    today only because nothing in `conf/` diverges from a default; the first one that did
    would have stamped every prediction row with a digest for a configuration nobody ran,
    which is the same shape as the `cfg_digest = "default"` defect this module's `fit`
    records two screens down. `hub.config.resolved_config` is now the one answer, here and
    in the fetch layer's provenance line.
    """
    from hub.config import resolved_config
    return resolved_config()


def forecaster() -> Forecaster:
    """The model this run publishes, named by its interface rather than by its class.

    Its own function for `live_config`'s reason -- a test can substitute one, and there is
    a single source -- and for one more: the return type is ADR-0002's protocol, so this is
    the seam Track A arrives at. Replacing the passthrough is a substitution here, not an
    edit to the middle of `fit`, and `hub.models.base.forecast` will check whatever arrives
    the same way.

    Still `MarketBaseline`, and the module docstring says why that is honest: it writes
    under the baseline's own name and version, so nothing in the track record can later
    credit it with an edge it never had.
    """
    return MarketBaseline()


def quarterback_state(cache: Path | None) -> tuple[pl.DataFrame | None, str]:
    """`hub.fetch.nfeloqb`'s per-team state off its cache, and one sentence about it.

    `cache` is the nflverse cache root a fit was given, and the nfeloqb file is looked for
    under it -- `<cache>/nfeloqb/` -- so a test that redirects one redirects both and a
    developer's own `data/raw/nfeloqb/` cannot reach a test through the default. With no
    cache given, the module's own default.

    Three answers, and the sentence says which: a state, no file (a fresh clone, or a pull
    that has never run), or a file the contract refused. The last two both mean the
    passthrough, and neither is an error -- CLAUDE.md's rule is that a missing fetch serves
    what can be served, and what can be served here is the betting market's number as it was.
    """
    where = None if cache is None else cache / "nfeloqb"
    try:
        state = nfeloqb.read_state(where)
    except ContractViolation as e:
        return None, f"the cached nfeloqb file was refused, so no rating moved: {e}"
    if state is None:
        return None, "no nfeloqb file cached, so no rating moved (hub.fetch.nfeloqb --refresh)"
    # The pin (#271): which commit the file came from, and -- said again here, because the
    # pull said it once on the day and this run may be days later -- whether its bytes
    # matched the pin. A source change is served, and it is never served silently.
    record = nfeloqb.stamp(where)
    commit = record.get("commit")
    at = (f"pulled {record.get('captured_at')} at commit "
          f"{str(commit)[:12] if commit else 'none (unpinned, the default branch)'}")
    said = f"nfeloqb state for {state.height} teams, {at}"
    if (change := nfeloqb.source_change(where)):
        said += f"; {change}"
    return state, said


def rated_games(season: int, *, at: datetime | None = None, cache: Path | None = None,
                base: Path | None = None) -> tuple[pl.DataFrame, str]:
    """The season's games as this module rates them, and one sentence about the state used.

    `hub.schedule.priced_games` with `hub.models.quarterback.apply` over it: every game
    priced from the betting market where a live price exists, and quarterback-adjusted where
    the staleness field marks none. The one seam for both readers -- the weekly fit below and
    `hub.season.survivor.grid_from_schedule` -- for the reason `hub.schedule` exists at all:
    two readers of one rule, so they cannot disagree about the same game.

    `at` defaults to now, naive UTC, the way `priced_games` defaults it, and the same moment
    decides both which snapshot priced a game and how long that quote had stood.
    """
    moment = at or datetime.now(UTC).replace(tzinfo=None)
    games = schedule.priced_games(season, at=moment, cache=cache, base=base)
    state, said = quarterback_state(cache)
    if state is not None and (unknown := nfeloqb.unknown_teams(state, games)):
        # A spelling the source uses and nflverse does not is a team that never adjusts,
        # and nothing downstream would say so. See `nfeloqb.ABBREVIATIONS`.
        said += (f"; {len(unknown)} team(s) in the state match no game and never adjust: "
                 f"{', '.join(unknown)} -- a spelling nfeloqb.ABBREVIATIONS does not map?")
    return quarterback.apply(games, state, at=moment), said


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
    """Pick a week, drive a `Forecaster` through `base.forecast`, write the rows it checked."""
    games, qb_said = rated_games(season, at=at, cache=cache, base=base)
    wk = week if week is not None else target_week(games, at)

    # Fit through the week before the one being predicted. The gap is the leakage
    # tripwire, and it is load-bearing: leakage looks like success rather than failure, so
    # it has to be structural rather than remembered -- which is why the fit, the
    # prediction and the check now happen together in `base.forecast` against this one
    # spec, rather than as three statements here that can drift apart.
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

    model = forecaster()
    preds = forecast(model, spec, slate)
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
    #
    # By version rather than by `price_source`, since #218: the version carries the source
    # *and* whether the quarterback layer moved the row, and a partition of one source can
    # hold both kinds -- a frozen quote beside a fresh one. Before #218 the two keys were the
    # same partition, so nothing already written is filed differently.
    for ver in sorted(set(preds["version"].to_list())):
        part = preds.filter(pl.col("version") == ver)
        name = f"{ver}-{digest}"
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
    # Read off the fitted object rather than off a class name, so the line says what ran
    # and not what this module used to import.
    print(f"    model={model.name} version={model.version}"
          f" fit_through_week={spec.through_week}")
    # #218's last criterion: the change to the published numbers, said once per run, with
    # the count of games it touched. Off the slate that was predicted, so the line is about
    # what this run wrote and not about the season.
    print(f"    {quarterback.report_line(slate)}")
    print(f"    {qb_said}")
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
