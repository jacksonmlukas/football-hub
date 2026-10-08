"""Does the shipped weekly interval cover, and does the survivor price hold by spread?

Two distributions that no gate has ever scored, graded here as committed code because
[ADR-0007](../../../docs/adr/0007-measurements-that-steer-the-product-are-committed-code.md)
requires it of a number that steers anything -- and these now do, through `--gate`.

**What is graded is the deployed moment function.** `hub.models.predict.moments` is what
`draft/optimize.py`, `season/roster.py` and `season/lineup_gate.py` actually call, and this
harness calls the same function rather than restating `WEEKLY_K[pos] * sqrt(mu)` beside it.
The interval bounds come from `predict.skewed` at `z = Phi^-1(p)` -- the transform is
monotone in `z`, so that is the quantile -- which is the same function `draft/season.py`
draws through. Nothing about the distribution is re-implemented here; a restatement that
agreed today is how the grader and the graded drift apart later.

**The centre is the whole argument.** `docs/weekly-coverage.md` centred on each
player-season's own realised mean, and said so: that is a *lookahead*, because the number
being used to build the interval was computed from the very weeks the interval is then
scored against. It flatters the result in one specific way -- the centre is exactly right by
construction, so the only spread the interval has to cover is the week-to-week noise, and
none of the error in knowing where the centre is.

So `--centre prior` is the default and is the real measurement: for each week, the centre is
the mean of that player's **strictly earlier** weeks in the same season. That is the rule
`hub.models.conformal` already states for its calibration window and the one
`docs/track-record.md` rule 1 forbids breaking, applied here. `--centre realised` reproduces
the document, and is kept for exactly that -- a superseded number you cannot reproduce is a
number nobody can check.

The two answer different questions and the difference is large; see the 2026-09-07
restatement in `docs/weekly-coverage.md`.

**The survivor half.** `season/survivor.py` prices every pick as
`normal_cdf(close_spread / MARGIN_SD)` and then multiplies those numbers into a survival
probability. Nothing has ever asked whether that price holds *at the spreads survivor
actually picks at*, which are the big favourites and nowhere near the middle of the
distribution. Graded by spread bucket rather than by probability bucket, because a miss
concentrated in one spread range is what a pick rule would walk into.

**What the gate holds the interval to** is the claim, not the label (#289). Until #309 the
interval was served as 80% and measured to cover 77.4% of the unclipped weeks; `CLAIMED_COV80`
is that restated claim and the gate refuses when the deployed function leaves `BAND` of it in
either direction. The label is `LEVELS`, which has not moved.

**What is graded since #309 is the published interval, conformalised, over the whole board.**
The parametric interval above carried no estimation error and its lower bound was a clipped
parametric one; the published interval is `predict.predictive_sd`'s scale
(`sd * sqrt(1 + 1/n)`) with its bounds calibrated *empirically* on the rolling window of
strictly earlier `(season, week)` cells -- within position, with a named fallback to the pooled
window below 200 calibration rows (`hub.models.conformal.mondrian`). The gate grades every
player-week that has a calibration, clipped or not, and says how many of the board it could
not score. The parametric interval is kept in the result as `parametric` -- the prior values --
and is what `--shape` still reads. The design is `docs/weekly-coverage.md`, 2026-10-06.

    uv run python -m hub.models.coverage --measure
    uv run python -m hub.models.coverage --measure --centre realised
    uv run python -m hub.models.coverage --survivor
    uv run python -m hub.models.coverage --gate          # exits 1 on a refusal
    uv run python -m hub.models.coverage --shape         # the skew law under CRPS (#292)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from hub import atomic, jsonio
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS, SEASON_AHEAD
from hub.declare import not_an_input
from hub.ledger import WIDTH_STATE, Ledger, code_digest
from hub.ledger import recipe as _recipe
from hub.models import conformal, predict
from hub.models.experiment import (
    Actions,
    GateRun,
    Harness,
)
from hub.models.scoring_rules import (
    crps_from_quantiles,
    normal_quantile,
    quantile_levels,
    reliability_by,
)
from hub.paths import STATE_DIR

# Nothing here is fitted. Every number below is either a filter this measurement inherits
# from the document it supersedes, a nominal level, or a threshold pre-registered before the
# answer was looked at -- and none of them reaches a prediction, because this module grades
# `hub.models.predict` and never calls it to forecast anything. Registering them in
# `config_digest` would stamp a prediction with the settings of its own grader.
Centre = Literal["prior", "realised"]

# The window `docs/weekly-coverage.md` measured over, so `--centre realised` reproduces it
# without the caller having to know which five seasons those were.
SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

# The positions the weekly laws are fitted for, read from the league rather than written out
# again: `WEEKLY_K` and `WEEKLY_SKEW` fall back to a pooled value for anything else, and
# grading a fallback tells you about the fallback. A superflex league would move
# `RosterConfig` and this follows it.
POSITIONS: tuple[str, ...] = DRAFTED_POSITIONS

# A player-season is in the sample if it has this many scoring weeks and averages this many
# points. Both from the document, unchanged, so the two centres are compared on one filter.
MIN_WEEKS = 8
MIN_MU = not_an_input(
    2.0,
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# Under `--centre prior` a week is scored only once the player has this many earlier weeks
# behind him. Below it the centre is mostly noise and the measurement becomes a statement
# about small-sample means rather than about the interval. Four is the smallest number at
# which the centre's own standard error is under half the weekly spread.
MIN_PRIOR = 4

# The two nominal levels, as (lower p, upper p). 80% is the interval `season/lineup.py`'s
# win probability is effectively asserting; 68% is one sigma, reported because a miss that
# is about the *shape* rather than the width shows up as the two disagreeing.
LEVELS: tuple[tuple[float, float], ...] = not_an_input(
    ((0.10, 0.90), (0.16, 0.84)),
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# **What the interval is measured to cover, which is what the gate holds it to (#289).** The
# interval is built at (p10, p90) and served under the label 80% -- `LEVELS` above is
# unchanged and `season/lineup.py` still asserts it -- and on the unclipped weeks it covers
# 77.4% (10,536 player-weeks, 2021-2025, prior centre; `docs/weekly-coverage.md`, restated
# 2026-09-13). The band was pre-registered at 80 +/- 2 and the deployed function sits outside
# it. Three things could follow: widen sigma by a fitted factor, refit the shape law, or say
# what the interval covers. The first is a data-chosen cut -- the number would be picked to
# make this gate green, which is #287's fault elsewhere; the second is #292, pre-registered
# and decided by CRPS; this is the third. So the claim is restated to the measured rate and
# the gate is re-registered against it: **77 +/- 2, on the deployed function.** The gate goes
# green on a truthful claim and not on a number chosen to make it green, and it goes red
# again if the deployed function drifts *either* way from what the doc says -- an interval
# that came to cover 80% would be a stale claim upward, and the gate says so.
#
# Not a fitted constant and not a choice a prediction reads: a restated claim about a
# measurement, pinned so that the gate and the doc cannot say two different numbers.
#
# **Restated again, 2026-10-07 (#310, revisiting #289): 0.80.** Conformalisation changed the
# kind of claim. The 0.77 above was honest for a parametric interval nothing made cover 80%;
# the published interval is now calibrated on the rolling window, so 80% is a *marginal* claim
# the construction asserts and the gate's job is to test whether it holds -- a gate that can
# only restate its own label is not a decision (ADR-0015). The 0.77 and its reasoning stay in
# `docs/weekly-coverage.md` as the record of the parametric interval.
CLAIMED_COV80 = not_an_input(
    0.80,
    "the claim the grading harness holds the deployed interval to; it grades "
    "predictions already made and makes none of its own")

# The pre-registered audit rule of #310 (the maintainer's ADOPTED comment, 2026-09-17), in code
# so the weekly run and the audit cannot disagree about it (rule 1). Not fitted, not read by
# any prediction.
#
# Three looks per claim-life, alpha spent across them (0.05 / 3 = 0.0167 a look, two-sided, so
# z = 2.394): the verdict is the full-history gate taken *at each audit*, because a gate read
# weekly on accumulating rows is a sequential test -- eighteen looks a season on one hypothesis.
LOOKS = not_an_input(
    3, "the number of audit looks a claim is allowed to live through, pre-registered (#310)")
ALPHA_PER_LOOK = not_an_input(
    0.05 / 3, "the type-I error spent on each audit look, pre-registered (#310)")
Z_PER_LOOK = not_an_input(
    statistics.NormalDist().inv_cdf(1.0 - 0.05 / 3 / 2),
    "the critical value of one audit look, derived from ALPHA_PER_LOOK (#310)")
# A position returns a verdict only at this many player-weeks: Delta = 0.02 (the band itself),
# power 0.80, variance under p = 0.80: (z_alpha + z_0.8)^2 * 0.32 / 0.02^2. The adopted rule
# names 8,377; the formula at the rounded z's is 8,375.3 (`derived_n_min`). **The adopted number
# binds** -- it is two weeks the more conservative, and changing a pre-registered figure after
# seeing which groups it admits is rule 1's laundering.
N_MIN_GROUP = not_an_input(
    8377, "the player-weeks below which a position cannot rule, pre-registered (#310)")
# The weekly run's smoke alarm: marginal coverage further than this from the claim (eight
# MDEs; it cannot fire on drift or noise, only on a broken pipeline), or a group with nothing
# scored. This is what survives of #273's statistical role in the weekly step.
SMOKE_BAND = not_an_input(
    0.10, "the weekly smoke alarm's distance from the claim, pre-registered (#310)")
# The 95% interval a position's coverage must sit in, at sqrt(2) x the binomial SE.
GROUP_Z = not_an_input(
    1.959963984540054, "the 95% two-sided normal quantile of the per-position interval (#310)")


def derived_n_min() -> float:
    """The minimum group size the adopted rule's formula gives; 8,375.3 beside the adopted 8,377."""
    zb = statistics.NormalDist().inv_cdf(0.80)
    return (Z_PER_LOOK + zb) ** 2 * 0.32 / BAND ** 2

# Pre-registered before the prior-centre numbers were looked at, and the only other thing
# `--gate` reads: empirical coverage must sit within this of the claim. Two points is roughly
# three standard errors at n = 10,000, so it is a band a truthful claim clears comfortably and
# a stale one does not.
BAND = not_an_input(
    0.02,
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# The gate is read off every player-week that has a calibration -- the whole scored board,
# clipped and unclipped (#309). Before it was the weeks whose parametric interval was *not*
# pinned at the zero floor, on the reasoning that a clipped lower bound cannot be fallen below
# and pooling those weeks in hides a miss. That excluded 5,525 of 16,061 player-weeks -- the
# low-mean third of the board, where the interval was degenerate -- and made the gated figure
# a statement about high-mean players. The remedy for a flattering subset is to measure the
# population and report the clipped share beside the rate, not to measure the half that
# cannot embarrass the interval. The floor split is still printed, as a diagnostic.
GATE_SUBSET = "all"

# The calibration window, in `(season, week)` cells: the span one season's scored weeks occupy
# (week 5, the first with four earlier weeks, through week 18) -- "the prior full season", the
# default #309 names, at which QB calibrates on ~370 rows. Chosen before the number
# (`docs/weekly-coverage.md`, 2026-10-06) and not fitted: a window picked to make the gate
# green is #287's fault.
CAL_WINDOW = not_an_input(
    14,
    "a setting of the grading harness's rolling calibration, which measures coverage of "
    "predictions already made and makes none of its own")

# How the rows are cut by games of evidence behind the centre, as (label, low, high) inclusive.
# The cuts `docs/weekly-coverage.md` printed in its 2026-09-07 restatement, kept so the two are
# comparable: the estimation error's whole claim is that it matters at the thin end.
PRIOR_BUCKETS: tuple[tuple[str, int, int | None], ...] = not_an_input(
    (("4-5", 4, 5), ("6-8", 6, 8), ("9-12", 9, 12), ("13+", 13, None)),
    "a pre-registered cut of the grading harness, which measures interval coverage of "
    "predictions already made and makes none of its own")

# Spread buckets for the survivor price, favourite-relative and in points: under 3, 3-7,
# 7-10, 10-14, 14 and up (#293). Each bucket is closed at its low edge and open at its high
# one, and the last takes its top edge, so a 7-point favourite is in 7-10 -- the same side
# of the line the `SURVIVOR_SPREAD` headline counts it on -- and 3 and 7, the two spreads
# the betting market lands on most, each start a bucket rather than splitting one. The top
# bucket is where survivor lives: `season/survivor.py` picks the biggest favourite on the
# board, so a bucket that pools a 3-point favourite with a 13-point one answers a question
# nobody asks of it. The edges before #293 were (3, 6, 9, 14), which put 7 in a 6-9 bucket
# no pick rule reads; the pooled verdict and `favourite_gap` do not read the buckets and did
# not move.
SPREAD_EDGES: tuple[float, ...] = not_an_input(
    (0.0, 3.0, 7.0, 10.0, 14.0, 30.0),
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# How the buckets are labelled where they are printed and published, one per pair of
# edges: `<3` is [0, 3), `3-7` is [3, 7), `14+` is [14, 30]. A label that said `<=3` would
# be lying about the 3-point favourites, which are in the next one.
SPREAD_LABELS: tuple[str, ...] = ("<3", "3-7", "7-10", "10-14", "14+")

# A bucket holding fewer games than this is shown and marked thin, never dropped (#293).
# Fifty is where the standard error of a win rate near 0.85 is about five points -- wide
# enough that a gap inside it says nothing, and a reader is owed the count to see that.
# An empty bucket is a thin bucket: four lines where five were expected read as a bucket
# that was dropped, which is the thing this exists to stop.
MIN_BUCKET = not_an_input(
    50,
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# What counts as "the favourites survivor actually picks", for the headline the gate reads.
SURVIVOR_SPREAD = not_an_input(
    7.0,
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# Where `--measure --write` and `--survivor --write` leave their answers and where
# `hub.publish` reads them from. Under `state/`, which is committed, rather than
# `data/processed/`, which is gitignored: a file written there never reached the runner, so
# the publisher read nothing and the track record shipped with no coverage field while the
# doc said it carried one (#273). Not under `site/data/` because the site copy is published
# output and this is the measurement behind it; the publisher joins the two.
ARTIFACT = STATE_DIR / "interval_coverage.json"

# --- the current season's row, and the backtest that is read back (#423) ---------------
#
# **The 2021-2025 backtest is a fixed past.** `SEASONS` is closed, so its figure cannot move; the
# slate re-pulled five seasons twice a week to reproduce it, and its diff was the timestamp and
# the sixteenth digit of four ratios (Audit V, V5). It is measured once per *key* -- the config,
# the code that computes it, and the identity of the data it reads -- and read back from the state
# file while the key holds. The data side of the key is the declared identity of a closed span
# (the season list), not a hash of the bytes: hashing them would need the pull this avoids, so a
# restated nflverse season is the one thing it cannot see, and `--remeasure` is the lever for it.
# A span that includes the current season is not closed and is never read back.
#
# **The current season is a separate, labelled, out-of-sample row.** The published interval is
# calibrated on the last `CAL_WINDOW` cells before the cell it is scored on, so the 2026 weeks are
# graded against the 2025 weeks behind them; `CURRENT_HISTORY` seasons are read ahead of the
# current one so that window is full (a window that reached past the data read would be a
# different calibration from the one the full history gives; a unit test holds the two equal).
# It is not pooled into the backtest and carries no verdict: looks 2 and 3 of the claim read it
# (`LOOK_WEEKS`, pre-registered 2026-10-07 in `docs/weekly-coverage.md`) and nothing else does.
CURRENT_SEASON = SEASON_AHEAD
CURRENT_HISTORY = not_an_input(
    2, "how many seasons are read ahead of the current one so the calibration window is full; "
       "a reading choice of the grader, which makes no prediction")
# Look k reads the current-season row once week LOOK_WEEKS[k] is complete (look 1 is the backtest).
LOOK_WEEKS = not_an_input(
    {2: 14, 3: 18},
    "the week after which each look that reads the current-season row may be taken, "
    "pre-registered 2026-10-07 (#423, docs/weekly-coverage.md)")
# A week is complete once its last game was played this many days ago and its rows are in the
# data; the same one-day lag `hub.season.weekly_forward.LAG_DAYS` uses for #432's horizon, and a
# unit test holds the two equal.
LAG_DAYS = not_an_input(
    1, "days after a week's last game before its rows are expected in nflverse; the grader's "
       "horizon, the same as #432's")
# Written floats are rounded to this many places, so a run that changes nothing writes the same
# bytes and the slate commits nothing. Six places is a millionth of a rate, far below the
# 1/sqrt(n) noise of any row here (0.003 at n = 16,000) and above the sixteenth-digit wobble
# Audit V found.
ROUND_DIGITS = not_an_input(
    6, "decimal places the artifact is written at; a presentation precision of the grader")
_CODE = ("hub.models.coverage", "hub.models.predict", "hub.models.conformal")
_SURVIVOR_CODE = ("hub.models.coverage", "hub.models.margin", "hub.models.market")


class NotEnoughWeeks(Exception):
    """No player-season survived the filters, so there is nothing to grade."""


def _z(p: float) -> float:
    """The standard normal quantile, from the standard library rather than scipy.

    `hub.models.market.normal_cdf` is this function's inverse and is written out the same
    way, with `math`. Adding a dependency to invert a normal is not a trade this repo makes.
    """
    return statistics.NormalDist().inv_cdf(p)


def player_weeks(stats: pl.DataFrame) -> pl.DataFrame:
    """(player_id, position, season, week, points) for regular-season skill players.

    Weeks are weeks he recorded stats. Byes and missed games are therefore out of the sample
    entirely, which is a real limit and the same one the document names: this grades the
    distribution of a week he played, not the distribution of a week.
    """
    need = {"player_id", "position", "season", "week", "season_type",
            "fantasy_points_ppr"}
    missing = sorted(need - set(stats.columns))
    if missing:
        raise ValueError(f"player_stats is missing {missing}")
    return (stats.filter(pl.col("season_type") == "REG")
                 .filter(pl.col("position").is_in(list(POSITIONS)))
                 .select("player_id", "position", "season", "week",
                         pl.col("fantasy_points_ppr").fill_null(0.0)
                           .cast(pl.Float64).alias("points"))
                 .sort(["player_id", "season", "week"]))


def centred(pw: pl.DataFrame, centre: Centre = "prior", *, min_weeks: int = MIN_WEEKS,
            min_prior: int = MIN_PRIOR, min_mu: float = MIN_MU) -> pl.DataFrame:
    """Attach the centre each week's interval is built around.

    `"realised"` is the document's: one number per player-season, that season's own mean.
    Every week is inside its own centre, which is the lookahead.

    `"prior"` is the honest one: the running mean of the weeks *before* this one. A week is
    kept once `min_prior` earlier weeks exist, so the first few weeks of every player-season
    leave the sample rather than being graded against a centre built from one game.

    The `min_mu` filter applies to whichever centre is in use, so the two are compared on the
    same rule and not on the same *rows* -- they cannot be, and pretending otherwise by
    fixing the row set to the realised-centre sample would smuggle the lookahead back in
    through the sample definition.
    """
    if centre not in ("prior", "realised"):
        raise ValueError(f"centre must be 'prior' or 'realised', got {centre!r}")
    if centre == "realised":
        by = ["player_id", "season"]
        g = pw.group_by(by).agg(pl.len().alias("weeks"),
                                pl.col("points").mean().alias("centre"))
        keep = g.filter((pl.col("weeks") >= min_weeks) & (pl.col("centre") >= min_mu))
        out = (pw.join(keep.select(*by, "centre"), on=by, how="inner")
                 .with_columns(pl.lit(None, dtype=pl.Int64).alias("n_prior")))
    else:
        over = ["player_id", "season"]
        out = (pw.with_columns(
                    pl.col("points").cum_sum().over(over).alias("_cs"),
                    pl.col("points").cum_count().over(over).alias("_cn"))
                 .with_columns(
                    ((pl.col("_cs") - pl.col("points"))
                     / (pl.col("_cn") - 1)).alias("centre"),
                    (pl.col("_cn") - 1).cast(pl.Int64).alias("n_prior"))
                 .drop("_cs", "_cn")
                 .filter((pl.col("n_prior") >= min_prior) & (pl.col("centre") >= min_mu)))
    if out.is_empty():
        raise NotEnoughWeeks(
            f"no player-week survived centre={centre!r} with min_weeks={min_weeks}, "
            f"min_prior={min_prior}, min_mu={min_mu}")
    return out


def graded(sample: pl.DataFrame) -> pl.DataFrame:
    """The sample with the deployed moments and the interval bounds attached.

    `predict.moments` reads its mean off `proj_blend`/`proj_ppg`/`xfp_per_game`, so the
    centre is handed to it as `proj_ppg` -- the same door a projection comes through. What
    comes back is the shipped `mu`, `sd` and `skew` and not a local copy of the laws behind
    them.

    `p10_raw` is the lower bound *before* `skewed()`'s clip at zero, and is the only piece of
    that function reproduced here. It has to be: the clip is what the floor split tests, and
    there is no way to ask whether a bound was clipped from the clipped bound alone.
    """
    m = predict.moments(sample.with_columns(pl.col("centre").alias("proj_ppg")))
    mu, sd, sk = (m["mu"].to_numpy(), m["sd"].to_numpy(), m["skew"].to_numpy())
    cols = {f"q{int(p * 100):02d}": predict.skewed(mu, sd, sk, _z(p))
            for pair in LEVELS for p in pair}
    a = np.maximum(sk, predict.MIN_SKEW) / 6.0
    z10 = _z(LEVELS[0][0])
    cols["p10_raw"] = mu + sd * (z10 + a * (z10 ** 2 - 1.0)) / np.sqrt(1.0 + 2.0 * a ** 2)
    # The counterfactual for "does the skew earn its place": the same mu and sd read through
    # a plain normal. Deliberately *unclipped*, which is what the document compared against
    # -- the clip is a separate effect and the floor split is where it is isolated.
    cols["n10"] = mu + sd * z10
    cols["n90"] = mu + sd * _z(LEVELS[0][1])
    # The scale a published interval needs (#309): the shape law's sd plus the centre's
    # estimation error. Only where the centre was estimated from earlier weeks -- the realised
    # centre is the lookahead and has no estimation error to add, which is what it flatters.
    if "n_prior" in m.columns:
        n = m["n_prior"].cast(pl.Float64).to_numpy()
        sd_pred = np.where(np.isnan(n), sd, predict.predictive_sd(sd, n))
    else:
        sd_pred = sd
    cols["sd_pred"] = sd_pred
    # The parametric interval at that scale, uncalibrated: separates what estimation error
    # alone closes from what the empirical calibration adds on top.
    cols["pe10"] = predict.skewed(mu, sd_pred, sk, z10)
    cols["pe90"] = predict.skewed(mu, sd_pred, sk, _z(LEVELS[0][1]))
    return m.with_columns(**{k: pl.Series(k, v) for k, v in cols.items()})


def _row(label: str, g: pl.DataFrame) -> dict[str, Any]:
    """One line of the coverage table."""
    y = g["points"].to_numpy()
    lo, hi = g["q10"].to_numpy(), g["q90"].to_numpy()
    lo68, hi68 = g["q16"].to_numpy(), g["q84"].to_numpy()
    n10, n90 = g["n10"].to_numpy(), g["n90"].to_numpy()
    # Realised spread against model spread, one reading per player-season so that a player
    # with nineteen weeks does not count nineteen times, then averaged over player-seasons.
    # The residual is taken around each row's own mu, which is what makes this work for a
    # moving centre as well as a fixed one.
    per = (g.with_columns((pl.col("points") - pl.col("mu")).alias("_r"))
            .group_by(["player_id", "season"])
            .agg(pl.col("_r").std().alias("_sd"), pl.col("sd").mean().alias("_m"),
                 pl.len().alias("_n"))
            .filter((pl.col("_n") >= 2) & (pl.col("_m") > 0)))
    ratio = (float(cast(float, (per["_sd"] / per["_m"]).mean()))
             if per.height else float("nan"))
    return {
        "group": label, "n": int(g.height),
        "cov80": float(np.mean((y >= lo) & (y <= hi))),
        "cov68": float(np.mean((y >= lo68) & (y <= hi68))),
        "below_p10": float(np.mean(y < lo)), "above_p90": float(np.mean(y > hi)),
        "cov80_no_skew": float(np.mean((y >= n10) & (y <= n90))),
        "sd_ratio": ratio,
    }


def table(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The coverage table: one row per position, then the pool."""
    rows = [_row(p, g.filter(pl.col("position") == p)) for p in POSITIONS
            if g.filter(pl.col("position") == p).height]
    rows.append(_row("all", g))
    return rows


def floor_split(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The same coverage, split on whether the model's own p10 survived the zero clip.

    `skewed()` clips at zero, so for a low-`mu` player the 10th percentile lands at or below
    zero and *nothing can fall below it*. Those weeks report a coverage the interval did not
    earn. This is the document's diagnosis and it survives the centre change; what does not
    survive is the conclusion drawn from it.
    """
    return [_row(name, g.filter(sel)) for name, sel in
            (("clipped at zero", pl.col("p10_raw") <= 0.0),
             ("strictly positive", pl.col("p10_raw") > 0.0))
            if g.filter(sel).height]


def calibrate(g: pl.DataFrame, *, window: int | None = CAL_WINDOW,
              floor: int = conformal.MIN_GROUP_CALIBRATION) -> pl.DataFrame:
    """Attach the published interval: bounds calibrated on the rolling window, per position.

    The score of a played week is `(points - mu) / sd_pred`, signed. For each `(season, week)`
    cell the calibration is the scores of the cells strictly before it, the last `window`
    (`conformal.history`, the module's own timeline), and a position is calibrated on its own
    rows there when it has `floor` of them and on the pooled window otherwise
    (`conformal.mondrian`). The bounds are `mu + sd_pred * q_lo` and `mu + sd_pred * q_hi`
    at each of `LEVELS`; the lower bound is clipped at zero as the interval always has been
    and `raw_lo` keeps it before the clip, because the clip is what `clipped` tests.

    Added columns: `raw_lo`, `lo`, `hi`, `lo68`, `hi68`, `n_cal` (the position's own calibration
    rows), `n_pool` (the pooled window's), `fell_back` and `scored`. A cell whose pooled
    window is under `floor` is not a calibration and its rows come back `scored = False` with
    null bounds -- named by the caller, never dropped here.
    """
    need = {"season", "week", "position", "points", "mu", "sd_pred"}
    missing = sorted(need - set(g.columns))
    if missing:
        raise ValueError(f"calibrate needs {missing}")
    n = g.height
    season = g["season"].to_numpy().astype(int)
    week = g["week"].to_numpy().astype(int)
    pos = g["position"].to_numpy().astype(str)
    mu = g["mu"].to_numpy().astype(float)
    sdp = g["sd_pred"].to_numpy().astype(float)
    score = (g["points"].to_numpy().astype(float) - mu) / sdp
    cell_of = list(zip(season.tolist(), week.tolist(), strict=True))
    by_cell: dict[tuple[int, int], np.ndarray] = {}
    for c in sorted(set(cell_of)):
        by_cell[c] = np.flatnonzero((season == c[0]) & (week == c[1]))
    cells = sorted(by_cell)

    nan = np.full(n, np.nan)
    raw_lo, hi, lo68, hi68 = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    n_cal = np.zeros(n, dtype=np.int64)
    n_pool = np.zeros(n, dtype=np.int64)
    fell = np.zeros(n, dtype=bool)
    scored = np.zeros(n, dtype=bool)
    l80, u80 = LEVELS[0]
    l68, u68 = LEVELS[1]
    for c in cells:
        past = conformal.history(cells, c, window)
        if not past:
            continue
        pi = np.concatenate([by_cell[x] for x in past])
        ps, pg = score[pi], pos[pi]
        rows = by_cell[c]
        for p_ in np.unique(pos[rows]):
            r = rows[pos[rows] == p_]
            cal80 = conformal.mondrian(ps, pg, str(p_), lower_p=l80, upper_p=u80, floor=floor)
            cal68 = conformal.mondrian(ps, pg, str(p_), lower_p=l68, upper_p=u68, floor=floor)
            n_pool[r] = ps.size
            n_cal[r] = int((pg == p_).sum())
            if cal80 is None or cal68 is None:
                continue
            scored[r] = True
            fell[r] = cal80.fell_back
            raw_lo[r] = mu[r] + sdp[r] * cal80.q_lo
            hi[r] = mu[r] + sdp[r] * cal80.q_hi
            lo68[r] = np.maximum(mu[r] + sdp[r] * cal68.q_lo, 0.0)
            hi68[r] = mu[r] + sdp[r] * cal68.q_hi
    return g.with_columns(
        pl.Series("raw_lo", raw_lo), pl.Series("lo", np.maximum(raw_lo, 0.0)),
        pl.Series("hi", hi), pl.Series("lo68", lo68), pl.Series("hi68", hi68),
        pl.Series("n_cal", n_cal), pl.Series("n_pool", n_pool),
        pl.Series("fell_back", fell), pl.Series("scored", scored))


# sqrt(2) times the binomial standard error under p = 0.80, the form #310 pre-registered for a
# group: the calibration draw's share of the variance equals the test binomial's. Printed
# beside a group and decides nothing here -- whether a group may rule is #310's.
_SIGMA_SCALE = math.sqrt(2.0)


def _published_row(label: str, g: pl.DataFrame, position: str | None) -> dict[str, Any]:
    """One line of the published interval's table, over the scored rows of `g`.

    `position` is the group's own code, or `None` for a pool -- a row that carried only
    `group` made a reader know the sizes to tell which 0.757 was QB (#309). `n_cal` is the
    median of the rows' own-position calibration counts and `n_cal_min` the smallest;
    `n_fallback` is how many rows borrowed the pooled window and `fallback_cells` names the
    `(season, week)` cells they sit in, so a group on pooled calibration says so instead of
    being read as a test of the conditional claim. `sigma` is the deviation from 0.80 over
    sqrt(2) times the binomial standard error at 0.80.
    """
    g = g.filter(pl.col("scored"))
    nn = int(g.height)
    if not nn:
        return {"group": label, "position": position, "n": 0}
    y = g["points"].to_numpy()
    lo, hi = g["lo"].to_numpy(), g["hi"].to_numpy()
    cov = float(np.mean((y >= lo) & (y <= hi)))
    nominal = LEVELS[0][1] - LEVELS[0][0]
    se = _SIGMA_SCALE * math.sqrt(nominal * (1.0 - nominal) / nn)
    fb = g.filter(pl.col("fell_back"))
    cells = sorted({(int(a), int(b)) for a, b in zip(fb["season"], fb["week"], strict=True)})
    return {
        "group": label, "position": position, "n": nn,
        "n_cal": int(np.median(g["n_cal"].to_numpy())),
        "n_cal_min": int(np.min(g["n_cal"].to_numpy())),
        "n_fallback": int(fb.height), "fallback_cells": [list(c) for c in cells],
        "pooled_calibration": bool(fb.height),
        "cov80": cov,
        "cov68": float(np.mean((y >= g["lo68"].to_numpy()) & (y <= g["hi68"].to_numpy()))),
        "below_p10": float(np.mean(y < lo)), "above_p90": float(np.mean(y > hi)),
        "deviation": cov - nominal, "se": se, "sigma": (cov - nominal) / se,
        "clipped_share": float(np.mean(g["raw_lo"].to_numpy() <= 0.0)),
    }


def published_table(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The published interval's coverage table: one row per position, then the pool."""
    rows = [_published_row(p, g.filter(pl.col("position") == p), p) for p in POSITIONS
            if g.filter(pl.col("position") == p).height]
    rows.append(_published_row("all", g, None))
    return rows


def published_floor_split(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The same coverage split on whether the published lower bound is pinned at zero.

    A diagnostic and not a population (#309): it says where the interval is working for a
    reason other than being right, and it decides nothing.
    """
    out = []
    for name, sel in (("clipped at zero", pl.col("raw_lo") <= 0.0),
                      ("strictly positive", pl.col("raw_lo") > 0.0)):
        part = g.filter(pl.col("scored") & sel)
        if part.height:
            out.append(_published_row(name, part, None))
    return out


def published_by_prior(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The published interval's coverage by games of evidence behind the centre.

    The conditional the estimation error is *for*: a parametric interval built on four games
    under-covers at the thin end and over-covers at the thick, and a marginal 80% hides the
    cancellation. Only where the centre was estimated (`n_prior` is null under the lookahead
    centre, and the cut has nothing to say there).
    """
    if "n_prior" not in g.columns or g["n_prior"].null_count() == g.height:
        return []
    out = []
    for label, low, high in PRIOR_BUCKETS:
        sel = pl.col("n_prior") >= low
        if high is not None:
            sel = sel & (pl.col("n_prior") <= high)
        part = g.filter(sel)
        if part.filter(pl.col("scored")).height:
            out.append(_published_row(label, part, None))
    return out


def gate_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The row the gate reads: the pool of everything that was scored, clipped and unclipped.

    Its own function so that "the gate measures the whole board" is a seam with a name -- a
    gate that reads a subset of the board is one line changed here, and
    `test_the_gate_reads_the_whole_scored_board_not_a_subset` is the control that plants it.
    """
    return next(r for r in rows if r["group"] == GATE_SUBSET)


def verdict(row: dict[str, Any], claim: float = CLAIMED_COV80, band: float = BAND) -> str:
    """COVERS, UNDER-COVERS or OVER-COVERS, against the pre-registered band around the claim.

    The claim and not the label (#289): `CLAIMED_COV80` is what the doc says the interval
    covers, and the verdict is whether the deployed function still does. Three answers
    rather than a boolean because the two failures have opposite fixes: an interval that is
    too narrow makes a lineup rule overconfident, and one that is too wide makes it refuse to
    distinguish players it could -- and either way the published claim is the thing to
    restate first.
    """
    gap = row["cov80"] - claim
    if abs(gap) <= band:
        return "COVERS"
    return "UNDER-COVERS" if gap < 0 else "OVER-COVERS"


def _mde(n: int) -> float:
    """The smallest miss from 0.80 a group of `n` weeks can resolve at 80% power at one look:
    (z_alpha + z_0.8) * sqrt(0.32 / n), the variance taken under p = 0.80 (#310). 0.0200 at
    8,377, 0.0144 on the whole 16,061-week board -- printed beside every group."""
    zb = statistics.NormalDist().inv_cdf(0.80)
    return (Z_PER_LOOK + zb) * math.sqrt(0.32 / n) if n else float("nan")


def group_verdict(row: dict[str, Any]) -> dict[str, Any]:
    """One position's audit reading: a verdict only at `N_MIN_GROUP` player-weeks.

    At or above it, the verdict is whether the 95% interval at sqrt(2) x the binomial SE
    (`row["se"]`, which already carries the root two) contains 0.80. Below it the group
    returns **NOT-RUNNABLE** and reports its deviation and sigma -- no verdict, not
    invisibility (rule 16: nothing ships on a conditional verdict, so this is not the harmful
    case). `n_required` is printed so the exemption names itself.
    """
    n = int(row["n"])
    out = {"group": row["group"], "position": row.get("position"), "n": n,
           "n_cal": row.get("n_cal"), "cov80": row["cov80"], "deviation": row["deviation"],
           "se": row["se"], "sigma": row["sigma"], "mde": _mde(n),
           "n_required": N_MIN_GROUP, "pooled_calibration": bool(row.get("n_fallback"))}
    if n < N_MIN_GROUP:
        out["verdict"] = "NOT-RUNNABLE"
    elif abs(row["deviation"]) <= GROUP_Z * row["se"]:
        out["verdict"] = "COVERS"
    else:
        out["verdict"] = "UNDER-COVERS" if row["deviation"] < 0 else "OVER-COVERS"
    return out


def audit_verdict(rows: list[dict[str, Any]], claim: float = CLAIMED_COV80,
                  band: float = BAND) -> dict[str, Any]:
    """The audit-time verdict (#310, adopted 2026-09-17): marginal 80 +/- 2 on the whole
    board, and per position the sqrt(2) interval -- a verdict only at `N_MIN_GROUP`.

    `passes` is the marginal COVERS and no *runnable* group missing; a NOT-RUNNABLE group
    neither passes nor fails the audit, and says so. Only the position rows and the pool are
    read: the clipped/unclipped split is a diagnostic the maintainer decided on 2026-10-07 not
    to make a claim, and its rows (no `position`) are not inputs here.
    """
    pool = next(r for r in rows if r["group"] == GATE_SUBSET)
    marginal = verdict(pool, claim=claim, band=band)
    groups = [group_verdict(r) for r in rows if r.get("position")]
    return {"marginal": marginal, "claim": claim, "band": band, "n": pool["n"],
            "cov80": pool["cov80"], "mde": _mde(int(pool["n"])), "groups": groups,
            "passes": marginal == "COVERS" and all(
                g["verdict"] in ("COVERS", "NOT-RUNNABLE") for g in groups)}


def smoke_alarm(got: dict[str, Any]) -> list[str]:
    """What the weekly step still fails the run on: a broken pipeline, not a drift.

    Marginal coverage further than `SMOKE_BAND` from the claim -- eight MDEs, so noise and
    drift cannot reach it -- or a group whose rows could not be scored at all. Empty means the
    week is fine. Supersedes #273's statistical role (a verdict read weekly is a sequential
    test) and keeps its purpose: a run that measures nothing sensible is a red run.
    """
    reasons = []
    miss = got["gate_cov80"] - got["gate_claim"]
    if abs(miss) > SMOKE_BAND:
        reasons.append(f"marginal coverage {got['gate_cov80']:.1%} is {abs(miss):.1%} from the "
                       f"claimed {got['gate_claim']:.0%}, further than {SMOKE_BAND:.0%}")
    # Every position the weekly laws are fitted for, not only the ones the table happens to
    # carry: `published_table` emits a position only if it has board rows, so a position with
    # nothing at all never appears and an alarm over the table alone is blind to it (#438).
    have = {r["group"]: r for r in got["by_position"]}
    for pos in POSITIONS:
        r = have.get(pos)
        if r is None:
            reasons.append(f"{pos} is missing from the table (no rows at all, so none "
                           f"calibrated)")
        elif not r["n"]:
            reasons.append(f"{pos} reported zero calibration rows (nothing scored)")
        elif not r.get("n_cal"):
            reasons.append(f"{pos} reported zero calibration rows of its own")
    return reasons


def _season_coverage(g: pl.DataFrame) -> dict[str, Any] | None:
    """The newest scored season's own coverage, reported beside the full-history figure and
    never gated (#310): a season is a look at one year, and the claim is the history's."""
    s = g.filter(pl.col("scored"))
    if not s.height:
        return None
    last = int(cast(int, s["season"].max()))
    row = _published_row(str(last), s.filter(pl.col("season") == last), None)
    return {"season": last, "n": row["n"], "cov80": row["cov80"]}


def _uncalibrated_cells(g: pl.DataFrame) -> list[list[int]]:
    left = g.filter(~pl.col("scored"))
    return [list(c) for c in sorted({(int(a), int(b)) for a, b in
                                     zip(left["season"], left["week"], strict=True)})]


def _estimation_only(g: pl.DataFrame) -> dict[str, Any]:
    """Coverage of the parametric interval at the estimation-aware scale, uncalibrated.

    The same rows the parametric table reads, the same clipped bounds, with only
    `sd -> sd_pred` changed: what estimation error closes by itself, before the empirical
    calibration is asked for anything.
    """
    y = g["points"].to_numpy()
    cov = (y >= g["pe10"].to_numpy()) & (y <= g["pe90"].to_numpy())
    clipped = g["p10_raw"].to_numpy() <= 0.0
    return {"n": int(g.height), "cov80": float(np.mean(cov)),
            "cov80_unclipped": float(np.mean(cov[~clipped])) if (~clipped).any() else None,
            "cov80_clipped": float(np.mean(cov[clipped])) if clipped.any() else None}


def measure(stats: pl.DataFrame, centre: Centre = "prior", *,
            min_weeks: int = MIN_WEEKS, min_prior: int = MIN_PRIOR,
            min_mu: float = MIN_MU, band: float = BAND,
            claim: float = CLAIMED_COV80, window: int | None = CAL_WINDOW,
            floor: int = conformal.MIN_GROUP_CALIBRATION) -> dict[str, Any]:
    """The whole weekly-interval measurement, as one dict.

    The graded interval is the *published* one (#309): `calibrate`'s conformalised bounds at
    the estimation-aware scale, over every player-week that has a calibration. `by_position`,
    `floor_split` and `gate_*` read it. The parametric interval the gate graded until #309 --
    no estimation error, a clipped parametric lower bound, the unclipped weeks only -- is kept
    whole under `parametric`, so the prior values are in the file and the two are on one set
    of rows; `estimation_only` is the middle step.

    `nominal` is the label the interval is served under; `gate_claim` is what the doc says
    it covers and what `verdict` was read against. Both are carried because a reader of
    `COVERS` beside `77.4%` needs the number it was compared with on the same line (#289).
    `n` is the board; `gate_n` is how much of it was scored and `n_uncalibrated` is the rest,
    with the cells -- the gate says how much of the board it graded.
    """
    g = graded(centred(player_weeks(stats), centre, min_weeks=min_weeks,
                       min_prior=min_prior, min_mu=min_mu))
    cal = calibrate(g, window=window, floor=floor)
    if not int(cal["scored"].sum()):
        raise NotEnoughWeeks(
            f"no player-week had a calibration of at least {floor} rows in a window of "
            f"{window} cells; the board has {g.height} player-weeks")
    rows = published_table(cal)
    split = floor_split(g)
    param_rows = table(g)
    param_gated = next((r for r in split if r["group"] == "strictly positive"),
                       param_rows[-1])
    gated = gate_population(rows)
    # Each group row carries the audit reading beside its figures (#310): a verdict at
    # N_MIN_GROUP, NOT-RUNNABLE below it, never a silent omission.
    for r in rows:
        r["verdict"] = (verdict(r, claim=claim, band=band) if r["group"] == GATE_SUBSET
                        else group_verdict(r)["verdict"])
        r["mde"] = _mde(int(r["n"]))
    clipped_param = next((r["n"] for r in split if r["group"] == "clipped at zero"), 0)
    return {
        "centre": centre, "lookahead": centre == "realised", "interval": "conformal",
        "n": int(g.height), "nominal": {"cov80": 0.80, "cov68": 0.68},
        "band": band, "by_position": rows, "floor_split": published_floor_split(cal),
        "by_prior": published_by_prior(cal),
        "calibration": {"window": window, "floor": floor, "scale": "sd*sqrt(1+1/n)",
                        "unit": "(season, week) cells"},
        "n_uncalibrated": int(g.height - gated["n"]),
        "uncalibrated_cells": _uncalibrated_cells(cal),
        "gate_subset": GATE_SUBSET, "gate_n": gated["n"], "gate_cov80": gated["cov80"],
        "gate_clipped_share": gated["clipped_share"],
        "gate_claim": claim, "verdict": verdict(gated, claim=claim, band=band),
        "season_coverage": _season_coverage(cal),
        # Reported beside the verdict and read by none (the 2026-10-07 decision, #310): the
        # clipped/unclipped split is not a claim, and a clip-conditional claim would be a new
        # ticket with its own pre-registered threshold.
        "diagnostics": ["floor_split", "by_prior"],
        "estimation_only": _estimation_only(g),
        "parametric": {
            "by_position": param_rows, "floor_split": split,
            "gate_subset": "unclipped", "gate_n": param_gated["n"],
            "gate_cov80": param_gated["cov80"],
            "clipped_share": clipped_param / g.height,
            "verdict": verdict(param_gated, claim=claim, band=band)},
    }


# --- the backtest's key, and the current season's row (#423) --------------


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()
                          ).hexdigest()[:12]


def backtest_key(seasons: Sequence[int], centre: Centre = "prior", *,
                 min_prior: int = MIN_PRIOR, band: float = BAND,
                 claim: float = CLAIMED_COV80) -> dict[str, Any] | None:
    """What a stored backtest must have been measured under to be read back, or `None`.

    `None` for a span that is not closed (it holds the current season or later): such a
    measurement can still move, so it is never read back. Otherwise the pair the issue names:
    `config_digest` over every setting the figure depends on and `code_digest` over the modules
    that compute it, and `data_digest` over the declared identity of the span read.
    """
    if not seasons or max(seasons) >= CURRENT_SEASON:
        return None
    config = {"centre": centre, "min_prior": min_prior, "min_weeks": MIN_WEEKS,
              "min_mu": MIN_MU, "levels": LEVELS, "window": CAL_WINDOW,
              "floor": conformal.MIN_GROUP_CALIBRATION, "band": band, "claim": claim,
              "n_min_group": N_MIN_GROUP, "group_z": GROUP_Z, "gate_subset": GATE_SUBSET,
              "positions": POSITIONS, "prior_buckets": PRIOR_BUCKETS}
    return {"config_digest": _digest(config), "code_digest": code_digest(_CODE),
            "data_digest": _digest(["nflverse.player_stats", sorted(seasons)]),
            "seasons": sorted(seasons)}


def survivor_key(seasons: Sequence[int]) -> str | None:
    """The same idea for the survivor block: config, code and the closed span of schedules."""
    if not seasons or max(seasons) >= CURRENT_SEASON:
        return None
    config = {"edges": SPREAD_EDGES, "spread": SURVIVOR_SPREAD, "min_bucket": MIN_BUCKET}
    return _digest([_digest(config), code_digest(_SURVIVOR_CODE),
                    _digest(["nflverse.schedules", sorted(seasons)])])


_STATE_ONLY = ("name", "generated_at", "survivor", "audit", "audit_looks", "current_season",
               "backtest_key")


def read_back_backtest(key: dict[str, Any] | None, path: Path | None = None
                       ) -> dict[str, Any] | None:
    """The stored backtest if it was measured under exactly `key`, else `None`.

    A key that is `None` (a span still open), a file with no key (written before #423), or any
    digest that differs is a miss, and a miss is a re-measurement -- never a stale read.
    """
    have = _existing(path or ARTIFACT)
    if key is None or have.get("backtest_key") != json.loads(json.dumps(key)):
        return None
    if "verdict" not in have or "by_position" not in have:
        return None
    return {k: v for k, v in have.items() if k not in _STATE_ONLY}


def completed_weeks(schedule: pl.DataFrame, stats: pl.DataFrame, season: int,
                    as_of: date) -> int:
    """How many of `season`'s regular-season weeks are complete: the longest run of weeks from
    week 1 whose last game was played more than `LAG_DAYS` before `as_of` *and* whose rows are in
    `stats`. Dates then data (#437): a week the calendar says is over but the data lacks is not
    complete, and a week with a game still to play is not either (a Saturday run holds
    Thursday's game of a week that has three days left).
    """
    from hub.fetch import consensus
    sched = schedule.filter(pl.col("season") == season)
    have = set(stats.filter(pl.col("season") == season)["week"].to_list())
    weeks = sorted({int(w) for w in sched.filter(pl.col("game_type") == "REG")["week"]})
    through = 0
    for w in weeks:
        last = consensus.last_game_day(sched, w)
        if last is None or as_of <= last + timedelta(days=LAG_DAYS) or w not in have:
            break
        through = w
    return through


def look_date_reached(look: int, schedule: pl.DataFrame, as_of: date,
                      season: int | None = None) -> str | None:
    """`None` if the calendar has passed look `look`'s week; else the reason it has not.

    Reads the schedule only -- no outcome is loaded to say not yet (the shape of #432's
    horizon). The data half is `completed_weeks`, run once the stats are in.
    """
    from hub.fetch import consensus
    season = CURRENT_SEASON if season is None else season
    week = LOOK_WEEKS[look]
    last = consensus.last_game_day(schedule.filter(pl.col("season") == season), week)
    if last is None:
        return f"the {season} schedule has no week {week}"
    if as_of <= last + timedelta(days=LAG_DAYS):
        return (f"look {look} is taken after week {week} is complete; its last game is {last} "
                f"and this run is {as_of}")
    return None


def _current_row(label: str, position: str | None, g: pl.DataFrame) -> dict[str, Any]:
    """One group of the 2026 row: its figures and its MDE, and **no verdict** (#460).

    The pre-registration gives this row looks 2 and 3 "and no other verdict", so whether a
    position may rule is left to `--audit --look 2|3`, which reads `group_verdict` at the look.
    Here a position says only how many weeks it has against the `n_required` that look will ask
    for (`runnable`), so a position that reached 8,377 weeks is not handed a COVERS or a miss by
    the weekly run.
    """
    row = _published_row(label, g, position)
    n = int(row["n"])
    row["mde"] = _mde(n) if n else None
    if position:
        row.update(runnable=n >= N_MIN_GROUP, n_required=N_MIN_GROUP)
    return row


def measure_current(stats: pl.DataFrame, schedule: pl.DataFrame, as_of: date,
                    season: int | None = None, *, min_prior: int = MIN_PRIOR,
                    window: int | None = CAL_WINDOW,
                    floor: int = conformal.MIN_GROUP_CALIBRATION) -> dict[str, Any]:
    """The current season's completed weeks, scored by the published interval: the labelled
    out-of-sample row (#423). Not pooled into the backtest; no marginal verdict attaches.

    `stats` carries the current season and the `CURRENT_HISTORY` before it, so the rolling
    calibration window is full. Only cells of weeks that are complete are scored: a later
    partial week sits in the data and is left out. Positions carry n, coverage, sigma and the MDE
    and whether they have `N_MIN_GROUP` weeks (`runnable`); no group carries a verdict -- those
    are read at looks 2 and 3 (`--audit --look`), not here.
    """
    season = CURRENT_SEASON if season is None else season
    through = completed_weeks(schedule, stats, season, as_of)
    g = graded(centred(player_weeks(stats), "prior", min_prior=min_prior))
    cal = calibrate(g, window=window, floor=floor)
    cur = cal.filter((pl.col("season") == season) & (pl.col("week") <= through))
    by_pos = []
    for p in POSITIONS:
        part = cur.filter(pl.col("position") == p)
        if part.height:
            by_pos.append(_current_row(p, p, part))
        else:
            by_pos.append({"group": p, "position": p, "n": 0, "mde": None,
                           "runnable": False, "n_required": N_MIN_GROUP})
    pool = _current_row(GATE_SUBSET, None, cur)
    keys = ("n", "cov80", "cov68", "clipped_share", "deviation", "se", "sigma", "mde")
    return {
        "label": "out-of-sample", "season": season, "weeks_complete": through,
        "seasons_read": [season - CURRENT_HISTORY + i for i in range(CURRENT_HISTORY + 1)],
        "claim": CLAIMED_COV80, "band": BAND,
        **{k: pool.get(k) for k in keys},
        "by_position": [*by_pos, pool],
        "looks": {str(k): {"after_week": w, "reached": through >= w}
                  for k, w in sorted(LOOK_WEEKS.items())},
    }


# --- the survivor price ---------------------------------------------------


def _bucketed(sides: pl.DataFrame, edges: Sequence[float]) -> list[dict[str, Any]]:
    """The reliability rows by spread, each labelled and each saying whether it is thin.

    `reliability_by` bins on the favourite's spread (a non-negative `spread` is the
    favourite's side of a game, so the count is one per game and the bins never see the
    underdog rows). Labels come from `SPREAD_LABELS` when the edges are the pre-registered
    ones and from the bin string otherwise, so a caller grading at other edges gets an
    honest label and not one written for different edges. `thin` is `n < MIN_BUCKET`, and
    an empty bucket is kept and marked rather than dropped (#293).
    """
    rows = reliability_by(sides, list(edges), on="spread", prob="win_prob", outcome="won")
    labels = (SPREAD_LABELS if tuple(edges) == tuple(SPREAD_EDGES)
              else tuple(r["bin"] for r in rows))
    return [{**r, "label": label, "thin": r["n"] < MIN_BUCKET}
            for r, label in zip(rows, labels, strict=True)]


def survivor_price(schedules: pl.DataFrame, edges: Sequence[float] = SPREAD_EDGES,
                   *, sd: float | None = None) -> dict[str, Any]:
    """Survivor's win probability, graded by spread bucket.

    Both sides of every game go in, home-relative spread negated for the away row, which is
    exactly the grid `survivor.grid_from_schedule` builds -- a diagram over home rows alone
    would grade one half of the pick space and survivor picks from both. The buckets then
    read the favourite's side of each game, so `n` in a bucket is games, and the realised
    rate is the favourite's win rate against the price it was given -- a survivor pick wins
    outright or it does not; nothing here is about covering the spread (#293).

    The headline -- `favourite_*` and `verdict` -- reads `SURVIVOR_SPREAD` and never a
    bucket, so the buckets can be re-cut without the verdict moving, and were (#293).

    The price and the tie convention are read from the modules that own them
    (`hub.models.margin`), not restated: a survivor pick and a weekly prediction are not
    allowed to disagree about a game they both price, and a grader holding a third copy of
    the formula is how they would.
    """
    from hub.models.margin import home_win_prob, home_won
    from hub.models.market import MARGIN_SD

    need = {"spread_line", "result"}
    missing = sorted(need - set(schedules.columns))
    if missing:
        raise ValueError(f"schedules is missing {missing}")
    scored = home_won(schedules.drop_nulls("spread_line").select(
        pl.col("spread_line").cast(pl.Float64), pl.col("result").cast(pl.Float64)))
    if scored.is_empty():
        raise NotEnoughWeeks("no completed game carries both a spread and a result")
    s = scored["spread_line"].to_numpy()
    won = scored["home_won"].to_numpy()
    margin_sd = MARGIN_SD if sd is None else sd
    p = home_win_prob(s, margin_sd)
    sides = pl.DataFrame({"spread": np.concatenate([s, -s]),
                          "win_prob": np.concatenate([p, 1.0 - p]),
                          "won": np.concatenate([won, 1 - won])})
    fav = sides.filter(pl.col("spread") >= SURVIVOR_SPREAD)
    n = fav.height
    pred = float(cast(float, fav["win_prob"].mean())) if n else float("nan")
    act = float(cast(float, fav["won"].mean())) if n else float("nan")
    se = math.sqrt(act * (1.0 - act) / n) if n else float("nan")
    return {
        "margin_sd": margin_sd, "n_games": scored.height, "n_sides": sides.height,
        "buckets": _bucketed(sides, edges), "min_bucket": MIN_BUCKET,
        "favourite_spread": SURVIVOR_SPREAD, "favourite_n": n,
        "favourite_predicted": pred, "favourite_actual": act,
        "favourite_gap": act - pred, "favourite_se": se,
        "favourite_sigma": ((act - pred) / se) if se else float("nan"),
        "verdict": ("HOLDS" if not se or abs(act - pred) <= 2.0 * se
                    else "UNDER-CONFIDENT" if act > pred else "OVER-CONFIDENT"),
    }


# --- the shape law, decided by CRPS (#292) --------------------------------
#
# Pre-registered in `docs/gate-power.md` on 2026-09-13, before this was written: the rows,
# the two arms, the paired difference, the cluster, the MDE and the ceiling arm are all
# fixed there, and this is the harness that runs them. CRPS is a gate input for exactly this
# decision and a diagnostic everywhere else (#274).

# The declared ceiling arm, by name, on the ceiling line -- distinct from the three gates'
# arms so no two numbers can be tabulated as one (`docs/gate-power.md`, #138). Per position,
# the skew that minimises mean CRPS on the same rows, chosen from `SKEW_GRID` plus the
# deployed value so the bound is a bound. In-sample by construction: it says how much *any*
# per-position skew law could gain over the deployed one here, and the skew-free arm is one
# such law.
CEILING_ARM = "the best per-position skew, chosen on these rows"
SKEW_GRID: tuple[float, ...] = not_an_input(
    tuple(round(0.05 * i, 2) for i in range(41)),
    "the candidate skews the ceiling arm is chosen from, a grading harness's search grid "
    "that no prediction reads")

# #335, ADR-0019's amendment: this gate's within-season repeated-measure unit is the player --
# `docs/method.md` rule 3's own unit, `paired["player_id"]`.
WITHIN: tuple[str, ...] = ("player_id",)

SHAPE_ACTIONS = Actions(
    adopt="The skew-free interval scores better under CRPS: the deployed function is the "
          "maintainer's to change, and #289's claim is re-registered against it first.",
    remove="The skew earns its place: the deployed interval stays as served.",
    show="The skew stays; the CRPS comparison could not remove it.")

# #387: this module's Harness -- one of the seven -- read by `shape_law` through `Harness.run`,
# `run_gate`'s composition (render, stamp, width history) rather than the pure `decide`:
# unlike `margin`'s two verdicts, this is one gate run a caller publishes.
SHAPE_HARNESS = Harness(name="interval_shape", arm_a="skew-free", arm_b="deployed skew",
                        within=WITHIN, ceiling_arm=CEILING_ARM, actions=SHAPE_ACTIONS,
                        unit="CRPS points per player-week", places=4,
                        arm_modules=("hub.models.base", "hub.models.components",
                                     "hub.models.conformal", "hub.models.coverage",
                                     "hub.models.margin", "hub.models.market",
                                     "hub.models.predict", "hub.models.scoring_rules",
                                     "hub.models.volume"))


def shape_scores(g: pl.DataFrame) -> pl.DataFrame:
    """CRPS of the deployed distribution and of the same function without its skew, per row.

    Arm B is `predict.skewed(mu, sd, skew, z)` on the graded moments at the CRPS quantile
    grid, clip included, which is the path `hub.exhibits.weekly_projection.parametric_quantiles`
    takes. Arm A is the same call with the skew zeroed -- which `skewed` floors at `MIN_SKEW`, so
    the arm is exactly what would be served with `WEEKLY_SKEW` removed, floor and all, and not a
    normal written out beside it. `diff` is deployed minus skew-free: positive when the skew-free
    arm scores lower, which is the sign `experiment.gate` adopts on.

    `ceiling_diff` is deployed minus the oracle -- the best per-position skew on these rows,
    `CEILING_ARM` -- and `best_skew` says which skew that was, so a reader can see how far
    the deployed law sits from the in-sample optimum.
    """
    z = normal_quantile(quantile_levels())[None, :]
    mu, sd, sk = (g[c].to_numpy().astype(float)[:, None] for c in ("mu", "sd", "skew"))
    y = g["points"].to_numpy().astype(float)
    pos = g["position"].to_numpy()
    deployed = crps_from_quantiles(predict.skewed(mu, sd, sk, z), y)
    skewfree = crps_from_quantiles(predict.skewed(mu, sd, 0.0, z), y)
    oracle = np.empty_like(deployed)
    best = np.empty_like(deployed)
    for name in np.unique(pos):
        rows = pos == name
        candidates = sorted(set(SKEW_GRID) | {float(sk[rows][0, 0])})
        scored = {c: crps_from_quantiles(predict.skewed(mu[rows], sd[rows], c, z), y[rows])
                  for c in candidates}
        pick = min(candidates, key=lambda c: float(scored[c].mean()))
        oracle[rows] = scored[pick]
        best[rows] = pick
    return g.select("season", "player_id", "week", "position", "points", "p10_raw").with_columns(
        pl.Series("crps_deployed", deployed), pl.Series("crps_skewfree", skewfree),
        pl.Series("diff", deployed - skewfree), pl.Series("ceiling_diff", deployed - oracle),
        pl.Series("best_skew", best))


def shape_law(stats: pl.DataFrame, *, min_weeks: int = MIN_WEEKS, min_prior: int = MIN_PRIOR,
              min_mu: float = MIN_MU, seed: int = 0,
              ledger: Ledger | None = None) -> tuple[GateRun, pl.DataFrame, dict[str, Any]]:
    """The pre-registered comparison: one gate run, the paired rows, and the pooled diagnostic.

    The rows are the coverage gate's -- prior centre, the same filters -- restricted to the
    unclipped subset it reads, fixed in `docs/gate-power.md` so the subset cannot be chosen
    after the sign is seen. The season is the cluster, the MDE comes from the interval's own
    bootstrap, and the ceiling is `CEILING_ARM` on the same rows. The pooled figure over every
    row, clipped weeks included, is returned beside it as a diagnostic and decides nothing.

    `ledger` replaces `width_path` (#385) -- `None` is `run_gate`'s own default, a file-backed
    `hub.ledger.Ledger()` writing to `hub.ledger.WIDTH_STATE`. `main` builds its own explicit
    `Ledger` from this module's own `WIDTH_STATE`, read at call time rather than defaulted
    here, so a test that monkeypatches `coverage.WIDTH_STATE` still isolates it.
    """
    g = graded(centred(player_weeks(stats), "prior", min_weeks=min_weeks,
                       min_prior=min_prior, min_mu=min_mu))
    scored = shape_scores(g)
    paired = scored.filter(pl.col("p10_raw") > 0.0)
    pooled = {"n": int(scored.height), "mean": float(cast(float, scored["diff"].mean()))}
    # #384: the arm is the filters and the seed -- what `shape_law` takes besides the frame.
    run = SHAPE_HARNESS.run(
        paired, seed=seed, ledger=ledger,
        recipe=_recipe(min_weeks=min_weeks, min_prior=min_prior, min_mu=min_mu, seed=seed))
    return run, paired, pooled


# --- what reads the answer ------------------------------------------------


def write_summary(result: dict[str, Any], path: Path | None = None) -> Path:
    """Leave the weekly measurement where a publisher can read it.

    ADR-0007's second consequence: the result is written rather than printed, so a later
    disagreement is between two recorded runs instead of between two memories. A survivor
    block already in the file is kept (#273): one file, two verdicts, one reader.
    """
    p = path or ARTIFACT
    have = _existing(p)
    # The survivor block and the last audit are not this run's to erase: a weekly write is a
    # report, and the audit's verdict stands until the next audit replaces it (#310). The
    # current-season row is kept when this run could not measure one (a failed fetch is last-good,
    # not an erasure) and replaced when it could.
    keep = {k: have[k] for k in ("survivor", "audit", "audit_looks") if have.get(k)}
    last_good = {k: have[k] for k in ("current_season",) if have.get(k) and k not in result}
    # **A run that changes nothing writes nothing (#423).** The body is built with the stamp the
    # file already carries and rendered at `ROUND_DIGITS`; if that is the file's own text the
    # file is left alone -- the stamp is the time the *numbers* last moved, not the time anyone
    # looked, and the git history, which is the pre-registration, carries no noise.
    body = {"name": "interval_coverage", "generated_at": have.get("generated_at"),
            **result, **last_good, **keep}
    text = jsonio.dumps(rounded_body(body), indent=2)
    if _same_text(p, text):
        return p
    body["generated_at"] = jsonio.stamp()
    atomic.write_text(p, jsonio.dumps(rounded_body(body), indent=2))
    return p


# **A recorded look is never rounded and never rewritten.** `audit` and `audit_looks` are the
# pre-registration's evidence -- the figure a look was taken on, as computed, stamped when it was
# taken -- so a later weekly write carries them through untouched rather than re-rendering them at
# `ROUND_DIGITS`, which would change values that a recorded look has already fixed (look 1 of
# 2026-10-07 reads 0.80035718841689). Everything else in the file is a report and is rounded.
_RECORDED = ("audit", "audit_looks")


def rounded_body(body: dict[str, Any]) -> dict[str, Any]:
    """`body` with its reports rounded to `ROUND_DIGITS` and its recorded looks left as they are."""
    return {k: (v if k in _RECORDED else round_values(v)) for k, v in body.items()}


def round_values(x: Any) -> Any:
    """Every float in `x`, at any depth, rounded to `ROUND_DIGITS` (#423)."""
    if isinstance(x, float):
        return round(x, ROUND_DIGITS) if math.isfinite(x) else x
    if isinstance(x, dict):
        return {k: round_values(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_values(v) for v in x]
    return x


def _same_text(p: Path, text: str) -> bool:
    try:
        return p.read_text() == text
    except OSError:
        return False


def write_audit(got: dict[str, Any], audit: dict[str, Any], look: int,
                path: Path | None = None, *, reads: str = "backtest") -> Path:
    """Record an audit's verdict beside the weekly measurement it was read from (#310).

    The block carries the claim the verdict was compared against (#289's rule), the look
    number and alpha per look, and every group's position, n, n_cal, coverage, sigma and
    verdict-or-NOT-RUNNABLE -- what an auditor needs to read the verdict without the run.

    `reads` says which row the look read: `"backtest"` (look 1; `got` is the weekly measurement,
    written beside it as ever) or `"current_season"` (looks 2 and 3, #423; `got` is the
    out-of-sample row, which is *not* the weekly summary and must not overwrite the backtest the
    file's top level holds -- the weekly run writes that row under `current_season`).
    """
    p = path or ARTIFACT
    current = reads == "current_season"
    if not current:
        p = write_summary(got, p)
    block = {"look": look, "looks": LOOKS, "alpha_per_look": ALPHA_PER_LOOK,
             "z_per_look": Z_PER_LOOK, "taken_at": jsonio.stamp(), "claim": audit["claim"],
             "band": audit["band"], "marginal": audit["marginal"], "passes": audit["passes"],
             "n": audit["n"], "cov80": audit["cov80"], "mde": audit["mde"], "reads": reads,
             "season_coverage": ({"season": got["season"], "n": got["n"], "cov80": got["cov80"],
                                  "weeks_complete": got["weeks_complete"]}
                                 if current else got.get("season_coverage")),
             "groups": audit["groups"]}
    have = _existing(p)
    taken = looks_taken(p)
    # Refuse, never overwrite: a look is spent when it is written, and the three looks are
    # the claim's whole alpha budget (#310). The CLI refuses earlier; this is the seam.
    if look in taken:
        raise LookAlreadyTaken(f"look {look} is already recorded in {p.name}")
    record = {"look": look, "taken_at": block["taken_at"], "marginal": block["marginal"],
              "passes": block["passes"], "cov80": block["cov80"], "n": block["n"],
              "claim": block["claim"], "reads": reads}
    atomic.write_text(p, jsonio.dumps(rounded_body(
        {"name": "interval_coverage", **have, "audit": block,
         "audit_looks": [*have.get("audit_looks", []), record]}), indent=2))
    return p


class LookAlreadyTaken(Exception):
    """The audit look asked for is already recorded; a look is never taken twice."""


def looks_taken(path: Path | None = None) -> list[int]:
    """The audit looks already recorded in the artifact, in the order they were taken."""
    got = _existing(path or ARTIFACT).get("audit_looks")
    return [int(r["look"]) for r in got if isinstance(r, dict) and "look" in r] \
        if isinstance(got, list) else []


def write_survivor(result: dict[str, Any], path: Path | None = None) -> Path:
    """Leave the survivor verdict beside the weekly one, in the same file (#273).

    A companion block, not a summary of its own: each block carries its own `generated_at`,
    and `published_summary` reads nothing until the weekly measurement has been written,
    because the page's consumers key on the weekly `verdict`. The slate runs `--measure
    --survivor --write` together, which is the order that leaves both. Like the summary, a
    block that has not changed keeps its stamp and writes nothing (#423); `key` is the digest
    it was measured under, which is what lets the next run read it back.
    """
    p = path or ARTIFACT
    have = _existing(p)
    block = {k: result.get(k) for k in
             ("verdict", "favourite_spread", "favourite_predicted", "favourite_actual",
              "favourite_gap", "favourite_sigma", "favourite_n", "n_games", "margin_sd",
              # The buckets ride along whole (#293): the page shows where the price holds
              # and where it is thin, and it cannot without the count in each.
              "buckets", "min_bucket", "key")}
    prior = have.get("survivor")
    block["generated_at"] = prior.get("generated_at") if isinstance(prior, dict) else None
    text = jsonio.dumps(rounded_body({"name": "interval_coverage", **have, "survivor": block}),
                        indent=2)
    if _same_text(p, text):
        return p
    block["generated_at"] = jsonio.stamp()
    atomic.write_text(p, jsonio.dumps(rounded_body(
        {"name": "interval_coverage", **have, "survivor": block}), indent=2))
    return p


def _existing(p: Path) -> dict[str, Any]:
    try:
        got = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def published_summary(path: Path | None = None) -> dict[str, Any] | None:
    """The last measurement, small enough to publish, or None if none has been made.

    None rather than a raise, and last-good rather than a fresh run: the publisher runs
    unattended on a Sunday, and a page that cannot render because a research measurement is
    missing is the failure mode `CLAUDE.md` names. An unreadable file is the same nothing as
    an absent one -- what the page needs is a field or no field, not a traceback.
    """
    p = path or ARTIFACT
    try:
        got = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(got, dict) or "verdict" not in got:
        return None
    out = {k: got.get(k) for k in
           ("centre", "lookahead", "n", "gate_subset", "gate_n", "gate_cov80", "gate_claim",
            "band", "verdict", "generated_at",
            # #309: what the page needs to say that the number is the whole scored board and
            # how much of the board that is. Absent in an artifact written before it, which
            # the page then words as it always did.
            "interval", "gate_clipped_share", "n_uncalibrated", "season_coverage",
            # #423: the current season's labelled out-of-sample row, beside the backtest and
            # never pooled with it. Absent before the first run that measured one.
            "current_season")}
    if isinstance(got.get("audit"), dict):
        out["audit"] = got["audit"]
    # An artifact written before the claim was carried (#289) had its verdict read against
    # the label, so that is the claim it is published with -- a reader printing `verdict`
    # beside `gate_claim` then says what that run actually compared.
    if out["gate_claim"] is None:
        out["gate_claim"] = LEVELS[0][1] - LEVELS[0][0]
    if isinstance(got.get("survivor"), dict):
        out["survivor"] = got["survivor"]
    return out


def _print_published(rows: list[dict[str, Any]]) -> None:
    """The published interval's table: coverage, n_cal, and the deviation in sigma (#309)."""
    print(f"    {'':<18}{'n':>7}{'n_cal':>7}{'80%':>8}{'68%':>8}{'<p10':>8}{'>p90':>8}"
          f"{'clipped':>9}{'dev':>8}{'sigma':>7}")
    for r in rows:
        if not r["n"]:
            print(f"    {r['group']:<18}{0:>7}   (nothing scored)")
            continue
        n_cal = f"{r['n_cal']:>7,}" if r.get("n_cal") else f"{'--':>7}"
        pooled = ("  pooled calibration on " + f"{r['n_fallback']:,} rows"
                  if r["n_fallback"] else "")
        print(f"    {r['group']:<18}{r['n']:>7,}{n_cal}{r['cov80']:>8.1%}{r['cov68']:>8.1%}"
              f"{r['below_p10']:>8.1%}{r['above_p90']:>8.1%}{r['clipped_share']:>9.1%}"
              f"{r['deviation']:>+8.1%}{r['sigma']:>+7.1f}{pooled}")


def _stats(seasons: Sequence[int], cache: Path | None) -> pl.DataFrame:
    from hub.fetch import nflverse
    return nflverse.load("player_stats", seasons=list(seasons),
                         cols=list(nflverse.PLAYER_STATS_COLS), cache=cache)


def _schedules(seasons: Sequence[int], cache: Path | None) -> pl.DataFrame:
    from hub.fetch import nflverse
    return nflverse.load("schedules", seasons=list(seasons), cache=cache)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.coverage",
        description="Grade the shipped weekly interval, and the survivor price by spread.")
    ap.add_argument("--measure", action="store_true", help="the weekly interval table")
    ap.add_argument("--survivor", action="store_true",
                    help="survivor win probability by spread bucket")
    ap.add_argument("--gate", action="store_true",
                    help="the weekly step: measure and report, exit 1 only on the smoke alarm "
                         "(marginal >10pp from the claim, or a group with nothing scored)")
    ap.add_argument("--audit", action="store_true",
                    help="the audit-time verdict (#310): marginal 80 +/- 2 on the whole board, "
                         "per position only at N >= 8,377; exit 1 if it fails")
    ap.add_argument("--look", type=int, default=None,
                    help=f"which of the {LOOKS} audit looks this is; required by --audit, "
                         f"and a look already recorded in the artifact is refused (#438)")
    ap.add_argument("--shape", action="store_true",
                    help="the skew law under CRPS, season-clustered, as pre-registered (#292)")
    ap.add_argument("--centre", default="prior", choices=("prior", "realised"),
                    help="'prior' uses only earlier weeks; 'realised' is the document's "
                         "lookahead centre and is kept to reproduce it")
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    ap.add_argument("--min-prior", type=int, default=MIN_PRIOR)
    ap.add_argument("--band", type=float, default=BAND)
    ap.add_argument("--write", action="store_true",
                    help=f"write the result to {ARTIFACT.name} for the publisher")
    ap.add_argument("--cache", default=None, help="raw-cache root; defaults to this repo's")
    ap.add_argument("--as-of", default=None,
                    help="ISO date the run is taken on (default: today, UTC); decides which "
                         "weeks of the current season are complete")
    ap.add_argument("--remeasure", action="store_true",
                    help="measure the closed 2021-2025 backtest and survivor again instead of "
                         "reading back the stored one (a restated nflverse season is the only "
                         "reason; the key cannot see it)")
    a = ap.parse_args(argv)

    if not (a.measure or a.survivor or a.gate or a.shape or a.audit):
        ap.print_help()
        return 0
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    cache = Path(a.cache) if a.cache else None

    as_of = date.fromisoformat(a.as_of) if a.as_of else datetime.now(UTC).date()

    if a.survivor:
        skey = survivor_key(seasons)
        stored = _existing(ARTIFACT).get("survivor")
        if (skey is not None and not a.remeasure and isinstance(stored, dict)
                and stored.get("key") == skey):
            # Measured once per key and read back (#423): the schedules are not pulled.
            print(f"  survivor price: read back from {ARTIFACT.name}, measured "
                  f"{stored.get('generated_at')} under key {skey}; {stored['favourite_n']:,} "
                  f"favourite sides over {stored['n_games']:,} games -> {stored['verdict']}")
        else:
            try:
                sched = _schedules(seasons, cache)
            except Exception as e:
                # Broad on `hub.cli.unavailable`'s reasoning: an empty cache with no network is
                # what a fresh clone hands this command, and the answer to it is a sentence and
                # a non-zero exit rather than nflreadpy's traceback.
                return unavailable("hub.models.coverage", "nflverse schedules", e)
            try:
                got = survivor_price(sched)
            except (ValueError, NotEnoughWeeks) as e:
                print(f"hub.models.coverage: {e}", file=sys.stderr)
                return 1
            got["key"] = skey
            print(f"  survivor price, {got['n_games']:,} games over seasons "
                  f"{seasons[0]}-{seasons[-1]}, margin sd {got['margin_sd']}")
            print(f"    {'spread':<14}{'n':>7}{'predicted':>12}{'actual':>10}{'gap':>10}")
            for b in got["buckets"]:
                # Every bucket, the empty and the thin ones marked rather than dropped (#293).
                rate = (f"{b['predicted']:>12.3f}{b['actual']:>10.3f}{b['gap']:>+10.3f}"
                        if b["n"] else f"{'--':>12}{'--':>10}{'--':>10}")
                thin = f"   under {got['min_bucket']} games" if b["thin"] else ""
                print(f"    {b['label']:<14}{b['n']:>7,}{rate}{thin}")
            print(f"    favourites of {got['favourite_spread']:.0f}+ : predicted "
                  f"{got['favourite_predicted']:.3f}, actual {got['favourite_actual']:.3f}, "
                  f"gap {got['favourite_gap']:+.3f} at {got['favourite_sigma']:+.1f} se "
                  f"over {got['favourite_n']:,} sides -> {got['verdict']}")
            if a.write:
                print(f"    written to {write_survivor(got)}")

    if a.shape:
        try:
            stats = _stats(seasons, cache)
        except Exception as e:
            return unavailable("hub.models.coverage", "nflverse player_stats", e)
        try:
            run, paired, pooled = shape_law(stats, min_prior=a.min_prior,
                                            ledger=Ledger(WIDTH_STATE))
        except (ValueError, NotEnoughWeeks) as e:
            print(f"hub.models.coverage: {e}", file=sys.stderr)
            return 1
        print(f"  the weekly interval's shape law under CRPS (#292), centre=prior, "
              f"{paired.height:,} unclipped player-weeks over seasons "
              f"{seasons[0]}-{seasons[-1]}; skew-free is the arm under test")
        for line in run.lines:
            print(line)
        by = paired.group_by("position").agg(pl.col("best_skew").first(),
                                             pl.col("diff").mean().alias("diff"),
                                             pl.len().alias("n")).sort("position")
        print("\n  per position: the deployed skew, the best skew on these rows, and the "
              "skew-free gain")
        for r in by.iter_rows(named=True):
            print(f"    {r['position']:<4}{r['n']:>7,}   deployed "
                  f"{predict.WEEKLY_SKEW.get(r['position'], predict.WEEKLY_SKEW_POOLED):.2f}"
                  f"   best {r['best_skew']:.2f}   skew-free {r['diff']:+.4f}")
        print(f"\n  pooled over every row, clipped weeks included: {pooled['mean']:+.4f} over "
              f"{pooled['n']:,}  (a diagnostic; the verdict reads the unclipped rows)")
        print(f"\n  {run.verdict[1]}")

    if a.audit and a.look is None:
        print("hub.models.coverage: --audit needs --look k; a look is spent when it is taken, "
              "so there is no default to take by accident (#438)", file=sys.stderr)
        return 1
    if a.audit and a.look in looks_taken():
        print(f"hub.models.coverage: look {a.look} is already recorded in {ARTIFACT.name} "
              f"(taken: {looks_taken()}); a look is never taken twice, and a recorded one is "
              f"never overwritten", file=sys.stderr)
        return 1
    if a.audit and not 1 <= a.look <= LOOKS:
        print(f"hub.models.coverage: look {a.look} is outside the {LOOKS} looks a claim is "
              f"allowed to live through; a claim that outlives them is a restatement trigger "
              f"(docs/weekly-coverage.md, #310), not a fourth read", file=sys.stderr)
        return 1

    if a.audit and a.look in LOOK_WEEKS:
        return _audit_current(a, cache, as_of)

    if a.measure or a.gate or a.audit:
        # An audit measures fresh -- the look is spent on what it reads, not on a stored copy.
        # The weekly run reads the closed backtest back while its key holds (#423).
        key = backtest_key(seasons, a.centre, min_prior=a.min_prior, band=a.band)
        got = None if (a.audit or a.remeasure) else read_back_backtest(key)
        if got is not None and key is not None:
            got["backtest_key"] = key
            print(f"  backtest {seasons[0]}-{seasons[-1]}: read back from {ARTIFACT.name} "
                  f"(measured once under key {key['config_digest']}/{key['code_digest']}/"
                  f"{key['data_digest']}; --remeasure to measure again)")
        else:
            try:
                stats = _stats(seasons, cache)
            except Exception as e:
                return unavailable("hub.models.coverage", "nflverse player_stats", e)
            try:
                got = measure(stats, a.centre, min_prior=a.min_prior, band=a.band)
            except (ValueError, NotEnoughWeeks) as e:
                print(f"hub.models.coverage: {e}", file=sys.stderr)
                return 1
            if key is not None:
                got["backtest_key"] = key
        lookahead = "  LOOKAHEAD CENTRE" if got["lookahead"] else ""
        print(f"  weekly interval, centre={got['centre']}, {got['n']:,} player-weeks on the "
              f"board, {got['gate_n']:,} scored{lookahead}")
        print(f"    the published interval: conformal, calibrated within position on the last "
              f"{got['calibration']['window']} cells (floor {got['calibration']['floor']}), "
              f"scale {got['calibration']['scale']}")
        _print_published(got["by_position"])
        print("    -- split on whether the published lower bound is pinned at zero --")
        _print_published(got["floor_split"])
        if got["by_prior"]:
            print("    -- by games of evidence behind the centre --")
            _print_published(got["by_prior"])
        par = got["parametric"]
        print(f"    for the record, the parametric interval this replaced: "
              f"{par['gate_cov80']:.1%} of {par['gate_n']:,} {par['gate_subset']} weeks "
              f"({par['clipped_share']:.1%} of the board clipped) -> {par['verdict']}")
        print(f"    {got['n_uncalibrated']:,} of {got['n']:,} player-weeks had no calibration "
              f"and were not scored"
              + (f" (cells {got['uncalibrated_cells']})" if got["uncalibrated_cells"] else ""))
        pooled = [r["group"] for r in got["by_position"] if r.get("pooled_calibration")
                  and r["position"]]
        if pooled:
            print(f"    on pooled calibration for part of the window, so not a test of the "
                  f"conditional claim there: {', '.join(pooled)}")
        print(f"    gate reads every week it scored ({got['gate_subset']}): "
              f"{got['gate_cov80']:.1%} against the claimed {got['gate_claim']:.1%} "
              f"+/- {got['band']:.0%} (interval labelled {got['nominal']['cov80']:.0%}) over "
              f"{got['gate_n']:,}, {got['gate_clipped_share']:.1%} of them clipped "
              f"-> {got['verdict']}")
        sc = got["season_coverage"]
        if sc:
            print(f"    the season's own coverage ({sc['season']}): {sc['cov80']:.1%} over "
                  f"{sc['n']:,} -- reported beside the full-history figure and never gated")
        # The row not being measurable is a failed run -- the same non-zero exit an unreachable
        # source has always given this command (`tests/contracts/test_cli_surface.py`) -- but it
        # is not a lost one: the backtest is still reported and written, and the last committed
        # row stays in the file (`write_summary`), so the page degrades to last-good.
        degraded = False
        if not a.audit:
            cur = _current_or_none(cache, as_of)
            if cur is not None:
                got["current_season"] = cur
                _print_current(cur)
            else:
                degraded = True
        if a.audit:
            audit = audit_verdict(got["by_position"], claim=got["gate_claim"], band=got["band"])
            _print_audit(audit, a.look)
            if a.write:
                print(f"    written to {write_audit(got, audit, a.look)}")
            return 0 if audit["passes"] else 1
        if a.write:
            print(f"    written to {write_summary(got)}")
        if a.gate:
            # The weekly run reports and never gates (#310): a verdict read weekly on
            # accumulating rows is a sequential test. What stays is the smoke alarm.
            print(f"    the weekly run reports and never gates; the verdict is taken at audits "
                  f"(--audit --look k, {LOOKS} looks). Smoke alarm: marginal within "
                  f"{SMOKE_BAND:.0%} of the claim and every group scored.")
            alarm = smoke_alarm(got)
            if alarm:
                print("    smoke alarm -- the pipeline looks broken, not drifted: "
                      + "; ".join(alarm), file=sys.stderr)
                return 1
        if degraded:
            return 1
    return 0


def _current_inputs(cache: Path | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(schedule, stats) for the current-season row: the season's schedule, and its player-weeks
    with the `CURRENT_HISTORY` seasons behind them."""
    seasons = [CURRENT_SEASON - CURRENT_HISTORY + i for i in range(CURRENT_HISTORY + 1)]
    return _schedules([CURRENT_SEASON], cache), _stats(seasons, cache)


def _current_or_none(cache: Path | None, as_of: date) -> dict[str, Any] | None:
    """The weekly run's current-season row, or `None` and a stderr line if it could not be made.

    Broad on `hub.cli.unavailable`'s reasoning, and for `CLAUDE.md`'s degradation rule: a fetch
    that fails must leave the last committed row in the file (`write_summary` keeps it) and the
    backtest and smoke alarm untouched, not take the run down.
    """
    try:
        schedule, stats = _current_inputs(cache)
        return measure_current(stats, schedule, as_of)
    except Exception as e:
        print(f"hub.models.coverage: the {CURRENT_SEASON} out-of-sample row could not be "
              f"measured ({type(e).__name__}: {e}); the last committed row stands",
              file=sys.stderr)
        return None


def _print_current(cur: dict[str, Any]) -> None:
    print(f"  {cur['season']} out-of-sample row, {cur['weeks_complete']} weeks complete "
          f"(not pooled with the backtest; no verdict until looks "
          f"{', '.join(str(k) for k in sorted(LOOK_WEEKS))} at weeks "
          f"{', '.join(str(LOOK_WEEKS[k]) for k in sorted(LOOK_WEEKS))})")
    _print_published(cur["by_position"])


REFUSED = 2


def _audit_current(a: argparse.Namespace, cache: Path | None, as_of: date) -> int:
    """Look 2 or 3: the current-season row against the marginal bar, once its week is complete.

    Dates, then data, the shape of #432's horizon (#437): the calendar is checked from the
    schedule alone, before any outcome is loaded, and then the week's rows must be in the data.
    A refusal exits `REFUSED` (2), which is neither a pass nor a failed audit, and spends nothing.
    """
    look, week = a.look, LOOK_WEEKS[a.look]
    try:
        schedule = _schedules([CURRENT_SEASON], cache)
    except Exception as e:
        return unavailable("hub.models.coverage", "nflverse schedules", e)
    why = look_date_reached(look, schedule, as_of)
    if why:
        print(f"hub.models.coverage: look {look} REFUSED -- {why}. No outcome was loaded and no "
              f"look was spent (docs/weekly-coverage.md, pre-registered 2026-10-07).",
              file=sys.stderr)
        return REFUSED
    try:
        stats = _stats([CURRENT_SEASON - CURRENT_HISTORY + i
                        for i in range(CURRENT_HISTORY + 1)], cache)
    except Exception as e:
        return unavailable("hub.models.coverage", "nflverse player_stats", e)
    try:
        cur = measure_current(stats, schedule, as_of, min_prior=a.min_prior)
    except (ValueError, NotEnoughWeeks) as e:
        print(f"hub.models.coverage: {e}", file=sys.stderr)
        return 1
    if cur["weeks_complete"] < week or not cur["n"]:
        print(f"hub.models.coverage: look {look} REFUSED -- the dates are past week {week} but "
              f"the data holds {cur['weeks_complete']} complete weeks (week {week}'s rows are "
              f"not in nflverse yet). No look was spent.", file=sys.stderr)
        return REFUSED
    _print_current(cur)
    audit = audit_verdict([r for r in cur["by_position"] if r["n"]], claim=cur["claim"],
                          band=cur["band"])
    _print_audit(audit, look)
    if a.write:
        print(f"    written to {write_audit(cur, audit, look, reads='current_season')}")
    return 0 if audit["passes"] else 1


def _print_audit(audit: dict[str, Any], look: int) -> None:
    """The audit's verdict with the look count and alpha beside it, and each group (#310)."""
    print(f"  AUDIT look {look} of {LOOKS}, alpha {ALPHA_PER_LOOK:.4f} per look "
          f"(z = {Z_PER_LOOK:.3f}): marginal {audit['cov80']:.1%} against the claimed "
          f"{audit['claim']:.0%} +/- {audit['band']:.0%} over {audit['n']:,} "
          f"(MDE {audit['mde']:.3f}) -> {audit['marginal']}")
    for g in audit["groups"]:
        note = (f"NOT-RUNNABLE: needs {g['n_required']:,} player-weeks to rule"
                if g["verdict"] == "NOT-RUNNABLE" else g["verdict"])
        pooled = "  (pooled calibration for part of the window)" if g["pooled_calibration"] else ""
        print(f"    {g['group']:<4}{g['n']:>7,} weeks  {g['cov80']:.1%}  dev {g['deviation']:+.1%}"
              f"  sigma {g['sigma']:+.1f}  MDE {g['mde']:.3f}  -> {note}{pooled}")
    print(f"    audit {'PASSES' if audit['passes'] else 'FAILS'}: the marginal binds; a position "
          f"below the minimum neither passes nor fails it.")


if __name__ == "__main__":
    sys.exit(main())
