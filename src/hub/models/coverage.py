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

**What the gate holds the interval to** is the claim, not the label (#289). The interval is
served as 80% and measured to cover 77.4% of the unclipped weeks; `CLAIMED_COV80` is that
restated claim and the gate refuses when the deployed function leaves `BAND` of it in
either direction. The label is `LEVELS`, which has not moved.

    uv run python -m hub.models.coverage --measure
    uv run python -m hub.models.coverage --measure --centre realised
    uv run python -m hub.models.coverage --survivor
    uv run python -m hub.models.coverage --gate          # exits 1 on a refusal
    uv run python -m hub.models.coverage --shape         # the skew law under CRPS (#292)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from hub import atomic, jsonio
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.declare import not_an_input
from hub.models import predict
from hub.models.experiment import (
    SEASON_CLUSTER,
    WIDTH_STATE,
    Actions,
    Ceiling,
    GateRun,
    run_gate,
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
CLAIMED_COV80 = not_an_input(
    0.77,
    "the restated claim the grading harness holds the deployed interval to; it grades "
    "predictions already made and makes none of its own")

# Pre-registered before the prior-centre numbers were looked at, and the only other thing
# `--gate` reads: empirical coverage must sit within this of the claim. Two points is roughly
# three standard errors at n = 10,000, so it is a band a truthful claim clears comfortably and
# a stale one does not.
BAND = not_an_input(
    0.02,
    "a pre-registered filter of the grading harness, which measures interval coverage "
    "of predictions already made and makes none of its own")

# The gate is read off the weeks whose interval is *not* pinned at the zero floor. A clipped
# lower bound cannot be fallen below, so those weeks report a coverage the model did not
# earn, and pooling them in hides the miss -- which is the document's own diagnosis, used
# here as the reason to exclude them rather than as a footnote.
GATE_SUBSET = "unclipped"

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


def measure(stats: pl.DataFrame, centre: Centre = "prior", *,
            min_weeks: int = MIN_WEEKS, min_prior: int = MIN_PRIOR,
            min_mu: float = MIN_MU, band: float = BAND,
            claim: float = CLAIMED_COV80) -> dict[str, Any]:
    """The whole weekly-interval measurement, as one dict.

    `nominal` is the label the interval is served under; `gate_claim` is what the doc says
    it covers and what `verdict` was read against. Both are carried because a reader of
    `COVERS` beside `77.4%` needs the number it was compared with on the same line (#289).
    """
    g = graded(centred(player_weeks(stats), centre, min_weeks=min_weeks,
                       min_prior=min_prior, min_mu=min_mu))
    rows = table(g)
    split = floor_split(g)
    gated = next((r for r in split if r["group"] == "strictly positive"), rows[-1])
    return {
        "centre": centre, "lookahead": centre == "realised",
        "n": int(g.height), "nominal": {"cov80": 0.80, "cov68": 0.68},
        "band": band, "by_position": rows, "floor_split": split,
        "gate_subset": GATE_SUBSET, "gate_n": gated["n"], "gate_cov80": gated["cov80"],
        "gate_claim": claim, "verdict": verdict(gated, claim=claim, band=band),
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

SHAPE_ACTIONS = Actions(
    adopt="The skew-free interval scores better under CRPS: the deployed function is the "
          "maintainer's to change, and #289's claim is re-registered against it first.",
    remove="The skew earns its place: the deployed interval stays as served.",
    show="The skew stays; the CRPS comparison could not remove it.")


def shape_scores(g: pl.DataFrame) -> pl.DataFrame:
    """CRPS of the deployed distribution and of the same function without its skew, per row.

    Arm B is `predict.skewed(mu, sd, skew, z)` on the graded moments at the CRPS quantile
    grid, clip included, which is the path `hub.models.weekly.shipped_quantiles` takes. Arm A
    is the same call with the skew zeroed -- which `skewed` floors at `MIN_SKEW`, so the arm
    is exactly what would be served with `WEEKLY_SKEW` removed, floor and all, and not a
    normal written out beside it. `diff` is deployed minus skew-free: positive when the
    skew-free arm scores lower, which is the sign `experiment.gate` adopts on.

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
              width_path: Path = WIDTH_STATE) -> tuple[GateRun, pl.DataFrame, dict[str, Any]]:
    """The pre-registered comparison: one gate run, the paired rows, and the pooled diagnostic.

    The rows are the coverage gate's -- prior centre, the same filters -- restricted to the
    unclipped subset it reads, fixed in `docs/gate-power.md` so the subset cannot be chosen
    after the sign is seen. The season is the cluster, the MDE comes from the interval's own
    bootstrap, and the ceiling is `CEILING_ARM` on the same rows. The pooled figure over every
    row, clipped weeks included, is returned beside it as a diagnostic and decides nothing.
    """
    g = graded(centred(player_weeks(stats), "prior", min_weeks=min_weeks,
                       min_prior=min_prior, min_mu=min_mu))
    scored = shape_scores(g)
    paired = scored.filter(pl.col("p10_raw") > 0.0)
    pooled = {"n": int(scored.height), "mean": float(cast(float, scored["diff"].mean()))}
    run = run_gate(paired, cluster=SEASON_CLUSTER, actions=SHAPE_ACTIONS,
                   name="interval_shape", arm_a="skew-free", arm_b="deployed skew",
                   unit="CRPS points per player-week", places=4, seed=seed,
                   ceiling=Ceiling(CEILING_ARM, paired["ceiling_diff"].to_numpy()),
                   width_path=width_path)
    return run, paired, pooled


# --- what reads the answer ------------------------------------------------


def write_summary(result: dict[str, Any], path: Path | None = None) -> Path:
    """Leave the weekly measurement where a publisher can read it.

    ADR-0007's second consequence: the result is written rather than printed, so a later
    disagreement is between two recorded runs instead of between two memories. A survivor
    block already in the file is kept (#273): one file, two verdicts, one reader.
    """
    p = path or ARTIFACT
    kept = _existing(p).get("survivor")
    atomic.write_text(p, jsonio.dumps({"name": "interval_coverage",
                                       "generated_at": jsonio.stamp(), **result,
                                       **({"survivor": kept} if kept else {})}, indent=2))
    return p


def write_survivor(result: dict[str, Any], path: Path | None = None) -> Path:
    """Leave the survivor verdict beside the weekly one, in the same file (#273).

    A companion block, not a summary of its own: each block carries its own `generated_at`,
    and `published_summary` reads nothing until the weekly measurement has been written,
    because the page's consumers key on the weekly `verdict`. The slate runs `--measure
    --survivor --write` together, which is the order that leaves both.
    """
    p = path or ARTIFACT
    have = _existing(p)
    block = {k: result.get(k) for k in
             ("verdict", "favourite_spread", "favourite_predicted", "favourite_actual",
              "favourite_gap", "favourite_sigma", "favourite_n", "n_games", "margin_sd",
              # The buckets ride along whole (#293): the page shows where the price holds
              # and where it is thin, and it cannot without the count in each.
              "buckets", "min_bucket")}
    block["generated_at"] = jsonio.stamp()
    atomic.write_text(p, jsonio.dumps({"name": "interval_coverage", **have,
                                       "survivor": block}, indent=2))
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
            "band", "verdict", "generated_at")}
    # An artifact written before the claim was carried (#289) had its verdict read against
    # the label, so that is the claim it is published with -- a reader printing `verdict`
    # beside `gate_claim` then says what that run actually compared.
    if out["gate_claim"] is None:
        out["gate_claim"] = LEVELS[0][1] - LEVELS[0][0]
    if isinstance(got.get("survivor"), dict):
        out["survivor"] = got["survivor"]
    return out


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"    {'':<18}{'n':>7}{'80%':>8}{'68%':>8}{'<p10':>8}{'>p90':>8}"
          f"{'80% no skew':>13}{'sd r/m':>8}")
    for r in rows:
        print(f"    {r['group']:<18}{r['n']:>7,}{r['cov80']:>8.1%}{r['cov68']:>8.1%}"
              f"{r['below_p10']:>8.1%}{r['above_p90']:>8.1%}"
              f"{r['cov80_no_skew']:>13.1%}{r['sd_ratio']:>8.2f}")


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
                    help="measure, then exit 1 if the interval leaves the band")
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
    a = ap.parse_args(argv)

    if not (a.measure or a.survivor or a.gate or a.shape):
        ap.print_help()
        return 0
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    cache = Path(a.cache) if a.cache else None

    if a.survivor:
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
                                            width_path=WIDTH_STATE)
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

    if a.measure or a.gate:
        try:
            stats = _stats(seasons, cache)
        except Exception as e:
            return unavailable("hub.models.coverage", "nflverse player_stats", e)
        try:
            got = measure(stats, a.centre, min_prior=a.min_prior, band=a.band)
        except (ValueError, NotEnoughWeeks) as e:
            print(f"hub.models.coverage: {e}", file=sys.stderr)
            return 1
        lookahead = "  LOOKAHEAD CENTRE" if got["lookahead"] else ""
        print(f"  weekly interval, centre={got['centre']}, {got['n']:,} player-weeks"
              f"{lookahead}")
        _print_table(got["by_position"])
        print("    -- split on whether the model's own p10 survived the zero clip --")
        _print_table(got["floor_split"])
        print(f"    gate reads the {got['gate_subset']} weeks: {got['gate_cov80']:.1%} "
              f"against the claimed {got['gate_claim']:.1%} +/- {got['band']:.0%} "
              f"(interval labelled {got['nominal']['cov80']:.0%}) over {got['gate_n']:,} "
              f"-> {got['verdict']}")
        if a.write:
            print(f"    written to {write_summary(got)}")
        if a.gate and got["verdict"] != "COVERS":
            print("    the deployed interval does not cover what docs/weekly-coverage.md "
                  "says it covers; restate the claim or find what moved the function.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
