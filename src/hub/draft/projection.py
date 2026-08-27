"""Adjustment to consensus, not a projection from scratch.

Thirteen days is not enough to beat FantasyPros ECR, and trying is the wrong use of the
time. ECR already aggregates hundreds of analysts who each watched more tape than you will.
A from-scratch model built in under two weeks reliably loses to it.

What consensus does badly is regression. Human rankers anchor on last season's realized
fantasy points, which bake in touchdown variance that does not repeat. ffopportunity gives
you expected points from opportunity alone, so the gap between expected and actual is a
clean, free signal that consensus systematically underweights.

So: start from ECR and nudge it. Shrink the nudge hard, because consensus is a strong prior
and your signal is one noisy season.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Nudge strength, set to zero by measurement rather than judgment.
#
# The reasoning below is intact and may still be right in principle -- consensus does
# anchor on realised points, and expected points is a cleaner number. Three holdouts say
# this particular adjustment cannot be relied on:
#
#   lam=0.08 delta, top-50 points, six holdouts 2019->20 through 2024->25:
#     -62   -13   -110   -87   +184   -94
#
# Five of six negative. The one positive year, 2023->24, is large enough to drag the pooled
# t to -0.68, so this is not significant either way -- but nothing supports a nonzero
# lambda, and the best value per season is 0.00 or 0.02 in four of six.
#
# The one unambiguous finding: a large nudge is reliably harmful. At lam=0.32 all six
# seasons are negative, t = -6.83, sign test p = 0.031. Spearman falls monotonically with
# lambda in every season without exception.
#
# The decisive finding is not in those numbers, it is underneath them. At lam=0.08 the
# adjustment moves only 2-3 players across the top-50 boundary per season, so the whole
# delta is a handful of individual outcomes. In 2023->24 the biggest of them was Rashee
# Rice: most negative z on the board, demoted hardest, then 64.9 points in 2024 -- because
# he tore his ACL in week 3. Replay that season with Rice healthy and +184 becomes +26,
# and the pooled six-season effect goes from t = -0.68 to t = -2.64.
#
# The reason is simpler than any of that, and it is measurable directly. This signal
# barely persists year over year:
#
#   corr(z in season N, z in season N+1), same player, four transitions:
#     uniform  +0.209    8wk  +0.193    4wk  +0.147    2wk  +0.090
#
# r = 0.21 is about 4% shared variance. Roughly 96% of the expected-versus-actual gap does
# not carry into the next season, so there was never enough here to tune. Recency weighting
# makes it strictly worse -- shorter windows average fewer weeks, so more of what survives
# is week-to-week noise, which is the signature of noise rather than of staleness.
#
# See docs/lambda-sweep.md for the full path: six seasons, three metrics, and a
# recency-weighted variant, all arriving at zero.
DEFAULT_LAMBDA = 0.0

# Evidence is clipped before it moves anybody. Three standard deviations is already far past
# anything the signal supports, and without a clip one bad `z_regress` becomes an unbounded
# multiplier on a player's rank.
Z_CLIP = 3.0


def regression_signal(df: pl.DataFrame) -> pl.DataFrame:
    """Positive means the player underperformed his opportunity: a buy candidate.

    Standardized within position, because the expected-vs-actual gap scales with volume and
    a WR1's residual is not comparable to a TE2's.
    """
    gap = -(pl.col("fp") - pl.col("xfp"))          # underperformance is positive
    return df.with_columns(
        ((gap - gap.mean().over("position")) / gap.std().over("position"))
        .fill_nan(None).alias("z_regress")
    )


def weighted_signal(weekly: pl.DataFrame, half_life: float | None = None,
                    min_weeks: int = 1) -> pl.DataFrame:
    """The regression signal with recent weeks weighted more heavily.

    The season-total version treats week 1 and week 17 as equally informative about next
    season. Roles change: a back who took over in November and one who lost the job in
    October can post identical totals and be entirely different assets going forward.

    `half_life` is in weeks -- 4.0 means a week four back counts half as much as the most
    recent one. `None` is uniform weighting, and it reproduces `regression_signal` on
    season totals exactly, so a comparison between the two is like-for-like rather than a
    comparison between two different quantities.

    Weights are renormalised to sum to the games played, so the signal keeps the scale of
    a season total. Without that, a shorter half-life would shrink every value and a
    lambda tuned on one weighting would mean something different on the other.
    """
    last = weekly["week"].max()
    if half_life is None:
        w = pl.lit(1.0)
    else:
        w = (0.5 ** ((pl.lit(last) - pl.col("week")) / half_life))

    per = (weekly.with_columns(w.alias("_w"))
           .group_by(["full_name", "position"])
           .agg(((pl.col("fp") - pl.col("xfp")) * pl.col("_w")).sum().alias("_num"),
                pl.col("_w").sum().alias("_den"),
                pl.len().alias("_n")))

    # Renormalise to season scale, then flip so underperformance is positive.
    gap = -(pl.col("_num") / pl.col("_den") * pl.col("_n"))
    scored = per.with_columns(
        pl.when(pl.col("_n") >= min_weeks).then(gap).otherwise(None).alias("_gap"))

    return (scored.with_columns(
                ((pl.col("_gap") - pl.col("_gap").mean().over("position"))
                 / pl.col("_gap").std().over("position"))
                .fill_nan(None).alias("z_regress"))
            .select("full_name", "position", "z_regress"))


def adjusted(df: pl.DataFrame, lam: float = DEFAULT_LAMBDA) -> pl.DataFrame:
    """`adj_ecr = ecr * exp(-lam * z)`, the one implementation of it.

    Multiplicative rather than additive on purpose. One standard deviation of evidence should
    move a player further at pick 150 than at pick 5, because consensus is tightly packed and
    well-informed at the top and loose at the bottom. Additive adjustment gets this backwards
    and will shove a fringe signal into your first round.

    Players with no signal (rookies, anyone without prior-season snaps) keep their ECR.

    This formula was written three times -- here, in `tune.apply`, and inline in
    `evaluate.simulate_draft` -- and the copy that had no caller was the one the other two
    were copies of. Changing the clip, or making the adjustment additive, meant finding three
    sites and a passing test on the version nothing ran.

    `adjust_consensus` and `build` are gone with it. They also produced a `ranks_moved`
    column, which nothing outside its own test ever read.
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    z = pl.col("z_regress").fill_null(0.0).clip(-Z_CLIP, Z_CLIP)
    return df.with_columns((pl.col("ecr") * np.e ** (-lam * z)).alias("adj_ecr"))
