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
# So the honest reason for zero is not "the signal is harmful". It is that a metric riding
# on ~15 player-outcomes across six seasons cannot tell. See docs/lambda-sweep.md.
DEFAULT_LAMBDA = 0.0


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


def adjust_consensus(df: pl.DataFrame, lam: float = DEFAULT_LAMBDA) -> pl.DataFrame:
    """adjusted_ecr = ecr * exp(-lam * z)

    Multiplicative rather than additive on purpose. One standard deviation of evidence should
    move a player further at pick 150 than at pick 5, because consensus is tightly packed and
    well-informed at the top and loose at the bottom. Additive adjustment gets this backwards
    and will shove a fringe signal into your first round.

    Players with no signal (rookies, anyone without 2025 snaps) keep their ECR untouched.
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    z = pl.col("z_regress").fill_null(0.0).clip(-3.0, 3.0)
    return df.with_columns([
        (pl.col("ecr") * np.e ** (-lam * z)).alias("adj_ecr"),
        (pl.col("ecr") - pl.col("ecr") * np.e ** (-lam * z)).alias("ranks_moved"),
    ])


def build(df: pl.DataFrame, lam: float = DEFAULT_LAMBDA) -> pl.DataFrame:
    return adjust_consensus(regression_signal(df), lam).sort("adj_ecr")
