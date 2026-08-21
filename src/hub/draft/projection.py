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

# Nudge strength. 0.08 moves a z=1 player about 8% up the board, which is ~1 rank at the top
# and ~12 ranks at pick 150. That asymmetry is intentional -- see adjust_consensus.
DEFAULT_LAMBDA = 0.08


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
