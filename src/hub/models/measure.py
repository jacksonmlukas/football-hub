"""Scoring a decision rule against what actually happened.

The half of `hub.draft.backtest` that is not about drafts. It was extracted on 2026-08-24
when the weekly-lineup gate needed the same two pieces -- realised points keyed the way the
board joins, and a paired bootstrap -- and importing them from a module called
`hub.draft.backtest` would have made that name a lie.

What deliberately did NOT move: `verdict`. Its branches name championship equity and the
draft-night output, so it belongs to that question. Every gate writes its own verdict, which
is the point -- a pre-registered rule is specific to what it is deciding, and a shared one
would drift toward being decorative.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hub.draft.state import _norm

# The paired bootstrap, matching `hub.models.eval.compare`.
BOOTSTRAP = 4000


def realised_ppg(stats: pl.DataFrame) -> pl.DataFrame:
    """Realised fantasy points per player per week, from nflverse weekly player stats.

    Returns (player, week, points). `player` is normalised with `state._norm`, the same key
    the board joins on, because nflverse and FantasyPros disagree about suffixes and
    punctuation and an exact join drops the disagreements silently.
    """
    out = stats.select(
        pl.col("player_display_name").map_elements(_norm, return_dtype=pl.Utf8)
          .alias("player"),
        pl.col("week").cast(pl.Int64),
        pl.col("fantasy_points_ppr").fill_null(0.0).cast(pl.Float64).alias("points"),
    )
    return out.group_by(["player", "week"]).agg(pl.col("points").sum())


def summarise(paired: pl.DataFrame, *, bootstrap: int = BOOTSTRAP,
              seed: int = 0) -> dict[str, float]:
    """Mean paired difference, a bootstrap interval, and P(optimizer better).

    Bootstrapped over *paired* observations rather than over each arm separately, matching
    `hub.models.eval.compare`: the arms share a room and a seed, so resampling them
    independently would throw away the pairing that the design exists to create.
    """
    d = paired["diff"].to_numpy()
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_better": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(bootstrap, n))
    means = d[idx].mean(axis=1)
    return {"n": float(n), "mean": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "p_better": float((means > 0).mean())}
