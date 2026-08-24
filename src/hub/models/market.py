"""The market baseline. Everything else is scored against this.

The closing line is the aggregate of every informed opinion with money behind it, and it
is very hard to beat. A model that cannot beat it has not learned anything, so this has to
exist before any model does -- a residual model with nothing to take a residual against is
unfalsifiable rather than good.

The conversion is deliberately plain. A spread is an expected margin; NFL margins scatter
around it roughly normally with a standard deviation near 13.5 points, which is the number
that turns a spread into a win probability. That single constant is the whole model, and
its calibration is checkable: a three-point home favourite comes out near 59%, which is
what three-point home favourites actually do.
"""
from __future__ import annotations

import math
from datetime import datetime

import polars as pl

from hub.models.base import FitSpec

# Standard deviation of the actual margin around the closing spread. Stable across decades
# of NFL results and the reason a 3-point favourite is a ~59% shot rather than a ~90% one.
# Lower it and the model claims precision the market does not have; raise it and every
# game drifts toward a coin flip.
MARGIN_SD = 13.5

# 80% interval. z for the two-sided 80% of a normal.
Z_80 = 1.2816


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class MarketBaseline:
    """Closing spread in, calibrated probability and margin interval out.

    `fit` stores what it was fit through and nothing else: there are no parameters to
    estimate, which is precisely why this is the benchmark rather than a competitor.
    """

    name = "market_baseline"

    def __init__(self, margin_sd: float = MARGIN_SD):
        self.margin_sd = margin_sd
        self._spec: FitSpec | None = None

    def fit(self, spec: FitSpec) -> "MarketBaseline":
        self._spec = spec
        return self

    @property
    def version(self) -> str:
        return f"market-{self._spec.digest}" if self._spec else "market-unfit"

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        if self._spec is None:
            raise ValueError("MarketBaseline.predict called before fit")
        if "close_spread" not in games.columns:
            raise ValueError("games must carry close_spread to have a market prior")

        # A game with no line has no market prior. Dropping it is honest; filling it with
        # zero would silently assert every unpriced game is a coin flip.
        priced = games.filter(pl.col("close_spread").is_not_null())

        spread = priced["close_spread"].to_numpy()
        probs = [_norm_cdf(float(s) / self.margin_sd) for s in spread]
        half = Z_80 * self.margin_sd

        return priced.select(
            pl.col("game_id"),
            pl.col("league"),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.Series("home_win_prob", probs, dtype=pl.Float64),
            pl.col("close_spread").cast(pl.Float64).alias("margin_mean"),
            (pl.col("close_spread") - half).cast(pl.Float64).alias("margin_lo"),
            (pl.col("close_spread") + half).cast(pl.Float64).alias("margin_hi"),
            pl.lit(self.name).alias("model"),
            pl.lit(self.version).alias("version"),
            pl.lit(self._spec.through_week).cast(pl.Int32).alias("fit_through_week"),
            pl.lit(datetime.now().replace(microsecond=0)).alias("predicted_at"),
        )
