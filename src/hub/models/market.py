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
from datetime import UTC, datetime

import polars as pl

from hub.models.base import FitSpec

# Standard deviation of the actual margin around the closing spread, and the reason a 3-point
# favourite is a ~60% shot rather than a ~90% one. Lower it and the model claims precision the
# market does not have; raise it and every game drifts toward a coin flip.
#
# FITTED 2026-08-24 by `hub.models.margin`, written up in `docs/margin-sd.md`. Trailing ten
# seasons, n=3018: 12.741 +/- 0.164. It was 13.5 before, asserted under a comment claiming it
# was "stable across decades" -- with no fit, no interval and no write-up, while sitting in
# `config.FITTED_MODULES` and being hashed into every model version as though it had been
# measured. ADR-0006 draws that line; this number was on the wrong side of it.
#
# Two things the fit found that the assertion could not:
#
#   * It is NOT stable. Per-season sd runs 11.5 to 14.4 and trends down 0.037 a year
#     (-2.4 se) -- the market has got sharper. A trailing window beat all-history on
#     held-out log-loss, which is why this is a decade rather than the full 1999-2025 record.
#   * The accuracy gain is real but small: +0.0002 mean held-out log-loss over 26
#     walk-forward seasons, about 0.057 nats a season. The win here is provenance, not
#     precision.
#
# **This is a trailing window, so it goes stale.** Re-run `python -m hub.models.margin --fit`
# after each season; a fixed constant fitted to a moving target needs a refit date, and a
# full-history value would not have.
MARGIN_SD = 12.741

# 80% interval. z for the two-sided 80% of a normal.
Z_80 = 1.2816


def normal_cdf(x: float) -> float:
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

    def fit(self, spec: FitSpec) -> MarketBaseline:
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
        probs = [normal_cdf(float(s) / self.margin_sd) for s in spread]
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
            # UTC, not local. `predicted_at` is the provenance stamp on every prediction
            # row and the whole claim is that December can audit August -- a naive local
            # timestamp is ambiguous the moment the machine or the season changes. Written
            # tz-naive after conversion so the column dtype is unchanged, which is the same
            # shape `hub.fetch.odds` already uses.
            pl.lit(datetime.now(UTC).replace(tzinfo=None, microsecond=0))
              .alias("predicted_at"),
        )
