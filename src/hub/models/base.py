"""The one interface every forecaster implements.

You said everything should connect, with NFL and CFB separate. That resolves to: a single
protocol, with league as a *field* rather than a subclass hierarchy. A fitted model is always
league-scoped (you never fit one model across both), but the type is shared, so `model-eval`
can put any two forecasters head to head without knowing what either one is.

The payoff is that "does the Bayesian model beat the market" becomes a comparison of two
objects implementing the same protocol, rather than a bespoke script per model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable
import hashlib
import polars as pl

League = Literal["nfl", "cfb"]

# Every forecaster returns exactly this. Enforced by contract at write time, which is what
# makes cross-model comparison possible at all.
PREDICTION_SCHEMA = {
    "game_id": pl.Utf8,
    "league": pl.Utf8,
    "season": pl.Int32,
    "week": pl.Int32,
    "home_win_prob": pl.Float64,     # calibrated probability, not a score
    "margin_mean": pl.Float64,       # home minus away, positive favors home
    "margin_lo": pl.Float64,         # interval bounds; equal to mean if the model
    "margin_hi": pl.Float64,         # has no uncertainty estimate
    "model": pl.Utf8,
    "version": pl.Utf8,
    "fit_through_week": pl.Int32,    # leakage tripwire: must be < week
    "predicted_at": pl.Datetime,
}


@dataclass(frozen=True)
class FitSpec:
    """What a model was fit on, and under which config.

    `cfg_digest` comes from hub.config.config_digest(). Folding it in means two runs that
    differ only in a hyperparameter cannot collide in the track record -- which matters
    because the public claim is that a prediction was made by a specific model, not by
    "the Bayesian model" as a category.
    """
    league: League
    season: int
    through_week: int
    seed: int = 0
    cfg_digest: str = "default"

    @property
    def digest(self) -> str:
        raw = f"{self.league}:{self.season}:{self.through_week}:{self.seed}:{self.cfg_digest}"
        return hashlib.sha256(raw.encode()).hexdigest()[:8]


@runtime_checkable
class Forecaster(Protocol):
    """Implemented by every track: market baseline, Bayesian ratings, sequence model.

    Deliberately narrow. Anything a specific model needs beyond this belongs in its own
    constructor, not in the shared interface.
    """

    name: str

    def fit(self, spec: FitSpec) -> "Forecaster":
        """Fit on data through spec.through_week. Returns self for chaining."""
        ...

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        """Predict the given games. Must conform to PREDICTION_SCHEMA."""
        ...

    @property
    def version(self) -> str:
        ...


def validate_predictions(df: pl.DataFrame, spec: FitSpec) -> pl.DataFrame:
    """Schema plus the two invariants that actually catch bugs.

    Leakage: a model fit through week N must never predict week <= N. This has caught more
    real errors in backtests than every other check combined, because leakage looks like
    success rather than failure.
    """
    missing = set(PREDICTION_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"prediction schema missing: {sorted(missing)}")

    p = df["home_win_prob"]
    if p.min() is not None and (cast(float, p.min()) < 0 or cast(float, p.max()) > 1):
        raise ValueError(f"home_win_prob outside [0,1]: [{p.min()}, {p.max()}]")

    leaked = df.filter(pl.col("week") <= pl.col("fit_through_week"))
    if leaked.height:
        raise ValueError(
            f"LEAKAGE: {leaked.height} predictions for weeks at or before "
            f"fit_through_week={spec.through_week}"
        )

    bad = df.filter(pl.col("margin_lo") > pl.col("margin_hi"))
    if bad.height:
        raise ValueError(f"{bad.height} rows have margin_lo > margin_hi")
    return df


class Conformalized:
    """Wraps any Forecaster and replaces its intervals with calibrated ones.

    Composition rather than inheritance: a Conformalized(BayesianRatings(...)) is itself a
    Forecaster, so it drops into model-eval unchanged and you can compare a model against
    its own conformalized version. Track C therefore applies to every other track for free
    instead of being reimplemented per model.
    """

    def __init__(self, base: Forecaster, alpha: float = 0.2):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.base, self.alpha = base, alpha
        self.name = f"conformal({base.name})"
        self._q: float | None = None

    @property
    def version(self) -> str:
        return f"{self.base.version}+cp{self.alpha}"

    def fit(self, spec: FitSpec) -> "Conformalized":
        self.base.fit(spec)
        return self

    def calibrate(self, residuals: pl.Series) -> "Conformalized":
        """Split-conformal quantile of absolute residuals on a held-out window."""
        n = residuals.len()
        if n < 20:
            raise ValueError(f"need >=20 calibration points, got {n}")
        k = min(1.0, (1 - self.alpha) * (n + 1) / n)   # finite-sample correction
        self._q = float(cast(float, residuals.abs().quantile(k)))
        return self

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        out = self.base.predict(games)
        if self._q is None:
            raise RuntimeError("calibrate() before predict()")
        return out.with_columns([
            (pl.col("margin_mean") - self._q).alias("margin_lo"),
            (pl.col("margin_mean") + self._q).alias("margin_hi"),
            pl.lit(self.name).alias("model"),
            pl.lit(self.version).alias("version"),
        ])
