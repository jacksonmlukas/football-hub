"""The one interface every forecaster implements.

You said everything should connect, with NFL and CFB separate. That resolves to: a single
protocol, with league as a *field* rather than a subclass hierarchy. A fitted model is always
league-scoped (you never fit one model across both), but the type is shared, so `model-eval`
can put any two forecasters head to head without knowing what either one is.

The payoff is that "does the Bayesian model beat the market" becomes a comparison of two
objects implementing the same protocol, rather than a bespoke script per model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import polars as pl

from hub.models.conformal import DEFAULT_MIN_CALIBRATION as MIN_CALIBRATION
from hub.models.conformal import interval

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

    def fit(self, spec: FitSpec) -> Forecaster:
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

    # GUARD leakage-tripwire [unit/test_models_base.py]: fit through N never predicts <= N
    leaked = df.filter(pl.col("week") <= pl.col("fit_through_week"))
    if leaked.height:
    # /GUARD
        raise ValueError(
            f"LEAKAGE: {leaked.height} predictions for weeks at or before "
            f"fit_through_week={spec.through_week}"
        )

    bad = df.filter(pl.col("margin_lo") > pl.col("margin_hi"))
    if bad.height:
        raise ValueError(f"{bad.height} rows have margin_lo > margin_hi")
    return df


def forecast(model: Forecaster, spec: FitSpec, games: pl.DataFrame) -> pl.DataFrame:
    """Fit, predict, check. The one path from a `Forecaster` to a row anything publishes.

    ADR-0002 says leakage is enforced at the type boundary, and until now that sentence
    described nothing: the boundary had no code on it. `hub.models.ratings` -- the module
    that writes the published predictions -- built a spec, named a concrete class, fitted
    it, predicted, and then remembered to call `validate_predictions` on the way past. So
    the highest-value check in this codebase was reachable from exactly one function that
    nothing typed, and Track A replacing "the middle of that function" would have carried
    the obligation to remember it again. A check nothing can reach reports nothing and
    cannot fail, which is this repo's recurring defect rather than a new one.

    The three steps are here together because they have to agree about one `spec`. Split
    across a caller they are three statements that can drift: fit through one week, predict
    another, and check against a third reads as success at every step. Leakage does not
    crash -- it looks good -- so the gap has to be structural.

    `model` is typed as the protocol and not as a class, which is the point of ADR-0002
    holding at all: what replaces the passthrough is substituted here, and cannot arrive
    without the check. `fit` returns self by the protocol's own contract, so a caller that
    kept its reference reads the fitted `version` off it afterwards.

    Nothing about this settles #136. It is the half of that issue that needs no answer:
    whatever is decided about where the protocol's payoff lands, the shipped writer going
    around it and the tripwire hanging off an untyped call are wrong either way.
    """
    out = model.fit(spec).predict(games)
    # GUARD leakage-is-reached [unit/test_ratings.py unit/test_models_base.py]: delete it
    # and a leaking model publishes clean, from the CLI `Makefile:12` runs. Ratings leads
    # the selectors because reachability is the claim: the harness runs under `-x`, so the
    # first file it reddens is the one this guard is evidence about.
    validate_predictions(out, spec)
    # /GUARD
    return out


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

    def fit(self, spec: FitSpec) -> Conformalized:
        self.base.fit(spec)
        return self

    def calibrate(self, residuals: pl.Series) -> Conformalized:
        """Split-conformal quantile of absolute residuals on a held-out window.

        Delegates to `hub.models.conformal.interval`, which is the one implementation of
        this statistic. It used to be written out here as well -- same finite-sample
        correction, same docstring explaining that the correction is not decoration, and a
        *different* minimum-n invariant: 20 here against 40 there. Two copies of a formula
        are a nuisance; two copies with disagreeing invariants are a defect waiting for the
        first caller who reads the wrong one.
        """
        n = residuals.len()
        if n < MIN_CALIBRATION:
            raise ValueError(f"need >={MIN_CALIBRATION} calibration points, got {n}")
        self._q = interval(residuals, self.alpha)
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
