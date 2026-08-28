from datetime import datetime
from typing import cast

import polars as pl
import pytest

from hub.models.base import (
    PREDICTION_SCHEMA,
    Conformalized,
    FitSpec,
    Forecaster,
    validate_predictions,
)


class Dummy:
    """Minimal conforming forecaster, used as the protocol's executable spec."""
    name = "dummy"

    def __init__(self, margin: float = 3.0):
        self.margin = margin
        self._spec: FitSpec | None = None

    @property
    def version(self) -> str:
        return f"dummy-{self._spec.digest}" if self._spec else "dummy-unfit"

    def fit(self, spec: FitSpec):
        self._spec = spec
        return self

    def predict(self, games: pl.DataFrame) -> pl.DataFrame:
        return games.select(["game_id", "league", "season", "week"]).with_columns([
            pl.lit(0.6).alias("home_win_prob"),
            pl.lit(self.margin).alias("margin_mean"),
            pl.lit(self.margin).alias("margin_lo"),
            pl.lit(self.margin).alias("margin_hi"),
            pl.lit(self.name).alias("model"),
            pl.lit(self.version).alias("version"),
            pl.lit(cast(FitSpec, self._spec).through_week).cast(pl.Int32).alias("fit_through_week"),
            pl.lit(datetime(2026, 9, 1)).alias("predicted_at"),
        ])


def _games(week=7, n=30):
    return pl.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "league": ["nfl"] * n,
        "season": pl.Series([2026] * n, dtype=pl.Int32),
        "week": pl.Series([week] * n, dtype=pl.Int32),
    })


def test_dummy_satisfies_the_protocol():
    assert isinstance(Dummy(), Forecaster)


def test_version_changes_with_fit_data():
    a = Dummy().fit(FitSpec("nfl", 2026, 6)).version
    b = Dummy().fit(FitSpec("nfl", 2026, 7)).version
    assert a != b, "two fits on different data must not share a version"


def test_valid_predictions_pass():
    spec = FitSpec("nfl", 2026, 6)
    out = Dummy().fit(spec).predict(_games(week=7))
    assert validate_predictions(out, spec).height == 30
    assert set(PREDICTION_SCHEMA).issubset(out.columns)


def test_leakage_is_caught():
    """Fit through week 7, predict week 7. This is the bug that looks like success."""
    spec = FitSpec("nfl", 2026, 7)
    out = Dummy().fit(spec).predict(_games(week=7))
    with pytest.raises(ValueError, match="LEAKAGE"):
        validate_predictions(out, spec)


def test_probability_out_of_range_is_caught():
    spec = FitSpec("nfl", 2026, 6)
    out = Dummy().fit(spec).predict(_games()).with_columns(pl.lit(1.4).alias("home_win_prob"))
    with pytest.raises(ValueError, match="outside"):
        validate_predictions(out, spec)


def test_conformal_wrapper_is_itself_a_forecaster():
    """Composition, not inheritance: Track C applies to every other track for free."""
    c = Conformalized(Dummy(), alpha=0.2)
    assert isinstance(c, Forecaster)


def test_conformal_widens_the_interval():
    spec = FitSpec("nfl", 2026, 6)
    c = Conformalized(Dummy(), alpha=0.2).fit(spec)
    c.calibrate(pl.Series([float(x) for x in range(-25, 26)]))
    out = c.predict(_games(week=7))
    assert cast(float, (out["margin_hi"] - out["margin_lo"]).min()) > 0


def test_conformal_refuses_to_predict_before_calibration():
    c = Conformalized(Dummy()).fit(FitSpec("nfl", 2026, 6))
    with pytest.raises(RuntimeError, match="calibrate"):
        c.predict(_games(week=7))


def test_conformal_rejects_bad_alpha():
    with pytest.raises(ValueError):
        Conformalized(Dummy(), alpha=1.5)


def test_conformal_needs_enough_calibration_points():
    c = Conformalized(Dummy()).fit(FitSpec("nfl", 2026, 6))
    with pytest.raises(ValueError, match=">=40"):
        c.calibrate(pl.Series([1.0, 2.0, 3.0]))


def test_config_digest_participates_in_model_version():
    """A lambda change must produce a different model version, even on identical data."""
    a = Dummy().fit(FitSpec("nfl", 2026, 6, cfg_digest="aaaa1111")).version
    b = Dummy().fit(FitSpec("nfl", 2026, 6, cfg_digest="bbbb2222")).version
    assert a != b


def test_the_two_conformal_implementations_are_one():
    """`Conformalized.calibrate` and `conformal.interval` computed the same split-conformal
    quantile with the same finite-sample correction -- and disagreed on the minimum-n
    invariant, 20 against 40. Two copies of a formula are a nuisance; two copies with
    different invariants are a defect waiting for whoever reads the wrong one."""
    import inspect

    from hub.models import base as base_mod
    from hub.models import conformal

    residuals = pl.Series([float(x) for x in range(-40, 41)])
    alpha = 0.2
    c = Conformalized(Dummy(), alpha=alpha).calibrate(residuals)
    assert c._q == conformal.interval(residuals, alpha)
    src = inspect.getsource(base_mod.Conformalized.calibrate)
    assert "quantile(" not in src, "the statistic must not be written out here as well"
    assert base_mod.MIN_CALIBRATION == conformal.DEFAULT_MIN_CALIBRATION
