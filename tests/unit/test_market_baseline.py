"""The market baseline, and the ratings passthrough that ships it.

`Makefile:12` invokes `hub.models.ratings --fit` and it did not exist, which is the last
thing standing between `make slate` and running end to end.

Two things live here for one reason. The plan puts `MarketBaseline` before any model
because it is what every model is scored against -- a residual model with nothing to take
a residual against is unfalsifiable. And it puts `ratings` in as a passthrough because
naive-but-real beats sophisticated-but-absent: returning the market prior unchanged makes
`make slate` run, which makes every later improvement a diff against something working.

The interesting tests are the sign convention and the leakage tripwire. Both are the kind
of error that leaves a backtest running happily while being exactly wrong.
"""
import datetime as dt

import polars as pl
import pytest

from hub.models.base import FitSpec, Forecaster, PREDICTION_SCHEMA, validate_predictions
from hub.models.market import MarketBaseline


def _games(rows):
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "league": ["nfl"] * len(rows),
         "season": pl.Series([2025] * len(rows), dtype=pl.Int32),
         "week": pl.Series([r[1] for r in rows], dtype=pl.Int32),
         "close_spread": [r[2] for r in rows]})


@pytest.fixture
def fitted():
    return MarketBaseline().fit(FitSpec("nfl", 2025, 5))


# --- the protocol ---------------------------------------------------------

def test_it_is_a_forecaster(fitted):
    """The plan's done-when for 3.1, stated literally."""
    assert isinstance(fitted, Forecaster)


def test_predictions_satisfy_the_shared_schema(fitted):
    out = fitted.predict(_games([("g1", 6, -3.0)]))
    assert set(PREDICTION_SCHEMA) <= set(out.columns)
    validate_predictions(out, FitSpec("nfl", 2025, 5))


def test_version_changes_with_what_it_was_fit_on(fitted):
    other = MarketBaseline().fit(FitSpec("nfl", 2025, 6))
    assert fitted.version != other.version


def test_version_changes_with_config(fitted):
    other = MarketBaseline().fit(FitSpec("nfl", 2025, 5, cfg_digest="beef1234"))
    assert fitted.version != other.version, \
        "two runs differing only in config must not collide in the track record"


def test_predicting_before_fitting_refuses():
    with pytest.raises(ValueError):
        MarketBaseline().predict(_games([("g1", 6, -3.0)]))


# --- the sign convention --------------------------------------------------

def test_a_home_favourite_is_favoured(fitted):
    """nflverse: positive spread means the home team is favoured. Invert this and every
    number the project publishes is confidently backwards."""
    out = fitted.predict(_games([("g1", 6, 7.0)]))
    assert out["home_win_prob"][0] > 0.5
    assert out["margin_mean"][0] > 0


def test_a_home_underdog_is_not_favoured(fitted):
    out = fitted.predict(_games([("g1", 6, -7.0)]))
    assert out["home_win_prob"][0] < 0.5


def test_a_pick_em_is_a_coin_flip(fitted):
    out = fitted.predict(_games([("g1", 6, 0.0)]))
    assert out["home_win_prob"][0] == pytest.approx(0.5)


def test_a_bigger_spread_is_a_bigger_probability(fitted):
    out = fitted.predict(_games([("a", 6, 3.0), ("b", 6, 10.0), ("c", 6, 17.0)]))
    probs = out.sort("game_id")["home_win_prob"].to_list()
    assert probs == sorted(probs)


def test_probabilities_stay_inside_zero_and_one(fitted):
    out = fitted.predict(_games([("a", 6, -60.0), ("b", 6, 60.0)]))
    assert out["home_win_prob"].min() > 0.0
    assert out["home_win_prob"].max() < 1.0


def test_a_three_point_home_favourite_lands_near_the_known_rate(fitted):
    """A field goal favourite wins about 58-60% of the time historically. A model that
    said 90% here would pass every structural test above and still be useless."""
    p = fitted.predict(_games([("g1", 6, 3.0)]))["home_win_prob"][0]
    assert 0.55 < p < 0.65


# --- intervals ------------------------------------------------------------

def test_the_interval_brackets_the_mean(fitted):
    out = fitted.predict(_games([("g1", 6, 3.0)]))
    assert out["margin_lo"][0] < out["margin_mean"][0] < out["margin_hi"][0]


def test_the_interval_is_wide_enough_to_be_honest(fitted):
    """NFL margins scatter enormously around the spread. A tight interval here would be a
    false claim of precision the market itself does not make."""
    out = fitted.predict(_games([("g1", 6, 3.0)]))
    assert out["margin_hi"][0] - out["margin_lo"][0] > 15


# --- leakage --------------------------------------------------------------

def test_fit_through_week_is_recorded(fitted):
    out = fitted.predict(_games([("g1", 6, 3.0)]))
    assert out["fit_through_week"][0] == 5


def test_predicting_a_week_already_fit_on_is_caught(fitted):
    """validate_predictions treats fit_through_week >= week as leakage."""
    with pytest.raises(ValueError):
        validate_predictions(fitted.predict(_games([("g1", 5, 3.0)])), FitSpec("nfl", 2025, 5))


def test_a_missing_spread_is_dropped_rather_than_guessed(fitted):
    out = fitted.predict(_games([("a", 6, 3.0), ("b", 6, None)]))
    assert out["game_id"].to_list() == ["a"], \
        "a game with no line has no market prior; inventing one is worse than skipping it"
