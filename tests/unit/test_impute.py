"""Imputing expected points for players with no history.

2026 rookies have no 2025 xFP, but the market drafts them inside the top 168, so the
market clearly expects production. Leaving mu at zero told the season simulation that a
second-round rookie RB is an empty roster slot -- and made any opponent who drafted one
strictly worse off, which is how P(win) came out at 85%.

Consensus rank is the available signal for what a player is expected to do. Imputation
is within position, because a TE and a WR at the same rank are not the same asset.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft.board import _impute_xfp, _pava_decreasing


def _df(rows):
    return pl.DataFrame(rows, schema={"player": pl.Utf8, "pos": pl.Utf8,
                                      "ecr": pl.Float64, "xfp_per_game": pl.Float64})


def test_missing_value_is_filled_from_neighbours_in_rank():
    out = _impute_xfp(_df([
        {"player": "A", "pos": "RB", "ecr": 10.0, "xfp_per_game": 20.0},
        {"player": "B", "pos": "RB", "ecr": 20.0, "xfp_per_game": None},
        {"player": "C", "pos": "RB", "ecr": 30.0, "xfp_per_game": 10.0},
    ]))
    v = out.filter(pl.col("player") == "B")["xfp_per_game"][0]
    assert 10.0 <= v <= 20.0


def test_imputation_is_within_position():
    """A TE must not inherit a WR's curve."""
    rows: list[dict[str, object]] = [
        {"player": f"W{i}", "pos": "WR", "ecr": float(i), "xfp_per_game": 30.0}
        for i in range(1, 11)]
    rows.extend({"player": f"T{i}", "pos": "TE", "ecr": float(i), "xfp_per_game": 5.0}
                for i in range(1, 11))
    rows.append({"player": "TX", "pos": "TE", "ecr": 5.0, "xfp_per_game": None})
    out = _impute_xfp(_df(rows))
    assert out.filter(pl.col("player") == "TX")["xfp_per_game"][0] == pytest.approx(5.0)


def test_known_values_are_untouched():
    out = _impute_xfp(_df([
        {"player": "A", "pos": "RB", "ecr": 1.0, "xfp_per_game": 22.0},
        {"player": "B", "pos": "RB", "ecr": 2.0, "xfp_per_game": None},
    ]))
    assert out.filter(pl.col("player") == "A")["xfp_per_game"][0] == 22.0


def test_imputation_is_monotone_in_rank():
    """A worse consensus rank must not impute a better projection."""
    rows: list[dict[str, object]] = [
        {"player": f"P{i}", "pos": "RB", "ecr": float(i), "xfp_per_game": 25.0 - i}
        for i in range(1, 41)]
    rows.extend([{"player": "Early", "pos": "RB", "ecr": 5.5, "xfp_per_game": None},
                 {"player": "Late", "pos": "RB", "ecr": 35.5, "xfp_per_game": None}])
    out = _impute_xfp(_df(rows))
    e = out.filter(pl.col("player") == "Early")["xfp_per_game"][0]
    ln = out.filter(pl.col("player") == "Late")["xfp_per_game"][0]
    assert e > ln


def test_position_with_no_known_values_falls_back_to_zero_not_a_crash():
    out = _impute_xfp(_df([{"player": "X", "pos": "K", "ecr": 1.0, "xfp_per_game": None}]))
    assert out.filter(pl.col("player") == "X")["xfp_per_game"][0] == 0.0


def test_nothing_missing_is_a_noop():
    d = _df([{"player": "A", "pos": "RB", "ecr": 1.0, "xfp_per_game": 5.0}])
    assert _impute_xfp(d)["xfp_per_game"].to_list() == [5.0]


# --- the monotone fit is the least-squares one, not the greedy projection (#315 item 3) ---
#
# The running minimum this replaced clamped every point down to the smallest value seen so far,
# so wherever the smoothed curve rose it sat at or below the data and never above: a one-sided
# bias that no residual the fit could be scored on would average out. PAVA pools the violating
# neighbours instead, and the pooled mean is the least-squares non-increasing fit.

def _greedy(y):
    return np.minimum.accumulate(np.asarray(y, dtype=float))


def test_pava_is_the_least_squares_nonincreasing_fit_and_greedy_is_not():
    """Three properties a greedy clamp cannot have on a series that rises. On [10, 4, 8, 3]
    the violating pair (4, 8) pools to 6: the fit is [10, 6, 6, 3], its sum is the data's sum
    (the residuals average zero), and its squared error is the smallest any non-increasing
    sequence can have. The greedy projection gives [10, 4, 4, 3]: sum 21 against 25, every
    residual of the clamp is >= 0, and a strictly larger squared error.

    Rule 18, planted: swapping `_pava_decreasing` for `np.minimum.accumulate` fails the
    sum and squared-error assertions here, and the closed-form imputation below."""
    y = np.array([10.0, 4.0, 8.0, 3.0])
    fit = _pava_decreasing(y)
    assert fit.tolist() == [10.0, 6.0, 6.0, 3.0]
    assert fit.sum() == pytest.approx(y.sum())
    sse = float(((y - fit) ** 2).sum())
    greedy = _greedy(y)
    assert float(((y - greedy) ** 2).sum()) > sse
    assert (y - greedy).min() >= 0.0 and (y - greedy).max() > 0.0   # one-sided: always below
    # Optimal against any non-increasing competitor: random monotone sequences never beat it.
    rng = np.random.default_rng(0)
    for _ in range(500):
        other = -np.sort(-rng.uniform(0.0, 12.0, size=y.size))
        assert float(((y - other) ** 2).sum()) >= sse - 1e-9


def test_pava_properties_on_random_series():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = rng.normal(10.0, 4.0, size=int(rng.integers(1, 40)))
        fit = _pava_decreasing(y)
        assert fit.shape == y.shape
        assert (np.diff(fit) <= 1e-12).all()                 # non-increasing
        assert fit.sum() == pytest.approx(y.sum())          # unbiased: residuals sum to zero
        assert _pava_decreasing(fit) == pytest.approx(fit)  # a monotone series is its own fit
        assert float(((y - fit) ** 2).sum()) <= float(((y - _greedy(y)) ** 2).sum()) + 1e-9
    assert _pava_decreasing(np.array([])).size == 0


def test_the_imputation_pools_a_rise_instead_of_clamping_it():
    """Through `_impute_xfp`: six veterans whose smoothed curve is [15, 16, 10, 8, 6, 5], a
    rise at ranks 1-2. A rookie at rank 1.5 reads the pooled value 15.5 (the mean of the
    violating pair) where the greedy clamp read 15.0."""
    ys = [20.0, 10.0, 16.0, 8.0, 6.0, 4.0]
    rows: list[dict[str, object]] = [
        {"player": f"V{i}", "pos": "WR", "ecr": float(i + 1), "xfp_per_game": y}
        for i, y in enumerate(ys)]
    rows.append({"player": "R", "pos": "WR", "ecr": 1.5, "xfp_per_game": None})
    got = _impute_xfp(_df(rows)).filter(pl.col("player") == "R")["xfp_per_game"][0]
    assert got == pytest.approx(15.5)
