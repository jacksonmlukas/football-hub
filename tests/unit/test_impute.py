"""Imputing expected points for players with no history.

2026 rookies have no 2025 xFP, but the market drafts them inside the top 168, so the
market clearly expects production. Leaving mu at zero told the season simulation that a
second-round rookie RB is an empty roster slot -- and made any opponent who drafted one
strictly worse off, which is how P(win) came out at 85%.

Consensus rank is the available signal for what a player is expected to do. Imputation
is within position, because a TE and a WR at the same rank are not the same asset.
"""
import polars as pl
import pytest

from hub.draft.board import _impute_xfp


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
