import polars as pl
import pytest
from hub.draft.projection import adjust_consensus, build, regression_signal


def _df():
    return pl.DataFrame({
        "player": ["Early", "Late", "Rookie"],
        "position": ["WR", "WR", "WR"],
        "ecr": [10.0, 150.0, 200.0],
        "fp": [100.0, 100.0, None],
        "xfp": [140.0, 140.0, None],
    })


def test_zero_lambda_is_identity():
    out = adjust_consensus(regression_signal(_df()), lam=0.0)
    assert out["adj_ecr"].to_list() == out["ecr"].to_list()


def test_missing_signal_leaves_ecr_untouched():
    """Rookies have no 2025 snaps. That is expected, and they must not be moved."""
    out = build(_df())
    r = out.filter(pl.col("player") == "Rookie")
    assert abs(r["adj_ecr"][0] - 200.0) < 1e-9


def test_underperformer_moves_up_the_board():
    df = pl.DataFrame({
        "player": ["Unlucky", "Lucky"], "position": ["WR", "WR"],
        "ecr": [50.0, 50.0], "fp": [80.0, 160.0], "xfp": [140.0, 140.0],
    })
    out = build(df)
    assert out["player"][0] == "Unlucky"


def test_adjustment_is_larger_deeper_in_the_board():
    """Same evidence must move a pick-150 player further than a pick-10 player."""
    df = pl.DataFrame({
        "player": ["Early", "Late"], "position": ["WR", "WR"],
        "ecr": [10.0, 150.0], "fp": [80.0, 80.0], "xfp": [140.0, 140.0],
        # identical z by construction; only ecr differs
    }).with_columns(pl.lit(1.0).alias("z_regress"))
    out = adjust_consensus(df, lam=0.08)
    moved = dict(zip(out["player"], out["ranks_moved"]))
    assert moved["Late"] > moved["Early"] * 5


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        adjust_consensus(_df().with_columns(pl.lit(0.0).alias("z_regress")), lam=-0.1)
