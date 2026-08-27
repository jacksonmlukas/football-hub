import polars as pl
import pytest

from hub.draft.projection import adjusted, regression_signal


def _df():
    return pl.DataFrame({
        "player": ["Early", "Late", "Rookie"],
        "position": ["WR", "WR", "WR"],
        "ecr": [10.0, 150.0, 200.0],
        "fp": [100.0, 100.0, None],
        "xfp": [140.0, 140.0, None],
    })


def test_zero_lambda_is_identity():
    out = adjusted(regression_signal(_df()), lam=0.0)
    assert out["adj_ecr"].to_list() == out["ecr"].to_list()


def test_missing_signal_leaves_ecr_untouched():
    """Rookies have no prior-season snaps. That is expected, and they must not be moved."""
    out = adjusted(regression_signal(_df()), lam=0.08)
    r = out.filter(pl.col("player") == "Rookie")
    assert abs(r["adj_ecr"][0] - 200.0) < 1e-9


def test_underperformer_moves_up_the_board():
    """At the shipped lambda of 0.0 both players get the same adj_ecr and this passed on
    stable sort order alone -- it asserted nothing about the adjustment. Run at a lambda
    that actually moves somebody."""
    df = pl.DataFrame({
        "player": ["Unlucky", "Lucky"], "position": ["WR", "WR"],
        "ecr": [50.0, 50.0], "fp": [80.0, 160.0], "xfp": [140.0, 140.0],
    })
    out = adjusted(regression_signal(df), lam=0.08).sort("adj_ecr")
    assert out["player"][0] == "Unlucky"
    assert out["adj_ecr"][0] < out["adj_ecr"][1], "they must not be tied"


def test_adjustment_is_larger_deeper_in_the_board():
    """Same evidence must move a pick-150 player further than a pick-10 player."""
    df = pl.DataFrame({
        "player": ["Early", "Late"], "position": ["WR", "WR"],
        "ecr": [10.0, 150.0], "fp": [80.0, 80.0], "xfp": [140.0, 140.0],
        # identical z by construction; only ecr differs
    }).with_columns(pl.lit(1.0).alias("z_regress"))
    out = adjusted(df, lam=0.08).with_columns(
        (pl.col("ecr") - pl.col("adj_ecr")).alias("moved"))
    moved = dict(zip(out["player"], out["moved"], strict=True))
    assert moved["Late"] > moved["Early"] * 5


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        adjusted(_df().with_columns(pl.lit(0.0).alias("z_regress")), lam=-0.1)


def test_one_implementation_of_the_adjustment():
    """It was written three times -- here, in tune.apply, and inline in numpy inside
    evaluate.simulate_draft -- and the copy with no caller was the one the others copied.
    Both now go through this function."""
    import inspect

    from hub.draft import evaluate, tune
    # The executable spellings, not the prose: tune's module docstring legitimately writes
    # the formula out to explain what the tuner tunes.
    for mod in (tune, evaluate):
        src = inspect.getsource(mod)
        assert "np.e ** (-lam" not in src, f"{mod.__name__} kept a polars copy"
        assert "np.exp(-lam" not in src, f"{mod.__name__} kept a numpy copy"
        assert "from hub.draft.projection import adjusted" in src
    assert "adjusted(" in inspect.getsource(tune.score)
