import numpy as np
import polars as pl
import pytest
from hub.draft.availability import availability, blended_adp, pick_value


def _board(n=60):
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(1, n + 1)],
        "ecr": [float(i) for i in range(1, n + 1)],
        "adp": [float(i) for i in range(1, n + 1)],
        "vor": [float(n - i) for i in range(n)],
    })


def test_blend_endpoints_collapse_to_single_boards():
    df = _board().with_columns(pl.col("adp") + 10)
    assert blended_adp(df, w=1.0)["mu_pick"].to_list() == df["adp"].to_list()
    assert blended_adp(df, w=0.0)["mu_pick"].to_list() == df["ecr"].to_list()


def test_blend_rejects_weight_outside_unit_interval():
    with pytest.raises(ValueError):
        blended_adp(_board(), w=1.4)


def test_availability_decreases_as_your_pick_gets_later():
    av = availability(_board(), picks=[10, 40], n_sims=2000)
    early, late = av["avail_10"].to_numpy(), av["avail_40"].to_numpy()
    assert (late <= early + 1e-9).all()


def test_top_of_board_is_gone_by_a_late_pick():
    av = availability(_board(), picks=[50], n_sims=2000)
    assert av.filter(pl.col("player") == "P1")["avail_50"][0] < 0.05


def test_availability_is_a_probability():
    av = availability(_board(), picks=[5, 25], n_sims=1000)
    for c in ("avail_5", "avail_25"):
        assert np.all((av[c].to_numpy() >= 0) & (av[c].to_numpy() <= 1))


def test_cost_of_waiting_prefers_the_player_who_will_not_return():
    """Two players of equal VOR: the one going sooner costs more to pass on.

    Embedded in a full board, because availability ranks within the pool it is given --
    a two-row frame can never produce a rank of 10.
    """
    df = _board(60).with_columns(pl.col("vor") * 0.1)
    df = pl.concat([df, pl.DataFrame({
        "player": ["Soon", "Later"], "ecr": [12.0, 45.0],
        "adp": [12.0, 45.0], "vor": [50.0, 50.0],
    })])
    pv = pick_value(df, now=10, next_pick=30, n_sims=3000)
    assert pv["player"][0] == "Soon"


def test_empty_board_does_not_crash():
    empty = _board().head(0)
    assert availability(empty, picks=[1]).height == 0
