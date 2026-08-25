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


# --- the pick-noise constants, now measured rather than assumed -----------

def test_the_noise_law_is_the_fitted_one():
    """`fit_pick_noise` existed and was never called: the heuristic 2.0 + 0.18*mu was still
    hardcoded in two places, including the opponent model inside the win-probability
    simulation. Fitted on 734 real picks from this league's 2022-25 drafts it comes out at
    1.00 + 0.253*mu.

    The difference is not cosmetic deep in the board. At ADP 100 the heuristic says sigma is
    20 and the fit says 26, so the heuristic is over-confident about who survives -- which
    inflates cost_of_waiting and pushes the board toward 'take him now' on players who would
    in fact have lasted."""
    from hub.draft.availability import PICK_NOISE_INTERCEPT, PICK_NOISE_SLOPE
    assert PICK_NOISE_INTERCEPT == pytest.approx(1.00, abs=0.01)
    assert PICK_NOISE_SLOPE == pytest.approx(0.253, abs=0.005)


def test_the_fitted_law_is_wider_late_and_tighter_early():
    """The shape of the correction, pinned so a refit that inverts it is noticed."""
    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    early_fit, early_heur = A + B * 3, 2.0 + 0.18 * 3
    late_fit, late_heur = A + B * 100, 2.0 + 0.18 * 100
    assert early_fit < early_heur
    assert late_fit > late_heur


def test_sigma_uses_the_fitted_law_when_there_is_no_consensus_spread():
    import polars as pl

    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    from hub.draft.availability import _sigma
    df = pl.DataFrame({"mu_pick": [10.0, 100.0]})
    got = _sigma(df)
    assert got[0] == pytest.approx(A + B * 10.0)
    assert got[1] == pytest.approx(A + B * 100.0)


# --- which column the consensus spread is read from -----------------------
#
# Only the fallback above was ever tested, so the branch that reads the consensus spread ran
# in production untested. That is why the column could be renamed with the whole suite green
# -- and why the collision below could have gone live without anything failing.

def test_sigma_prefers_the_consensus_spread_when_it_is_there():
    import polars as pl

    from hub.draft.availability import _sigma
    df = pl.DataFrame({"mu_pick": [10.0, 100.0], "ecr_sd": [7.0, 40.0]})
    assert _sigma(df).tolist() == [7.0, 40.0]


def test_a_zero_or_null_consensus_spread_falls_back_per_player():
    """A player FantasyPros has no spread for must not come out as sigma 0 -- certainty
    about where he goes, from an absence of data."""
    import polars as pl

    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    from hub.draft.availability import _sigma
    df = pl.DataFrame({"mu_pick": [10.0, 20.0], "ecr_sd": [0.0, None]})
    got = _sigma(df)
    assert got[0] == pytest.approx(A + B * 10.0)
    assert got[1] == pytest.approx(A + B * 20.0)


def test_a_points_spread_is_not_mistaken_for_a_pick_spread():
    """The collision this rename exists to prevent.

    `sd` meant two different things on frames in this pipeline: the spread of a player's
    consensus *rank*, in picks, and the spread of his weekly *points*. Both are small
    positive floats and neither looks wrong. `hub.models.predict.moments` writes the points
    one onto a board-derived frame, so an availability sim run on a scored frame would have
    read points as picks and produced a confident, plausible, entirely wrong curve.
    """
    import polars as pl

    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    from hub.draft.availability import _sigma
    scored = pl.DataFrame({"mu_pick": [10.0], "sd": [7.9]})   # weekly points, not picks
    assert _sigma(scored)[0] == pytest.approx(A + B * 10.0)
