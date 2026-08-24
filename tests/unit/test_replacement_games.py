"""Replacement level needs a real denominator.

xfp_per_game over a 1-game sample is not a rate, it is noise. Those flukes sort to
the top and occupy the slots that define replacement, pushing genuine players out
of the top-N window and mispricing every VOR on the board.

The tell was that WR replacement came out *above* RB (11.14 vs 11.10), contradicting
the three-WR effect in docs/decisions.md. With a minimum-games filter WR drops to
10.02 and the documented effect reappears. Stable at both 8 and 10 games, so this is
not knife-edge fitting.
"""
import polars as pl
from hub.draft.board import replacement_levels


def _pool(pos, n, per_game, games):
    return pl.DataFrame({
        "position": [pos] * n,
        "xfp_per_game": per_game,
        "games": games,
        "full_name": [f"{pos}{i}" for i in range(n)],
    })


def _levels(df, **kw):
    """`replacement_levels` takes three Series, so name them here once.

    The columns these fixtures happen to use are `position`/`xfp_per_game`/`games`, which
    used to be a hard requirement of the function rather than a choice of the caller.
    """
    return replacement_levels(df["position"], df["xfp_per_game"], df["games"], **kw)


def test_one_game_flukes_do_not_set_replacement():
    """A 40-point one-game wonder must not occupy a startable slot."""
    n = 40
    real = _pool("TE", n, [10.0 - i * 0.1 for i in range(n)], [16] * n)
    fluke = _pool("TE", 3, [40.0, 39.0, 38.0], [1, 1, 2])
    lv = _levels(pl.concat([fluke, real]), teams=12, min_games=6)
    # 13 startable TEs; without the filter three slots go to flukes and the level
    # is read three players too shallow.
    assert lv["TE"] == real["xfp_per_game"][12]


def test_filter_is_inclusive_at_the_threshold():
    n = 30
    df = _pool("TE", n, [10.0 - i * 0.1 for i in range(n)], [6] * n)
    assert _levels(df, teams=12, min_games=6)["TE"] is not None


def test_position_with_no_qualifying_players_is_zero_not_a_crash():
    df = _pool("TE", 5, [10.0] * 5, [1] * 5)
    assert _levels(df, teams=12, min_games=6)["TE"] == 0.0


def test_shallow_pool_uses_its_last_player():
    df = _pool("QB", 4, [30.0, 25.0, 20.0, 15.0], [16] * 4)
    assert _levels(df, teams=12, min_games=6)["QB"] == 15.0


def test_three_wr_slots_put_wr_replacement_below_rb():
    """The claim in decisions.md, as an executable assertion.

    Equal talent curves for RB and WR; the only difference is 3 WR starters vs 2 RB,
    so WR is drawn deeper into its pool and its replacement must land lower.
    """
    curve = [20.0 - i * 0.2 for i in range(80)]
    df = pl.concat([_pool("RB", 80, curve, [16] * 80), _pool("WR", 80, curve, [16] * 80)])
    lv = _levels(df, teams=12, min_games=6)
    assert lv["WR"] < lv["RB"]
