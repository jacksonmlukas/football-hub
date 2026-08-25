"""Weeks 15-17 strength of schedule.

The failure this must not have is a silently plausible number: a player we could not
place on a team getting 1.0 and reading as "average playoff schedule".
"""
import polars as pl
import pytest

from hub.draft.playoff_sos import (
    PLAYOFF_WEEKS,
    _canon_team,
    _dvp_from_stats,
    _opponents_from_schedule,
    _sos_from,
    attach_sos,
)


def _stats(rows):
    return pl.DataFrame(rows, schema={"opponent_team": pl.Utf8, "position": pl.Utf8,
                                      "week": pl.Int64, "fantasy_points_ppr": pl.Float64})


def _sched(rows):
    return pl.DataFrame(rows, schema={"week": pl.Int64, "home_team": pl.Utf8,
                                      "away_team": pl.Utf8})


# --- team codes -----------------------------------------------------------

@pytest.mark.parametrize("board,nflverse", [("JAC", "JAX"), ("LAR", "LA"), ("KC", "KC")])
def test_team_aliases(board, nflverse):
    assert _canon_team(board) == nflverse


def test_free_agents_have_no_team():
    assert _canon_team("FA") is None
    assert _canon_team(None) is None


# --- defence vs position --------------------------------------------------

def test_points_are_summed_within_a_week_then_averaged():
    """A committee backfield is still one week of points allowed."""
    d = _dvp_from_stats(_stats([
        {"opponent_team": "AAA", "position": "RB", "week": 1, "fantasy_points_ppr": 10.0},
        {"opponent_team": "AAA", "position": "RB", "week": 1, "fantasy_points_ppr": 10.0},
        {"opponent_team": "AAA", "position": "RB", "week": 2, "fantasy_points_ppr": 30.0},
    ]))
    assert d["ppg_allowed"][0] == pytest.approx(25.0)   # (20 + 30) / 2 weeks


def test_ratio_is_indexed_to_the_league_average_for_that_position():
    d = _dvp_from_stats(_stats([
        {"opponent_team": "SOFT", "position": "WR", "week": 1, "fantasy_points_ppr": 30.0},
        {"opponent_team": "HARD", "position": "WR", "week": 1, "fantasy_points_ppr": 10.0},
    ]))
    assert d["dvp_ratio"].mean() == pytest.approx(1.0)
    soft = d.filter(pl.col("defense") == "SOFT")["dvp_ratio"][0]
    assert soft > 1.0


def test_positions_are_scaled_independently():
    """A defence soft against WRs is not necessarily soft against RBs."""
    d = _dvp_from_stats(_stats([
        {"opponent_team": "A", "position": "WR", "week": 1, "fantasy_points_ppr": 40.0},
        {"opponent_team": "B", "position": "WR", "week": 1, "fantasy_points_ppr": 20.0},
        {"opponent_team": "A", "position": "RB", "week": 1, "fantasy_points_ppr": 5.0},
        {"opponent_team": "B", "position": "RB", "week": 1, "fantasy_points_ppr": 25.0},
    ]))
    a_wr = d.filter((pl.col("defense") == "A") & (pl.col("pos") == "WR"))["dvp_ratio"][0]
    a_rb = d.filter((pl.col("defense") == "A") & (pl.col("pos") == "RB"))["dvp_ratio"][0]
    assert a_wr > 1.0 and a_rb < 1.0


def test_non_scoring_positions_are_excluded():
    d = _dvp_from_stats(_stats([
        {"opponent_team": "A", "position": "WR", "week": 1, "fantasy_points_ppr": 10.0},
        {"opponent_team": "A", "position": "CB", "week": 1, "fantasy_points_ppr": 99.0},
    ]))
    assert d["pos"].unique().to_list() == ["WR"]


# --- schedule -------------------------------------------------------------

def test_both_sides_of_every_game_appear():
    o = _opponents_from_schedule(_sched([{"week": 15, "home_team": "KC", "away_team": "BUF"}]))
    assert set(o["team"].to_list()) == {"KC", "BUF"}
    assert o.filter(pl.col("team") == "KC")["opponent"][0] == "BUF"


def test_only_the_playoff_weeks_are_used():
    o = _opponents_from_schedule(_sched([
        {"week": 1, "home_team": "KC", "away_team": "BUF"},
        {"week": 16, "home_team": "KC", "away_team": "DEN"},
    ]))
    assert o["week"].unique().to_list() == [16]


def test_default_weeks_are_the_fantasy_playoffs():
    assert PLAYOFF_WEEKS == (15, 16, 17)


# --- sos ------------------------------------------------------------------

def _three_softs():
    dvp = pl.DataFrame({"defense": ["S1", "S2", "S3", "H1"], "pos": ["WR"] * 4,
                        "ppg_allowed": [30.0, 30.0, 30.0, 10.0],
                        "dvp_ratio": [1.5, 1.5, 1.5, 0.5]})
    opp = pl.DataFrame({"team": ["EASY"] * 3 + ["HARD"] * 3,
                        "week": [15, 16, 17] * 2,
                        "opponent": ["S1", "S2", "S3", "H1", "H1", "H1"]})
    return _sos_from(dvp, opp)


def test_a_soft_playoff_slate_scores_above_one():
    s = _three_softs()
    assert s.filter(pl.col("team") == "EASY")["wk15_17_sos"][0] == pytest.approx(1.5)
    assert s.filter(pl.col("team") == "HARD")["wk15_17_sos"][0] == pytest.approx(0.5)


def test_game_count_is_reported_so_a_partial_slate_is_visible():
    assert _three_softs()["sos_games"].to_list() == [3, 3]


# --- attaching to the board ----------------------------------------------

def test_board_join_uses_canonical_team_codes():
    board = pl.DataFrame({"player": ["A"], "team": ["JAC"], "pos": ["WR"]})
    sos = pl.DataFrame({"team": ["JAX"], "pos": ["WR"], "wk15_17_sos": [1.2],
                        "sos_games": [3]})
    assert attach_sos(board, sos)["wk15_17_sos"][0] == pytest.approx(1.2)


def test_unplaceable_player_gets_null_not_a_default():
    """A silent 1.0 would read as 'average schedule' for someone we could not place."""
    board = pl.DataFrame({"player": ["Free Agent"], "team": ["FA"], "pos": ["WR"]})
    sos = pl.DataFrame({"team": ["JAX"], "pos": ["WR"], "wk15_17_sos": [1.2],
                        "sos_games": [3]})
    assert attach_sos(board, sos)["wk15_17_sos"][0] is None


def test_helper_column_does_not_leak():
    board = pl.DataFrame({"player": ["A"], "team": ["KC"], "pos": ["WR"]})
    sos = pl.DataFrame({"team": ["KC"], "pos": ["WR"], "wk15_17_sos": [1.0], "sos_games": [3]})
    assert not [c for c in attach_sos(board, sos).columns if c.startswith("_")]
