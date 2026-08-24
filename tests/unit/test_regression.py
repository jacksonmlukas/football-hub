"""Touchdown luck: last season's actuals against the current board.

The chain this rests on, each link measured separately:

- Touchdown rate per yard has **zero** year-over-year persistence -- +0.004 receiving,
  -0.030 rushing (`docs/component-projection.md`). A player's touchdown rate tells you
  nothing about his next one.
- Fully regressing it beats carrying his points forward, and the fitted optimal shrink is
  1.0 (`docs/talent-cv.md`, `docs/volume-model.md`).
- The draft market does **not** fully regress it. Weighting prior touchdown points against
  prior yardage points, the room prices touchdowns at 1.02 relative to volume for
  quarterbacks where their true predictive weight is -0.05 (`docs/td-luck.md`).

So a player whose prior season carried more touchdowns than his yardage supports is priced
on something that will not repeat. That is the signal, and it is not what the board's
existing `fp_over_expected` column measures -- the two correlate at +0.16 on the live board,
and a genuine overlap would show as a strong negative.
"""
import polars as pl
import pytest

from hub.draft import regression as R
from hub.models import components as C


def _season(rows):
    """rows: (player, pos, games, rec_yds, rec_td, rush_yds, rush_td, pass_yds, pass_td)"""
    keys = ("player", "pos", "g", "receiving_yards", "receiving_tds", "rushing_yards",
            "rushing_tds", "passing_yards", "passing_tds")
    return pl.DataFrame({k: [r[i] for r in rows] for i, k in enumerate(keys)})


def test_a_player_scoring_exactly_the_positional_rate_has_no_luck():
    """The zero point. If this drifts every number in the column is offset."""
    yards = 1000.0
    tds = yards * C.td_rate("WR", "rec")
    got = R.td_luck(_season([("A", "WR", 16, yards, tds, 0.0, 0.0, 0.0, 0.0)]))
    assert got["td_luck"][0] == pytest.approx(0.0, abs=1e-9)


def test_scoring_above_the_rate_is_positive_luck():
    got = R.td_luck(_season([("A", "WR", 16, 1000.0, 12.0, 0.0, 0.0, 0.0, 0.0)]))
    assert got["td_luck"][0] > 0


def test_scoring_below_the_rate_is_negative_luck():
    got = R.td_luck(_season([("A", "WR", 16, 1400.0, 4.0, 0.0, 0.0, 0.0, 0.0)]))
    assert got["td_luck"][0] < 0


def test_positions_are_held_to_their_own_rate():
    """Tight ends score more touchdowns per yard than receivers -- 0.00705 against 0.00605 --
    so the same line is lucky for one and not the other."""
    line = (1000.0, 6.0, 0.0, 0.0, 0.0, 0.0)
    wr = R.td_luck(_season([("A", "WR", 16, *line)]))["td_luck"][0]
    te = R.td_luck(_season([("B", "TE", 16, *line)]))["td_luck"][0]
    assert wr > te


def test_a_passing_touchdown_is_worth_four_and_a_rushing_one_six():
    """Fantasy scoring, not touchdown counts. Getting this wrong understates quarterbacks by
    a third, and quarterbacks are where the market misprices most."""
    qb = R.td_luck(_season([("A", "QB", 16, 0.0, 0.0, 0.0, 0.0, 4000.0, 40.0)]))
    exp_td = 4000.0 * C.td_rate("QB", "pass")
    assert qb["td_luck"][0] == pytest.approx(4.0 * (40.0 - exp_td) / 16, rel=1e-6)


def test_it_is_reported_per_game_not_per_season():
    """A player who missed half a season is not half as lucky per game."""
    full = R.td_luck(_season([("A", "WR", 16, 1000.0, 12.0, 0.0, 0.0, 0.0, 0.0)]))
    half = R.td_luck(_season([("B", "WR", 8, 500.0, 6.0, 0.0, 0.0, 0.0, 0.0)]))
    assert full["td_luck"][0] == pytest.approx(half["td_luck"][0], rel=1e-6)


def test_a_tiny_sample_is_dropped():
    """Two games of touchdown luck is noise wearing a number's clothes."""
    got = R.td_luck(_season([("A", "WR", 2, 200.0, 4.0, 0.0, 0.0, 0.0, 0.0)]))
    assert got.height == 0


# --- attaching to the board ------------------------------------------------

def test_attaching_leaves_unmatched_players_null_rather_than_dropping_them():
    """Rookies have no prior season. Dropping them would quietly shrink the board."""
    board = pl.DataFrame({"player": ["A", "Rookie"], "pos": ["WR", "WR"]})
    got = R.attach(board, _season([("A", "WR", 16, 1000.0, 12.0, 0.0, 0.0, 0.0, 0.0)]))
    assert got.height == 2
    assert got.filter(pl.col("player") == "Rookie")["td_luck"][0] is None


def test_the_column_survives_a_board_that_already_has_one():
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"], "td_luck": [99.0]})
    got = R.attach(board, _season([("A", "WR", 16, 1000.0, 12.0, 0.0, 0.0, 0.0, 0.0)]))
    assert got["td_luck"][0] != 99.0


def test_names_are_matched_on_a_normalised_form():
    """`A.J. Brown` and `AJ Brown` are the same player; nflverse and ESPN disagree about
    punctuation, and an exact join silently loses him."""
    board = pl.DataFrame({"player": ["A.J. Brown"], "pos": ["WR"]})
    got = R.attach(board, _season([("AJ Brown", "WR", 16, 1000.0, 12.0, 0.0, 0.0, 0.0, 0.0)]))
    assert got["td_luck"][0] is not None


# --- correcting the projection the win-probability sim runs on ------------

def test_a_lucky_quarterback_is_marked_down():
    """The point of the whole exercise: the signal has to reach the objective. ESPN's
    projection carries the same touchdown bias the draft room does, but only for
    quarterbacks -- ppg_next ~ proj + td_luck gives a td_luck coefficient of -0.540,
    95% CI [-1.057, -0.125], 99.5%. So the projection the simulator scores on is too high
    for a quarterback who got lucky, by about half a point per point of luck."""
    board = pl.DataFrame({"player": ["A"], "pos": ["QB"], "proj_blend": [22.0],
                          "td_luck": [4.0]})
    got = R.correct_projection(board)
    assert got["proj_blend"][0] == pytest.approx(22.0 + R.TD_LUCK_BETA["QB"] * 4.0)
    assert got["proj_blend"][0] < 22.0


def test_an_unlucky_quarterback_is_marked_up():
    board = pl.DataFrame({"player": ["A"], "pos": ["QB"], "proj_blend": [18.0],
                          "td_luck": [-2.0]})
    assert R.correct_projection(board)["proj_blend"][0] > 18.0


def test_running_backs_and_receivers_are_left_alone():
    """Measured and not applied. RB comes back at +0.253 with the wrong sign entirely
    (17%), and WR at -0.286 is only directional (89%) -- below what this repo ships on.
    Correcting them would be acting on noise in the name of a mechanism."""
    board = pl.DataFrame({"player": ["A", "B"], "pos": ["RB", "WR"],
                          "proj_blend": [14.0, 14.0], "td_luck": [3.0, 3.0]})
    got = R.correct_projection(board)
    assert got["proj_blend"].to_list() == [14.0, 14.0]


def test_a_player_without_a_prior_season_is_untouched():
    board = pl.DataFrame({"player": ["Rookie"], "pos": ["QB"], "proj_blend": [19.0],
                          "td_luck": [None]})
    assert R.correct_projection(board)["proj_blend"][0] == 19.0


def test_the_correction_cannot_drive_a_projection_negative():
    board = pl.DataFrame({"player": ["A"], "pos": ["QB"], "proj_blend": [1.0],
                          "td_luck": [40.0]})
    assert R.correct_projection(board)["proj_blend"][0] >= 0.0


def test_a_board_without_the_signal_passes_through():
    board = pl.DataFrame({"player": ["A"], "pos": ["QB"], "proj_blend": [22.0]})
    assert R.correct_projection(board)["proj_blend"][0] == 22.0
