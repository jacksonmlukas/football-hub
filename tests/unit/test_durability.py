"""Availability as a per-player trait, and today's injury news as a separate thing.

`TALENT_CV` already carried availability, but only as a population average: it was fitted on
points per *team* game, so missed time is inside it at the positional level. Every running
back therefore carried the same durability risk, and Christian McCaffrey and a back who has
never missed a snap were the same player to the simulation.

Two facts make a per-player version worth having (`docs/durability.md`):

- **Availability persists.** Games missed correlates +0.407 year over year. A player who
  missed six or more games last season misses three or more this season 76% of the time,
  against a 55% base rate; one who missed none does it 41% of the time.
- **The projection does not fully price it.** `ppg_next ~ proj_ppg + prior games missed`
  leaves a coefficient of -0.186 per game pooled, P(<0) = 100%.

The per-position cut is the surprise, and it is why this is not applied uniformly: running
backs come back at -0.065 (71%, nothing). The market already discounts running back
durability, because everyone knows running backs break. The inefficiency is at quarterback
and receiver.
"""
import polars as pl
import pytest

from hub.draft import durability as D


def _seasons(rows):
    """rows: (player, pos, games_played)"""
    return pl.DataFrame({"player": [r[0] for r in rows], "pos": [r[1] for r in rows],
                         "g": [r[2] for r in rows],
                         "ppg": [12.0] * len(rows)})


# --- the trait -------------------------------------------------------------

def test_games_missed_counts_against_the_team_season():
    got = D.games_missed(_seasons([("A", "WR", 13)]))
    assert got["missed"][0] == D.TEAM_GAMES - 13


def test_a_player_who_never_missed_a_game_carries_no_markdown():
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"], "proj_blend": [14.0]})
    got = D.correct_projection(D.attach(board, _seasons([("A", "WR", D.TEAM_GAMES)])))
    assert got["proj_blend"][0] == pytest.approx(14.0)


def test_a_fragile_receiver_is_marked_down():
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"], "proj_blend": [14.0]})
    got = D.correct_projection(D.attach(board, _seasons([("A", "WR", 11)])))
    assert got["proj_blend"][0] == pytest.approx(14.0 + D.BETA["WR"] * 6)
    assert got["proj_blend"][0] < 14.0


def test_quarterbacks_take_the_largest_markdown():
    """-0.457 against -0.151. If these ever invert, something upstream changed."""
    assert abs(D.BETA["QB"]) > abs(D.BETA["WR"])


def test_running_backs_are_left_alone():
    """Not a threshold call so much as a finding: -0.065 at 71%, indistinguishable from
    zero. The market already prices running back durability, so there is no residual to
    take. Marking them down again would be double-counting the market's own discount."""
    board = pl.DataFrame({"player": ["A"], "pos": ["RB"], "proj_blend": [14.0]})
    got = D.correct_projection(D.attach(board, _seasons([("A", "RB", 8)])))
    assert got["proj_blend"][0] == pytest.approx(14.0)


def test_a_backup_is_not_treated_as_fragile():
    """Games missed here is really games *not played*, which for a low-usage player is being
    a backup rather than being hurt. Requiring a real prior role is what keeps the two
    apart."""
    got = D.games_missed(_seasons([("Bench", "WR", 4)]).with_columns(
        pl.lit(1.5).alias("ppg")))
    assert got.height == 0


def test_a_player_with_no_prior_season_is_untouched():
    board = pl.DataFrame({"player": ["Rookie"], "pos": ["WR"], "proj_blend": [14.0]})
    got = D.correct_projection(D.attach(board, _seasons([("Someone", "WR", 10)])))
    assert got["proj_blend"][0] == pytest.approx(14.0)


def test_the_markdown_cannot_drive_a_projection_negative():
    board = pl.DataFrame({"player": ["A"], "pos": ["QB"], "proj_blend": [0.5]})
    got = D.correct_projection(D.attach(board, _seasons([("A", "QB", 1)])))
    assert got["proj_blend"][0] >= 0.0


# --- today's news, which is a different quantity ---------------------------

def test_current_injury_status_is_carried_but_not_priced():
    """A player hurt *now* is not the same as a player who was fragile *last year*, and
    there is nothing here to fit a coefficient against -- no history of preseason
    designations against outcomes. So it is surfaced for judgment and deliberately left out
    of the projection, rather than given an invented number."""
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"], "proj_blend": [14.0],
                          "injury_status": ["OUT"]})
    got = D.correct_projection(board)
    assert got["proj_blend"][0] == pytest.approx(14.0)


def test_a_status_worth_flagging_is_recognised():
    assert D.is_flagworthy("OUT") and D.is_flagworthy("INJURY_RESERVE")
    assert D.is_flagworthy("QUESTIONABLE") and D.is_flagworthy("DOUBTFUL")
    assert not D.is_flagworthy("ACTIVE")
    assert not D.is_flagworthy(None)
