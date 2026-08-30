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

def test_a_player_ruled_out_is_marked_down():
    """Fitted against week-1 injury designations, which are the closest historical analogue
    to an August one: -1.631 points per team game, 95% CI [-2.554, -0.736], P(<0) = 100%.
    Players carrying it played 7.7 games against 11.7 for the undesignated."""
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"], "proj_blend": [14.0],
                          "injury_status": ["OUT"]})
    got = D.correct_projection(board)
    assert got["proj_blend"][0] == pytest.approx(14.0 + D.INJURY_BETA["OUT"])
    assert got["proj_blend"][0] < 14.0


def test_injured_reserve_is_treated_as_at_least_as_bad_as_out():
    """Nobody on IR appears on a practice report, so IR has no fitted coefficient of its
    own. Starting a season on IR means missing at least four games by rule, so it is at
    least as severe as Out -- borrowing that coefficient understates it, which is the
    direction to be wrong in."""
    board = pl.DataFrame({"player": ["A", "B"], "pos": ["WR", "WR"],
                          "proj_blend": [14.0, 14.0],
                          "injury_status": ["INJURY_RESERVE", "OUT"]})
    got = D.correct_projection(board)["proj_blend"].to_list()
    assert got[0] <= got[1] < 14.0


def test_questionable_is_carried_but_deliberately_not_priced():
    """Two reasons, and the second matters more than the first. The fitted coefficient does
    not clear significance (-0.949, 90.2%, n=36). And an August QUESTIONABLE is a different
    population from the week-1 one it would be fitted on -- 12.6% of the August board
    against 2.9% at week 1, **4.4x more common**. Applying a week-1 number to it would price
    an eighth of the board on a coefficient estimated from a much sicker group."""
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"], "proj_blend": [14.0],
                          "injury_status": ["QUESTIONABLE"]})
    assert D.correct_projection(board)["proj_blend"][0] == pytest.approx(14.0)


def test_an_active_player_is_untouched():
    board = pl.DataFrame({"player": ["A"], "pos": ["RB"], "proj_blend": [14.0],
                          "injury_status": ["ACTIVE"]})
    assert D.correct_projection(board)["proj_blend"][0] == pytest.approx(14.0)


def test_the_injury_markdown_applies_to_every_position():
    """Unlike the durability trait, which the market prices for running backs. Being ruled
    out is not a trait the market has had a chance to discount -- it is news."""
    board = pl.DataFrame({"player": ["A", "B"], "pos": ["RB", "TE"],
                          "proj_blend": [14.0, 14.0], "injury_status": ["OUT", "OUT"]})
    got = D.correct_projection(board)["proj_blend"].to_list()
    assert all(v < 14.0 for v in got)


def test_a_status_worth_flagging_is_recognised():
    assert D.is_flagworthy("OUT") and D.is_flagworthy("INJURY_RESERVE")
    assert D.is_flagworthy("QUESTIONABLE") and D.is_flagworthy("DOUBTFUL")
    assert not D.is_flagworthy("ACTIVE")
    assert not D.is_flagworthy(None)


# --- the shape both prior-season signals share (improvements.md #15) --------

def test_the_join_keeps_a_null_for_a_player_with_no_prior_season():
    """Null and zero mean different things: zero is "played every game", null is "we do not
    know", and filling one with the other calls every rookie durable. Rookies are half an
    early board."""
    import polars as pl

    from hub.draft import prior_signal
    board = pl.DataFrame({"player": ["A.J. Brown", "Rookie Guy"], "pos": ["WR", "WR"]})
    signal = pl.DataFrame({"player": ["AJ Brown"], "missed": [3.0]})
    got = prior_signal.join_by_player(board, signal, "missed")
    assert got["missed"].to_list() == [3.0, None], "punctuation must not drop him"


def test_an_empty_signal_yields_a_null_column_not_a_missing_one():
    import polars as pl

    from hub.draft import prior_signal
    board = pl.DataFrame({"player": ["A"], "pos": ["WR"]})
    got = prior_signal.join_by_player(board, pl.DataFrame(), "td_luck")
    assert got["td_luck"].to_list() == [None]


def test_an_unlisted_position_is_priced_at_zero_not_at_a_pooled_default():
    """A position absent from a BETA is one the fit found nothing for, and inventing a
    coefficient for it would ship an effect nobody measured."""
    import polars as pl

    from hub.draft import prior_signal
    d = pl.DataFrame({"pos": ["QB", "TE"], "missed": [2.0, 2.0]})
    got = d.select(prior_signal.priced("missed", {"QB": -0.5}).alias("adj"))
    assert got["adj"].to_list() == [-1.0, 0.0]


def test_both_modules_use_the_shared_shape():
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "draft"
    for rel in ("durability.py", "regression.py"):
        names = {a.name for n in ast.walk(ast.parse((root / rel).read_text()))
                 if isinstance(n, ast.ImportFrom) for a in n.names}
        assert "prior_signal" in names, f"{rel} still carries its own copy of the join"
