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


# --- opponent adjustment (issue #180) --------------------------------------

def _games(rows):
    """One row per (defence, offence, position, week) with the points the offence scored."""
    return pl.DataFrame(rows, schema={"opponent_team": pl.Utf8, "team": pl.Utf8,
                                      "position": pl.Utf8, "week": pl.Int64,
                                      "fantasy_points_ppr": pl.Float64})


def _two_defences_two_schedules():
    """Two identical defences. One faced only the strong offence, one only the weak.

    Every offence scores its own level against everyone: STRONG scores 30 wherever it plays,
    WEAK scores 10. So the two defences are the same defence, and an unadjusted metric
    ranks whoever drew the harder slate as the worse unit -- which is the error the metric
    exists to avoid making about the *schedule*. A third defence faces both, which is what
    lets the offences' levels be identified at all.
    """
    rows = []
    for w, off in ((1, "STRONG"), (2, "STRONG"), (3, "STRONG")):
        rows.append({"opponent_team": "DEF_HARD", "team": off, "position": "WR",
                     "week": w, "fantasy_points_ppr": 30.0})
    for w, off in ((1, "WEAK"), (2, "WEAK"), (3, "WEAK")):
        rows.append({"opponent_team": "DEF_EASY", "team": off, "position": "WR",
                     "week": w, "fantasy_points_ppr": 10.0})
    for w, off, pts in ((1, "STRONG", 30.0), (2, "WEAK", 10.0), (3, "STRONG", 30.0),
                        (4, "WEAK", 10.0)):
        rows.append({"opponent_team": "DEF_BOTH", "team": off, "position": "WR",
                     "week": w, "fantasy_points_ppr": pts})
    return _games(rows)


def test_unadjusted_ranks_the_harder_slate_as_the_worse_defence():
    """The premise, held so the adjustment below is shown to change something real."""
    d = _dvp_from_stats(_two_defences_two_schedules())
    hard = d.filter(pl.col("defense") == "DEF_HARD")["dvp_ratio"][0]
    easy = d.filter(pl.col("defense") == "DEF_EASY")["dvp_ratio"][0]
    assert hard > easy, "premise: unadjusted, the harder slate looks like the worse defence"


def test_adjusted_does_not_rank_an_identical_defence_worse_for_its_schedule():
    """#180's second criterion, on the fixture built to show it.

    With the offences' levels taken out, DEF_HARD and DEF_EASY are the same defence and
    their adjusted ratios agree. Not exactly -- ridge shrinks every effect toward zero, so
    with three games each they meet somewhere short of a full correction -- but the gap
    closes to a small fraction of the unadjusted one, and it closes in the right direction.
    """
    from hub.draft.playoff_sos import _dvp_from_stats as dvp

    raw = dvp(_two_defences_two_schedules())
    adj = dvp(_two_defences_two_schedules(), ridge=1.0)
    gap_raw = (raw.filter(pl.col("defense") == "DEF_HARD")["dvp_ratio"][0]
               - raw.filter(pl.col("defense") == "DEF_EASY")["dvp_ratio"][0])
    gap_adj = (adj.filter(pl.col("defense") == "DEF_HARD")["dvp_ratio"][0]
               - adj.filter(pl.col("defense") == "DEF_EASY")["dvp_ratio"][0])
    assert gap_raw > 0.5, f"premise: the unadjusted gap is large ({gap_raw:.3f})"
    assert abs(gap_adj) < 0.25 * gap_raw, (
        f"adjusted gap {gap_adj:.3f} did not close against unadjusted {gap_raw:.3f}")


def test_the_adjustment_keeps_the_league_mean_at_one():
    """A ratio indexed to the league average has to average to one after adjustment too,
    or every team's number has moved for a reason that is not its schedule."""
    adj = _dvp_from_stats(_two_defences_two_schedules(), ridge=1.0)
    assert adj["dvp_ratio"].mean() == pytest.approx(1.0, abs=1e-9)


def test_early_season_shrinks_toward_average_rather_than_exploding():
    """#180's third criterion. One game per defence, and each faced a different offence.

    Unregularised, that system is underdetermined -- every defence effect can be traded
    off against its lone opponent's effect -- and a solver would return whatever the
    numerics hand it. Ridge returns a number that sits between the raw ratio and one, and
    the more it is penalised the closer to one it sits.
    """
    early = _games([
        {"opponent_team": "D1", "team": "O1", "position": "RB", "week": 1,
         "fantasy_points_ppr": 30.0},
        {"opponent_team": "D2", "team": "O2", "position": "RB", "week": 1,
         "fantasy_points_ppr": 10.0},
    ])
    raw = _dvp_from_stats(early).filter(pl.col("defense") == "D1")["dvp_ratio"][0]
    light = _dvp_from_stats(early, ridge=0.5).filter(pl.col("defense") == "D1")["dvp_ratio"][0]
    heavy = _dvp_from_stats(early, ridge=8.0).filter(pl.col("defense") == "D1")["dvp_ratio"][0]
    assert raw > light > heavy > 1.0, (
        f"expected raw {raw:.3f} > light {light:.3f} > heavy {heavy:.3f} > 1, the shrink "
        f"toward average growing with the penalty")


def test_the_sensitivity_reports_the_ranking_at_every_penalty():
    """The disposition: a reported range rather than a fitted penalty.

    What the table has to carry is whether the *ranking* moves across the range, because
    that is the only thing a reader can act on -- a penalty that reorders the slate is a
    decision, one that does not is a detail.
    """
    from hub.draft.playoff_sos import RIDGE_PENALTIES, sos_sensitivity

    games = _two_defences_two_schedules()
    sched = _sched([{"week": 15, "home_team": "A", "away_team": "B"},
                    {"week": 16, "home_team": "A", "away_team": "C"},
                    {"week": 17, "home_team": "B", "away_team": "C"}])
    # Teams A/B/C are offences on the schedule; the defences they face are looked up by
    # opponent. Map the fixture's defence names onto them.
    games = games.with_columns(pl.col("opponent_team").replace(
        {"DEF_HARD": "A", "DEF_EASY": "B", "DEF_BOTH": "C"}))
    table = sos_sensitivity(games, sched)
    assert set(table["ridge"].unique().to_list()) >= set(RIDGE_PENALTIES), (
        "a penalty in the declared range is missing from the report")
    assert (table["ridge"] == 0.0).any(), "the unadjusted metric is not in the table as ridge 0"
    assert "rank" in table.columns and "team" in table.columns


def test_stats_with_no_offence_column_are_refused_rather_than_adjusted_blind():
    """The adjustment removes the offence's effect, so a frame that cannot say who scored
    has nothing to remove. Refusing names both column names nflverse has used."""
    no_offence = _stats([{"opponent_team": "D", "position": "WR", "week": 1,
                          "fantasy_points_ppr": 10.0}])
    with pytest.raises(ValueError, match=r"team.*recent_team"):
        _dvp_from_stats(no_offence, ridge=1.0)


def test_ranking_movement_reads_the_default_from_config():
    """`rank_default` is the rank at the penalty the board actually reads -- which is a
    setting in `DraftConfig`, not a constant in this module, so the two cannot drift."""
    from hub.config import DraftConfig
    from hub.draft.playoff_sos import ranking_movement, sos_sensitivity

    games = _two_defences_two_schedules().with_columns(pl.col("opponent_team").replace(
        {"DEF_HARD": "A", "DEF_EASY": "B", "DEF_BOTH": "C"}))
    sched = _sched([{"week": 15, "home_team": "A", "away_team": "B"},
                    {"week": 16, "home_team": "A", "away_team": "C"},
                    {"week": 17, "home_team": "B", "away_team": "C"}])
    table = sos_sensitivity(games, sched)
    move = ranking_movement(table)
    assert "rank_default" in move.columns and move["rank_default"].null_count() == 0
    at_default = table.filter(pl.col("ridge") == DraftConfig().sos_ridge)
    assert at_default.height > 0, "the config default is not one of the swept penalties"
    assert move["spread"].dtype == pl.Int64


def test_playoff_sos_reads_the_config_default_and_honours_an_override(monkeypatch):
    """The entry point, with the two bulk pulls stood in for.

    Three things at once: the default penalty is the config's and not a literal here; `None`
    reproduces the unadjusted metric every published number was measured on; and a float
    overrides for one call. Covered because the two lines that resolve the default sit inside
    the network function, and a floor raised for them would have hidden the pull itself.
    """
    import types

    from hub.config import DraftConfig
    from hub.draft import playoff_sos as PS

    games = _two_defences_two_schedules().with_columns(
        pl.col("opponent_team").replace({"DEF_HARD": "A", "DEF_EASY": "B", "DEF_BOTH": "C"}),
        pl.lit("REG").alias("season_type"))
    sched = _sched([{"week": 15, "home_team": "A", "away_team": "B"},
                    {"week": 16, "home_team": "A", "away_team": "C"},
                    {"week": 17, "home_team": "B", "away_team": "C"}]
                   ).with_columns(pl.lit(PS.SEASON_AHEAD).alias("season"))
    fake = types.SimpleNamespace(load_player_stats=lambda seasons: games,
                                 load_schedules=lambda: sched)
    monkeypatch.setitem(__import__("sys").modules, "nflreadpy", fake)

    by_config = PS.playoff_sos()
    explicit = PS.playoff_sos(ridge=DraftConfig().sos_ridge)
    unadjusted = PS.playoff_sos(ridge=None)
    assert by_config.equals(explicit), "the default did not resolve to DraftConfig.sos_ridge"
    assert not by_config.equals(unadjusted), "the config default is the unadjusted metric"
    assert PS.playoff_sos(ridge=8.0).height == by_config.height
