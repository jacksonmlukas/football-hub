"""`edge` = where your room drafts a player, minus where consensus ranks him.

Two ways this has already been wrong:

  1. posRank (a *positional* rank) subtracted from an overall consensus rank.
  2. Real overall ADP subtracted from a consensus rank computed over a different
     population -- every board row (~1474) vs ESPN's draftable pool (~460).
     That makes edge a disguised restatement of -consensus_rank.

Both are the same underlying error: subtracting two numbers that are not on a
common scale. These tests pin the scale, not just the arithmetic.
"""
from typing import cast

import polars as pl

from hub.draft.board import _adp_join_failure_rate, _adp_saturation_cutoff, _attach_edge


def _board(players, ecrs):
    return pl.DataFrame({"player": players, "ecr": [float(e) for e in ecrs]})


def _adp(players, adps):
    return pl.DataFrame({"player": players, "adp": [float(a) for a in adps]},
                        schema={"player": pl.Utf8, "adp": pl.Float64})


# --- saturation detection -------------------------------------------------

def test_no_saturation_when_every_adp_is_distinct():
    assert _adp_saturation_cutoff(pl.Series("adp", [1.0, 2.0, 3.0, 4.0]), teams=12) is None


def test_tail_cluster_is_detected():
    """ESPN parks undrafted players in a tail band instead of leaving it null."""
    s = pl.Series("adp", [1.0, 2.0, 3.0] + [169.99] * 12)
    assert _adp_saturation_cutoff(s, teams=12) == 169.0


def test_jittered_band_is_caught_not_just_the_exact_spike():
    """The real shape: a dense band of near-distinct values, not one repeated number.

    Counting exact repeats would miss this entirely -- no value appears twice -- and
    every one of these would be priced as a genuine round-14 pick.
    """
    band = [169.0 + i / 100 for i in range(20)]
    s = pl.Series("adp", [1.0, 2.0, 3.0, *band])
    assert _adp_saturation_cutoff(s, teams=12) == 169.0


def test_cluster_smaller_than_league_size_is_not_saturation():
    """Eleven players genuinely going near pick 90 is a real cluster, not a sentinel."""
    s = pl.Series("adp", [1.0, 2.0] + [90.0] * 11)
    assert _adp_saturation_cutoff(s, teams=12) is None


def test_lowest_qualifying_cluster_wins():
    s = pl.Series("adp", [1.0] + [150.0] * 12 + [170.0] * 20)
    assert _adp_saturation_cutoff(s, teams=12) == 150.0


def test_empty_series_has_no_saturation_point():
    assert _adp_saturation_cutoff(pl.Series("adp", [], dtype=pl.Float64), teams=12) is None


# --- edge on a common scale -----------------------------------------------

def test_saturated_players_get_null_adp_and_null_edge():
    board = _board([f"P{i}" for i in range(15)], range(1, 16))
    adp = _adp([f"P{i}" for i in range(15)], [1, 2, 3] + [169.99] * 12)
    out = _attach_edge(board, adp, teams=12)
    sat = out.filter(pl.col("player").is_in([f"P{i}" for i in range(3, 15)]))
    assert sat["adp"].is_null().all()
    assert sat["edge"].is_null().all()


def test_consensus_is_reranked_within_the_drafted_pool():
    """The core fix: consensus_pick must count only drafted players, not the whole board.

    P0/P1/P2 are drafted and sit at overall ecr 1/2/3. Hundreds of undrafted players
    follow. Their consensus_pick must be 1/2/3 -- their rank among the drafted -- so
    edge stays on ADP's scale instead of inheriting the full board's.
    """
    players = [f"P{i}" for i in range(200)]
    board = _board(players, range(1, 201))
    adp = _adp(players, [10.0, 20.0, 30.0] + [169.99] * 197)
    out = _attach_edge(board, adp, teams=12)
    drafted = out.filter(pl.col("adp").is_not_null()).sort("ecr")
    assert drafted["consensus_pick"].to_list() == [1, 2, 3]
    assert drafted["edge"].to_list() == [9.0, 18.0, 27.0]


def test_positive_edge_means_the_room_lets_him_fall():
    board = _board(["Falls", "Reaches"], [1, 2])
    adp = _adp(["Falls", "Reaches"], [40.0, 1.0])
    out = _attach_edge(board, adp, teams=12)
    assert out.filter(pl.col("player") == "Falls")["edge"][0] > 0
    assert out.filter(pl.col("player") == "Reaches")["edge"][0] < 0


def test_edge_magnitude_stays_within_the_draftable_pool():
    """Regression on the 1474-vs-463 bug: edge must not scale with board size."""
    players = [f"P{i}" for i in range(1000)]
    board = _board(players, range(1, 1001))
    adp = _adp(players, [float(i + 1) for i in range(50)] + [169.99] * 950)
    out = _attach_edge(board, adp, teams=12)
    assert cast(float, out["edge"].abs().max()) < 100


def test_players_without_adp_survive_the_join():
    board = _board(["A", "B"], [1, 2])
    adp = _adp(["A"], [5.0])
    out = _attach_edge(board, adp, teams=12)
    assert out.height == 2
    assert out.filter(pl.col("player") == "B")["edge"].is_null().all()


# --- #370 (S12): the ESPN-ADP-to-board join-failure rate --------------------

def test_an_exact_match_is_not_a_failure():
    board = _board(["A.J. Brown"], [1])
    adp = _adp(["A.J. Brown"], [1.0])
    assert _adp_join_failure_rate(adp, board) == (0, 1)


def test_a_spelling_variant_recoverable_only_under_relaxed_key_is_counted():
    """`_attach_edge` joins on the raw string, so `Mike Thomas` fails that join even though
    it is the same player as `Michael Thomas` -- and `player_key` does not collapse a
    nickname the way it collapses punctuation, so `michael thomas` and `mike thomas` are two
    keys. `relaxed_key` (first initial, surname) is the looser comparison that still catches
    it: `m thomas` either way."""
    board = _board(["Michael Thomas"], [1])
    adp = _adp(["Mike Thomas"], [1.0])
    assert _adp_join_failure_rate(adp, board) == (1, 1)


def test_a_name_the_board_never_ranks_is_not_counted():
    """Kickers and defenses: ESPN prices them, FantasyPros' consensus board this repo builds
    from does not by design, and a name matching nothing even loosely is that, not a spelling
    failure."""
    board = _board(["A.J. Brown"], [1])
    adp = _adp(["San Francisco D/ST"], [150.0])
    assert _adp_join_failure_rate(adp, board) == (0, 1)


def test_the_rate_is_reported_against_every_adp_name_not_only_the_failures():
    board = _board(["A", "Bobby Smith"], [1, 2])
    adp = _adp(["A", "Bob Smith", "Kicker Guy"], [1.0, 2.0, 150.0])
    # "A" matches exactly; "Bob Smith" relaxes to "b smith", the same as "Bobby Smith";
    # "Kicker Guy" relaxes to "k guy", which nothing on the board carries.
    assert _adp_join_failure_rate(adp, board) == (1, 3)


def test_no_adp_names_reports_zero_over_zero_rather_than_dividing_by_it():
    board = _board(["A"], [1])
    adp = _adp([], [])
    assert _adp_join_failure_rate(adp, board) == (0, 0)


def test_attach_market_prints_the_join_failure_rate(monkeypatch, capsys):
    from hub.draft import board as board_mod

    b = pl.DataFrame({
        "player": ["A", "Bobby Smith"], "pos": ["WR", "RB"], "ecr": [1.0, 2.0],
        "games": [17, 17], "xfp_per_game": [10.0, 8.0],
    })
    adp = _adp(["A", "Bob Smith"], [1.0, 2.0])
    monkeypatch.setattr(board_mod, "replacement_levels", lambda *a, **k: {"WR": 0.0, "RB": 0.0})
    monkeypatch.setattr(board_mod.durability, "correct_projection", lambda df: df)
    monkeypatch.setattr(board_mod, "blend", lambda: pl.col("ecr").alias("proj_blend"))
    # `corrected_adp` is imported inside `_attach_market`'s own body, so the patch has to
    # land on the module it is imported *from*, not on `board_mod`.
    monkeypatch.setattr("hub.draft.optimize.corrected_adp", lambda df: pl.col("ecr"))

    board_mod._attach_market(b, adp, league_size=12, season=2025, season_ahead=2026)
    out = capsys.readouterr().out
    assert "join failures 1/2" in out
