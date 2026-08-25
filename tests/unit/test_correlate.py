"""Within-game correlation of standardised weekly points.

`docs/correlation.md` said opponent correlation was not modelled and nobody priced it. This
measures it. The method is deliberately identical to the teammate measurement so the two
numbers are comparable — running the same function with `same_team=True` reproduces
`TEAMMATE_RHO`, which is the check that it is the same method.

All offline.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import correlate


def _stats(rows):
    """(season, week, game_id, team, player_id, position, points)."""
    cols = ["season", "week", "game_id", "team", "player_id", "position",
            "fantasy_points_ppr"]
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"season": pl.Int32, "week": pl.Int32, "game_id": pl.Utf8,
                                "team": pl.Utf8, "player_id": pl.Utf8,
                                "position": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _game(week, home, away, players):
    """players: (player_id, team, pos, pts)."""
    gid = f"2024_{week:02d}_{away}_{home}"
    return [(2024, week, gid, t, pid, pos, pts) for pid, t, pos, pts in players]


# --- standardising --------------------------------------------------------

def test_a_players_weeks_become_z_scores_within_his_own_season():
    rows = []
    for w in range(1, 13):
        rows += _game(w, "KC", "LV", [("a", "KC", "QB", float(w)),
                                      ("b", "LV", "QB", 10.0 + w)])
    z = correlate.standardised(_stats(rows))
    a = z.filter(pl.col("player_id") == "a")["z"].to_numpy()
    assert a.mean() == pytest.approx(0.0, abs=1e-9)
    assert a.std(ddof=1) == pytest.approx(1.0, rel=1e-6)


def test_a_player_with_too_few_weeks_is_excluded():
    """Below eight weeks one big game defines the standardisation, and a correlation built
    on that is a correlation between two accidents."""
    rows = []
    for w in range(1, 4):
        rows += _game(w, "KC", "LV", [("a", "KC", "QB", float(w)),
                                      ("b", "LV", "QB", float(w))])
    assert correlate.standardised(_stats(rows)).is_empty()


def test_a_player_with_no_variance_is_excluded():
    """Dividing by a zero spread gives infinities that poison every pair he is in."""
    rows = []
    for w in range(1, 13):
        rows += _game(w, "KC", "LV", [("flat", "KC", "QB", 10.0),
                                      ("varies", "LV", "QB", float(w))])
    z = correlate.standardised(_stats(rows))
    assert "flat" not in z["player_id"].to_list()
    assert "varies" in z["player_id"].to_list()


def test_kickers_and_defences_are_excluded():
    rows = []
    for w in range(1, 13):
        rows += _game(w, "KC", "LV", [("k", "KC", "K", float(w)),
                                      ("q", "LV", "QB", float(w))])
    assert set(correlate.standardised(_stats(rows))["position"].to_list()) == {"QB"}


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="game_id"):
        correlate.standardised(pl.DataFrame({"season": [2024], "week": [1]}))


# --- pairing --------------------------------------------------------------

def _two_qbs(corr_sign, n=300, seed=0):
    """Two QBs on opposite teams whose weeks move together (or apart)."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n + 1):
        shared = rng.normal()
        a = shared + rng.normal(0, 0.3)
        b = corr_sign * shared + rng.normal(0, 0.3)
        rows += _game(w, "KC", "LV", [("a", "KC", "QB", 15.0 + 5 * a),
                                      ("b", "LV", "QB", 15.0 + 5 * b)])
    return _stats(rows)


def test_opponents_are_paired_and_teammates_are_not():
    z = correlate.standardised(_two_qbs(+1))
    opp = correlate.pair_correlations(z, same_team=False)
    same = correlate.pair_correlations(z, same_team=True)
    assert opp.height == 1 and opp["rho"][0] > 0.7
    assert same.is_empty(), "the two QBs are on opposite teams"


def test_a_negative_relationship_comes_back_negative():
    """RB-RB came back negative in the real data -- one team running out the clock means the
    other is throwing. The sign has to survive the pipeline."""
    z = correlate.standardised(_two_qbs(-1))
    got = correlate.pair_correlations(z, same_team=False)
    assert got["rho"][0] < -0.7


def test_players_in_different_games_are_never_paired():
    """Correlation here is a within-game quantity. Pairing across games would measure
    league-wide scoring weather instead."""
    rows = []
    for w in range(1, 13):
        rows += _game(w, "KC", "LV", [("a", "KC", "QB", float(w))])
        rows += _game(w, "SF", "SEA", [("b", "SF", "QB", float(w))])
    z = correlate.standardised(_stats(rows))
    assert correlate.pair_correlations(z, same_team=False, min_pairs=1).is_empty()


def test_position_pairs_are_unordered():
    """QB-WR and WR-QB are one relationship, not two half-sized ones."""
    rows = []
    for w in range(1, 13):
        rows += _game(w, "KC", "LV", [("q", "KC", "QB", float(w)),
                                      ("r", "LV", "WR", float(w) * 1.5)])
        rows += _game(w, "SF", "SEA", [("w2", "SF", "WR", float(w)),
                                       ("q2", "SEA", "QB", float(w) * 1.5)])
    z = correlate.standardised(_stats(rows))
    got = correlate.pair_correlations(z, same_team=False, min_pairs=1)
    assert got.height == 1
    assert (got["pos_a"][0], got["pos_b"][0]) == ("QB", "WR")


def test_thin_pairs_are_not_reported():
    """At n=200 the standard error is 0.071, wider than every teammate effect found."""
    z = correlate.standardised(_two_qbs(+1, n=20))
    assert correlate.pair_correlations(z, same_team=False).is_empty()
    assert not correlate.pair_correlations(z, same_team=False, min_pairs=5).is_empty()


def test_the_standard_error_is_the_fisher_one():
    z = correlate.standardised(_two_qbs(+1, n=400))
    got = correlate.pair_correlations(z, same_team=False)
    assert got["se"][0] == pytest.approx(1 / np.sqrt(got["n"][0] - 3))


# --- reporting ------------------------------------------------------------

def test_significance_filters_on_standard_errors_not_size():
    """A tiny effect measured very precisely outranks a larger one measured loosely. QB here
    is 0.15/0.033 = 4.5 se; RB is 0.01/0.0005 = 20 se on a hundredth of the correlation."""
    table = pl.DataFrame({"pos_a": ["QB", "RB"], "pos_b": ["QB", "RB"], "n": [900, 900000],
                          "rho": [0.15, 0.01], "se": [0.033, 0.0005]})
    got = correlate.significant(table)
    assert got.height == 2
    assert got["pos_a"].to_list()[0] == "RB", "sorted by standard errors away, not by rho"


def test_an_imprecise_effect_is_filtered_out_however_large():
    """The failure this guards: a big-looking rho on 30 pairs is not a finding. At n=30 the
    standard error is 0.192, so 0.30 is only 1.6 se -- and 0.40 would actually clear, which
    is a useful reminder of how little precision thirty pairs buys."""
    thin = pl.DataFrame({"pos_a": ["TE"], "pos_b": ["TE"], "n": [30],
                         "rho": [0.30], "se": [0.192]})
    assert correlate.significant(thin).is_empty()


def test_an_empty_table_is_handled():
    assert correlate.significant(pl.DataFrame(
        schema={"pos_a": pl.Utf8, "pos_b": pl.Utf8, "n": pl.Int64,
                "rho": pl.Float64, "se": pl.Float64})).is_empty()


def test_the_cli_runs_offline(monkeypatch, capsys, tmp_path):
    import nflreadpy as nfl
    monkeypatch.setattr(nfl, "load_player_stats", lambda *a, **k: _two_qbs(+1, n=400))
    out = tmp_path / "c.parquet"
    assert correlate.main(["--seasons", "2024", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "OPPONENT correlation" in text
    assert out.exists()


def test_the_cli_can_report_teammates_instead(monkeypatch, capsys):
    import nflreadpy as nfl
    rows = []
    for w in range(1, 40):
        rows += _game(w, "KC", "LV", [("q", "KC", "QB", float(w)),
                                      ("r", "KC", "WR", float(w) * 1.2)])
    monkeypatch.setattr(nfl, "load_player_stats", lambda *a, **k: _stats(rows))
    assert correlate.main(["--seasons", "2024", "--teammates"]) == 0
    assert "TEAMMATE correlation" in capsys.readouterr().out
