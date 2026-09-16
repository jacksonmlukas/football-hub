"""`hub.draft.impute_cv`: IMPUTE_CV measured on the players actually imputed (#277).

The shipped constant was measured by blanking observed veterans inside the top 200 and
re-imputing them from the rest. The players the board imputes are rookies with no prior NFL
season, and a veteran's rank-to-production residual is not a rookie's. This module measures
the rookie residual; these tests hold the population rules on synthetic frames so the script
that runs it on the archive can be read for its numbers rather than its plumbing.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hub.draft import impute_cv


def _board(rows):
    """(player, pos, ecr, xfp_per_game, xfp_imputed) in the board's shape."""
    return pl.DataFrame({
        "player": [r[0] for r in rows], "pos": [r[1] for r in rows],
        "ecr": [float(r[2]) for r in rows], "xfp_per_game": [float(r[3]) for r in rows],
        "xfp_imputed": [bool(r[4]) for r in rows],
    })


def _stats(rows):
    """(player_display_name, season) weekly lines, one per row."""
    return pl.DataFrame({"player_display_name": [r[0] for r in rows],
                         "season": [int(r[1]) for r in rows],
                         "week": [1] * len(rows), "fantasy_points_ppr": [1.0] * len(rows)})


def _realised(rows):
    """(full_name, position, fp, xfp, games) as `board.expected_points` returns them."""
    return pl.DataFrame({"full_name": [r[0] for r in rows], "position": [r[1] for r in rows],
                         "fp": [float(r[2]) for r in rows], "xfp": [float(r[3]) for r in rows],
                         "games": [int(r[4]) for r in rows]})


def test_a_rookie_is_a_player_whose_first_season_line_is_the_board_season():
    """The proxy for nflverse `rookie_year`, stated: the first season a player has any weekly
    line. A player whose first line predates the board season is a veteran however he is
    ranked, and one with no line at all has no season to measure."""
    stats = _stats([("Vet A", 2022), ("Vet A", 2023), ("Rookie B", 2024), ("Rookie B", 2025),
                    ("Later C", 2025)])
    first = impute_cv.first_seasons(stats)
    got = dict(zip(first["player"], first["first_season"], strict=True))
    assert got == {"vet a": 2022, "rookie b": 2024, "later c": 2025}


def test_the_population_is_rookies_inside_the_top_200_who_were_imputed_and_played():
    """Four rules, each counted: inside the top 200 by consensus; a rookie; imputed by the
    board (a rookie carrying a prior xFP is a join collision, not a measurement); and enough
    games for a per-game rate to mean something. An imputed player with no weekly line in
    any season is counted apart: the proxy cannot say what he is."""
    board = _board([
        ("Rookie A", "WR", 30, 12.0, True),    # in: rookie, imputed, played 8
        ("Rookie B", "RB", 60, 10.0, True),    # out: 3 games
        ("Rookie C", "TE", 250, 6.0, True),    # out: past the top 200
        ("Vet D", "WR", 40, 14.0, False),      # out: veteran
        ("Rookie E", "QB", 90, 15.0, False),   # out: a rookie with a prior xFP -- a collision
        ("Rookie F", "WR", 120, 8.0, True),    # out: no season line at all
    ])
    stats = _stats([("Rookie A", 2024), ("Rookie B", 2024), ("Rookie C", 2024),
                    ("Vet D", 2021), ("Vet D", 2024), ("Rookie E", 2024)])
    # Rookie A at exactly `min_games`: the floor is inclusive, eight games is a rate.
    realised = _realised([("Rookie A", "WR", 120.0, 112.0, 8), ("Rookie B", "RB", 30.0, 28.0, 3),
                          ("Rookie C", "TE", 50.0, 60.0, 10), ("Vet D", "WR", 200.0, 190.0, 14),
                          ("Rookie E", "QB", 300.0, 280.0, 16)])
    rows, counts = impute_cv.rookie_rows(board, stats, realised, season=2024, min_games=8)
    assert rows["player"].to_list() == ["rookie a"]
    assert rows["ppg"][0] == pytest.approx(15.0) and rows["xfp_pg"][0] == pytest.approx(14.0)
    assert rows["imputed"][0] == 12.0 and rows["pos"][0] == "WR"
    # Rookie F has no weekly line in any season, so the proxy cannot classify him: he is
    # neither a rookie nor a veteran here, and he is counted so the omission is visible.
    assert counts == {"top200": 5, "rookies": 3, "imputed": 2, "with_line": 2, "played": 1,
                      "collisions": 1, "no_line": 1}


def test_the_residual_cv_is_the_spread_of_realised_over_imputed_by_position_and_pooled():
    """`sd(realised / imputed - 1)`, the statistic the shipped constant is, so the two are
    comparable; a position with one row has no spread and says so rather than 0."""
    rows = pl.DataFrame({
        "pos": ["WR", "WR", "WR", "RB", "RB", "TE"],
        "imputed": [10.0, 10.0, 10.0, 8.0, 8.0, 5.0],
        "ppg": [12.0, 8.0, 10.0, 12.0, 4.0, 5.0],
    })
    got = impute_cv.residual_cv(rows, "ppg")
    ratios_wr = np.array([0.2, -0.2, 0.0])
    assert got["WR"]["cv"] == pytest.approx(ratios_wr.std(ddof=1))
    assert got["WR"]["n"] == 3
    assert got["RB"]["cv"] == pytest.approx(np.array([0.5, -0.5]).std(ddof=1))
    assert got["TE"]["cv"] is None and got["TE"]["n"] == 1
    pooled = np.array([0.2, -0.2, 0.0, 0.5, -0.5, 0.0])
    assert got["pooled"]["cv"] == pytest.approx(pooled.std(ddof=1))
    assert got["pooled"]["n"] == 6
    assert got["pooled"]["median"] == pytest.approx(0.0)


def test_the_interval_is_clustered_on_the_season_with_a_t_reference():
    """The season is the cluster: the pooled CV is measured once per season, the interval
    is the mean of those under a t on k-1 degrees of freedom, and the distance from the
    shipped value is read on that unit. A season with one rookie has no spread and is
    dropped from the clusters rather than read as zero."""
    from hub.models.experiment import t_quantile
    rows = pl.DataFrame({
        "season": [2023] * 3 + [2024] * 3 + [2025] * 3 + [2021],
        "pos": ["WR"] * 10, "imputed": [10.0] * 10,
        "ppg": [12.0, 8.0, 10.0, 15.0, 5.0, 10.0, 11.0, 9.0, 10.0, 30.0],
    })
    got = impute_cv.season_clustered(rows, "ppg", shipped=0.260)
    assert got is not None
    per = {2023: np.std([0.2, -0.2, 0.0], ddof=1), 2024: np.std([0.5, -0.5, 0.0], ddof=1),
           2025: np.std([0.1, -0.1, 0.0], ddof=1)}
    assert got["k"] == 3 and set(got["per_season"]) == {2023, 2024, 2025}
    for s, cv in per.items():
        assert got["per_season"][s]["cv"] == pytest.approx(cv) and got["per_season"][s]["n"] == 3
    vals = np.array(list(per.values()))
    mean, se = vals.mean(), vals.std(ddof=1) / np.sqrt(3)
    assert got["mean"] == pytest.approx(mean) and got["se"] == pytest.approx(se)
    half = t_quantile(0.975, 2) * se
    assert got["lo"] == pytest.approx(mean - half) and got["hi"] == pytest.approx(mean + half)
    assert got["t_vs_shipped"] == pytest.approx((mean - 0.260) / se)
    assert got["clears"] == (abs(got["t_vs_shipped"]) >= t_quantile(0.975, 2))
    assert impute_cv.season_clustered(rows.head(3), "ppg", shipped=0.260) is None  # one cluster


def test_a_held_out_season_leaves_the_measurement_and_is_named():
    """`--exclude-season` (#294): the rows of that season are not in the fit, and the sentence
    says which season was held out and which remain."""
    rows = pl.DataFrame({"season": [2023, 2023, 2024, 2024, 2025, 2025],
                         "pos": ["WR"] * 6, "imputed": [10.0] * 6,
                         "ppg": [12.0, 8.0, 15.0, 5.0, 11.0, 9.0]})
    kept, said = impute_cv.hold_out(rows, 2024)
    assert kept["season"].unique().sort().to_list() == [2023, 2025]
    assert "2024" in said and "held out" in said
    same, said_none = impute_cv.hold_out(rows, None)
    assert same.height == 6 and "nothing held out" in said_none


def test_the_script_records_the_constant_as_not_refitted_with_the_rookie_number_in_the_reason(
        monkeypatch, tmp_path, capsys):
    """`--exclude-season N --record` (#294) writes IMPUTE_CV and IMPUTE_CV_BY_POS into the
    season's set as *not refitted*. Since #298 the shipped constant *is* this script's
    measurement on all five seasons, so the reason says that, carries the hold-out's own
    pooled value so the replay's run line shows what it did not use, and no longer calls
    the adoption an open decision or cites the veteran 0.260 as the shipped number."""
    from hub import holdout

    monkeypatch.setattr(holdout, "SETS", tmp_path)
    result = {"n": 5, "seasons": [2021, 2022], "clusters": 1,
              "xfp_pg": {"by_position": {"pooled": {"cv": 0.3125, "n": 5, "median": 0.0}}}}
    monkeypatch.setattr(impute_cv, "measure", lambda *a, **k: (result, ["  a line"]))
    assert impute_cv.main(["--exclude-season", "2022", "--record"]) == 0
    got = holdout.load(2022)
    assert got.values == {}
    assert set(got.missing) == {"predict.IMPUTE_CV", "predict.IMPUTE_CV_BY_POS"}
    why = got.missing["predict.IMPUTE_CV"]
    assert "pooled 0.312" in why and "#298" in why
    assert "0.260" not in why and "open decision" not in why
    assert "0.315" in why, "the reason names the shipped value the hold-out is read against"
    out = capsys.readouterr().out
    assert "unchanged by this run" not in out, "the pre-#298 closing line: no longer true"
    assert "#298" in out and "0.315" in out


def test_the_measurement_reports_n_the_clusters_and_the_shipped_value_beside_each_rookie_cv():
    """The whole run on two synthetic boards through the three seams: every line a script
    prints carries n, the season count as the clusters, both realised quantities, and the
    shipped number beside the rookie one -- and a held-out season is named on the run line."""
    boards = {
        2024: _board([("R1", "WR", 10, 10.0, True), ("R2", "WR", 20, 10.0, True),
                      ("R3", "RB", 30, 8.0, True), ("V1", "RB", 5, 20.0, False)]),
        2025: _board([("R4", "WR", 10, 10.0, True), ("R5", "RB", 20, 8.0, True),
                      ("R6", "RB", 40, 8.0, True)]),
    }
    stats = _stats([("R1", 2024), ("R2", 2024), ("R3", 2024), ("V1", 2020), ("V1", 2024),
                    ("R4", 2025), ("R5", 2025), ("R6", 2025)])
    realised = {
        2024: _realised([("R1", "WR", 120.0, 110.0, 10), ("R2", "WR", 80.0, 90.0, 10),
                         ("R3", "RB", 96.0, 80.0, 8), ("V1", "RB", 300.0, 280.0, 15)]),
        2025: _realised([("R4", "WR", 100.0, 100.0, 10), ("R5", "RB", 40.0, 48.0, 8),
                         ("R6", "RB", 120.0, 96.0, 12)]),
    }
    result, lines = impute_cv.measure(
        [2024, 2025], build_board=lambda yr: boards[yr], load_stats=lambda: stats,
        load_realised=lambda yr: realised[yr])
    text = "\n".join(lines)
    assert result["n"] == 6 and result["clusters"] == 2
    assert "n = 6 rookies over 2 seasons (the clusters)" in text
    assert "against realised PPR points per game" in text and "own xFP per game" in text
    assert "nothing held out" in text
    ratios = np.array([0.2, -0.2, 0.5, 0.0, -0.375, 0.25])
    assert result["ppg"]["by_position"]["pooled"]["cv"] == pytest.approx(ratios.std(ddof=1))
    assert result["ppg"]["se"] is not None and 0 < result["ppg"]["se"] < 0.2
    from hub.models.predict import IMPUTE_CV
    assert f"{IMPUTE_CV:.3f}" in text     # the shipped value, printed beside the rookie one

    held, lines_held = impute_cv.measure(
        [2024, 2025], exclude=2025, build_board=lambda yr: boards[yr],
        load_stats=lambda: stats, load_realised=lambda yr: realised[yr])
    assert held["n"] == 3 and held["clusters"] == 1
    assert "season 2025 held out; fitted on [2024]" in "\n".join(lines_held)
