"""The unwired state-space rating (#365): what it recovers, what it must not gain, and that it
cannot see ahead.

The controls are planted leagues, because the real archive cannot say whether the filter works
-- only a league whose strengths are known can. Rule 18: each assertion is one a wrong filter
would fail. A filter that ignored the walk would tie the passthrough on the drifting league; a
filter that over-fitted would beat a perfect close on the static one; a filter that read the
season it scores would move an early fold when a later season is flipped.
"""
import numpy as np
import polars as pl
import pytest

from hub.exhibits import state_space_rating as S

NOISE = 8.0
HOME = 2.0


def league(*, drift_sd: float, seasons: int = 9, weeks: int = 16, teams: int = 20,
           seed: int = 0, perfect_close: bool = False) -> pl.DataFrame:
    """A planted league. Strengths take a random walk of `drift_sd` per week and carry over
    between seasons. The close knows each team's strength at the *start* of the season, which
    is all a stale price knows; `perfect_close` hands it the true expected margin instead."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 4.0, teams)
    rows = []
    for season in range(2000, 2000 + seasons):
        start = theta.copy()
        for week in range(1, weeks + 1):
            theta = theta + rng.normal(0.0, drift_sd, teams)
            order = rng.permutation(teams)
            for i in range(0, teams, 2):
                h, a = int(order[i]), int(order[i + 1])
                mean = theta[h] - theta[a] + HOME
                stale = start[h] - start[a] + HOME
                rows.append({
                    "game_id": f"{season}_{week:02d}_{h}_{a}", "season": season, "week": week,
                    "game_type": "REG", "home_team": f"T{h}", "away_team": f"T{a}",
                    "result": float(round(mean + rng.normal(0.0, NOISE))) or 1.0,
                    "spread_line": float(mean if perfect_close else stale)})
    return S.prepare(pl.DataFrame(rows))


@pytest.fixture(scope="module")
def drifting():
    g = league(drift_sd=2.0)
    return g, *S.walk_forward(g, arm_a_sd=NOISE)


@pytest.fixture(scope="module")
def static():
    g = league(drift_sd=0.0, perfect_close=True, seed=1)
    return g, *S.walk_forward(g, arm_a_sd=NOISE)


def test_a_league_with_a_known_drift_is_recovered_and_beats_the_passthrough(drifting):
    """The close here is the preseason truth, stale by exactly the drift. A filter that tracks
    the walk gains on it in the seasons it has had time to learn from, and the fitted step is
    the order of the planted one rather than the lower bound it would sit at if it saw none."""
    _, _, table = drifting
    assert table["gap"].mean() > 0.02
    assert (table["gap"] > 0).sum() >= table.height - 1
    assert abs(table["week_step_sd"].tail(4).mean() - 2.0) < 0.3, "the planted step is 2.0"
    assert table["season_shrink"].tail(4).mean() < 0.1, "the planted league carries over whole"


def test_a_static_league_with_a_perfect_close_gains_nothing(static):
    """True strengths never move and the close is the true expected margin. Nothing a rating
    fitted to results can add: the gap must not be positive beyond the noise of estimating
    the strengths, and the fitted walk must be near its floor."""
    _, _, table = static
    assert table["gap"].max() < 0.01
    assert table["gap"].mean() < 0.0
    assert table["week_step_sd"].tail(4).mean() < 0.1, "no walk was planted"


def test_no_leak_flipping_a_later_season_moves_no_earlier_fold():
    g = league(drift_sd=1.0, seasons=5)
    _, base = S.walk_forward(g, arm_a_sd=NOISE)
    flipped = g.with_columns(
        pl.when(pl.col("season") == 2004).then(-pl.col("result")).otherwise(pl.col("result"))
        .alias("result"))
    _, other = S.walk_forward(flipped, arm_a_sd=NOISE)
    early = lambda t: t.filter(pl.col("season") < 2004)  # noqa: E731
    assert early(base).equals(early(other))
    assert not base.filter(pl.col("season") == 2004).equals(
        other.filter(pl.col("season") == 2004)), "the control must be able to see the flip"


def test_a_prediction_never_sees_its_own_result():
    """Within a season: changing a game's result changes no prediction made in that week or
    before, and does change a later one."""
    g = league(drift_sd=1.0, seasons=2)
    p = S.start_params()
    base, _ = S.run_filter(g, p)
    target = g.filter((pl.col("season") == 2001) & (pl.col("week") == 4)).row(0, named=True)
    idx = g["game_id"].to_list().index(target["game_id"])
    bumped = g.with_columns(pl.when(pl.col("game_id") == target["game_id"])
                            .then(pl.col("result") + 30.0).otherwise(pl.col("result"))
                            .alias("result"))
    moved, _ = S.run_filter(bumped, p)
    wk = g["week"].to_numpy()
    sn = g["season"].to_numpy()
    same_or_before = (sn < 2001) | ((sn == 2001) & (wk <= 4))
    assert np.array_equal(base[same_or_before], moved[same_or_before])
    assert not np.array_equal(base[idx + 1:], moved[idx + 1:])


def test_the_rating_reads_results_not_the_close(drifting):
    """Rule 6: rewriting every spread leaves the rating arm's log-loss exactly where it was."""
    g, _, table = drifting
    rewritten = g.with_columns((pl.col("spread_line") * 0.0 + 7.0).alias("spread_line"))
    _, other = S.walk_forward(rewritten, arm_a_sd=NOISE)
    assert table["ll_rating"].to_list() == other["ll_rating"].to_list()
    assert table["ll_close"].to_list() != other["ll_close"].to_list()


def test_ties_and_unpriced_games_feed_the_filter_but_are_never_scored():
    g = league(drift_sd=1.0, seasons=3)
    tie = g.row(0, named=True)["game_id"]
    unpriced = g.row(1, named=True)["game_id"]
    frame = g.with_columns(
        pl.when(pl.col("game_id") == tie).then(0.0).otherwise(pl.col("result")).alias("result"),
        pl.when(pl.col("game_id") == unpriced).then(None).otherwise(pl.col("spread_line"))
        .alias("spread_line"))
    out = S.prepare(frame.with_columns(pl.lit("REG").alias("game_type")))
    scored = out.filter(pl.col("scored"))["game_id"].to_list()
    assert tie not in scored and unpriced not in scored
    assert out.height == g.height, "both stay in the filter"
    per_game, _ = S.walk_forward(out, arm_a_sd=NOISE)
    assert tie not in per_game["game_id"].to_list()


def test_postseason_and_unplayed_games_are_not_ingested():
    g = league(drift_sd=1.0, seasons=2)
    raw = g.with_columns(pl.lit("REG").alias("game_type"))
    extra = raw.head(2).with_columns(pl.Series("game_type", ["POST", "REG"]),
                                     pl.Series("result", [5.0, None]))
    out = S.prepare(pl.concat([raw, extra]))
    assert out.height == raw.height


def test_a_franchise_that_moved_is_one_team():
    g = pl.DataFrame({"game_id": ["a", "b"], "season": [2000, 2020], "week": [1, 1],
                      "game_type": ["REG", "REG"], "home_team": ["OAK", "LV"],
                      "away_team": ["SD", "LAC"], "result": [3.0, 3.0],
                      "spread_line": [1.0, 1.0]})
    out = S.prepare(g)
    assert set(out["home_team"]) == {"LV"} and set(out["away_team"]) == {"LAC"}


def test_the_table_carries_no_verdict_word(drifting):
    _, _, table = drifting
    text = "\n".join(S.report(table)).lower()
    for word in ("adopt", "remove", "show", "verdict", "wins", "loses", "beats", "better"):
        assert word not in text


def test_an_empty_table_reports_without_a_verdict():
    assert S.report(pl.DataFrame()) == ["  no held-out season"]


def test_the_gate_with_no_ceiling_is_not_runnable_and_names_the_archive(drifting, tmp_path):
    from hub.models.experiment import Ledger

    _, per_game, _ = drifting
    run = S.HARNESS._replace(ledger=Ledger(path=None), bootstrap=100).run(per_game)
    assert run.verdict[0] == "NOT-RUNNABLE"
    (tmp_path / "2026").mkdir()
    line = S.not_runnable_line(S.archive_seasons(tmp_path))
    assert "NOT-RUNNABLE" in line and "#383" in line and "1 season(s)" in line
    assert S.archive_seasons(tmp_path / "missing") == 0


def test_the_constants_are_all_declared_not_an_input():
    from hub import declare

    mine = [d for d in declare.declarations() if d.module == S.__name__]
    assert mine and all(not d.covered for d in mine)


# --- the entry point -----------------------------------------------------------------

def _raw(g: pl.DataFrame) -> pl.DataFrame:
    return g.drop("scored").with_columns(pl.lit("REG").alias("game_type"))


def test_main_without_a_flag_prints_help_and_exits_zero(capsys):
    assert S.main([]) == 0
    assert "--fit" in capsys.readouterr().out


def test_main_fit_prints_the_table_and_no_verdict(monkeypatch, capsys):
    from hub.fetch import nflverse

    g = league(drift_sd=1.0, seasons=3)
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: _raw(g))
    assert S.main(["--fit"]) == 0       # no season here is complete: through falls back to all
    out = capsys.readouterr().out
    assert "mean gap over 2 seasons" in out and "NOT-RUNNABLE" not in out


def test_main_through_stops_the_walk(monkeypatch, capsys):
    from hub.fetch import nflverse

    g = league(drift_sd=1.0, seasons=4)
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: _raw(g))
    assert S.main(["--fit", "--through", "2001"]) == 0
    assert "mean gap over 1 seasons" in capsys.readouterr().out


def test_main_gate_prints_not_runnable_and_names_the_archive(monkeypatch, capsys):
    from hub.fetch import nflverse
    from hub.models.experiment import Ledger

    g = league(drift_sd=1.0, seasons=3)
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: _raw(g))
    monkeypatch.setattr(S, "HARNESS", S.HARNESS._replace(ledger=Ledger(path=None), bootstrap=50))
    assert S.main(["--gate", "--through", "2002"]) == 0
    out = capsys.readouterr().out
    assert "NOT-RUNNABLE" in out and "#383" in out


def test_main_names_a_schedule_it_cannot_read(monkeypatch, capsys):
    from hub.fetch import nflverse

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(nflverse, "load", boom)
    assert S.main(["--fit"]) != 0
    assert "nflverse schedules" in capsys.readouterr().err


def test_prepare_refuses_a_frame_missing_a_column():
    with pytest.raises(ValueError, match="missing"):
        S.prepare(pl.DataFrame({"season": [2000]}))


def test_a_single_season_has_nothing_to_hold_out():
    g = league(drift_sd=1.0, seasons=1)
    per_game, table = S.walk_forward(g)
    assert per_game.is_empty() and table.is_empty()


def test_the_close_sigma_falls_back_when_nothing_was_priced():
    g = league(drift_sd=1.0, seasons=1).with_columns(pl.lit(False).alias("scored"))
    assert S.trailing_sd(g) == S.MARGIN_SD


def test_a_fit_on_no_games_returns_the_start():
    assert S.fit_constants(league(drift_sd=1.0, seasons=1).head(0)) == S.start_params()
