"""The ratings passthrough.

`Makefile:12` calls `hub.models.ratings --fit`. Until now that line dead-ended, so
`make slate` had never run end to end and no part of the weekly pipeline had been
exercised together.

The tests that matter are about honesty rather than accuracy. A passthrough must not be
mistakable for a model: it writes under the market's own name and version so the track
record cannot later credit it with an edge it never had. And it must pick a week it is
allowed to predict, because leakage looks like success.
"""
import polars as pl
import pytest

from hub import store
from hub.models import ratings


def _sched(rows):
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([r[1] for r in rows], dtype=pl.Int32),
         "spread_line": [r[2] for r in rows],
         "result": [r[3] for r in rows]})


@pytest.fixture
def schedule(monkeypatch):
    def _install(rows):
        monkeypatch.setattr(ratings.nflverse, "load",
                            lambda source, seasons, cache=None: _sched(rows))
    return _install


# --- picking a week it is allowed to predict ------------------------------

def test_targets_the_first_unplayed_week(schedule):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None), ("c", 3, 2.0, None)])
    assert ratings.target_week(ratings.games_for(2026)) == 2


def test_a_finished_season_falls_back_to_the_last_week(schedule):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, 3)])
    assert ratings.target_week(ratings.games_for(2026)) == 2


def test_a_week_with_no_line_is_not_a_target(schedule):
    """An unplayed week nobody has priced cannot be predicted from the market."""
    schedule([("a", 1, None, None), ("b", 2, -1.0, None)])
    assert ratings.target_week(ratings.games_for(2026)) == 2


# --- what it writes -------------------------------------------------------

def test_it_writes_versioned_predictions(schedule, tmp_path):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    got = store.sql("SELECT * FROM preds", base=tmp_path / "store")
    assert got.height == 1
    assert got["version"][0].startswith("market-")


def test_it_does_not_claim_to_be_a_model(schedule, tmp_path):
    """A passthrough credited as a model would poison the track record permanently."""
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got["model"][0] == "market_baseline"


def test_it_never_predicts_a_week_it_was_fit_through(schedule, tmp_path):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert (got["fit_through_week"] < got["week"]).all()


def test_an_explicit_week_is_honoured(schedule, tmp_path):
    schedule([("a", 1, 3.0, None), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, week=2, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got["week"].to_list() == [2]


def test_a_slate_with_no_lines_writes_nothing_rather_than_guessing(schedule, tmp_path):
    schedule([("a", 1, None, None)])
    got = ratings.fit(2026, week=1, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got.height == 0


def test_it_prints_a_summary_not_rows(schedule, tmp_path, capsys):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert len(capsys.readouterr().out.splitlines()) <= 8


def test_main_without_fit_prints_help(capsys):
    assert ratings.main([]) == 0
    assert "fit" in capsys.readouterr().out
