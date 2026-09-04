"""The ratings passthrough.

`Makefile:12` calls `hub.models.ratings --fit`. Until now that line dead-ended, so
`make slate` had never run end to end and no part of the weekly pipeline had been
exercised together.

The tests that matter are about honesty rather than accuracy. A passthrough must not be
mistakable for a model: it writes under the market's own name and version so the track
record cannot later credit it with an edge it never had. And it must pick a week it is
allowed to predict, because leakage looks like success.
"""
import datetime as dt

import polars as pl
import pytest

from hub import schedule, store
from hub.fetch import nflverse
from hub.models import ratings


def _sched(rows):
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([r[1] for r in rows], dtype=pl.Int32),
         "home_team": [r[4] if len(r) > 4 else "KC" for r in rows],
         "away_team": [r[5] if len(r) > 5 else "LV" for r in rows],
         "gameday": [r[6] if len(r) > 6 else "2026-12-25" for r in rows],
         "gametime": [r[7] if len(r) > 7 else "13:00" for r in rows],
         "spread_line": [r[2] for r in rows],
         "result": [r[3] for r in rows]})


@pytest.fixture
def sched(monkeypatch):
    def _install(rows):
        monkeypatch.setattr(nflverse, "load",
                            lambda source, seasons, cache=None: _sched(rows))
    return _install


# --- picking a week it is allowed to predict ------------------------------

def test_targets_the_first_unplayed_week(sched, tmp_path):
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None), ("c", 3, 2.0, None)])
    assert ratings.target_week(schedule.priced_games(2026, base=tmp_path)) == 2


def test_a_finished_season_falls_back_to_the_last_week(sched, tmp_path):
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, 3)])
    assert ratings.target_week(schedule.priced_games(2026, base=tmp_path)) == 2


def test_a_week_with_no_line_is_not_a_target(sched, tmp_path):
    """An unplayed week nobody has priced cannot be predicted from the market."""
    sched([("a", 1, None, None), ("b", 2, -1.0, None)])
    assert ratings.target_week(schedule.priced_games(2026, base=tmp_path)) == 2


# --- what it writes -------------------------------------------------------

def test_it_writes_versioned_predictions(sched, tmp_path):
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    got = store.sql("SELECT * FROM preds", base=tmp_path / "store")
    assert got.height == 1
    assert got["version"][0].startswith("market-")


def test_it_does_not_claim_to_be_a_model(sched, tmp_path):
    """A passthrough credited as a model would poison the track record permanently."""
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got["model"][0] == "market_baseline"


def test_it_never_predicts_a_week_it_was_fit_through(sched, tmp_path):
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert (got["fit_through_week"] < got["week"]).all()


def test_an_explicit_week_is_honoured(sched, tmp_path):
    sched([("a", 1, 3.0, None), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, week=2, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got["week"].to_list() == [2]


def test_a_slate_with_no_lines_writes_nothing_rather_than_guessing(sched, tmp_path):
    sched([("a", 1, None, None)])
    got = ratings.fit(2026, week=1, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got.height == 0


def test_it_prints_a_summary_not_rows(sched, tmp_path, capsys):
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert len(capsys.readouterr().out.splitlines()) <= 8


def test_main_without_fit_prints_help(capsys):
    assert ratings.main([]) == 0
    assert "fit" in capsys.readouterr().out


# The pricing rule itself -- which source priced a game, and the fallback -- is
# `hub.schedule`'s and is tested in test_schedule.py, beside the module that owns it. What
# is asserted here is what the fit does with it.

def _snap(base, season, week, rows, at):
    store.write(
        pl.DataFrame({"game_id": [r[0] for r in rows],
                      "close_spread": [r[1] for r in rows],
                      "captured_at": [at] * len(rows)},
                     schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                             "captured_at": pl.Datetime}),
        "lines", "nfl", season, week, base=base, name=f"snap-{at:%Y%m%dT%H%M%S}")


# --- provenance travels onto the prediction --------------------------------

def test_the_version_changes_with_the_source(sched, tmp_path):
    """A prediction priced from a dated snapshot is not the same artifact as one priced
    from a moving field, even when the two numbers agree."""
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    got = ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path,
                      cache=tmp_path / "cache")
    by = dict(zip(got["game_id"].to_list(), got["version"].to_list(), strict=True))
    assert by["b"].endswith("-snapshot") and by["c"].endswith("-schedule")
    assert by["b"] != by["c"], "identical numbers, different artifacts"


def test_the_two_sources_are_written_as_separate_partitions(sched, tmp_path):
    """One file per provenance, so a partition is homogeneous in what priced it and a
    re-run under one source cannot overwrite the record of the other."""
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path, cache=tmp_path / "cache")
    written = sorted(p.name for p in (tmp_path / "preds").rglob("*.parquet"))
    assert len(written) == 2, written
    assert any("snapshot" in n for n in written) and any("schedule" in n for n in written)


def test_the_fit_reports_how_many_games_each_source_priced(sched, tmp_path, capsys):
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None),
              ("d", 2, None, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path, cache=tmp_path / "cache")
    out = capsys.readouterr().out
    assert "1 from a dated snapshot" in out
    assert "1 from the moving field" in out
    assert "1 unpriced" in out


def test_a_store_with_no_snapshots_at_all_still_fits(sched, tmp_path):
    """Graceful degradation: a fresh clone has no `lines` view, and the fit must serve the
    moving field rather than raising."""
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got.height == 1
    assert got["price_source"].to_list() == ["schedule"]


# --- a prediction is not written for a game that has already started ------------
#
# `docs/track-record.md` rule 1: a prediction counts only if it was committed before kickoff.
# A human running the fit before the week starts satisfies that by habit. A schedule does not
# -- a Sunday-morning run would publish a "prediction" for Thursday night's finished game,
# which is the one thing the record cannot survive. So the rule stops being a habit.

def _week_in_flight(rows):
    """A week mid-flight: Thursday kicked off, Sunday has not."""
    return _sched(rows)


def test_a_game_that_has_kicked_off_is_not_predicted(sched, tmp_path):
    sched([("thu", 2, 3.0, None, "KC", "LV", "2026-09-10", "20:15"),
           ("sun", 2, -1.0, None, "SF", "SEA", "2026-09-13", "13:00")])
    got = ratings.fit(2026, 2, at=dt.datetime(2026, 9, 11, 12, 0),
                      base=tmp_path, cache=tmp_path / "c")
    assert got["game_id"].to_list() == ["sun"]


def test_a_finished_game_is_not_predicted_even_with_no_kickoff_to_read(sched, tmp_path):
    """Belt and braces: a schedule with no times still must not price a played game."""
    sched([("done", 2, 3.0, 7, "KC", "LV", None, None),
           ("todo", 2, -1.0, None, "SF", "SEA", None, None)])
    got = ratings.fit(2026, 2, at=dt.datetime(2026, 9, 11), base=tmp_path,
                      cache=tmp_path / "c")
    assert got["game_id"].to_list() == ["todo"]


def test_the_run_says_how_many_games_it_withheld(sched, tmp_path, capsys):
    """Silently predicting fewer games than the slate holds is how a reader concludes the
    week was light rather than that the fit ran late."""
    sched([("thu", 2, 3.0, None, "KC", "LV", "2026-09-10", "20:15"),
           ("sun", 2, -1.0, None, "SF", "SEA", "2026-09-13", "13:00")])
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 11, 12, 0), base=tmp_path,
                cache=tmp_path / "c")
    assert "1 already under way" in capsys.readouterr().out


def test_a_week_entirely_under_way_is_not_the_target(sched, tmp_path):
    """The week to predict is the first one still forecastable, not the first one unfinished.
    A run that fired late must move to next week rather than predict a slate in progress."""
    sched([("a", 1, 3.0, None, "KC", "LV", "2026-09-10", "20:15"),
           ("b", 2, -1.0, None, "SF", "SEA", "2026-09-17", "20:15")])
    # 20:15 ET on the 10th is 00:15 UTC on the 11th -- asking at midnight UTC is *before*
    # kickoff, not after. I wrote this test the other way round first.
    games = schedule.priced_games(2026, at=dt.datetime(2026, 9, 11, 12), base=tmp_path)
    assert ratings.target_week(games, at=dt.datetime(2026, 9, 11, 12)) == 2


def test_a_slate_that_has_entirely_started_writes_nothing_rather_than_backdating(sched,
                                                                                 tmp_path):
    sched([("a", 1, 3.0, None, "KC", "LV", "2026-09-10", "20:15")])
    got = ratings.fit(2026, 1, at=dt.datetime(2026, 9, 11, 12), base=tmp_path,
                      cache=tmp_path / "c")
    assert got.height == 0


def test_a_later_run_does_not_delete_a_prediction_it_can_no_longer_make(sched, tmp_path):
    """The defect a schedule creates, and the direction that makes it serious.

    A second run in the same week re-fits the games that have not started and writes the same
    partition. Without this, the games that *have* started simply vanish -- so the record
    quietly loses exactly the predictions reality has already tested, which is the direction a
    dishonest record would trim in. Rule 1 says the commit is the timestamp; a later commit
    must not un-say an earlier one.
    """
    rows = [("thu", 1, 3.0, None, "KC", "LV", "2026-09-10", "20:15"),
            ("sun", 1, -1.0, None, "SF", "SEA", "2026-09-13", "13:00")]
    sched(rows)
    ratings.fit(2026, 1, at=dt.datetime(2026, 9, 9), base=tmp_path, cache=tmp_path / "c")
    ratings.fit(2026, 1, at=dt.datetime(2026, 9, 12), base=tmp_path, cache=tmp_path / "c")
    got = store.predictions(season=2026, week=1, base=tmp_path)
    assert sorted(got["game_id"].to_list()) == ["sun", "thu"]


def test_the_later_run_still_refreshes_what_it_may(sched, tmp_path):
    """Keeping the committed row must not freeze the rest of the slate: a game that has not
    kicked off can legitimately be re-priced, and the commit still predates its kickoff."""
    sched([("thu", 1, 3.0, None, "KC", "LV", "2026-09-10", "20:15"),
           ("sun", 1, -1.0, None, "SF", "SEA", "2026-09-13", "13:00")])
    ratings.fit(2026, 1, at=dt.datetime(2026, 9, 9), base=tmp_path, cache=tmp_path / "c")
    first = store.predictions(season=2026, week=1, base=tmp_path)
    was = dict(zip(first["game_id"].to_list(), first["predicted_at"].to_list(), strict=True))
    ratings.fit(2026, 1, at=dt.datetime(2026, 9, 12), base=tmp_path, cache=tmp_path / "c")
    now = store.predictions(season=2026, week=1, base=tmp_path)
    is_ = dict(zip(now["game_id"].to_list(), now["predicted_at"].to_list(), strict=True))
    assert is_["thu"] == was["thu"], "a started game keeps the prediction it was committed with"
    assert is_["sun"] >= was["sun"], "an unstarted game is re-priced"
