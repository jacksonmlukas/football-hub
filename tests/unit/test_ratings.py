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

def test_targets_the_first_unplayed_week(schedule, tmp_path):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None), ("c", 3, 2.0, None)])
    assert ratings.target_week(ratings.games_for(2026, base=tmp_path)) == 2


def test_a_finished_season_falls_back_to_the_last_week(schedule, tmp_path):
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, 3)])
    assert ratings.target_week(ratings.games_for(2026, base=tmp_path)) == 2


def test_a_week_with_no_line_is_not_a_target(schedule, tmp_path):
    """An unplayed week nobody has priced cannot be predicted from the market."""
    schedule([("a", 1, None, None), ("b", 2, -1.0, None)])
    assert ratings.target_week(ratings.games_for(2026, base=tmp_path)) == 2


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


# --- which source priced it ------------------------------------------------
#
# A prediction is worth what it can be re-derived from. `spread_line` moves as the week
# runs and is empty for far weeks, so a number taken from it cannot be shown afterwards,
# only asserted. Where a dated snapshot exists it prices the game; where none does the
# moving field still prices it, and the row says which.

def _snap(base, season, week, rows, at):
    store.write(
        pl.DataFrame({"game_id": [r[0] for r in rows],
                      "close_spread": [r[1] for r in rows],
                      "captured_at": [at] * len(rows)},
                     schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                             "captured_at": pl.Datetime}),
        "lines", "nfl", season, week, base=base, name=f"snap-{at:%Y%m%dT%H%M%S}")


def test_a_dated_snapshot_prices_the_game_rather_than_the_moving_field(schedule, tmp_path):
    schedule([("a", 1, 3.0, None)])
    _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, 5))
    got = ratings.games_for(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["close_spread"].to_list() == [6.5]
    assert got["price_source"].to_list() == ["snapshot"]


def test_a_game_with_no_snapshot_still_prices_from_the_moving_field(schedule, tmp_path):
    """Week 18 will never carry a snapshot when the fit first runs. It must not drop out
    of the slate for that."""
    schedule([("a", 18, 3.0, None)])
    got = ratings.games_for(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["close_spread"].to_list() == [3.0]
    assert got["price_source"].to_list() == ["schedule"]


def test_a_snapshot_taken_after_the_moment_does_not_price_the_game(schedule, tmp_path):
    schedule([("a", 1, 3.0, None)])
    _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, 10))
    got = ratings.games_for(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["price_source"].to_list() == ["schedule"]


def test_a_game_nobody_has_priced_carries_no_source(schedule, tmp_path):
    schedule([("a", 18, None, None)])
    got = ratings.games_for(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["price_source"].to_list() == [None]


def test_the_snapshot_that_priced_a_row_is_named_by_its_capture_time(schedule, tmp_path):
    """"Which source" is only half of provenance -- there are many snapshots and the row
    has to say which one, or re-deriving it means guessing."""
    schedule([("a", 1, 3.0, None)])
    _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, 5, 11))
    got = ratings.games_for(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["priced_at"].to_list() == [dt.datetime(2026, 9, 5, 11)]


# --- provenance travels onto the prediction --------------------------------

def test_the_version_changes_with_the_source(schedule, tmp_path):
    """A prediction priced from a dated snapshot is not the same artifact as one priced
    from a moving field, even when the two numbers agree."""
    schedule([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    got = ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path,
                      cache=tmp_path / "cache")
    by = dict(zip(got["game_id"].to_list(), got["version"].to_list(), strict=True))
    assert by["b"].endswith("-snapshot") and by["c"].endswith("-schedule")
    assert by["b"] != by["c"], "identical numbers, different artifacts"


def test_the_two_sources_are_written_as_separate_partitions(schedule, tmp_path):
    """One file per provenance, so a partition is homogeneous in what priced it and a
    re-run under one source cannot overwrite the record of the other."""
    schedule([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path, cache=tmp_path / "cache")
    written = sorted(p.name for p in (tmp_path / "preds").rglob("*.parquet"))
    assert len(written) == 2, written
    assert any("snapshot" in n for n in written) and any("schedule" in n for n in written)


def test_the_fit_reports_how_many_games_each_source_priced(schedule, tmp_path, capsys):
    schedule([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None),
              ("d", 2, None, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path, cache=tmp_path / "cache")
    out = capsys.readouterr().out
    assert "1 from a dated snapshot" in out
    assert "1 from the moving field" in out
    assert "1 unpriced" in out


def test_a_store_with_no_snapshots_at_all_still_fits(schedule, tmp_path):
    """Graceful degradation: a fresh clone has no `lines` view, and the fit must serve the
    moving field rather than raising."""
    schedule([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    got = ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert got.height == 1
    assert got["price_source"].to_list() == ["schedule"]
