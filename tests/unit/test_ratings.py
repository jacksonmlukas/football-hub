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
from hub.models.base import Forecaster


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


# --- the interface, and the tripwire on it (issue #205) ---------------------
#
# `test_it_never_predicts_a_week_it_was_fit_through` above asserts the *outcome* the
# tripwire produces, on a fixture that could not leak anyway -- the shape that left eight
# guards green and dead in the week of 2026-09-04. These assert that the check is reached
# from this module, which is what `make slate` runs.

class _Leaks:
    """A conforming `Forecaster` that predicts the week it was fit through.

    Stamping `fit_through_week` from the week it is handed, rather than from the spec, is
    exactly how a real leak arrives: nothing crashes, every column is present, and the rows
    look better than they should.
    """
    name = "leaks"
    version = "leaks-1"

    def fit(self, spec):
        return self

    def predict(self, games):
        return games.select(["game_id", "league", "season", "week"]).with_columns([
            pl.lit(0.6).alias("home_win_prob"),
            pl.lit(1.0).alias("margin_mean"),
            pl.lit(0.0).alias("margin_lo"),
            pl.lit(2.0).alias("margin_hi"),
            pl.lit(self.name).alias("model"),
            pl.lit(self.version).alias("version"),
            pl.lit("schedule").alias("price_source"),
            pl.col("week").cast(pl.Int32).alias("fit_through_week"),
            pl.lit(dt.datetime(2026, 9, 1)).alias("predicted_at"),
        ])


def test_the_model_this_module_publishes_is_named_by_the_interface():
    """ADR-0002's protocol, and not a class this module happens to import."""
    assert isinstance(ratings.forecaster(), Forecaster)


def test_a_leaking_model_is_refused_by_the_writer_rather_than_published(sched, tmp_path,
                                                                       monkeypatch):
    """The tripwire, reached from the CLI `Makefile:12` runs.

    Substituted at `forecaster`, so what is exercised is the whole of `fit` -- week choice,
    spec, write path -- with only the model swapped. A passthrough cannot leak, which is
    why the old arrangement could lose the check without a test noticing.
    """
    sched([("a", 1, 3.0, 7), ("b", 2, -1.0, None)])
    monkeypatch.setattr(ratings, "forecaster", _Leaks)
    with pytest.raises(ValueError, match="LEAKAGE"):
        ratings.fit(2026, cache=tmp_path / "cache", base=tmp_path / "store")
    assert not (tmp_path / "store" / "preds").exists(), (
        "a refused fit must reach no partition; the record is the timestamp")


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
    assert by["b"].endswith("-live") and by["c"].endswith("-schedule")
    assert by["b"] != by["c"], "identical numbers, different artifacts"


def test_the_two_sources_are_written_as_separate_partitions(sched, tmp_path):
    """One file per provenance, so a partition is homogeneous in what priced it and a
    re-run under one source cannot overwrite the record of the other."""
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path, cache=tmp_path / "cache")
    written = sorted(p.name for p in (tmp_path / "preds").rglob("*.parquet"))
    assert len(written) == 2, written
    assert any("live" in n for n in written) and any("schedule" in n for n in written)


def test_the_fit_reports_how_many_games_each_source_priced(sched, tmp_path, capsys):
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None), ("c", 2, 3.0, None),
              ("d", 2, None, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 5))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 6), base=tmp_path, cache=tmp_path / "cache")
    out = capsys.readouterr().out
    assert "1 from a live snapshot" in out
    assert "0 from a stale one" in out
    assert "1 from the moving field" in out
    assert "1 unpriced" in out


def test_the_fit_reports_a_stale_snapshot_as_such(sched, tmp_path, capsys):
    """#281: a frozen quote with no moving field behind it prices the game and is counted
    as stale, not as a dated snapshot -- the wording that used to cover both."""
    sched([("a", 1, 3.0, 7), ("b", 2, None, None), ("c", 2, 3.0, None)])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 8, 20))
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 12), base=tmp_path, cache=tmp_path / "cache")
    out = capsys.readouterr().out
    assert "0 from a live snapshot" in out and "1 from a stale one" in out


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


# --- the number is the betting market's on every row (#299) --------------------------------
#
# From #218 to #299 the writer routed the slate through `hub.models.quarterback.apply`, and a
# row the staleness field marked as having no live price was moved from the nfeloqb state and
# said so. #299 pulled it: the source names a new starter only after his first game (#291),
# so the adjustment could only ever move a line that had already priced him. What is held
# here is the pull itself -- the row a stale poll priced is written as the betting market's
# number, under the baseline's own name, whatever nfeloqb file is cached -- and
# `tests/contracts/test_the_quarterback_adjustment_is_not_a_dependency.py` holds that the
# module is not on the writer's import path at all.

def _qb_state(cache):
    """The hand-built nfeloqb file, cached where the fit used to look for it. Its presence
    is the mutation the tests below are held against: a writer that read it would move `c`."""
    import json
    from pathlib import Path

    from hub.fetch import nfeloqb
    rows = json.loads((Path(__file__).resolve().parents[1] / "golden" / "fixtures"
                       / "nfeloqb_qb_elos.synthetic.json").read_text())
    cols = list(rows[0])
    text = "\n".join([",".join(cols)] + [",".join("" if r[c] is None else str(r[c]) for c in cols)
                                         for r in rows]) + "\n"
    (cache / "nfeloqb").mkdir(parents=True)
    (cache / "nfeloqb" / nfeloqb.FILE).write_text(text)


def test_a_frozen_quote_is_written_as_the_betting_markets_number_whatever_state_is_cached(
        sched, tmp_path):
    """Two games, one snapshot each. `b`'s quote was polled an hour before the fit and is
    live; `c`'s has stood untouched for three weeks and yields to the moving field, which
    carries the same number. LV's starter in the cached fixture is a fresh backup -- the
    row #218 would have moved. Since #299 neither row moves, neither carries an
    adjustment column, and both are the baseline's own name."""
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None, "KC", "LAC"), ("c", 2, 3.0, None, "KC", "LV")])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 12, 11))
    _snap(tmp_path, 2026, 2, [("c", 3.0)], dt.datetime(2026, 8, 20))
    _qb_state(tmp_path / "cache")
    got = ratings.fit(2026, 2, at=dt.datetime(2026, 9, 12, 12), base=tmp_path,
                      cache=tmp_path / "cache")
    assert "adjusted_by" not in got.columns and "qb_adjustment" not in got.columns
    by = {r["game_id"]: r for r in got.to_dicts()}
    assert by["b"]["margin_mean"] == 3.0 and by["c"]["margin_mean"] == 3.0
    assert by["b"]["version"].endswith("-live") and by["c"]["version"].endswith("-schedule")
    assert by["b"]["model"] == by["c"]["model"] == "market_baseline"
    assert not any("-qb" in v for v in got["version"].to_list())


def test_the_run_line_is_the_passthrough_and_counts_no_adjustment(sched, tmp_path, capsys):
    """From #284 to #299 the first line counted the priced games the quarterback layer
    moved and said `passthrough` only at zero; the count left with the layer, and no line
    of the run mentions the adjustment or the nfeloqb file."""
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None, "KC", "LAC"), ("c", 2, 3.0, None, "KC", "LV")])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 12, 11))
    _snap(tmp_path, 2026, 2, [("c", 3.0)], dt.datetime(2026, 8, 20))
    _qb_state(tmp_path / "cache")
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 12, 12), base=tmp_path,
                cache=tmp_path / "cache")
    out = capsys.readouterr().out
    assert "ratings (passthrough): season 2026 week 2" in out.splitlines()[0]
    assert "quarterback" not in out and "nfeloqb" not in out


def test_the_partition_of_a_stale_poll_is_the_schedule_partition_and_nothing_else(sched,
                                                                                  tmp_path):
    """A partition is homogeneous in what priced it, and since #299 the version names the
    source alone: nothing is filed under `-qb`."""
    sched([("a", 1, 3.0, 7), ("b", 2, 3.0, None, "KC", "LAC"), ("c", 2, 3.0, None, "KC", "LV")])
    _snap(tmp_path, 2026, 2, [("b", 3.0)], dt.datetime(2026, 9, 12, 11))
    _snap(tmp_path, 2026, 2, [("c", 3.0)], dt.datetime(2026, 8, 20))
    _qb_state(tmp_path / "cache")
    ratings.fit(2026, 2, at=dt.datetime(2026, 9, 12, 12), base=tmp_path, cache=tmp_path / "cache")
    written = sorted(p.name for p in (tmp_path / "preds").rglob("*.parquet"))
    assert len(written) == 2, written
    assert not any("qb" in n for n in written)


def test_the_seam_is_the_priced_slate_in_the_slates_own_order(sched, tmp_path):
    """`rated_games` is `hub.schedule.priced_games` and nothing else: the same rows, the
    same columns, the same order, with a played week and a week ahead on the slate and a
    state cached that #272 would have rated the played week from."""
    sched([("w1", 1, 3.0, 7, "KC", "LV", "2026-09-14", "20:15"),
           ("w2", 2, 3.0, None, "KC", "LV", "2026-09-27", "13:00")])
    _qb_state(tmp_path / "cache")
    at = dt.datetime(2026, 9, 20, 12)
    games = ratings.rated_games(2026, at=at, cache=tmp_path / "cache", base=tmp_path)
    priced = schedule.priced_games(2026, at=at, cache=tmp_path / "cache", base=tmp_path)
    assert games.equals(priced)
    assert games["game_id"].to_list() == ["w1", "w2"]
    assert games["close_spread"].to_list() == [3.0, 3.0]
