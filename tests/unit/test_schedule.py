"""The season's games, priced, and the record of what priced them.

This rule had one reader and now has two. `hub.models.ratings` prices a weekly prediction
from it and `hub.season.survivor` prices a whole season of survivor picks, and the two must
not be able to disagree about the same game -- which they did for a day, after the weekly
prediction moved onto the dated snapshots and survivor was left reading the field that moves.

The tests that matter are about provenance, not accuracy. Both inputs quote the same
quantity, so what is asserted here is which one was used, that the fallback fires where
there is no snapshot, and that a game nobody has priced is absent rather than guessed at.
"""
import datetime as dt

import polars as pl
import pytest

from hub import schedule, store
from hub.fetch import nflverse


def _sched(rows):
    """A schedule shaped like nflverse's: one row per game, both teams, a lookahead spread."""
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([r[1] for r in rows], dtype=pl.Int32),
         "home_team": [r[4] if len(r) > 4 else "KC" for r in rows],
         "away_team": [r[5] if len(r) > 5 else "LV" for r in rows],
         "spread_line": [r[2] for r in rows],
         "result": [r[3] for r in rows]})


@pytest.fixture
def sched(monkeypatch):
    def _install(rows):
        monkeypatch.setattr(nflverse, "load",
                            lambda source, seasons, cache=None: _sched(rows))
    return _install


def _snap(base, season, week, rows, at):
    store.write(
        pl.DataFrame({"game_id": [r[0] for r in rows],
                      "close_spread": [r[1] for r in rows],
                      "captured_at": [at] * len(rows)},
                     schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                             "captured_at": pl.Datetime}),
        "lines", "nfl", season, week, base=base, name=f"snap-{at:%Y%m%dT%H%M%S}")


# --- which source priced it ------------------------------------------------

def test_a_dated_snapshot_prices_the_game_rather_than_the_moving_field(sched, tmp_path):
    sched([("a", 1, 3.0, None)])
    _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, 5))
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["close_spread"].to_list() == [6.5]
    assert got["price_source"].to_list() == ["snapshot"]


def test_a_game_with_no_snapshot_still_prices_from_the_moving_field(sched, tmp_path):
    sched([("a", 18, 3.0, None)])
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["close_spread"].to_list() == [3.0]
    assert got["price_source"].to_list() == ["schedule"]


def test_a_snapshot_taken_after_the_moment_does_not_price_the_game(sched, tmp_path):
    sched([("a", 1, 3.0, None)])
    _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, 10))
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["price_source"].to_list() == ["schedule"]


def test_a_game_nobody_has_priced_carries_no_source(sched, tmp_path):
    sched([("a", 18, None, None)])
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["price_source"].to_list() == [None]
    assert got["close_spread"].to_list() == [None]


def test_the_snapshot_that_priced_a_row_is_named_by_its_capture_time(sched, tmp_path):
    """"Which source" is only half of provenance -- there are many snapshots and the row has
    to say which one, or re-deriving it means guessing."""
    sched([("a", 1, 3.0, None)])
    _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, 5, 11))
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["priced_at"].to_list() == [dt.datetime(2026, 9, 5, 11)]


def test_a_store_with_no_snapshots_at_all_prices_from_the_moving_field(sched, tmp_path):
    """Graceful degradation: a fresh clone has no `lines` view, and the read must serve the
    moving field rather than raising."""
    sched([("a", 1, 3.0, None)])
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["close_spread"].to_list() == [3.0]


def test_both_teams_survive_so_a_caller_can_price_a_side(sched, tmp_path):
    """`survivor` needs to know who is playing whom, not only what the game is worth."""
    sched([("a", 1, 3.0, None, "KC", "LV")])
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 6), base=tmp_path)
    assert got["home_team"].to_list() == ["KC"] and got["away_team"].to_list() == ["LV"]


def test_a_snapshot_does_not_fan_a_game_out_into_two_rows(sched, tmp_path):
    """Many snapshots per game is the point of the store. A join taking all of them would
    double every game in every consumer."""
    sched([("a", 1, 3.0, None)])
    for day in (4, 5, 6):
        _snap(tmp_path, 2026, 1, [("a", 6.5)], dt.datetime(2026, 9, day))
    got = schedule.priced_games(2026, at=dt.datetime(2026, 9, 7), base=tmp_path)
    assert got.height == 1
    assert got["priced_at"].to_list() == [dt.datetime(2026, 9, 6)]
