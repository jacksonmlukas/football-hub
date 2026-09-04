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


# --- what a reader can obtain, and what only this machine can ------------------
#
# #6 gave every prediction a `price_source` and a `priced_at` under a stated purpose: a
# prediction that can be re-derived later from the data that produced it. Against the store on
# this machine that citation is complete. Against the public repository it is not, and nothing
# said so -- `data/processed/` is gitignored as redistributed third-party data the repo cannot
# publish, so the timestamp names a file no reader can open.

def test_every_source_the_rule_can_emit_is_classified():
    """The assertion that catches the next source added.

    Read off the pricing rule itself rather than off a fixture: a behavioural test only sees
    the sources its own frame happens to exercise, so a fourth branch would ship unclassified
    and the artifact would over-claim again.

    Targeted at the branch chain -- the literals handed to `.then(pl.lit(...))` -- rather than
    at every string in the function. My first version was a denylist of column names, which
    would have failed the day someone added a column and taught the next reader to edit the
    exclusion list instead of the classification.
    """
    import ast
    import inspect

    emitted = set()
    for node in ast.walk(ast.parse(inspect.getsource(schedule.priced_games))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "then" and node.args):
            continue
        arg = node.args[0]
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "lit" and arg.args
                and isinstance(arg.args[0], ast.Constant)
                and isinstance(arg.args[0].value, str)):
            emitted.add(arg.args[0].value)

    assert emitted == {"snapshot", "schedule"}, (
        f"the branch chain now emits {sorted(emitted)}; this test reads the rule and the "
        f"rule has changed shape")
    unclassified = emitted - set(schedule.PROVENANCE)
    assert not unclassified, (
        f"{sorted(unclassified)} can price a prediction and nothing says whether a reader "
        f"can obtain it. Classify it in `schedule.PROVENANCE`.")


def test_an_unclassified_source_is_a_loud_failure_not_a_default():
    """A convenient default is how the artifact comes to over-claim quietly."""
    with pytest.raises(KeyError, match="consensus"):
        schedule.provenance("consensus")


def test_neither_source_is_obtainable_by_a_reader_today_and_for_different_reasons():
    """The finding worth publishing, and it is not the one the ticket assumed. A snapshot is
    immutable and unpublished; the moving field is published and has since moved. Both fail
    re-derivation, and the *reason* is what tells a reader which could change."""
    snap, sched = schedule.provenance("snapshot"), schedule.provenance("schedule")
    assert not snap.reader_can_obtain and not sched.reader_can_obtain
    assert snap.why != sched.why
    assert "publish" in snap.why and "moved" in sched.why


def test_the_reason_cites_the_constraint_rather_than_restating_it():
    """A future reader has to be able to tell a licence constraint from an oversight."""
    assert ".gitignore" in schedule.provenance("snapshot").why


def test_the_classification_is_data_a_caller_can_serialise():
    got = schedule.provenance("snapshot")
    assert isinstance(got.reader_can_obtain, bool) and isinstance(got.why, str)
