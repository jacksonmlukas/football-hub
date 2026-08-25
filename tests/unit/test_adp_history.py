"""Keeping ESPN ADP, which ESPN does not keep.

`fit_espn_weight` returns a prior instead of a fit, the opponent model is blocked, and
ADR-0010 says `edge` can never be validated -- all three because ESPN returns the undrafted
sentinel for past seasons. The data exists only on the day, and until this module the repo
overwrote it on every `make draft`.

All offline.
"""
from datetime import date

import polars as pl

from hub.draft import adp_history as H


def _board(players, adp, pos=None, **cols):
    df = pl.DataFrame({"player": players, "pos": pos or ["RB"] * len(players),
                       "adp": adp}, schema_overrides={"adp": pl.Float64})
    return df.with_columns(**cols) if cols else df


# --- writing --------------------------------------------------------------

def test_a_snapshot_lands_under_its_own_date(tmp_path):
    p = H.snapshot(_board(["A"], [1.5]), on=date(2026, 8, 25), base=tmp_path)
    assert p is not None and p.parent.name == "date=2026-08-25"
    assert pl.read_parquet(p)["adp"].to_list() == [1.5]


def test_rebuilding_the_same_day_replaces_rather_than_duplicates(tmp_path):
    """Draft morning means several builds. Five copies of one day is not a history."""
    day = date(2026, 9, 3)
    H.snapshot(_board(["A"], [1.5]), on=day, base=tmp_path)
    H.snapshot(_board(["A"], [2.5]), on=day, base=tmp_path)
    got = H.history(tmp_path)
    assert got.height == 1
    assert got["adp"].to_list() == [2.5], "the later build of the day wins"


def test_a_board_with_no_adp_is_not_recorded(tmp_path):
    """The documented degraded path. Absent and zero are different claims, and writing an
    empty day would say the market had no opinion that day."""
    assert H.snapshot(pl.DataFrame({"player": ["A"], "pos": ["RB"]}), base=tmp_path) is None
    assert H.days(tmp_path) == []


def test_a_board_whose_adp_is_entirely_null_is_not_recorded(tmp_path):
    board = _board(["A", "B"], [None, None])
    assert H.snapshot(board, base=tmp_path) is None


def test_players_without_adp_are_dropped_but_the_day_is_kept(tmp_path):
    board = _board(["A", "B"], [1.5, None])
    H.snapshot(board, on=date(2026, 8, 25), base=tmp_path)
    got = H.history(tmp_path)
    assert got["player"].to_list() == ["A"]


def test_only_the_archival_columns_are_kept(tmp_path):
    """An archive that has to be readable in a year, not a second copy of the board."""
    board = _board(["A"], [1.5], vor=[3.0], td_luck=[0.5])
    p = H.snapshot(board, on=date(2026, 8, 25), base=tmp_path)
    assert p is not None
    assert set(pl.read_parquet(p).columns) == {"player", "pos", "adp", "date"}


# --- reading --------------------------------------------------------------

def test_history_is_oldest_first_across_days(tmp_path):
    H.snapshot(_board(["A"], [3.0]), on=date(2026, 9, 1), base=tmp_path)
    H.snapshot(_board(["A"], [1.0]), on=date(2026, 8, 20), base=tmp_path)
    got = H.history(tmp_path)
    assert got["date"].to_list() == ["2026-08-20", "2026-09-01"]
    assert got["adp"].to_list() == [1.0, 3.0], "ADP moved, which is the point of keeping it"


def test_an_empty_archive_reads_as_an_empty_frame_not_a_crash(tmp_path):
    got = H.history(tmp_path / "nothing")
    assert got.is_empty() and "adp" in got.columns


def test_days_lists_what_has_been_captured(tmp_path):
    H.snapshot(_board(["A"], [1.0]), on=date(2026, 8, 20), base=tmp_path)
    H.snapshot(_board(["A"], [2.0]), on=date(2026, 9, 3), base=tmp_path)
    assert H.days(tmp_path) == ["2026-08-20", "2026-09-03"]


def test_days_on_an_absent_archive_is_empty(tmp_path):
    assert H.days(tmp_path / "nothing") == []


def test_columns_that_differ_between_days_still_concatenate(tmp_path):
    """A board built without ECR on one day must not make the whole archive unreadable."""
    H.snapshot(_board(["A"], [1.0], ecr=[2.0]), on=date(2026, 8, 20), base=tmp_path)
    H.snapshot(_board(["A"], [2.0]), on=date(2026, 8, 21), base=tmp_path)
    assert H.history(tmp_path).height == 2
