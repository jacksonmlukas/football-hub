"""#383: every capture a poll makes is readable from a fresh checkout.

`data/processed/` is gitignored and an Actions runner discards it, so the captures the scheduled
polls made were unreadable the moment the job ended. Each poll now also writes its validated
frame to a committed, append-only file under `state/odds/`, and `store.connect` unions that tree
under the `lines` name. These are the rule-18 controls, each built from fixtures (never `data/`):

* **fresh checkout** -- an empty local store and only committed files: `lines` and `lines_as_of`
  return the captures. The files below are written by hand, not through the writer under test,
  so the reader is not graded against its own output.
* **append-only** -- a second poll adds a file and never rewrites the first.
* **lookahead** -- `lines_as_of(at)` ignores a committed snapshot captured after `at`.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from hub import store

AT = dt.datetime(2025, 9, 4, 12)


def _commit(base: Path, week: int, stamp: str, rows: list[tuple[str, float, str]],
            season: int = 2025) -> Path:
    """A committed snapshot written by hand, in the format and place `state/odds/` uses."""
    f = base / "_state" / "odds" / str(season) / f"wk{week:02d}" / f"snap-{stamp}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({
        "season": season, "week": week, "captured_at": rows[0][2],
        "rows": [{"game_id": g, "close_spread": s, "spread_price": -110.0,
                  "close_total": 44.5, "total_price": -110.0, "captured_at": at,
                  "polls_unmoved": 1, "unmoved_since": at} for g, s, at in rows]}))
    return f


def _frame(game="g1", spread=-3.0, at=AT) -> pl.DataFrame:
    return pl.DataFrame([{"game_id": game, "close_spread": spread, "spread_price": -110.0,
                          "close_total": 44.5, "total_price": -110.0, "captured_at": at,
                          "polls_unmoved": 1, "unmoved_since": at}])


# --- fresh checkout ---------------------------------------------------------------------

def test_a_fresh_checkout_reads_the_committed_captures(tmp_path):
    _commit(tmp_path, 1, "20250902T110000", [("g1", -3.0, "2025-09-02T11:00:00")])
    _commit(tmp_path, 1, "20250903T110000", [("g1", -3.5, "2025-09-03T11:00:00")])
    assert not any(tmp_path.glob("lines")), "the local store must be empty for this control"

    got = store.lines(2025, base=tmp_path).sort("captured_at")
    assert got["close_spread"].to_list() == [-3.0, -3.5]
    assert got["week"].to_list() == [1, 1]
    assert "lines" in store.tables(tmp_path)

    asof = store.lines_as_of(dt.datetime(2025, 9, 3, 12), 2025, base=tmp_path)
    assert asof["close_spread"].to_list() == [-3.5], "the latest capture at or before `at`"


def test_a_fresh_checkout_with_nothing_committed_is_still_the_empty_answer(tmp_path):
    assert store.lines(2025, base=tmp_path).is_empty()
    assert store.lines_as_of(AT, 2025, base=tmp_path).is_empty()


def test_the_committed_tree_is_unioned_with_the_local_store_and_deduped(tmp_path):
    """The maintainer's own run writes a capture to both places; it must not count twice."""
    store.write(_frame(at=AT).drop("polls_unmoved", "unmoved_since"), "lines", "nfl", 2025, 1,
                base=tmp_path, name="snap-20250904T120000")
    _commit(tmp_path, 1, "20250904T120000", [("g1", -3.0, AT.isoformat())])      # same capture
    _commit(tmp_path, 1, "20250905T110000", [("g1", -4.0, "2025-09-05T11:00:00")])  # only in git
    got = store.lines(2025, base=tmp_path).sort("captured_at")
    assert got["close_spread"].to_list() == [-3.0, -4.0]


def test_the_committed_tree_is_scoped_to_the_season_asked_for(tmp_path):
    _commit(tmp_path, 1, "20240902T110000", [("old", -1.0, "2024-09-02T11:00:00")], season=2024)
    _commit(tmp_path, 1, "20250902T110000", [("g1", -3.0, "2025-09-02T11:00:00")])
    assert store.lines(2025, base=tmp_path)["game_id"].to_list() == ["g1"]


# --- append-only ------------------------------------------------------------------------

def test_a_second_poll_adds_a_file_and_never_rewrites_the_first(tmp_path):
    first = store.write_snapshot(_frame(spread=-3.0), 2025, 1, AT, base=tmp_path)
    before = first.read_bytes()
    second = store.write_snapshot(_frame(spread=-3.5, at=AT + dt.timedelta(days=1)), 2025, 1,
                                  AT + dt.timedelta(days=1), base=tmp_path)
    assert second != first and second.exists()
    assert first.read_bytes() == before
    assert sorted(p.name for p in first.parent.glob("snap-*.json")) == [
        "snap-20250904T120000.json", "snap-20250905T120000.json"]


def test_a_capture_that_disagrees_with_the_file_already_there_is_refused(tmp_path):
    p = store.write_snapshot(_frame(spread=-3.0), 2025, 1, AT, base=tmp_path)
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        store.write_snapshot(_frame(spread=-9.0), 2025, 1, AT, base=tmp_path)
    assert p.read_bytes() == before
    # An identical re-run is idempotent rather than an error.
    assert store.write_snapshot(_frame(spread=-3.0), 2025, 1, AT, base=tmp_path) == p


def test_what_the_writer_commits_is_what_the_reader_returns(tmp_path):
    store.write_snapshot(_frame("g1", -3.0, AT), 2025, 1, AT, base=tmp_path)
    got = store.lines(2025, base=tmp_path)
    assert got.row(0, named=True)["game_id"] == "g1"
    assert got["captured_at"].to_list() == [AT]
    assert got["unmoved_since"].to_list() == [AT]


# --- lookahead --------------------------------------------------------------------------

def test_lines_as_of_ignores_a_committed_snapshot_captured_after_the_moment(tmp_path):
    _commit(tmp_path, 1, "20250902T110000", [("g1", -3.0, "2025-09-02T11:00:00")])
    _commit(tmp_path, 1, "20250905T110000", [("g1", -7.0, "2025-09-05T11:00:00")])
    got = store.lines_as_of(dt.datetime(2025, 9, 4), 2025, base=tmp_path)
    assert got["close_spread"].to_list() == [-3.0]


def test_a_game_only_captured_after_the_moment_is_absent_not_priced_from_the_future(tmp_path):
    _commit(tmp_path, 1, "20250905T110000", [("g1", -7.0, "2025-09-05T11:00:00")])
    assert store.lines_as_of(dt.datetime(2025, 9, 4), 2025, base=tmp_path).is_empty()


# --- the writer is wired into the poll --------------------------------------------------

def test_a_poll_commits_its_snapshot_beside_the_local_store(tmp_path, monkeypatch):
    from hub.fetch import odds
    monkeypatch.setattr(odds, "_game_index", lambda season: ({}, {}))
    monkeypatch.setattr(odds, "_match_game", lambda ev, index: ("2025_01_DAL_PHI", 1))
    monkeypatch.setattr(odds, "_median_home_spread", lambda ev, home: (-3.0, -110.0))
    monkeypatch.setattr(odds, "_median_game_total", lambda ev: (44.5, -110.0))
    odds._record([{"home_team": "x"}], {"x-requests-remaining": "400"}, 2025, AT,
                 tmp_path / "state.json", tmp_path, 50)
    files = list(store.snapshot_root(tmp_path).glob("2025/wk01/snap-*.json"))
    assert [f.name for f in files] == ["snap-20250904T120000.json"]
    fresh = tmp_path / "fresh"
    # The same file, read with no local store at all: a checkout holding only git.
    target = store.snapshot_root(fresh) / "2025" / "wk01"
    target.mkdir(parents=True)
    (target / files[0].name).write_text(files[0].read_text())
    assert store.lines(2025, base=fresh)["game_id"].to_list() == ["2025_01_DAL_PHI"]
