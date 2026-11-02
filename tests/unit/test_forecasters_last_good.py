"""Last-good serving for the packet's sources (CLAUDE.md, "Graceful degradation").

A failed fetch serves the last copy that was fetched; it is never allowed to skip an as-of
check. These tests prove both halves: the copy comes back with its *content* timestamps
intact (so it is judged on when the source said it was written, not on when it was cached),
and a cached copy that breaks a check is refused exactly as a live one would be.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from forecasters_fixtures import (
    FETCHED,
    RUN_START,
    _build,
    _injuries,
    _prov,
    _qb,
    _schedule,
)

from hub.forecasters.inputs import InjuryReport, QbSource, Schedule
from hub.forecasters.last_good import NoLastGood, serve
from hub.forecasters.packet import PacketRefused


def _boom() -> InjuryReport:
    raise ConnectionError("espn unreachable")


def test_a_live_fetch_is_returned_and_remembered(tmp_path: Path):
    inj = _injuries()
    got = serve("injuries", lambda: inj, InjuryReport, tmp_path)
    assert got.value == inj and got.degraded is False and got.why is None
    assert (tmp_path / "injuries.json").exists()


def test_a_failed_fetch_serves_the_last_copy_with_its_content_times_and_says_so(tmp_path: Path):
    inj = _injuries(RUN_START - timedelta(days=2))
    serve("injuries", lambda: inj, InjuryReport, tmp_path)
    got = serve("injuries", _boom, InjuryReport, tmp_path)
    assert got.degraded is True and "espn unreachable" in (got.why or "")
    assert got.value.published_at == inj.published_at              # the source's own time
    assert got.value.stamp.fetched_at == FETCHED              # the original fetch time
    assert got.value.stamp.served_from == "last_good"
    assert got.value.entries == inj.entries


def test_a_failed_fetch_with_nothing_cached_is_refused_not_invented(tmp_path: Path):
    with pytest.raises(NoLastGood, match="espn unreachable"):
        serve("injuries", _boom, InjuryReport, tmp_path)


def test_an_unreadable_last_copy_is_no_last_copy(tmp_path: Path):
    (tmp_path / "injuries.json").write_text("{not json")
    with pytest.raises(NoLastGood):
        serve("injuries", _boom, InjuryReport, tmp_path)


def test_all_three_cached_sources_round_trip(tmp_path: Path):
    for name, cls, value in (("injuries", InjuryReport, _injuries()), ("qb", QbSource, _qb()),
                             ("schedule", Schedule, _schedule())):
        serve(name, lambda v=value: v, cls, tmp_path)
        got = serve(name, lambda: (_ for _ in ()).throw(OSError("down")), cls, tmp_path)
        assert got.degraded and got.value.stamp.served_from == "last_good"
        assert got.value.stamp.source == value.stamp.source


def test_a_stale_but_valid_last_good_report_passes_and_is_marked_last_good(tmp_path: Path):
    serve("injuries", lambda: _injuries(RUN_START - timedelta(days=9)), InjuryReport, tmp_path)
    got = serve("injuries", _boom, InjuryReport, tmp_path)
    p = _build(injuries=got.value)
    assert p.stamp["injuries"].served_from == "last_good"
    assert p.injuries.published_at == RUN_START - timedelta(days=9)


def test_a_last_good_report_published_after_run_start_is_still_refused(tmp_path: Path):
    # The copy was fetched in a later run than the one being rebuilt (a re-run of an old
    # slate, say): last-good is no exemption from the check.
    serve("injuries", lambda: _injuries(RUN_START + timedelta(hours=1)), InjuryReport, tmp_path)
    got = serve("injuries", _boom, InjuryReport, tmp_path)
    with pytest.raises(PacketRefused) as exc:
        _build(injuries=got.value)
    assert {f.check for f in exc.value.failures} == {"injury_published"}


def test_a_last_good_qb_source_updated_after_run_start_is_still_refused(tmp_path: Path):
    late = _qb(RUN_START + timedelta(minutes=5))
    serve("qb", lambda: late, QbSource, tmp_path)
    got = serve("qb", lambda: (_ for _ in ()).throw(OSError("down")), QbSource, tmp_path)
    with pytest.raises(PacketRefused) as exc:
        _build(qb_source=got.value)
    assert {f.check for f in exc.value.failures} == {"qb_source_updated"}


def test_a_last_good_schedule_cannot_smuggle_a_late_game_in(tmp_path: Path):
    from hub.forecasters.inputs import PlayedGame
    late = PlayedGame("late", 2026, "REG", RUN_START - timedelta(hours=1), "KC", "NYG", 100, 0,
                      True, RUN_START + timedelta(minutes=1))
    sched = Schedule(_prov("nflverse:schedules"), (*_schedule().games, late))
    serve("schedule", lambda: sched, Schedule, tmp_path)
    got = serve("schedule", lambda: (_ for _ in ()).throw(OSError("down")), Schedule, tmp_path)
    assert _build(schedule=got.value).records.home.games == 4


def test_the_timestamps_survive_as_aware_utc(tmp_path: Path):
    serve("qb", lambda: _qb(datetime(2026, 10, 9, 18, 0, tzinfo=UTC)), QbSource, tmp_path)
    got = serve("qb", lambda: (_ for _ in ()).throw(OSError("down")), QbSource, tmp_path)
    assert got.value.updated_at == datetime(2026, 10, 9, 18, 0, tzinfo=UTC)
    assert got.value.updated_at is not None and got.value.updated_at.tzinfo is not None
