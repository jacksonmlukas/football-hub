"""`hub.fetch.consensus` writes the weekly consensus page down before kickoff, or refuses (#432).

Method rule 18 is the point of this file: the capture's whole value is that it was taken before
the week's first game, so the check that refuses a late one is only trusted once a late one has
been planted and seen refused -- on both sides of the boundary, and again at the read, where a
file that reached the tree by another route has to be named and left out. No test here touches
the network: the fetch is the one function that does and `main`'s tests replace it.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from hub.fetch import consensus
from hub.models import panel

# Week 5 of the fixture schedule: a Thursday game on 2026-10-08, 00:00 Eastern = 04:00 UTC.
FIRST = date(2026, 10, 8)
DEADLINE = datetime(2026, 10, 8, 4, 0, tzinfo=UTC)


def _schedule(first_thursday: date = date(2026, 9, 10), weeks: int = 18) -> pl.DataFrame:
    rows = []
    for w in range(1, weeks + 1):
        thu = first_thursday + timedelta(days=7 * (w - 1))
        for off in (0, 3, 4):
            rows.append({"season": 2026, "week": w, "game_type": "REG",
                         "gameday": (thu + timedelta(days=off)).isoformat()})
    rows.append({"season": 2026, "week": 19, "game_type": "WC", "gameday": "2027-01-10"})
    return pl.DataFrame(rows)


def _archive(*scrapes: tuple[str, str, str, float]) -> pl.DataFrame:
    """(page, scrape_date, player, ecr) rows, plus a different page that must never leak."""
    rows = [{"page_type": p, "scrape_date": d, "player": n, "pos": "WR", "team": "KC", "ecr": e}
            for p, d, n, e in scrapes]
    return pl.DataFrame(rows, schema={"page_type": pl.Utf8, "scrape_date": pl.Utf8,
                                      "player": pl.Utf8, "pos": pl.Utf8, "team": pl.Utf8,
                                      "ecr": pl.Float64})


def _doc(captured_at: datetime, *, week: int = 5, first: date = FIRST) -> dict:
    rows = pl.DataFrame({"player": ["A", "B"], "pos": ["WR", "RB"], "team": ["KC", "SF"],
                         "ecr": [1.0, 2.0]})
    return consensus.build(2026, week, rows, first - timedelta(days=1), first, captured_at,
                           "abc123")


def test_the_page_and_the_lead_are_the_panels_own():
    """A fetcher may not import a model, so the two constants are spelled twice -- and this is
    what holds the spellings equal. A capture of a different page than the gate reads, or with a
    different notion of this week's scrape, would be a capture of the wrong incumbent."""
    assert consensus.PAGE == panel.CONSENSUS_PAGE
    assert consensus.MAX_LEAD_DAYS == panel.CONSENSUS_MAX_LEAD_DAYS


def test_the_deadline_is_the_start_of_the_first_game_day_eastern():
    assert consensus.deadline_for(FIRST) == DEADLINE                      # EDT, UTC-4
    assert consensus.deadline_for(date(2026, 12, 3)) == datetime(2026, 12, 3, 5, 0, tzinfo=UTC)


def test_first_and_last_game_days_read_regular_season_only():
    sched = _schedule()
    days = consensus.first_game_days(sched)
    assert days[1] == date(2026, 9, 10) and days[5] == FIRST and 19 not in days
    assert consensus.last_game_day(sched, 14) == date(2026, 12, 14)
    assert consensus.last_game_day(sched, 19) is None            # a playoff week is not REG


def test_a_weeks_page_cannot_be_a_scrape_that_still_ranks_the_week_before():
    """Wednesday's page for week 5 is eight days ahead of week 6 and is week 5's: a scrape on
    or before the previous week's last game day belongs to that week (`assign_weeks`'s own rule),
    so the floor of week 6 is the day after week 5's Monday game, not eight days before its
    first. Week 1 has nothing earlier and is bounded by the lead alone."""
    floors = consensus.scrape_floors(_schedule())
    assert floors[6] == date(2026, 10, 13)             # Monday 10-12 is week 5's last game
    assert floors[1] == date(2026, 9, 10) - timedelta(days=8)
    assert floors[5] == date(2026, 10, 6)              # the day after week 4's Monday
    arch = _archive(("weekly-op", "2026-10-07", "Wk5Page", 1.0))
    with pytest.raises(consensus.StaleArchive):
        consensus.window_scrape(arch, date(2026, 10, 15), floors[6])
    assert consensus.window_scrape(arch, FIRST, floors[5])[0] == date(2026, 10, 7)


def test_next_week_is_the_first_whose_deadline_is_ahead():
    days = consensus.first_game_days(_schedule())
    assert consensus.next_week(days, DEADLINE - timedelta(seconds=1)) == (5, FIRST)
    # at the deadline itself the week is gone: the next one is what a run then captures
    assert consensus.next_week(days, DEADLINE) == (6, date(2026, 10, 15))
    assert consensus.next_week(days, datetime(2027, 3, 1, tzinfo=UTC)) is None


def test_the_window_takes_the_newest_scrape_of_the_page_before_the_game_day():
    arch = _archive(("weekly-op", "2026-10-05", "Old", 9.0),
                    ("weekly-op", "2026-10-06", "Early", 3.0),
                    ("weekly-op", "2026-10-07", "Late", 1.0),
                    ("weekly-op", "2026-10-07", "Late2", 2.0),
                    ("redraft-overall", "2026-10-07", "OtherPage", 1.0),
                    ("weekly-op", "2026-10-08", "OnTheGameDay", 1.0))
    when, rows = consensus.window_scrape(arch, FIRST)
    assert when == date(2026, 10, 7)
    assert rows["player"].to_list() == ["Late", "Late2"]            # the newest day, by ecr
    assert rows.columns == list(consensus.CAPTURE_COLUMNS)


def test_a_scrape_of_last_weeks_page_is_stale_not_this_weeks():
    """Plant (rule 18): the archive's newest scrape is nine days before kickoff, the previous
    week's page. Filing it under this week would be a lookahead ranking of the wrong games."""
    arch = _archive(("weekly-op", "2026-09-29", "LastWeek", 1.0))
    with pytest.raises(consensus.StaleArchive, match="not published yet"):
        consensus.window_scrape(arch, FIRST)
    # ...and the game day's own scrape is not a lookahead one either
    with pytest.raises(consensus.StaleArchive):
        consensus.window_scrape(_archive(("weekly-op", "2026-10-08", "X", 1.0)), FIRST)
    # the boundary the other way: exactly MAX_LEAD_DAYS ahead is still this week's
    ok = _archive(("weekly-op", (FIRST - timedelta(days=consensus.MAX_LEAD_DAYS)).isoformat(),
                   "Edge", 1.0))
    assert consensus.window_scrape(ok, FIRST)[0] == FIRST - timedelta(days=8)


def test_a_capture_written_after_kickoff_is_refused_and_nothing_lands(tmp_path):
    """Rule 18's control, planted: the capture one second past the deadline is refused, and the
    one a second before is written -- so the check is seen to fire on the condition it exists
    for and to leave the on-time case alone."""
    late = _doc(DEADLINE)                                           # at the deadline: late
    with pytest.raises(consensus.LateCapture, match="nothing was written"):
        consensus.write_capture(late, tmp_path)
    assert list(tmp_path.rglob("*")) == []                          # not even a directory
    with pytest.raises(consensus.LateCapture):
        consensus.write_capture(_doc(DEADLINE + timedelta(hours=9)), tmp_path)
    path = consensus.write_capture(_doc(DEADLINE - timedelta(seconds=1)), tmp_path)
    assert path.exists() and path.parent.name == "wk05"


def test_a_capture_is_append_only_atomic_and_idempotent(tmp_path):
    doc = _doc(DEADLINE - timedelta(hours=20))
    first = consensus.write_capture(doc, tmp_path)
    assert consensus.write_capture(doc, tmp_path) == first          # same document: a no-op
    assert [p.name for p in first.parent.iterdir()] == [first.name]  # no `.part` scratch left
    changed = dict(doc, rows=doc["rows"][:1])
    with pytest.raises(FileExistsError, match="append-only"):
        consensus.write_capture(changed, tmp_path)
    assert consensus.read_capture(first).rows.height == 2          # the original stands


def test_a_capture_reads_back_as_written(tmp_path):
    at = DEADLINE - timedelta(hours=20)
    consensus.write_capture(_doc(at), tmp_path)
    (cap,) = consensus.read_captures(2026, tmp_path)
    assert (cap.season, cap.week, cap.first_game_day) == (2026, 5, FIRST)
    assert cap.captured_at == at and cap.deadline == DEADLINE
    assert cap.scrape_date == FIRST - timedelta(days=1) and cap.digest == "abc123"
    assert cap.rows["player"].to_list() == ["A", "B"] and cap.path is not None
    assert consensus.read_captures(2025, tmp_path) == []


def test_capture_week_cuts_the_next_weeks_page_from_the_archive(tmp_path):
    arch = _archive(("weekly-op", "2026-10-07", "Star", 1.0), ("weekly-op", "2026-10-07", "Two", 2.0))
    now = DEADLINE - timedelta(hours=12)
    path, week = consensus.capture_week(_schedule(), arch, 2026, now, "d1", tmp_path)
    assert week == 5 and path.parent.name == "wk05"
    cap = consensus.read_capture(path)
    assert cap.rows["player"].to_list() == ["Star", "Two"] and cap.scrape_date == date(2026, 10, 7)
    # the same run an hour past the deadline targets week 6 -- whose page is not there yet
    with pytest.raises(consensus.StaleArchive):
        consensus.capture_week(_schedule(), arch, 2026, DEADLINE + timedelta(hours=1), "d1",
                               tmp_path)
    with pytest.raises(ValueError, match="no regular-season week"):
        consensus.capture_week(_schedule(), arch, 2026, datetime(2027, 3, 1, tzinfo=UTC), "d1",
                               tmp_path)


def test_main_answers_a_failed_fetch_with_a_sentence_and_a_stale_page_with_another(
        monkeypatch, capsys):
    """Graceful degradation: a fetch that fails is one line and a non-zero exit, the committed
    captures untouched; a page not yet published is a different, named exit the next cron retries."""
    assert consensus.main([]) == 2

    def boom(_season):
        raise ConnectionError("no route")

    monkeypatch.setattr(consensus, "_fetch", boom)
    assert consensus.main(["--capture"]) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "ConnectionError" in err and "Traceback" not in err

    far = _schedule(first_thursday=date(2099, 9, 10))
    monkeypatch.setattr(consensus, "_fetch", lambda _s: (far, _archive(), "d"))
    assert consensus.main(["--capture"]) == 3
    assert "nothing captured" in capsys.readouterr().err


@pytest.fixture
def spy(monkeypatch):
    """`nflverse.load` and friends replaced by a recorder: what each call site actually passes
    as `cols`, with no network. Returns `{source: cols}` as the calls arrive."""
    from hub.fetch import nflverse

    asked: dict[str, tuple[str, ...]] = {}

    def load(source, seasons, cols=None, **_kw):
        asked[source] = tuple(cols or ())
        if source == "schedules":
            return _schedule()
        return pl.DataFrame({"season": [2026], "week": [14]})

    monkeypatch.setattr(nflverse, "load", load)
    monkeypatch.setattr(nflverse, "load_rankings", lambda *a, **k: _archive())
    monkeypatch.setattr(nflverse, "data_pin", lambda *a, **k: None)
    return asked


def _missing(cols, contract):
    return sorted(set(contract.required) - set(cols))


def test_the_columns_each_call_site_asks_for_satisfy_the_contracts(spy, monkeypatch, tmp_path):
    """The fetch used to be uncovered, and its first live run failed with a contract violation:
    the schedule was asked for four columns and the contract requires five. Here the call sites
    themselves are driven against a spy and the `cols` each really passes is held against the
    contract's required set -- not a constant beside them."""
    from hub.contracts import PLAYER_STATS, SCHEDULES
    from hub.fetch import nflverse
    from hub.season import weekly_forward

    consensus._fetch(2026)
    assert _missing(spy["schedules"], SCHEDULES) == []
    spy.clear()
    assert weekly_forward.week14_loaded() is True
    assert _missing(spy["player_stats"], PLAYER_STATS) == []
    spy.clear()
    monkeypatch.setattr(weekly_forward.consensus, "read_captures", lambda _s: [])
    assert weekly_forward.main(["--as-of", "2026-10-07"]) == 0         # NOT-YET: nothing read
    assert _missing(spy["schedules"], SCHEDULES) == []
    assert "player_stats" not in spy            # the data is not asked about before the date
    assert nflverse.load is not None


def test_a_four_column_ask_at_the_call_site_is_seen_to_fail(spy):
    """The plant. Rewrite `_fetch`'s own source so the schedule call carries the old inline
    four-column tuple, run it against the same spy, and the check must go red naming the three
    columns the live run named -- so the test above is able to see the defect at a call site."""
    import inspect
    import textwrap

    from hub.contracts import SCHEDULES

    src = textwrap.dedent(inspect.getsource(consensus._fetch))
    assert "cols=SCHEDULE_COLS" in src
    bad = src.replace("cols=SCHEDULE_COLS", 'cols=("season", "week", "game_type", "gameday")')
    ns = dict(vars(consensus))
    exec(bad, ns)
    ns["_fetch"](2026)
    assert _missing(spy["schedules"], SCHEDULES) == ["away_team", "game_id", "home_team"]
