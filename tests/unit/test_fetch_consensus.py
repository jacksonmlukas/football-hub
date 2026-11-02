"""`hub.fetch.consensus` writes the weekly consensus page down before kickoff, or refuses (#432).

Method rule 18 is the point of this file: the capture's whole value is that it was taken before
the week's first game, so the check that refuses a late one is only trusted once a late one has
been planted and seen refused -- on both sides of the boundary, and again at the read, where a
file that reached the tree by another route has to be named and left out. No test here touches
the network: the fetch is the one function that does and `main`'s tests replace it.
"""
from __future__ import annotations

import json
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


# --- the lineup capture: after Thursday, before Sunday, kept apart ---------------------------
#
# Week 5 of the fixture: week 4's last game Monday 10-05, Thursday 10-08, Sunday 10-11 (whose
# start, Eastern, is 04:00 UTC), Monday 10-12. The archive publishes on Fridays.

SUNDAY = date(2026, 10, 11)
SUNDAY_DEADLINE = datetime(2026, 10, 11, 4, 0, tzinfo=UTC)
FRIDAY_RUN = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)        # 32h before Sunday's deadline


def _lineup_doc(captured_at: datetime, scrape: date = date(2026, 10, 9)) -> dict:
    rows = pl.DataFrame({"player": ["A", "B"], "pos": ["WR", "RB"], "team": ["KC", "SF"],
                         "ecr": [1.0, 2.0]})
    return consensus.build_lineup(2026, 5, rows, scrape, date(2026, 10, 6), FIRST, SUNDAY,
                                  captured_at, "abc123")


def test_the_first_sunday_and_the_lineup_week_follow_the_sunday_deadline():
    sched = _schedule()
    assert consensus.first_sundays(sched)[5] == SUNDAY and 19 not in consensus.first_sundays(sched)
    assert consensus.lineup_deadline_for(SUNDAY) == SUNDAY_DEADLINE
    # Friday, after week 5's Thursday deadline: the week capture has moved on, the Friday one has not
    assert consensus.next_week(consensus.first_game_days(sched), FRIDAY_RUN) == (
        6, date(2026, 10, 15))
    assert consensus.lineup_week(sched, FRIDAY_RUN) == (5, FIRST, SUNDAY)
    assert consensus.lineup_week(sched, SUNDAY_DEADLINE) == (
        6, date(2026, 10, 15), date(2026, 10, 18))
    assert consensus.lineup_week(sched, datetime(2027, 3, 1, tzinfo=UTC)) is None


def test_the_lineup_window_takes_fridays_page_and_refuses_either_side_of_it(tmp_path):
    sched = _schedule()
    # a scrape on week 4's Monday game day still ranks week 4: rejected
    with pytest.raises(consensus.StaleArchive):
        consensus.capture_lineup(sched, _archive(("weekly-op", "2026-10-05", "Mon", 1.0)),
                                 2026, FRIDAY_RUN, "d", tmp_path)
    # a scrape dated Sunday is not before Sunday's kickoff: rejected
    with pytest.raises(consensus.StaleArchive):
        consensus.capture_lineup(sched, _archive(("weekly-op", "2026-10-11", "Sun", 1.0)),
                                 2026, FRIDAY_RUN, "d", tmp_path)
    assert list(tmp_path.rglob("*.json")) == []
    # Friday's page, after Thursday's game: accepted, and filed apart from the week captures
    arch = _archive(("weekly-op", "2026-10-06", "Tue", 5.0), ("weekly-op", "2026-10-09", "Fri", 1.0))
    path, week = consensus.capture_lineup(sched, arch, 2026, FRIDAY_RUN, "d", tmp_path)
    assert week == 5 and path.parent == tmp_path / "2026" / "lineup" / "wk05"
    assert path.name.startswith("lineup-")
    [cap] = consensus.read_lineup_captures(2026, tmp_path)
    assert cap.scrape_date == date(2026, 10, 9) and cap.rows["player"].to_list() == ["Fri"]
    assert cap.first_sunday == SUNDAY and cap.deadline == SUNDAY_DEADLINE
    assert consensus.read_captures(2026, tmp_path) == [], "the week reader does not see it"
    with pytest.raises(ValueError, match="no regular-season week"):
        consensus.capture_lineup(sched, arch, 2026, datetime(2027, 3, 1, tzinfo=UTC), "d",
                                 tmp_path)


def test_a_lineup_capture_written_after_sunday_starts_is_refused_and_nothing_lands(tmp_path):
    """Rule 18's control for the lineup capture: at Sunday's deadline refused, a second before
    written; and a capture after the deadline cannot be smuggled in through the read either."""
    with pytest.raises(consensus.LateCapture, match="nothing was written"):
        consensus.write_lineup_capture(_lineup_doc(SUNDAY_DEADLINE), tmp_path)
    with pytest.raises(consensus.LateCapture, match="outside the week's window"):
        consensus.write_lineup_capture(_lineup_doc(FRIDAY_RUN, scrape=SUNDAY), tmp_path)
    with pytest.raises(consensus.LateCapture, match="purpose"):
        consensus.write_lineup_capture(_doc(FRIDAY_RUN), tmp_path)   # a week doc, mislabelled
    assert list(tmp_path.rglob("*")) == []
    path = consensus.write_lineup_capture(_lineup_doc(SUNDAY_DEADLINE - timedelta(seconds=1)),
                                          tmp_path)
    assert consensus.write_lineup_capture(_lineup_doc(SUNDAY_DEADLINE - timedelta(seconds=1)),
                                          tmp_path) == path        # append-only, idempotent
    with pytest.raises(FileExistsError):
        consensus.write_lineup_capture(
            dict(_lineup_doc(SUNDAY_DEADLINE - timedelta(seconds=1)), digest="other"), tmp_path)
    late = consensus.lineup_path(2026, 5, SUNDAY_DEADLINE, tmp_path)
    late.write_text(json.dumps(_lineup_doc(SUNDAY_DEADLINE)))
    assert [c.path for c in consensus.read_lineup_captures(2026, tmp_path)] == [path]


def test_the_week_reader_drops_a_lineup_capture_even_filed_as_a_week_one(tmp_path):
    """The plant: a lineup document copied to `wk05/cap-*.json` matches the week glob. The
    `purpose` check is what keeps it out of the reader whose captures are labelled in time for every player."""
    ok = consensus.write_capture(_doc(DEADLINE - timedelta(hours=20)), tmp_path)
    smuggled = consensus.capture_path(2026, 5, DEADLINE - timedelta(hours=2), tmp_path)
    smuggled.write_text(json.dumps(_lineup_doc(DEADLINE - timedelta(hours=2))))
    assert [c.path for c in consensus.read_captures(2026, tmp_path)] == [ok]


@pytest.mark.parametrize(("hours_left", "code"), [
    (23.9, consensus.NOTHING_AND_LATE), (1.0, consensus.NOTHING_AND_LATE),
    (24.0, consensus.NOTHING_YET), (40.0, consensus.NOTHING_YET),
    (-1.0, consensus.NOTHING_YET), (None, consensus.NOTHING_YET)])
def test_an_empty_run_is_red_only_inside_a_day_of_its_deadline(hours_left, code):
    deadline = (None if hours_left is None
                else SUNDAY_DEADLINE)
    now = SUNDAY_DEADLINE - timedelta(hours=hours_left or 0)
    assert consensus.empty_exit(deadline, now) == code


def test_main_lineup_exits_by_deadline_proximity_and_writes_when_the_page_is_there(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(consensus, "ROOT", tmp_path)
    assert consensus.main(["--capture", "--lineup"]) == 2
    monkeypatch.setattr(consensus, "_fetch", lambda _s: (_schedule(), _archive(), "d"))
    monkeypatch.setattr(consensus, "_now", lambda: FRIDAY_RUN)              # 32h left
    assert consensus.main(["--lineup"]) == consensus.NOTHING_YET == 3
    assert "nothing captured for the lineup" in capsys.readouterr().err
    saturday = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)                    # 14h left
    monkeypatch.setattr(consensus, "_now", lambda: saturday)
    assert consensus.main(["--lineup"]) == consensus.NOTHING_AND_LATE == 4
    assert "14.0h away" in capsys.readouterr().err
    arch = _archive(("weekly-op", "2026-10-09", "Fri", 1.0))
    monkeypatch.setattr(consensus, "_fetch", lambda _s: (_schedule(), arch, "d"))
    assert consensus.main(["--lineup"]) == 0
    assert len(consensus.read_lineup_captures(2026, tmp_path)) == 1
    assert consensus.read_captures(2026, tmp_path) == []
    # the week capture's empty run stays a notice whatever the clock says
    monkeypatch.setattr(consensus, "_now", lambda: DEADLINE - timedelta(hours=2))
    monkeypatch.setattr(consensus, "_fetch", lambda _s: (_schedule(), _archive(), "d"))
    assert consensus.main(["--capture"]) == 3


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
    monkeypatch.setattr(weekly_forward.consensus, "read_every_capture", lambda _s: [])
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


# --- in time per player (ADR-0027, #515) ---------------------------------------------------------
#
# Week 7 of 2026 as the schedule has it: Thursday 10-22 DAL@PHI 20:15 ET, Sunday 10-25 KC@BUF
# 13:00 ET, Monday 10-26 NYJ@MIA 20:15 ET, all in daylight time. JAX plays Sunday too, so the
# archive's `JAC` has a game to be placed at, and SF is on a bye.

def _week7(gametime: bool = True) -> pl.DataFrame:
    rows = [("2026-10-22", "20:15", "DAL", "PHI"), ("2026-10-25", "13:00", "KC", "BUF"),
            ("2026-10-25", "13:00", "JAX", "TEN"), ("2026-10-26", "20:15", "NYJ", "MIA"),
            ("2026-11-01", "13:00", "SF", "SEA")]
    df = pl.DataFrame([{"season": 2026, "week": 8 if d == "2026-11-01" else 7,
                        "game_type": "REG", "gameday": d, "gametime": t, "away_team": a,
                        "home_team": h} for d, t, a, h in rows])
    return df if gametime else df.drop("gametime")


THURSDAY7 = date(2026, 10, 22)
FRIDAY7 = datetime(2026, 10, 23, 15, tzinfo=UTC)
W7 = pl.DataFrame({"player": ["Thu Guy", "Sun Guy", "Jag Guy", "Mon Guy", "Bye Guy", "Nobody"],
                   "pos": ["WR"] * 6, "team": ["PHI", "KC", "JAC", "MIA", "SF", "FA"],
                   "ecr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def test_kickoffs_are_the_schedules_eastern_times_on_one_utc_clock():
    """`hub.schedule.kickoff_expr`'s conversion, with UTC attached: 13:00 EDT on 10-25 is 17:00
    UTC and 13:00 EST on 11-08 is 18:00 UTC -- a fixed offset would be right for one of them
    only. A game with no time is placed at the start of its day, Eastern (`deadline_for`)."""
    k = consensus.kickoffs(_week7())
    assert k[(7, "KC")] == k[(7, "BUF")] == datetime(2026, 10, 25, 17, 0, tzinfo=UTC)
    assert k[(7, "PHI")] == datetime(2026, 10, 23, 0, 15, tzinfo=UTC)
    after_dst = pl.DataFrame([{"week": 10, "game_type": "REG", "gameday": "2026-11-08",
                               "gametime": "13:00", "away_team": "KC", "home_team": "BUF"}])
    assert consensus.kickoffs(after_dst)[(10, "KC")] == datetime(2026, 11, 8, 18, 0, tzinfo=UTC)
    untimed = consensus.kickoffs(_week7(gametime=False))
    assert untimed[(7, "KC")] == consensus.deadline_for(date(2026, 10, 25))
    assert consensus.kickoffs(_schedule()) == {}              # no teams: nobody to place


def test_a_friday_capture_is_in_time_for_sunday_players_and_not_for_a_thursday_player():
    """ADR-0027's test, at the rows: Friday is after Thursday's kickoff and before Sunday's and
    Monday's. `JAC` is the archive's spelling of JAX, placed by `TEAM_ALIASES`; a bye has no game
    to have seen; a team the schedule does not know cannot be shown in time and is left out."""
    kicks = consensus.kickoffs(_week7())
    kept, out = consensus.in_time_rows(W7, FRIDAY7, 7, THURSDAY7, kicks)
    assert kept["player"].to_list() == ["Sun Guy", "Jag Guy", "Mon Guy", "Bye Guy"]
    assert out == 2
    # a capture taken before the week's first game day precedes every kickoff: all stand
    wed = datetime(2026, 10, 21, 20, tzinfo=UTC)
    kept_wed, out_wed = consensus.in_time_rows(W7, wed, 7, THURSDAY7, kicks)
    assert kept_wed.equals(W7) and out_wed == 0
    # Thursday 20:14 ET is in time for Thursday's game too; 20:15 ET is not
    before = datetime(2026, 10, 23, 0, 14, tzinfo=UTC)
    assert "Thu Guy" in consensus.in_time_rows(W7, before, 7, THURSDAY7, kicks)[0]["player"]
    at = datetime(2026, 10, 23, 0, 15, tzinfo=UTC)
    assert "Thu Guy" not in consensus.in_time_rows(W7, at, 7, THURSDAY7, kicks)[0]["player"]
    # Sunday 14:00 ET: in time only for Monday's player and the bye
    sunday = datetime(2026, 10, 25, 18, tzinfo=UTC)
    assert consensus.in_time_rows(W7, sunday, 7, THURSDAY7, kicks)[0]["player"].to_list() == [
        "Mon Guy", "Bye Guy"]


def test_the_team_aliases_are_the_draft_modules_own():
    from hub.draft import playoff_sos
    assert consensus.TEAM_ALIASES == playoff_sos.TEAM_ALIASES


def test_every_capture_reads_both_kinds_and_the_week_reader_only_week_captures(tmp_path):
    week = consensus.write_capture(_doc(DEADLINE - timedelta(hours=20)), tmp_path)
    fri = consensus.write_lineup_capture(_lineup_doc(FRIDAY_RUN), tmp_path)
    every = consensus.read_every_capture(2026, tmp_path)
    assert [(c.path, c.purpose) for c in every] == [(week, consensus.WEEK),
                                                    (fri, consensus.LINEUP)]
    assert every[1].deadline == SUNDAY_DEADLINE and every[1].scrape_date == date(2026, 10, 9)
    assert [c.path for c in consensus.read_captures(2026, tmp_path)] == [week]
