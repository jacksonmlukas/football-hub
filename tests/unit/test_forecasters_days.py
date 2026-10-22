"""`hub.forecasters.days`: which day's run forecasts a game, and whether a trigger is in window.

The prereg: a game belongs to its kickoff date's run if kickoff is at or after 13:00 ET, else to
the day before's. The run window is 09:30-11:30 ET. Both are *Eastern wall-clock* rules, so the
tests that matter are the ones a fixed UTC offset would get wrong: 1 November 2026 is the
Sunday daylight time ends, and the same UTC instant is 13:00 one week and 12:00 the next.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hub.forecasters import days


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


# --- day assignment -----------------------------------------------------------------------

@pytest.mark.parametrize(("kickoff", "run_day"), [
    (utc(2026, 10, 11, 17, 0), date(2026, 10, 11)),    # Sun 13:00 EDT -> Sunday's run
    (utc(2026, 10, 11, 16, 59), date(2026, 10, 10)),   # Sun 12:59 EDT -> Saturday's
    (utc(2026, 10, 9, 0, 15), date(2026, 10, 8)),      # Thu 20:15 EDT (Fri 00:15Z) -> Thursday's
    (utc(2026, 10, 13, 0, 15), date(2026, 10, 12)),    # Mon 20:15 EDT -> Monday's
])
def test_kickoffs_at_or_after_one_pm_eastern_belong_to_their_own_day(kickoff, run_day):
    assert days.run_day(kickoff) == run_day


def test_the_dst_change_is_read_on_eastern_wall_clock_not_a_fixed_offset():
    # 1 Nov 2026, standard time from 02:00: 13:00 EST is 18:00Z, and 17:00Z is only 12:00 EST.
    assert days.run_day(utc(2026, 11, 1, 18, 0)) == date(2026, 11, 1)
    assert days.run_day(utc(2026, 11, 1, 17, 0)) == date(2026, 10, 31)   # -4h would say Nov 1
    # the week before, 17:00Z *is* 13:00 EDT
    assert days.run_day(utc(2026, 10, 25, 17, 0)) == date(2026, 10, 25)


def test_an_early_thanksgiving_game_is_forecast_the_day_before():
    # Thursday 26 Nov 2026, 12:30 EST -> Wednesday's run; the 16:30 game is Thursday's
    assert days.run_day(utc(2026, 11, 26, 17, 30)) == date(2026, 11, 25)
    assert days.run_day(utc(2026, 11, 26, 21, 30)) == date(2026, 11, 26)


def test_an_international_morning_game_is_forecast_the_day_before():
    # London, Sunday 09:30 EDT (13:30Z) -> Saturday's run
    assert days.run_day(utc(2026, 10, 18, 13, 30)) == date(2026, 10, 17)
    # Munich or Madrid after the change, 09:30 EST (14:30Z) -> Saturday's
    assert days.run_day(utc(2026, 11, 8, 14, 30)) == date(2026, 11, 7)


def test_a_naive_kickoff_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        days.run_day(datetime(2026, 10, 11, 17, 0))


def test_due_selects_the_games_of_one_run_day():
    ks = {"a": utc(2026, 10, 11, 17, 0), "b": utc(2026, 10, 11, 13, 30), "c": utc(2026, 10, 13, 0, 15)}
    assert days.due(ks, date(2026, 10, 11)) == ["a"]
    assert days.due(ks, date(2026, 10, 10)) == ["b"]


def test_every_same_day_kickoff_leaves_at_least_ninety_minutes_after_the_window():
    """The prereg's arithmetic: the window ends 11:30 ET and a same-day game kicks off at or
    after 13:00 ET, so only a run longer than 30 minutes can hit the 60-minute rule."""
    for minute in range(0, 24 * 60, 5):
        k = datetime(2026, 11, 1, tzinfo=days.ET).replace(hour=minute // 60, minute=minute % 60)
        d = days.run_day(k.astimezone(UTC))
        if d == k.date():
            end = datetime(d.year, d.month, d.day, 11, 30, tzinfo=days.ET)
            assert (k - end).total_seconds() >= 90 * 60


# --- the trigger window -------------------------------------------------------------------

@pytest.mark.parametrize(("now", "inside"), [
    (utc(2026, 10, 10, 13, 30), True),    # 09:30 EDT, the first trigger
    (utc(2026, 10, 10, 13, 29), False),
    (utc(2026, 10, 10, 15, 29), True),    # 11:29 EDT
    (utc(2026, 10, 10, 15, 30), False),   # 11:30 EDT: the window has closed
    (utc(2026, 10, 10, 16, 30), False),   # the last cron, 12:30 EDT
    (utc(2026, 11, 7, 13, 30), False),    # 08:30 EST -- the cron fires, the code says no
    (utc(2026, 11, 7, 14, 30), True),     # 09:30 EST
    (utc(2026, 11, 7, 16, 15), True),     # 11:15 EST, only reachable in standard time
    (utc(2026, 11, 7, 16, 30), False),    # 11:30 EST
])
def test_the_window_is_half_past_nine_to_half_past_eleven_eastern(now, inside):
    assert days.in_window(now) is inside


def test_the_crons_cover_the_window_in_both_daylight_and_standard_time():
    """Every 15 minutes 13:30-16:30 UTC: each 15-minute ET slot of the window is hit by at
    least one trigger whether ET is UTC-4 or UTC-5."""
    triggers = days.CRON_TRIGGERS_UTC
    for day in (date(2026, 10, 10), date(2026, 11, 7)):
        hits = {days.eastern(datetime(day.year, day.month, day.day, h, m, tzinfo=UTC)).time()
                for h, m in triggers if days.in_window(datetime(day.year, day.month, day.day, h, m, tzinfo=UTC))}
        assert len(hits) == 8                             # 09:30, 09:45, ..., 11:15


def test_should_run_needs_the_window_and_an_unfinished_day():
    now = utc(2026, 10, 10, 14, 0)
    assert days.should_run(now, done_days=set()) == (True, date(2026, 10, 10))
    assert days.should_run(now, done_days={date(2026, 10, 10)})[0] is False
    assert days.should_run(utc(2026, 10, 10, 12, 0), done_days=set())[0] is False


def test_the_run_day_is_the_eastern_date_even_when_utc_has_rolled_over():
    # never true in the window, but the date must come from ET, not the UTC calendar
    assert days.eastern(utc(2026, 10, 11, 2, 0)).date() == date(2026, 10, 10)


def test_should_run_reports_the_eastern_date_even_outside_the_window():
    assert days.should_run(utc(2026, 10, 11, 2, 0), done_days=set()) == (False, date(2026, 10, 10))
