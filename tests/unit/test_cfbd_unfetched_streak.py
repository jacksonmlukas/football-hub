"""The unfetched streak and its escalation, and the week-1 anchor (#424).

A source that is unfetched is a warning by design (an optional source must not fail a
Sunday), and the same warning repeated for four weeks was not read. `record_run` carries the
streak's start forward and says when it has become more than a week, inside the regular
season. The clock is passed in on every call; nothing here reads the real one.
"""
import json
from datetime import UTC, datetime

import pytest

from hub.fetch import cfbd


@pytest.fixture
def env(monkeypatch):
    values: dict[str, str] = {}
    monkeypatch.setattr(cfbd, "_env", lambda: values)
    return values


@pytest.fixture(autouse=True)
def _anchor(env):
    """The in-season window is derived from week 1 (#433), so every run here has an anchor:
    week 1's first game on 2026-09-03, its week opening Tuesday 2026-09-01."""
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-03"


@pytest.fixture
def quota(tmp_path):
    return tmp_path / "quota.json"


def _ran(path, quota, on: tuple[int, int], *, fetched=False):
    """One run on a pinned day."""
    when = datetime(2026, on[0], on[1], 12, tzinfo=UTC)
    return cfbd.record_run(
        2026, 3 if fetched else None, rows={"games": 1} if fetched else None,
        why=None if fetched else "nothing was fetched: CFB_WEEK_ONE is not set",
        path=path, quota_path=quota, now=when)


def test_an_unfetched_streak_escalates_only_after_more_than_a_week_in_season(tmp_path, quota):
    p = tmp_path / "cfbd.json"
    first = _ran(p, quota, (9, 9))
    assert first["unfetched_since"] == "2026-09-09" and first["escalate"] is False
    # Exactly seven days is still a warning; the eighth day is the first escalation.
    assert _ran(p, quota, (9, 16))["escalate"] is False
    got = _ran(p, quota, (9, 17))
    assert got["unfetched_since"] == "2026-09-09", "the streak start must be carried forward"
    assert got["escalate"] is True


def test_a_fetched_run_resets_the_streak(tmp_path, quota):
    """The control for the carry-forward: if a success did not clear it, one old miss would
    escalate every later miss forever."""
    p = tmp_path / "cfbd.json"
    _ran(p, quota, (9, 9))
    assert _ran(p, quota, (9, 30), fetched=True)["unfetched_since"] is None
    got = _ran(p, quota, (10, 7))
    assert got["unfetched_since"] == "2026-10-07" and got["escalate"] is False


def test_the_offseason_and_postseason_never_escalate(tmp_path, quota):
    """Past week 15 `configured_week` fetches nothing by design; a streak there is not a fault."""
    p = tmp_path / "cfbd.json"
    _ran(p, quota, (11, 1))
    assert _ran(p, quota, (12, 20))["escalate"] is False, "postseason"
    q = tmp_path / "summer.json"
    _ran(q, quota, (6, 1))
    assert _ran(q, quota, (7, 20))["escalate"] is False, "summer"
    assert cfbd.season_in_progress(datetime(2026, 10, 6, tzinfo=UTC).date())


def test_a_record_from_before_the_field_existed_counts_from_its_own_date(tmp_path, quota):
    """The committed record that motivated the ticket has no `unfetched_since`; it must
    not restart the count, or the four weeks already spent would be forgiven."""
    p = tmp_path / "cfbd.json"
    p.write_text(json.dumps({"fetched": False, "generated_at": "2026-09-09T11:00:00+00:00"}))
    got = _ran(p, quota, (10, 6))
    assert got["unfetched_since"] == "2026-09-09" and got["escalate"] is True


def test_an_unreadable_previous_record_starts_the_count_today(tmp_path, quota):
    p = tmp_path / "cfbd.json"
    p.write_text("{not json")
    got = _ran(p, quota, (10, 6))
    assert got["unfetched_since"] == "2026-10-06" and got["escalate"] is False


def test_the_anchor_is_week_ones_first_game_not_week_zeros(env):
    """ESPN numbers the 08-29 Week-0 games as week 1 too, so the anchor is 09-03 and the
    count matches ESPN's at 10-10 (week 6); anchored on 08-29 it says 7."""
    at = datetime(2026, 10, 10, 12, tzinfo=UTC)
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-03"
    assert cfbd.configured_week(at).week == 6
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-08-29"
    assert cfbd.configured_week(at).week == 7


def test_a_summer_streak_does_not_escalate_before_week_one_opens(tmp_path, quota):
    """Rule-18 control for #433. Summer records are unfetched by design, so on the first runs
    of the season the carried streak is months old. Counted from `unfetched_since` that is
    red from Aug 24 (the old hand-kept window) until week 1 opens; counted from the season's
    start it is green here and turns red only a week after week 1 opened (Tue 09-01)."""
    p = tmp_path / "cfbd.json"
    _ran(p, quota, (6, 1))
    for day in ((8, 24), (8, 28), (8, 31)):
        got = _ran(p, quota, day)
        assert got["unfetched_since"] == "2026-06-01", "the record still tells the truth"
        assert got["escalate"] is False, f"false red on {day}, before week 1 opened"
    assert _ran(p, quota, (9, 8))["escalate"] is False, "exactly seven days into the season"
    assert _ran(p, quota, (9, 9))["escalate"] is True, "the first day past a week of season"


def test_the_window_is_derived_from_the_week_one_anchor(env):
    """One anchor, not two calendars: moving CFB_WEEK_ONE moves the window with it."""
    from datetime import date
    assert cfbd.season_window() == (date(2026, 9, 1), date(2026, 12, 14))
    env[cfbd.CFB_WEEK_ONE_ENV] = "2027-09-02"
    assert cfbd.season_window() == (date(2027, 8, 31), date(2027, 12, 13))
    assert not cfbd.season_in_progress(date(2027, 8, 30))
    assert cfbd.season_in_progress(date(2027, 8, 31))
    env[cfbd.CFB_WEEK_ONE_ENV] = ""
    assert cfbd.season_window() is None and not cfbd.season_in_progress(date(2026, 10, 6))
