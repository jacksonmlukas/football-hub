"""`slate.yml` polls the odds market on exactly the crons `hub.fetch.odds.POLL_SCHEDULE` names.

The module decides what a poll costs and what the cadence is for; the workflow decides when a
run starts. Those are two statements of one schedule, and a cron edited without the constant
-- or the constant without the cron -- is exactly the drift #370 (S12) found: two crons in
`slate.yml`'s comment claiming "about 9 a month" while nothing in `src/` recorded the number a
reader could check that against. `test_bigten_schedule.py` is the same shape, one workflow
smaller, and this file follows it line for line.
"""
from __future__ import annotations

import re
from pathlib import Path

from hub.fetch import odds

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "slate.yml"

_CRON = re.compile(r'^\s*-\s*cron:\s*"([^"]+)"', re.M)


def _crons() -> list[str]:
    """Live crons only: a commented-out block is not a schedule."""
    return [" ".join(c.split()) for c in _CRON.findall(WORKFLOW.read_text())]


def test_the_workflow_fires_on_exactly_the_modules_polls():
    assert sorted(_crons()) == sorted(p.cron() for p in odds.POLL_SCHEDULE), (
        f"slate.yml's crons {_crons()} and hub.fetch.odds.POLL_SCHEDULE "
        f"{[p.cron() for p in odds.POLL_SCHEDULE]} disagree; a run would poll on a schedule "
        f"the module does not account for, or a poll the module names would never fire")


def test_the_scan_sees_the_crons():
    """Non-vacuity, the same premise `test_bigten_schedule.py` asserts of its own scan: a
    regex that matched nothing would make the comparison above two empty lists and pass."""
    assert len(_crons()) >= 4


def test_the_schedule_is_four_to_six_polls_a_week():
    """#370's acceptance range, checked against the constant everything else here reads."""
    assert 4 <= len(odds.POLL_SCHEDULE) <= 6


def test_thursday_and_a_sunday_morning_capture_are_both_named():
    """The two #370 names specifically, distinct from the pre-existing Wednesday/Saturday
    pair: cron weekday 4 is Thursday, weekday 0 is Sunday, and "morning" here means before
    the early kickoffs -- `hub.fetch.odds`'s own comment puts those at 17:00 UTC."""
    by_weekday = {p.weekday: p for p in odds.POLL_SCHEDULE}
    assert 4 in by_weekday, "no Thursday poll"
    sundays = [p for p in odds.POLL_SCHEDULE if p.weekday == 0]
    assert sundays and all(p.hour < 17 for p in sundays), (
        "no Sunday poll before the early kickoffs")


def test_only_the_original_two_crons_run_the_full_slate():
    """The split #370's design rests on: the other four polls must not also refetch nflverse,
    CFBD and the roster four extra times a week. `refresh`'s `if:` is the enforcement; this
    reads it back rather than trusting the comment beside it."""
    text = WORKFLOW.read_text()
    job = text.split("\n  refresh:", 1)[1].split("\n  poll_odds:", 1)[0]
    m = re.search(r"if:\s*(.+)", job)
    assert m, "the refresh job no longer conditions itself on which cron fired"
    for cron in ("0 11 * * 3", "0 14 * * 6"):
        assert cron in m.group(1), f"{cron!r} is missing from refresh's condition"
    for poll in odds.POLL_SCHEDULE:
        if poll.cron() not in ("0 11 * * 3", "0 14 * * 6"):
            assert poll.cron() not in m.group(1), (
                f"{poll.cron()!r} ({poll.name}) would also trigger the full slate")


def test_poll_odds_spends_nothing_but_an_odds_credit():
    """The job that runs on the other four: no `make slate`, no nflverse, no CFBD, no roster
    -- just the one CLI call #370 is about."""
    text = WORKFLOW.read_text()
    job = text.split("\n  poll_odds:", 1)[1]
    assert "hub.fetch.odds --snapshot" in job
    for forbidden in ("make slate", "hub.fetch.nflverse", "hub.fetch.cfbd", "hub.season.roster",
                     "hub.publish"):
        assert forbidden not in job, f"poll_odds runs {forbidden!r}, which is more than a poll"


def test_every_scheduled_job_has_a_timeout():
    """`test_scheduled_jobs_are_bounded.py` already asserts this repo-wide; restated here so
    a reviewer of this file alone sees the new job is covered rather than having to trust
    the repo-wide scan caught it."""
    text = WORKFLOW.read_text()
    for job in ("refresh", "poll_odds"):
        body = text.split(f"\n  {job}:", 1)[1].split("\n  poll_odds:", 1)[0] \
            if job == "refresh" else text.split(f"\n  {job}:", 1)[1]
        assert re.search(r"timeout-minutes:\s*\d+", body), f"{job} has no timeout-minutes"
