"""Every game window has backup starts ahead of its first kickoff (#470, from #416).

On 2026-10-13 the 00:15Z kickoff had no `live` run and no `watchdog` check, because GitHub never
delivered the scheduled start (#391); #468 measured the starts it does deliver landing 4-8.5
hours late. The window crons are all inside the window's own hours, so one bad stretch of the
scheduler takes every request at once. `live.yml` therefore carries backup crons an hour
ahead of each window, and this holds that a window cannot be left with a single request before
its first kickoff, and that a duplicate start is harmless because one group serialises them.

The rule is read off the file's crons and the Eastern kickoff times, in both daylight-saving
states, and the planted control removes the backups and requires the same rule to fail -- a
check that cannot go red on the thing it guards is not a check.
"""
import datetime as dt
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
ET = ZoneInfo("America/New_York")
EDT_WEEK = dt.date(2026, 10, 11)   # a Sunday in daylight saving
EST_WEEK = dt.date(2026, 12, 13)   # a Sunday in standard time

# The first kickoff each window is for, as (cron day-of-week, Eastern hour, minute). Sunday
# NFL at 13:00, Saturday college at noon, the Thursday and Monday night games at 20:00 (the
# 20:15 kickoffs, rounded to the hour the window opens for). Sunday and Saturday *night* are
# the tails of the daytime windows and have no kickoff of their own.
FIRST_KICKOFFS = {
    "Sunday daytime": (0, 13, 0),
    "Saturday daytime": (6, 12, 0),
    "Thursday night": (4, 20, 0),
    "Monday night": (1, 20, 0),
}
MIN_ENTRIES = 2                       # distinct cron entries that must fire before kickoff
LEAD = dt.timedelta(minutes=10)       # a start this close to kickoff is no start
REACH = dt.timedelta(hours=3)         # and one this far ahead is not about this kickoff


def _crons_in(text: str) -> list[str]:
    return re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', text, flags=re.MULTILINE)


def _expand(field: str, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part == "*":
            lo, top = 0, hi
        elif "-" in part:
            lo, top = (int(x) for x in part.split("-"))
        else:
            lo = top = int(part)
            if step != 1:
                top = hi
        out |= set(range(lo, top + 1, step))
    return out


def _fires(cron: str, since: dt.datetime, until: dt.datetime) -> list[dt.datetime]:
    """Every UTC minute `cron` fires in [since, until]."""
    minute, hour, _dom, _mon, dow = cron.split()
    minutes, hours = _expand(minute, 59), _expand(hour, 23)
    dows = {d % 7 for d in _expand(dow, 7)}
    out, t = [], since.replace(minute=0, second=0, microsecond=0)
    while t <= until:
        if (t.weekday() + 1) % 7 in dows and t.hour in hours:
            out += [t.replace(minute=m) for m in sorted(minutes)
                    if since <= t.replace(minute=m) <= until]
        t += dt.timedelta(hours=1)
    return out


def _kickoff_utc(week_of: dt.date, dow: int, hour: int, minute: int) -> dt.datetime:
    sunday = week_of - dt.timedelta(days=(week_of.weekday() + 1) % 7)
    local = dt.datetime.combine(sunday + dt.timedelta(days=dow), dt.time(hour, minute), tzinfo=ET)
    return local.astimezone(dt.UTC)


def shortfalls(live_text: str) -> list[str]:
    """Window slots (one per window and daylight-saving state) with fewer than MIN_ENTRIES
    distinct cron entries firing a usable start ahead of the first kickoff."""
    crons = _crons_in(live_text)
    out = []
    for week_of, state in ((EDT_WEEK, "EDT"), (EST_WEEK, "EST")):
        for name, (dow, hour, minute) in FIRST_KICKOFFS.items():
            kick = _kickoff_utc(week_of, dow, hour, minute)
            usable = {c for c in set(crons) if _fires(c, kick - REACH, kick - LEAD)}
            if len(usable) < MIN_ENTRIES:
                out.append(f"{name} under {state} (kickoff {kick:%a %H:%MZ}): "
                           f"{len(usable)} entries ({sorted(usable)})")
    return out


def _without_backups(text: str) -> str:
    """The same workflow with every cron that is not a `*/10` window cron removed."""
    return "\n".join(line for line in text.splitlines()
                     if not re.match(r'\s*-\s*cron:\s*"(?!\*/10)', line))


def test_every_window_has_two_starts_requested_before_its_first_kickoff():
    got = shortfalls((WORKFLOWS / "live.yml").read_text())
    assert not got, (
        "these windows have a single cron entry ahead of kickoff, so one dropped delivery "
        f"leaves them with no refresher (#416, #391, #468): {got}")


def test_the_backups_are_not_the_whole_schedule():
    """Premise of the planted control: there are backups to remove, and the primaries stay."""
    text = (WORKFLOWS / "live.yml").read_text()
    stripped = _crons_in(_without_backups(text))
    assert len(_crons_in(text)) > len(stripped) >= 5
    assert all(c.startswith("*/10") for c in stripped)


def test_planted_a_copy_without_the_backups_goes_red():
    """Rule 18: the assertion above must be able to fail on exactly this defect."""
    planted = _without_backups((WORKFLOWS / "live.yml").read_text())
    got = shortfalls(planted)
    assert len(got) >= len(FIRST_KICKOFFS), (
        f"the rule passes with the backup crons removed; it guards nothing: {got}")


def test_a_duplicate_start_meets_one_named_group():
    """The backups add starts to a window that already has several. What makes the extra ones
    harmless is a single constant group, so two starts are never two loops."""
    doc = yaml.safe_load((WORKFLOWS / "live.yml").read_text())
    group = doc["concurrency"]["group"]
    assert group and "${{" not in group, (
        "a group keyed on the run or the event is a group per start, i.e. no group")
    assert "cancel-in-progress" in doc["concurrency"], "the duplicate's fate must be stated"
    assert "concurrency" not in doc["jobs"]["refresh"], (
        "a job-level group would override the workflow's and split the starts")


def test_the_workflow_says_why_the_backups_exist():
    text = (WORKFLOWS / "live.yml").read_text()
    for issue in ("#416", "#391", "#468", "#470"):
        assert issue in text, f"live.yml does not cite {issue}"


def test_the_check_step_still_runs_in_a_replaced_run():
    """A duplicate start replaces the running loop. Its check must still happen, or the
    replaced run is a hole in the window the look-back counts."""
    doc = yaml.safe_load((WORKFLOWS / "live.yml").read_text())
    step = next(s for s in doc["jobs"]["refresh"]["steps"] if s.get("name") == "Watchdog check")
    assert step["if"] == "always()"
