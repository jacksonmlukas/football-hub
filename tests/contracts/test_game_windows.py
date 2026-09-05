"""The crons that refresh the scores, held against the clock the games are played on.

A cron fires in UTC. Games are played on Eastern time, and Eastern is UTC-4 for half the
season and UTC-5 for the other half. The deploy and watchdog windows were written as fixed
UTC hours annotated as Eastern, so the whole window slid an hour earlier the moment the
season crossed out of daylight saving -- gaining dead time before the first kickoff and
losing an hour of tail, which is where a Sunday night game running past midnight lives.

This repo has already paid for the fixed-offset assumption once: `hub.fetch.odds._game_date`
records an odds snapshot that lost every primetime game to it, and `hub.schedule._kickoff`
converts rather than offsetting for the same reason. Prose did not hold it the first two
times, so this is the third statement of the rule and the first one a test can fail.

Three properties, and the second and third are about the monitor rather than the deploy:

  1. Every Eastern hour a game can be in progress is covered, in both daylight-saving states.
  2. The watchdog never watches an hour the deploy does not refresh -- a monitor wider than
     the thing it monitors reports the gap between them as an outage.
  3. The watchdog never runs in the *opening* hour of a window. The overlay it checks is
     refreshed on the same cadence, so the first tick of a window looks at an artifact last
     written before the window opened, and files an incident by construction -- twice a week,
     for eighteen weeks.
"""
import datetime as dt
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
ET = ZoneInfo("America/New_York")
UTC = dt.UTC

# A Sunday in daylight saving and a Sunday in standard time, both inside the NFL season.
# The rule has to hold on both, which is the whole point -- week 1 and week 15 are not on
# the same offset and a cron does not know that.
EDT_WEEK = dt.date(2026, 10, 11)
EST_WEEK = dt.date(2026, 12, 13)

# The hours a game can be in progress, in *Eastern*, as (weekday, hour) with Sunday=0 to
# match cron's own day-of-week numbering.
#
# Sunday runs noon to the end of the night game; Saturday is college, which starts at noon
# and whose marquee kickoffs are at 19:30 and 20:00; Thursday and Monday are the standalone
# night games. Each night block runs into the small hours of the next day because a game
# that kicks at 20:15 is still being played at 23:45 and can reach midnight.
DAYTIME = tuple(range(12, 24))
NIGHT = tuple(range(20, 24))

GAME_HOURS: set[tuple[int, int]] = (
    {(0, h) for h in DAYTIME} | {(1, 0)}          # Sunday, into Monday morning
    | {(6, h) for h in DAYTIME} | {(0, 0)}        # Saturday college, into Sunday morning
    | {(4, h) for h in NIGHT} | {(5, 0)}          # Thursday night
    | {(1, h) for h in NIGHT} | {(2, 0)}          # Monday night
)


def _crons_in(text: str) -> list[str]:
    return re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', text, flags=re.MULTILINE)


def _crons(name: str) -> list[str]:
    """The cron expressions in one workflow, ignoring commented-out ones."""
    return _crons_in((WORKFLOWS / name).read_text())


def _hours(field: str) -> set[int]:
    """Expand a cron hour field: `16-23`, `0-5`, `3`, or a comma-separated mix."""
    out: set[int] = set()
    for part in field.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out |= set(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _slots(crons: list[str]) -> set[tuple[int, int]]:
    """(day-of-week, hour) in UTC for every hour these crons fire in.

    The minute field is not read. Every window cron here runs `*/10`, and what these
    properties are about is which hours are covered at all.
    """
    out: set[tuple[int, int]] = set()
    for c in crons:
        _minute, hour, dom, mon, dow = c.split()
        assert dom == "*" and mon == "*", f"{c!r}: a window cron names days of week only"
        for d in _hours(dow):
            out |= {(d, h) for h in _hours(hour)}
    return out


def _in_eastern(slots: set[tuple[int, int]], week_of: dt.date) -> set[tuple[int, int]]:
    """The same slots read on the Eastern clock, for the week containing `week_of`.

    Converted rather than offset by a constant -- which is the defect this file exists for.
    """
    sunday = week_of - dt.timedelta(days=(week_of.weekday() + 1) % 7)
    out = set()
    for day, hour in slots:
        moment = dt.datetime.combine(sunday + dt.timedelta(days=day),
                                     dt.time(hour), tzinfo=UTC).astimezone(ET)
        out.add(((moment.weekday() + 1) % 7, moment.hour))
    return out


# --- the premise, which a regex over YAML has to earn ------------------------

def test_the_scan_finds_the_crons_that_exist():
    """A parser matching nothing would make every assertion below vacuously true, which is
    the failure mode of a guard shaped like this one."""
    assert len(_crons("pages.yml")) >= 5
    assert len(_crons("watchdog.yml")) >= 3
    assert _slots(_crons("pages.yml")), "no hours parsed out of the deploy crons"


def test_a_commented_out_cron_is_not_counted_as_one():
    """The premise above only holds if the regex can tell a live block from a dormant one,
    and that is not hypothetical here: until 2026-09-04 both `watchdog.yml` and `ci.yml`
    carried a `SCHEDULE DISABLED` comment above a `schedule:` block that was in fact live.
    Had the block matched the comment, `_crons` would have returned nothing and every
    property in this file would have passed by having nothing to check."""
    live = '  schedule:\n    - cron: "*/10 17-23 * * 0"\n'
    dead = '  # schedule:\n  #   - cron: "*/10 17-23 * * 0"\n'
    assert _crons_in(live) == ["*/10 17-23 * * 0"]
    assert _crons_in(dead) == [], "a commented-out cron would be counted as coverage"


# --- 1. the window follows Eastern, in both states ---------------------------

@pytest.mark.parametrize("week_of,label", [(EDT_WEEK, "EDT"), (EST_WEEK, "EST")])
def test_every_hour_a_game_can_be_in_progress_is_refreshed(week_of, label):
    covered = _in_eastern(_slots(_crons("pages.yml")), week_of)
    missing = sorted(GAME_HOURS - covered)
    assert not missing, (
        f"under {label} these Eastern (day, hour) slots have no deploy: {missing}. A fixed "
        f"UTC window is right for one half of the season and wrong for the other.")


def test_the_window_is_not_merely_wide_enough_to_pass():
    """The opposite failure: covering Eastern by running all week. A cron that ran always
    would satisfy the test above and deploy a few thousand times a month to publish a
    scoreboard with nothing on it."""
    assert len(_slots(_crons("pages.yml"))) <= 60, "7 days x 24 hours is 168; stay a window"


# --- 2 and 3. the monitor, and what it must not report -----------------------

def test_the_watchdog_never_watches_an_hour_nothing_refreshes():
    deploy, watch = _slots(_crons("pages.yml")), _slots(_crons("watchdog.yml"))
    assert not (watch - deploy), (
        f"watched but never refreshed: {sorted(watch - deploy)}. The gap between a monitor "
        f"and the thing it monitors is reported as an outage.")


def test_a_window_opening_does_not_by_itself_file_an_incident():
    """The false incident, twice a week for eighteen weeks. Staleness is measured against
    the last refresh, and the refresh runs on the same cadence -- so the first tick of a
    window looks at an artifact written before the window opened. The fix is structural:
    the watchdog starts an hour into a window it already knows is running."""
    deploy, watch = _slots(_crons("pages.yml")), _slots(_crons("watchdog.yml"))
    opening = {(d, h) for d, h in deploy
               if ((d, h - 1) if h else ((d - 1) % 7, 23)) not in deploy}
    assert not (watch & opening), (
        f"the watchdog runs in the opening hour of a window: {sorted(watch & opening)}. "
        f"Nothing has refreshed the overlay since the previous window closed.")


def test_the_watchdog_still_covers_the_windows_it_is_for():
    """The guard above is satisfiable by a watchdog that never runs. It has to keep
    watching -- the whole window bar its first hour."""
    deploy, watch = _slots(_crons("pages.yml")), _slots(_crons("watchdog.yml"))
    opening = {(d, h) for d, h in deploy
               if ((d, h - 1) if h else ((d - 1) % 7, 23)) not in deploy}
    assert watch == deploy - opening, (
        f"unwatched: {sorted(deploy - opening - watch)}; watched and should not be: "
        f"{sorted(watch - (deploy - opening))}")


# --- the recovery branch closes its own incidents ----------------------------

def test_the_watchdog_marks_the_incidents_it_files_and_closes_only_those():
    """`gh issue close` ran over every open issue carrying the `incident` label, which
    includes one a human filed -- closed with a comment about the heartbeat. The label says
    what an issue is about; it does not say who filed it."""
    text = (WORKFLOWS / "watchdog.yml").read_text()
    marker = re.search(r"<!--\s*(watchdog-[a-z-]+)\s*-->", text)
    assert marker, "the incident body carries no machine marker to recognise it by"
    tag = marker.group(1)
    assert text.count(tag) >= 2, (
        f"{tag} is written into the body but never read back; the close branch still has "
        f"no way to tell its own incident from a human's")
    close = text.split("Close the incident on recovery", 1)[1]
    assert tag in close, "the recovery branch does not filter on the marker"
