"""The crons that start the refresher, held against the clock the games are played on.

The window schedule moved from `pages.yml` to `live.yml` in issue #91 -- the ten-minute
deploy cron was measured being delivered every ~106 minutes, so the cron now starts a
long-lived looping job instead of a deploy. These properties moved with it unchanged, plus
two the move itself needs: the schedule must exist in exactly one workflow, and every cron
must say which league its window is for.

A cron fires in UTC. Games are played on Eastern time, and Eastern is UTC-4 for half the
season and UTC-5 for the other half. The deploy and watchdog windows were written as fixed
UTC hours annotated as Eastern, so the whole window slid an hour earlier the moment the
season crossed out of daylight saving -- gaining dead time before the first kickoff and
losing an hour of tail, which is where a Sunday night game running past midnight lives.

This repo has already paid for the fixed-offset assumption once: `hub.fetch.odds._game_date`
records an odds snapshot that lost every primetime game to it, and `hub.schedule._kickoff`
converts rather than offsetting for the same reason. Prose did not hold it the first two
times, so this is the third statement of the rule and the first one a test can fail.

Five properties. The second and third are about the monitor rather than the refresher, and
the last two are what the move to `live.yml` has to keep true:

  1. Every Eastern hour a game can be in progress is covered, in both daylight-saving states.
  2. The watchdog never watches an hour nothing refreshes -- a monitor wider than the thing
     it monitors reports the gap between them as an outage.
  3. The watchdog never runs in the *opening* hour of a window. The overlay it checks is
     refreshed on the same cadence, so the first tick of a window looks at an artifact last
     written before the window opened, and files an incident by construction -- twice a week,
     for eighteen weeks.
  4. The window schedule exists in exactly one workflow. Two of them is two deploy sources
     for one window, against a deployment rate nobody has measured a ceiling for.
  5. Every window cron says which league it is for. The artifact is single-league, the league
     follows the window -- college Saturday, professional Sunday -- and a cron edited without
     its mapping would publish the wrong board for a whole day, which is what happened on
     2026-09-05: professional pre-season games on the page while the college board was the
     one with games in progress.
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


def _hours(field: str, every: int = 24) -> set[int]:
    """Expand a cron hour field: `16-23`, `0-5`, `3`, `*`, or a comma-separated mix.

    `every` is what `*` means -- 24 for an hour field, 7 for a day-of-week one. A window cron
    never writes `*` in either, but the workflows this file compares the window against do,
    and reading `*` as "no hours at all" would let a daily job hide inside the window.
    """
    if field == "*":
        return set(range(every))
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
    assert len(_crons("live.yml")) >= 5
    assert len(_crons("watchdog.yml")) >= 3
    assert _slots(_crons("live.yml")), "no hours parsed out of the window crons"


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


# --- 4. one schedule, in one workflow ----------------------------------------

def test_the_window_schedule_lives_in_exactly_one_workflow():
    """`pages.yml` carried these crons until issue #91 and now carries none.

    Leaving them in both would have been the safe-looking half-move: two deploy sources for
    one window, twice the deploy rate, and a second definition of the window to drift. The
    deploy workflow keeps its push, its `workflow_run` and its dispatch -- all of which are
    events rather than clocks.
    """
    assert _crons("pages.yml") == [], (
        f"the deploy workflow has window crons again: {_crons('pages.yml')}. The window is "
        f"`live.yml`'s, and `pages.yml` deploys when something asks it to.")
    others = {name: _crons(name) for name in ("slate.yml", "ci.yml")}
    for name, crons in others.items():
        assert not (_slots(crons) & _slots(_crons("live.yml"))), (
            f"{name} also fires inside the game window ({crons}); a second refresher is a "
            f"second answer to the same question")


# --- 5. the league follows the window ----------------------------------------

def test_every_window_cron_says_which_league_it_is_for():
    """The artifact is single-league and the crons are what know which league is playing.

    A mapping keyed on the cron string is only as good as its coverage: an edited cron falls
    through to whatever the default is, and publishes the wrong board for a whole Saturday
    with nothing failing. So the mapping is required to name every cron, exactly, and the
    workflow's own fall-through is required to be an error rather than a guess.
    """
    text = (WORKFLOWS / "live.yml").read_text()
    step = text.split("Which board this window is for", 1)
    assert len(step) == 2, "live.yml no longer chooses a league; the overlay would be nfl always"
    mapped = dict(re.findall(r'^\s*"([^"]+)"\)\s*LEAGUE=(\w+)', step[1], flags=re.MULTILINE))

    missing = [c for c in _crons("live.yml") if c not in mapped]
    assert not missing, (
        f"these window crons are not in the league mapping: {missing}. They would fall "
        f"through to the error branch and the window would go unrefreshed.")
    assert set(mapped.values()) <= {"cfb", "nfl"}, mapped

    # Saturday is college and every other window is professional -- read off the crons
    # themselves rather than restated, so this cannot agree with a mapping that has drifted
    # from the schedule. A window that touches Eastern Saturday at all is the college one,
    # including the primetime block that spills past midnight into Sunday morning: those are
    # Saturday's games still being played, which is exactly the hour the overlay held
    # professional pre-season games on 2026-09-05.
    for week, label in ((EDT_WEEK, "EDT"), (EST_WEEK, "EST")):
        for cron, league in mapped.items():
            if cron not in _crons("live.yml"):
                continue
            eastern_days = {d for d, _h in _in_eastern(_slots([cron]), week)}
            expected = "cfb" if 6 in eastern_days else "nfl"
            assert league == expected, (
                f"under {label} {cron!r} covers Eastern day(s) {sorted(eastern_days)} and is "
                f"mapped to {league!r}; a window touching Saturday is the college board and "
                f"every other window is the professional one")


def test_the_league_the_window_chose_reaches_the_deploy_that_publishes_it():
    """The half of property 5 that is easy to leave decorative.

    The loop and the deploy are separate runs, so the loop's refresh is a check and the
    deploy's is the one that writes the file the page serves. A loop that chose `cfb` and
    dispatched a deploy that refreshed the default board would publish an empty professional
    scoreboard over a Saturday of college scores, five minutes after getting it right --
    and everything would look like it was working.
    """
    live, pages = (WORKFLOWS / "live.yml").read_text(), (WORKFLOWS / "pages.yml").read_text()
    assert "-f league=" in live, (
        "the loop does not tell the deploy which board it is refreshing, so its own league "
        "choice never reaches the published file")
    assert re.search(r"inputs:\s*\n\s*league:", pages), (
        "`pages.yml` does not accept a league, so `-f league=` would fail the dispatch")
    assert "--league" in pages, (
        "the deploy refreshes the overlay without saying which board, which is the default "
        "one whatever the window is")


# --- 1. the window follows Eastern, in both states ---------------------------

@pytest.mark.parametrize("week_of,label", [(EDT_WEEK, "EDT"), (EST_WEEK, "EST")])
def test_every_hour_a_game_can_be_in_progress_is_refreshed(week_of, label):
    covered = _in_eastern(_slots(_crons("live.yml")), week_of)
    missing = sorted(GAME_HOURS - covered)
    assert not missing, (
        f"under {label} these Eastern (day, hour) slots have no deploy: {missing}. A fixed "
        f"UTC window is right for one half of the season and wrong for the other.")


def test_the_window_is_not_merely_wide_enough_to_pass():
    """The opposite failure: covering Eastern by running all week. A cron that ran always
    would satisfy the test above and deploy a few thousand times a month to publish a
    scoreboard with nothing on it."""
    assert len(_slots(_crons("live.yml"))) <= 60, "7 days x 24 hours is 168; stay a window"


# --- 2 and 3. the monitor, and what it must not report -----------------------

def test_the_watchdog_never_watches_an_hour_nothing_refreshes():
    deploy, watch = _slots(_crons("live.yml")), _slots(_crons("watchdog.yml"))
    assert not (watch - deploy), (
        f"watched but never refreshed: {sorted(watch - deploy)}. The gap between a monitor "
        f"and the thing it monitors is reported as an outage.")


def test_a_window_opening_does_not_by_itself_file_an_incident():
    """The false incident, twice a week for eighteen weeks. Staleness is measured against
    the last refresh, and the refresh runs on the same cadence -- so the first tick of a
    window looks at an artifact written before the window opened. The fix is structural:
    the watchdog starts an hour into a window it already knows is running."""
    deploy, watch = _slots(_crons("live.yml")), _slots(_crons("watchdog.yml"))
    opening = {(d, h) for d, h in deploy
               if ((d, h - 1) if h else ((d - 1) % 7, 23)) not in deploy}
    assert not (watch & opening), (
        f"the watchdog runs in the opening hour of a window: {sorted(watch & opening)}. "
        f"Nothing has refreshed the overlay since the previous window closed.")


def test_the_watchdog_still_covers_the_windows_it_is_for():
    """The guard above is satisfiable by a watchdog that never runs. It has to keep
    watching -- the whole window bar its first hour."""
    deploy, watch = _slots(_crons("live.yml")), _slots(_crons("watchdog.yml"))
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
