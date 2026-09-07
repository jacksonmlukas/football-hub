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

Six properties. The second, third and sixth are about the monitor rather than the refresher,
and the fourth and fifth are what the move to `live.yml` has to keep true:

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
  6. A watchdog *run* knows whether it is still inside the window it was scheduled for, and
     says so rather than measuring when it is not.

Property 6 is the one properties 2 and 3 cannot reach, and issue #208 is what it costs. Those
two hold the *crons* inside the windows; nothing held the *runs*. A cron is a request for a
start and GitHub drops scheduled workflows under load -- `live.yml` records these starts
arriving 97-126 minutes apart -- so on 2026-09-07 the run behind `*/10 0-5  * * 1` (Monday
00:00-05:59 UTC) executed at 09:21Z, measured a heartbeat that nothing was refreshing because
its window had closed, and filed issue #193. Every cron property in this file passed that day
and was passing while it happened.

Its preseason half is the same defect one step earlier: before the season's first game there
is no window at all, so there is nothing to be late for and nothing to measure. #193 could
not clear because a healthy check needs a window, the first window needs the first game, and
the first game was two days out.
"""
import datetime as dt
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

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


# --- 6. the run, not the cron ------------------------------------------------
#
# Everything above is about where the crons are written. A cron is a request for a start,
# and what this section adds is about where a run actually landed.

WINDOW_STATE = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "window_state.sh"
WATCHDOG = WORKFLOWS / "watchdog.yml"

# The real dates from #193 and #208. The season's first game is the Wednesday; the incident
# was filed on the Monday before it by a run scheduled for 00:00-05:59Z that executed at
# 09:21Z; the next Monday-adjacent cron was the Tuesday.
FIRST_GAME = "2026-09-09"
FILED_AT = dt.datetime(2026, 9, 7, 9, 21, tzinfo=UTC)
NEXT_CRON_AT = dt.datetime(2026, 9, 8, 3, 0, tzinfo=UTC)
# A Monday in the season proper, so the preseason branch is not what is being read.
IN_SEASON = dt.datetime(2026, 10, 12, tzinfo=UTC)


def _state(cron: str, when: dt.datetime, season_opens: str = FIRST_GAME) -> str:
    got = subprocess.run(
        [str(WINDOW_STATE), cron, season_opens, str(int(when.timestamp()))],
        capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def test_a_run_that_executes_inside_its_window_measures():
    """The case that has to keep working. Suppressing the false alarm is easy if the check is
    allowed to stop checking, and that is the failure this file's subject has had twice."""
    got = _state("*/10 0-5  * * 1", IN_SEASON.replace(hour=4, minute=12))
    assert got.startswith("inside "), got
    assert int(got.split()[1]) == 108 * 60, "the window runs to 05:59:59, so 04:12 has 108m"


def test_a_run_delayed_past_its_window_says_so_rather_than_measuring():
    """**The load-bearing case, and the one #208 is about.**

    Scheduled inside the window by a cron every property above holds in place, executed after
    it closed. There is nothing to measure: the `live` loop has stopped, so the published
    heartbeat is expected to age and its age is not evidence of anything.
    """
    got = _state("*/10 0-5  * * 1", IN_SEASON.replace(hour=9, minute=21))
    assert got.startswith("late "), got
    assert int(got.split()[1]) == 3 * 3600 + 21 * 60, "09:21 is 3h21m after a 06:00 close"


def test_the_run_that_filed_193_is_not_a_run_that_measures():
    """The incident itself, replayed on both of its counts.

    On the day, the preseason branch is what catches it -- the first game was two days out.
    Given a season already under way, the delay is: same cron, same clock, still not a
    measurement. Either alone is enough to keep it from filing, which is why both are asked.
    """
    assert _state("*/10 0-5  * * 1", FILED_AT) == f"preseason {FIRST_GAME}"
    assert _state("*/10 0-5  * * 1", FILED_AT, "2026-09-01") == "late 12060"


def test_before_the_first_game_a_run_inside_its_window_still_has_nothing_to_watch():
    """The preseason case, which is not the delayed one and is the one most easily skipped.

    This run is exactly where it was asked to be -- the Tuesday cron, executing in the middle
    of its own window. The crons fire every week of the year, so being inside a window says
    nothing about a game being played inside it. This is the run that clears #193.
    """
    assert _state("*/10 1-5  * * 2", NEXT_CRON_AT) == f"preseason {FIRST_GAME}"


def test_a_season_date_already_past_suppresses_nothing():
    """Which way the constant rots. `SEASON_OPENS` is a date somebody has to change each
    August, so the question is what a stale one costs: an old date means every run measures,
    which is the behaviour without this check at all. A monitor that goes quiet when nobody
    updates it is the other direction and is not reachable here."""
    assert _state("*/10 1-5  * * 2", NEXT_CRON_AT, "2026-09-01").startswith("inside ")
    assert _state("*/10 1-5  * * 2", NEXT_CRON_AT, "").startswith("inside ")


def test_a_cron_this_check_cannot_read_measures_rather_than_going_quiet():
    """A step in the hour field is not a window shape this repo writes, and widening it to
    "every hour" would report a run as inside a window it was never scheduled in. Refused --
    and refused *towards* measuring, because the workflow treats an unreadable cron as a
    reason to check rather than as a reason to stay silent."""
    assert _state("*/10 */2 * * 1", IN_SEASON.replace(hour=4)) == "unparseable"
    assert _state("*/10 0-5 17 * 1", IN_SEASON.replace(hour=4)) == "unparseable"
    assert _state("", IN_SEASON.replace(hour=9)) == "unscheduled"


def test_a_run_that_is_merely_early_is_not_reported_as_a_week_late():
    """Walking backwards from a weekly cron always finds last week's window. GitHub delays
    rather than advances, so this is unreachable in production and is here because "six days
    late" is the reading a one-directional search would have printed for it."""
    got = _state("*/10 17-23 * * 0", dt.datetime(2026, 10, 11, 16, 30, tzinfo=UTC))
    assert got == "early 1800", got


def test_every_watchdog_cron_can_be_read_by_the_check_that_gates_on_it():
    """The premise. A check answering `unparseable` for every real cron would be one that
    never suppresses anything, and every assertion above would still pass."""
    for cron in _crons("watchdog.yml"):
        _minute, hour, _dom, _mon, dow = cron.split()
        day, at = min(_hours(dow)), min(_hours(hour)) + 1
        # The Sunday of the in-season week above, plus the cron's own day and hour.
        sunday = IN_SEASON - dt.timedelta(days=(IN_SEASON.weekday() + 1) % 7)
        when = (sunday + dt.timedelta(days=day)).replace(hour=at)
        assert _state(cron, when).startswith("inside "), (
            f"{cron!r} is not readable as a window by window_state.sh; the watchdog would "
            f"measure every run without ever establishing where it landed")


# --- 6, in the workflow: what the run does with the answer --------------------


def _run_step(name: str, **expressions: str) -> dict[str, str]:
    """Run one step's shell out of `watchdog.yml`, with its `${{ }}` expressions bound.

    The step is executed rather than grepped for. A pair of greps that finds "preseason" and
    finds "closes=true" in the same file does not say the first produces the second, and the
    workflow schedule check `test_guards_are_load_bearing.py` was written for was built
    exactly that way.
    """
    steps = yaml.safe_load(WATCHDOG.read_text())["jobs"]["heartbeat"]["steps"]
    script = next(s["run"] for s in steps if s.get("name") == name)

    def bind(m: re.Match) -> str:
        key = m.group(1).strip()
        assert key in expressions, f"{name!r} reads {key!r}, which this test does not bind"
        return expressions[key]

    script = re.sub(r"\$\{\{([^}]+)\}\}", bind, script)
    out = Path(subprocess.run(["mktemp"], capture_output=True, text=True,
                              check=True).stdout.strip())
    got = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                         env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                              "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": "/dev/null"})
    assert got.returncode == 0, got.stderr
    parsed: dict[str, str] = {}
    lines = out.read_text().splitlines()
    while lines:
        key, _, value = lines.pop(0).partition("=")
        if "<<" in key:
            key, _, delim = key.partition("<<")
            value = ""
            while lines and lines[0] != delim:
                value += lines.pop(0) + "\n"
            if lines:
                lines.pop(0)
        parsed[key] = value.strip()
    return parsed


VERDICT = "What this run can conclude"


def _verdict(state: str, detail: str = "", stale: str = "false") -> dict[str, str]:
    return _run_step(VERDICT, **{
        "steps.window.outputs.state": state,
        "steps.window.outputs.detail": detail,
        "steps.window.outputs.at": "09:21:07Z",
        "steps.window.outputs.cron": "*/10 0-5  * * 1",
        "steps.check.outputs.stale": stale,
    })


def test_a_stale_heartbeat_outside_a_window_is_not_an_incident():
    """Both halves of #208, read off the workflow's own branch rather than off its prose. The
    heartbeat is stale in every one of these -- what changes is whether anything was supposed
    to be refreshing it."""
    for state, detail in (("late", "12060"), ("early", "1800")):
        got = _verdict(state, detail, stale="true")
        assert got["verdict"] == "no-window", got
        assert got["closes"] == "false", "a delayed run cannot observe health either"
    assert _verdict("preseason", FIRST_GAME, stale="true")["verdict"] == "no-window"


def test_an_incident_the_season_has_not_started_for_closes_without_waiting_for_a_game():
    """#193's actual exit. A healthy check needs a window, the first window needs the first
    game, and the incident would otherwise sit open across the two days before kickoff --
    which is what teaches an operator to skim this workflow on the Sunday."""
    got = _verdict("preseason", FIRST_GAME, stale="true")
    assert got["closes"] == "true", got
    assert FIRST_GAME in got["close_note"], got["close_note"]
    assert "no healthy check is possible yet" in got["close_note"]
    # The distinction the ticket record keeps once the comment has scrolled away: this
    # incident did not end, it was never founded. A bare close records `completed` for both.
    assert got["close_reason"] == "not planned", got
    assert _verdict("inside", "6480")["close_reason"] == "completed"


def test_a_stale_heartbeat_inside_a_window_is_still_an_incident():
    """The alarm this workflow exists to raise. Every suppression above is also reachable by
    a check that has quietly stopped checking, and this says it has not."""
    got = _verdict("inside", "6480", stale="true")
    assert got["verdict"] == "stale" and got["closes"] == "false", got
    got = _verdict("inside", "6480", stale="false")
    assert got["verdict"] == "healthy" and got["closes"] == "true", got

    # A dispatch and an unreadable cron measure too: an operator asked, or nothing could be
    # established, and neither is a reason to go quiet.
    for state in ("unscheduled", "unparseable"):
        assert _verdict(state, stale="true")["verdict"] == "stale", state


def test_the_incident_body_says_which_window_its_number_came_from():
    """The out-of-window case has to be tellable from a stall *in the issue text*, by a reader
    who is not going to go and work out what the clock was doing."""
    inside = _verdict("inside", "6480", stale="true")["measured"]
    assert "inside the window this run was scheduled for" in inside, inside
    assert "09:21:07Z" in inside and "*/10 0-5  * * 1" in inside
    assert "`" not in inside, (
        "this text is substituted into a double-quoted bash string in the filing step, where "
        "a backtick is a command substitution rather than markdown")
    assert "dispatch" in _verdict("unscheduled", stale="true")["measured"]


def test_the_verdict_is_what_gates_filing_and_closing():
    """The tie between the branch tested above and the steps that act on it. Without it the
    verdict could be computed correctly and ignored, which is the shape every guard in
    `tests/contracts/test_guards_are_load_bearing.py` was written for."""
    steps = {s["name"]: s for s in
             yaml.safe_load(WATCHDOG.read_text())["jobs"]["heartbeat"]["steps"] if "name" in s}
    assert VERDICT in steps, "the workflow no longer decides what a run can conclude"

    filing = next(s for n, s in steps.items() if n.startswith("File or update"))
    assert filing["if"] == "steps.verdict.outputs.verdict == 'stale'", (
        f"the incident is filed on {filing['if']!r} rather than on the verdict, so a run that "
        f"landed outside its window would file one again")
    assert "steps.verdict.outputs.measured" in filing["run"], (
        "the body does not say which window its number was measured in")

    closing = next(s for n, s in steps.items() if n.startswith("Close the incident"))
    assert closing["if"] == "steps.verdict.outputs.closes == 'true'"
    assert "--reason" in closing["run"], (
        "a bare `gh issue close` silently records `completed`, so the two ways an alarm stops "
        "standing become indistinguishable (docs/agents/issue-tracker.md)")

    window = next(s for n, s in steps.items() if n.startswith("Which window"))
    assert "github.event.schedule" in yaml.safe_dump(window), (
        "the run does not ask which cron scheduled it, so it has no window to compare against "
        "and is back to measuring against a clock it cannot see")
    assert "SEASON_OPENS" in window["run"], "the preseason case is never consulted"


def test_the_season_opens_on_a_date_the_workflow_can_read():
    opens = yaml.safe_load(WATCHDOG.read_text())["env"]["SEASON_OPENS"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(opens)), opens
    assert dt.date.fromisoformat(str(opens))
