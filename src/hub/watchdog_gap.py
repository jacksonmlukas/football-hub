"""Report a game window in which no watchdog check ran, after the fact (#391).

**The failure this is for.** On the Monday night of 2026-09-28/29 PHI @ CHI kicked off at
00:15Z and was played start to finish with no `live` run and no `watchdog` run. GitHub drops
scheduled workflows under load, and `live.yml` and `watchdog.yml` share that scheduler, so one
dropped-cron episode took out the refresher and the alarm about it together. The watchdog's
own check can only say "the heartbeat is stale" when it *runs*; a window with zero runs is
silence, and silence reads as health.

So the watchdog's first run after such a window -- the 06:42Z run, which on that Tuesday was
the next one delivered -- asks the question backwards: for every window that has closed
recently, did any check run inside it, and was there a game in it? A window with a game and no
check is reported. That is all this does; it does not repair the window, because by then it is
over.

**Where "window" comes from.** `live.yml`'s crons, read at run time from the file. They are
the only definition of a window and this module does not restate one: no hour, no weekday and
no length of a game is written below. A game belongs to a window when its kickoff, as ESPN
reports it, falls inside the window -- which is how the windows were drawn (each opens at or
before the earliest kickoff it is for).

**What counts as a check.** Any run of the watchdog workflow created inside the window, of any
event and any conclusion. A run that failed still ran, and a failed *job* is its own report.
A run created after the window closed -- the late delivery that does the reporting -- is not
inside it, which is the point.

**What it cannot see.** A window in which one check ran at the very end and nothing before it
is not reported: "no check ran" is the claim, and that window had one. A refresher that was
never started is `hub.live_chain`'s neighbour in the ticket and is not this module's concern;
this one makes the absence visible regardless of what the refresher did.

**Closing.** The gap itself cannot recover; it is a fact about a past window. The incident
closes when a look-back shows no such window at all, so one that is fixed ages out after
`LOOKBACK_DAYS` and one that is not stays open. Nothing here needs an operator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = dt.UTC
ET = ZoneInfo("America/New_York")

MARKER = "<!-- watchdog-gap -->"
LOOKBACK_DAYS = 7
HOUR = dt.timedelta(hours=1)

# Events ESPN reports for a game that was never played. Not "in progress" in any window.
NOT_PLAYED = frozenset({"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED"})


# --- the windows, read off the crons --------------------------------------------------------

def crons_in(text: str) -> list[str]:
    """The cron expressions in a workflow's text, ignoring commented-out ones."""
    return re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', text, flags=re.MULTILINE)


def _expand(field: str, span: int) -> set[int]:
    """`16-23`, `0-5`, `3`, `*` or a comma-separated mix. A step is refused, not widened:
    reading `*/2` as every hour would invent a window, and a window invented here is a gap
    reported for an hour nothing was ever scheduled."""
    if field == "*":
        return set(range(span))
    out: set[int] = set()
    for part in field.split(","):
        if "/" in part:
            raise ValueError(f"{field!r}: a step is not a window this reads")
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def _slots(crons: Iterable[str]) -> set[tuple[int, int]]:
    """(day-of-week with Sunday as 0, hour) in UTC for every hour these crons fire in. The
    minute field is not read: what a window is, is which hours are covered at all."""
    out: set[tuple[int, int]] = set()
    for c in crons:
        parts = c.split()
        if len(parts) != 5 or parts[2] != "*" or parts[3] != "*":
            raise ValueError(f"{c!r}: a window cron names days of the week only")
        _minute, hours, _dom, _mon, days = parts
        # cron writes Sunday as 0 or 7.
        dows = {d % 7 for d in _expand(days, 7)}
        out |= {(d, h) for d in dows for h in _expand(hours, 24)}
    return out


@dataclass(frozen=True)
class Window:
    """A maximal run of consecutive hours the crons fire in: `[start, end)`, UTC."""
    start: dt.datetime
    end: dt.datetime

    def contains(self, when: dt.datetime) -> bool:
        return self.start <= when < self.end

    @property
    def id(self) -> str:
        return self.start.strftime("%Y-%m-%dT%H:%MZ")


def windows(crons: Sequence[str], since: dt.datetime, until: dt.datetime) -> list[Window]:
    """Every window that closed in `(since, until]`.

    Consecutive hours merge across the day boundary and across cron lines, so `*/10 16-23 * * 0`
    and `*/10 0-5 * * 1` are one window, which is what they are: a Sunday of games that runs
    past midnight. Computed from a day before `since` so a window already open at `since` is
    whole rather than clipped into a gap it never had.
    """
    slots = _slots(crons)
    hour = (since - dt.timedelta(days=1)).astimezone(UTC).replace(
        minute=0, second=0, microsecond=0)
    out: list[Window] = []
    start: dt.datetime | None = None
    while hour <= until:
        inside = ((hour.weekday() + 1) % 7, hour.hour) in slots
        if inside and start is None:
            start = hour
        if not inside and start is not None:
            out.append(Window(start, hour))
            start = None
        hour += HOUR
    # A window still open at `until` has not closed and is nobody's gap yet.
    return [w for w in out if since < w.end <= until]


# --- the question -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Gap:
    """A window with no check in it. `kickoffs` is None when ESPN could not say whether there
    was a game -- reported anyway, because a monitor that goes quiet on a case it cannot
    establish is the failure this repo has already paid for twice."""
    window: Window
    kickoffs: tuple[dt.datetime, ...] | None


def find_gaps(wins: Iterable[Window], checks: Sequence[dt.datetime],
              kickoffs_in: Callable[[Window], Sequence[dt.datetime]]) -> list[Gap]:
    """The windows with no check inside them that had a game in them.

    Windows that *did* have a check are never put to `kickoffs_in`, so the lookup -- which is
    a network call per league per day in production -- costs nothing on a week where the
    scheduler behaved.
    """
    out: list[Gap] = []
    for w in wins:
        if any(w.contains(t) for t in checks):
            continue
        try:
            kicks = tuple(k for k in kickoffs_in(w) if w.contains(k))
        except Exception:
            out.append(Gap(w, None))
            continue
        if kicks:
            out.append(Gap(w, tuple(sorted(kicks))))
    return out


def unseen(gaps: Iterable[Gap], known_text: str) -> list[Gap]:
    """The gaps an existing incident does not already name. A window is named by its id, so a
    gap seen again on every later run is commented once rather than every ten minutes."""
    return [g for g in gaps if g.window.id not in known_text]


def render(gaps: Sequence[Gap], watchdog: str = "watchdog") -> str:
    """The incident text. Backticks are safe here: it is written to a file and read with
    `--body-file`, never substituted into a shell string."""
    lines = [
        MARKER,
        f"No `{watchdog}` check ran during {len(gaps)} game window(s). The workflow shares a "
        f"scheduler with `live`, and GitHub drops scheduled runs under load, so a window can "
        f"pass with no refresher and no alarm about it (#391). This was found after the fact, "
        f"by the first run delivered since.",
        "",
    ]
    for g in gaps:
        w = g.window
        span = f"{w.start:%Y-%m-%d %H:%MZ} to {w.end:%Y-%m-%d %H:%MZ}"
        if g.kickoffs is None:
            lines.append(f"- `{w.id}` ({span}): ESPN could not be asked whether a game was "
                         f"played, and no check ran.")
        else:
            kicks = ", ".join(f"{k:%H:%MZ}" for k in g.kickoffs)
            lines.append(f"- `{w.id}` ({span}): {len(g.kickoffs)} kickoff(s) at {kicks}, "
                         f"and no check ran.")
    lines += [
        "",
        "Nothing can be repaired for a window that is over. The overlay for it was not "
        "refreshed unless the `live` loop ran without the watchdog, which this cannot tell.",
        f"This issue closes when a look-back of {LOOKBACK_DAYS} days finds no such window.",
    ]
    return "\n".join(lines) + "\n"


# --- the sources ------------------------------------------------------------------------------

def parse_kickoff(raw: str) -> dt.datetime:
    """ESPN writes `2026-09-29T00:15Z`, which `fromisoformat` reads from 3.11."""
    when = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def kickoffs_of(payload: dict) -> list[dt.datetime]:
    """Kickoffs of the games in one scoreboard payload that were actually played."""
    out = []
    for ev in payload.get("events") or []:
        name = ((ev.get("status") or {}).get("type") or {}).get("name", "")
        if name in NOT_PLAYED or not ev.get("date"):
            continue
        out.append(parse_kickoff(ev["date"]))
    return out


def espn_kickoffs(window: Window) -> list[dt.datetime]:
    """Every kickoff either board lists on the days a window touches.

    Both leagues and the day either side, because the board is keyed by ESPN's own idea of a
    day and this does not guess it: the caller keeps only kickoffs inside the window, so the
    extra days cost a request and nothing else. Raises when ESPN cannot be reached, which
    `find_gaps` reports rather than reading as "no game".
    """
    from hub.fetch.espn import LEAGUE_PATHS, scoreboard
    days = {(window.start + dt.timedelta(days=d)).astimezone(ET).strftime("%Y%m%d")
            for d in (-1, 0, 1)} | {window.end.astimezone(ET).strftime("%Y%m%d")}
    out: list[dt.datetime] = []
    for league in LEAGUE_PATHS:
        for day in sorted(days):
            out += kickoffs_of(scoreboard(league, day))
    return out


def check_times(workflow: str, repo: str) -> list[dt.datetime]:
    """When each recent run of `workflow` was created. Raises when `gh` cannot say."""
    got = subprocess.run(
        ["gh", "run", "list", "--workflow", workflow, "--repo", repo, "--limit", "500",
         "--json", "createdAt"], capture_output=True, text=True, timeout=120, check=True)
    return [parse_kickoff(r["createdAt"]) for r in json.loads(got.stdout)]


# --- the CLI ----------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None,
         kickoffs_in: Callable[[Window], Sequence[dt.datetime]] = espn_kickoffs,
         checks_fn: Callable[[str, str], list[dt.datetime]] = check_times) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.watchdog_gap",
        description="Report game windows in which no watchdog check ran. Prints `gaps=N` and "
                    "`new=M` for $GITHUB_OUTPUT; writes the incident text to --report when "
                    "there is something new to say.")
    ap.add_argument("--live-yml", type=Path, default=Path(".github/workflows/live.yml"))
    ap.add_argument("--workflow", default="watchdog.yml")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--season-opens", default="",
                    help="ISO date; windows that closed before it are not looked at")
    ap.add_argument("--known-file", type=Path, default=None,
                    help="text of the open incident, so a gap already named is not repeated")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--now", default=None, help="ISO instant, for a replay")
    args = ap.parse_args(argv)

    now = parse_kickoff(args.now) if args.now else dt.datetime.now(UTC)
    try:
        wins = windows(crons_in(args.live_yml.read_text()),
                       now - dt.timedelta(days=LOOKBACK_DAYS), now)
        if args.season_opens:
            opens = dt.datetime.fromisoformat(args.season_opens).replace(tzinfo=UTC)
            wins = [w for w in wins if w.start >= opens]
        checks = checks_fn(args.workflow, args.repo)
    except Exception as e:
        # Not "no gaps". A check that cannot read the run list has established nothing, and
        # exiting 0 here is how it would stay quiet for as long as its token was wrong.
        print(f"watchdog_gap: could not establish the run history: {e!r}", file=sys.stderr)
        return 2

    gaps = find_gaps(wins, checks, kickoffs_in)
    known = args.known_file.read_text() if args.known_file else ""
    fresh = unseen(gaps, known)
    print(f"gaps={len(gaps)}")
    print(f"new={len(fresh)}")
    if fresh and args.report:
        args.report.write_text(render(fresh, args.workflow.removesuffix(".yml")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
