"""The 2026 forward measurement of the Weekly projection against consensus (#432).

**Read `docs/weekly-forward.md` first.** The design is written there, before any number this
prints, and this docstring says only where each piece of it lives.

This is `hub.season.weekly_gate`'s comparison on the one season it could not have been fitted
to, scored as the season is played, with the incumbent read from the first-party captures
`hub.fetch.consensus` writes under `state/consensus/` -- not from the archive, which cannot
show what could have been read before kickoff.

**What it admits.** A week counts if and only if a capture of that week's page exists whose
`captured_at` precedes the week's first game day, whose scrape is that week's page, and whose
file was first committed before that day as well (`first_commit_at`: the commit is the
timestamp, `hub.prereg`'s argument). Every other week is *named* with its reason, never silently
dropped, and the weeks played before any capture existed are not in the measurement at all
(`docs/weekly-forward.md`, *Weeks 1-5*).

**When it speaks.** Once, at the horizon: week 14's games are played. Before that it prints
NOT-YET, and -- the point of the structure -- it has not loaded a single 2026 outcome to say so:
`assemble`, the one function that reads results, is called after the horizon check and never
before it. `test_a_run_before_the_horizon_reads_no_outcome` plants an `assemble` that raises.

**The unit.** 2026 is one season, so the season cluster has no degrees of freedom; the interval
half clusters on the **week** and the season's Disposition on the **roster**, through the shipped
`experiment.gate` (#381's (C)). This is deliberately not a `Harness`: a `Harness` is a
season-clustered gate by construction, and this is the one place the repo clusters elsewhere,
for the reason `docs/weekly-forward.md` gives, with the variance decomposition.

    uv run python -m hub.season.weekly_forward
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import polars as pl

from hub.cli import unavailable
from hub.closure import import_closure, module_digests
from hub.config import FANTASY_WEEKS
from hub.declare import decision, not_an_input
from hub.fetch import consensus
from hub.fetch.consensus import Capture
from hub.models.experiment import PLAYER_STATS_COLS, Actions, Ceiling, GateRun, gate
from hub.season.weekly_gate import CEILING_ARM, WITHIN, void_condition

SEASON = 2026

# The player-stats columns the horizon's presence check asks for: the shared list, so it is one
# cache entry with every other reader and satisfies the `nflverse_player_stats` contract.
STATS_COLS = PLAYER_STATS_COLS
DESIGN = "docs/weekly-forward.md"

# The arm the design pinned (#456): the first-party import closure of the code the forward
# measurement runs, walked from the modules that define the arm (`ARM_ROOTS`) with the same
# walker and the same exemptions as the ledger's code declaration (`hub.closure.import_closure`,
# `CLOSURE_EXEMPT`, #439), each module's source hashed. A reading taken on a closure that differs
# in any module -- one edited, one added, one gone -- refuses, and the refusal names which. It
# replaces #432's pin of the blob of `hub/exhibits/weekly_projection.py` alone, which could not
# see `hub.models.panel` or the rest of what the arm reads (#315's opponent-adjusted DvP would
# have changed the arm and not the reading's willingness to read). Re-pinning is a dated
# amendment of `docs/weekly-forward.md` made before any 2026 outcome is read, with the closure's
# behavioural identity shown (`tests/unit/test_weekly_projection_move.py`), as #430 and #456 were.
ARM_ROOTS = ("hub.season.weekly_gate_data", "hub.exhibits.weekly_projection")
PINNED_ARM_MODULES: dict[str, str] = {
    "hub.draft.adp_history": "286f2d9c667d",
    "hub.draft.availability": "34b707bf7e75",
    "hub.draft.board": "6e61e56ecac0",
    "hub.draft.cohort": "5d6e03fc2cd5",
    "hub.draft.durability": "59fbe0402d35",
    "hub.draft.optimize": "314a78a8b5ef",
    "hub.draft.picks": "c695b87cc5e8",
    "hub.draft.playoff_sos": "42bc422f88eb",
    "hub.draft.prior_signal": "4b7421479470",
    "hub.draft.regression": "3ba22b3a649e",
    "hub.draft.report": "df1f8e20569e",
    "hub.draft.season": "601db0b4f9f1",
    "hub.draft.state": "ad958507df6a",
    "hub.exhibits.weekly_projection": "372f7180071e",
    "hub.holdout": "a8d095db300c",
    "hub.league": "1d50b1eaf7ef",
    "hub.models.base": "8c41cd5f26d1",
    "hub.models.components": "b23c5d7d0c49",
    "hub.models.conformal": "79827e566bba",
    "hub.models.coverage": "3c2d88cc6908",
    "hub.models.margin": "8d58c201cc4b",
    "hub.models.market": "098baf487556",
    "hub.models.panel": "99d19fba99e0",
    "hub.models.predict": "ddf960a1dfe6",
    "hub.models.scoring_rules": "3c201f3a0d6f",
    "hub.models.volume": "31e5321b1e6d",
    "hub.names": "93e503040186",
    "hub.season.weekly_gate": "73cc434a55db",
    "hub.season.weekly_gate_data": "1823ccae01c3",
}


def arm_digest(modules: Mapping[str, str]) -> str:
    """16-char hash of a closure's `{module: digest}`, the one value the reading prints and the
    document records beside the table."""
    return hashlib.sha256(json.dumps(dict(sorted(modules.items()))).encode()).hexdigest()[:16]


PINNED_ARM_DIGEST = arm_digest(PINNED_ARM_MODULES)


def arm_difference(arm: Mapping[str, str] | None) -> list[str]:
    """What differs between `arm` and the pin, module by module; empty if they are the same.
    `None` is an arm that could not be identified, and differs from everything."""
    if arm is None:
        return ["the arm's modules could not be read"]
    out = [f"{m} changed" for m in sorted(arm.keys() & PINNED_ARM_MODULES.keys())
           if arm[m] != PINNED_ARM_MODULES[m]]
    out += [f"{m} is new in the arm's imports"
            for m in sorted(arm.keys() - PINNED_ARM_MODULES.keys())]
    out += [f"{m} is no longer in the arm's imports"
            for m in sorted(PINNED_ARM_MODULES.keys() - arm.keys())]
    return out

# The last week the measurement reads: the fantasy regular season's, `GATE_WEEKS`' own (15-17
# are reported apart by the gate and are not read here).
HORIZON_WEEK = max(FANTASY_WEEKS)

# Below this many admitted weeks the design cannot reach its question: chosen, not fitted and
# set before any 2026 outcome was read (`docs/weekly-forward.md`, *The decision rule*).
MIN_WEEKS = not_an_input(
    6, "the fewest admitted weeks the forward reading is pre-registered to issue a verdict on: "
       "a harness threshold chosen before any outcome, which no prediction reads")

# What one independent observation is in the interval half. 2026 is one season, so the season
# has no degrees of freedom; the week is the unit that absorbs what the rosters of a week share.
CLUSTER_WEEK: tuple[str, ...] = ("week",)

# The data lags the schedule: nflverse's weekly tables land the day after the last game.
LAG_DAYS = 1

ACTIONS = Actions(
    adopt="ADOPT (forward, one season): the 2026 measurement reverses the weekly gate's REMOVE "
          "as evidence. ADR-0016 is re-opened for the maintainer's decision and the weekly "
          "tickets held behind #432 are released; it does not by itself set a lineup.",
    remove="REMOVE (forward, one season): the spent seasons' REMOVE is confirmed on a season it "
           "could not have been fitted to. The weekly tickets stay blocked.",
    show="SHOW: nothing moves. The REMOVE stands and the weekly tickets stay blocked. At this "
         "design's power that is the branch it reaches at any plausible effect, and it is an "
         "exemption, not a finding (docs/weekly-forward.md, *The power at that horizon*).")

CAVEATS = (
    "the arm under test reads the game's spread as nflverse holds it at run time (the one that "
    "stood at the game), the incumbent a page captured before kickoff: an information "
    "asymmetry that favours the projection, so an ADOPT must be read with it and a REMOVE or "
    "SHOW is not weakened by it",
    "one season of 20 simulated frozen rosters: it speaks for 2026 and for the admitted weeks "
    "only, and the spent seasons are not pooled into it",
)


class Admission(NamedTuple):
    """Which weeks count, and why each of the others does not."""
    admitted: dict[int, Capture]
    refused: dict[int, str]           # week -> why every capture of it was turned away
    uncaptured: tuple[int, ...]       # weeks 1-14 with no capture file at all


class Reading(NamedTuple):
    """What a run says: a status, the lines to print, and the gate's own run when it ran one."""
    status: str
    lines: list[str]
    run: GateRun | None = None
    admission: Admission | None = None


def first_commit_at(path: Path) -> datetime | None:
    """When `path` was first committed (aware UTC), or None where git cannot say.

    None for a path outside a repository, an untracked file, a shallow clone (its history stops
    at an arbitrary commit, and every file older than that would look committed at the boundary)
    or a failing git. `admit` refuses a capture whose time is None: the file's own `captured_at`
    is never a substitute for the commit.
    """
    try:
        cwd = path.parent
        if subprocess.run(["git", "-C", str(cwd), "rev-parse", "--is-shallow-repository"],
                          capture_output=True, text=True, check=True).stdout.strip() != "false":
            return None
        out = subprocess.run(
            ["git", "-C", str(cwd), "log", "--diff-filter=A", "--format=%ct", "--", path.name],
            capture_output=True, text=True, check=True).stdout.split()
        return datetime.fromtimestamp(int(out[-1]), UTC) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@decision
def admit(captures: Sequence[Capture], days: dict[int, date],
          floors: dict[int, date], *,
          first_commit: Callable[[Path], datetime | None]) -> Admission:
    """The weeks the measurement may read, by the rule the design fixed before any outcome.

    A capture is turned away when it was written at or after its week's first game day
    (`consensus.is_late`, judged against the schedule *as it is now*, so a game moved earlier
    makes a capture later, not earlier), when its scrape is not that week's page, or when its
    file was first committed after the deadline whatever the file says of itself. Of the
    captures that stand the latest is the page that could last have been read, and is the one
    used. A week with captures and no valid one is **refused with its reason**; a week with none
    is `uncaptured`; neither is scored.
    """
    admitted: dict[int, Capture] = {}
    refused: dict[int, str] = {}
    uncaptured: list[int] = []
    for week in FANTASY_WEEKS:
        if week not in days:
            continue
        first = days[week]
        mine = [c for c in captures if c.week == week]
        if not mine:
            uncaptured.append(week)
            continue
        valid: list[Capture] = []
        why: list[str] = []
        for c in mine:
            deadline = consensus.deadline_for(first)
            if consensus.is_late(c.captured_at, first):
                why.append(f"{c.captured_at:%Y-%m-%dT%H:%M}Z is at or after the deadline "
                           f"{deadline:%Y-%m-%dT%H:%M}Z")
                continue
            if not (floors[week] <= c.scrape_date < first):
                why.append(f"its scrape of {c.scrape_date} is not week {week}'s page")
                continue
            committed = first_commit(c.path) if c.path is not None else None
            if committed is None:
                # Never the file's own word: it is the one thing a late capture can set.
                where = c.path.name if c.path is not None else "(no path)"
                why.append(f"{where}: first-commit time unknown (shallow clone, untracked "
                           f"file or no history) -- not admitted; read with full history "
                           f"(`fetch-depth: 0`)")
                continue
            if committed >= deadline:
                why.append(f"first committed {committed:%Y-%m-%dT%H:%M}Z, after the deadline")
                continue
            valid.append(c)
        if valid:
            admitted[week] = max(valid, key=lambda c: c.captured_at)
        else:
            refused[week] = "; ".join(why)
    return Admission(admitted, refused, tuple(uncaptured))


def horizon_reached(as_of: date, last_game_day: date | None) -> bool:
    """Week 14's games are played and the data has had a day to land. No schedule, no horizon."""
    return last_game_day is not None and as_of > last_game_day + timedelta(days=LAG_DAYS)


def _admission_lines(adm: Admission) -> list[str]:
    out = [f"  weeks admitted ({len(adm.admitted)}): {sorted(adm.admitted) or 'none'}"]
    for week, why in sorted(adm.refused.items()):
        out.append(f"  week {week} REFUSED: {why}")
    if adm.uncaptured:
        out.append(f"  weeks with no capture (not in the measurement): {list(adm.uncaptured)}")
    return out


@decision
def read_forward(*, as_of: date, schedule: pl.DataFrame, captures: Sequence[Capture],
                 assemble: Callable[[dict[int, Capture]], tuple[pl.DataFrame, dict | None]],
                 first_commit: Callable[[Path], datetime | None],
                 horizon_data: Callable[[], bool], arm: Mapping[str, str] | None,
                 seed: int = 0) -> Reading:
    """One reading of the forward measurement, or the statement that it is not time.

    `horizon_data` says whether week 14's rows are in nflverse (a presence check, run only once
    the date has passed); `arm` is the `{module: digest}` of the arm's import closure as it stands
    (`arm_closure()`), and any value but `PINNED_ARM_MODULES` refuses the reading, naming what
    differs.

    `assemble` is the only function that reads outcomes -- admitted weeks in, a paired frame
    (`season`, `roster`, `week`, `diff`, and `ceiling_diff`) and the gate's coverage out -- and it
    is called after the horizon check and after the admission, never before either: a run before
    the horizon loads nothing it could be tempted to look at.
    """
    days = consensus.first_game_days(schedule)
    last = consensus.last_game_day(schedule, HORIZON_WEEK)
    if not horizon_reached(as_of, last):
        when = "no schedule for it" if last is None else f"its last game is {last}"
        return Reading("NOT-YET", [
            f"NOT-YET: the horizon is the end of week {HORIZON_WEEK} ({when}); this run is "
            f"{as_of}. No outcome was loaded and none will be before it ({DESIGN})."])
    if not horizon_data():
        return Reading("NOT-YET", [
            f"NOT-YET: the dates are past week {HORIZON_WEEK}, but its rows are not in nflverse "
            f"yet. No outcome was read."])
    if (diff := arm_difference(arm)):
        return Reading("REFUSED", [
            f"REFUSED: the arm under test is not the pinned one ({'; '.join(diff)}). "
            f"A change to code the projection runs is a new arm and needs a new pre-registration "
            f"({DESIGN}); no verdict is read and no outcome was loaded."])
    adm = admit(captures, days, consensus.scrape_floors(schedule), first_commit=first_commit)
    head = _admission_lines(adm)
    if len(adm.admitted) < MIN_WEEKS:
        return Reading("NOT-RUNNABLE", [
            f"NOT RUNNABLE: {len(adm.admitted)} admitted week(s), fewer than the {MIN_WEEKS} the "
            f"design was pre-registered to read ({DESIGN}). Not a failed arm: the capture "
            f"missed too many weeks. No outcome was loaded.", *head], None, adm)
    paired, cover = assemble(adm.admitted)
    top = Ceiling(CEILING_ARM, paired["ceiling_diff"].to_numpy()) if (
        "ceiling_diff" in paired.columns) else None
    run = gate(paired.drop("ceiling_diff", strict=False), cluster=CLUSTER_WEEK, within=WITHIN,
               ceiling=top, actions=ACTIONS, void=void_condition(cover), seed=seed)
    s = run.summary
    nan = float("nan")
    interval = f"[{s.get('t_lo', nan):+.3f}, {s.get('t_hi', nan):+.3f}]"
    lines = [f"{run.verdict[0]}: {run.verdict[1]}",
             f"  2026, {int(s.get('clusters', 0))} week clusters "
             f"({int(s.get('n', 0))} roster-weeks): weekly - consensus "
             f"{s.get('mean', nan):+.3f} points per team-week, t interval {interval}, "
             f"MDE {s.get('mde', nan):.3f}",
             f"  Disposition: {run.resolved} resolved of {run.seasons.height}, "
             f"{run.abstained} abstained",
             *head]
    lines.append(f"  arm under test: closure {PINNED_ARM_DIGEST} of {len(PINNED_ARM_MODULES)} "
                 f"modules (pinned, {DESIGN})")
    lines += [f"  caveat: {c}" for c in CAVEATS]
    return Reading(run.verdict[0], lines, run, adm)


def arm_closure() -> dict[str, str] | None:  # pragma: no cover - reads the working tree
    """The arm's import closure as it stands: `{module: digest}`, or `None` if it cannot be read."""
    try:
        return module_digests(import_closure(ARM_ROOTS))
    except (OSError, ValueError, SyntaxError, ImportError):
        return None


def consensus_frame(admitted: dict[int, Capture]) -> pl.DataFrame:
    """The captures as the frame `weekly_consensus` returns: `season`, `week`, `key`, `ecr`."""
    from hub.names import player_key
    frames = [c.rows.select(pl.col("player").map_elements(player_key, return_dtype=pl.Utf8)
                            .alias("key"), pl.col("ecr").cast(pl.Float64))
              .with_columns(season=pl.lit(c.season, dtype=pl.Int64),
                            week=pl.lit(c.week, dtype=pl.Int64),
                            lead_days=pl.lit((c.first_game_day - c.scrape_date).days,
                                             dtype=pl.Int64))
              for c in admitted.values()]
    return (pl.concat(frames).group_by(["season", "week", "key"])
              .agg(pl.col("ecr").min(), pl.col("lead_days").first()))


def assemble_2026(admitted: dict[int, Capture]
                  ) -> tuple[pl.DataFrame, dict]:  # pragma: no cover - network
    """Rosters, realised points and both arms' scores for 2026, on the admitted weeks only.

    2025 is the walk-forward buffer `expanding_seasons` consumes to fit 2026's shrinkage on; it
    is not scored (its consensus is absent from the frame passed in, so no 2025 week is covered).
    """
    from hub.season import weekly_gate
    from hub.season.weekly_gate_data import assemble_universe
    g = assemble_universe([SEASON - 1, SEASON], drafts=20, seed=0,
                          ecr=consensus_frame(admitted))
    weeks = sorted(admitted)
    paired = weekly_gate.compare(g, weeks=weeks, ceiling=True)
    return paired.filter(pl.col("season") == SEASON), weekly_gate.coverage(g, weeks)


def week14_loaded() -> bool:
    """Whether nflverse has any player-week row for the horizon week: presence only."""
    from hub.fetch import nflverse
    stats = nflverse.load("player_stats", [SEASON], cols=STATS_COLS)
    return bool((stats["week"] == HORIZON_WEEK).any())


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--as-of", help="ISO date this reading is taken on (default: today, UTC)")
    a = ap.parse_args(argv)
    as_of = date.fromisoformat(a.as_of) if a.as_of else datetime.now(UTC).date()
    try:
        from hub.fetch import nflverse
        schedule = nflverse.load("schedules", [SEASON], cols=consensus.SCHEDULE_COLS)
    except Exception as exc:
        return unavailable("hub.season.weekly_forward", "the 2026 schedule", exc)
    reading = read_forward(as_of=as_of, schedule=schedule,
                           captures=consensus.read_captures(SEASON), assemble=assemble_2026,
                           first_commit=first_commit_at, horizon_data=week14_loaded,
                           arm=arm_closure())
    print("\n".join(reading.lines))
    return 0 if reading.status in {"ADOPT", "REMOVE", "SHOW", "NOT-YET"} else 1


if __name__ == "__main__":
    sys.exit(main())

