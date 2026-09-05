"""Survivor as one assignment problem, not eighteen choices.

The greedy pick -- take the biggest favourite each week -- is what most entrants do, and
it is reliably wrong for a structural reason: spending Kansas City in week 1 against a bad
team costs you Kansas City in week 12 when the alternative is a coin flip. The schedule is
known in advance, so the whole season is a single integer program.

**Survival is multiplicative.** The objective is the sum of log win probabilities, not the
sum of probabilities. Maximising the sum would happily trade a 0.95 week for two 0.60s,
which is a worse season and an easy mistake to make with a linear objective.

Pool-aware play -- deliberately picking a contrarian team when the field is large enough
that survival alone is not sufficient -- is not implemented. It needs the pool size and
payout structure, which `docs/decisions.md` lists as the last open blocker, and the doc is
specific: under ~20 entries survival is close to optimal, above ~100 the objective becomes
P(finish first) and is materially more contrarian. Guessing at that would be worse than
maximising survival, which is right for the small case and defensible for the large one.

    uv run python -m hub.season.survivor --season 2026
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub import schedule
from hub.config import SEASON_AHEAD
from hub.paths import SITE

# Below this a team is treated as unpickable rather than fed to log(). A survivor pick at
# 1% is never the answer, and log(0) is negative infinity.
MIN_PROB = 1e-4

# Fewer than three priced games in a week. The "choice" is then which side of one or two
# games to take, which is worth saying out loud before anyone treats it as a plan.
THIN_WEEK = 6

# How long the season a plan covers is. Here rather than in `hub.publish`, which had its own
# `NFL_WEEKS = 18` beside this module's `--weeks` default of 18: one number, spelled twice,
# in the two places that plan the same season.
NFL_WEEKS = 18


def published_plan(path: Path | None = None) -> list[dict]:
    """The rows of the last published survivor artifact, or nothing.

    Read here so the CLI and `hub.publish.survivor` answer "what has been spent" from the
    same file. An unreadable or absent artifact is no history rather than an error -- a
    fresh clone has none, and refusing to plan because of that would be the
    operator-dependence `CLAUDE.md` warns about.
    """
    try:
        got = json.loads(Path(path or (SITE / "survivor.json")).read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(got, dict) or not isinstance(rows := got.get("rows"), list):
        return []
    # The season rides on the envelope, not on the rows. Carried down so `spent_teams` can
    # stay a function of rows -- and so it can tell a plan from this season apart from one
    # left over from the last, which is the collision issue #23 fixed for the weekly
    # artifact. `site/data/survivor.json` is committed and `data/processed/` is not, so a
    # scheduled run that starts a season mid-way reads last season's plan as this one's.
    season = got.get("season")
    return [{**r, **({"season": season} if season is not None else {})}
            for r in rows if isinstance(r, dict)]


def _solver():
    """Prefer COIN_CMD, fall back to the bundled CBC.

    PULP_CBC_CMD is deprecated and removed in PuLP 4.0, but COIN_CMD needs a system CBC
    that is not installed here. Preferring the supported one means this keeps working when
    CBC arrives or PuLP drops the old name, without requiring either today.
    """
    import warnings

    import pulp

    try:
        coin = pulp.COIN_CMD(msg=False)
        if coin.available():
            return coin
    except Exception:
        pass
    with warnings.catch_warnings():
        # Suppressed narrowly: the fallback is deliberate and the warning is about a
        # future PuLP version, not about this call being wrong.
        warnings.simplefilter("ignore", DeprecationWarning)
        return pulp.PULP_CBC_CMD(msg=False)


class Infeasible(Exception):
    """No assignment covers every week without reusing a team."""


def solve(grid: pl.DataFrame, weeks: Sequence[int] | None = None,
          spent: Sequence[str] = ()) -> pl.DataFrame:
    """One team per week, no repeats, maximising the probability of surviving them all.

    `grid` is (week, team, win_prob) for every team playing in every week.

    `spent` is the teams already used in weeks that are behind us. The no-repeat constraint
    below only binds *within* one plan, so a mid-season solve would otherwise hand back a
    team spent in September and the entry would be infeasible the moment it was entered.
    Passed in rather than inferred: this module is given a grid, not a history, and guessing
    which of the plan's own earlier picks were actually entered would be a different claim.
    """
    import pulp

    weeks = list(weeks) if weeks is not None else sorted(set(grid["week"].to_list()))
    # Teams, where `hub.publish.survivor` had a `gone` holding weeks a few lines from its
    # own `spent`. One word for two kinds of thing, in one neighbourhood.
    unavailable = {str(t) for t in spent}
    usable = grid.filter(pl.col("win_prob") > MIN_PROB)

    options = [(int(r["week"]), str(r["team"]), float(r["win_prob"]))
               for r in usable.iter_rows(named=True)
               if int(r["week"]) in weeks and str(r["team"]) not in unavailable]
    if not options:
        raise Infeasible("no pickable team in any week")

    prob = pulp.LpProblem("survivor", pulp.LpMaximize)
    # add_variable rather than LpVariable(...): the direct constructor is deprecated and
    # goes away in PuLP 4.0, and a 18x32 grid emits a warning per variable.
    x = {(w, t): prob.add_variable(f"x_{w}_{t}", cat="Binary") for w, t, _ in options}

    # log, because surviving every week is the product of surviving each one.
    prob += pulp.lpSum(math.log(p) * x[(w, t)] for w, t, p in options)

    for w in weeks:
        wk = [x[(w, t)] for ww, t, _ in options if ww == w]
        if not wk:
            raise Infeasible(f"no pickable team in week {w}")
        prob += pulp.lpSum(wk) == 1, f"one_pick_wk{w}"

    for team in {t for _, t, _ in options}:
        appearances = [x[(w, t)] for w, t, _ in options if t == team]
        if len(appearances) > 1:
            prob += pulp.lpSum(appearances) <= 1, f"once_{team}"

    status = prob.solve(_solver())
    if pulp.LpStatus[status] != "Optimal":
        raise Infeasible(
            f"no full-season plan: {pulp.LpStatus[status]}. With {len(weeks)} weeks and "
            f"{len({t for _, t, _ in options})} distinct teams, one team per week cannot "
            f"be covered without a repeat.")

    picked = [(w, t, p) for w, t, p in options if x[(w, t)].value() == 1]
    return pl.DataFrame(
        {"week": [w for w, _, _ in picked], "team": [t for _, t, _ in picked],
         "win_prob": [p for _, _, p in picked]}).sort("week")


def played(grid: pl.DataFrame, at: datetime | None = None) -> list[int]:
    """Weeks the season has already run: in the grid, and absent from what is ahead.

    Derived by difference rather than by comparing a week number to a date, because the two
    would disagree the first time a week straddled a boundary -- and because
    `schedule.forecastable` is then the only place the rule is written. This module wrapped
    that call as `forthcoming` for a while, under a docstring saying the rule was
    "unchanged and unrestated" while the name restated it; `docs/agents/domain.md` is
    specific about not drifting to a synonym, so the wrapper is gone and its callers ask
    `hub.schedule` directly.
    """
    ahead = set(schedule.forecastable(grid, at)["week"].to_list())
    return sorted({int(w) for w in grid["week"].to_list()} - {int(w) for w in ahead})


def spent_teams(prior: Sequence[Mapping[str, Any]], weeks: Sequence[int],
                season: int | None = None) -> list[str]:
    """Teams a previous plan assigned to weeks that are now behind us, in this season.

    The best available answer to "what has this entry already used", and stated as what it
    is: a reading of the last plan published, not a record of what was entered. An entrant
    who deviated has deviated from this too. It is still strictly better than assuming
    nothing was spent, which is what the plan did -- and which made every mid-season plan
    infeasible against the real remaining pool while looking exactly like a plan.

    **Scoped to one season**, for the reason `hub.publish._keeping_published` is: a week
    number does not identify a slate. A `survivor.json` left from last season would
    otherwise contribute its weeks-1..N picks here, and the committed artifact outlives the
    gitignored store, so a run that starts a season mid-way is exactly when it happens.
    A row carrying no season is read as this one's -- artifacts written before the plan
    recorded it, and forgetting what a running entry spent is the worse of the two errors.

    A row with no usable week is skipped rather than raising. `int(None)` took the whole
    panel down through `publish`'s broad except, which then reported an unavailable
    schedule: the wrong cause, for the wrong reason.
    """
    gone = {int(w) for w in weeks}
    out = set()
    for r in prior:
        if not r.get("team") or r.get("week") is None:
            continue
        if season is not None and r.get("season") not in (None, season):
            continue
        if int(r["week"]) in gone:
            out.add(str(r["team"]))
    return sorted(out)


class Coverage(NamedTuple):
    """The requested weeks, sorted into what the market has done about them.

    Typed rather than a dict, because `RemainingPlan` used to copy `covered`, `missing` and
    `thin` out of it one key at a time -- three fields restating one answer, and a reader of
    the plan could not tell they came from a single question.

    The per-week game counts that decide these three are not carried. They were, and nothing
    read them: `thin` is the question anyone actually asks of a count, and a field no caller
    needs is a promise this has to keep.
    """
    covered: list[int]
    missing: list[int]
    thin: list[int]


def coverage(grid: pl.DataFrame, weeks: Sequence[int]) -> Coverage:
    """Which requested weeks the market has actually priced.

    In August the board runs a handful of weeks deep, so a solve over "the season" quietly
    becomes a solve over whatever is posted. The plan is still the best available answer
    for the weeks it covers -- it just is not a season, and must not print like one.
    """
    usable = grid.filter(pl.col("win_prob") > MIN_PROB)
    counts = {int(r["week"]): int(r["len"])
              for r in usable.group_by("week").len().iter_rows(named=True)}
    weeks = list(weeks)
    return Coverage(
        covered=[w for w in weeks if counts.get(w, 0) > 0],
        missing=[w for w in weeks if counts.get(w, 0) == 0],
        thin=[w for w in weeks if 0 < counts.get(w, 0) < THIN_WEEK],
    )


def snapshot_only_weeks(grid: pl.DataFrame, weeks: Sequence[int]) -> list[int]:
    """Weeks no game reaches through the schedule's own moving field.

    Reported rather than assumed, because it is the quantity that decides whether reading
    the store was worth anything -- and because it moves: upstream fills `spread_line` in as
    the season approaches, so a week in this list today is not in it in December.
    """
    if "moving_field" not in grid.columns:
        return []
    have = set(grid.filter(pl.col("moving_field"))["week"].to_list())
    return [int(w) for w in weeks if int(w) not in have]


class RemainingPlan(NamedTuple):
    """A plan, and the scope it is a plan over.

    Returned together because the plan alone is unreadable: a survival probability means
    nothing without the weeks it is over, a reader looking at a plan that starts in week 9
    should not have to infer why, and a week the market has not priced still needs a pick
    from the entrant. The site panel and the CLI both print all of it.

    `coverage` rides along whole rather than unpacked into three lists here, and `survival`
    is a property rather than a call every caller makes on the way out -- both callers did
    `survival(got.picks)` on the next line, which is a behaviour of the plan and not of the
    caller.
    """
    picks: pl.DataFrame
    coverage: Coverage
    played: list[int]
    spent: list[str]
    snapshot_only: list[int]

    @property
    def survival(self) -> float:
        """Probability of surviving every week this plan covers."""
        # The module-level function, not this property: a method body's names resolve
        # through the module, never through the class it is written in.
        return survival(self.picks)


def plan_remaining(grid: pl.DataFrame, season: int, *,
                   prior: Sequence[Mapping[str, Any]] = (),
                   season_weeks: int = NFL_WEEKS,
                   at: datetime | None = None) -> RemainingPlan:
    """The plan for the weeks still ahead, against the teams still unspent.

    **From here, not from week 1.** Survivor is one assignment problem *because* spending a
    team early costs you that team later -- so a grid that still prices played weeks hands
    the solver its strongest teams for games that are over, and every remaining pick comes
    from a pool degraded by picks that were never available. The reported survival
    probability is then the product over games already won or lost. Both are wrong
    in-season and neither shows in the output: the plan looks like a plan.

    **Written once, because it was written twice.** These six steps -- what is ahead, which
    weeks are behind, what those weeks spent, which of the weeks left are priced, solve
    against the rest -- were verbatim in `hub.publish.survivor` and in this module's `main`.
    That sequence *is* the rule issue #24 was about, so a drift between the copies would
    have the panel and the CLI planning different seasons with nothing in either output
    saying so.

    `at` and `season_weeks` are arguments rather than reads of the clock and of a constant,
    so a test can put the season anywhere in itself; `prior` is the last published plan's
    rows, passed in for the reason `solve` takes `spent` rather than inferring it -- this
    module is given data, not a history. `season_weeks` is a count where every other `weeks`
    in this module is a list of week numbers, and it is spelled apart for the reason `spent`
    and the solver's teams are.

    Raising when no week is covered, rather than handing `solve` an empty week list: "no
    pickable team in any week" is true of the grid and is not the answer to what was asked,
    which was weeks 1 to `season_weeks`.
    """
    ahead = schedule.forecastable(grid, at)
    behind = played(grid, at)
    spent = spent_teams(prior, behind, season=season)
    # Against every week still to come, not against the weeks the grid happens to have --
    # asking coverage about its own weeks makes `missing` empty by construction and the
    # panel would never say a week needs a pick. A week already played is in neither list.
    cov = coverage(ahead, [w for w in range(1, season_weeks + 1) if w not in behind])
    if not cov.covered:
        raise Infeasible(f"no week in 1-{season_weeks} has a posted spread yet")
    return RemainingPlan(
        picks=solve(ahead, weeks=cov.covered, spent=spent),
        coverage=cov, played=behind, spent=spent,
        snapshot_only=snapshot_only_weeks(ahead, cov.covered))


def survival(plan: pl.DataFrame) -> float:
    """Probability of surviving every week in the plan."""
    out = 1.0
    for p in plan["win_prob"].to_list():
        out *= float(p)
    return out


def grid_from_schedule(season: int, cache: Path | None = None, *,
                       at: datetime | None = None,
                       base: Path | None = None) -> pl.DataFrame:
    """Win probability for every team in every week it plays.

    Two things are shared rather than restated, and both were claims this function used to
    make in prose while nothing enforced them.

    The spread-to-probability conversion is `MarketBaseline`'s, so a survivor pick and a
    weekly prediction cannot disagree about a game they both price. And the *spread* is
    `hub.schedule`'s, so they cannot disagree about which number that is either -- which
    they did, for a day: the weekly prediction moved onto the dated snapshots and this was
    left reading nflverse's own field, which upstream leaves empty for the late season. That
    planned twelve of eighteen weeks and reported the rest unpriced while the store held
    every game of the season, week 18 included.

    Spending a team early costs you that team later, so a plan over twelve weeks followed by
    a plan over the remaining six, with the best teams already gone, is strictly worse than
    one plan over eighteen. Which weeks are reachable is therefore not a display detail.

    A game neither source prices is dropped rather than filled at a coin flip, and
    `coverage` still names the week so an entrant knows it is theirs to fill.
    """
    from hub.models.market import MARGIN_SD, normal_cdf

    games = schedule.priced_games(season, at=at, cache=cache, base=base)
    rows = []
    for r in games.filter(pl.col("close_spread").is_not_null()).iter_rows(named=True):
        # close_spread is positive when the home team is favoured, both sources alike.
        home_p = normal_cdf(float(r["close_spread"]) / MARGIN_SD)
        # Whether the schedule's own field *could* have priced this game, which is not the
        # same as which source won. With the store covering every game, `price_source` reads
        # "snapshot" everywhere and says nothing about what the fallback would have reached
        # -- I reported all eighteen weeks as snapshot-only before the real data caught it.
        moving = r["schedule_spread"] is not None
        # Kickoff and result ride along per row, so `schedule.forecastable` can ask the same
        # question of this grid that `ratings` asks of the games it was built from. Deriving them
        # again here from a week number would be the second implementation of one idea that
        # this module's own docstring warns about.
        kick, res = r.get("kickoff"), r.get("result")
        rows.append((int(r["week"]), r["home_team"], home_p, moving, kick, res))
        rows.append((int(r["week"]), r["away_team"], 1.0 - home_p, moving, kick, res))
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows],
                         "moving_field": [r[3] for r in rows],
                         "kickoff": [r[4] for r in rows],
                         "result": [r[5] for r in rows]},
                        schema={"week": pl.Int64, "team": pl.Utf8, "win_prob": pl.Float64,
                                "moving_field": pl.Boolean, "kickoff": pl.Datetime,
                                "result": pl.Float64})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.survivor",
        description="Plan a full survivor season as one assignment problem.")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--weeks", type=int, default=NFL_WEEKS)
    a = ap.parse_args(argv)

    grid = grid_from_schedule(a.season)
    try:
        # The same call `hub.publish.survivor` makes, which is the point of it existing.
        got = plan_remaining(grid, a.season, prior=published_plan(),
                             season_weeks=a.weeks)
    except Infeasible as e:
        print(f"hub.season.survivor: {e}", file=sys.stderr)
        return 1
    # `got.picks` and `got.coverage` rather than locals called `plan` and `cov`: CONTEXT.md
    # avoids "the plan" unqualified, because a plan for the season and a plan from here are
    # different objects and this CLI prints the second one.
    cov = got.coverage
    print(f"  survivor plan, {a.season}, {len(cov.covered)} of "
          f"{len(cov.covered) + len(cov.missing)} remaining weeks priced")
    if got.played:
        print(f"  {len(got.played)} week(s) already played and absent from the plan: "
              + ", ".join(f"wk {w}" for w in got.played))
    if got.spent:
        print(f"  unavailable, already spent: {', '.join(got.spent)}")
    for r in got.picks.iter_rows(named=True):
        thin = "  (thin: one or two games priced)" if r["week"] in cov.thin else ""
        print(f"    wk {r['week']:>2}  {r['team']:<4} {r['win_prob']:.3f}{thin}")
    print(f"  survives the {got.picks.height} planned weeks: {got.survival:.1%}")
    # What the snapshot store actually buys, said out loud. These are the weeks nflverse's
    # lookahead field does not price, and the difference between a season plan and most of
    # one -- a team spent in week 3 is unavailable in week 17 whether or not the plan could
    # see week 17 when it chose.
    if got.snapshot_only:
        print(f"  {len(got.snapshot_only)} of these weeks are priced only by the dated "
              "snapshots: " + ", ".join(f"wk {w}" for w in got.snapshot_only))
    if cov.missing:
        # Not a failure: weeks with no spread are weeks the market has not posted, and a
        # plan over what exists beats no plan. But an entrant still has to pick in them.
        print("  not priced yet, so absent from the plan and still needing a pick: "
              + ", ".join(f"wk {w}" for w in cov.missing))
        print("  re-run once those weeks are on the board -- the teams spent early are "
              "not available to cover them.")
    print("  objective is survival only -- pool-aware play needs the pool config that "
          "docs/decisions.md still lists as open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
