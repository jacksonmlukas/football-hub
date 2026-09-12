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
from hub.cli import unavailable
from hub.config import SEASON_AHEAD, PoolConfig
from hub.paths import SITE

NOT_FITTED_BECAUSE = (
    "MIN_PROB is a floor that keeps a zero out of a log, and THIN_ROWS a count of grid rows. "
    "Both are settings; the win probabilities themselves come from hub.models.market at run "
    "time. "
)

# Below this a team is treated as unpickable rather than fed to log(). A survivor pick at
# 1% is never the answer, and log(0) is negative infinity.
MIN_PROB = 1e-4

# The row count below which a week is thin. Rows, not games and not weeks: the grid carries
# one row per team per game, so 6 is three games. It was `THIN_WEEK`, which named neither the
# quantity nor its unit -- a reader checking "is 6 three games or six?" had only the comment.
# Fewer than three priced games and the "choice" is which side of one or two games to take,
# which is worth saying out loud before anyone treats it as a plan for the weeks ahead.
THIN_ROWS = 6

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
    out = [{**r, **({"season": season} if season is not None else {})}
           for r in rows if isinstance(r, dict)]
    # The envelope's own ledger, carried forward as rows of a week already behind. The rows
    # are the *remaining* plan: the first publish after a played week trims that week's row
    # out, so the second publish of the same week found no played rows, read the ledger as
    # empty, and offered the spent team again (review 2026-09-12, reproduced). A ledger row
    # carries week 0 -- behind every real week -- so `spent_teams` counts it whatever `weeks`
    # it is handed, and the season the envelope names so last season's ledger does not.
    for team in got.get("spent") or []:
        if isinstance(team, str) and team:
            out.append({"week": 0, "team": team, "ledger": True,
                        **({"season": season} if season is not None else {})})
    return out


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
          spent: Sequence[str] = (), pool: PoolConfig | None = None) -> pl.DataFrame:
    """The weeks a pool asks for, no repeats, maximising the probability of surviving them all.

    One team a week, except where the pool says two. Weeks 13-18 of this pool take a pair and
    both have to win, which the objective already expresses: surviving both is the product of
    surviving each, so the sum of logs it maximises is the joint probability without a change.
    The no-repeat constraint needed no change either -- it already spans the season, and a
    24-pick path is what it was always counting.

    What is new is that two picks in one week must not be the two sides of one fixture. Under
    one pick that was unreachable; under two it is reachable, guaranteed fatal, and *attractive*
    -- not for the week, where two opposite sides multiply to at most a quarter and any two
    independent favourites match that, but across the season, where spending two coin flips at
    once conserves two favourites for later. The optimiser will take that trade unless it is
    forbidden.

    A caller that passes no `pool` gets one pick a week: these are the pool's rules, not the
    game's, and a solver handed no rules should not invent them.

    `grid` is (week, team, win_prob) for every team playing in every week.

    `spent` is the teams already used in weeks that are behind us. The no-repeat constraint
    below only binds *within* one plan, so a mid-season solve would otherwise hand back a
    team spent in September and the entry would be infeasible the moment it was entered.
    Passed in rather than inferred: this module is given a grid, not a history, and guessing
    which of its own earlier picks were actually entered would be a different claim.
    """
    import pulp

    doubles = tuple(pool.double_pick_weeks) if pool is not None else ()
    weeks = list(weeks) if weeks is not None else sorted(set(grid["week"].to_list()))
    if doubles and "game_id" not in grid.columns and any(w in doubles for w in weeks):
        raise ValueError(
            "a double-pick week needs `game_id` on the grid: without it the two picks cannot "
            "be stopped from being the two sides of one fixture, which loses by construction")
    picks_in = {w: (2 if w in doubles else 1) for w in weeks}
    # Teams, where `hub.publish.survivor` had a `gone` holding weeks a few lines from its
    # own `spent`. One word for two kinds of thing, in one neighbourhood.
    unavailable = {str(t) for t in spent}
    usable = grid.filter(pl.col("win_prob") > MIN_PROB)

    has_gid = "game_id" in usable.columns
    options = [(int(r["week"]), str(r["team"]), float(r["win_prob"]),
                str(r["game_id"]) if has_gid else "")
               for r in usable.iter_rows(named=True)
               if int(r["week"]) in weeks and str(r["team"]) not in unavailable]
    if not options:
        raise Infeasible("no pickable team in any week")

    prob = pulp.LpProblem("survivor", pulp.LpMaximize)
    # add_variable rather than LpVariable(...): the direct constructor is deprecated and
    # goes away in PuLP 4.0, and a 18x32 grid emits a warning per variable.
    x = {(w, t): prob.add_variable(f"x_{w}_{t}", cat="Binary") for w, t, _, _ in options}

    # log, because surviving every week is the product of surviving each one -- and, in a
    # double-pick week, of surviving both of that week's games.
    prob += pulp.lpSum(math.log(p) * x[(w, t)] for w, t, p, _ in options)

    for w in weeks:
        wk = [x[(w, t)] for ww, t, _, _ in options if ww == w]
        need = picks_in[w]
        # Fixtures, not rows -- the same correction `coverage` already carries. Two picks
        # have to come from two *games*: both sides of one fixture cannot both win, so a
        # double-pick week whose only usable teams are the two sides of one game has one
        # usable fixture and no way to cover itself. Counted on rows it passes here, and the
        # `one_side` constraint below then makes the whole season infeasible -- which raises
        # for the season and names none of the week that caused it.
        fixtures = {gid for ww, _, _, gid in options if ww == w and gid}
        have, unit = (len(fixtures), "usable fixture(s)") if fixtures \
            else (len(wk), "pickable team(s)")
        if have < need:
            raise Infeasible(
                f"week {w} needs {need} pick(s) and has {have} {unit}"
                + (f", from {len(wk)} pickable team(s) -- two picks cannot come from both "
                   "sides of one fixture, because one of them loses"
                   if fixtures and len(wk) > have else ""))
        prob += pulp.lpSum(wk) == need, f"picks_wk{w}"
        # Never both sides of one fixture: one of them loses, so the week is lost.
        by_game: dict[str, list] = {}
        for ww, tt, _, gid in options:
            if ww == w and gid:
                by_game.setdefault(gid, []).append(x[(w, tt)])
        for gid, sides in by_game.items():
            if len(sides) > 1:
                prob += pulp.lpSum(sides) <= 1, f"one_side_wk{w}_{gid}"

    for team in {t for _, t, _, _ in options}:
        appearances = [x[(w, t)] for w, t, _, _ in options if t == team]
        if len(appearances) > 1:
            prob += pulp.lpSum(appearances) <= 1, f"once_{team}"

    status = prob.solve(_solver())
    if pulp.LpStatus[status] != "Optimal":
        raise Infeasible(
            f"no full-season plan: {pulp.LpStatus[status]}. {len(weeks)} weeks asking for "
            f"{sum(picks_in.values())} picks against "
            f"{len({t for _, t, _, _ in options})} distinct teams cannot be covered "
            f"without a repeat.")

    picked = [(w, t, p) for w, t, p, _ in options if x[(w, t)].value() == 1]
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
    nothing was spent, which is what `plan_remaining` did -- and which made every mid-season
    infeasible against the real remaining pool while looking exactly like a plan.

    **Scoped to one season**, for the reason `hub.publish._keeping_published` is: a week
    number does not identify a slate. A `survivor.json` left from last season would
    otherwise contribute its weeks-1..N picks here, and the committed artifact outlives the
    gitignored store, so a run that starts a season mid-way is exactly when it happens.
    A row carrying no season is read as this one's -- artifacts written before a remaining plan
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
        if int(r["week"]) in gone or r.get("ledger"):
            out.add(str(r["team"]))
    return sorted(out)


class WeekFixtures(NamedTuple):
    """One week's fixtures, sorted by which of the two questions each one can answer.

    There is one rule for a usable week and it has two halves, because *pickable* and
    *drawable* are different statements about a fixture and were never written down as such.

    **Pickable** is what `solve`, `auto_pick` and `pool.weekly` need: a side priced above
    `MIN_PROB`, which is a floor on what may be *taken* -- a survivor pick at 1% is never the
    answer, and log(0) is negative infinity.

    **Drawable** is what `pool.weeks_from_grid` needs: *both* sides priced, at any
    probability, because a game whose opponent has no row cannot be played out. The floor does
    not belong here. A fixture priced 0.9999 against 0.0001 is perfectly drawable and its
    hopeless side is not pickable, and folding those into one test would refuse to simulate a
    week that is fully priced.

    They were an unstated rule apiece and disagreed three ways, each verified: the floor was
    applied on the coverage side and not on the simulator side, so a team `auto_pick` refuses
    was still available to our own entry inside the simulation; a fixture priced on one side
    only counted as coverage and was refused a draw, so `plan_remaining` published a plan for
    a week `pool` would not price at all; and a double-pick week counted its picks against
    fixtures here and against nothing there, so a week `solve` calls infeasible was simulated
    as the whole field eliminated -- the exact reading `UnpricedWeek` exists to prevent.

    `needs` rides along because it is the third thing both consumers have to agree on and the
    one they disagreed about most expensively.
    """
    week: int
    needs: int                  # picks this week takes: 1, or 2 in a double-pick week
    pickable: tuple[str, ...]   # fixtures with a side priced above MIN_PROB
    drawable: tuple[str, ...]   # fixtures with both sides priced, at any probability
    half: tuple[str, ...]       # the rest: priced on one side only, so drawable by neither


def week_fixtures(grid: pl.DataFrame, weeks: Sequence[int],
                  pool: PoolConfig | None = None) -> list[WeekFixtures]:
    """What each requested week has to offer, counted in fixtures rather than rows.

    Fixtures, not rows, everywhere: the grid carries one row per team per game, so a count of
    rows says nothing about how many *games* a week has to take two picks from. Both callers
    had reached that correction separately and neither had said so in a place the other read.

    A grid with no `game_id` cannot identify a fixture at all. Every priced team is then its
    own pickable option -- which is the old row count, and what `coverage` fell back to -- and
    nothing is drawable, which is why `pool.weeks_from_grid` refuses such a grid outright
    rather than guessing which rows are two sides of one game.
    """
    doubles = tuple(pool.double_pick_weeks) if pool is not None else ()
    has_gid = "game_id" in grid.columns
    out = []
    for raw in weeks:
        w = int(raw)
        wk = grid.filter(pl.col("week") == w)
        needs = 2 if w in doubles else 1
        if not has_gid:
            out.append(WeekFixtures(w, needs, tuple(
                str(t) for t in wk.filter(pl.col("win_prob") > MIN_PROB)["team"]), (), ()))
            continue
        sides: dict[str, list[float]] = {}
        for r in wk.iter_rows(named=True):
            sides.setdefault(str(r["game_id"]), []).append(float(r["win_prob"]))
        out.append(WeekFixtures(
            w, needs,
            tuple(sorted(g for g, ps in sides.items() if any(p > MIN_PROB for p in ps))),
            tuple(sorted(g for g, ps in sides.items() if len(ps) == 2)),
            tuple(sorted(g for g, ps in sides.items() if len(ps) != 2))))
    return out


class Coverage(NamedTuple):
    """The requested weeks, sorted into what the betting market has done about them.

    Typed rather than a dict, because `RemainingPlan` used to copy `covered`, `missing` and
    `thin` out of it one key at a time -- three fields restating one answer, and a reader of
    the remaining plan could not tell they came from a single question.

    The per-week game counts that decide these three are not carried. They were, and nothing
    read them: `thin` is the question anyone actually asks of a count, and a field no caller
    needs is a promise this has to keep.
    """
    covered: list[int]
    missing: list[int]
    thin: list[int]


def coverage(grid: pl.DataFrame, weeks: Sequence[int],
             pool: PoolConfig | None = None) -> Coverage:
    """Which requested weeks the betting market has actually priced.

    In August the board runs a handful of weeks deep, so a solve over "the season" quietly
    becomes a solve over whatever is posted. A remaining plan is still the best available
    answer for the weeks it covers -- it just is not a season, and must not print like one.

    **Covered is the pickable half of `week_fixtures`**, which is the question this asks: can
    `solve` plan the week. It is not the drawable half, and the difference is real rather than
    an oversight -- a fixture priced on one side only offers a team to pick and no game to
    play out, so it is covered here and refused by `pool.weeks_from_grid`. Named, because it
    was decided twice by two functions that never mentioned each other.
    """
    fx = week_fixtures(grid, weeks, pool)
    # Rows, for `thin` alone -- `THIN_ROWS` is a row count and says so.
    counts = {int(r["week"]): int(r["len"])
              for r in grid.filter(pl.col("win_prob") > MIN_PROB)
              .group_by("week").len().iter_rows(named=True)}
    return Coverage(
        covered=[f.week for f in fx if len(f.pickable) >= f.needs],
        missing=[f.week for f in fx if len(f.pickable) < f.needs],
        thin=[f.week for f in fx if 0 < counts.get(f.week, 0) < THIN_ROWS],
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

    Returned together because the picks alone are unreadable: a survival probability means
    nothing without the weeks it is over, a reader looking at a plan that starts in week 9
    should not have to infer why, and a week the betting market has not priced still needs a
    pick from the entrant. The site panel and the CLI both print all of it.

    `coverage` rides along whole rather than unpacked into three lists here, and `survival`
    is a property rather than a call every caller makes on the way out -- both callers did
    `survival(got.picks)` on the next line, which is a behaviour of the remaining plan and not
    of the caller.
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
                   at: datetime | None = None,
                   pool: PoolConfig | None = None) -> RemainingPlan:
    """The remaining plan: the weeks still ahead, against the teams still unspent.

    **From here, not from week 1.** Survivor is one assignment problem *because* spending a
    team early costs you that team later -- so a grid that still prices played weeks hands
    the solver its strongest teams for games that are over, and every remaining pick comes
    from a pool degraded by picks that were never available. The reported survival
    probability is then the product over games already won or lost. Both are wrong
    in-season and neither shows in the output: a whole-season solve prints exactly like a
    plan from here.

    **Written once, because it was written twice.** These five steps -- what is ahead, which
    weeks are behind, what those weeks spent, which of the weeks left are priced, solve
    against the rest -- were verbatim in `hub.publish.survivor` and in this module's `main`.
    (It read "six steps" over five items, in both copies, until issue #53: a hand-kept count
    of a list printed beside it.) That sequence *is* the rule issue #24 was about, so a drift
    between the copies would have the panel and the CLI planning different seasons with
    nothing in either output saying so.

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
    cov = coverage(ahead, [w for w in range(1, season_weeks + 1) if w not in behind], pool)
    if not cov.covered:
        raise Infeasible(f"no week in 1-{season_weeks} has a posted spread yet")
    return RemainingPlan(
        picks=solve(ahead, weeks=cov.covered, spent=spent, pool=pool),
        coverage=cov, played=behind, spent=spent,
        snapshot_only=snapshot_only_weeks(ahead, cov.covered))


def survival(plan: pl.DataFrame) -> float:
    """Probability of surviving every week in the frame given."""
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
    `hub.models.ratings.rated_games`'s -- `hub.schedule`'s number, quarterback-adjusted
    where the staleness field marks no live price (#218) -- so they cannot disagree about
    which number that is either. They did, for a day: the weekly prediction moved onto the
    dated snapshots and this was left reading nflverse's own field, which upstream leaves
    empty for the late season. That planned twelve of eighteen weeks and reported the rest
    unpriced while the store held every game of the season, week 18 included. The survivor
    horizon is where the adjustment matters most: for most of it the betting market's
    number is a posted lookahead that no news has touched, and a starter ruled out in
    October reaches a week-14 pick through nothing else.

    Spending a team early costs you that team later, so a plan over twelve weeks followed by
    a plan over the remaining six, with the best teams already gone, is strictly worse than
    one plan over eighteen. Which weeks are reachable is therefore not a display detail.

    A game neither source prices is dropped rather than filled at a coin flip, and
    `coverage` still names the week so an entrant knows it is theirs to fill.

    Each row also carries the `game_id` it came from. Nothing reads it yet: it is here so
    that a week taking two picks can be stopped from taking both sides of one fixture, which
    is unreachable while a week takes one pick and guaranteed fatal once it takes two.

    And since #218 each row carries the game's `close_spread` (the home side's, as rated),
    `qb_adjustment` and `adjusted_by`, so the CLI can say once what the adjustment did to
    the season it planned -- `hub.models.quarterback.report_line` reads exactly those three.
    """
    from hub.models import ratings
    from hub.models.market import MARGIN_SD, normal_cdf

    games, _ = ratings.rated_games(season, at=at, cache=cache, base=base)
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
        # The fixture both rows came from. A week that takes *two* picks must not take both
        # sides of one game, and nothing else on the row can say which rows those are:
        # `kickoff` groups a dozen unrelated Sunday-afternoon games into one slot, and
        # `result` is null for every week still being planned. nflverse's own `game_id`
        # rather than a key assembled here, because `hub.schedule` already carries it and a
        # second spelling of one identifier is the drift this module keeps being bitten by.
        gid = r["game_id"]
        rated = (float(r["close_spread"]), r.get("qb_adjustment"), r.get("adjusted_by"))
        rows.append((int(r["week"]), r["home_team"], home_p, moving, kick, res, gid, *rated))
        rows.append((int(r["week"]), r["away_team"], 1.0 - home_p, moving, kick, res, gid,
                     *rated))
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows],
                         "moving_field": [r[3] for r in rows],
                         "kickoff": [r[4] for r in rows],
                         "result": [r[5] for r in rows],
                         "game_id": [r[6] for r in rows],
                         "close_spread": [r[7] for r in rows],
                         "qb_adjustment": [r[8] for r in rows],
                         "adjusted_by": [r[9] for r in rows]},
                        schema={"week": pl.Int64, "team": pl.Utf8, "win_prob": pl.Float64,
                                "moving_field": pl.Boolean, "kickoff": pl.Datetime,
                                "result": pl.Float64, "game_id": pl.Utf8,
                                "close_spread": pl.Float64, "qb_adjustment": pl.Float64,
                                "adjusted_by": pl.Utf8})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.survivor",
        description="Plan a full survivor season as one assignment problem.")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--weeks", type=int, default=NFL_WEEKS)
    a = ap.parse_args(argv)

    try:
        grid = grid_from_schedule(a.season)
    except Exception as e:
        return unavailable("hub.season.survivor", f"the {a.season} schedule and its prices", e)
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
        print(f"  {len(got.played)} week(s) already played and absent from this plan: "
              + ", ".join(f"wk {w}" for w in got.played))
    if got.spent:
        print(f"  unavailable, already spent: {', '.join(got.spent)}")
    for r in got.picks.iter_rows(named=True):
        thin = "  (thin: one or two games priced)" if r["week"] in cov.thin else ""
        print(f"    wk {r['week']:>2}  {r['team']:<4} {r['win_prob']:.3f}{thin}")
    wk_n = got.picks["week"].n_unique()
    print(f"  survives the {wk_n} planned weeks "
          f"({got.picks.height} picks): {got.survival:.1%}")
    # #218's last criterion, for the survivor numbers: once, over the games the grid holds
    # -- one row per game rather than per side, so a game is counted once.
    if "adjusted_by" in grid.columns and "game_id" in grid.columns:
        from hub.models import quarterback
        print(f"  {quarterback.report_line(grid.unique(subset=['game_id'], keep='first'))}")
    # What the snapshot store actually buys, said out loud. These are the weeks nflverse's
    # lookahead field does not price, and the difference between a season plan and most of
    # one -- a team spent in week 3 is unavailable in week 17 whether or not this plan could
    # see week 17 when it chose.
    if got.snapshot_only:
        print(f"  {len(got.snapshot_only)} of these weeks are priced only by the dated "
              "snapshots: " + ", ".join(f"wk {w}" for w in got.snapshot_only))
    if cov.missing:
        # Not a failure: weeks with no spread are weeks the betting market has not posted,
        # plan over what exists beats no plan. But an entrant still has to pick in them.
        print("  not priced yet, so absent from this plan and still needing a pick: "
              + ", ".join(f"wk {w}" for w in cov.missing))
        print("  re-run once those weeks are on the board -- the teams spent early are "
              "not available to cover them.")
    print("  objective is survival only -- pool-aware play needs the pool config that "
          "docs/decisions.md still lists as open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
