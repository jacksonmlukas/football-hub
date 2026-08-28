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
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from hub.config import SEASON_AHEAD

# Below this a team is treated as unpickable rather than fed to log(). A survivor pick at
# 1% is never the answer, and log(0) is negative infinity.
MIN_PROB = 1e-4

# Fewer than three priced games in a week. The "choice" is then which side of one or two
# games to take, which is worth saying out loud before anyone treats it as a plan.
THIN_WEEK = 6


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


def solve(grid: pl.DataFrame, weeks: Sequence[int] | None = None) -> pl.DataFrame:
    """One team per week, no repeats, maximising the probability of surviving them all.

    `grid` is (week, team, win_prob) for every team playing in every week.
    """
    import pulp

    weeks = list(weeks) if weeks is not None else sorted(set(grid["week"].to_list()))
    usable = grid.filter(pl.col("win_prob") > MIN_PROB)

    options = [(int(r["week"]), str(r["team"]), float(r["win_prob"]))
               for r in usable.iter_rows(named=True) if int(r["week"]) in weeks]
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


def coverage(grid: pl.DataFrame, weeks: Sequence[int]) -> dict:
    """Which requested weeks the market has actually priced.

    In August the board runs a handful of weeks deep, so a solve over "the season" quietly
    becomes a solve over whatever is posted. The plan is still the best available answer
    for the weeks it covers -- it just is not a season, and must not print like one.
    """
    usable = grid.filter(pl.col("win_prob") > MIN_PROB)
    counts = {int(r["week"]): int(r["len"])
              for r in usable.group_by("week").len().iter_rows(named=True)}
    weeks = list(weeks)
    return {
        "covered": [w for w in weeks if counts.get(w, 0) > 0],
        "missing": [w for w in weeks if counts.get(w, 0) == 0],
        "thin": [w for w in weeks if 0 < counts.get(w, 0) < THIN_WEEK],
        "counts": counts,
    }


def survival(plan: pl.DataFrame) -> float:
    """Probability of surviving every week in the plan."""
    out = 1.0
    for p in plan["win_prob"].to_list():
        out *= float(p)
    return out


def grid_from_schedule(season: int, cache: Path | None = None) -> pl.DataFrame:
    """Win probability for every team in every week it plays.

    Uses the same spread-to-probability conversion as `MarketBaseline`, so a survivor pick
    and a weekly prediction cannot disagree about the same game.
    """
    from hub.fetch import nflverse
    from hub.models.market import MARGIN_SD, normal_cdf

    sched = nflverse.load("schedules", seasons=[season], cache=cache)
    rows = []
    for r in sched.filter(pl.col("spread_line").is_not_null()).iter_rows(named=True):
        # spread_line is positive when the home team is favoured.
        home_p = normal_cdf(float(r["spread_line"]) / MARGIN_SD)
        rows.append((int(r["week"]), r["home_team"], home_p))
        rows.append((int(r["week"]), r["away_team"], 1.0 - home_p))
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows]})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.survivor",
        description="Plan a full survivor season as one assignment problem.")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--weeks", type=int, default=18)
    a = ap.parse_args(argv)

    grid = grid_from_schedule(a.season)
    requested = list(range(1, a.weeks + 1))
    cov = coverage(grid, requested)
    weeks = cov["covered"]
    if not weeks:
        print(f"hub.season.survivor: no week in 1-{a.weeks} has a posted spread yet",
              file=sys.stderr)
        return 1
    try:
        plan = solve(grid, weeks=weeks)
    except Infeasible as e:
        print(f"hub.season.survivor: {e}", file=sys.stderr)
        return 1

    print(f"  survivor plan, {a.season}, {len(weeks)} of {len(requested)} weeks priced")
    for r in plan.iter_rows(named=True):
        thin = "  (thin: one or two games priced)" if r["week"] in cov["thin"] else ""
        print(f"    wk {r['week']:>2}  {r['team']:<4} {r['win_prob']:.3f}{thin}")
    s = survival(plan)
    print(f"  survives the {plan.height} planned weeks: {s:.1%}")
    if cov["missing"]:
        # Not a failure: weeks with no spread are weeks the market has not posted, and a
        # plan over what exists beats no plan. But an entrant still has to pick in them.
        print("  not priced yet, so absent from the plan and still needing a pick: "
              + ", ".join(f"wk {w}" for w in cov["missing"]))
        print("  re-run once those weeks are on the board -- the teams spent early are "
              "not available to cover them.")
    print("  objective is survival only -- pool-aware play needs the pool config that "
          "docs/decisions.md still lists as open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
