"""What this league's structure actually rewards, measured rather than argued.

`docs/championship-leverage.md` derives its draft-time strategy from *"12 teams, 8 make
playoffs, 3 weeks (15-17), no byes"*, concludes the regular season is *"nearly a
formality"* and that *"dP(champ)/d(regular-season win) is close to zero"*, and lands on
*"never sacrifice ceiling for a marginal regular-season win."* The live league is 6 of 12
with byes for seeds 1-2. This module measures the same quantities against the real shape.

The findings are written up in `docs/six-of-twelve.md`. The short version is that the
doc's ceiling instinct survives for one quantity and dies for another, and it treats them
as one thing:

- **Season-long outcome spread** -- how uncertain it is what a player *becomes* -- is worth
  paying for at every roster strength, because the payoff in seeding is steeply convex.
- **Weekly boom-bust** -- spread in what he does on a given Sunday, given his talent -- is
  not. Head-to-head wastes surplus, so at a fixed mean it is neutral for a weak roster and
  negative for a strong one.

**The measurement trap this module exists to avoid.** A starting lineup is the best legal
subset of a roster, so it is a max. Raising player spread therefore raises the expected
maximum: the naive "same mean, more variance" sweep is silently adding points, and finds
that variance helps enormously at every strength. `calibrate` rescales projections to put
the team's mean weekly score back where it was, so the sweep moves one quantity at a time.
`team_mean` at two spreads shows the size of the trap: 1.8x spread is worth +15% of mean.

Everything here is inside the model. Opponents are twelve copies of one synthetic roster
and players are independent, so the correlation layer does not exist. `TALENT_CV` -- which
the season-long sweep below is a sweep in -- was fitted on 2026-08-23 against this league's
own past drafts (`hub.draft.calibrate`, `docs/talent-cv.md`); refitting it from 0.35 to 0.41
strengthened every conclusion here rather than changing one. Read the directions, not the
decimals.

    uv run python -m hub.draft.leverage
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np

from hub.draft.season import (PLAYOFF_TEAMS, REG_SEASON_WEEKS, WEEKLY_K,
                              WEEKLY_K_POOLED, _round_robin, simulate_weeks,
                              talent_cv_for)

TEAMS = 12
CHUNK = 4000

# One archetype roster replicated across all twelve teams: 14 players in ESPN's shape. The
# league is symmetric by construction so that any asymmetry in the output is the structure
# under test rather than the inputs.
POS = np.array(["QB", "QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "WR",
                "TE", "TE", "RB"])
MU = np.array([19., 11., 15., 12., 10., 7., 14., 12., 10., 8., 6., 9., 5., 5.])
# The square-root law from docs/weekly-spread.md, not the constant it replaced. This file
# used to hardcode MU * 0.55 and so kept simulating under a superseded model.
SD = np.array([WEEKLY_K.get(str(p), WEEKLY_K_POOLED) for p in POS]) * np.sqrt(MU)

N = len(POS)
POOL_POS = np.tile(POS, TEAMS)
ROSTERS = [np.arange(t * N, (t + 1) * N) for t in range(TEAMS)]
# Seventeen weeks so the three playoff rounds get their own draws. `champion_probability`
# recycles regular-season weeks 1-3, which is fine for ranking rosters but would couple a
# team's title odds to its week 1 result -- not something to measure a bye against.
SIM_WEEKS = REG_SEASON_WEEKS + 3


def _talent_cv(cv_mult: float) -> np.ndarray:
    """Per-player talent dispersion for the pool, with team 0's scaled by `cv_mult`."""
    cv = talent_cv_for(POOL_POS)
    cv[:N] *= cv_mult
    return cv


def _season(k, vol, cv_mult, n, base_seed):
    """Weekly points and seeds for one chunk of simulated seasons. Team 0 is the subject.

    Draws go through `simulate_weeks` rather than being re-implemented here. They were
    re-implemented once, and this file silently kept the old weekly model -- proportional
    spread, normal draws, spread keyed to the projection -- after the simulator moved on.
    """
    mu = np.tile(MU, TEAMS).astype(float)
    sd = np.tile(SD, TEAMS).astype(float)
    mu[:N] *= k
    sd[:N] *= vol

    # Common random numbers: the seed depends only on position in the run, so two
    # configurations see the same underlying stream and their difference is the change.
    pts = simulate_weeks(ROSTERS, mu, sd, POOL_POS, n_sims=n, weeks=SIM_WEEKS,
                         rng=np.random.default_rng(base_seed),
                         talent_cv=_talent_cv(cv_mult))

    wins = np.zeros((n, TEAMS))
    for w, pairs in enumerate(_round_robin(TEAMS, REG_SEASON_WEEKS)):
        for a, b in pairs:
            a_won = pts[:, w, a] > pts[:, w, b]
            wins[:, a] += a_won
            wins[:, b] += ~a_won

    # Seed on wins, tie-break on total points -- playoffSeedingRule TOTAL_POINTS_SCORED.
    key = wins * 10_000.0 + pts[:, :REG_SEASON_WEEKS, :].sum(axis=1)
    return pts, wins, np.argsort(-key, axis=1)[:, :PLAYOFF_TEAMS]


def _champion(pts, seeds, s):
    """Winner of one bracket: seeds 1-2 bye, 3v6 and 4v5, then semis, then the final."""
    f = seeds[s]
    w = REG_SEASON_WEEKS
    alive = [f[0], f[1]]
    for a, b in ((f[2], f[5]), (f[3], f[4])):
        alive.append(a if pts[s, w, a] > pts[s, w, b] else b)
    semi = [alive[0] if pts[s, w + 1, alive[0]] > pts[s, w + 1, alive[3]] else alive[3],
            alive[1] if pts[s, w + 1, alive[1]] > pts[s, w + 1, alive[2]] else alive[2]]
    return semi[0] if pts[s, w + 2, semi[0]] > pts[s, w + 2, semi[1]] else semi[1]


def simulate(k: float = 1.0, vol: float = 1.0, cv_mult: float = 1.0,
             n_sims: int = 20000, seed: int = 0) -> dict:
    """Outcomes for team 0, whose projections are scaled by `k`, weekly spread by `vol`
    and season-long talent spread by `cv_mult`. Everyone else stays at baseline.
    """
    made = bye = champ = 0
    wins_sum = 0.0
    done = 0
    while done < n_sims:
        n = min(CHUNK, n_sims - done)
        pts, wins, seeds = _season(k, vol, cv_mult, n, seed * 1_000_003 + done)
        made += int((seeds == 0).any(axis=1).sum())
        bye += int((seeds[:, :2] == 0).sum())
        wins_sum += float(wins[:, 0].sum())
        champ += sum(_champion(pts, seeds, s) == 0 for s in range(n))
        done += n
    return {"playoff": made / n_sims, "bye": bye / n_sims, "title": champ / n_sims,
            "wins": wins_sum / n_sims}


def seed_value(n_sims: int = 20000, seed: int = 0) -> np.ndarray:
    """P(win the title | finishing in seed 1..6), pooled over twelve identical teams.

    Symmetric by construction, so this isolates what the bracket alone pays for seeding,
    with none of it attributable to the seeded team being better.
    """
    won = np.zeros(PLAYOFF_TEAMS)
    done = 0
    while done < n_sims:
        n = min(CHUNK, n_sims - done)
        pts, _, seeds = _season(1.0, 1.0, 1.0, n, seed * 1_000_003 + done)
        for s in range(n):
            won[list(seeds[s]).index(_champion(pts, seeds, s))] += 1
        done += n
    return won / n_sims


def team_mean(k: float = 1.0, vol: float = 1.0, cv_mult: float = 1.0,
              n: int = 60000, seed: int = 7) -> float:
    """Expected weekly points for one team at these settings.

    Not a diagnostic -- it is the control. See the module docstring on why a spread sweep
    without this measures points rather than variance.
    """
    pts = simulate_weeks([np.arange(N)], MU * k, SD * vol, POS, n_sims=n, weeks=1,
                         rng=np.random.default_rng(seed),
                         talent_cv=talent_cv_for(POS) * cv_mult)
    return float(pts[:, 0, 0].mean())


def calibrate(target: float, vol: float = 1.0, cv_mult: float = 1.0,
              lo: float = 0.3, hi: float = 1.8) -> float:
    """The projection multiplier that puts the team's mean weekly score back on `target`."""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if team_mean(mid, vol, cv_mult) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.leverage",
        description="Measure what 6-of-12 with two byes rewards.")
    ap.add_argument("--sims", type=int, default=20000)
    a = ap.parse_args(argv)
    n = a.sims

    print("  roster strength -> outcomes")
    print(f"  {'mu x':>6} {'E[wins]':>8} {'playoff':>9} {'bye':>7} {'title':>7}")
    prev = None
    for k in (0.85, 0.95, 1.00, 1.05, 1.15):
        r = simulate(k=k, n_sims=n)
        grad = ("" if prev is None else
                f"   {100 * (r['title'] - prev['title']) / (r['wins'] - prev['wins']):+.1f} pp/win")
        print(f"  {k:>6.2f} {r['wins']:>8.2f} {r['playoff']:>8.1%} {r['bye']:>7.1%} "
              f"{r['title']:>7.1%}{grad}")
        prev = r

    print("\n  what the bracket pays for seeding (identical rosters)")
    for i, p in enumerate(seed_value(n_sims=n), 1):
        print(f"    seed {i}{' (bye)' if i <= 2 else '      '}: {p:>6.1%}")

    print("\n  variance at a fixed team mean -- weekly spread vs season-long spread")
    print(f"  {'roster':>8} {'kind':>8} {'x':>5} {'playoff':>9} {'bye':>7} {'title':>7}")
    for label, k0 in (("weak", 0.90), ("strong", 1.10)):
        tgt = team_mean(k=k0)
        for vol in (0.7, 1.8):
            k = calibrate(tgt, vol=vol)
            r = simulate(k=k, vol=vol, cv_mult=k0 / k, n_sims=n)
            print(f"  {label:>8} {'weekly':>8} {vol:>5.1f} {r['playoff']:>8.1%} "
                  f"{r['bye']:>7.1%} {r['title']:>7.1%}")
        for cvm in (0.5, 2.0):
            k = calibrate(tgt, cv_mult=cvm)
            r = simulate(k=k, cv_mult=cvm, n_sims=n)
            print(f"  {label:>8} {'season':>8} {cvm:>5.1f} {r['playoff']:>8.1%} "
                  f"{r['bye']:>7.1%} {r['title']:>7.1%}")
    print("\n  see docs/six-of-twelve.md; directions are the finding, not the decimals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
