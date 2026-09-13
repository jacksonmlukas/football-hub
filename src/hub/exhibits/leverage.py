"""What this league's structure actually rewards, measured rather than argued.

**This is an exhibit.** It was `hub.draft.leverage` until #198, a CLI with no importer
sitting in the draft package as if the product read it; the package docstring says what an
exhibit is. The measurement is `docs/six-of-twelve.md`, and this is the harness that re-runs
it -- nothing else, and nothing in `hub.draft` reaches it.

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

    uv run python -m hub.exhibits.leverage
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

import numpy as np

from hub.declare import not_an_input
from hub.draft.season import (
    PLAYOFF_ROUNDS,
    PLAYOFF_TEAMS,
    REG_SEASON_WEEKS,
    WEEKLY_K,
    WEEKLY_K_POOLED,
    champion,
    seed_table,
    simulate_weeks,
    talent_cv_for,
)

TEAMS = 12
CHUNK = 4000

# One archetype roster replicated across all twelve teams: 14 players in ESPN's shape. The
# league is symmetric by construction so that any asymmetry in the output is the structure
# under test rather than the inputs.
POS = np.array(["QB", "QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "WR",
                "TE", "TE", "RB"])
MU = not_an_input(
    np.array([19., 11., 15., 12., 10., 7., 14., 12., 10., 8., 6., 9., 5., 5.]),
    "a synthetic fixture league for the variance sweep exhibit, which predicts "
    "nothing and is not a dependency of the product")
# The square-root law from docs/weekly-spread.md, not the constant it replaced. This file
# used to hardcode MU * 0.55 and so kept simulating under a superseded model.
SD = np.array([WEEKLY_K.get(str(p), WEEKLY_K_POOLED) for p in POS]) * np.sqrt(MU)

N = len(POS)
POOL_POS = np.tile(POS, TEAMS)
ROSTERS = [np.arange(t * N, (t + 1) * N) for t in range(TEAMS)]
# `champion_probability` used to recycle regular-season weeks 1-3 as the three playoff
# rounds, so this file simulated its own seventeen weeks and forked the seeding loop and the
# bracket to escape it. season.py now gives the playoffs their own draws, and both live
# there -- this is the same constant, no longer a workaround.
SIM_WEEKS = REG_SEASON_WEEKS + PLAYOFF_ROUNDS

# How far a sweep row's calibrated team mean may sit from its target and still be read as a
# fixed-mean comparison. #173's second criterion is that this is *stated* -- an unstated
# tolerance is one nobody can find a row outside.
#
# A setting rather than a measurement, and its size is set by the bisection rather than by
# anything about football: `calibrate` stops at a bracket 1e-3 wide in the multiplier, and
# the mean is very nearly proportional to it, so a converged row lands inside about 0.1%.
# Half a per cent leaves room for the Monte Carlo wobble in `team_mean` without admitting the
# 15% the uncalibrated sweep produced -- the confound this module exists to remove is two
# orders of magnitude larger than this line.
CALIBRATION_TOL = not_an_input(
    0.005,
    "the tolerance the exhibit checks its own fixture's calibration to; an exhibit "
    "predicts nothing and is not a dependency of the product")


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

    wins, seeds = seed_table(pts)
    return pts, wins, seeds


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
        champ += sum(champion(pts, seeds, s) == 0 for s in range(n))
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
            won[list(seeds[s]).index(champion(pts, seeds, s))] += 1
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


def calibrate(target: float, vol: float = 1.0,
              cv_mult: float | Callable[[float], float] = 1.0,
              lo: float = 0.3, hi: float = 1.8) -> float:
    """The projection multiplier that puts the team's mean weekly score back on `target`.

    **`cv_mult` may be a function of the multiplier being solved for, and issue #173 is what
    happens when it cannot be.** The weekly-spread sweep holds *season-long* spread fixed in
    absolute points while it rescales the mean, so its talent multiplier is `k0 / k` -- a
    function of the very `k` this bisection is looking for. With only a float accepted, the
    loop below solved at `cv_mult = 1.0` and the row was then simulated at `k0 / k`, so the
    mean the row reported as fixed was not the mean it had been calibrated to. Passing a
    callable closes the loop: each probe is evaluated at the settings that probe implies.

    Still one bisection and not a fixed-point iteration, because `team_mean(k, vol, k0/k)` is
    monotone in `k` -- raising the multiplier raises the mean while the absolute talent
    spread it is paired with stays put, so there is exactly one crossing and bracketing finds
    it directly.
    """
    at: Callable[[float], float] = (
        cv_mult if callable(cv_mult) else (lambda _k, c=float(cv_mult): c))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if team_mean(mid, vol, at(mid)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)


def calibrated(target: float, vol: float = 1.0,
               cv_mult: float | Callable[[float], float] = 1.0) -> tuple[float, float, float]:
    """`(multiplier, the talent multiplier it resolves to, the team mean it achieves)`.

    Three numbers rather than one, because the settings a row is *evaluated* at are what
    #173 is about and a caller that re-derives them is a caller that can re-derive them
    differently -- which is exactly what the volatility loop did. `sweep_row` is the only
    caller and it feeds all three onward; nothing recomputes `k0 / k` a second time.

    `calibrate` is unchanged and still returns the multiplier on its own: the multiplier loop
    was already consistent, and this adds measurement beside it rather than a different
    calibration. `test_the_multiplier_loop_calibration_is_unchanged` holds that.
    """
    k = calibrate(target, vol=vol, cv_mult=cv_mult)
    at = cv_mult(k) if callable(cv_mult) else float(cv_mult)
    return k, at, team_mean(k, vol, at)


def sweep_row(target: float, *, vol: float = 1.0,
              cv_mult: float | Callable[[float], float] = 1.0,
              n_sims: int = 20000, seed: int = 0) -> tuple[dict, float]:
    """One row of the variance sweep: `(outcomes, how far its mean missed `target`)`.

    **This function is #173's fix, and the fix is that it exists.** The bug was not a wrong
    constant -- it was that the loop calibrated in one statement and simulated in another,
    with the talent multiplier written out twice and the two spellings disagreeing: solved at
    the default 1.0, run at `k0 / k`. So the row's target team mean was not the mean it had
    been calibrated to, while the row's own label said it was.

    Here the resolved settings are computed once and every consumer is handed the same ones.
    `team_mean` measures the mean at those settings and `simulate` runs at those settings,
    from one `at`, so a row cannot be calibrated under settings it is not simulated under --
    not because a caller remembered to match them, but because there is nothing to mismatch.
    A checking loop that only *compared* two spellings would go on passing the moment
    somebody wrote the third.

    The miss comes back beside the outcomes rather than being raised on. A row outside
    `CALIBRATION_TOL` is a finding about the sweep, and refusing to render the table would
    take the other seven rows down with it.
    """
    k, at, got = calibrated(target, vol=vol, cv_mult=cv_mult)
    outcomes = simulate(k=k, vol=vol, cv_mult=at, n_sims=n_sims, seed=seed)
    return outcomes, missed_target(got, target)


def missed_target(got: float, target: float) -> float:
    """How far a calibrated row's mean sits from its target, as a fraction of the target."""
    return abs(got - target) / target if target else float("inf")


def _miss(err: float) -> str:
    """The `mean err` cell, marked when the row is not the fixed-mean comparison it claims."""
    return f"{err:.2%}" + ("" if err <= CALIBRATION_TOL else " OFF")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.exhibits.leverage",
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
    print(f"  {'roster':>8} {'kind':>8} {'x':>5} {'playoff':>9} {'bye':>7} {'title':>7}"
          f" {'mean err':>9}")
    for label, k0 in (("weak", 0.90), ("strong", 1.10)):
        tgt = team_mean(k=k0)
        for vol in (0.7, 1.8):
            # `cv_mult` as a function of the multiplier being solved for, which is #173. The
            # row holds season-long spread fixed in *absolute* points while the mean is
            # rescaled, so its talent multiplier is `k0 / k` -- and this loop used to solve
            # at the default 1.0 and then simulate at `k0 / k`, so the mean it reported as
            # fixed was not the mean it had been calibrated to. `sweep_row` resolves the
            # settings once and hands the same ones to both.
            r, err = sweep_row(tgt, vol=vol, cv_mult=lambda m, b=k0: b / m, n_sims=n)
            print(f"  {label:>8} {'weekly':>8} {vol:>5.1f} {r['playoff']:>8.1%} "
                  f"{r['bye']:>7.1%} {r['title']:>7.1%} {_miss(err):>9}")
        for cvm in (0.5, 2.0):
            # Unchanged by #173: this loop already calibrated at the multiplier it evaluates,
            # and it is the pattern the one above was brought onto.
            r, err = sweep_row(tgt, cv_mult=cvm, n_sims=n)
            print(f"  {label:>8} {'season':>8} {cvm:>5.1f} {r['playoff']:>8.1%} "
                  f"{r['bye']:>7.1%} {r['title']:>7.1%} {_miss(err):>9}")
    print(f"\n  mean err is the calibrated team mean against its target; the sweep's claim "
          f"is that\n  only one quantity moved per row, so anything over {CALIBRATION_TOL:.1%}"
          f" is marked OFF and the row is not\n  a fixed-mean comparison. See #173.")
    print("\n  see docs/six-of-twelve.md; directions are the finding, not the decimals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
