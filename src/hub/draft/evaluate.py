"""Evaluate a board by the roster it drafts, not by the order it prints.

Every other metric in `hub.draft.tune` scores one static ordering. A draft is not an
ordering. It is a sequence of decisions against a pool that eleven other people are
emptying, and a board is worth exactly the roster it ends up producing.

That difference is not academic. The top-50-points metric turned on two or three players
sitting nearest an arbitrary cut, so six seasons came to roughly fifteen player-outcomes
and a single torn ACL moved the pooled result by two units of t. Here every pick reads the
whole board, a trial is sixteen rounds, and each season is replayed from all twelve seats.

Scoring is starters only -- QB1 RB2 WR3 TE1 FLEX1 -- because a surplus player scores zero.
A board that stacks five good WRs has not built a better roster than one that fills a
lineup, and any metric that cannot see that is measuring the wrong thing.

    uv run python -m hub.draft.evaluate --sweep
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.draft.season import FLEX_FROM, STARTERS

TEAMS = 12
ROUNDS = 16

# How loosely opponents follow their board. Nobody drafts strictly off a list, and a room
# of perfect consensus-followers would make any deviation look better than it is.
OPP_NOISE = 8.0

MIN_SIGMA = 2.0


def starter_points(roster: pl.DataFrame) -> float:
    """Season points from the best legal lineup this roster can field."""
    total, leftovers = 0.0, []
    for pos, n in STARTERS.items():
        got = sorted(roster.filter(pl.col("pos") == pos)["actual_points"].to_list(),
                     reverse=True)
        total += sum(got[:n])
        if pos in FLEX_FROM and len(got) > n:
            leftovers.append(got[n])
    return total + (max(leftovers) if leftovers else 0.0)


def simulate_draft(pool: pl.DataFrame, lam: float, my_slot: int,
                   rng: np.random.Generator, teams: int = TEAMS,
                   rounds: int = ROUNDS, opp_noise: float = OPP_NOISE) -> list[list[int]]:
    """One snake draft. Returns row indices per team.

    I read a board adjusted by `lam`; the room reads consensus with noise. That is the
    comparison that matters -- does the adjustment help against a room following ECR?
    """
    ecr = pool["ecr"].to_numpy()
    z = np.nan_to_num(pool["z_regress"].fill_null(0.0).to_numpy(), nan=0.0).clip(-3, 3)
    mine = ecr * np.exp(-lam * z)
    theirs = ecr + rng.normal(0.0, opp_noise, ecr.size)

    my_order = np.argsort(mine)
    their_order = np.argsort(theirs)
    gone = np.zeros(ecr.size, dtype=bool)
    rosters: list[list[int]] = [[] for _ in range(teams)]

    for overall in range(1, teams * rounds + 1):
        rnd = (overall - 1) // teams + 1
        seat = (overall - 1) % teams
        team = seat if rnd % 2 else teams - 1 - seat
        order = my_order if team == my_slot - 1 else their_order
        live = order[~gone[order]]
        if live.size == 0:
            break
        pick = int(live[0])
        gone[pick] = True
        rosters[team].append(pick)
    return rosters


def trial(pool: pl.DataFrame, lam: float, my_slot: int,
          rng: np.random.Generator, **kw) -> float:
    """Starter points I end up with, drafting this board from this seat."""
    rosters = simulate_draft(pool, lam, my_slot, rng, **kw)
    return starter_points(pool[rosters[my_slot - 1]])


def evaluate(pool: pl.DataFrame, lams: Sequence[float], n_sims: int = 12,
             teams: int = TEAMS, seed: int = 0) -> pl.DataFrame:
    """Mean starter points per lambda, over every seat and `n_sims` draft realisations.

    Paired on purpose: every lambda faces the same opponent draws from the same seat, so
    the lift between two lambdas is a within-draft comparison rather than two independent
    noisy estimates. The same discipline as common random numbers in `hub.draft.optimize`,
    and for the same reason -- the difference is far smaller than either level.
    """
    rows = []
    base: np.ndarray | None = None
    for lam in lams:
        got = []
        for s in range(n_sims):
            for slot in range(1, teams + 1):
                # Same seed for the same (sim, slot) across every lambda: the room drafts
                # identically and only my board changes.
                rng = np.random.default_rng(seed + s * 1000 + slot)
                got.append(trial(pool, lam, slot, rng, teams=teams))
        arr = np.array(got)
        if base is None:
            base = arr
        diff = arr - base
        rows.append({
            "lam": float(lam),
            "starter_points": float(arr.mean()),
            "lift": float(diff.mean()),
            "lift_se": float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0,
            "n_trials": int(arr.size),
        })
    return pl.DataFrame(rows)


def best_lambda(evaluated: pl.DataFrame, min_sigma: float = MIN_SIGMA) -> float:
    """The best lambda whose lift clears the noise, else zero."""
    if evaluated.is_empty():
        return 0.0
    credible = evaluated.filter(
        (pl.col("lam") > 0)
        & (pl.col("lift") > 0)
        & (pl.col("lift") > min_sigma * pl.col("lift_se")))
    if credible.is_empty():
        return 0.0
    return float(credible.sort(["lift", "lam"], descending=[True, False])["lam"][0])


def main(argv: Sequence[str] | None = None) -> int:
    from hub.draft.tune import DEFAULT_GRID, holdout

    ap = argparse.ArgumentParser(
        prog="hub.draft.evaluate",
        description="Score projection_lambda by the roster it drafts.")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--signal-season", type=int, default=2023)
    ap.add_argument("--board-season", type=int, default=2024)
    ap.add_argument("--sims", type=int, default=12)
    a = ap.parse_args(argv)
    if not a.sweep:
        ap.print_help()
        return 0

    pool = holdout(a.signal_season, a.board_season)
    got = evaluate(pool, DEFAULT_GRID, n_sims=a.sims)
    print(f"  {a.signal_season}->{a.board_season}: {pool.height} players, "
          f"{got['n_trials'][0]:,} drafts per lambda")
    print(f"  {'lam':>6} {'starter pts':>12} {'lift':>9} {'se':>7} {'sigma':>7}")
    for r in got.iter_rows(named=True):
        sig = r["lift"] / r["lift_se"] if r["lift_se"] else 0.0
        print(f"  {r['lam']:>6.2f} {r['starter_points']:>12,.1f} {r['lift']:>+9.1f} "
              f"{r['lift_se']:>7.1f} {sig:>+7.2f}")
    print(f"  best: {best_lambda(got):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
