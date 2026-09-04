"""The rosters a season-level gate is scored on.

A **Cohort** is many simulated drafts from one seat, across a season: the sample a gate
measures an arm against. It is not the **Room**, which is the eleven opponents inside a single
draft.

It existed twice. `hub.season.lineup_gate` and `hub.season.weekly_gate_data` both drafted
twenty rosters a season, in the configured seat, over fourteen rounds, with the market
strategy, seeded `seed + 1000 * season + k` -- and each wrote that recipe out. The seed formula
was hand-copied, so a drift in either would have left two gates scoring different cohorts while
both described themselves as identical harnesses.

**The cause was a narrow interface.** `backtest.play` returns player names and positions and
throws away the board indices and the undrafted remainder. The weekly gate's waiver arm needs
both -- a roster that adds and drops cannot be a fixed list of names -- so it could not use
`play`, and the second copy went inside a `main()` where no test reached it. That is the shape
`tests/contracts/test_cli_surface.py` was written after three bugs of: coverage measures lines
executed, not whether the seam between a module and the real world exists.

So this returns the widest useful form and lets each gate narrow it. Building the narrow form
and reconstructing the wide one is what produced two copies in the first place.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl

from hub.config import RosterConfig
from hub.draft.backtest import market_strategy
from hub.draft.optimize import simulate_remaining_draft
from hub.draft.state import DraftState

_CFG = RosterConfig()
SLOT = _CFG.slot
TEAMS = _CFG.teams

# Fourteen rounds, not the league's sixteen: the last two are kickers and defences, which the
# board does not carry (ADR-0008). Twenty drafts a season is what both gates have always used
# and what every published result was measured on -- these are the recipe, and changing one
# invalidates a recorded number rather than improving a default.
ROUNDS = 14
DRAFTS = 20


def seed_for(seed: int, season: int, k: int) -> int:
    """The draw for one draft, stated once.

    The season is in it so that a four-season gate scores four cohorts rather than one cohort
    four times, and `k` so that twenty drafts are twenty drafts. It was written out in both
    gates; a formula copied by hand into two places is one that eventually differs in one.
    """
    return seed + 1000 * season + k


class Cohort(NamedTuple):
    """One season's simulated rosters, as indices into the board they were drafted from.

    Indices rather than names, for the reason `ADR-0008` gives for the simulator: a name has
    to be looked up again to be used, and every lookup is a chance to look up the wrong frame.
    `pos` is carried alongside so a caller filling a lineup does not re-read the board either.
    """
    rosters: list[list[int]]      # per draft: the scored seat's players
    pool: list[list[int]]         # per draft: everyone no seat drafted
    pos: list[str]                # per board row: position, "NA" where the board has none


def cohort(board: pl.DataFrame, season: int, *, drafts: int = DRAFTS, seed: int = 0,
           my_slot: int = SLOT, teams: int = TEAMS, rounds: int = ROUNDS) -> Cohort:
    """Draft `drafts` rosters from `board`, and say who was left.

    Every seat plays, not only the scored one: the pool a waiver arm adds from has to exclude
    the eleven other rosters, or a gate would offer a player another team already holds.
    """
    pos = [str(p) if p else "NA" for p in board["pos"].to_list()]
    n = board.height
    rosters, pool = [], []
    for k in range(drafts):
        room = simulate_remaining_draft(
            board, DraftState(taken=[]), my_slot=my_slot, teams=teams, rounds=rounds,
            rng=np.random.default_rng(seed_for(seed, season, k)),
            my_pick=market_strategy())
        taken = {int(i) for seat in room for i in seat}
        rosters.append([int(i) for i in room[my_slot - 1]])
        pool.append([i for i in range(n) if i not in taken])
    return Cohort(rosters, pool, pos)
