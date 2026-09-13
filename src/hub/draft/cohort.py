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

**Nothing here imports the harness.** Until #257 `market_strategy` was read from
`hub.draft.backtest`, and the harness imports the exhibit -- so every Gate that scored a
Cohort loaded championship equity on the way to its own verdict, and the contract test that
holds the exhibit off the product never asked about a Gate. The strategy is declared with
the room in `hub.draft.optimize`, and `tests/contracts/test_the_exhibit_is_not_a_dependency.py`
now walks the three season Gates' imports, function-local ones included.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl

from hub.config import RosterConfig
from hub.declare import chosen
from hub.draft.board import BuildReport
from hub.draft.optimize import DEFAULT_ROUNDS, market_strategy, simulate_remaining_draft
from hub.draft.state import DraftState

_CFG = RosterConfig()
SLOT = _CFG.slot
TEAMS = _CFG.teams

# Fourteen rounds, not the league's sixteen: the last two are kickers and defences, which the
# board does not carry (ADR-0008). One number, declared in `hub.draft.optimize` where the
# simulator defaults to it, and reached here under the name both season-side gates and
# the digest already use -- it was a second literal 14 until #200, under a second
# reason ("bench depth beyond the eight starting slots"), and two declarations of one number
# are two numbers as soon as one is edited. Twenty drafts a season is what every Gate has
# always used and what every published result was measured on: three CLIs default to this
# name rather than restating the literal. These are the recipe, and changing one invalidates
# a recorded number rather than improving a default.
ROUNDS = chosen(DEFAULT_ROUNDS)
DRAFTS = chosen(20)


def seed_for(seed: int, season: int, k: int) -> int:
    """The draw for one draft, stated once.

    The season is in it so that a four-season gate scores four cohorts rather than one cohort
    four times, and `k` so that twenty drafts are twenty drafts. It was written out in both
    gates; a formula copied by hand into two places is one that eventually differs in one.

    **The draft backtest does not draw its rooms from this, and the difference is deliberate
    rather than a drift.** `backtest.draft_root` descends from the `SeedSequence` tree #195
    built, because arm B evaluates draft futures *inside* the room it is scored in and the
    integer arithmetic let rollout 0 be that room. A Cohort has no arm B and no rollout under
    it -- one market draft per `k`, scored once -- so the leak #195 closed cannot occur here,
    and both season-side gates' published figures were measured on exactly this draw
    (`test_the_cohort_is_what_the_gates_used_to_build_for_themselves` pins it). Moving it
    onto the tree would be a re-measurement of two Gates that persist nothing and cannot be
    re-run, not a refactor; #200 states the reason here instead. `tests/unit/test_cohort.py`
    guards that neither recipe is written anywhere but its declaration.
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
           my_slot: int = SLOT, teams: int = TEAMS, rounds: int = ROUNDS,
           report: BuildReport | None = None) -> Cohort:
    """Draft `drafts` rosters from `board`, and say who was left.

    Every seat plays, not only the scored one: the pool a waiver arm adds from has to exclude
    the eleven other rosters, or a gate would offer a player another team already holds.

    **`report` is the recorded answer to "what did the run that built this frame do", and it
    travels rather than being re-derived.** #199 gave the Board a report that rides with the
    frame and #146 moved the consumer sites inside `hub.draft` onto it; this was the last seam
    where a caller *held* one and dropped it. Both season-side gates unpack
    `board, report = board_as_of(yr)` -- they have to, `require_corrections` reads it -- and
    then handed the frame on alone, so `optimize.simulate_remaining_draft` sent it through
    `board.report_for`, which derives a fresh one by looking at the frame's columns.

    Nothing ranks differently for it today, and that is the reason to fix it rather than a
    reason not to. `BuildReport`'s own docstring says the report layer exists because consumers
    "used to infer what had happened by sniffing for columns"; a caller holding the answer and
    letting it be inferred again is that same act one level up, and the two agreeing today is
    exactly the condition that stops holding without anything saying so. A derivation cannot
    see an all-null stage at all, which is the case where the two part company.

    Optional, and it stays optional -- the expand half of expand-then-contract, so every caller
    and fixture that has only a frame keeps working and gets the derivation it already got.
    """
    pos = [str(p) if p else "NA" for p in board["pos"].to_list()]
    n = board.height
    rosters, pool = [], []
    for k in range(drafts):
        room = simulate_remaining_draft(
            board, DraftState(taken=[]), my_slot=my_slot, teams=teams, rounds=rounds,
            rng=np.random.default_rng(seed_for(seed, season, k)),
            my_pick=market_strategy(), report=report)
        taken = {int(i) for seat in room for i in seat}
        rosters.append([int(i) for i in room[my_slot - 1]])
        pool.append([i for i in range(n) if i not in taken])
    return Cohort(rosters, pool, pos)
