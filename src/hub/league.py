"""How this league is shaped, and the one rule for filling it.

A leaf, for the same reason `hub.paths` and `hub.names` are: it imports `hub.config` and
nothing else, so anything may ask how many receivers it starts without dragging a draft
simulator along.

These names lived in `hub.draft.season` -- a module whose actual job is simulating a season
to a champion in order to *score a draft*. That put the league's roster shape inside the
draft package, and four of the six `hub.season` modules reached back into it: `season -> draft`
was the second-heaviest edge in the codebase at twelve imports, five of them for nothing more
than `STARTERS` and `starting_lineup`.

**The codebase had already started this move.** `hub.models.weekly_screen` needed the fantasy
week list on 2026-09-04, could not import it -- `test_models_does_not_reach_into_draft` forbids
`models -> draft` -- and so `FANTASY_WEEKS` was moved to `hub.config`. That fix was right and
stopped one name short, which is why `REG_SEASON_WEEKS` was until now defined in `config`,
imported into `draft.season`, and imported back out of it by `season.lineup_gate`: a constant
round-tripping through a package that has nothing to do with it.

`config` declares what is *configurable* -- the roster slots, the season length. This module
holds what is *derived* from that and used everywhere, plus the rule that reads it. The two
week constants are re-exported so there is one address for "how does this league work"; they
are still declared in `config`, because that is the file a commissioner change edits.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hub.config import (  # noqa: F401  -- re-exported: see the module docstring
    FANTASY_WEEKS,
    REG_SEASON_WEEKS,
    RosterConfig,
    flex_capacity,
    flex_positions,
    required_starters,
)

# QB1 / RB2 / WR3 / TE1 / FLEX1 -- confirmed against the live league, not the ESPN default.
# Derived from `hub.config.RosterConfig` rather than restated: this module is "how a league
# works", so it is the right place to *read* the shape and the wrong place to declare it.
ROSTER = RosterConfig()
STARTERS = required_starters(ROSTER)
FLEX_FROM = flex_positions(ROSTER)
FLEX_SLOTS = ROSTER.flex
# The most flex-eligible players a team can start at once. Was a bare `7` in optimize.py.
FLEX_CAPACITY = flex_capacity(ROSTER)
PLAYOFF_TEAMS = 6
# Quarter-final, semi-final, final. The bracket needs its own weeks: scoring the playoffs on
# draws that already decided seeding couples a team's title odds to its week 1 result.
PLAYOFF_ROUNDS = 3


def starting_lineup(pos: Sequence[str], score: Sequence[float] | np.ndarray) -> list[int]:
    """Indices of the lineup: each required **Slot** to the best available, then the flex.

    One rule, in the module that owns the shape it reads. `STARTERS`, `FLEX_FROM` and
    `FLEX_SLOTS` are declared just above, and six modules import them -- so the rule that
    fills them belongs beside them rather than in whichever caller needed it first.

    It was written twice, character for character, in `hub.season.lineup_gate` and
    `hub.season.weekly_gate`, kept in agreement by a docstring reading *"the same rule as
    lineup_gate.projection_lineup_points"*. This repo has now twice found something real
    underneath an invariant asserted in prose -- `store.tables` claiming it and `connect` could
    not disagree, and the `prior_signal` join -- so the second copy is gone rather than
    annotated.

    Greedy, and deliberately not the enumeration in `hub.season.lineup`. That one searches
    every legal lineup to maximise a win probability; this is *start your highest*, the simple
    rule both gates measure against. They are different questions and both are wanted.

    `score` orders the choice and nothing else -- a projection, a negated consensus rank, a
    lower confidence bound. What it means is the caller's business.
    """
    order = np.argsort(-np.asarray(score, dtype=float))
    counts: dict[str, int] = {}
    starters: list[int] = []
    flex: list[int] = []
    for j in order:
        i = int(j)
        p = pos[i]
        if p in STARTERS and counts.get(p, 0) < STARTERS[p]:
            counts[p] = counts.get(p, 0) + 1
            starters.append(i)
        elif p in FLEX_FROM:
            flex.append(i)
    # `order` is already descending, so the first FLEX_SLOTS leftovers are the best ones.
    starters.extend(flex[:FLEX_SLOTS])
    return starters
