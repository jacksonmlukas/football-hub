"""Re-run `pool.leverage`'s "first run" at `LEVERAGE_STUDY_TRIALS` per arm (#368, S10).

`pool.leverage` was unresolved at two standard errors on 2026-09-11's first run -- 2 of 25
comparisons cleared the bar where chance gives about 1.1, 24 of 25 positive in sign, and
`pool.leverage`'s own docstring says why: "the resolution is the advanced arm's", and
resolving a +$3 term at concentration 16 needs roughly 4,000 trials per arm against the 1600
that first run spent. `WEEKLY_TRIALS` stayed at 400 because that is what the weekly path
needs, not because 4,000 was ever tried -- so "unresolved" was a statement about a constant
in the config, not about the pool.

This is the same measurement, same board, same seed, more trials: ADR-0007's harness for a
number that is cited as a reason (whether `LEVERAGE` below `pool_digest` states a resolved
term or an unresolved one).

**The board is the synthetic 32-team ladder `tests/unit/test_pool.py::_board` builds** --
sixteen fixtures a week, every team playing every week, matchups rotating by a circle so a
plan has something to solve -- rebuilt here rather than imported from the test module: no
script imports test code, and `pythonpath` deliberately does not put the repo root (and so
`scripts/`) on the path a test could import back either (`pyproject.toml`). The formula
below is copied verbatim from that fixture and should be kept identical to it by hand; a
board that drifted from the one the first run used would not be re-running the same study.

    uv run python scripts/leverage_study.py
    uv run python scripts/leverage_study.py --trials 4000

Nothing here writes a constant or a journal row. It prints the same lines
`hub.season.pool --leverage` would for this board and week, so the result can be read the
way an operator reads a live one.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.config import PoolConfig
from hub.season import pool
from hub.season.pool import LEVERAGE_STUDY_TRIALS

SCRIPT = "scripts/leverage_study.py"

# Verbatim from `tests/unit/test_pool.py`: a 32-team board, T00 weakest to T31 strongest, a
# strength ladder with a wobble so every fixture's price is distinct (see that module's
# `_board` for why the wobble is load-bearing).
TEAMS = tuple(f"T{i:02d}" for i in range(32))
STRENGTH = [-1.2 + 2.4 * i / 31 + 0.03 * math.sin(7.3 * i) for i in range(32)]


def season_grid(weeks: Sequence[int]) -> pl.DataFrame:
    """The same 32-team board `tests/unit/test_pool.py::_board(weeks)` builds, reproduced
    here so this script depends on no test code. See that function's docstring for why the
    rotation and the wobble are both load-bearing rather than decorative."""
    order = list(range(32))
    rows = []
    for w in range(1, max(weeks) + 1):
        pairs = [(order[i], order[31 - i]) for i in range(16)]
        if w in weeks:
            for i, (a, b) in enumerate(pairs):
                p = 1.0 / (1.0 + math.exp(-(STRENGTH[a] - STRENGTH[b])))
                rows += [(w, TEAMS[a], p, f"{w}-{i}"), (w, TEAMS[b], 1.0 - p, f"{w}-{i}")]
        order = [order[0], order[-1], *order[1:-1]]
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows], "game_id": [r[3] for r in rows]},
                        schema={"week": pl.Int64, "team": pl.Utf8, "win_prob": pl.Float64,
                                "game_id": pl.Utf8})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=SCRIPT,
        description="Re-run the pool leverage study at more trials per arm than the weekly "
                    "path spends, on the same synthetic board and seed as the first run.")
    ap.add_argument("--trials", type=int, default=LEVERAGE_STUDY_TRIALS)
    ap.add_argument("--week", type=int, default=2, help="the week being decided; week 1 is "
                    "already played, matching the first run's 'week 1 decided'")
    ap.add_argument("--entries", type=int, default=21)
    ap.add_argument("--pot", type=float, default=420.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    grid = season_grid(range(1, 15))
    cfg = PoolConfig(co_elimination_rule="split", co_survivor_rule="split")
    field = pool.Field(grid, list(range(1, 15)), cfg)
    print(f"leverage study: {a.trials} trials per arm, {a.entries} entries, pot ${a.pot:.2f}, "
          f"week {a.week}, seed {a.seed}, rules {pool.pool_digest(cfg)}, "
          f"board {pool.grid_digest(grid)}")

    rows = pool.leverage(field, week=a.week, entries=a.entries, pot=a.pot,
                         trials=a.trials, rng=np.random.default_rng(a.seed))
    for line in pool.leverage_report(rows):
        print(line)

    hits = sum(r.resolvable for r in rows)
    chance = pool.expected_by_chance(len(rows))
    positive = sum(r.term > 0 for r in rows)
    print(f"\n  {hits} of {len(rows)} comparisons resolvable at {pool.DECISIVE_SIGMA:.0f} "
          f"standard errors (chance gives about {chance:.1f}); {positive} of {len(rows)} "
          f"terms positive; resolvable_on_the_axis={pool.resolvable_on_the_axis(rows)}")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
