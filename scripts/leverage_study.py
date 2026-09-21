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

**The free pick and the five candidates it is compared against are pinned, not re-ranked**
(#377). The 2026-09-21 re-run at 4,000 trials could not be compared with the 2026-09-11 run
at 1600: the free pick had moved T31 -> T30 on the identical board, seed and week, because
intervening commits changed what `candidate_ranking` returns for it, so the two runs shared
no arm. `tests/golden/fixtures/pool_leverage_candidates.json` is the committed answer to
"what does today's ranking say", dated, and this script reads it rather than asking
`hub.season.pool` fresh on every run -- so a re-run studies the same six teams the last one
did, and a ranking change is a diff against that file instead of a silent confound.

    uv run python scripts/leverage_study.py
    uv run python scripts/leverage_study.py --trials 4000
    uv run python scripts/leverage_study.py --refit

`--refit` recomputes the free pick and candidates from today's ranking, prints the diff
against the committed fixture, writes the regenerated file, and exits -- it spends no
trials and runs no study. `tests/unit/test_pool.py` carries the test that would have caught
the T31 -> T30 move without anyone running `--refit` at all: it asserts the committed
fixture still matches what today's ranking produces.

Nothing else here writes a constant or a journal row. It prints the same lines
`hub.season.pool --leverage` would for this board and week, so the result can be read the
way an operator reads a live one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from hub.config import PoolConfig
from hub.season import pool
from hub.season.pool import LEVERAGE_STUDY_TRIALS

SCRIPT = "scripts/leverage_study.py"

# tests/golden/fixtures/pool_leverage_candidates.json: the free pick and five candidates
# `candidate_ranking` returned for this board and week the day the fixture was last written.
# See that file's own "note" field and docs/pool-leverage.md for what it is and why it is
# pinned rather than recomputed on every run.
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "golden" / "fixtures" \
    / "pool_leverage_candidates.json"

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


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _candidates_payload(grid: pl.DataFrame, week: int, cfg: PoolConfig) -> dict:
    """Today's free pick and five candidates, in the fixture's own `{free_pick, candidates}`
    shape -- what `--refit` compares against the committed file and, if asked, writes back."""
    _, teams = pool.candidate_ranking(grid, week, cfg, top=6)
    return {
        "free_pick": {"team": teams[0][0], "win_prob": teams[0][1]},
        "candidates": [{"team": t, "win_prob": p} for t, p, _ in teams[1:]],
    }


def _ranking_from_payload(payload: dict
                          ) -> tuple[str, list[tuple[str, float, tuple[str, ...]]]]:
    """The `(free, candidates)` shape `pool.leverage(..., ranking=...)` takes, read back off
    a fixture's `{free_pick, candidates}` -- the inverse of `_candidates_payload`. `takes` is
    always the team alone: this board's `week` is never one of `PoolConfig.double_pick_weeks`
    (the default config passed here declares none), so a candidate is never a pair."""
    free_row = payload["free_pick"]
    rows = [free_row, *payload["candidates"]]
    return free_row["team"], [(r["team"], r["win_prob"], (r["team"],)) for r in rows]


def _diff(committed: dict, today: dict) -> list[str]:
    """What changed between the committed fixture and today's ranking, read for a human."""
    old_rows = [committed["free_pick"], *committed["candidates"]]
    new_rows = [today["free_pick"], *today["candidates"]]
    old_teams, new_teams = [r["team"] for r in old_rows], [r["team"] for r in new_rows]
    lines = []
    if old_teams == new_teams:
        lines.append(f"  same six teams, same order: {', '.join(old_teams)}")
    else:
        lines.append(f"  committed: {', '.join(old_teams)}")
        lines.append(f"  today:     {', '.join(new_teams)}")
        moved = [t for t in old_teams if t not in new_teams]
        arrived = [t for t in new_teams if t not in old_teams]
        if moved:
            lines.append(f"  left the top six: {', '.join(moved)}")
        if arrived:
            lines.append(f"  entered the top six: {', '.join(arrived)}")
    for old, new in zip(old_rows, new_rows, strict=False):
        if old["team"] == new["team"] and abs(old["win_prob"] - new["win_prob"]) > 1e-9:
            lines.append(f"  {old['team']}: win_prob {old['win_prob']:.6f} -> "
                         f"{new['win_prob']:.6f}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=SCRIPT,
        description="Re-run the pool leverage study at more trials per arm than the weekly "
                    "path spends, on the same synthetic board, seed and candidate fixture "
                    "as the last run.")
    ap.add_argument("--trials", type=int, default=LEVERAGE_STUDY_TRIALS)
    ap.add_argument("--week", type=int, default=2, help="the week being decided; week 1 is "
                    "already played, matching the first run's 'week 1 decided'")
    ap.add_argument("--entries", type=int, default=21)
    ap.add_argument("--pot", type=float, default=420.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true",
                    help="regenerate tests/golden/fixtures/pool_leverage_candidates.json "
                         "from today's ranking, print the diff against the committed file, "
                         "write it, and exit -- spends no trials and runs no study")
    a = ap.parse_args(argv)

    grid = season_grid(range(1, 15))
    cfg = PoolConfig(co_elimination_rule="split", co_survivor_rule="split")

    if a.refit:
        committed = _load_fixture()
        today = _candidates_payload(grid, a.week, cfg)
        print(f"refitting against week {a.week}, board {pool.grid_digest(grid)}, "
              f"committed fixture dated {committed.get('date', 'unknown')}:")
        for line in _diff(committed, today):
            print(line)
        fixture = {**committed, **today,
                  "date": dt.datetime.now(tz=dt.UTC).date().isoformat()}
        fixture["board"] = {**committed["board"], "week": a.week,
                            "grid_digest": pool.grid_digest(grid),
                            "pool_digest": pool.pool_digest(cfg)}
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
        print(f"  wrote {FIXTURE}")
        return 0

    field = pool.Field(grid, list(range(1, 15)), cfg)
    fixture = _load_fixture()
    board = fixture["board"]
    if pool.grid_digest(grid) != board["grid_digest"]:
        raise ValueError(
            f"the board this script builds digests to {pool.grid_digest(grid)}, not the "
            f"{board['grid_digest']} the fixture was written against: the formula in this "
            "file has drifted from `tests/unit/test_pool.py::_board`, which `--refit` "
            "cannot fix -- that is a copy-paste error to correct by hand, per this "
            "script's own module docstring.")
    if a.week != board["week"]:
        raise ValueError(
            f"the committed fixture is pinned to week {board['week']}; asked for week "
            f"{a.week} instead. Run `--refit --week {a.week}` to pin a new candidate set "
            "for that week before studying it.")
    ranking = _ranking_from_payload(fixture)
    print(f"leverage study: {a.trials} trials per arm, {a.entries} entries, pot ${a.pot:.2f}, "
          f"week {a.week}, seed {a.seed}, rules {pool.pool_digest(cfg)}, "
          f"board {pool.grid_digest(grid)}, candidates fixture dated {fixture['date']} "
          f"(free {ranking[0]}, vs {', '.join(t for t, _, _ in ranking[1] if t != ranking[0])})")

    rows = pool.leverage(field, week=a.week, entries=a.entries, pot=a.pot,
                         trials=a.trials, rng=np.random.default_rng(a.seed), ranking=ranking)
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
