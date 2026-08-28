"""Assembly for `hub.season.weekly_gate`: rosters, realised points, and both arms' scores.

Separate from the gate for the reason `backtest` and `lineup_gate` split the same way -- the
statistics have to be testable without a network, and a gate that only runs against live
nflverse is one nobody re-runs.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.draft.season import REG_SEASON_WEEKS
from hub.models.experiment import realised_ppg
from hub.names import player_key
from hub.season.weekly_gate import UNRANKED

WEEKS = REG_SEASON_WEEKS


def _matrix(keys: Sequence[str], lookup: dict[tuple[str, int], float],
            default: float) -> np.ndarray:
    """(roster, weeks) from a {(player key, week): value} map, with a stated default."""
    out = np.full((len(keys), WEEKS), default, dtype=float)
    for i, k in enumerate(keys):
        for w in range(1, WEEKS + 1):
            v = lookup.get((k, w))
            if v is not None:
                out[i, w - 1] = float(v)
    return out


def assemble(seasons: Sequence[int], *, drafts: int = 20, seed: int = 0):
    # pragma: no cover - network
    """`(rosters, realised, consensus, weekly, covered)`, keyed by `(season, roster index)`.

    `covered` is the set of `(season, week)` the consensus page actually has a scrape for.
    Historical coverage is not complete -- 2024 has nothing before week 5 -- and a week with
    no incumbent is not a hard week, it is no comparison at all.

    The weekly arm **falls back to the consensus score wherever it has no history** -- a player
    with no games played yet has no Usage to multiply, and projecting him at zero would bench
    him on the arm's own ignorance rather than on a forecast. Deferring to the incumbent there
    means the arm is never worse-informed *or* better-informed than what it is measured
    against, and it biases the measured difference toward zero rather than toward us.
    """
    import nflreadpy as nfl

    from hub.config import RosterConfig
    from hub.draft.backtest import market_strategy, play
    from hub.draft.board import board_as_of
    from hub.models.experiment import PLAYER_STATS_COLS, expanding_seasons
    from hub.models.weekly import VOLUME, fit_multiplier, project
    from hub.models.weekly_screen import build_panel, weekly_consensus

    cfg = RosterConfig()
    panel = build_panel(seasons, consensus=False)
    ecr = weekly_consensus(seasons)

    # One projection per (season, week) player, fitted on strictly earlier seasons only.
    projected = []
    for _season, past, now in expanding_seasons(panel):
        coefs = {c: fit_multiplier(past, c) for c in VOLUME}
        projected.append(project(now, coefs).select("key", "season", "week", "mu"))
    proj = pl.concat(projected) if projected else pl.DataFrame(
        schema={"key": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "mu": pl.Float64})

    rosters: dict[int, list] = {}
    realised: dict[tuple[int, int], np.ndarray] = {}
    consensus: dict[tuple[int, int], np.ndarray] = {}
    weekly: dict[tuple[int, int], np.ndarray] = {}

    for yr in sorted(set(proj["season"].unique().to_list()) & set(seasons)):
        print(f"  building the {yr} board as of {yr}-09-01 ...", flush=True)
        board = board_as_of(yr)[0]
        stats = nfl.load_player_stats(seasons=[yr]).select(list(PLAYER_STATS_COLS))
        pts = realised_ppg(stats)
        pts_of = {(r["player"], int(r["week"])): float(r["points"])
                  for r in pts.iter_rows(named=True)}
        e = ecr.filter(pl.col("season") == yr)
        ecr_of = {(r["key"], int(r["week"])): -float(r["ecr"])
                  for r in e.iter_rows(named=True)}
        m = proj.filter(pl.col("season") == yr)
        mu_of = {(r["key"], int(r["week"])): float(r["mu"])
                 for r in m.iter_rows(named=True)}

        made = []
        for k in range(drafts):
            names, pos = play(board, market_strategy(), my_slot=cfg.slot, teams=cfg.teams,
                              rounds=14, rng=np.random.default_rng(seed + 1000 * yr + k))
            keys = [player_key(n) for n in names]
            made.append(list(zip(names, pos, strict=True)))
            realised[(yr, k)] = _matrix(keys, pts_of, 0.0)
            cons = _matrix(keys, ecr_of, UNRANKED)
            consensus[(yr, k)] = cons
            wk = _matrix(keys, mu_of, float("nan"))
            weekly[(yr, k)] = np.where(np.isnan(wk), cons, wk)
        rosters[yr] = made
    covered = {(int(r["season"]), int(r["week"]))
               for r in ecr.select("season", "week").unique().iter_rows(named=True)}
    return rosters, realised, consensus, weekly, covered
