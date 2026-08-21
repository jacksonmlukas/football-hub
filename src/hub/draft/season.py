"""Simulate a season to a champion.

`cost_of_waiting` answers "who will I regret not taking". It does not answer "which pick
most improves my chance of winning", and those come apart in ways that matter: a fourth
good WR has real VOR and almost no effect on a championship, because you can only start
three. Roster construction, positional saturation and week-to-week variance only show up
if you play the season out.

The model is deliberately shallow where shallowness is cheap and honest where it is not:

  * weekly points ~ Normal(mu, sd) truncated at zero, per player, independent
  * mu is expected points per game (xFP), which is less noisy than realised points
  * sd is that player's own week-to-week dispersion
  * lineups are set optimally in hindsight each week

The last one is the biggest simplification: real managers start the wrong guy. It inflates
everyone's scores roughly equally, so it distorts P(win) far less than it distorts points.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# QB1 / RB2 / WR3 / TE1 / FLEX1 -- confirmed against the live league, not the ESPN default.
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX_FROM = ("RB", "WR", "TE")
REG_SEASON_WEEKS = 14
PLAYOFF_TEAMS = 6

# How wrong a preseason projection typically is about a player's season, as a fraction of
# his projected per-game points. This is the single most important number in the model:
# at 0 the projection is truth and drafting on it is clairvoyance; the larger it gets, the
# more a draft is a lottery and the flatter every candidate's championship equity becomes.
# 0.35 is in line with the historical spread between preseason ranks and end-of-season
# finishes for skill players. It has NOT been fitted here -- see the note in the module
# docstring of optimize.py before reading small differences as real.
TALENT_CV = 0.35


def weekly_moments(xp: pl.DataFrame, floor_sd: float = 2.0) -> pl.DataFrame:
    """Per-player weekly mean and dispersion.

    mu comes from expected points rather than realised: xFP already strips the
    week-to-week luck we are about to re-add, so using realised points would double-count
    variance and make every roster look more volatile than it is.
    """
    # Prefer the market's projection for the season being drafted. xFP describes the
    # season just gone, and using it as truth makes any strategy ranked on xFP look
    # prescient: it is scoring against the very data it optimised. Fall back to xFP only
    # where no projection exists.
    cols = [pl.col(c) for c in ("proj_blend", "proj_ppg") if c in xp.columns]
    mu = pl.coalesce(*cols, pl.col("xfp_per_game")).fill_null(0.0)
    return xp.with_columns(
        mu.alias("mu"),
        pl.max_horizontal(mu * 0.55, pl.lit(floor_sd)).alias("sd"),
    )


def _lineup_points(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Best legal lineup, vectorised over (sims, weeks).

    scores: (sims, weeks, roster) -- one draw per player per week
    pos:    (roster,) -- position string per player
    """
    total = np.zeros(scores.shape[:2])
    leftovers = []
    for p, n in STARTERS.items():
        idx = np.flatnonzero(pos == p)
        if idx.size == 0:
            continue
        block = -np.sort(-scores[:, :, idx], axis=2)      # descending
        total += block[:, :, :n].sum(axis=2)
        if p in FLEX_FROM and block.shape[2] > n:
            leftovers.append(block[:, :, n])              # best bench player at this slot
    if leftovers:
        total += np.max(np.stack(leftovers, axis=-1), axis=-1)
    return total


def simulate_weeks(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                   pos: np.ndarray, n_sims: int, weeks: int = REG_SEASON_WEEKS,
                   rng: np.random.Generator | None = None,
                   talent_cv: float = TALENT_CV) -> np.ndarray:
    """Weekly points for every team. Returns (sims, weeks, teams).

    Realised talent is drawn once per season, then weekly points are drawn around it.
    Without that first draw the projection IS the truth, and any strategy that ranks on
    the projection drafts with perfect foresight while an ADP-following opponent does
    not -- which is not an edge, it is a leak.
    """
    rng = rng or np.random.default_rng(0)
    true_mu = mu[None, :] * (1.0 + rng.normal(0.0, talent_cv, size=(n_sims, mu.size)))
    np.clip(true_mu, 0.0, None, out=true_mu)
    draws = rng.normal(true_mu[:, None, :], sd[None, None, :],
                       size=(n_sims, weeks, mu.size))
    np.clip(draws, 0.0, None, out=draws)
    out = np.empty((n_sims, weeks, len(rosters)))
    for t, r in enumerate(rosters):
        out[:, :, t] = _lineup_points(draws[:, :, r], pos[r]) if r.size else 0.0
    return out


def _round_robin(teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method schedule, repeated until the season is full."""
    rot = list(range(teams))
    base = []
    for _ in range(teams - 1):
        base.append([(rot[i], rot[teams - 1 - i]) for i in range(teams // 2)])
        rot = [rot[0], rot[-1], *rot[1:-1]]
    return [base[w % len(base)] for w in range(weeks)]


def champion_probability(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                         pos: np.ndarray, n_sims: int = 400,
                         rng: np.random.Generator | None = None,
                         talent_cv: float = TALENT_CV) -> np.ndarray:
    """P(each team wins the league). Returns (teams,) summing to 1.

    14-week H2H regular season, top 6 seeds, two byes, then single elimination on
    one-week matchups -- read off the live league, not assumed.
    """
    rng = rng or np.random.default_rng(0)
    teams = len(rosters)
    pts = simulate_weeks(rosters, mu, sd, pos, n_sims, REG_SEASON_WEEKS, rng, talent_cv)

    wins = np.zeros((n_sims, teams))
    for w, pairs in enumerate(_round_robin(teams, REG_SEASON_WEEKS)):
        for a, b in pairs:
            a_won = pts[:, w, a] > pts[:, w, b]
            wins[:, a] += a_won
            wins[:, b] += ~a_won

    # Seed on wins, break ties on total points -- the standard ESPN rule.
    total = pts.sum(axis=1)
    key = wins * 10_000.0 + total
    seeds = np.argsort(-key, axis=1)[:, :PLAYOFF_TEAMS]

    champs = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        field = list(seeds[s])
        wk = REG_SEASON_WEEKS
        # Seeds 1-2 bye; 3v6 and 4v5 in round one.
        alive = [field[0], field[1]]
        for a, b in ((field[2], field[5]), (field[3], field[4])):
            alive.append(a if _week_winner(pts, s, wk, a, b) else b)
        wk += 1
        semi = [alive[0] if _week_winner(pts, s, wk, alive[0], alive[3]) else alive[3],
                alive[1] if _week_winner(pts, s, wk, alive[1], alive[2]) else alive[2]]
        wk += 1
        champs[s] = semi[0] if _week_winner(pts, s, wk, semi[0], semi[1]) else semi[1]

    return np.bincount(champs, minlength=teams) / n_sims


def _week_winner(pts: np.ndarray, sim: int, week: int, a: int, b: int) -> bool:
    """True if `a` beats `b`. Playoff weeks reuse regular-season draws cyclically."""
    w = week % pts.shape[1]
    return bool(pts[sim, w, a] > pts[sim, w, b])
