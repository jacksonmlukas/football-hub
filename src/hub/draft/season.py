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

# Player-level prediction lives in `hub.models.predict`: what a player does in a week, as
# opposed to how a league works, which is this module. These names are re-exported because
# calibrate, leverage, optimize and the tests import them from here, and a refactor that
# breaks its callers to be tidier is not an improvement.
from hub.models.predict import (  # noqa: F401
    MIN_SKEW, TALENT_CV, TALENT_CV_BY_POS, WEEKLY_K, WEEKLY_K_POOLED, WEEKLY_SKEW,
    WEEKLY_SKEW_POOLED, correlated_normal, skewed, talent_cv_for, weekly_skew_for,
)
from hub.models.predict import moments as weekly_moments  # noqa: F401

_skewed = skewed
_correlated_normal = correlated_normal


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
                   talent_cv: float | np.ndarray | None = None,
                   nfl_team: np.ndarray | None = None) -> np.ndarray:
    """Weekly points for every team. Returns (sims, weeks, teams).

    Realised talent is drawn once per season, then weekly points are drawn around it.
    Without that first draw the projection IS the truth, and any strategy that ranks on
    the projection drafts with perfect foresight while an ADP-following opponent does
    not -- which is not an edge, it is a leak.
    """
    rng = rng or np.random.default_rng(0)
    # None means per position; a scalar is still accepted, which is what the sweeps in
    # hub.draft.leverage need in order to vary one thing at a time.
    if talent_cv is None:
        talent_cv = talent_cv_for(pos)
    true_mu = mu[None, :] * (1.0 + rng.normal(0.0, talent_cv, size=(n_sims, mu.size)))
    np.clip(true_mu, 0.0, None, out=true_mu)
    # Weekly spread follows *realised* talent, not the projection. Keying it to the
    # projection made busts impossible: a player projected at 15 whose talent went to zero
    # was drawn from N(0, 8.25) and clipped, which is a half-normal averaging 3.3 points a
    # game -- 22% of his projection, manufactured entirely by the clip. Scaling by the
    # realised ratio leaves an average player untouched and lets a collapsed one score
    # nothing, which is what a bust is.
    # Spread follows *realised* talent. Under sd = k*sqrt(mu) that means scaling by
    # sqrt(realised/projected), not by the ratio -- a player who realises a quarter of his
    # projection keeps half his weekly spread, not a quarter of it.
    ratio = np.divide(true_mu, mu[None, :], out=np.zeros_like(true_mu),
                      where=mu[None, :] > 0)
    sd_eff = (sd[None, :] * np.sqrt(ratio))[:, None, :]
    z = _correlated_normal(rng, (n_sims, weeks, mu.size), pos, nfl_team)
    draws = _skewed(true_mu[:, None, :], sd_eff,
                    weekly_skew_for(pos)[None, None, :], z)
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
                         talent_cv: float | np.ndarray | None = None,
                         nfl_team: np.ndarray | None = None) -> np.ndarray:
    """P(each team wins the league). Returns (teams,) summing to 1.

    14-week H2H regular season, top 6 seeds, two byes, then single elimination on
    one-week matchups -- read off the live league, not assumed.
    """
    rng = rng or np.random.default_rng(0)
    teams = len(rosters)
    pts = simulate_weeks(rosters, mu, sd, pos, n_sims, REG_SEASON_WEEKS, rng, talent_cv,
                         nfl_team)

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
