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

from collections.abc import Sequence

import numpy as np

from hub.config import RosterConfig, flex_capacity, flex_positions, required_starters

# QB1 / RB2 / WR3 / TE1 / FLEX1 -- confirmed against the live league, not the ESPN default.
# Derived from `hub.config.RosterConfig` rather than restated: this module is "how a league
# works", so it is the right place to *read* the shape and the wrong place to declare it.
ROSTER = RosterConfig()
STARTERS = required_starters(ROSTER)
FLEX_FROM = flex_positions(ROSTER)
FLEX_SLOTS = ROSTER.flex
# The most flex-eligible players a team can start at once. Was a bare `7` in optimize.py.
FLEX_CAPACITY = flex_capacity(ROSTER)
REG_SEASON_WEEKS = 14
# The weeks a fantasy regular season is played over, as a tuple, derived rather than restated.
# It was written out twice as `GATE_WEEKS = tuple(range(1, 15))` -- in `weekly_screen` and in
# `weekly_gate` -- which is a literal `15` in two files for a league length that already has an
# owner one line up.
FANTASY_WEEKS: tuple[int, ...] = tuple(range(1, REG_SEASON_WEEKS + 1))
PLAYOFF_TEAMS = 6
# Quarter-final, semi-final, final. The bracket needs its own weeks: scoring the playoffs on
# draws that already decided seeding couples a team's title odds to its week 1 result.
PLAYOFF_ROUNDS = 3

# Player-level prediction lives in `hub.models.predict`: what a player does in a week, as
# opposed to how a league works, which is this module. These constants are re-exported
# because calibrate, leverage and the tests import them from here, and a refactor that
# breaks its callers to be tidier is not an improvement.
#
# `moments` is deliberately NOT re-exported. It was, under the old name `weekly_moments`,
# and that alias outlived the function it named: `moments` returns three moments, so the
# name promised a shape the object no longer had. Callers wanting a player-level
# prediction import it from `hub.models.predict`, which is where it lives.
from hub.models.predict import (  # noqa: F401,E402
    MIN_SKEW,
    TALENT_CV,
    TALENT_CV_BY_POS,
    WEEKLY_K,
    WEEKLY_K_POOLED,
    WEEKLY_SKEW,
    WEEKLY_SKEW_POOLED,
    correlated_normal,
    skewed,
    talent_cv_for,
    weekly_skew_for,
)

_skewed = skewed
_correlated_normal = correlated_normal



def starting_lineup(pos: Sequence[str], score: Sequence[float] | np.ndarray) -> list[int]:
    """Indices of the lineup: each required **Slot** to the best available, then the flex.

    One rule, in the module that owns the shape it reads. `STARTERS`, `FLEX_FROM` and
    `FLEX_SLOTS` are declared three lines up, and five modules import them -- so the rule that
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


def lineup_points(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Best legal lineup, vectorised over (sims, weeks).

    scores: (sims, weeks, roster) -- one draw per player per week
    pos:    (roster,) -- position string per player

    Public because `hub.draft.backtest` scores realised seasons with it. A backtest using
    its own lineup rule would answer a slightly different question than the simulator it is
    auditing, and the difference would be invisible -- both would read as "best legal
    lineup". One rule, one owner.
    """
    total = np.zeros(scores.shape[:2])
    bench = []
    for p, n in STARTERS.items():
        idx = np.flatnonzero(pos == p)
        if idx.size == 0:
            continue
        block = -np.sort(-scores[:, :, idx], axis=2)      # descending
        total += block[:, :, :n].sum(axis=2)
        if p in FLEX_FROM and block.shape[2] > n:
            # Up to FLEX_SLOTS of them, not one: with two flex slots the best pair can come
            # from the same position, and taking one candidate per position cannot see that.
            bench.append(block[:, :, n:n + FLEX_SLOTS])
    if bench and FLEX_SLOTS:
        pool = np.concatenate(bench, axis=2)
        total += -np.sort(-pool, axis=2)[:, :, :FLEX_SLOTS].sum(axis=2)
    return total


def simulate_weeks(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                   pos: np.ndarray, n_sims: int, weeks: int = REG_SEASON_WEEKS,
                   rng: np.random.Generator | None = None,
                   talent_cv: float | np.ndarray | None = None,
                   nfl_team: np.ndarray | None = None,
                   skew: np.ndarray | None = None) -> np.ndarray:
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
    # Skew comes from the caller when it has it. `hub.models.predict.moments` already
    # returns a skew column alongside mu and sd; recomputing it here from `pos` agreed only
    # because both routes read the same table, and would go on agreeing right up until one
    # of them stopped being a function of position alone.
    if skew is None:
        skew = weekly_skew_for(pos)
    draws = _skewed(true_mu[:, None, :], sd_eff, skew[None, None, :], z)
    out = np.empty((n_sims, weeks, len(rosters)))
    for t, r in enumerate(rosters):
        out[:, :, t] = lineup_points(draws[:, :, r], pos[r]) if r.size else 0.0
    return out


def _round_robin(teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method schedule, repeated until the season is full."""
    rot = list(range(teams))
    base = []
    for _ in range(teams - 1):
        base.append([(rot[i], rot[teams - 1 - i]) for i in range(teams // 2)])
        rot = [rot[0], rot[-1], *rot[1:-1]]
    return [base[w % len(base)] for w in range(weeks)]


def seed_table(pts: np.ndarray, *, reg_weeks: int = REG_SEASON_WEEKS,
               playoff_teams: int = PLAYOFF_TEAMS) -> tuple[np.ndarray, np.ndarray]:
    """Regular-season wins and the playoff field, from weekly points.

    Seeds on wins, ties broken on total points -- ESPN's `playoffSeedingRule
    TOTAL_POINTS_SCORED`. **Regular-season points only**: `pts` now carries playoff weeks
    too, and summing those into the tiebreak would let a team's semi-final performance
    decide the seed it entered the playoffs with.

    Shared rather than duplicated. `leverage.py` re-implemented this loop line for line
    because `champion_probability` gave it no way to ask for the pieces.
    """
    n_sims, _, teams = pts.shape
    wins = np.zeros((n_sims, teams))
    for w, pairs in enumerate(_round_robin(teams, reg_weeks)):
        for a, b in pairs:
            a_won = pts[:, w, a] > pts[:, w, b]
            wins[:, a] += a_won
            wins[:, b] += ~a_won
    key = wins * 10_000.0 + pts[:, :reg_weeks, :].sum(axis=1)
    return wins, np.argsort(-key, axis=1)[:, :playoff_teams]


def champion(pts: np.ndarray, seeds: np.ndarray, sim: int, *,
             reg_weeks: int = REG_SEASON_WEEKS) -> int:
    """Winner of one bracket: seeds 1-2 bye, 3v6 and 4v5, then semis, then the final.

    Playoff weeks index `pts` directly. They used to be taken modulo the array width, so a
    14-week simulation replayed weeks 1-3 as the three playoff rounds -- meaning the games
    that decided the title were the same draws that had already set the seeding.
    """
    f = seeds[sim]
    w = reg_weeks
    alive = [f[0], f[1]]
    for a, b in ((f[2], f[5]), (f[3], f[4])):
        alive.append(a if pts[sim, w, a] > pts[sim, w, b] else b)
    semi = [alive[0] if pts[sim, w + 1, alive[0]] > pts[sim, w + 1, alive[3]] else alive[3],
            alive[1] if pts[sim, w + 1, alive[1]] > pts[sim, w + 1, alive[2]] else alive[2]]
    return int(semi[0] if pts[sim, w + 2, semi[0]] > pts[sim, w + 2, semi[1]] else semi[1])


def champion_probability(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                         pos: np.ndarray, n_sims: int = 400,
                         rng: np.random.Generator | None = None,
                         talent_cv: float | np.ndarray | None = None,
                         nfl_team: np.ndarray | None = None,
                         skew: np.ndarray | None = None) -> np.ndarray:
    """P(each team wins the league). Returns (teams,) summing to 1.

    14-week H2H regular season, top 6 seeds, two byes, then single elimination on one-week
    matchups -- read off the live league, not assumed. Three further weeks are simulated so
    the bracket has draws of its own.
    """
    rng = rng or np.random.default_rng(0)
    teams = len(rosters)
    pts = simulate_weeks(rosters, mu, sd, pos, n_sims,
                         REG_SEASON_WEEKS + PLAYOFF_ROUNDS, rng, talent_cv, nfl_team, skew)
    _, seeds = seed_table(pts)
    champs = np.array([champion(pts, seeds, s) for s in range(n_sims)])
    return np.bincount(champs, minlength=teams) / n_sims


