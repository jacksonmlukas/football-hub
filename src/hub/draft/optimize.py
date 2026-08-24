"""Pick the player who most raises your chance of winning the league.

This is a different question from every other ranking in this package, and it disagrees
with them in specific places. VOR prices talent against a positional baseline.
cost_of_waiting prices talent you will not get back. Neither knows that you can only
start three WRs, that a league is won in a 6-team playoff, or that a boom/bust roster and
a steady one with identical projected points do not win at the same rate.

The method is nested simulation:

  1. take the candidate,
  2. play out the rest of the draft -- opponents off a noisy blended board, you greedily
     filling your own lineup,
  3. simulate the season to a champion many times,
  4. read off how often you win.

The output is a probability, so it is directly comparable across candidates and directly
interpretable: "this pick is worth 1.4 points of championship equity".

READ THE LIFT, NOT THE LEVEL.
---------------------------
The absolute P(win) this reports is far too high -- around 30-50% for a 12-team league
where the baseline is 8.3% -- and the reason is structural, not a tuning problem.

The season is scored against `proj_blend`, and the greedy ranks candidates on
`proj_blend`. The simulation therefore hands this drafter the exact quantity that
defines its own truth, while opponents navigate by ADP, a proxy for it. Measured that
way a projection-follower cannot lose. A null control confirms the machinery is
otherwise sound: probabilities sum to one, and the eleven ADP-following seats show a
sensible snake gradient from 7.4% down to 2.9%. But the level is an artefact of assuming
ESPN's projection IS the season, and in reality ADP aggregates the whole market and is
plausibly the better estimator of the two.

Settling it needs an out-of-sample test, not a parameter: replay 2025 with 2025 preseason
projections and 2025 ADP, and score both against what actually happened. Until then the
levels are not credible.

What survives is the COMPARISON. Every candidate is evaluated against identical simulated
futures (common random numbers), so the paired lift between two candidates is a real
within-model quantity with a usable standard error, and the leak above applies equally to
both sides of it. Rank on `lift`, check it clears two standard errors, and ignore `p_win`.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hub.draft.availability import DEFAULT_ESPN_WEIGHT, blended_adp
from hub.draft.picks import snake_picks
from hub.draft.season import STARTERS, champion_probability, weekly_moments
from hub.draft.state import DraftState, remaining

# Bench depth beyond the 8 starting slots. Deep enough that saturation is punished,
# shallow enough that the simulated draft stays cheap.
DEFAULT_ROUNDS = 14


def _need_score(counts: dict[str, int], pos: str) -> int:
    """How badly an unfilled starting slot wants this position.

    Filling an empty required slot beats any surplus, because a surplus player scores
    zero. Flex-eligible depth ranks between the two.
    """
    if pos in STARTERS and counts.get(pos, 0) < STARTERS[pos]:
        return 2
    if pos in ("RB", "WR", "TE") and sum(counts.get(p, 0) for p in ("RB", "WR", "TE")) < 7:
        return 1
    return 0


def draft_pool(board: pl.DataFrame, state: DraftState,
               w: float = DEFAULT_ESPN_WEIGHT) -> pl.DataFrame:
    """Players still available, in the one canonical order.

    Roster indices returned by simulate_remaining_draft() index THIS frame, not the full
    board. Everything that consumes those rosters -- mu, sd, pos -- must be taken from
    here too, or the simulation silently scores the wrong players.
    """
    return blended_adp(remaining(board, state), w)


def simulate_remaining_draft(board: pl.DataFrame, state: DraftState, *, my_slot: int,
                             teams: int = 12, rounds: int = DEFAULT_ROUNDS,
                             forced: str | None = None,
                             w: float = DEFAULT_ESPN_WEIGHT, opp_noise: float = 1.0,
                             rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Play out the draft. Returns one array of draft_pool() row indices per team."""
    rng = rng or np.random.default_rng(0)
    pool = draft_pool(board, state, w)

    mu_pick = pool["mu_pick"].fill_null(999.0).to_numpy()
    # How loosely opponents follow their own board. The fitted sigma in availability.py
    # measures deviation from FantasyPros ECR, which is NOT what this is: a manager
    # drafting off ESPN's app barely deviates from ESPN's order, and ESPN's order is very
    # close to ESPN's projections. Using the ECR-fitted spread here credits opponents with
    # far more error than the premise allows, and every point of that error becomes edge
    # for the greedy. Scale it explicitly and treat the result as a sensitivity, not a
    # measurement -- the quantity that would settle it (deviation from historical ADP) is
    # exactly the one ESPN does not retain. See fit_espn_weight.
    noise = rng.normal(0.0, opp_noise * (2.0 + 0.18 * mu_pick))
    order = np.argsort(mu_pick + noise)          # opponents' perceived board

    # Rank on the same forecast the season is scored against. Using a different signal
    # here would measure the gap between two projections, not the value of the pick.
    key = "vor_proj" if "vor_proj" in pool.columns else "vor"
    vor = pool[key].fill_null(-99.0).to_numpy()
    pos = pool["pos"].fill_null("NA").to_numpy()
    names = pool["player"].to_list()

    my_picks = set(snake_picks(my_slot, teams, rounds))
    rosters: list[list[int]] = [[] for _ in range(teams)]
    # Every team tracks its own roster shape, not just mine. An opponent who takes the
    # next name on his board regardless of position ends the draft with an empty TE slot
    # scoring zero every week -- which is not a league, it is a walkover. Filling your
    # starters is the one thing every real drafter does.
    counts_by_team: list[dict[str, int]] = [{} for _ in range(teams)]
    counts = counts_by_team[my_slot - 1]
    gone = np.zeros(pool.height, dtype=bool)

    if forced is not None:
        f = names.index(forced)
        gone[f] = True
        rosters[my_slot - 1].append(f)
        counts[pos[f]] = counts.get(pos[f], 0) + 1

    start = state.n_taken + 1 + (1 if forced is not None else 0)
    for overall in range(start, teams * rounds + 1):
        rnd = (overall - 1) // teams + 1
        seat = (overall - 1) % teams
        team = seat if rnd % 2 else teams - 1 - seat
        if overall in my_picks and team == my_slot - 1:
            live = np.flatnonzero(~gone)
            if live.size == 0:
                break
            # Greedy on lineup value: an empty starting slot outranks raw VOR.
            pick = max(live, key=lambda i: (_need_score(counts, pos[i]), vor[i]))
        else:
            nxt = order[~gone[order]]
            if nxt.size == 0:
                break
            # Opponents follow their own noisy board, but skip a position they have
            # already filled -- they take the best player they still have room to start,
            # falling back to best available once every starting slot is covered.
            tc = counts_by_team[team]
            need = [i for i in nxt[:40] if _need_score(tc, pos[i]) > 0]
            pick = int(need[0]) if need else int(nxt[0])
        counts_by_team[team][pos[pick]] = counts_by_team[team].get(pos[pick], 0) + 1
        gone[pick] = True
        rosters[team].append(int(pick))

    return [np.array(r, dtype=int) for r in rosters]


def win_probability(board: pl.DataFrame, state: DraftState, candidates: list[str], *,
                    my_slot: int, teams: int = 12, rounds: int = DEFAULT_ROUNDS,
                    n_draft_sims: int = 12, n_season_sims: int = 150,
                    w: float = DEFAULT_ESPN_WEIGHT, seed: int = 0) -> pl.DataFrame:
    """P(you win the league) for each candidate, averaged over simulated drafts."""
    pool = draft_pool(board, state, w)
    moments = weekly_moments(pool)
    mu = moments["mu"].fill_null(0.0).to_numpy()
    sd = moments["sd"].fill_null(2.0).to_numpy()
    pos = pool["pos"].fill_null("NA").to_numpy()
    # NFL team, so the simulator can correlate a quarterback with his own pass catchers.
    # Without it a stacked roster is drawn independent and comes out less volatile than it
    # is -- see docs/correlation.md, where independence gives a nominal 80% interval that
    # covers 72.9%.
    nfl_team = (pool["team"].to_numpy() if "team" in pool.columns else None)

    # Common random numbers. Every candidate is evaluated against the SAME simulated
    # futures -- same draft rollouts, same talent draws, same weekly scores -- so the
    # only thing that differs between two columns is the player at this pick. The levels
    # still carry the full noise of the simulation, but the *difference* between two
    # candidates is paired, and the paired difference is what the decision needs.
    mat = np.empty((len(candidates), n_draft_sims))
    for i, c in enumerate(candidates):
        for k in range(n_draft_sims):
            rosters = simulate_remaining_draft(board, state, my_slot=my_slot, teams=teams,
                                               rounds=rounds, forced=c, w=w,
                                               rng=np.random.default_rng(seed + k))
            p = champion_probability(rosters, mu, sd, pos, n_sims=n_season_sims,
                                     rng=np.random.default_rng(seed + 1000 + k),
                                     nfl_team=nfl_team)
            mat[i, k] = p[my_slot - 1]

    field = mat.mean(axis=0)                       # the field, per simulated future
    diff = mat - field[None, :]                    # paired lift, same futures
    se = diff.std(axis=1, ddof=1) / np.sqrt(n_draft_sims) if n_draft_sims > 1 else \
        np.zeros(len(candidates))

    return pl.DataFrame({
        "player": candidates,
        "p_win": mat.mean(axis=1),
        "lift": diff.mean(axis=1),
        "lift_se": se,
    }).sort("lift", descending=True)
