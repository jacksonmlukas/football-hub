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

from typing import Callable

import numpy as np
import polars as pl

from hub.draft.availability import DEFAULT_ESPN_WEIGHT, blended_adp
from hub.draft.picks import snake_picks
from hub.draft.season import FLEX_CAPACITY, FLEX_FROM, STARTERS, champion_probability
from hub.draft.state import DraftState, _norm, remaining, roster_for
from hub.models.predict import WEEKLY_SKEW_POOLED, moments

# Bench depth beyond the 8 starting slots. Deep enough that saturation is punished,
# shallow enough that the simulated draft stays cheap.
DEFAULT_ROUNDS = 14

# What a pluggable draft strategy is handed, and what it returns.
#
#   pool   -- the board, with `mu_pick` attached; indices are board rows
#   live   -- row indices still on the board
#   counts -- how many of each position I already hold
#   taken  -- every player already off the board, in pick order, so a strategy that needs a
#             `DraftState` can rebuild one exactly
#
# It returns a row index into `pool`, i.e. into the board. Anything else raises.
MyPick = Callable[[pl.DataFrame, np.ndarray, dict[str, int], list[str]], int]


def _need_score(counts: dict[str, int], pos: str) -> int:
    """How badly an unfilled starting slot wants this position.

    Filling an empty required slot beats any surplus, because a surplus player scores
    zero. Flex-eligible depth ranks between the two.
    """
    if pos in STARTERS and counts.get(pos, 0) < STARTERS[pos]:
        return 2
    if pos in FLEX_FROM and sum(counts.get(p, 0) for p in FLEX_FROM) < FLEX_CAPACITY:
        return 1
    return 0


def simulate_remaining_draft(board: pl.DataFrame, state: DraftState, *, my_slot: int,
                             teams: int = 12, rounds: int = DEFAULT_ROUNDS,
                             forced: str | None = None,
                             w: float = DEFAULT_ESPN_WEIGHT, opp_noise: float = 1.0,
                             rng: np.random.Generator | None = None,
                             my_pick: MyPick | None = None) -> list[np.ndarray]:
    """Play out the draft. Returns one array of `board` row indices per team.

    Two separable things live here: the **room** -- eleven opponents following a noisy board
    and filling their own starting slots -- and **my** strategy, which is greedy on lineup
    value. `my_pick` replaces the second and leaves the first alone.

    That seam exists because `hub.draft.backtest` needs to run other strategies against this
    exact room. Copying the room into the backtest would have been the alternative, and a
    copied room drifts: the thing being measured would slowly stop being the thing that
    ships. Default is `None`, which keeps the greedy rule and every existing caller
    unchanged.

    **Indices are into `board`.** They used to index a derived frame of only-available
    players, which carried a standing warning that everything consuming a roster had to be
    read from that same frame. Indexing the board removes the class of error rather than
    warning about it, and it is what lets every seat start from the roster it already holds
    -- see below, which is the bug this shape exists to fix.
    """
    rng = rng or np.random.default_rng(0)
    pool = blended_adp(board, w)

    mu_pick = pool["mu_pick"].fill_null(999.0).to_numpy()
    # How loosely opponents follow their own board. The fitted sigma in availability.py
    # measures deviation from FantasyPros ECR, which is NOT what this is: a manager
    # drafting off ESPN's app barely deviates from ESPN's order, and ESPN's order is very
    # close to ESPN's projections. Using the ECR-fitted spread here credits opponents with
    # far more error than the premise allows, and every point of that error becomes edge
    # for the greedy. Scale it explicitly and treat the result as a sensitivity, not a
    # measurement -- the quantity that would settle it (deviation from historical ADP) is
    # exactly the one ESPN does not retain. See fit_espn_weight.
    from hub.draft.availability import pick_noise
    noise = rng.normal(0.0, opp_noise * pick_noise(mu_pick))
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

    # Every seat starts from the roster it already holds.
    #
    # This is the fix for a defect that shipped: rosters used to be built only from picks
    # made *during* the simulation, because the pool was `remaining(board, state)` and your
    # existing players were not in it. Championship equity was therefore blind to what you
    # already owned -- holding a quarterback, it ranked a second quarterback above a
    # startable back, and on the live 2026 board it named a third and fourth running back at
    # three of your first six turns while QB, WR and TE sat empty.
    #
    # All twelve seats, not just mine: seeding only mine would leave eleven opponents
    # fielding future picks alone, making them weaker than they are and inflating my p_win
    # -- a number the output already has to apologise for.
    #
    # A recorded pick that is not on the board (K, DST, or a misspelling) is skipped. Those
    # are expected -- `DRAFTED_POSITIONS` excludes kickers and defences on purpose -- and
    # `suggest_unmatched` already flags a misspelling where a human can still fix it.
    by_norm = {_norm(n): i for i, n in enumerate(names)}
    for seat in range(1, teams + 1):
        for held_name in roster_for(state, seat, teams, rounds):
            i = by_norm.get(_norm(held_name))
            if i is None or gone[i]:
                continue
            gone[i] = True
            rosters[seat - 1].append(i)
            counts_by_team[seat - 1][pos[i]] = (
                counts_by_team[seat - 1].get(pos[i], 0) + 1)
    # Pick *order*, not pool order. `np.flatnonzero(gone)` would give the latter, and
    # `state.roster_for` attributes picks to seats by walking the snake -- so a strategy
    # rebuilding a DraftState from an unordered list would be handed someone else's roster.
    order_taken: list[str] = list(state.taken)

    if forced is not None:
        f = names.index(forced)
        gone[f] = True
        rosters[my_slot - 1].append(f)
        counts[pos[f]] = counts.get(pos[f], 0) + 1
        order_taken.append(names[f])

    start = state.n_taken + 1 + (1 if forced is not None else 0)
    for overall in range(start, teams * rounds + 1):
        rnd = (overall - 1) // teams + 1
        seat = (overall - 1) % teams
        team = seat if rnd % 2 else teams - 1 - seat
        if overall in my_picks and team == my_slot - 1:
            live = np.flatnonzero(~gone)
            if live.size == 0:
                break
            if my_pick is None:
                # Greedy on lineup value: an empty starting slot outranks raw VOR.
                pick = max(live, key=lambda i: (_need_score(counts, pos[i]), vor[i]))
            else:
                pick = int(my_pick(pool, live, dict(counts), list(order_taken)))
                if gone[pick]:
                    raise ValueError(
                        f"my_pick returned {names[pick]!r}, who is already drafted")
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
        order_taken.append(names[pick])

    return [np.array(r, dtype=int) for r in rosters]


def win_probability(board: pl.DataFrame, state: DraftState, candidates: list[str], *,
                    my_slot: int, teams: int = 12, rounds: int = DEFAULT_ROUNDS,
                    n_draft_sims: int = 24, n_season_sims: int = 300,
                    w: float = DEFAULT_ESPN_WEIGHT, seed: int = 0) -> pl.DataFrame:
    """P(you win the league) for each candidate, averaged over simulated drafts.

    Scored over the *whole board*, and rosters include the players each seat already holds.
    They used to include only picks made during the simulation, which made the objective
    blind to your own roster: holding a quarterback, it ranked a second one above a
    startable back.
    """
    pool = blended_adp(board, w)
    pred = moments(pool)
    mu = pred["mu"].fill_null(0.0).to_numpy()
    sd = pred["sd"].fill_null(2.0).to_numpy()
    # Read the skew off the same frame that produced mu and sd. The simulator used to
    # recompute it from `pos`, which agreed only because both routes read one table --
    # a per-player skew (from components, say) would have been computed here and silently
    # dropped on the way in.
    skew = pred["skew"].fill_null(WEEKLY_SKEW_POOLED).to_numpy()
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
                                     nfl_team=nfl_team, skew=skew)
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


def rank_tiers(wp: pl.DataFrame) -> pl.DataFrame:
    """Mark which candidates the simulation cannot separate from the best one.

    The objective is the player with the highest chance of winning the league, and sometimes
    that is two players. At pick 3 the top two differ by 0.17 points of championship equity
    with standard errors near 0.5 -- printing a strict order there asserts a distinction the
    simulation cannot make, and re-running names a different one.

    Gaps are in standard errors of the *difference*, since both candidates carry sampling
    error, and anything inside two of them is reported as tied for the lead.
    """
    top = wp.sort("lift", descending=True)
    lead_lift = float(top["lift"][0])
    lead_se = float(top["lift_se"][0] or 0.0)
    pooled = (pl.col("lift_se").fill_null(0.0) ** 2 + lead_se ** 2).sqrt()
    gap = (pl.lit(lead_lift) - pl.col("lift"))
    return top.with_columns(
        pl.when(pooled > 0).then(gap / pooled).otherwise(
            pl.when(gap.abs() > 0).then(pl.lit(float("inf"))).otherwise(pl.lit(0.0))
        ).alias("gap_se")
    ).with_columns((pl.col("gap_se") < 2.0).alias("co_leader"))


def tag_for(co_leader: bool, lift: float, lift_se: float) -> str:
    """How a candidate should be labelled on the board.

    Three states, and the sign matters. `avoid` is for candidates significantly *worse* than
    the field -- an earlier version fired it on any candidate significantly different from
    zero, which labelled a player at +0.95 lift as one to avoid.
    """
    if co_leader:
        return "TAKE"
    if lift < 0 and abs(lift) > 2 * (lift_se or 0.0):
        return "avoid"
    return ""


def market_pick(pool: pl.DataFrame, counts: dict[str, int],
                by: str = "adp") -> str | None:
    """Best available in a market, that fills an unfilled starting slot.

    What the board leads with. P0 measured this against championship equity on realised
    outcomes across three seasons: market +3.11 against the room, equity +3.15, difference
    +0.04 with a 95% interval of [-3.64, +3.58] at n=36. No detectable difference.

    The simpler arm leads because it cannot be shown worse and is instant, and the burden
    sits on the complicated thing -- not because the optimizer is bad. That claim was made
    from a run at a quarter of the optimizer's shipped budget and has been withdrawn.

    Lexicographic, matching `simulate_remaining_draft`: an unfilled starting slot outranks
    any amount of market position, and the market breaks ties within a need tier.

    `by` names which market. `adp` is the **draft market** -- where your room actually takes
    a player -- and is what draft night uses. `ecr` is **consensus**, and is what a replay of
    a past season has to use, because ESPN publishes ADP for the current season only. Both
    are lower-is-better rankings, so one rule covers them; see `CONTEXT.md`, which keeps the
    two markets apart on purpose.
    """
    if pool.height == 0 or by not in pool.columns:
        return None
    # An all-null column means that market has nothing to say, and ESPN is the first thing
    # to fall over on draft night. Ranking by a column that is entirely null returns
    # whichever row came first -- a confident-looking recommendation with nothing behind it,
    # which is worse than admitting the input is missing.
    if pool[by].null_count() == pool.height:
        return None
    names = pool["player"].to_list()
    pos = pool["pos"].fill_null("NA").to_list()
    # a null position is an undrafted player, not the first pick of the round
    rank = pool[by].fill_null(999.0).to_list()
    best = min(range(len(names)),
               key=lambda i: (-_need_score(counts, pos[i]), rank[i]))
    return names[best]
