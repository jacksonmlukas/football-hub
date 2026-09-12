"""The room, the seeding tree, and THE PICK: what a draft-night decision runs on.

Three things live here, and they are what is left after ADR-0009's REMOVE was finished by
#198:

  * **The room.** `simulate_remaining_draft` plays the rest of a draft -- eleven opponents
    off a noisy blended board, each filling its own starting slots before taking depth, you
    in your seat by whatever rule the caller hands in. Every Gate in the repo draws its
    rosters through it: `hub.draft.cohort` for the two season-side Gates, `hub.draft.backtest`
    for the draft Gate. It indexes the board rather than naming players (ADR-0008).
  * **The seeding tree.** `root_seed` and `stream` are how one experiment's levels get
    unrelated random streams instead of offsets on one integer (#195); the block below says
    what the offsets cost.
  * **THE PICK.** `the_pick`, `market_pick` and `corrected_adp` are the rule the draft-night
    output leads with: best available in the draft market, corrected by what this repo has
    measured the draft market to miss, that fills an unfilled starting slot (ADR-0011). Both
    draft-night tools reach it here so they cannot disagree.

**What is not here any more, and why.** This module used to open with championship equity --
"pick the player who most raises your chance of winning the league" -- and hold
`win_probability`, `_lift_frame`, `rank_tiers` and `tag_for`. Equity was gated against the
draft market on realised outcomes and lost by 17 to 20 points a team-game in every season
(ADR-0009), so it does not pick; ADR-0007 keeps the measurement re-runnable, so the code was
not deleted. It is `hub.exhibits.championship_equity` now, where a reader can tell it from
the code the product reads without a call-site census (#198). It imports the room from here;
nothing here imports it.
"""
from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from hub.config import DraftConfig
from hub.draft import durability
from hub.draft.availability import DEFAULT_ESPN_WEIGHT, blended_adp
from hub.draft.picks import MY_SLOT, TEAMS, snake_picks
from hub.draft.state import DraftState, remaining, roster_for
from hub.league import FLEX_CAPACITY, FLEX_FROM, STARTERS
from hub.names import player_key

if TYPE_CHECKING:                  # `board` reaches this module function-locally, both ways
    from hub.draft.board import BuildReport

# Bench depth beyond the 8 starting slots. Deep enough that saturation is punished,
# shallow enough that the simulated draft stays cheap.
DEFAULT_ROUNDS = 14

# --- the seeding tree (issue #195) ------------------------------------------------------
#
# Three different things in this package need randomness, and they are at three different
# *levels* of the same experiment:
#
#   ROOM        the eleven opponents you are actually drafting against, and which is the
#               room your roster is scored in;
#   ROLLOUT     the draft futures a candidate is evaluated over, inside `win_probability`;
#   SEASON_SIM  the seasons each of those futures is played out for.
#
# Until #195 these were arithmetic offsets on one integer -- the room was `seed`, rollout k
# opened `default_rng(seed + k)`, and season sim k opened `default_rng(seed + 1000 + k)`.
# Offsets on a shared line are not separate streams, they are one line read from different
# places, and a line read from different places eventually reads itself:
#
#   * rollout 0 opened `default_rng(seed)`, bit-for-bit the room being scored, so one of
#     arm B's twelve evaluation futures WAS the room it was about to be graded in, at every
#     pick. Arm A got nothing of the kind, so the leak is one-directional;
#   * consecutive drafts, whose rooms sat one apart, shared 11 of their 12 futures;
#   * `seed + 1000 + k` landed on the next season's room lattice, because the room itself
#     was `seed + 1000 * season + k`.
#
# The fix is not a bigger offset. A bigger offset is the same defect with a larger constant,
# and it stays correct only for as long as nobody changes `n_draft_sims` or the number of
# drafts. `SeedSequence` spawn coordinates are what actually separate streams: two distinct
# coordinate paths hash to unrelated states, so a level cannot arrive at another level's
# stream by counting, whatever the counts are.
ROOM, ROLLOUT, SEASON_SIM = 0, 1, 2

# `SeedSequence` refuses a negative entropy word, and `--seed` is a plain `int` from argparse.
# Fold rather than raise: a negative seed used to run, and a seeding change is not the place
# to start rejecting one.
_ENTROPY_MASK = (1 << 64) - 1


def root_seed(*parts: int | np.random.SeedSequence) -> np.random.SeedSequence:
    """The root every stream of one experiment descends from.

    Pass the coordinates that identify the experiment -- for the backtest, `(seed, season,
    draft)`. A `SeedSequence` passed alone is returned as-is, which is what lets a caller
    that already holds a root hand it down instead of flattening it back to an integer and
    re-deriving something that only looks like it.
    """
    if len(parts) == 1 and isinstance(parts[0], np.random.SeedSequence):
        return parts[0]
    return np.random.SeedSequence([int(p) & _ENTROPY_MASK for p in parts])  # type: ignore[arg-type]


def stream(root: np.random.SeedSequence, *path: int) -> np.random.Generator:
    """One level's generator, at coordinate `path` under `root`.

    Deterministic and repeatable: calling this twice with the same root and path returns two
    generators in the same state, which is exactly what "the same room, twice" needs. What it
    does not do is let two different paths meet, which is what the arithmetic did.
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=root.entropy, spawn_key=(*root.spawn_key, *path)))

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


def _greedy_currency(pool: pl.DataFrame, report: BuildReport) -> np.ndarray:
    """What the greedy fallback ranks on, and a comprehensible failure when it is not there.

    The report decides the currency -- that is #199, and asking the frame instead is what let
    the live room rank on `proj_blend` while every backtested room ranked on prior-season xFP
    with nothing recording the difference. This does not ask the frame what stage ran.

    What it does handle is an incoherent pair: a frame whose report says the draft-market stage
    ran, carrying `adp` but not `vor_proj`. `build` cannot emit one -- the stage runs under
    `_stage(..., absorbs=())`, so a failure inside it aborts the build rather than returning a
    half-written Board -- but these are public functions taking a bare frame, and review found
    a hand-built one raising `ColumnNotFoundError` from four frames deep.

    **It refuses rather than falling back to `vor`.** Falling back is what the sniff did, and
    quietly ranking on a different currency than the caller believes is the defect, not the
    mitigation. So the answer is the same refusal with an explanation attached.

    **This is issue #147's decision, and it is the first of the two that ticket offered.**
    #147 asked for the choice on the record rather than for a particular one: thread the
    report through the simulator, or leave the fallback and annotate what ranking on `vor`
    instead of `vor_proj` costs every arm, cohort and gate built on it. #199 threaded it, so
    the annotation branch is moot -- what a reader wants instead is the gap that survives
    threading, and that is named where it bites, in `backtest.LIMITATIONS`: no historical
    ESPN projection exists to give a past room, so the live room and every backtested room
    rank in different currencies whatever this function does. Read off the report, that is a
    stated limitation; read off a column, it was an accident nothing recorded.

    One seam is still open and is a season-side change rather than this one: `cohort.cohort`
    takes no report, so the two gates that call it -- `season.lineup_gate` and
    `season.weekly_gate_data`, both holding one from `board_as_of` -- have it re-derived by
    `report_for`. Nothing ranks differently for it today (the cohort supplies its own
    `my_pick`, so this function never runs there, and a historical board's derived report
    matches the one it was built with), which is why it is recorded rather than fixed here.
    """
    column = "vor_proj" if report.adp else "vor"
    try:
        return pool[column].fill_null(-99.0).to_numpy()
    except pl.exceptions.ColumnNotFoundError:
        raise ValueError(
            f"the greedy fallback ranks on {column!r} because this frame's report says the "
            f"draft-market stage {'ran' if report.adp else 'did not run'}, and the column is "
            f"not here. A Board from `build` always carries it; a frame that does not is "
            f"either a fixture describing a build that cannot have happened, or a board off "
            f"disk predating the column. Pass the report that belongs to this frame, or give "
            f"the fixture the column its report claims.") from None


def simulate_remaining_draft(board: pl.DataFrame, state: DraftState, *, my_slot: int,
                             teams: int = 12, rounds: int = DEFAULT_ROUNDS,
                             forced: str | None = None,
                             w: float = DEFAULT_ESPN_WEIGHT, opp_noise: float = 1.0,
                             rng: np.random.Generator | None = None,
                             my_pick: MyPick | None = None,
                             report: BuildReport | None = None) -> list[np.ndarray]:
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

    **`report` says which currency this room ranks in**, and it is the whole of the sharp
    half of issue #199 -- see the `key` line below for what it decides and what that costs.
    """
    rng = rng or np.random.default_rng(0)
    from hub.draft.board import report_for
    report = report_for(board, report)
    pool = blended_adp(board, w, report=report)

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
    #
    # **Which forecast that is, is the report's answer and not this frame's schema.** Both
    # halves of the coupling are the draft-market stage's: `vor_proj` is the last of
    # `board.STAGE_COLUMNS["adp"]`, and `moments` -- what scores the season below -- reads
    # `proj_blend` and `proj_ppg` from that same list, falling through to `xfp_per_game`,
    # which is the scale `vor` is on. So one flag keys both sides, and asking it here is how
    # they are kept in step rather than kept in step by accident.
    #
    # It used to be `"vor_proj" in pool.columns`, and that had a live consequence rather
    # than only an owner: `vor_proj` exists only where ESPN ADP does, so the live room ranked
    # on the blended projection while every backtested room ranked on prior-season xFP, with
    # nothing anywhere recording that the two differ. Closing it is not available -- there is
    # no historical ESPN projection to give a past room, which is the same wall
    # `LIMITATIONS` hits twice already -- so it is named there instead. The gap is now
    # decided and stated; before, it was inherited from a column.
    #
    # Read only where it is used, which is the `my_pick is None` branch below and nowhere
    # else. Every caller that supplies a strategy -- `backtest.play`, `cohort`, both arms of
    # the backtest -- replaces the greedy outright, so eagerly resolving a currency none of
    # them ranks in made the room's ranking a precondition on runs that never ranked.
    vor = _greedy_currency(pool, report) if my_pick is None else None
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
    by_norm = {player_key(n): i for i, n in enumerate(names)}
    for seat in range(1, teams + 1):
        for held_name in roster_for(state, seat, teams, rounds):
            i = by_norm.get(player_key(held_name))
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
                assert vor is not None       # resolved above for exactly this branch
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


# How far a correction may move a player, as a fraction of his own ADP.
#
# Proportional rather than absolute on purpose: an absolute clamp is least protective exactly
# where damage is worst. Twelve picks at ADP 150 is noise; twelve picks at ADP 3 is
# catastrophic. At 20% the bound is 0.6 picks at the top of round one -- where consensus is
# tightest and a correction has least to add -- and about one round at ADP 60. A round is the
# unit that means something: a correction may say "this player is a round cheaper than the
# market thinks", not "he is a different tier".
#
# It lives in `config.DraftConfig` because it is a choice rather than a measurement, which is
# ADR-0006's line -- and the test that walks the tree for unregistered fitted constants
# caught it sitting here as a bare float.


def market_curve(adp: np.ndarray, proj: np.ndarray, window: int = 15):
    """The pick the market takes a given projection at. Returns (proj_asc, adp) for interp.

    Built from the board itself rather than fitted, because the relationship is steeply
    non-linear: near the top, players are separated by small projection gaps and few picks,
    so a point of projection is worth a handful of picks; by round twelve the same point
    spans dozens. A single fitted slope would over-move the top of the board and barely move
    the bottom, which is the opposite of where a correction can be trusted.

    Smoothed with a rolling median and forced monotone -- a higher projection never maps to
    a later pick -- so that noise in one player's ADP cannot bend the curve under everyone.
    """
    ok = np.isfinite(adp) & np.isfinite(proj)
    a, pr = np.asarray(adp)[ok], np.asarray(proj)[ok]
    if a.size == 0:
        return np.array([0.0]), np.array([0.0])
    order = np.argsort(pr)
    pr, a = pr[order], a[order]
    if window > 1 and a.size >= window:
        pad = window // 2
        a = np.array([np.median(a[max(0, i - pad):i + pad + 1]) for i in range(a.size)])
    # non-increasing in projection: better projection, never a later pick
    a = np.maximum.accumulate(a[::-1])[::-1]
    return pr, a


def corrected_adp(board: pl.DataFrame, clamp_frac: float | None = None) -> pl.Series:
    """ADP adjusted for the corrections this repo has measured and the market has not priced.

    THE PICK ranks on this. Ranking on raw ADP while *printing* a fitted correction beside it
    was an inconsistency: it said "our measurements show the market is wrong about this
    player", and then ordered the board by the market anyway.

    Every input carries a fitted coefficient with an interval -- touchdown luck (-0.540 QB,
    -0.286 WR), durability (-0.457 QB, -0.151 WR), current designation (-1.631 OUT/DOUBTFUL/
    IR). What is *not* validated is the assembly, and it cannot be: scoring a ranking against
    outcomes needs historical ADP, which ESPN does not retain (see ADR-0010). Hence the clamp
    -- the pieces are measured, the combination is bounded.

    A player with no correction does not move. That is by construction and it is the first
    half of the tripwire in `hub.draft.backtest`.
    """
    if clamp_frac is None:
        clamp_frac = DraftConfig().correction_clamp_frac
    # Columns read for arithmetic rather than for provenance, which is why these two survive
    # issue #199 rather than becoming a `report.adp` -- the same guard, on three of the same
    # columns, that `backtest.correction_report` keeps and #164 wrote the reason for. What
    # follows reads all three and subtracts two, so their presence is a precondition on the
    # operation rather than a second guess at what `build` did.
    #
    # It also could not ask a report if it wanted to. `board._attach_market` calls this
    # *during* the build, before the stage that would set the flag has been marked -- so the
    # only report in existence at this moment is one that does not yet describe this frame.
    # A producer is not a consumer of the report; see `regression.correct_projection` and
    # `durability.correct_projection`, which are inside `_stage` for the same reason.
    if not {"adp", "proj_blend", "proj_correction"} <= set(board.columns):
        return board["adp"] if "adp" in board.columns else pl.Series("adp", [], pl.Float64)

    adp = board["adp"].to_numpy().astype(float)
    proj = board["proj_blend"].to_numpy().astype(float)
    corr = board["proj_correction"].fill_null(0.0).to_numpy().astype(float)

    # `proj_blend` arrives already corrected -- `_attach_market` applies both corrections to
    # it in place and derives `proj_correction` as the delta -- so the uncorrected projection
    # is the difference, and it is where the curve has to be read (#39).
    #
    # Reading at `proj` and `proj + corr` measured each shift from the corrected point to a
    # *twice*-corrected one: one full correction too far along a curve that is steeply
    # non-linear by construction, so the error was largest exactly where the curve is
    # steepest and corrections are most trusted.
    raw = proj - corr
    # And the curve is built on the uncorrected projection. Built on the corrected one it
    # moved under every player whenever any correction changed -- so absorbing a single
    # advisory stage re-priced the whole board rather than the players that stage touched,
    # which is what #121 measured at 346 of 457 players.
    xs, ys = market_curve(adp, raw)
    # Both sides read off the curve, so a player whose own ADP already differs from it keeps
    # that difference -- the shift is the correction's effect, not a re-pricing of the player.
    shift = np.interp(raw + corr, xs, ys) - np.interp(raw, xs, ys)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    bound = clamp_frac * np.abs(np.nan_to_num(adp, nan=0.0))
    shift = np.clip(shift, -bound, bound)
    out = adp + np.where(np.isfinite(adp), shift, 0.0)
    # Null where `adp` is null, not NaN (#243). `to_numpy` turned every null into a NaN
    # and the NaN rode back into the frame as a float: on the served board of 2026-09-11
    # that was 293 of 457 rows passing `is_not_null()`, uncounted by `hub.inspect
    # --nulls`, and -- because `market_pick`'s `min` never replaces a key it cannot compare
    # against -- picked ahead of every drafted player whenever the undrafted one came
    # first in the pool. The corrected column takes the shape of the column it corrects.
    return pl.Series("adp_corrected", out).zip_with(
        board["adp"].is_not_null(), pl.Series([None] * len(out), dtype=pl.Float64))



@dataclass(frozen=True)
class ThePick:
    """The recommendation, and everything a renderer needs to explain it."""
    player: str
    pos: str | None
    via: str            # which market chose it, prose for the screen
    rank: float | None  # the number, in `rank_label` units
    rank_label: str     # what `rank` IS -- carried, never inferred from `via`
    notes: list[str]    # corrections, in points per game


def the_pick(board: pl.DataFrame, state: DraftState, *,
             my_slot: int = MY_SLOT, teams: int = TEAMS) -> ThePick | None:
    """What to draft right now. The one recommendation, for every tool that gives one.

    Draft market first, consensus behind it -- the same lexicographic rule either way, so
    the fallback is not a consolation prize: consensus-following is arm A of P0b, the arm
    that beat championship equity by twenty points a team-game.

    **This exists because there were two answers.** `hub.draft.board` led with this rule
    while `hub.draft.live` -- the poller you actually have open on the clock -- ranked by
    `edge` from round four and by `vor_live` before it, with no need term in the ordering
    at all; need was printed as a `*` and never sorted on. Two draft-night tools, two
    different recommendations, and no test could see the disagreement because they shared
    no code. They share this.

    Returns None only when the board carries neither ADP nor ECR, which means it did not
    build.
    """
    avail = remaining(board, state)
    held = held_positions(board, state, my_slot=my_slot, teams=teams)
    # Corrected ADP first: the market's pick, moved by what we have measured it to miss.
    # Raw ADP behind it, for a board built before `adp_corrected` existed. Consensus last.
    # (column to rank on, prose for the screen, column to *report*, its unit).
    #
    # Reported separately from ranked: on corrected ADP the drafter is comparing against a
    # room that sees raw ADP, so raw ADP is the useful number. The unit travels with the
    # value instead of being recovered from the prose -- `board.main` used to infer it with
    # `"ADP" in via`, which silently labelled an ADP value "ECR" the moment the prose changed
    # to "draft market, corrected".
    routes = (("adp_corrected", "draft market, corrected", "adp", "ADP"),
              ("adp", "draft market (ADP)", "adp", "ADP"),
              ("ecr", "consensus (ECR) -- no ADP today", "ecr", "ECR"))
    for by, via, report, label in routes:
        name = market_pick(avail, held, by=by)
        if name is None:
            continue
        row = board.filter(pl.col("player") == name).row(0, named=True)
        return ThePick(player=name, pos=row.get("pos"), via=via,
                       rank=row.get(report), rank_label=label, notes=pick_notes(row))
    return None


# --- the decision layer -----------------------------------------------------
#
# `held_positions` and `pick_notes` moved here from `hub.draft.board` on 2026-08-24. They
# were pulled out of print blocks that morning because they are decisions rather than
# rendering; this finishes the move, so that both draft-night tools can reach them without
# `hub.draft.live` importing a CLI. `board` and `live` render what this module decides.


# Show a touchdown-luck note beside THE PICK only when it is worth a drafter's attention.
# Private and lower-cased in intent: this is a display threshold, not a fitted constant, so
# it must not move the model version. See `hub.config.FITTED_MODULES`.
_TD_LUCK_NOTE = 0.5


def held_positions(board: pl.DataFrame, state: DraftState, *,
                   my_slot: int = MY_SLOT, teams: int = TEAMS) -> dict[str, int]:
    """How many of each position you already hold. Feeds positional need.

    Pulled out of the print block in `main` because it is a decision, not a rendering: it
    is what makes THE PICK fill a need rather than take the best player left. Inline, it
    read

        (remaining(board, st).is_empty() and []) or [ ... ]

    and `(x and []) or y` is `y` for every value of `x` -- an empty list is falsy, so the
    `or` always takes the right branch. The guard never fired, and it called `remaining()`
    on the whole board to throw the answer away.
    """
    mine = roster_for(state, my_slot, teams)
    if not mine:
        return {}
    got = board.filter(pl.col("player").is_in(mine))["pos"].drop_nulls().to_list()
    return dict(collections.Counter(got))


def pick_notes(row: dict) -> list[str]:
    """Where this repo's measurements say the draft market is wrong about THE PICK.

    Also a decision rather than a rendering -- each entry encodes a threshold for what is
    worth interrupting a drafter with, and those were previously spelled out inside an
    f-string block where nothing could test them.

    Deliberately not a score. These are corrections the market has not priced, shown so the
    drafter weighs them; folding them into a single number would hide which one fired.
    """
    notes = []
    luck = row.get("td_luck")
    if luck is not None and abs(luck) > _TD_LUCK_NOTE:
        notes.append(f"td luck {luck:+.2f}/gm")
    if row.get("missed"):
        notes.append(f"missed {int(row['missed'])} last season")
    if durability.is_flagworthy(row.get("injury_status")):
        notes.append(str(row["injury_status"]))
    return notes


def market_pick(pool: pl.DataFrame, counts: dict[str, int],
                by: str = "adp") -> str | None:
    """Best available in a market, that fills an unfilled starting slot.

    What the board leads with, and now the only thing that picks.

    P0 could not separate this from championship equity (+0.04, 95% CI [-3.64, +3.58],
    n=36), so equity was demoted to a tiebreaker. P0b -- the same question asked with a
    committed harness, `recommend()`'s shortlist as the design had always specified, four
    seasons and n=80 -- separated them decisively: **-19.66 points per team game, 95% CI
    [-23.16, -16.20]**, equity losing in every season and winning 9 of 80 drafts. Per the
    rule fixed before that run, equity left the output. See docs/adr/0009 -- restated
    2026-09-11: that interval is row-clustered (season-clustered it is [-26.68, -11.45]) and
    the magnitude is not quotable, re-runs at fixed inputs having returned -17.30, -12.48 and
    -11.59 (docs/gate-power.md); the 4-of-4 and the verdict are what stand.

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
