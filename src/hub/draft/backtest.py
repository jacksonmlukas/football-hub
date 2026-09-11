"""Does championship equity beat the draft market? Measured on realised outcomes.

This is P0 built as code, per ADR-0007 (a measurement that steers the product must be
committed code).
The original P0 ran in an afternoon and committed nothing: both its commits touch
`docs/next.md` and nothing else, so the +0.04 [-3.64, +3.58] that demoted championship equity
to a tiebreaker cannot be reproduced. Two departures from its own pre-registered design went
unrecorded as a result -- the shortlist was top-8 by VOR rather than `recommend()`'s, and n
was 36 rather than the 60 the design fixed.

**The design, pre-registered.** Two arms on the same room and the same seed, so the comparison
is paired:

  * **A, the market**: best available by that season's consensus that fills an unfilled
    starting slot, lexicographically -- `optimize.market_pick(by="ecr")`.
  * **B, the optimizer**: the top of `win_probability` over `recommend()`'s shortlist, ties
    broken by consensus.

**Outcome**: realised weekly points. Best legal lineup each week from what the players
actually scored, summed and divided by team games. Never a projection -- escaping the
circularity is the entire point, since one object both ranks candidates and scores seasons.

**Why arm A ranks on consensus rather than ADP.** ESPN publishes ADP for the current season
only, and the FantasyPros archive carries no historical ADP. Consensus is a real market and
genuinely contemporaneous; the cost is that arm A is a *consensus*-follower while the shipped
THE PICK is a *draft-market*-follower. That is the first of four named gaps between this
harness and the tool it audits, all listed under LIMITATIONS below.

**Decision rule, fixed before the numbers.** Asymmetric on purpose: the market already leads,
and the burden is on the complicated thing.

    CI excludes zero favouring B  -> promote equity back to the headline; P1 fires
    CI contains zero              -> nothing changes
    CI excludes zero favouring A  -> remove equity from the draft-night output entirely

    uv run python -m hub.draft.backtest --seasons 2022,2023,2024,2025 --drafts 20
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.config import DraftConfig, RosterConfig, drafted_positions
from hub.draft.board import BuildReport, board_as_of
from hub.draft.optimize import (
    DEFAULT_ROUNDS,
    ROOM,
    market_pick,
    root_seed,
    simulate_remaining_draft,
    stream,
)
from hub.draft.season import CorrelationReport, lineup_points
from hub.draft.state import DraftState

# The arm under test, from the exhibit. This harness is the one production reader
# `hub.exhibits` has -- ADR-0007 keeps the measurement re-runnable and ADR-0009 says
# reopening it means re-running this file -- and
# `tests/contracts/test_the_exhibit_is_not_a_dependency.py` holds it to being the only one.
from hub.exhibits.championship_equity import rank_tiers, win_probability
from hub.league import REG_SEASON_WEEKS
from hub.models.experiment import (
    BOOTSTRAP,  # noqa: F401 -- re-exported: tests reach it as `bt.BOOTSTRAP`
    SEASON_CLUSTER,
    Actions,
    Ceiling,
    realised_ppg,  # noqa: F401 -- same
    run_gate,
    stamped_for_publication,  # noqa: F401 -- same; it was written here and moved (#135)
    summarise,  # noqa: F401 -- same
    walk_forward_inputs,
)
from hub.names import player_key, relaxed_key

NOT_FITTED_BECAUSE = (
    "the draft gate's harness. VOID_FLOOR is the share of drafted names lost to a join failure "
    "above which a run is not reported at all -- a pre-registered guard, not a fitted quantity, "
    "and the same shape as weekly_gate.VOID_FLOOR. Nothing here predicts; the arms and the "
    "constants they draw on live in hub.draft.optimize and hub.models.predict. "
)

# Gaps between this harness and the tool it audits. Written here rather than in the result,
# because a limitation discovered after the numbers is a rationalisation.
LIMITATIONS = (
    "arm A follows consensus (ECR); the shipped THE PICK follows the draft market (ADP), "
    "which ESPN publishes for the current season only",
    "arm B scores seasons on prior-season xFP; the live board scores on proj_blend, which "
    "blends in an ESPN projection that does not exist for past seasons",
    "arm B breaks ties by consensus; the shipped tool refuses to break them and asks you",
    "the room is simulated -- consensus plus fitted pick noise, lexicographic need -- not "
    "the eleven people who were actually in those drafts",
    "arm B is the POST-FIX optimizer: win_probability now seeds every seat with the roster "
    "it already holds. P0 measured the pre-fix one, which was blind to your own roster, so "
    "any movement from P0's +0.04 cannot be read as 'the shortlist tipped it'",
    # Issue #196, and the one limitation here that is about comparability between runs rather
    # than between this harness and the product. Left as a limitation rather than fixed: both
    # draws are vectorised over the whole frame in one call, so making a player's draw a
    # function of his identity rather than of his row would re-pair every player in every
    # existing figure -- a larger change to the thing being measured than either #195 or #196
    # is, and one that should be made deliberately with a re-run rather than in passing. What
    # closes the reporting half of it is `board_digest`: the coupling is undetectable without
    # a stamp naming the frame, and detectable with one.
    # Issue #199. Named here rather than closed, because closing it needs a preseason ESPN
    # projection for a past season and ESPN publishes one for the current season only -- the
    # same wall the first two limitations hit. What #199 changed is that the choice is now
    # made by the Board's `BuildReport` instead of falling out of whether a column happened
    # to be on the frame, so it is a decision with a line of code behind it and a limitation
    # with a line here, rather than an accident nothing recorded.
    "the ROOM ranks in a different currency in the two worlds: on a live Board the greedy "
    "and the simulated opponents rank on vor_proj (replacement-adjusted proj_blend, which "
    "blends in ESPN's projection), and on every backtested Board they rank on vor "
    "(replacement-adjusted prior-season xFP). It follows the draft-market stage, because "
    "the season underneath is scored on the same split -- models.predict.moments coalesces "
    "proj_blend and proj_ppg before falling through to xfp_per_game -- so ranking and "
    "scoring stay in one currency. What cannot be made to match is the two harnesses: arm "
    "B's room is a prior-season-xFP room and the shipped room is not",
    "the noise is drawn per Board ROW, not per player: both stochastic quantities are arrays "
    "whose last axis is the Board's height (optimize.simulate_remaining_draft draws one "
    "pick-noise normal per row; predict.correlated_normal draws (n_sims, weeks, mu.size)). "
    "Probed directly: drop one player and everyone ABOVE him keeps his pick-noise draw, "
    "nobody below him does, and no player's season draw survives at all. So any commit that "
    "moves Board membership -- MIN_GAMES, the xFP imputation, a join key, the as-of boundary "
    "by a day -- re-pairs the whole board and is a total re-draw, not a small perturbation. "
    "Two runs are comparable only at an identical board_digest",
)

def score_roster(names: Sequence[str], pos: Sequence[str], realised: pl.DataFrame,
                 weeks: int = REG_SEASON_WEEKS) -> float:
    """Points per team game from the best legal lineup each week.

    Weekly rather than on season totals. Totals would silently reward even production and
    erase bye weeks and mid-season injuries -- the things a roster is built to survive -- and
    would score the backtest on a different objective than the one being tested.

    A player with no realised row scored nothing, which is correct: he was hurt, cut, or
    never played. Zero, not null, because the lineup rule has to be able to bench him.
    """
    if not names:
        return 0.0
    keys = [player_key(n) for n in names]
    grid = np.zeros((len(keys), weeks))
    idx = {k: i for i, k in enumerate(keys)}
    for row in realised.filter(pl.col("player").is_in(keys)).iter_rows(named=True):
        w = int(row["week"])
        if 1 <= w <= weeks:
            grid[idx[row["player"]], w - 1] = float(row["points"])
    # (sims=1, weeks, roster) -- the same rule the season simulator uses.
    total = lineup_points(grid.T[None, :, :], np.asarray(pos, dtype=object))
    return float(total.sum() / weeks)


class RealisedNames(NamedTuple):
    """The names one season's realised frame carries, under both comparisons."""
    exact: frozenset[str]      # every `player_key` with a realised row
    relaxed: frozenset[str]    # the `relaxed_key` of each


def realised_names(realised: pl.DataFrame) -> RealisedNames:
    """Both key sets, built once per season rather than once per roster."""
    keys = [str(k) for k in realised["player"].unique().to_list() if k is not None]
    return RealisedNames(frozenset(keys), frozenset(relaxed_key(k) for k in keys))


def join_failures(names: Sequence[str], known: RealisedNames) -> int:
    """How many drafted names failed to join the season they were scored against.

    `score_roster` scores a name with no realised row as zero, which is right for a player
    who was hurt, cut or never played and wrong for one whose name the stats source spelled
    differently -- and nothing in the score tells the two apart. The weekly gate separates
    them by position in a matrix; here the only key *is* the name, so the discriminator is a
    crosswalk: a drafted name absent from the realised set that matches a realised name under
    `relaxed_key` (first initial and surname) is a join failure, and one matching nothing is
    never-played. That reads a genuine rookie who never took a snap as the harness being
    right, and an abbreviated spelling as the harness being wrong, which is the distinction
    issue #46 exists to draw.
    """
    failed = 0
    for name in names:
        key = player_key(name)
        if key in known.exact:
            continue
        if relaxed_key(key) in known.relaxed:
            failed += 1
    return failed


def _pool_index(pool: pl.DataFrame, name: str) -> int:
    return pool["player"].to_list().index(name)


def market_strategy(by: str = "ecr"):
    """Arm A. Best available in `by` that fills an unfilled starting slot."""
    def pick(pool, live, counts, taken):
        avail = pool[[int(i) for i in live]]
        name = market_pick(avail, counts, by=by)
        if name is None:
            return int(live[0])
        return _pool_index(pool, name)
    return pick


FORESIGHT = "_foresight"


def with_foresight(board: pl.DataFrame, realised: pl.DataFrame) -> pl.DataFrame:
    """The board plus a market that already knows how the season went.

    `FORESIGHT` is a lower-is-better ranking of realised season points, which is exactly the
    shape `market_pick` reads -- so the foresight arm is the *incumbent arm reading a perfect
    ranking*, not a second strategy. That matters for what the ceiling means: it isolates the
    value of knowing the ranking, holding the drafting rule fixed, so the gap between it and
    the arm under test is attributable to the ranking rather than to two different players
    behaving differently.

    Row order is preserved because `play` indexes the board by row.

    A player with no realised row ranks last, not null. He scored nothing, which is what
    `score_roster` already assumes about him, and a null would sort him into the middle of a
    lower-is-better column.
    """
    seen = (realised.group_by("player")
                    .agg(pl.col("points").sum().alias("_pts")))
    keyed = board.with_columns(
        pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("_k"))
    joined = keyed.join(seen, left_on="_k", right_on="player", how="left")
    return (joined.with_columns(pl.col("_pts").fill_null(-1.0))
                  .with_columns(pl.col("_pts").rank("ordinal", descending=True)
                                  .cast(pl.Float64).alias(FORESIGHT))
                  .drop("_k", "_pts"))


def draft_root(seed: int, season: int, k: int) -> np.random.SeedSequence:
    """The root every stream of one (season, draft) descends from.

    One function rather than a repeated expression, because `compare` and `ceiling` are
    paired against each other -- `ceiling`'s incumbent column has to be `compare`'s incumbent
    column, drawn against the same field -- and two copies of a seeding rule are two rules as
    soon as one of them is edited. It used to be the expression `seed + 1000 * season + k`
    written out twice.

    What descends from it is `optimize.ROOM`, `optimize.ROLLOUT` and `optimize.SEASON_SIM`;
    the tree is documented there.

    **Not `cohort.seed_for`, and #200 is where the difference is stated rather than left to
    look like a drift.** The Cohort the other two Gates score still opens
    `default_rng(seed + 1000 * season + k)`, the integer draw this harness used before #195.
    This harness left it because arm B evaluates draft futures inside the room it is scored
    in, and on the integer line rollout 0 *was* that room; a Cohort has no arm B under it,
    so the leak cannot occur there, and both season-side gates' published figures were
    measured on the integer draw. Two recipes, each declared once, each with its reason --
    `tests/unit/test_cohort.py` guards that neither is written anywhere else.
    """
    return root_seed(seed, season, k)


def optimizer_strategy(board: pl.DataFrame, *, my_slot: int, teams: int, rounds: int,
                       n_draft_sims: int, n_season_sims: int,
                       seed: int | np.random.SeedSequence,
                       tiebreak: str = "ecr",
                       report: BuildReport | None = None,
                       correlation: CorrelationReport | None = None):
    """Arm B. Top of `win_probability` over `recommend()`'s shortlist, ties broken by `by`.

    The tie-break is not a tidy default: `rank_tiers` exists because the top two candidates
    routinely sit inside two standard errors of each other, and its own docstring notes that
    re-running names a different one. The shipped tool hands that tie to a human. A harness
    has no human, so it needs a rule the product does not have -- and taking the consensus
    among co-leaders makes arm B exactly "equity leads, market breaks ties", the mirror of
    arm A. The lead is then the only difference between the two arms, which is the question.
    """
    from hub.draft.board import recommend

    def pick(pool, live, counts, taken):
        state = DraftState(taken=list(taken))
        overall = len(taken) + 1
        try:
            _, rec = recommend(board, overall, rounds=rounds, state=state, report=report)
        except ValueError:
            rec = pool[[int(i) for i in live]].head(10)
        names = [n for n in rec["player"].to_list()
                 if n in set(pool["player"][[int(i) for i in live]].to_list())]
        if not names:
            return int(live[0])
        if len(names) == 1:
            return _pool_index(pool, names[0])
        wp = win_probability(board, state, names, my_slot=my_slot, teams=teams,
                             rounds=rounds, n_draft_sims=n_draft_sims,
                             n_season_sims=n_season_sims, seed=seed,
                             report=report, correlation=correlation)
        leaders = rank_tiers(wp).filter(pl.col("co_leader"))["player"].to_list()
        ranked = (board.filter(pl.col("player").is_in(leaders))
                       .sort(tiebreak, nulls_last=True))
        return _pool_index(pool, ranked["player"][0] if ranked.height else leaders[0])
    return pick


def play(board: pl.DataFrame, strategy, *, my_slot: int, teams: int, rounds: int,
         rng: np.random.Generator,
         report: BuildReport | None = None) -> tuple[list[str], list[str]]:
    """Play one draft with `strategy` in my seat. Returns (my player names, my positions)."""
    rosters = simulate_remaining_draft(board, DraftState(taken=[]), my_slot=my_slot,
                                       teams=teams, rounds=rounds, rng=rng,
                                       my_pick=strategy, report=report)
    mine = rosters[my_slot - 1]
    names = [board["player"][int(i)] for i in mine]
    pos = [board["pos"][int(i)] or "NA" for i in mine]
    return names, pos


def compare(boards: dict[int, pl.DataFrame], realised: dict[int, pl.DataFrame], *,
            n_drafts: int = 20, seed: int = 0, my_slot: int | None = None,
            teams: int | None = None, rounds: int = DEFAULT_ROUNDS,
            n_draft_sims: int = 12, n_season_sims: int = 250,
            on_draft: Callable[[int, int, int], None] | None = None,
            correlation: CorrelationReport | None = None) -> pl.DataFrame:
    """Paired arm A against arm B, one row per (season, draft).

    Pure: takes frames, returns a frame, touches no network. That is what makes the
    statistics testable, and a backtest whose statistics can only be exercised by hitting
    ESPN is one nobody re-runs.

    `n_draft_sims` and `n_season_sims` default to 12 x 250, the budget every published figure
    in ADR-0009 was measured at. It used to read "pinned at the shipped 12 x 250", and there
    is no shipped path: equity left the draft-night output under that ADR, so the budget is
    the measurement's own and not a product setting it mirrors (#198). The first P0 run used
    6 x 120 -- a quarter of it -- and produced -5.79 [-9.17, -2.45], an artifact that
    vanished at adequate power. Lower them and you are measuring a different optimizer.

    **The season stays the cluster, and #195 is why the question was asked.** Before it, the
    rows were dependent for two separate reasons: they share a board, a player pool and one
    realisation of the year (issue #45's argument), and consecutive drafts' arm-B evaluations
    shared 11 of their 12 futures, which is a dependence the seeding manufactured and which
    no clustering key named. The second reason is gone -- each draft's futures are now its
    own -- and the first is untouched, because it is a claim about the data rather than about
    the code. So the answer is unchanged and the reasoning behind it is now one reason
    instead of two; `tests/contracts/test_gates_cluster_on_the_season.py` holds the call.
    """
    cfg = RosterConfig()
    my_slot = cfg.slot if my_slot is None else my_slot
    teams = cfg.teams if teams is None else teams

    rows = []
    for season in sorted(boards):
        board, real = boards[season], realised[season]
        known = realised_names(real)
        arm_a = market_strategy()
        for k in range(n_drafts):
            # Common random numbers: the same room, twice. The only thing that differs
            # between the arms is who sits in my seat. `stream(root, ROOM)` is a pure
            # function of the root, so the two calls below open one room and both arms play
            # it -- the pairing is unchanged by #195 and is asserted, not assumed.
            #
            # What #195 changed is the level *below* this line. Arm B's evaluation rollouts
            # and its season simulations now hang off the same root as separate coordinates,
            # so they can no longer be the room they are scored in, and two consecutive
            # drafts no longer share futures.
            root = draft_root(seed, season, k)
            a_names, a_pos = play(board, arm_a, my_slot=my_slot, teams=teams,
                                  rounds=rounds, rng=stream(root, ROOM))
            arm_b = optimizer_strategy(board, my_slot=my_slot, teams=teams, rounds=rounds,
                                       n_draft_sims=n_draft_sims,
                                       n_season_sims=n_season_sims, seed=root,
                                       correlation=correlation)
            b_names, b_pos = play(board, arm_b, my_slot=my_slot, teams=teams,
                                  rounds=rounds, rng=stream(root, ROOM))
            # A callback, not a print, so `compare` stays pure and the tests stay quiet. This
            # run takes long enough that a caller needs to know it is alive: the first attempt
            # was killed at 49 minutes having emitted nothing at all, because the only output
            # was buffered behind a pipe and the per-season lines never reached anyone.
            if on_draft is not None:
                on_draft(season, k + 1, n_drafts)
            rows.append({
                "season": season, "draft": k,
                "market": score_roster(a_names, a_pos, real),
                "optimizer": score_roster(b_names, b_pos, real),
                # Beside the scores, never inside them (#46): how many of each arm's drafted
                # names were scored zero for a spelling rather than for a season. Both arms
                # draft `rounds` players, so one `picks` serves both rates.
                "market_failed": join_failures(a_names, known),
                "optimizer_failed": join_failures(b_names, known),
                "picks": len(a_names),
            })
    out = pl.DataFrame(rows)
    return out.with_columns((pl.col("optimizer") - pl.col("market")).alias("diff"))


# What this gate's ceiling *is*, spelled where it is printed (#138). The draft gate's ceiling
# is genuine foresight: `ceiling` hands the arm under test the season in advance, so the
# largest effect any board could show is what a drafter who already knew the season achieves.
# That is the same *kind* of arm as the weekly gate's and a different kind from the lineup
# gate's variance oracle -- and `docs/gate-power.md` stage 2 compares each gate's MDE against
# its own declared arm, never another's. Until this constant existed the line read
# `paired_report`'s default, which happened to be the right words; a default that happens to
# be right is the shape this repo keeps finding, and #138 is where it was written down.
CEILING_ARM = "perfect foresight -- the season known in advance"


def ceiling(boards: dict[int, pl.DataFrame], realised: dict[int, pl.DataFrame], *,
            n_drafts: int = 20, seed: int = 0, my_slot: int | None = None,
            teams: int | None = None, rounds: int = DEFAULT_ROUNDS,
            on_draft: Callable[[int, int, int], None] | None = None) -> pl.DataFrame:
    """What a drafter who already knew the season achieves against the incumbent arm.

    This bounds what *any* board could deliver, which is the number the underpowered-gate rule
    in `docs/gate-power.md` compares an effect against. A gate whose MDE exceeds this cannot
    resolve a real effect from a perfect one, and reporting an interval from it would be
    reporting noise with a decimal point.

    **Its own paired frame, not a third column on `compare`'s.** Widening that frame would
    move the sweep digest `tests/unit/test_experiment.py` pins and re-price every verdict that
    rests on it, as a side effect of adding a diagnostic -- and `docs/track-record.md` rule 1
    makes those numbers commit-dated. The two frames share `season`, `draft` and the same
    seeds, so they pair row for row.

    **Recomputed per season set, never cached.** A ceiling served from a run over other
    seasons is a bound on a different question, and it would be indistinguishable from a right
    answer: same units, same shape, plausible size. Taking `boards` by value and returning a
    frame is what makes that impossible to get wrong.

    Common random numbers with `compare`: the same `draft_root(seed, season, k)` room, so the
    incumbent column here is that same column there, drawn against the same field. Both reach
    the room through that one function rather than through two copies of an expression, which
    is what keeps the claim true after a seeding change rather than only before one.
    """
    cfg = RosterConfig()
    my_slot = cfg.slot if my_slot is None else my_slot
    teams = cfg.teams if teams is None else teams

    rows = []
    for season in sorted(boards):
        board, real = boards[season], realised[season]
        seeing = with_foresight(board, real)
        arm_a, arm_c = market_strategy(), market_strategy(by=FORESIGHT)
        for k in range(n_drafts):
            root = draft_root(seed, season, k)
            a_names, a_pos = play(board, arm_a, my_slot=my_slot, teams=teams,
                                  rounds=rounds, rng=stream(root, ROOM))
            c_names, c_pos = play(seeing, arm_c, my_slot=my_slot, teams=teams,
                                  rounds=rounds, rng=stream(root, ROOM))
            if on_draft is not None:
                on_draft(season, k + 1, n_drafts)
            rows.append({
                "season": season, "draft": k,
                "market": score_roster(a_names, a_pos, real),
                "foresight": score_roster(c_names, c_pos, real),
            })
    out = pl.DataFrame(rows)
    return out.with_columns((pl.col("foresight") - pl.col("market")).alias("diff"))


# Your first six turns from slot 3 of 12. Fixed rather than read from the live draft state,
# because `--diagnose` is run twice at two different commits and anything state-dependent
# would not be comparable between them.
#
# The spread matters: at pick 3 you hold nothing, so seeding a roster is a no-op and the
# objective is unaffected by it. By pick 70 you hold five players, and any blindness to them
# has had five chances to show.
DIAGNOSE_PICKS = (3, 22, 27, 46, 51, 70)


def diagnose(board: pl.DataFrame, report: BuildReport, *,
             picks: Sequence[int] = DIAGNOSE_PICKS,
             my_slot: int | None = None, teams: int | None = None,
             rounds: int = DEFAULT_ROUNDS, n_draft_sims: int = 12,
             n_season_sims: int = 250, seed: int = 0,
             correlation: CorrelationReport | None = None) -> pl.DataFrame:
    """What championship equity recommends at each of your first turns.

    Run once before a change to the objective and once after, and diff. The draft is advanced
    by the *market* rather than by equity, so the path through the draft is identical in both
    runs and the only thing that can differ is what equity says about the same situation.

    Returns one row per pick: what you hold, who equity names, and by how much.

    **`report` says which market that is, and it is required rather than defaulted for the
    same reason `board_as_of` stopped letting a caller drop one.** Which market to advance by
    is a question about what `build` did -- did the stage that leaves `adp` run -- and this
    site used to answer it a second time, privately, by looking for the column. Issue #131
    gave that question one owner; this site was left out of that change because the file was
    owned elsewhere at the time, and what it was left holding is its own copy of the answer
    and of the string `"adp"` -- which is `board.STAGE_COLUMN["adp"]` restated somewhere
    nothing would ever update it.

    No board changes hands differently for this. On every board reachable today the column
    and the flag agree -- a stage that leaves no column is one the report already calls
    absorbed -- so this moves who owns the question rather than what `--diagnose` draws. That
    is the point: the two agreeing is a property of today's stages, not a rule, and the
    report is where the rule lives.

    **`correlation` is the second report, and it is about the simulation rather than the
    board.** `BuildReport` says which stages built the frame this reads; a `CorrelationReport`
    says how much of the season simulation underneath was actually correlated. They are two
    reports because they are two runs: the board was built once, and the equity below is
    thousands of draws made now. Passing one in is what carries the count out to `main`,
    which is the only place with an output to put it in.
    """
    cfg = RosterConfig()
    my_slot = cfg.slot if my_slot is None else my_slot
    teams = cfg.teams if teams is None else teams
    want = set(picks)
    rows: list[dict] = []
    # The same seeding tree `compare` uses, and for the same reason: the draft advanced below
    # is the room every `win_probability` call in it is evaluated against, so on the old
    # arithmetic rollout 0 was that room. Two runs at two commits still walk an identical
    # path, which is the property this function exists for -- the root is a function of
    # `seed` alone.
    root = root_seed(seed)

    from hub.draft.board import recommend

    def pick(pool, live, counts, taken):
        overall = len(taken) + 1
        avail = pool[[int(i) for i in live]]
        if overall in want:
            state = DraftState(taken=list(taken))
            try:
                _, rec = recommend(board, overall, rounds=rounds, state=state,
                                   report=report)
                names = [n for n in rec["player"].to_list()
                         if n in set(avail["player"].to_list())]
            except ValueError:
                names = []
            if len(names) >= 2:
                wp = rank_tiers(win_probability(
                    board, state, names, my_slot=my_slot, teams=teams, rounds=rounds,
                    n_draft_sims=n_draft_sims, n_season_sims=n_season_sims, seed=root,
                    report=report, correlation=correlation))
                top = wp.row(0, named=True)
                pos_of = dict(zip(board["player"].to_list(), board["pos"].to_list(), strict=True))
                # Does any co-leader fill a slot you cannot currently start? The tripwire
                # needs this: a need-filling candidate the simulation cannot separate from
                # the leader means the objective has not *rejected* need, it has declined
                # to distinguish -- which is a tie, not a defect.
                from hub.config import required_starters
                req = required_starters(cfg)
                short = [p for p, n in req.items() if counts.get(p, 0) < n]
                co = wp.filter(pl.col("co_leader"))["player"].to_list()
                need_co_led = any(pos_of.get(c) in short for c in co)
                rows.append({
                    "need_co_led": bool(need_co_led),
                    "pick": overall,
                    "held": ", ".join(f"{k}{v}" for k, v in sorted(counts.items())) or "-",
                    "leader": top["player"],
                    "leader_pos": pos_of.get(top["player"]),
                    "lift": float(top["lift"]),
                    "co_leaders": int(wp["co_leader"].sum()),
                    "candidates": len(names),
                    # Typed rather than parsed back out of `held`, which is for reading.
                    **{f"held_{p.lower()}": int(counts.get(p, 0))
                       for p in drafted_positions()},
                })
        # Advance by the market so the path is identical across runs. Which market is the
        # report's answer rather than this frame's: `avail` is a row slice of the board, so
        # its columns are the board's, and asking them here is the sniffing issue #131 gave
        # one owner. A board carrying no draft market is replayed on consensus, which is what
        # a past season gets in any case -- see `market_pick`, which keeps the two apart.
        name = market_pick(avail, counts, by="adp" if report.adp else "ecr")
        return _pool_index(pool, name) if name else int(live[0])

    simulate_remaining_draft(board, DraftState(taken=[]), my_slot=my_slot, teams=teams,
                             rounds=rounds, rng=stream(root, ROOM), my_pick=pick,
                             report=report)
    return pl.DataFrame(rows)


def tripwire(board: pl.DataFrame, diagnosed: pl.DataFrame) -> list[str]:
    """Whether an objective is fit to pick with: does it ever name a player at a required
    position you have already filled, ahead of one filling a slot you have not?

    **This is a quality check on the objective, not a regression gate on a code change**, and
    conflating those two jobs is how it got talked past. On 2026-08-24 it fired at picks 46
    and 70 for naming a running back while WR and QB sat empty. Read as a regression gate --
    "did my refactor break something?" -- that looked like a false positive, because the
    need-filling alternatives were co-leaders and a tie is not a rejection. So a co-leader
    clause was added and the gate went quiet.

    Read as a quality check, it was correct and the clause was wrong. P0b then measured that
    same running-back-over-need preference at **-19.66 points per team game** against
    consensus-following, across four seasons, losing in all of them. The clause is reverted:
    a need-filling co-leader means the objective cannot tell the difference between filling a
    hole and not filling one, which is exactly the thing worth knowing.

    Deliberately still not a threshold on how much a recommendation moved. A correctness fix
    that changes recommendations is doing its job.
    """
    from hub.config import required_starters
    required = required_starters(RosterConfig())
    bad = []
    for r in diagnosed.iter_rows(named=True):
        held = {p: int(r.get(f"held_{p.lower()}", 0)) for p in required}
        pos = r["leader_pos"]
        if pos in required and held.get(pos, 0) >= required[pos]:
            unfilled = [p for p, n in required.items() if held.get(p, 0) < n]
            if unfilled:
                co = " (a co-leader fills one, which is not a defence)" if r.get(
                    "need_co_led", False) else ""
                bad.append(f"pick {r['pick']}: named {r['leader']} ({pos}), but {pos} is "
                           f"full and {'/'.join(unfilled)} is not{co}")
    return bad


def correction_report(board: pl.DataFrame) -> pl.DataFrame:
    """Which players corrected ADP moves, and by how much. Sorted by size of move.

    The diagnostic for ADR-0011, and the shape is deliberately the same as the one that
    gated the roster-seeding fix: run it, read it, and check the tripwire below before
    shipping a change to what the draft ranks on.
    """
    # Columns read for arithmetic rather than for provenance, which is why this one survives
    # issue #131 rather than becoming a `report.adp`. What follows selects these five and
    # subtracts two of them, so their presence is a precondition on the operation -- the same
    # guard, on three of the same columns, that `optimize.corrected_adp` keeps for the same
    # reason. It is not a second guess at what `build` did, and it must not become one: this
    # is also reached from `correction_tripwire` with a hand-built frame and no report behind
    # it, and answering "can I subtract these" out of a report would refuse that frame for a
    # stage it never claimed to have run.
    #
    # The provenance question a frame genuinely cannot answer -- absent Corrected ADP, or one
    # that moved nobody -- belongs to the caller, and `main` asks the report before it renders
    # a word of this.
    need = {"player", "pos", "adp", "adp_corrected", "proj_correction"}
    if not need <= set(board.columns):
        return pl.DataFrame(schema={"player": pl.Utf8, "pos": pl.Utf8, "adp": pl.Float64,
                                    "adp_corrected": pl.Float64, "move": pl.Float64,
                                    "proj_correction": pl.Float64})
    return (board.select("player", "pos", "adp", "adp_corrected", "proj_correction")
                 .drop_nulls("adp")
                 .with_columns((pl.col("adp_corrected") - pl.col("adp")).alias("move"))
                 .filter(pl.col("move").abs() > 1e-9)
                 .sort(pl.col("move").abs(), descending=True))


def correction_tripwire(board: pl.DataFrame, clamp_frac: float | None = None) -> list[str]:
    """Fixed before the numbers. Fires only on things that cannot happen if the code is right.

    Deliberately NOT "did the recommendation change" -- it is supposed to change, and gating
    on that would reject the change for working. That was the mistake made with the
    seeding-fix tripwire on the morning of 2026-08-24 and corrected the same day.

    Two impossibilities:
      * a player carrying no fitted correction moves at all -- the shift is a function of the
        correction, so zero in must give zero out;
      * any move exceeds the clamp -- the clamp is applied unconditionally.

    Either means the exchange rate or the clamp is wrong, not that the corrections disagree
    with the market.
    """
    from hub.config import DraftConfig
    if clamp_frac is None:
        clamp_frac = DraftConfig().correction_clamp_frac
    rep = correction_report(board)
    bad: list[str] = []
    for r in rep.iter_rows(named=True):
        if abs(r["proj_correction"] or 0.0) < 1e-12:
            bad.append(f"{r['player']} has no correction but moved "
                       f"{r['move']:+.2f} picks")
        if abs(r["move"]) > clamp_frac * abs(r["adp"]) + 1e-6:
            bad.append(f"{r['player']} moved {r['move']:+.2f} picks, past the "
                       f"{clamp_frac:.0%} clamp of {clamp_frac * abs(r['adp']):.2f}")
    return bad


# The pre-registered actions, fixed before the numbers. The rule that chooses between them
# is `experiment.gate` -- one implementation, ADR-0019 -- and these are the three sentences
# only this gate can write.
ACTIONS = Actions(
    adopt="PROMOTE: championship equity returns to the headline, and P1 (break the "
          "circularity) fires.",
    remove="REMOVE: championship equity leaves the draft-night output. A tiebreaker "
           "measurably worse than the market steers close calls the wrong way.",
    show="NO CHANGE: the market leads and equity stays a tiebreaker.")


# Above this share of either arm's drafted names lost to a join failure, the run is VOID
# rather than reported. **Pre-registered here on 2026-09-11, before any run was made under
# it** (issue #46): the value is the weekly gate's `VOID_FLOOR`, adopted for the same reason
# that one was set tight -- the error is directional. A name that fails to join scores zero
# for the season, the two arms draft different players, and so a differential failure rate is
# a differential bias in the headline number rather than noise around it. Each gate's floor is
# its own pre-registration about its own join, not one shared bar declared twice: `MIN_SE` is
# one claim about significance, and this is a claim about what a drafted name is worth when
# the stats source cannot find him.
VOID_FLOOR = 0.02


def join_failure_rates(paired: pl.DataFrame) -> dict[str, float]:
    """Both arms' share of drafted names that failed to join, over every row of the run.

    Computed from the counts `compare` carries beside the scores, so the rates describe what
    was actually drafted and scored rather than a second draft that happened to agree. A run
    with no rows has no rate: NaN, and `picks` of zero, which `void_condition` reads as
    nothing to void.
    """
    picks = float(paired["picks"].sum()) if paired.height else 0.0
    if not picks:
        return {"picks": 0.0, "market": float("nan"), "optimizer": float("nan")}
    return {"picks": picks,
            "market": float(paired["market_failed"].sum()) / picks,
            "optimizer": float(paired["optimizer_failed"].sum()) / picks}


def void_condition(rates: dict[str, float]) -> str | None:
    """The precondition this gate hands `experiment.run_gate`, already phrased.

    Either arm above the floor voids. The bias is differential, and which arm carries it
    does not matter: a failure on arm A alone understates the incumbent, on arm B alone
    understates the arm under test, and a verdict read over either is a verdict about the
    spelling. Below the floor the run is exactly the run there was -- `run_gate` with no void
    -- which `tests/unit/test_backtest.py` holds on the summary, the seasons and the verdict.
    """
    if not rates["picks"]:
        return None
    if max(rates["market"], rates["optimizer"]) <= VOID_FLOOR:
        return None
    return (f"VOID: {rates['market']:.1%} of arm A's drafted names and "
            f"{rates['optimizer']:.1%} of arm B's are a join failure -- the player has a "
            f"realised row under another spelling and was scored zero for the season -- "
            f"against a pre-registered floor of {VOID_FLOOR:.0%}.\n  The two arms draft "
            f"different players, so this is a differential bias in the headline number "
            f"rather than noise around it. Fix the join before reading any number below.")


def join_report(rates: dict[str, float]) -> list[str]:
    """Both arms' rates beside the floor, and a second line when the arms differ.

    Said on every run and not only a void one, because a rate under the floor is still a
    fact about what the interval below was measured on -- and because "the two rates
    differ" is the sentence that says the bias is differential, which is the whole reason
    the floor exists. Nothing when the run drafted nothing.
    """
    if not rates["picks"]:
        return []
    lines = [f"  join failures: market {rates['market']:.1%}, optimizer "
             f"{rates['optimizer']:.1%} of {int(rates['picks'])} drafted names "
             f"(floor {VOID_FLOOR:.0%})"]
    if rates["market"] != rates["optimizer"]:
        lines.append(f"  the arms' failure rates differ by "
                     f"{abs(rates['market'] - rates['optimizer']):.1%}: a differential "
                     f"failure rate is a differential bias in the effect above, not noise "
                     f"around it")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    # Function-local for the same reason `build` is below: `hub.draft.cohort` imports
    # `market_strategy` from this module, so a module-level import here would be a cycle.
    from hub.draft.cohort import DRAFTS

    ap = argparse.ArgumentParser(
        prog="hub.draft.backtest",
        description="Championship equity against the market, on realised outcomes.")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--drafts", type=int, default=DRAFTS, help="drafts per season")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--draft-sims", type=int, default=12)
    ap.add_argument("--season-sims", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true",
                    help="one line per draft, so a long run can be watched rather than trusted")
    ap.add_argument("--ceiling", action="store_true",
                    help="also play a foresight arm and report what any board could deliver")
    ap.add_argument("--out", default=None, help="write the paired rows to this parquet path")
    ap.add_argument("--board", default=None,
                    help="parquet snapshot of the board. Written if absent, reused if "
                         "present, so two --diagnose runs at two commits compare the same "
                         "board rather than two live ADP fetches.")
    ap.add_argument("--diagnose-corrections", action="store_true",
                    help="which players corrected ADP moves and by how much, on the live "
                         "board, plus the pre-registered tripwire. Gates ADR-0011.")
    ap.add_argument("--diagnose", action="store_true",
                    help="what equity recommends at each of your first turns, on the live "
                         "board. Run before and after a change to the objective and diff.")
    a = ap.parse_args(argv)

    from hub.draft.board import build

    if a.diagnose_corrections:
        print("  building the live board ...")
        try:
            board, report = build()
        except Exception as e:
            return unavailable("hub.draft.backtest", "the live board", e)
        # Why the board has no Corrected ADP is the report's to answer, and it has to be
        # answered *here*, before anything below reads the frame. `correction_report` is
        # honest about its own frame -- no corrected column, no moves -- but zero moves
        # rendered by the lines below is "the corrections are clean", and on a board whose
        # market stage never ran that is a gate reporting a pass it did not run. Same shape
        # as issue #121 one level down: an absent stage arriving indistinguishable from a
        # stage that had nothing to say.
        if not report.adp:
            print("\n  no draft market reached this board, so it carries no ADP and no "
                  "Corrected ADP for ADR-0011 to gate.\n  Nothing moved because nothing was "
                  "computed -- which is not the same as nothing needing to move.")
            return 1
        rep = correction_report(board)
        print(f"\n  Corrected ADP moves {rep.height} of {board.height} players.")
        print(f"  Clamp: {DraftConfig().correction_clamp_frac:.0%} of each player's own ADP.\n")
        print(f"  {'player':<24} {'pos':<4} {'ADP':>6} {'->':>2} {'corrected':>9} "
              f"{'move':>7} {'ppg corr':>9}")
        for r in rep.head(20).iter_rows(named=True):
            print(f"  {str(r['player'])[:24]:<24} {r['pos'] or ''!s:<4} "
                  f"{r['adp']:>6.1f} {'->':>2} {r['adp_corrected']:>9.1f} "
                  f"{r['move']:>+7.1f} {r['proj_correction']:>+9.2f}")
        if rep.height > 20:
            print(f"  ... and {rep.height - 20} more")
        bad = correction_tripwire(board)
        print()
        if bad:
            print("  TRIPWIRE TRIPPED -- these are impossible if the code is right:")
            for line in bad:
                print(f"    {line}")
            return 1
        print("  tripwire clear: every move is a function of a real correction, "
              "and none exceeds the clamp.")
        if a.out:
            rep.write_parquet(a.out)
            print(f"  wrote {rep.height} rows to {a.out}")
        return 0

    if a.diagnose:
        # Pin the board. `--diagnose` is run twice at two commits to gate a change, and
        # `build()` refetches live ESPN ADP every time -- ADP moves, so two runs minutes
        # apart are not the same experiment. First run writes the snapshot, later runs
        # reuse it, so the only thing that differs between them is the code.
        snap = Path(a.board) if a.board else None
        if snap and snap.exists():
            board = pl.read_parquet(snap)
            # A pinned board is a board off disk, which is what `of_served` is for: the run
            # that wrote it is over and its columns are the only evidence of it there is. It
            # also keeps the pin intact -- the report is a function of the pinned board, so
            # two runs at two commits get the same one, which a second `build()` would not.
            report = BuildReport.of_served(board)
            print(f"  board pinned from {snap}")
        else:
            print("  building the live board ...")
            try:
                board, report = build()
            except Exception as e:
                return unavailable("hub.draft.backtest", "the live board", e)
            if snap:
                board.write_parquet(snap)
                print(f"  board snapshot written to {snap}")
        # Owned here rather than inside `diagnose`, so the count survives the call. Every
        # simulated season below writes into it; `note()` is said whether or not anything
        # failed, because a line that appears only on a bad run reads the same as no line.
        correlation = CorrelationReport()
        got = diagnose(board, report, rounds=a.rounds, n_draft_sims=a.draft_sims,
                       n_season_sims=a.season_sims, seed=a.seed, correlation=correlation)
        if got.is_empty():
            print("  no pick produced a rankable shortlist; nothing to compare.")
            return 1
        print(f"\n  Championship equity at your first {got.height} turns."
              f"  {a.draft_sims} x {a.season_sims} sims.")
        print("  The draft is advanced by the market, so the path is identical across runs\n"
              "  and the only thing that can differ is what equity says.\n")
        print(f"  {'pick':>4}  {'held':<16} {'leader':<24} {'pos':<4} "
              f"{'lift':>7}  {'co-led':>6}  {'cands':>5}")
        for r in got.iter_rows(named=True):
            print(f"  {r['pick']:>4}  {r['held']:<16} {str(r['leader'])[:24]:<24} "
                  f"{r['leader_pos'] or ''!s:<4} {r['lift']*100:>+6.2f}%  "
                  f"{r['co_leaders']:>6}  {r['candidates']:>5}")
        print(f"\n  {correlation.note()}")
        # The per-team repair record (#187). Printed beside the note rather than folded into
        # it: the note says how many blocks were repaired and this says how far each one
        # moved, which is the figure that decides whether a repair mattered.
        for line in correlation.repair_lines():
            print(line)
        bad = tripwire(board, got)
        print()
        if bad:
            print("  TRIPWIRE TRIPPED -- equity named a filled position over an empty one:")
            for line in bad:
                print(f"    {line}")
        else:
            print("  tripwire clear: no pick named a filled required position "
                  "over an unfilled one.")
        if a.out:
            got.write_parquet(a.out)
            print(f"  wrote {got.height} rows to {a.out}")
        return 0

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    try:
        boards, realised = walk_forward_inputs(
            seasons, board_as_of,
            on_season=lambda yr: print(f"  building the {yr} board as of {yr}-09-01 ..."))
    except Exception as e:
        return unavailable("hub.draft.backtest", "the boards these seasons are drafted from", e)

    print(f"  playing {a.drafts} drafts x {len(seasons)} seasons, "
          f"{a.draft_sims} x {a.season_sims} sims per optimizer call ...")
    def _tick(label: str):
        """One line per draft, flushed. A long run that says nothing is a run someone kills."""
        def say(season: int, k: int, of: int) -> None:
            print(f"    {label} {season}: draft {k}/{of}", flush=True)
        return say

    # Owned here for the same reason `diagnose`'s is: every simulated season inside `compare`
    # writes into it, and a count that lives inside the call dies with its stack frame.
    correlation = CorrelationReport()
    paired = compare(boards, realised, n_drafts=a.drafts, seed=a.seed, rounds=a.rounds,
                     n_draft_sims=a.draft_sims, n_season_sims=a.season_sims,
                     on_draft=_tick("paired") if a.progress else None,
                     correlation=correlation)
    print(f"\n  {correlation.note()}")
    for line in correlation.repair_lines():
        print(line)
    bound = None
    if a.ceiling:
        print("  measuring the ceiling: the same arm, given the season in advance ...")
        top = ceiling(boards, realised, n_drafts=a.drafts, seed=a.seed, rounds=a.rounds,
                      on_draft=_tick("ceiling") if a.progress else None)
        bound = Ceiling(CEILING_ARM, top["diff"])

    # `SEASON_CLUSTER`, not the row this gate used to take: the eighty (season, draft) rows
    # are twenty rooms drawn against four boards, and what varies independently between them
    # is the season. Issue #45; the effect is unmoved and the interval widens.
    #
    # Re-examined under #195, which removed a second and undeclared source of dependence
    # between the rows, and left unchanged: see `compare`'s docstring for why the surviving
    # reason is sufficient on its own. Stated here, at this gate's own call site, because the
    # run has no default for it (#135).
    # The join, reported on every run and voided above the floor (#46). Voiding is the run's;
    # what a join failure is, and what share of one this gate tolerates, is this gate's.
    rates = join_failure_rates(paired)
    run = run_gate(paired, cluster=SEASON_CLUSTER, actions=ACTIONS, name="draft",
                   arm_a="optimizer", arm_b="market", void=void_condition(rates),
                   ceiling=bound, seed=a.seed, boards=boards)
    for line in [*join_report(rates), *run.lines]:
        print(line)
    print(f"\n  {run.verdict[1]}")
    print("\n  Limitations, fixed before the run:")
    for line in LIMITATIONS:
        print(f"    - {line}")

    if a.out:
        run.stamped.write_parquet(a.out)
        print(f"\n  wrote {run.stamped.height} paired rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
