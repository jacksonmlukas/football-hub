"""Does a lineup set off the **Weekly projection** beat one set off weekly consensus rank?

This is **the** gate for `hub.models.weekly`.
[ADR-0015](../../../docs/adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md)
records why it has to be a decision and not an accuracy test: six seasons of historical
weekly FantasyPros consensus ship `ecr` and no `r2p_pts`, so the only incumbent worth beating
exists as a *ranking* and there is nothing to take a paired error against.

`hub.season.lineup_gate` asks a different question on the same harness. That one varies the
**search** over identical projections and returned a structural zero, because `sd = k*sqrt(mu)`
makes spread a deterministic function of the mean and the optimiser was handed no variance to
read (ADR-0012). This one varies the **projection** under an identical search, which is the
axis still open. Its own docstring names the gap this fills: *"projections are static across
the season, because weekly historical projections do not exist."* They do now.

THE ARMS, and they see the same information.

    consensus  fill each slot with your highest-ranked rostered player by `weekly-op` ECR
    weekly     fill each slot by `hub.models.weekly`'s projection for that week

The search is fixed at *start your highest* in both, per ADR-0012. Both are restricted to the
same roster and score against the same realised grid.

THE RULES, pre-registered in `docs/weekly-projection-plan.md` before this ran:

  * **Weeks 1-14**, the fantasy regular season. 15-17 reported apart and never pooled.
  * **Paired by roster-week**, with a **cluster bootstrap by roster** -- a roster's fourteen
    weeks are not fourteen observations, and quoting the raw n would inflate the interval's
    precision by roughly the square root of fourteen.
  * **A rostered player the model cannot price is not scored by either arm** -- #206, and the
    paragraph below. **This supersedes half of a pre-registered rule, and replaces it here
    rather than beside it**, because it is a rule the harness applies rather than a number it
    published: *"a rostered player missing from that week's consensus page is ranked last,
    because the absence is the incumbent saying 'do not start him' and it is real information
    it has."* A refusal to price is not a price. Reading it as one hands whichever arm refuses
    a free lineup rule -- bench him, for nothing -- which is the defect that disqualified
    `unscoreable` in the other direction. `coverage` still counts those cells and `VOID_FLOOR`
    still guards them; what changed is that they are no longer scored.
  * **Inactive weeks score zero**, not missing: starting a player who did not play is the most
    expensive weekly mistake there is and an honest lineup score has to eat it.

AND THERE IS NO FALLBACK IN THE RESULT, because there is no fallback. The arm's score column
is the model's own projection where there is one and something invented everywhere else, and
on 2026-09-07 that was 54.2% projection, 16.7% rank-interpolated, 29.1% unscoreable. The three
ways of treating the middle group were **1.5 points apart** -- further than any of the three
was from zero -- so the gate was measuring its own fallback (#207 measured all three; #206
disposed of them).

The disposition is not a fourth way of guessing. It is that both arms score **only the
roster-weeks both can price**, which is `priced_by_both` and is the `addable` mask this gate
already carried for exactly that idea. `method.md` rule 6 -- both arms must have the same
information -- then holds *by construction* rather than by two arms being trusted to apply a
fallback symmetrically, and the cells the three treatments disagreed in are the cells nothing
scores. A run prints the share of the slate that survives and scores the surviving rows under
all three treatments anyway, which is now a **check**: they can only differ where nothing is
scored, so a spread of anything but zero says the restriction is not holding.

**What it costs is stated in the output, not here alone.** The result speaks for the covered
share and for nothing else, and whether the weekly model beats consensus on the players it
cannot price is **unanswered** -- `priced_report` says so in those words on every run.

    uv run python -m hub.season.weekly_gate --run
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.config import FANTASY_WEEKS
from hub.league import STARTERS, starting_lineup
from hub.models.experiment import (
    SEASON_CLUSTER,
    Actions,
    Ceiling,
    Field,
    per_season,
    reading,
    run_gate,
    summarise,
)

NOT_FITTED_BECAUSE = (
    "Gate B for the Weekly projection. VOID_FLOOR is the share of roster-weeks lost to a join "
    "failure above which a run is not reported at all -- a pre-registered guard, not a fitted "
    "quantity. See docs/weekly-projection-plan.md "
)

# The fantasy regular season. 15-17 is the playoffs, reported apart; 18 is meaningless.
GATE_WEEKS = FANTASY_WEEKS

# What one independent observation is here -- `experiment.SEASON_CLUSTER`, declared once for
# every gate in the repo rather than three times in three harnesses.
#
# **This was `("season", "roster")` until #45, and the change is a widening.** The old comment
# was right about what it saw: a roster's fourteen weeks share its players, its bye and its
# draft, so resampling rows reports an interval about sqrt(14) too narrow -- signal-screens.md
# protocol item 3, the error that once produced an apparent 4-sigma result. The correction did
# not go far enough. Forty rosters within one season are drafted from one board, over one
# player pool, against one schedule, and score one realisation of that year; they are forty
# readings of a season, not forty independent observations, and the same argument that
# promoted the roster over the row promotes the season over the roster. There are four
# seasons, and the interval that says so is much wider than the one this gate published.
CLUSTER: tuple[str, ...] = SEASON_CLUSTER

# A player the consensus page does not list is ranked behind every player it does.
UNRANKED = -1e9

# Above this share of roster-weeks lost to a join failure, the run is VOID rather than
# reported. Pre-registered in docs/weekly-projection-plan.md.
#
# **The floor stands and its reason has changed, so the reason is restated rather than left.**
# It was tight because the error was *directional*: an unmatched player was ranked last, so a
# join failure did not add noise, it forced a bench on the incumbent's arm and biased the
# result toward us. Under #206 an unmatched player is not scored by either arm, so a join
# failure is no longer directional -- it is a hole in what the gate scores, and it lands in
# `priced_by_both` and is reported as coverage. That is a weaker fault and still a fault: a
# run that silently drops one roster-week in fifty to a name that did not match is answering
# for a different slate than the one it names, and the floor is what makes it say so.
VOID_FLOOR = 0.02


WAIVER_LOOK = 15

# This gate's own unit, named once so the effect and the ceiling printed under it cannot end
# up quoted in two different ones. Three places rather than two, because this gate's effect
# is a tenth the size of the draft gate's and rounds to +0.22 at two, while every doc and ADR
# quotes it as +0.215.
UNIT = "points per team-week"
PLACES = 3

# What this gate's ceiling *is*, spelled where it is printed. Full foresight is right here:
# what separates these two arms is the projection itself, so the largest effect any weekly
# projection could show is what a perfect one would. The lineup gate's arms already share a
# projection, so its ceiling is a perfect *spread* and the two bound different questions --
# and `docs/gate-power.md` stage 2 compares each gate's MDE against its own ceiling and never
# against another's, so the line says which one it is rather than leaving a reader to take
# three numbers in three units for one quantity.
CEILING_ARM = "perfect foresight"


class GateInputs(NamedTuple):
    """Everything one gate run reads, as one thing rather than nine.

    These were returned as a nine-value positional tuple, unpacked by name at the call site,
    and threaded on: `compare_universe` took **twelve parameters** to run thirty-six lines, and
    `coverage` took five of the same nine. The *ordering* was knowledge duplicated across the
    return, the unpack and two call sites, and checked nowhere -- swapping `consensus` and
    `weekly` is a silent inversion of the entire result, and nothing would have said so.

    Every array is over the season's **universe** -- every player on that season's board -- so
    a roster is a list of indices into it and can change week to week, which is what waiver
    churn needs and what a per-roster matrix cannot express.

    `projected` is the tenth, added by #207. `weekly` is a *mixture* -- the model's own number
    where it has one and a fallback everywhere else -- and until this mask travelled with it
    the gate could neither say how much of its own score column was which, nor re-score the
    same rows under a different treatment of the fallback. Both of those were done by hand
    against a document instead. It is a classification of cells and not a lineup rule: the
    arms' universes are set by `weekly` and `addable`, and this changes neither.
    """
    rosters: dict[int, list[list[int]]]         # season -> one index list per drafted roster
    pos: dict[int, Sequence[str]]               # season -> position per universe index
    realised: dict[int, np.ndarray]             # season -> (universe, weeks) points scored
    consensus: dict[int, np.ndarray]            # season -> (universe, weeks) -ecr, the incumbent
    weekly: dict[int, np.ndarray]               # season -> (universe, weeks) the arm under test
    pool: dict[int, list[list[int]]]            # season -> free agents, per drafted roster
    addable: dict[int, np.ndarray]              # season -> mask: both arms can score him
    se: dict[int, np.ndarray]                   # season -> standard error of the weekly mean
    covered: set[tuple[int, int]]               # the (season, week) pairs consensus ranks
    projected: dict[int, np.ndarray]            # season -> mask: `weekly` is the model's own


def lineup_projection(roster: Sequence[int], pos: Sequence[str],
                      score: np.ndarray) -> float:
    """What this roster's best legal lineup *projects* to score, by this arm's own numbers."""
    idx = starting_lineup([pos[i] for i in roster], score[list(roster)])
    return float(sum(score[roster[j]] for j in idx))


def fieldable(roster: Sequence[int], only: np.ndarray | None) -> list[int]:
    """The held players this arm may actually start this week, in roster order.

    One line, named once, because it is the whole of #206's restriction and it has to be the
    *same* line in the lineup and in the waiver decision. A swap chosen against a lineup the
    week will not field is a swap chosen against a projection the gate never scores.

    `only` is one week's column of `priced_by_both`, or `None` for the pre-#206 universe.
    """
    return list(roster) if only is None else [i for i in roster if only[i]]


def waiver_swap(roster: list[int], pool: list[int], pos: Sequence[str],
                score: np.ndarray, starters: int,
                *, look: int = WAIVER_LOOK,
                only: np.ndarray | None = None) -> tuple[int, int] | None:
    """One add/drop for a week, chosen by **what it does to the starting lineup**.

    The obvious rule -- add the highest-scoring free agent, drop the lowest-scoring bench
    player -- is wrong, and wrong in a way that only bites one arm. Absolute weekly points are
    much larger at quarterback than anywhere else, so it adds a backup quarterback every week:
    the first run of this experiment picked up Russell Wilson at a projected 24.4 (he scored
    5.1), Marcus Mariota at 16.5 (-2.1), Jayden Daniels at 18.5 (2.7), for a mean add of 19.6
    projected against 13.7 realised. You start one quarterback. A second is a roster spot spent
    on somebody who will never play.

    Consensus rank does not make that mistake, because a weekly *ranking* already prices
    positional scarcity -- so the naive rule handed the incumbent a free win that had nothing
    to do with either arm's forecasting. Scoring the swap by the projected starting lineup
    removes it: a backup quarterback cannot improve a lineup whose quarterback slot is already
    filled by someone better.

    Both arms run this identically over an identical pool. Only `score` differs.

    `only` restricts what the *lineup arithmetic* may count, not what may be held or dropped:
    a player nothing can price this week still occupies a roster spot and is still droppable,
    he simply cannot be the reason a swap looks like an improvement. Both arms are handed the
    same `only`, so this is which rows are scored and not a rule either arm gets to itself.
    """
    if not pool or len(roster) <= starters:
        return None
    held: dict[str, int] = {}
    for i in roster:
        held[pos[i]] = held.get(pos[i], 0) + 1
    droppable = [i for i in roster
                 if pos[i] not in STARTERS or held.get(pos[i], 0) > STARTERS[pos[i]]]
    if not droppable:
        return None

    base = lineup_projection(fieldable(roster, only), pos, score)
    best, gain = None, 0.0
    for add in sorted(pool, key=lambda i: score[i], reverse=True)[:look]:
        for drop in droppable:
            trial = [i for i in roster if i != drop] + [add]
            lift = lineup_projection(fieldable(trial, only), pos, score) - base
            if lift > gain:
                best, gain = (add, drop), lift
    return best


def season_points(realised: np.ndarray, pos: Sequence[str], score: np.ndarray,
                  roster: Sequence[int], pool: Sequence[int], weeks: Sequence[int],
                  *, churn: bool = False, addable: np.ndarray | None = None,
                  add_score: np.ndarray | None = None,
                  only: np.ndarray | None = None) -> dict[int, float]:
    """Points per week for one arm, optionally streaming a player a week.

    `realised`, `score` and `pos` are over the whole **universe** of board players, and a
    roster is a list of indices into it that evolves. With `churn=False` the roster never
    changes and this is the frozen-roster gate.

    `add_score` is what the **waiver decision** reads, where `score` is what the **lineup**
    reads. They differ when adds are ranked by a lower confidence bound: a waiver pick is the
    maximum over hundreds of candidates and so is biased upward, while a lineup is a choice
    among players you already hold and has no such selection. Defaults to `score`, which is
    the behaviour every earlier run had.

    `addable` masks the *pool* to the players both arms can score that week. Restricting it is
    what keeps this able to fail: consensus ranks only 35.8% of a 935-player free-agent pool,
    so an unmasked pool would let the arm under test add six hundred players the incumbent
    cannot score at all.

    `only` masks the **roster** to the same quantity, and until #206 this docstring said
    doing so "would bench a rostered player for being unranked, which is a different rule and
    the wrong one". It is the wrong rule when one arm gets it. Both arms are handed the same
    `only` by `compare`, so no arm benches anybody the other starts -- which is what makes
    the answer a property of the arm rather than of how the unpriceable were guessed at. A
    roster shorter than the slots is not an error here: `starting_lineup` fills what it can
    and an empty one scores zero, which is the honest reading of a week the gate cannot price.
    """
    need = sum(STARTERS.values())
    cur, free = list(roster), list(pool)
    decide = score if add_score is None else add_score
    out: dict[int, float] = {}
    for w in weeks:
        col = score[:, w - 1]
        if churn:
            eligible = ([i for i in free if addable[i, w - 1]]
                        if addable is not None else free)
            swap = waiver_swap(cur, eligible, pos, decide[:, w - 1], need,
                               only=None if only is None else only[:, w - 1])
            if swap is not None:
                add, drop = swap
                cur = [i for i in cur if i != drop] + [add]
                free = [i for i in free if i != add] + [drop]
        start_from = fieldable(cur, None if only is None else only[:, w - 1])
        idx = starting_lineup([pos[i] for i in start_from], col[start_from])
        chosen = [start_from[j] for j in idx]
        out[w] = float(realised[chosen, w - 1].sum()) if chosen else 0.0
    return out


def priced_by_both(g: GateInputs, season: int) -> np.ndarray:
    """The roster-weeks **both** arms can price: what this gate scores, since #206.

    One address for the restriction, read by the scoring and by the coverage line printed
    beside it -- so a run cannot score one set of cells and report the share of another. It is
    `addable`, unchanged and not re-derived: `(consensus > UNRANKED) & projected` is already
    "both arms can score him this week", which is what that mask was built to mean when it was
    only ever used on the free-agent pool. Deriving it a second time here from `projected` and
    `consensus` would be a second definition of one idea in two modules, which is how the two
    come to disagree.
    """
    return g.addable[season]


def compare(g: GateInputs, *, weeks: Sequence[int] = GATE_WEEKS, churn: bool = False,
            z: float = 0.0, mask_pool: bool = True, ceiling: bool = False,
            restrict: bool = True) -> pl.DataFrame:
    """One row per roster-week, with or without waiver churn.

    `churn=False` is the frozen gate. `z` ranks waiver adds by a lower confidence bound rather
    than by the mean; `mask_pool=False` opens the pool to players the incumbent cannot score,
    which is the sensitivity that is explicitly **not** the gate.

    `restrict` is #206 and defaults to on, which is the gate. Every arm -- both the two under
    comparison and the foresight arm, when one is scored -- is handed the *same*
    `priced_by_both` mask, so the restriction decides which rows are scored and is never a
    lineup rule one arm has and another lacks. `restrict=False` is the pre-#206 universe,
    where an unpriceable rostered player was scored on whatever the fallback had put in his
    cell. It is kept reachable because it is what every published figure through 2026-09-07
    was measured on and what `TREATMENTS` was measured across, and it is not the gate.

    It is deliberately independent of `mask_pool`: the pool mask decides who may be *added*
    and the restriction decides who may be *started*, and an open-pool run still starts nobody
    the other arm cannot price.
    """
    rows = []
    for season in sorted(g.rosters):
        wks = [w for w in weeks if (season, w) in g.covered]
        add = g.addable[season] if mask_pool else None
        only = priced_by_both(g, season) if restrict else None
        # Only the arm under test gets a lower confidence bound. Consensus is a *ranking* with
        # no uncertainty attached to subtract, so there is nothing to hand it -- and this is
        # our arm being more careful with its own estimate, not the incumbent being handicapped.
        lcb = (g.weekly[season] - z * g.se[season]) if z else None
        for k, roster in enumerate(g.rosters[season]):
            # The pool is per draft: who is a free agent depends on what the other eleven
            # teams took in that room.
            free = g.pool[season][k]
            a = season_points(g.realised[season], g.pos[season], g.consensus[season], roster,
                              free, wks, churn=churn, addable=add, only=only)
            b = season_points(g.realised[season], g.pos[season], g.weekly[season], roster,
                              free, wks, churn=churn, addable=add, add_score=lcb, only=only)
            # The ceiling: the same rule reading what actually happened. Full foresight is
            # right *here* -- unlike the lineup gate, where both arms already share a
            # projection -- because what separates these two arms is the projection itself,
            # so the largest effect any weekly projection could show is what a perfect one
            # would (#43). Its units are this gate's own points per roster-week and are never
            # compared against the draft backtest's.
            # Restricted with the other two: a foresight arm that could start players the
            # gate does not score would bound a different question from the one being asked,
            # and would bound it too high.
            c = (season_points(g.realised[season], g.pos[season], g.realised[season], roster,
                               free, wks, churn=churn, addable=add, only=only)
                 if ceiling else None)
            for w in wks:
                row = {"season": season, "roster": k, "week": w,
                       "consensus": a[w], "weekly": b[w]}
                if c is not None:
                    row["foresight"] = c[w]
                rows.append(row)
    out = pl.DataFrame(rows)
    if out.is_empty():
        return out
    out = out.with_columns((pl.col("weekly") - pl.col("consensus")).alias("diff"))
    if ceiling:
        out = out.with_columns(
            (pl.col("foresight") - pl.col("consensus")).alias("ceiling_diff"))
    return out


def declared_ceiling(paired: pl.DataFrame) -> Ceiling | None:
    """The foresight arm this run scored, named as this gate declares it -- or nothing.

    Nothing when the frame carries no ceiling, which is how a run without `--ceiling` prints
    exactly what it printed before -- and what a VOID run and an empty frame get, since
    neither carries the column. The arm's *name* travels with its rows so the one renderer in
    `experiment.run_gate` prints *perfect foresight* here and *a perfect spread* for the
    lineup gate (#135); until then this gate rendered its own line through a `ceiling_report`
    and handed `summarise` nothing, so the not-runnable rule could never read its ceiling.
    """
    if "ceiling_diff" not in paired.columns:
        return None
    return Ceiling(CEILING_ARM, paired["ceiling_diff"])


def void_condition(cover: dict[str, float] | None) -> str | None:
    """The one precondition only this gate has, phrased for `experiment.gate`'s `void`.

    This gate *voids* on a join failure, and **what a join failure does changed with #206
    while the floor did not**. It used to be directional: an unmatched player was ranked
    last, so the failure benched the incumbent's arm and biased the result toward us. He is
    now unpriceable by consensus and so is scored by neither arm, which makes the failure a
    hole in the covered share rather than a thumb on the scale. Either way it is not a verdict
    about the arm, it is a statement that there is no verdict to read at the slate the run
    names, and the floor it trips at is pre-registered in `docs/weekly-projection-plan.md`.
    """
    if cover is None or not cover["cells"] or cover["join_failure"] <= VOID_FLOOR:
        return None
    return (f"VOID: {cover['join_failure']:.1%} of roster-weeks are a join failure -- the "
            f"player was not on that week's consensus page and scored anyway -- against a "
            f"pre-registered floor of {VOID_FLOOR:.0%}.\n  Consensus cannot price him, so "
            f"since #206 neither arm scores him: this is that much of the slate silently "
            f"outside the result rather than a bias in it. Fix the join before reading any "
            f"number below.")


# The pre-registered actions, fixed in `docs/weekly-projection-plan.md` before this ran.
# Asymmetric on purpose: the Weekly projection is the complicated thing and the burden sits
# on it. The middle branch is the expected one and it carries an *action* rather than being a
# disappointment to explain away -- the same disposition the snap trend got in ADR-0013, and
# it was written down early precisely because "show it beside consensus" is a satisfying thing
# to decide after seeing a near-miss.
ACTIONS = Actions(
    adopt="ADOPT: the Weekly projection sets lineups.",
    remove="REMOVE: worse than a free ranking. Delete the module rather than shipping it as "
           "an option.",
    show="SHOW, NEVER RANK ON: printed beside consensus, never sorted on.")


def coverage(g: GateInputs, weeks: Sequence[int] = GATE_WEEKS) -> dict[str, float]:
    """How much of the incumbent's arm is missing, and how much of that is a defect.

    Two different things, and conflating them would either void every run or none:

      * **unranked** -- the player is not on that week's page. Usually because he is out, and
        the incumbent has nothing to say about him either way.
      * **join failure** -- unranked *and he scored*. Consensus would have ranked a player who
        played; the name did not match. This is the one `VOID_FLOOR` is measured against.

    Both are counted over every roster-week, including the ones #206 leaves unscored -- which
    is the point of counting them here rather than off `priced_by_both`. That mask says how
    much of the slate the result speaks for; this says how much of what it does not speak for
    is a property of the incumbent and how much is a defect in the join.
    """
    cells = unranked = failed = 0
    for season, made in g.rosters.items():
        for roster in made:
            for w in weeks:
                if (season, w) not in g.covered:
                    continue
                col = g.consensus[season][roster, w - 1]
                pts = g.realised[season][roster, w - 1]
                cells += col.size
                un = col == UNRANKED
                unranked += int(un.sum())
                failed += int((un & (pts > 0)).sum())
    if not cells:
        return {"cells": 0, "unranked": float("nan"), "join_failure": float("nan")}
    return {"cells": float(cells), "unranked": unranked / cells,
            "join_failure": failed / cells}


# --- the fallback that was removed, and the check that it stays removed ------
#
# The arm under test did not score in one currency. `weekly_gate_data._one_scale` builds its
# column from the model's own projection where there is one and a **fallback** everywhere
# else, and on 2026-09-07 that column was 54.2% projection, 16.7% fallback and 29.1%
# unscoreable. What the gate did with the middle group moved the answer by 1.5 points --
# further than any of the three answers was itself from zero -- and a run printed one of them.
#
# #207 made a run print all three, which is what established the spread. #206 then disposed of
# all three rather than choosing among them: the gate scores only `priced_by_both`, so no cell
# any treatment touches reaches the result. **The three below are therefore no longer three
# ways of getting an answer; they are the check that the answer no longer depends on them.**
# Scored on the restricted rows they can only agree, and `treatment_report` says so loudly
# when they do not.
#
# They are kept, rather than deleted with the fallback, for the reason the mixed scale was
# kept after #44 removed it: the numbers a document publishes have to stay reachable from the
# code that produced them. `docs/weekly-blend-gate.md` carries the 1.5-point spread as the
# record of *why* the fallback went, and `restrict=False` is how that record is re-derivable
# rather than transcribed.

# What `_one_scale` puts in the cells nothing scores. Not "primary" any more -- there is no
# primary treatment, because there is no treatment; this names which one the assembled column
# happens to carry, so `under_treatment` knows which one it need not rebuild.
COLUMN_TREATMENT = "rank-interpolated"


class Treatment(NamedTuple):
    """One way the gate used to score the players the model cannot price, and its standing.

    `role` is what this treatment *was*, and it is data rather than prose because the report
    has to be able to say which line each published figure came off. `why` is the argument
    that disqualified it, carried here so the treatments print with their reasons rather than
    as a bare table of three numbers a reader is left to rank by size.

    All three are past tense since #206. Nothing chooses among them, because the gate scores
    the rows none of them can reach.
    """
    name: str
    role: str
    why: str


TREATMENTS: tuple[Treatment, ...] = (
    Treatment(COLUMN_TREATMENT, "was primary",
              "read off nothing but the week's own paired observations, so neither arm got "
              "information the other lacked -- the only one of the three not disqualified on "
              "that ground, and still an estimator whose error the arm under test carried. "
              "The published -1.004 is this one. It is what the assembled column holds"),
    Treatment("unscoreable", "was a comparison",
              "benched every player the arm could not price while consensus still ranked and "
              "started him -- a free lineup rule the arm did not earn, which is the "
              "different-universes defect `_one_scale` names. Restricting the rows is that "
              "rule applied to both arms, which is why it is a smaller slate and not a treatment"),
    Treatment("mixed scale", "superseded by #44",
              "negated ranks and fantasy points in one column, so carrying a projection at "
              "all beat being ranked well; removed by #44 and kept reachable only because it "
              "is the treatment the published +0.215 was measured under"),
)


def under_treatment(g: GateInputs, name: str) -> GateInputs:
    """The same inputs with the fallback cells re-scored, and nothing else touched.

    Every treatment is a function of the two columns already assembled: `weekly` carries the
    model's projection wherever `projected` is true, so the cells the treatments differ in are
    exactly `~projected` and no re-assembly is needed to reach them. Which is why `projected`
    is carried rather than recomputed -- a second `np.isnan` here would be a second definition
    of what "the model could price him" means, in a different module from the one that
    decided it.

    **`addable` is deliberately not re-scored**, and since #206 that is load-bearing twice
    over. It is `(cons > UNRANKED) & projected`, so it already excluded every fallback cell
    from the waiver pool under all three treatments -- the pool is identical across them and
    the only thing that varied was how a *rostered* unprojected player was scored. It is now
    also `priced_by_both`, the rows the gate scores at all. Moving it with the treatment would
    move the scored rows with the treatment, and the check that the treatments agree would
    become three treatments agreeing about three different sets of rows.
    """
    if name == COLUMN_TREATMENT:
        return g
    if name not in {t.name for t in TREATMENTS}:
        raise ValueError(f"no such treatment: {name!r}")
    weekly = {}
    for season, col in g.weekly.items():
        priced = g.projected[season]
        # `unscoreable`: strip the interpolation, so a player this arm cannot price is one it
        # cannot start. `mixed scale`: the pre-#44 column, negated ECR in those same cells.
        other = UNRANKED if name == "unscoreable" else g.consensus[season]
        weekly[season] = np.where(priced, col, other)
    return g._replace(weekly=weekly)


def mixture(g: GateInputs, weeks: Sequence[int] = GATE_WEEKS) -> dict[str, float]:
    """What the arm's score column is made of, over the cells the gate actually reads.

    Three shares that sum to one, counted over roster-weeks on the weeks consensus covers --
    the same cells `coverage` walks, and for the same reason: what the lineup rule reads is a
    roster, so a share taken over the whole board would describe a column no arm ever sorts.

      * **projection** -- the model's own number.
      * **fallback** -- no projection, but ranked and in a week with something to calibrate
        against, so `_one_scale` interpolated one.
      * **unscoreable** -- no projection and nothing to interpolate from, so `UNRANKED`.

    Reported with the result rather than discovered by probing, which is what it took to learn
    the figures the published table carried.
    """
    cells = priced = fallback = unscoreable = 0
    for season, made in g.rosters.items():
        for roster in made:
            for w in weeks:
                if (season, w) not in g.covered:
                    continue
                on = g.projected[season][roster, w - 1]
                col = g.weekly[season][roster, w - 1]
                cells += on.size
                priced += int(on.sum())
                fallback += int((~on & (col > UNRANKED)).sum())
                unscoreable += int((~on & (col <= UNRANKED)).sum())
    if not cells:
        return {"cells": 0.0, "projection": float("nan"), "fallback": float("nan"),
                "unscoreable": float("nan")}
    return {"cells": float(cells), "projection": priced / cells,
            "fallback": fallback / cells, "unscoreable": unscoreable / cells}


def mixture_report(mix: dict[str, float], *,
                   fallback_name: str = COLUMN_TREATMENT) -> list[str]:
    """The mixture beside the effect. Lines, not prints, like every other block here.

    Nothing at all when nothing was counted, for the reason the ceiling line prints nothing
    without a ceiling: a percentage of no cells is not a small number, it is not a number.

    It describes the column **as assembled**, not as scored: since #206 only the first share
    is scored at all, which is what `priced_report` says on the next line. The other two are
    printed because the size of the group the gate declines to guess at is the cost of
    declining, and a cost nobody prints is a cost nobody weighs.
    """
    if not mix["cells"] or reading(mix, "projection") is not Field.VALUE:
        return []
    return ["\n" + _wrapped(
        f"the arm's score column as assembled, over {int(mix['cells'])} roster-week cells: "
        f"{mix['projection']:.1%} model projection, {mix['fallback']:.1%} {fallback_name}, "
        f"{mix['unscoreable']:.1%} unscoreable", "  ")]


def priced_share(g: GateInputs, weeks: Sequence[int] = GATE_WEEKS) -> dict[str, float]:
    """How much of the slate the result speaks for, and what the rest is out for.

    Counted over the same roster-weeks `coverage` and `mixture` walk, off the one mask
    `compare` scores on, so the share printed beside a number is the share that number was
    measured over. That equality is the whole of #206's fourth acceptance criterion: the
    figures the published table carried were obtained by probing the arm from outside and
    typing them into a document.

      * **scored** -- both arms can price him: the model has a projection and the consensus
        page has a rank.
      * **no projection** -- the model has none. The largest group, and the one whose three
        possible guesses were 1.5 points apart.
      * **no rank** -- the consensus page does not list him. Overlaps the above; both are
        shares of all cells rather than of the remainder, because a cell can fail both tests
        and there is no honest way to attribute it to one.

    And one thing that is not a share of cells: **forced**, the share of scored roster-weeks
    in which no lineup choice exists, because every player either arm can price that week
    starts. Restricting the rows is what creates this, and it is the failure mode the
    restriction has to be watched for -- two arms with no decision to make field the identical
    team, and the row's difference is zero by construction rather than by measurement. A gate
    that cannot disagree with itself is the defect ADR-0012 found in `lineup_gate` arriving by
    a different road, so it is counted and printed rather than left to be inferred from an
    effect suspiciously close to zero. It is read off `starting_lineup` rather than from a
    second copy of the slot arithmetic: how many of a roster start is not a function of their
    scores, so one call with a flat score answers it, and `hub.league` owns the rule.
    """
    cells = scored = unprojected = unranked = 0
    roster_weeks = forced = 0
    for season, made in g.rosters.items():
        for roster in made:
            for w in weeks:
                if (season, w) not in g.covered:
                    continue
                both = priced_by_both(g, season)[roster, w - 1]
                cells += both.size
                scored += int(both.sum())
                unprojected += int((~g.projected[season][roster, w - 1]).sum())
                unranked += int((g.consensus[season][roster, w - 1] <= UNRANKED).sum())
                can_start = [i for i, ok in zip(roster, both, strict=True) if ok]
                roster_weeks += 1
                forced += int(len(starting_lineup([g.pos[season][i] for i in can_start],
                                                  np.zeros(len(can_start)))) == len(can_start))
    if not cells:
        return {"cells": 0.0, "share": float("nan"), "no_projection": float("nan"),
                "no_rank": float("nan"), "roster_weeks": 0.0, "forced": float("nan")}
    return {"cells": float(cells), "share": scored / cells,
            "no_projection": unprojected / cells, "no_rank": unranked / cells,
            "roster_weeks": float(roster_weeks), "forced": forced / roster_weeks}


def priced_report(pop: dict[str, float]) -> list[str]:
    """The coverage beside the result, and the question it leaves open, in those words.

    Three things a reader must not have to derive: what the gate scored, what it did not, and
    that the answer for what it did not is **unanswered** rather than absent, negative or
    implied. #206's fifth acceptance criterion requires the last of those to be stated either
    way, and a result that speaks for half a slate while reading as though it speaks for all
    of it is the failure this whole ticket is about.

    Loud, in the register the ceiling check and the VOID line already use here, because it is
    the one line in the block that says what the verdict underneath it does *not* cover.

    **`forced` is printed with no threshold on purpose.** A share above which a run should be
    disbelieved would be a constant nobody fitted, arriving in a module whose whole subject is
    a modelling choice nobody justified. What is stated instead is the one reading that is a
    fact rather than a judgement: at 100% the arms never had a decision to make, so the effect
    is zero by construction and the run cannot fail. Between the two a reader has the number.
    """
    if not pop["cells"] or reading(pop, "share") is not Field.VALUE:
        return []
    forced = ([] if reading(pop, "forced") is not Field.VALUE else
              [_wrapped(f"no lineup choice in {pop['forced']:.1%} of "
                        f"{int(pop['roster_weeks'])} scored roster-weeks: every player either "
                        f"arm can price starts, so both field the same team and that row's "
                        f"difference is zero by construction. Restricting the rows is what "
                        f"creates this, and it is the cost to watch.", "  ")]
              + ([_wrapped(
                  "THE ARMS NEVER DISAGREE: no scored roster-week left either arm a lineup to "
                  "choose, so the effect below is zero by construction and this run could not "
                  "have failed. It is not a result about the weekly projection (#206).", "  ")]
                 if pop["forced"] >= 1.0 else []))
    return ["\n" + _wrapped(
        f"scored on the {pop['share']:.1%} of {int(pop['cells'])} roster-week cells both arms "
        f"can price -- the model projects him and the consensus page ranks him (#206). There "
        f"is no fallback in the result because there is no fallback: the cells the three "
        f"treatments below disagreed about are the cells nothing scores.", "  "),
        _wrapped(f"left out: {pop['no_projection']:.1%} the model cannot project, "
                 f"{pop['no_rank']:.1%} the consensus page does not rank (a cell can be "
                 f"both).", "  "),
        *forced,
        _wrapped(f"UNANSWERED: whether the weekly projection beats consensus on the "
                 f"{1.0 - pop['share']:.1%} it cannot price. Nothing below speaks for those "
                 f"roster-weeks, in either direction. What would answer it is a projection "
                 f"for those players -- a rule for scoring them without one is what #206 "
                 f"removed.", "  ")]


def _wrapped(text: str, indent: str, *, hang: bool = False, width: int = 92) -> str:
    """One paragraph, folded to a terminal. Hyphens are never a break point here.

    The treatments' reasons are sentences rather than fields, and an operator reading a
    180-column line in an 80-column terminal reads the first half of it. `break_on_hyphens`
    is off because the sentences carry `different-universes` and `rank-interpolated`, and a
    treatment's own name split across two lines is not greppable in a pasted run.

    `hang` indents the continuation, which is right for a list item and wrong for the
    paragraph above one: a prose line continuing at the depth of the rows underneath it reads
    as a row.
    """
    return textwrap.fill(text, width=width, initial_indent=indent, break_on_hyphens=False,
                         subsequent_indent=indent + ("  " if hang else ""))


class TreatmentEffect(NamedTuple):
    """One treatment, scored. The same summary shape every other block on this page reads."""
    treatment: Treatment
    summary: dict
    seasons: pl.DataFrame


def treatment_effects(g: GateInputs, *, weeks: Sequence[int] = GATE_WEEKS,
                      churn: bool = False, z: float = 0.0, mask_pool: bool = True,
                      seed: int = 0, primary: pl.DataFrame | None = None,
                      restrict: bool = True,
                      treatments: Sequence[Treatment] = TREATMENTS) -> list[TreatmentEffect]:
    """Score the identical rows under each treatment of the fallback.

    Identical in every other respect on purpose: same weeks, same churn rule, same pool mask,
    same seed, same `restrict`, and `summarise` under the same `CLUSTER` #45 fixed. A spread
    measured across three runs that also differed in their seed would be a spread across
    seeds, which is the reading this block exists to rule out -- #206 records seed 0 at -1.004
    and seed 7 at -1.188, so 0.18 points of seed noise against the 1.5 the fallback was worth.

    **Under `restrict=True` this is a check and not a measurement.** The treatments differ
    only in cells `priced_by_both` excludes, so all three frames are the same frame and the
    three means are one mean. They are still *scored* rather than asserted equal, because the
    claim being checked is about the code and not about the arithmetic: it is that the
    restriction reaches every arm, every week and the waiver rule too. `restrict=False` is
    where they can genuinely differ, and is what the published spread was measured on.

    `primary` is the paired frame the caller already built for the verdict, handed back in so
    a run scores three arms rather than four. It is only ever reused for `COLUMN_TREATMENT` --
    the frame the caller built is the assembled column's -- and handing in a frame from some
    other run would put a row in this table that the verdict was not read off, so the caller
    passes the frame it printed or nothing at all.
    """
    out = []
    for t in treatments:
        paired = (primary if t.name == COLUMN_TREATMENT and primary is not None
                  else compare(under_treatment(g, t.name), weeks=weeks, churn=churn, z=z,
                               mask_pool=mask_pool, restrict=restrict))
        out.append(TreatmentEffect(t, summarise(paired, cluster=CLUSTER, seed=seed),
                                   per_season(paired)))
    return out


def treatment_report(effects: Sequence[TreatmentEffect], *, unit: str = UNIT,
                     places: int = PLACES, restricted: bool = True) -> list[str]:
    """Every treatment's effect side by side, and how far apart they are.

    **The spread is computed, and so is its comparison with the effects.** The claim that
    matters -- for #207, that the choice of fallback moved the answer further than any of the
    answers was itself from zero; for #206, that it no longer moves it at all -- is exactly
    the kind a document restates until it is stale, so the line reads `larger` or `smaller`
    off the numbers in front of it rather than asserting what held on 2026-09-07.

    `restricted` says which of those two the block is. Under the gate's own scored rows the
    treatments reach no cell that is scored, so **a spread of anything but zero is a defect
    and says so loudly** -- it means the restriction missed an arm, a week or the waiver rule,
    and the number above it is once again partly the fallback. Under `restricted=False` the
    spread is the #207 measurement and is a finding rather than an alarm.

    Nothing at all under two scored treatments: one number side by side with itself is the
    single-treatment report this replaces, and a spread of zero printed across it would be
    worse than printing nothing. A treatment whose frame was empty carries no mean and is
    listed as such rather than dropped, because a missing row would read as one nobody ran.
    """
    scored = [e for e in effects if reading(e.summary, "mean") is Field.VALUE]
    if len(scored) < 2:
        return []
    label = {e.treatment.name: f"{e.treatment.name} ({e.treatment.role})" for e in effects}
    pad = max(len(v) for v in label.values())
    lead = ("the same scored rows under each treatment of the fallback #206 removed, same "
            "seed and same cluster. They can only differ in cells this run does not score, "
            "so this is a check on the restriction and not a choice between them:"
            if restricted else
            "the same rows under each treatment of the fallback, same seed and same cluster "
            "-- the pre-#206 universe, where the choice was still live:")
    lines = ["\n" + _wrapped(lead, "  ")]
    for e in effects:
        name = label[e.treatment.name]
        if reading(e.summary, "mean") is not Field.VALUE:
            lines.append(f"    {name:<{pad}}  nothing scored")
            continue
        lines.append(f"    {name:<{pad}}  {e.summary['mean']:+.{places}f}  "
                     f"95% CI [{e.summary['lo']:+.{places}f}, "
                     f"{e.summary['hi']:+.{places}f}]")
    means = [e.summary["mean"] for e in scored]
    spread, biggest = max(means) - min(means), max(abs(m) for m in means)
    lines.append(_wrapped(
        f"spread across treatments {spread:.{places}f} {unit} -- "
        f"{'larger' if spread > biggest else 'smaller'} than any effect any of them reports "
        f"({biggest:.{places}f}). How much of the verdict is the fallback.", "  "))
    if restricted and spread:
        lines.append(_wrapped(
            f"THE FALLBACK STILL REACHES THE RESULT: the treatments are scored on the rows "
            f"both arms can price, where they touch nothing, and they disagree by "
            f"{spread:.{places}f}. Something is scoring a cell `priced_by_both` excludes; the "
            f"effect above is not a property of the arm until it is found (#206).", "  "))
    lines.extend(_wrapped(f"{e.treatment.name}: {e.treatment.why}", "    ", hang=True)
                 for e in effects)
    return lines


def main(argv: Sequence[str] | None = None) -> int:      # pragma: no cover - network
    from hub.draft.cohort import DRAFTS

    ap = argparse.ArgumentParser(
        prog="hub.season.weekly_gate",
        description="Does the Weekly projection beat weekly consensus rank at setting lineups?")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--churn", action="store_true",
                    help="one waiver add/drop a week, both arms, from a pool both can score")
    ap.add_argument("--open-pool", action="store_true",
                    help=argparse.SUPPRESS)   # the asymmetric pool: NOT the gate
    # The pre-#206 universe, where a rostered player nothing could price was scored on
    # whatever the fallback had left in his cell. Hidden beside `--open-pool` and for the same
    # reason: it is how the published figures are re-derivable, and it is NOT the gate. A run
    # under it prints the #207 spread rather than the check, and says which it printed.
    ap.add_argument("--unrestricted", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--lcb", type=float, default=0.0, metavar="Z",
                    help="rank waiver adds by mu - Z*se; the pre-registered value is 1.0")
    ap.add_argument("--expected", action="store_true",
                    help="expected receptions and yardage in the priors, not realised")
    ap.add_argument("--shrink",
                    choices=("mae", "tail", "mae-market", "tail-market", "market-only"),
                    default=None,
                    help="shrink thin-sample projections toward the positional mean; "
                         "'mae' is the pre-registered fit, 'tail' the exploratory one")
    ap.add_argument("--ceiling", action="store_true",
                    help="also score a foresight arm -- the same lineup rule reading what "
                         "actually happened -- and report the largest effect any weekly "
                         "projection could show. `docs/gate-power.md` stage 2")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--drafts", type=int, default=DRAFTS, help="rosters per season")
    ap.add_argument("--seed", type=int, default=0)
    # The stamped paired rows, the same four stamps the other two gates write. This gate
    # wrote nothing until #135; `docs/gate-power.md` names a frozen paired frame as what
    # stage 2 needs for the two network-built gates, and this is how one is made.
    ap.add_argument("--out", default=None, help="write the stamped paired rows to this parquet")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    from hub.season.weekly_gate_data import assemble_universe
    try:
        inputs = assemble_universe(seasons, drafts=a.drafts, seed=a.seed, shrink=a.shrink,
                                   expected=a.expected)
    except Exception as e:
        return unavailable("hub.season.weekly_gate", "the gate's inputs", e)
    cover = coverage(inputs)
    mix = mixture(inputs)
    pop = priced_share(inputs)
    restrict = not a.unrestricted
    paired = compare(inputs, churn=a.churn, z=a.lcb, mask_pool=not a.open_pool,
                     ceiling=a.ceiling, restrict=restrict)
    # `SEASON_CLUSTER`, stated at this gate's own call site: the run has no default for it.
    run = run_gate(paired, cluster=SEASON_CLUSTER, actions=ACTIONS, name="weekly",
                   arm_a="weekly", arm_b="consensus", unit=UNIT, places=PLACES, show_n=False,
                   void=void_condition(cover), ceiling=declared_ceiling(paired), seed=a.seed)
    s, seasons_tbl = run.summary, run.seasons
    # The assembled column's frame is handed back rather than rebuilt, so this is two extra
    # scorings and not three, and the row it fills is the one the verdict is read off. No
    # `--ceiling` on the other two: the ceiling is a property of the harness rather than of
    # the fallback, and scoring a foresight arm three times would say the same thing thrice.
    effects = treatment_effects(inputs, churn=a.churn, z=a.lcb, mask_pool=not a.open_pool,
                                seed=a.seed, primary=paired, restrict=restrict)
    mode = ("one add/drop a week, pool both arms can score" if a.churn and not a.open_pool
            else "one add/drop a week, OPEN POOL -- not the gate" if a.churn
            else "frozen rosters")
    if not restrict:
        mode += ", UNRESTRICTED -- the pre-#206 universe, not the gate"
    if a.shrink:
        mode += f", shrink={a.shrink}"
    if a.expected:
        mode += ", expected priors"
    if a.lcb:
        mode += f", waiver LCB z={a.lcb}"
    print(f"\n  {int(s['n'])} roster-weeks over {int(s['clusters'])} seasons, "
          f"on the {len(inputs.covered)} weeks consensus covers   [{mode}]")
    # The arguments this run was given, printed by the run rather than recalled by whoever
    # pastes it. A gate once run with bare defaults was nearly reported as a re-run of a
    # published figure, caught only because the season count did not match the table (#237);
    # `docs/weekly-blend-gate.md` records what the published figures were measured under.
    print(f"  configuration: seasons={a.seasons} drafts={a.drafts} seed={a.seed} "
          f"shrink={a.shrink} churn={a.churn} lcb={a.lcb} expected={a.expected} "
          f"ceiling={a.ceiling} restricted={restrict} mask_pool={not a.open_pool}")
    print(f"  unranked {cover['unranked']:.1%}, of which a join failure "
          f"{cover['join_failure']:.1%} (floor {VOID_FLOOR:.0%})")
    print(seasons_tbl)
    print("\n".join([*run.lines,
                      *mixture_report(mix),
                      *(priced_report(pop) if restrict else []),
                      *treatment_report(effects, restricted=restrict)]))
    print(f"\n  {run.verdict[1]}")
    if a.out:
        run.stamped.write_parquet(a.out)
        print(f"\n  wrote {run.stamped.height} paired rows to {a.out}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
