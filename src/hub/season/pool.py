"""The pool as a whole, not one entry in it.

`hub.season.survivor` answers "which teams should I pick". This answers "how long does the
contest last, and how many people are still in it" -- which is a different question and the
one a dollar figure needs. A buyback is worth what the pot will be times the chance of taking
it, and both of those depend on the field, not on us.

**Every game is drawn once per trial.** All entries holding that team share the result. This
is the whole point: a field playing chalk dies *together*, and the correlation is what decides
whether the pool reaches week 13 at all. Drawing per entry would make eliminations independent,
which would thin the field smoothly and stretch the contest far past anything real.

**And no entry is handed both sides of one fixture**, ours or a rival's. One of them loses, so
a double-pick week spending them together is lost by construction and can eliminate the entry
outright -- which is a loss the simulator inflicted rather than one the pool did. The game
identity was read for the draw and never for the pick until #156; `_Week.fixture` is where it
is now stated, and `_pick` and `_Ours.plays` are the two places that read it.

**Rivals are sampled, not deterministic**, and that is a correction rather than a refinement.
Twenty-one rivals following one deterministic rule against identical empty ledgers pick the
identical team every week and die in the same week: the surviving count is 21 until it is 0,
the ending week is a point mass, and the partially-thinned field a buyback is priced against
never occurs. So each rival samples among the teams absent from its own ledger, weighted by win
probability -- concentrated on chalk, but not identical to it.

**How hard they crowd is a stated number, not a by-product of that rule.** Weighting by raw
`win_prob` is one point on an axis, and it is the flat end of it: it puts roughly a sixteenth
of the field on the week's best team, where real survivor fields concentrate several times
that. Since concentration is exactly what makes a field die together, the sampling rule was
quietly deciding whether the pool reaches week 13 -- the question this module exists to answer.
`PoolConfig.field_concentration` is the exponent `_pick` raises `win_prob` to, and 1.0 is the
old behaviour reproduced rather than a measurement.

What that buys is divergence. What it costs is a modelling choice with no measurement behind
it: nobody has observed this pool's rivals, and under Hidden Picks nobody can before the
deadline. The weighting is an assumption and is stated as one -- so nothing here reports a
figure at one concentration and calls it the answer. `sensitivity` runs the axis and reports
the range, which is what ADR-0024 asks of a parameter whose alternatives can be laid side by
side but not chosen between.

**Our own entry plays our plan, not the field's rule.** It is index 0 of the same arrays, and
for a while that was the whole of it: there was no branch for `i == 0`, so the entry every
dollar figure here is about picked a near-chalk random team each week under the rival rule
above. That answers "what is an entry worth to somebody who does not use this repo", the error
runs one way, and it grows with the horizon -- which is why it decided the buyback verdict
from a modelling choice rather than from the pool's economics. Index 0 now replays a plan
solved *once*, outside the trial loop: `hub.season.survivor.solve` on the drawable grid
carrying our ledger, or a stated best-available fallback when that cannot run. Rivals are
untouched. The sampling rule is theirs and only theirs, and the paragraph defending it above
is about them.

**Our plan is an input, not a by-product**, and that is a shape rather than an optimisation.
Re-solving an integer program inside the trial loop would multiply it by trials times weeks to
answer the same question every time -- what our entry plays does not depend on a trial's
outcomes until one of its teams is unavailable -- but the reason it is a parameter is that two
candidate plans have to be able to meet the *same* season. That is the pairing #159 needs, and
a plan computed inside the loop could not be handed to it.

**Ledgers are reconstructed, not observed**, under the same sampling rule -- so the model is
exactly degenerate in week 1, where every ledger is empty and it knows nothing about the field
at all. That is where the season starts, so early figures carry more model risk than late ones.

**A pool ends two ways, and both are priced.** Somebody outlasts the final week, and the pot
is divided by `PoolConfig.co_survivor_rule`; or the last entries standing all go out in the
same week, and it is divided by `PoolConfig.co_elimination_rule` among the entries eliminated
last. The second is the ending a field playing chalk usually reaches -- it dies *together*,
which is the correlation the first paragraph is about -- and until #157 it was the ending
this module measured as `PoolOutcome.ending_week` and then valued our stake in at zero. An
entry that outlives the whole field and then loses is that state with a count of one, and
takes the pot: this simulator keeps a lone survivor playing, and the pool would not have.
`trial_share` is the one place either rule meets a trial.

**Ties are not modelled** because the grid cannot express one: `win_prob` comes from a
continuous margin model, which prices a tie at zero. The pool's rule that a tie eliminates is
therefore satisfied vacuously here rather than enforced.

Iteration order is sorted everywhere a set would otherwise decide it. `docs/weekly-blend-gate.md`
records an interval that moved between identical runs because `.unique()` order fed a bootstrap;
same class of bug, and a reproducible simulator cannot afford it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.config import PoolConfig
from hub.season.survivor import MIN_PROB, Infeasible, solve, week_fixtures

NOT_FITTED_BECAUSE = (
    "nothing here is measured. DEFAULT_TRIALS and WEEKLY_TRIALS buy resolution and are "
    "traded against runtime; DEFAULT_CONCENTRATIONS is the axis `sensitivity` sweeps and not "
    "a value any figure is computed at -- the concentration a run actually uses is "
    "PoolConfig.field_concentration, which is a stated assumption covered by `pool_digest`. "
    "DECISIVE_SIGMA is the evidential bar the rest of the repo's gates are stated at, which "
    "is a convention rather than a quantity anybody measured. Moving any of the four changes "
    "how finely, over what range, or at what confidence this module reports; none of them "
    "changes a prediction, and none has a measurement behind it to move. "
)

# Enough that the ending-week distribution is stable to about a percentage point, which is
# finer than any decision downstream reads it at. Callers wanting a tighter tail pass more.
DEFAULT_TRIALS = 2000

# The axis `sensitivity` sweeps when the caller names none. Not candidate values to be chosen
# between, and none of them is favoured: 1.0 is the behaviour already in the tree, and it is
# the anchor rather than the centre.
#
# **Geometric, and it runs to 16 rather than to 4, because the lever is weaker than it looks.**
# The exponent is applied to every team's price and then renormalised, so raising it lifts the
# other fifteen favourites on a full slate almost as much as the chalk team -- the response in
# ownership is far flatter than the response in weight. Measured on the 32-team board in
# `tests/unit/test_pool.py`, the best team's share of a clean-ledger field goes 5.7% at 1.0,
# 8.0% at 2.0, 11.6% at 4.0, 17.3% at 8.0, 27.0% at 16.0. The several-times-a-sixteenth
# concentration real survivor fields show therefore lives near 12-16 on this axis and nowhere
# near 2, so an axis stopping at 4 would have swept only the region where the answer does not
# move and reported that the knob does not matter. A caller with a view passes its own.
DEFAULT_CONCENTRATIONS = (1.0, 2.0, 4.0, 8.0, 16.0)

# `weekly` runs one simulation per candidate rather than one per call, so its real cost is
# `top` times this -- 2400 trials at the default six, which is the same work as a single
# DEFAULT_TRIALS run. The resolution it publishes falls as 1/sqrt(trials), so a caller who
# needs a finer verdict than a week returns can raise this and watch the figure move.
WEEKLY_TRIALS = 400

# How many standard errors of the paired difference a week has to clear before `weekly` calls
# its recommendation distinguishable from the free pick. It was one, which is a two-sided
# false positive rate of about one in three -- so roughly a third of the weeks where the two
# picks are genuinely alike would have been reported as a departure worth taking, in a repo
# whose gates elsewhere ask for a bootstrap interval excluding zero and a consistent sign
# across held-out seasons. Two is the bar those gates are stated at; a stricter one would
# start refusing weeks the trials really can separate, and the alternative when a week is
# refused is free rather than costly, so the error is cheap in one direction and not the other.
DECISIVE_SIGMA = 2.0


class PoolOutcome(NamedTuple):
    """What one simulation says about the contest, as distributions rather than points."""
    trials: int
    ending_week: dict[int, float]      # week -> P(the last live entry dies in it)
    co_survivors: dict[int, float]     # n -> P(exactly n entries outlast the final week)
    alive_by_week: dict[int, float]    # week -> mean entries still live after it


class Plan(NamedTuple):
    """What our own entry picks, week by week, before any trial is run.

    Solved once and replayed, which is what makes it a *parameter* of the simulation rather
    than something the simulation produces. Two candidate plans can therefore be scored on the
    same trials and differ only by the picks -- `docs/method.md` rule 7, and the pairing #159
    is for. A plan built inside the trial loop could not be handed to anything.

    `source` says which rule produced it, because the two are not equally good and the gap
    between them is the model's own uncertainty rather than a detail of the run. It is prose
    and is meant to be printed: a figure computed off a best-available fallback is a weaker
    claim than the same figure off the optimiser, and nothing else in `EntryOutcome` would say
    so.

    `picks` is missing a week wherever the rule that built it ran out of teams. That is not a
    gap to be filled with a repeat -- it is the no-repeat ledger binding, and the entry is
    eliminated there for want of a legal pick, exactly as `_pick` returning None eliminates a
    rival.
    """
    picks: dict[int, tuple[str, ...]]   # week -> the teams taken in it, sorted
    source: str


class EntryOutcome(NamedTuple):
    """What one entry's own position is worth, from here.

    `sole` is the ticket's question and `share` is the one a dollar figure asks: with a pot
    split among co-survivors, finishing level with two others is worth a third, not nothing.
    Both come off the same trials because a second pass over this simulator is not cheap, and
    computing them apart would let them disagree about the same run.

    `plan` is what our entry actually played, carried back out so a figure can be read beside
    the picks that produced it -- and so a caller passing one in can see it was the one used.
    `replans` counts, per trial, the weeks where it would not play and had to be solved again;
    a figure with a high `replans` was mostly not a figure about the picks it names.

    `survivors_each` is the trial-by-trial record the three means above are means *of*,
    carried out rather than reduced here because a difference between two of these outcomes
    has to be taken **paired** to be worth anything. Two candidate plans on the same seed meet
    the same season trial for trial (`_play`), so the spread of `a - b` is far tighter than
    the spread of either arm -- and computing it from `share_sd` alone would throw that away
    and report an interval several times too wide.

    It is a *count* and not a share, which is what makes it the record rather than one reading
    of it. Every figure derives: the entry survived where it is non-zero, was alone where it
    is one, and what the trial paid depends on `PoolConfig.co_survivor_rule` -- which is
    unconfirmed, and under `rollover` pays nothing at all for a two-way finish. A vector of
    shares could not answer "did we survive" under that rule, so it would have needed a second
    vector beside it and the two could disagree about one set of trials.

    **`last_out_each` is the other way a trial pays, and until #157 it paid nothing.** A
    survivor pool ends one of two ways: somebody outlasts the final week, or the last entries
    standing all go out in the same week. The second is not a tail case -- a field crowding
    onto chalk dies *together*, and on the synthetic board `sensitivity` was first run on the
    field is gone before the last week in 99% of trials at the default concentration. This
    module measured that event as `PoolOutcome.ending_week` and then valued our stake in it
    at zero, which is not a rule any pool has: the entries eliminated last split the pot, or
    it rolls over, and `PoolConfig.co_elimination_rule` says which. The record here is how
    many entries went into the week the field emptied, ours among them, and 0 where the
    trial did not end that way or ours had already gone. `trial_share` reads the two records
    together, and it is the one place either rule is applied to a trial.

    An entry that outlives the whole field and then loses is in this state alone: the count
    is one and it takes the pot under every spelling of the rule. That is what the pool
    would have paid it the week the last rival went out, so `survives` and `sole` -- which
    are about the *final week* and unchanged -- can both be zero on a trial that paid in
    full. `share` is the money and the two above it are the season.
    """
    trials: int
    survives: float     # P(this entry outlasts the final week at all)
    sole: float         # P(it is the only one that does)
    share: float        # expected fraction of the pot under `PoolConfig.co_survivor_rule`
                        # and `PoolConfig.co_elimination_rule`, whichever the trial reached
    share_sd: float = 0.0   # spread of that fraction across trials, so a caller can say
                            # how finely two of these figures can be told apart
    plan: Plan | None = None    # the picks our entry played; None when it sampled as a rival
    replans: float = 0.0        # weeks per trial where that plan would not play
    survivors_each: tuple[int, ...] = ()    # entries left at the end of each trial, ours
                                            # included; 0 where ours was not one of them
    last_out: float = 0.0       # P(ours went out in the week the field emptied -- it was
                                # among the last standing, or was the last)
    last_out_each: tuple[int, ...] = ()     # entries that went into that week, ours
                                            # included; 0 where the trial did not end so

    @property
    def shared(self) -> float:
        """P(this entry survives and somebody else does too).

        The trials `PoolConfig.co_survivor_rule` decides, which is the one pool rule nobody
        has confirmed and the reason it is worth a name of its own. How often it is reachable
        is not a property of the pool -- it is a property of the field-concentration
        assumption, and `sensitivity` is where the two are reported against each other.
        """
        return self.survives - self.sole

    @property
    def co_eliminated(self) -> float:
        """P(this entry went out in the week the field emptied, and was not alone in it).

        The trials `PoolConfig.co_elimination_rule` decides, the counterpart of `shared` for
        the other way a pool ends. Read off the record rather than off two means, because
        the "alone" case is one it must exclude: an entry that outlasted everybody and then
        lost takes the pot under every spelling, so the rule decides nothing there.
        """
        if not self.last_out_each:
            return 0.0
        return sum(m > 1 for m in self.last_out_each) / len(self.last_out_each)


def _plural(n: int, word: str) -> str:
    """`1 team`, `2 teams`. A count printed beside a decision is read by a person."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class Buyback(NamedTuple):
    """A $20 decision, priced. Every field is reported rather than folded into a verdict.

    `net` is the answer and its sign is the recommendation: positive means the equity bought
    exceeds the fee paid. `breakeven` is the fee at which that flips, so the margin is visible
    instead of implied -- a net of +$0.40 and a net of +$40 are the same verdict and very
    different bets.

    **`breakeven` is a fixed point, not the equity** (#158). The pot is a function of the fee
    -- our re-entry and every rival's pay it in -- so the fee at which `net` is zero is the
    `f` solving `share * (pot + (1 + rivals) * f) = f`, which is `share * pot` over
    `1 - share * (1 + rivals)`. The equity is that answer holding the pot fixed at the
    configured fee, and it is low by exactly the pot's own growth: a tenth or so on the
    inputs this pool has. When `share * (1 + rivals)` reaches one, every dollar of fee adds
    at least a dollar of pot to our side and no fee flips the verdict; that is `inf`, and
    `report` says so in words rather than printing it.

    `ahead` is the weeks the re-entry was priced over: those after `week`, since an entry
    eliminated in a week re-enters for the next one. Carried so the figure is read beside
    its horizon, and so a caller can see it was not the whole season.
    """
    available: bool
    reason: str
    pot: float          # what the pot becomes once the expected buybacks are in
    field: int          # live entries once they are, ours included
    spent: int          # teams the re-entry inherits, which is why it is not a fresh entry
    share: float        # our expected fraction of that pot
    equity: float       # share * pot, in dollars
    fee: float
    net: float          # equity - fee. Positive means buy back
    breakeven: float    # the fee at which net is zero once the pot has grown by that fee
    recommend: bool
    ahead: tuple[int, ...] = ()     # the weeks priced: those after the elimination week
    rivals: int = 0                 # rival re-entries priced, exactly as the caller stated


def _ahead(weeks: Sequence[int], week: int) -> list[int]:
    """The weeks still to be played after `week`, in the order given.

    One definition for the two paths that need it. `weekly` prices a candidate on the weeks
    after the one being picked, and `buyback` prices a re-entry on the weeks after the one
    it was eliminated in; the second used to take the caller's list whole, so a caller
    handing it the season while eliminating in week 3 had the re-entry replay weeks 1 to 3
    against a ledger that had already spent them (#158). The two now cannot disagree about
    what "ahead" means.
    """
    return [w for w in weeks if w > week]


class Candidate(NamedTuple):
    """One legal pick for a week, and what taking it is worth.

    The two standard errors ride alongside the two simulated figures rather than being
    recoverable from them, because both are needed to print the figures honestly: a run that
    reports `survives 30.4%` off 400 trials is claiming a tenth of a point it cannot see, and
    `weekly_report` reads these to decide how many digits it is entitled to. They are the
    error on *this candidate alone*; the difference between two candidates is paired and
    tighter, which is `Weekly.resolution`.

    Zero means "no Monte Carlo error", not "unknown". A week with nothing ahead of it is
    priced in closed form and its figures are exact.
    """
    team: str
    win_prob: float
    survives: float             # P(the season is survived, having taken this now)
    expected_dollars: float     # net of what the entry has already paid in
    is_fallback: bool           # the team auto-pick would assign for nothing
    survives_se: float = 0.0        # standard error of `survives` at the trial count used
    dollars_se: float = 0.0         # standard error of `expected_dollars`, likewise


class Weekly(NamedTuple):
    """A week's recommendation, and the free alternative it has to beat.

    `given_up` is `fallback.survives - recommend.survives`, and it is the number
    `hub.season.journal` refuses a departure without: ADR-0014's logging duty is the week, the
    chalk pick, ours, and the probability cost accepted. `journal.record_weekly` is what
    carries this shape over there, and it recomputes the difference from the candidate
    actually entered rather than forwarding this field -- which is a cost against the
    *recommendation* and is the wrong comparison for an overridden pick.

    **It is not the quantity ADR-0014's threshold is stated in.** The rule fires on a
    win-probability cost of under ~8pp; this is a season-survival difference, and the two
    routinely disagree in sign, because buying survival later with a lower win probability now
    is the entire thesis of a survivor plan. Named in `journal.record_weekly`, which is where
    the column is written and where the choice between them has to be made.

    It is **signed on purpose**. A departure that survives better than the free pick has given
    up nothing and gained something, and clamping that to zero would file it as "the two plans
    survive alike" -- which is a different claim, and a false one. Zero is reserved for the case
    where they really do survive alike.

    **Which comparison the interval is for, said once.** `resolution` and `given_up_se` are
    both about *the recommended plan against the auto-pick plan*: two candidate plans, both
    replayed by our own entry, against one season drawn once and shared between them. They are
    **not** intervals on our arm against the field's sampling rule. That comparison cannot be
    paired at all -- replaying a plan consumes no draws where sampling consumes one a week, so
    the two arms walk the generator at different rates and their difference carries the
    variance of two independent means (#151, and the docstring of
    `test_our_entry_survives_materially_more_often_than_the_field_rule_gave_it` says so from
    the test side). Two plans against each other is the comparison this simulator can make
    honestly, and it is the one every figure here reports.
    """
    week: int
    recommend: str
    fallback: str | None       # None when auto-pick had no team left to assign
    matched: bool
    given_up: float
    pot: float
    resolution: float           # the dollar difference this many trials can actually resolve
    candidates: list[Candidate]
    given_up_se: float = 0.0    # standard error of `given_up`, paired across the same trials
    trials: int = 0             # trials each candidate was run for; 0 when none were needed
    cashed: int = 0             # of those, the ones where either arm took a share of the pot

    @property
    def unresolved(self) -> bool:
        """Whether these trials carry any information about the comparison at all.

        The degenerate regime, reported rather than folded into a tie. With our share often on
        the order of a thousandth, a few hundred trials can return zero for *every* candidate:
        both arms then have the identical figure, the spread of their difference is zero, and
        a comparison at any bar reads "not decisive". That lands the week on the auto-pick for
        a structural reason rather than because the auto-pick is right, and it is a different
        statement from two plans that were measured and found alike -- which is what
        `given_up == 0.0` off a set that did cash means.

        `docs/method.md` rule 12: a gate that cannot run is an absence of evidence, not
        evidence against. A week priced in closed form ran no trials and is not this -- there
        is no Monte Carlo error for it to be about.
        """
        return self.trials > 0 and self.cashed == 0

    @property
    def decisive(self) -> bool:
        """Whether the recommendation is distinguishable from the free pick at all.

        False does not mean the pick is wrong; it means the simulation cannot tell the two
        apart, and `docs/method.md` rule 12 is that a gate which cannot run is an absence of
        evidence rather than evidence against
        rather than refuted. A week that lands here should take the free pick, because it is
        free -- and should say that is why.

        The bar is `resolution`, and since #159 that is `DECISIVE_SIGMA` standard errors of
        the **paired** difference rather than one standard error of an unpaired one. One
        standard error is a roughly one-in-three two-sided false positive rate, in a repo
        whose gates ask for a bootstrap interval excluding zero and a positive sign in every
        held-out season; two is the bar the rest of the repo calls a finding.
        """
        if self.matched:
            return True
        if self.unresolved:
            return False
        fb = next((c for c in self.candidates if c.is_fallback), None)
        return fb is None or abs(self.candidates[0].expected_dollars
                                 - fb.expected_dollars) > self.resolution


class UnpricedWeek(ValueError):
    """A week the grid cannot price, which is a different answer from a week nobody survived.

    The two are indistinguishable downstream if this is not raised: a week with no usable
    fixture leaves every entry unable to field a pick, the field is eliminated at once, and
    `PoolOutcome.ending_week` reports the contest ending in a week that was never covered.
    A missing grid is not a result, so it is refused here rather than counted there.
    """


class _Week(NamedTuple):
    """One week's games and prices, in a shape the trial loop can use without re-querying.

    `games`, `teams` and `prob` all come off the same set of fixtures. They were not: the
    first two counted a fixture only when both sides were priced while `prob` was built from
    every row in the week, so a half-priced fixture put a team in `prob` that no draw could
    ever produce a result for.

    `teams` is every team a game here can return a result for; `pickable` is the subset an
    entry is allowed to *take*, which is `hub.season.survivor.MIN_PROB` applied here as it is
    applied in `auto_pick` and `weekly`. They are not the same set and the difference is the
    point: a hopeless side still has to be drawn, because its opponent's win depends on it,
    and it still must never be handed to an entry as a pick.

    `weight` is `prob` raised to `PoolConfig.field_concentration`, and it is the *rivals'*
    sampling weight and nothing else. `prob` is what a game is drawn at and what the optimiser
    and `_best_available` rank on; the two must not be confused, which is why the exponent is
    applied here, once per week per run, rather than inside `_pick` where a reader would have
    to check it had not leaked into a price. Only the teams in `pickable` have one, because
    they are the only teams the sampler can reach.

    `fixture` is which game each team is in, and it is carried rather than rebuilt because
    `_pick` needs it -- once per live entry per week per trial, not once per week per run.
    It was rebuilt on demand while the only readers were `_greedy` and `_Ours`, and the whole
    of #156 is that the *sampler* was not one of them: `games` is the only field on this week
    that still knows two rows are two sides of one fixture, and `teams`, `prob`, `pickable`
    and `weight` have each flattened that away. A rule stated on a shape the code doing the
    work cannot see is a rule nothing enforces.
    """
    games: tuple[tuple[str, str, float], ...]   # (team_a, team_b, P(team_a wins))
    teams: tuple[str, ...]                      # sorted, so iteration order is not a set's
    prob: dict[str, float]
    pickable: frozenset[str]                    # teams above MIN_PROB: what a pick may take
    weight: dict[str, float]                    # rival sampling weight; see above
    fixture: dict[str, int]                     # team -> which game in `games` it plays in
    picks: int                                  # 1, or 2 in a double-pick week
    dropped: int = 0                            # fixtures in the week priced on one side only


def weeks_from_grid(grid: pl.DataFrame, weeks: Sequence[int],
                    pool: PoolConfig | None = None) -> list[_Week]:
    """Reshape the win-probability grid into one entry per week.

    Needs `game_id`: a week that takes two picks must know which rows are two sides of one
    fixture, both so the game is drawn once and so nobody is handed both sides. Raises rather
    than falling back to per-team draws, which would let a team and its opponent both win.

    **Both halves of that sentence are now wired to something.** The first was: `games` pairs
    the rows and `_play` draws each pair once. The second was prose -- the identity was used
    for the draw and never for the pick, where the only exclusion applied was the entry's own
    ledger, so `_pick` handed a rival both sides of one game in a double-pick week and the
    entry lost by construction (#156). `_Week.fixture` is that identity in the shape the
    sampler can read, built here beside `games` so the two cannot drift.

    A fixture priced on one side only is dropped, and the count of what was dropped rides on
    the week rather than being discarded -- a week built from half of what was asked for is
    still an answer, but not the one the caller asked for, and the difference has to be
    readable. A week with fewer completely priced fixtures than it takes picks is refused
    outright: it is the case `hub.season.survivor.coverage` was hardened against by counting
    fixtures rather than rows, and it arrives here as a whole field eliminated in a week that
    was simply not covered.

    **The count is against the picks the week takes, not against zero.** A double-pick week
    with one priced fixture has two teams and no way to cover itself -- `_pick` hands the
    entry both sides of that one game, one of them loses, and every entry in the field dies
    with certainty in a week `survivor.solve` calls infeasible and `coverage` calls missing.
    That reads out as `ending_week` naming a week nobody could have played, which is exactly
    what refusing a wholly unpriced week was for; zero was where the guard stopped rather than
    where the reasoning did.

    **What is usable is `hub.season.survivor.week_fixtures`**, read here and by `coverage`,
    which is the one place the rule is stated. This side uses its *drawable* half -- both
    sides priced -- because a game with one row cannot be played out. `MIN_PROB` is not part
    of that test and is applied to `_Week.pickable` instead: a hopeless side is undrawable
    only in the sense that nobody may pick it, and dropping its fixture would refuse to
    simulate a week the board has fully priced.

    **`PoolConfig.field_concentration` is applied here and only here.** Raising `prob` to it
    once per week per run costs nothing next to doing it inside `_pick`, which runs once per
    live entry per week per trial -- but the reason it lives here is that this is the seam
    where a price stops being a price. `_Week.weight` is a sampling weight for rivals;
    `_Week.prob` stays the drawn probability, and every other reader of the week --
    `_best_available`, the optimiser's grid, the draw in `_play` -- goes on reading `prob`.
    """
    if "game_id" not in grid.columns:
        raise ValueError(
            "the grid must carry `game_id`; without it a game cannot be drawn once and a "
            "team and its opponent can both win in the same trial")
    cfg = pool or PoolConfig()
    k = float(cfg.field_concentration)
    # Refused rather than clamped, and negative is the case worth naming: it does not make the
    # field less concentrated, it inverts the knob into a field preferring the *worst* team
    # available, which is not a point on this axis and would read out of a sweep as one. Zero
    # is allowed and is the axis's floor -- a field picking uniformly among what it may still
    # take. A NaN would reach `rng.choice` as a probability vector several frames from here.
    if not np.isfinite(k) or k < 0.0:
        raise ValueError(
            f"`PoolConfig.field_concentration` is {cfg.field_concentration!r}: it is an "
            "exponent on `win_prob` and has to be finite and at or above zero. 1.0 samples "
            "the field proportional to win probability, 0.0 samples it uniformly among the "
            "teams it may still take, and above 1.0 crowds it onto the week's favourites. A "
            "negative value inverts the rule rather than relaxing it.")
    out = []
    for f in week_fixtures(grid, weeks, cfg):
        wk = grid.filter(pl.col("week") == f.week)
        games = []
        for gid in f.drawable:
            side = wk.filter(pl.col("game_id") == gid).sort("team")
            a, b = side["team"][0], side["team"][1]
            games.append((str(a), str(b), float(side["win_prob"][0])))
        if len(games) < f.needs:
            raise UnpricedWeek(
                (f"week {f.week} has no completely priced fixture: " if not games else
                 f"week {f.week} takes {_plural(f.needs, 'pick')} and has only "
                 f"{_plural(len(games), 'completely priced fixture')} to take them from: ")
                + f"{_plural(len(games) + len(f.half), 'fixture')} in the grid, "
                f"{_plural(len(f.half), 'fixture')} priced on one side only. Refused rather "
                "than simulated -- with no legal pick to field, every entry is eliminated at "
                "once and the ending week would report the contest ending in a week the grid "
                "never covered. Two picks cannot come from both sides of one fixture, because "
                "one of them loses, which is the same refusal `survivor.solve` already makes.")
        teams = tuple(sorted({t for g in games for t in (g[0], g[1])}))
        # Restricted to the fixtures `games` and `teams` were built from, and each side keeps
        # the win probability its own row carried rather than one minus its opponent's: the
        # grid prices both sides and they need not sum to exactly one.
        priced = set(teams)
        prob = {str(r["team"]): float(r["win_prob"]) for r in wk.iter_rows(named=True)
                if str(r["team"]) in priced}
        pickable = frozenset(t for t in teams if prob[t] > MIN_PROB)
        out.append(_Week(tuple(games), teams, prob, pickable,
                         {t: prob[t] ** k for t in sorted(pickable)},
                         {t: i for i, g in enumerate(games) for t in (g[0], g[1])},
                         f.needs, len(f.half)))
    return out


def _pick(rng: np.random.Generator, week: _Week, ledger: set[str], k: int) -> list[str] | None:
    """`k` teams this entry has not used, one per fixture, sampled toward the best available.

    None when the entry cannot field a legal pick -- it has spent too many teams to cover the
    week, which is elimination by the no-repeat rule rather than by losing.

    **One per fixture, which is #156 and was the whole of the defect.** Both sides of one game
    cannot both win, so a double-pick week spending them together is lost the moment it is
    entered -- and it can eliminate the entry outright. `hub.season.survivor.solve` forbids it
    as `one_side_wk` and `_best_available` takes one team per fixture, so our own entry was
    never exposed; this sampler, which is what prices the *money*, drew `k` teams out of a flat
    list and the only exclusion it applied was the ledger. The two counts that matter are
    therefore different counts: `k` distinct *teams* were always available where `k` distinct
    *fixtures* were not, and the second is the one a week can actually be covered from.

    The draw is sequential rather than one `size=k` call, because the legal set narrows after
    each pick: taking a team removes its opponent as well as itself. At `k == 1` that is the
    identical `rng.choice` call on the identical list, so nothing about a single-pick week --
    which is every week outside 13 through 18 -- moves by a float.

    **This is the whole of the field's sampling rule, and since #151 it is theirs alone**: our
    own entry replays a plan and reaches this function through no path. So how hard the field
    crowds onto the week's best team is a property of this one draw, and
    `PoolConfig.field_concentration` is where it is stated -- carried in as `_Week.weight`,
    which is `prob` raised to it. At 1.0 the weight *is* the probability and this samples
    exactly as it always has; above 1.0 the field concentrates, which is what makes it die
    together. Nobody has observed this pool's rivals and under Hidden Picks nobody can before
    a deadline, so the knob is stated rather than fitted and `sensitivity` reports what the
    answer does across it instead of asserting a value.

    **Drawn from `pickable`, not from `teams`.** A team below `MIN_PROB` is one `auto_pick`
    and `weekly` both refuse to hand us, and it was still reachable here -- so our own entry
    was valued over seasons in which it made picks the pick side would never have allowed.
    Its sampling weight was near zero either way, which is why the figures barely move; what
    moves is that one rule now says what may be taken, in both places that take one.

    That also retires a guard. This returned None a second time when the weights summed to
    zero -- a whole week of teams the betting market gives no chance -- which was the only way a
    zero-priced team could reach the sampler at all. Every member of `pickable` is above
    `MIN_PROB` by construction, so a non-empty `avail` cannot sum to zero and the branch was
    unreachable rather than merely untaken. Deleted, because a guard that cannot fire still
    reads as a case someone has thought about. That argument survives the exponent: a positive
    weight raised to a finite non-negative power is positive, and `weeks_from_grid` refuses
    every other power.
    """
    avail = [t for t in week.teams if t in week.pickable and t not in ledger]
    # Fixtures, not teams. A double-pick week whose only legal teams are the two sides of one
    # game cannot be covered at all, and saying so here is what makes that elimination the
    # no-repeat rule's rather than a loss the entry was handed.
    if len({week.fixture[t] for t in avail}) < k:
        return None
    out: list[str] = []
    while len(out) < k:
        w = np.array([week.weight[t] for t in avail], dtype=float)
        got = str(rng.choice(avail, size=1, replace=False, p=w / w.sum())[0])
        out.append(got)
        avail = [t for t in avail if week.fixture[t] != week.fixture[got]]
    return out


def _chalk_share(week: _Week) -> tuple[str, float]:
    """The week's best team, and the share of a clean-ledger field that draws it.

    The quantity the concentration knob is *about*, so it is reported rather than left to be
    inferred from a lifetime. Computed and not counted: with every ledger empty, `_pick`'s
    normalised weight on a team **is** its expected ownership, so estimating it by simulation
    would put Monte Carlo noise on a number already known exactly.

    Exact only while the ledgers are empty, which is the first week. The module's docstring
    already names week 1 as the week this model knows least about; this is the same fact from
    the other side -- after it, rivals have diverged and what they own has to be simulated.

    It is the share of one *draw*. A double-pick week takes two, so a rival's chance of
    holding the best team there is higher than this figure; in this pool the double weeks are
    13 through 18 and a sweep reported from week 1 is not one of them.

    Ties on `win_prob` break on the team name, which is `auto_pick`'s tiebreak and
    `_best_available`'s -- so "the best team" means the same team in all three.
    """
    best = min(sorted(week.pickable), key=lambda t: (-week.prob[t], t))
    total = sum(week.weight[t] for t in sorted(week.pickable))
    return best, week.weight[best] / total


def _best_available(week: _Week, ledger: set[str], k: int) -> tuple[str, ...] | None:
    """The `k` likeliest teams this entry may still take, one per fixture. None if it cannot.

    The stated fallback, and it is deliberately the greedy rule `hub.season.survivor`'s own
    docstring calls reliably wrong: take the biggest favourite, which spends the team you
    wanted in week 12. It is here because a figure produced by a rule nobody can name is worse
    than a figure produced by a bad rule that is named -- and because `CLAUDE.md`'s degradation
    rule says a module with no solver available must still answer.

    Sorted on the name after the probability, so a tie is not broken by fixture order: this is
    `auto_pick`'s tiebreak, applied where `auto_pick` cannot reach.
    """
    out: list[str] = []
    taken: set[int] = set()
    for t in sorted((t for t in week.teams if t in week.pickable and t not in ledger),
                    key=lambda t: (-week.prob[t], t)):
        if week.fixture[t] in taken:
            continue
        out.append(t)
        taken.add(week.fixture[t])
        if len(out) == k:
            return tuple(sorted(out))
    return None


def _pickable_grid(wks: Sequence[_Week], weeks: Sequence[int]) -> pl.DataFrame:
    """The weeks as a grid the optimiser can be handed, holding only what a trial can play.

    Not `grid` itself, and the difference is the drawable/pickable split
    `hub.season.survivor.week_fixtures` exists to name. `solve` reading the raw grid would
    plan a week from a fixture priced on one side only -- legal to *pick* and impossible to
    *draw*, so `weeks_from_grid` dropped it -- and hand back a team no trial could ever return
    a result for. Our plan would then be invalidated in every trial and re-solved into the
    same answer. Restricting the solver to `pickable` makes that unreachable instead of merely
    handled.

    `game_id` is synthesised per week from the fixture's position, because the fixture pairing
    is all `solve` reads it for and that constraint never spans two weeks.
    """
    rows = [(w, t, week.prob[t], f"{w}-{i}")
            for week, w in zip(wks, weeks, strict=True)
            for i, g in enumerate(week.games)
            for t in (g[0], g[1]) if t in week.pickable]
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows], "game_id": [r[3] for r in rows]},
                        schema={"week": pl.Int64, "team": pl.Utf8,
                                "win_prob": pl.Float64, "game_id": pl.Utf8})


def _greedy(wks: Sequence[_Week], weeks: Sequence[int], ledger: set[str], why: str) -> Plan:
    """A whole plan from `_best_available`, week by week, carrying what it spends."""
    led = set(ledger)
    out: dict[int, tuple[str, ...]] = {}
    for week, w in zip(wks, weeks, strict=True):
        got = _best_available(week, led, week.picks)
        if got is None:
            break       # nothing legal left: the entry is eliminated there, not given a repeat
        out[w] = got
        led.update(got)
    return Plan(out, f"best available week by week, because {why}")


def _solved(wks: Sequence[_Week], weeks: Sequence[int], ledger: set[str],
            cfg: PoolConfig) -> Plan:
    """`hub.season.survivor.solve` over these weeks with our ledger, or the stated fallback.

    Both outcomes are plans and the caller is told which by `Plan.source`, because they are
    not equally good: the optimiser maximises the probability of surviving every week at once,
    and the fallback is the greedy rule that optimiser exists to beat. The gap between them is
    the model's own uncertainty about what our entry is worth, and it is reported rather than
    smoothed over.

    Two failures, kept apart because they mean different things. `Infeasible` is an answer
    about the weeks and the ledger -- no assignment covers them without a repeat -- and is
    reachable from a ledger deep enough that no season remains. Anything else is the solver
    itself failing, which under `CLAUDE.md`'s degradation rule must still produce a usable
    figure rather than take the buyback decision down with it.
    """
    frame = _pickable_grid(wks, weeks)
    try:
        picks = solve(frame, weeks=list(weeks), spent=sorted(ledger), pool=cfg)
    except Infeasible as e:
        return _greedy(wks, weeks, ledger, f"no assignment covers these weeks: {e}")
    except Exception as e:      # broad on purpose: the solver is optional, see the docstring
        return _greedy(wks, weeks, ledger, f"the optimiser could not run: {e}")
    out: dict[int, list[str]] = {}
    for r in picks.iter_rows(named=True):
        out.setdefault(int(r["week"]), []).append(str(r["team"]))
    return Plan({w: tuple(sorted(t)) for w, t in out.items()},
                "the optimiser on the weeks ahead, carrying our ledger")


def solve_plan(grid: pl.DataFrame, weeks: Sequence[int], *, ledger: Sequence[str] = (),
               pool: PoolConfig | None = None) -> Plan:
    """What our entry would pick in each of these weeks, before any trial is run.

    The seam. `entry_outcome` calls this when it is handed no plan, and a caller that wants to
    compare two of them -- #159's pairing -- builds them here and passes each in, so the only
    difference between the two figures is the picks and not the season they met.

    Solved against the same weeks the simulator will play, which is why it takes a grid and
    reshapes it rather than taking `hub.season.survivor.plan_remaining`'s answer: a plan over
    weeks this simulator cannot draw is not a plan this simulator can score.
    """
    cfg = pool or PoolConfig()
    return _solved(weeks_from_grid(grid, weeks, cfg), weeks, set(ledger), cfg)


class _Ours:
    """Our entry's side of a trial: replay a plan, and re-solve only where it will not play.

    The invalidation rule, which is the whole of what makes replaying safe. A week's picks are
    taken from our plan unless one of them is **already spent** in this trial's ledger or is
    **not available to take** in the week as drawn -- below `MIN_PROB`, absent from the week
    entirely, or paired against our plan's other pick in one fixture. Then, and only then, the
    remaining weeks are solved again from the ledger the trial actually has.

    Nothing else can invalidate it, and that is why solving once is not an approximation: our
    entry's picks do not depend on which games were won, only on which teams are left, and
    what is left changes only when we ourselves have spent something. A trial that our entry
    is still alive in has spent exactly the picks it made.

    Re-solves are memoised on `(week, ledger)`. They are deterministic in both, so a
    trajectory that invalidates in one trial invalidates identically in all of them, and the
    integer program runs once for the run rather than once per trial. `replans` counts the
    invalidations rather than the solves, because the reader's question is how much of the
    figure was about the picks they were shown.

    A class and not a `NamedTuple`, unlike everything else named in this module: the memo and
    the tally are state that accumulates across trials, and a tuple that has to carry a
    one-element list to be written to is a value object pretending.
    """

    def __init__(self, wks: Sequence[_Week], weeks: Sequence[int], cfg: PoolConfig,
                 plan: Plan | None, ledger: set[str]) -> None:
        self.wks = tuple(wks)
        self.weeks = tuple(weeks)
        self.cfg = cfg
        self.plan = plan if plan is not None else _solved(wks, weeks, ledger, cfg)
        self.memo: dict[tuple[int, frozenset[str]], tuple[str, ...] | None] = {}
        self.replans = 0

    def plays(self, at: int, teams: Sequence[str], ledger: set[str]) -> bool:
        """Whether these picks are legal in this week for an entry holding this ledger.

        The three conditions are the same three `_pick` now applies to a rival, which is the
        point of #156: our entry has always been held to one team per fixture -- by `solve`'s
        `one_side_wk`, by `_best_available`, and by this line -- and the field was not. One
        rule, checked on both sides of the branch in `_play`.
        """
        week = self.wks[at]
        if len(teams) != week.picks:
            return False
        if any(t in ledger or t not in week.pickable for t in teams):
            return False
        return len({week.fixture[t] for t in teams}) == len(teams)

    def picks(self, at: int, ledger: set[str]) -> list[str] | None:
        """This week's picks: our plan's, or a fresh solve where our plan will not play."""
        want = self.plan.picks.get(self.weeks[at])
        if want is not None and self.plays(at, want, ledger):
            return list(want)
        self.replans += 1
        key = (self.weeks[at], frozenset(ledger))
        if key not in self.memo:
            self.memo[key] = self._again(at, ledger)
        got = self.memo[key]
        return list(got) if got is not None else None

    def _again(self, at: int, ledger: set[str]) -> tuple[str, ...] | None:
        """Solve the rest of the season from here, and take this week out of it.

        The whole remainder rather than this week alone, for the reason `solve` exists at all:
        the team that covers this week cheapest may be the one week 15 has no substitute for.
        Only this week's picks are kept -- the next week checks our plan again,
        and invalidates again if that is still wrong, which costs one memoised solve per week
        rather than replacing a plan the caller passed in and can still read back.
        """
        rest, wks = self.weeks[at:], self.wks[at:]
        got = _solved(wks, rest, ledger, self.cfg).picks.get(self.weeks[at])
        return got if got is not None and self.plays(at, got, ledger) else None


def _play(rng: np.random.Generator, wks: Sequence[_Week], weeks: Sequence[int],
          led: list[set[str]], alive: list[bool],
          ours: _Ours | None = None) -> tuple[int | None, list[int], int | None]:
    """Play one trial out. Returns the week everyone died -- None if somebody lasted -- the
    live count after each week, and the position in `weeks` at which entry 0 went out, or
    None if it lasted.

    The third is what `_entry_trials` needs to price the week the field empties (#157): once
    every entry is dead the `alive` list cannot say whether ours went out in that week or
    three weeks earlier, and the two are worth the whole pot and nothing respectively.
    `simulate` gets it too and reads it for nothing, because entry 0 is a rival there.

    Every game is drawn once here and the result shared by each entry holding either side.
    Extracted rather than written twice: `simulate` and `entry_outcome` ask different
    questions of the same trial, and a second copy of this loop is a second answer about it.

    `ours` is the one branch on entry index in this module, and it is the difference between
    the two questions. Given, entry 0 is *our* entry and replays our plan;
    left out, entry 0 is another member of the field and samples like one -- which is what
    `simulate` wants, because a field statistic has no us in it. Every other index samples
    either way, so a change to how we pick cannot reach a rival.

    **The whole season is drawn before anybody picks, and that is what makes two candidates
    comparable.** It reads like an ordering detail and is the pairing #159 is about. Drawn
    week by week inside the loop, the number of games this trial consumed from the generator
    depended on how far the trial got -- and the trial stops when the *last* entry dies, ours
    included. So the moment our entry outlived the field in one candidate and not in the
    other, that trial consumed a different number of draws, the stream offset, and every
    later trial met a different season. Pairing survived only up to the first week our fate
    differed, which is exactly the trials carrying the signal.

    Drawn up front, the count is `sum(len(wk.games))` whatever happens afterwards. Rivals then
    consume from the same stream, but only ever as a function of the results and their own
    ledgers -- our entry replays a plan and consumes nothing (#151) -- so two candidate plans
    handed the same seed meet the identical field, game for game and rival pick for rival
    pick. `weekly` relies on that and `test_two_candidates_meet_the_identical_field` is what
    holds it.
    """
    counts = []
    out_at: int | None = None
    # One draw per game per trial, and the same result read by every entry holding either
    # side: the correlation this module exists for, unchanged. What is new is *when*.
    results = [frozenset(a if rng.random() < p_a else b for a, b, p_a in wk.games)
               for wk in wks]
    for at, (wk, w) in enumerate(zip(wks, weeks, strict=True)):
        won = results[at]
        for i in range(len(alive)):
            if not alive[i]:
                continue
            picks = (ours.picks(at, led[i]) if ours is not None and i == 0
                     else _pick(rng, wk, led[i], wk.picks))
            if picks is None or not all(t in won for t in picks):
                alive[i] = False
                if i == 0:
                    out_at = at
                continue
            led[i].update(picks)
        counts.append(sum(alive))
        if counts[-1] == 0:
            return w, counts, out_at
    return None, counts, out_at


def share_of_pot(rule: str, survivors: int) -> float:
    """What finishing level with `survivors - 1` others is worth, as a fraction of the pot.

    `PoolConfig.co_survivor_rule` applied, and until #160 it was applied nowhere: `share` was
    `1 / n` and the rule was named only in the prose beside it. A rule stated in a docstring
    and absent from the arithmetic is not a setting, it is a comment -- and correcting it
    would have been a code change rather than the re-run `PoolConfig` exists to make it.

    **split** divides the pot evenly, which is what every dollar figure in this module assumed.

    **rollover** pays nobody when more than one entry is standing; the pot carries forward, so
    only an outright win collects. That is not a smaller figure by a constant -- it is worth
    nothing exactly where a split is worth a half, which is why the rule being unconfirmed
    matters more the more often a shared finish happens.

    **tiebreak** hands the whole pot to one survivor by a rule nobody has stated. With no rule
    to model it is priced at its expectation under an unbiased draw, which is `1 / n` and so
    identical to a split *in the mean*. It is not identical in spread: the realised payout is
    all or nothing, so `EntryOutcome.share_sd` understates it and any interval built from that
    understates it too. Named rather than smoothed over, because a figure right in the mean
    and wrong in the tail is exactly what a decision at the margin gets taken on. Simulating
    the draw instead was rejected: it would consume from the generator conditionally on our
    own survival, which is precisely what #159 removed to make two candidates comparable.

    An unrecognised rule raises. It is a closed set of three and a fourth spelling is a
    configuration nobody has modelled, which must not silently fall through to the default --
    that is how `co_survivor_rule` came to be unread in the first place.
    """
    if rule in ("split", "tiebreak"):
        return 1.0 / survivors
    if rule == "rollover":
        return 1.0 if survivors == 1 else 0.0
    raise ValueError(
        f"a pool rule is {rule!r}: `PoolConfig.co_survivor_rule` decides what a shared "
        "finish pays and `PoolConfig.co_elimination_rule` what the week the field empties "
        "pays, and each has to be one of 'split' (an even division), 'rollover' (nobody "
        "collects, the pot carries forward) or 'tiebreak' (one entry takes it, priced at its "
        "expectation). Both rules are unconfirmed for this pool, which is why they are "
        "settings; an unmodelled spelling of one is not a fourth option.")


def trial_share(cfg: PoolConfig, survivors: int, last_out: int) -> float:
    """What one trial paid our entry, as a fraction of the pot, from its two-count record.

    `survivors` is `EntryOutcome.survivors_each` for the trial and `last_out` is
    `EntryOutcome.last_out_each`; at most one of them is non-zero, because a trial ends
    either with entries outlasting the final week or with the field empty, never both. The
    first is priced by `PoolConfig.co_survivor_rule` and the second by
    `PoolConfig.co_elimination_rule`, each through `share_of_pot`, and zero is what a trial
    pays when ours went out strictly before the last entry did -- the one terminal state
    that really is worth nothing.

    The one place either rule meets a trial. `_entry_trials` builds the means from it and
    `weekly` rebuilds the per-trial vector from it, so the two cannot read the record
    differently -- which is what stored shares would have let them do.
    """
    if survivors:
        return share_of_pot(cfg.co_survivor_rule, survivors)
    if last_out:
        return share_of_pot(cfg.co_elimination_rule, last_out)
    return 0.0


def _entry_trials(rng: np.random.Generator, wks: Sequence[_Week], weeks: Sequence[int], *,
                  entries: int, ledger: set[str], ours: _Ours | None, trials: int,
                  pool: PoolConfig | None = None) -> EntryOutcome:
    """The trial loop `entry_outcome` reports, over an already-reshaped season.

    Split out because `ours=None` is a quantity worth being able to ask for: it is entry 0
    played under the *field's* sampling rule, which is what this module valued our own entry
    with until #151. Keeping it reachable is what lets the gap between the two be measured
    rather than asserted from memory -- and the gap is the whole finding, since it is what
    reverses #161's leverage claim. `entry_outcome` itself never passes None: from outside
    this module our entry plays our plan.

    Two counts per trial, and which one is set says how the trial ended. `n` is the entries
    left after the final week when ours is one of them; `m` is the entries that went *into*
    the week the field emptied when ours went out in it -- the live count after the week
    before, or the whole field if it was the first. `_play` says which week ours went out in
    because the `alive` list cannot: once the field is empty every entry reads the same, and
    an entry eliminated in the last week and one eliminated three weeks earlier are worth
    the pot and nothing.
    """
    cfg = pool or PoolConfig()
    survived = sole = last = 0
    # Kept per trial, not just summed: two candidate picks are compared by their means, and a
    # difference smaller than the spread of what was averaged is not a difference. Counts
    # rather than shares, so the record does not have either pool rule baked into it.
    each: list[int] = []
    out: list[int] = []
    share: list[float] = []
    for _ in range(trials):
        led = [set(ledger)] + [set() for _ in range(entries - 1)]
        alive = [True] * entries
        died, counts, out_at = _play(rng, wks, weeks, led, alive, ours)
        n = sum(alive) if alive[0] else 0
        m = 0
        if died is not None and out_at == len(counts) - 1:
            m = counts[-2] if len(counts) > 1 else entries
        each.append(n)
        out.append(m)
        share.append(trial_share(cfg, n, m))
        survived += int(n > 0)
        sole += int(n == 1)
        last += int(m > 0)
    return EntryOutcome(trials=trials, survives=survived / trials,
                        sole=sole / trials, share=sum(share) / trials,
                        share_sd=float(np.std(share)) if share else 0.0,
                        plan=ours.plan if ours is not None else None,
                        replans=(ours.replans / trials) if ours is not None else 0.0,
                        survivors_each=tuple(each), last_out=last / trials,
                        last_out_each=tuple(out))


def entry_outcome(grid: pl.DataFrame, weeks: Sequence[int], *, entries: int,
                  ledger: Sequence[str] = (), pool: PoolConfig | None = None,
                  plan: Plan | None = None, trials: int = DEFAULT_TRIALS,
                  rng: np.random.Generator | None = None) -> EntryOutcome:
    """What our own entry is worth from here, carrying the teams it has already spent.

    The quantity a buyback is priced against, and the reason it cannot be one-over-the-field:
    that number is blind to the ledger, so it is the same in week 2 and week 6 and a buyback
    figure built on it never moves with the teams already gone. Here a fuller ledger means
    fewer legal picks, which means more weeks the entry cannot cover.

    Our entry runs in the same trials as the field, so it shares game outcomes with every
    rival holding the same team -- which is why it cannot be computed on its own and then
    combined with a field number afterwards.

    **It picks like us and not like them**, which is the correction #151 was. Its picks come
    from `solve_plan` -- `hub.season.survivor.solve` over these weeks with this ledger -- or,
    where that cannot run, from the best-available fallback `_best_available` states. Which of
    the two produced them is on `EntryOutcome.plan.source`, because they are not equally good
    and the difference between them is the model's own uncertainty about this figure. It was
    the rival sampler, and every published number was therefore about an entry that picks a
    near-chalk random team every week for the rest of the season.

    `plan` is that plan as an *argument*. Left out, one is solved here from `ledger`; passed
    in, it is played as given and re-solved only where it will not play, so two candidates can
    be scored against the same trials and differ by the picks alone. Either way it comes back
    on the result.
    """
    cfg = pool or PoolConfig()
    rng = rng or np.random.default_rng(0)
    wks = weeks_from_grid(grid, weeks, cfg)
    spent = set(ledger)
    return _entry_trials(rng, wks, weeks, entries=entries, ledger=spent,
                         ours=_Ours(wks, weeks, cfg, plan, spent), trials=trials, pool=cfg)


def buyback(grid: pl.DataFrame, weeks: Sequence[int], *, week: int,
            ledger: Sequence[str], live_entries: int, pot: float,
            rival_buybacks: int = 0, used: int = 0, pool: PoolConfig | None = None,
            trials: int = DEFAULT_TRIALS,
            rng: np.random.Generator | None = None) -> Buyback:
    """Whether paying the fee to re-enter is worth it, and the fee at which that changes.

    `week` is the week our entry was eliminated in, and the re-entry is priced over the
    weeks of `weeks` **after** it -- `_ahead`, the same slice `weekly` takes. It used to be
    priced over the whole list, so a caller handing in the season while eliminating in week
    3 had the re-entry replay weeks 1 to 3 against a ledger that had already spent them
    (#158). A week with nothing priced after it is reported as unavailable rather than
    valued as a tie with everybody still standing, which is what an empty season prices to.

    Not a survival question. Re-entering buys a share of a pot that the buybacks themselves
    enlarge, against a field those same buybacks refill -- so a rival re-entry moves the
    numerator and the denominator together, and modelling only the first would make every
    buyback look better than it is.

    **The re-entry is not a fresh entry.** It carries the teams already spent, which the
    commissioner confirmed, so the same $20 buys less in week 6 than in week 2. That is why
    the equity comes from `entry_outcome` on the real ledger rather than from one over the
    field, which is ledger-blind and identical in both.

    That confirmation is `PoolConfig.buyback_restores_ledger`, and until #160 it was read by
    nothing -- the ledger was carried in unconditionally and the field was a fact about the
    pool recorded where nothing could act on it. It is read here now, so a commissioner who
    corrects it is a re-run: `False` means the re-entry starts clean, every team is available
    to it again, and the same $20 is worth the same in week 6 as in week 2. `Buyback.spent`
    moves with it, because a figure that says three teams are gone while pricing a fresh
    entry is worse than either answer on its own.

    **The equity is what the re-entry is worth to somebody who plays it well**, since
    `entry_outcome` values it on our plan, solved from that inherited ledger. It used to be
    what the re-entry was worth to somebody picking near-chalk at random, which is nobody, and
    the direction of that error is why this verdict was essentially fixed before the pool's
    economics were consulted. It is still a floor and the reasons below are unchanged.

    `rival_buybacks` is an argument, not a model. Nobody has observed this pool's rivals and
    Hidden Picks means nobody can before a deadline, so a propensity fitted here would be an
    invention wearing a number's clothes. The caller states an assumption and the figure moves
    with it.

    **And it is not clamped to the cap** (#158). `PoolConfig.buyback_cap` is a cap *per
    entry* -- how many times one entry may re-enter across the season, provisionally four --
    and it was applied here as a ceiling on the field's total, so a caller stating six rival
    re-entries was silently priced at four and the docstring's promise that the figure moves
    with the assumption was false above the cap. The figure could not honour the cap as
    written without knowing how many rivals are out and how many re-entries each has used,
    neither of which is an input, and inventing them is the propensity the paragraph above
    refuses. So the caller's count stands, a negative one is refused, and the per-entry cap
    is applied where it is per-entry: to *ours*, through `used`, the re-entries this entry
    has already taken. An entry at the cap cannot buy back whatever the equity says.

    Future re-entries are not priced. A buyback that is itself later lost could be bought back
    again while the rule still allows it, and that option has value this ignores -- so the
    figure is a floor rather than a point.

    **And it is one point on the field-concentration axis**, the one `cfg.field_concentration`
    names, which is a stated assumption about the rivals rather than anything measured. The
    net here is the net *at that concentration*; `sensitivity` is where the range across the
    axis is reported, and a verdict whose sign flips inside that range is a verdict about the
    assumption rather than about the pool's economics.

    **What the pot pays a co-survivor is `PoolConfig.co_survivor_rule`**, applied in
    `share_of_pot` and reaching this figure through `entry_outcome.share`. It defaults to an
    even split and is a rule nobody has confirmed, so this equity is conditional on it -- and
    how *much* it is conditional on it is not fixed: a shared finish is nearly unreachable at
    a flat field and common at a concentrated one, which is `sensitivity`'s `shared` column
    and an assumption rather than a fact about the pool.

    **And what it pays the week the field empties is `PoolConfig.co_elimination_rule`**,
    the other unconfirmed rule and since #157 the larger term: a re-entry priced against a
    field that dies together is mostly priced on the week it dies, and that week paid
    nothing before. `EntryOutcome.last_out` is how much of this equity is that week, and
    `co_eliminated` how much of it the rule can move.
    """
    cfg = pool or PoolConfig()
    if cfg.buyback_cap <= 0:
        return Buyback(False, "no buybacks: the cap is zero", pot, live_entries,
                       len(set(ledger)), 0.0, 0.0, cfg.buyback_fee, 0.0, 0.0, False)
    if used >= cfg.buyback_cap:
        return Buyback(False,
                       f"no buyback: this entry has used {_plural(used, 'buyback')} of "
                       f"the {cfg.buyback_cap} the cap allows each entry",
                       pot, live_entries, len(set(ledger)), 0.0, 0.0, cfg.buyback_fee,
                       0.0, 0.0, False)
    if week > cfg.buyback_cutoff_week:
        return Buyback(False,
                       f"no buyback: week {week} is past week "
                       f"{cfg.buyback_cutoff_week}, the last one that allows it",
                       pot, live_entries, len(set(ledger)), 0.0, 0.0, cfg.buyback_fee,
                       0.0, 0.0, False)
    if rival_buybacks < 0:
        raise ValueError(
            f"`rival_buybacks` is {rival_buybacks!r}: it is how many rivals the caller "
            "assumes re-enter alongside us, and a negative count is not an assumption "
            "about the field. State zero or more; there is no cap applied here, because "
            "`PoolConfig.buyback_cap` is per entry and this is the field's total.")
    ahead = _ahead(weeks, week)
    if not ahead:
        return Buyback(False,
                       f"no buyback: nothing is priced after week {week} to re-enter for",
                       pot, live_entries, len(set(ledger)), 0.0, 0.0, cfg.buyback_fee,
                       0.0, 0.0, False)

    # Ours plus theirs, as the caller states them. Both sides of the ledger move: the fees
    # enlarge the pot and the entries refill the field.
    rivals = int(rival_buybacks)
    field = live_entries + 1 + rivals
    grown = pot + (1 + rivals) * cfg.buyback_fee
    # The rule, not the argument: what the re-entry inherits is what the pool says it
    # inherits. The caller still passes the ledger, because a run under a pool that clears it
    # has to be able to say what was cleared.
    inherits = sorted(set(ledger)) if cfg.buyback_restores_ledger else []

    out = entry_outcome(grid, ahead, entries=field, ledger=inherits, pool=cfg,
                        trials=trials, rng=rng)
    equity = out.share * grown
    net = equity - cfg.buyback_fee
    yes = net > 0
    # The fee at which net is zero once the pot has grown by that fee and the rivals' -- the
    # fixed point, not the equity. `share` does not move with the fee, so `net(f)` is linear
    # in it with slope `share * (1 + rivals) - 1`; a slope at or above zero means no fee
    # flips the verdict and the breakeven is unbounded.
    slope = out.share * (1 + rivals)
    breakeven = out.share * pot / (1.0 - slope) if slope < 1.0 else float("inf")
    return Buyback(
        available=True,
        reason=(f"{'BUY BACK' if yes else 'DO NOT BUY BACK'}: "
                f"${equity:.2f} of equity against a ${cfg.buyback_fee:.2f} fee, "
                + (f"re-entering with {_plural(len(inherits), 'team')} already spent"
                   if cfg.buyback_restores_ledger else
                   "re-entering with a clean ledger, which is what this pool's rules say a "
                   "buyback restores")),
        pot=grown, field=field, spent=len(inherits), share=out.share,
        equity=equity, fee=cfg.buyback_fee, net=net, breakeven=breakeven, recommend=yes,
        ahead=tuple(ahead), rivals=rivals)


def auto_pick(grid: pl.DataFrame, week: int, ledger: Sequence[str] = ()) -> str | None:
    """What the pool assigns when nobody submits: the best available team by the betting market.

    Not a courtesy. `docs/method.md` rule 5 says gate against the simplest thing that already
    works, and this is that thing -- free, automatic, and available every week. A season of
    weeks where the recommendation matched it is a season the model earned nothing, which is
    only countable if the fallback is worked out and written down beside the pick.
    """
    spent = set(ledger)
    wk = grid.filter((pl.col("week") == week) & (pl.col("win_prob") > MIN_PROB))
    wk = wk.filter(~pl.col("team").is_in(list(spent))) if spent else wk
    if wk.is_empty():
        return None
    # Sorted on team as well, so a tie does not let row order pick the fallback.
    return str(wk.sort(["win_prob", "team"], descending=[True, False])["team"][0])


def weekly(grid: pl.DataFrame, weeks: Sequence[int], *, week: int,
           ledger: Sequence[str] = (), entries: int, pot: float, outlay: float = 0.0,
           pool: PoolConfig | None = None, top: int = 6,
           trials: int = WEEKLY_TRIALS, rng: np.random.Generator | None = None) -> Weekly:
    """This week's pick, what it is worth, and what it cost against the free one.

    A candidate is worth `P(it wins this week)` times what the rest of the season is worth
    with it spent -- which `entry_outcome` already answers, over the weeks still to come and
    against the field. Two things follow from using the simulator rather than a formula, and
    both are properties this had to have: future team value is discounted by the chance the
    pool is still running, because a season that ends in week 6 never reaches the weeks a
    reservation was made for; and a double-pick week nobody survives to contributes nothing,
    for the same reason and without a special case.

    **The approximation, stated.** The field is not advanced through this week before the rest
    is valued, so rival attrition in the current week is not credited to any candidate. That
    understates every figure by close to the same factor, which leaves the ranking -- the part
    a recommendation is -- intact.

    **Every candidate plays the same season.** Each one gets a generator reseeded to the same
    value, so the field draws the identical results and the only difference between two figures
    is the pick itself. Letting them draw independently was tried first and is what a naive
    reading of "use the rng you were passed" gives you: on a grid where the top teams are close,
    the ranking then moved with the seed, and the week recommended departing from the free pick
    over an edge of one survival point that was entirely sampling noise. Paired trials cost
    nothing and remove it.

    **Reseeding was not enough on its own, which is #159.** One seed and one stream is only a
    shared season while both candidates consume it at the same rate, and they did not: the
    trial stopped when the last live entry died, ours included, so the first trial our entry
    outlasted the field in one candidate and not the other consumed a different number of
    game draws and offset everything after it. The pairing held up to the first week our fate
    differed, which is precisely the trials the comparison is about. `_play` now draws the
    whole season before anybody picks, so the count is fixed and the field two candidates meet
    is the same field, rival pick for rival pick.

    **And the difference is taken paired, at two standard errors.** `resolution` is
    `DECISIVE_SIGMA` standard errors of the trial-by-trial difference between the recommended
    plan and the free one -- not the root-sum-square of two independent arms, which is a
    different and much wider interval, and not one standard error, which called a difference
    real about a third of the time when there was none. What the interval is a comparison
    *of* is stated on `Weekly`: two candidate plans, not our arm against the field's rule.

    **The figures are net of `outlay`** -- the entry fee, plus any buyback already paid. That
    money is spent whichever team is picked, so subtracting it shifts every candidate by the
    same amount and cannot reorder them. It is subtracted anyway, because the question "which
    of these is best" and the question "is any of them worth having" have different answers,
    and only the second one notices that the entry cost something.

    Only the `top` most likely teams are valued. Each one costs a simulation, and a team the
    betting market prices below the sixth-best is not a candidate for a pick whose whole
    purpose is surviving the week.
    """
    cfg = pool or PoolConfig()
    rng = rng or np.random.default_rng(0)
    spent = set(ledger)
    wk = grid.filter((pl.col("week") == week) & (pl.col("win_prob") > MIN_PROB))
    wk = wk.filter(~pl.col("team").is_in(list(spent))) if spent else wk
    if wk.is_empty():
        raise ValueError(f"week {week} has no legal pick left: {len(spent)} teams are spent")

    ahead = _ahead(weeks, week)
    free = auto_pick(grid, week, ledger)
    ranked = wk.sort(["win_prob", "team"], descending=[True, False]).head(top)

    seed = int(rng.integers(2 ** 32))
    cands: list[Candidate] = []
    # The per-trial figure behind each candidate's mean, kept so the comparison at the bottom
    # can be made trial by trial. A candidate's own standard error goes on the candidate; the
    # difference between two of them is a different and much tighter quantity and is computed
    # from these rather than from those.
    each: dict[str, np.ndarray] = {}
    live: dict[str, np.ndarray] = {}
    root = np.sqrt(trials)
    for r in ranked.iter_rows(named=True):
        team, p = str(r["team"]), float(r["win_prob"])
        if ahead:
            rest = entry_outcome(grid, ahead, entries=entries, ledger=[*spent, team],
                                 pool=cfg, trials=trials, rng=np.random.default_rng(seed))
            survives, share = rest.survives, rest.share
            # The pool rules applied here rather than read off a stored share: what a trial
            # paid is a *rule* about a count, and the counts are what the trial recorded.
            # `live` is survival and only survival; a trial the field emptied with us in it
            # paid under `co_elimination_rule` and is not a survival, which is why the money
            # and the season are read off different records.
            live[team] = np.asarray(rest.survivors_each, dtype=int) > 0
            each[team] = xs = np.array(
                [trial_share(cfg, n, m)
                 for n, m in zip(rest.survivors_each, rest.last_out_each, strict=True)],
                dtype=float)
            s_se = float(np.std(live[team])) / root * p
            d_se = float(np.std(xs)) / root * p * pot
        else:
            # Nothing left to play: surviving this week is surviving, and the pot is split
            # with whoever else is still standing -- which the field statistics cannot say
            # from here, so it is claimed as an outright win rather than guessed at. No trial
            # was run, so both errors are zero because the figures are exact.
            survives, share = 1.0, 1.0
            s_se = d_se = 0.0
        cands.append(Candidate(team, p, p * survives, p * share * pot - outlay, team == free,
                               s_se, d_se))

    cands.sort(key=lambda c: (-c.expected_dollars, c.team))
    best = cands[0]
    fb = next((c for c in cands if c.is_fallback), None)
    # The interval on the *difference*, taken paired. Both arms met the same season trial for
    # trial (`_play`), so `d` is a per-trial difference and not two independent means being
    # subtracted -- the correlation is what makes this tight enough to decide a week with.
    # `win_prob` is inside it rather than left out: the figures being compared are
    # `win_prob * share * pot`, and dropping the factor inflated the bar by roughly 1/p and
    # called that conservatism. It is not conservatism when it is also the bar that decides
    # whether an unresolved week is reported as one.
    res = gse = 0.0
    cashed = 0
    if ahead and fb is not None and fb.team != best.team:
        a, b = each[best.team], each[fb.team]
        d = pot * (best.win_prob * a - fb.win_prob * b)
        res = DECISIVE_SIGMA * float(np.std(d)) / root
        gse = float(np.std(live[fb.team].astype(float)
                          - live[best.team].astype(float))) / root
        # Took a share of the pot, which is not the same as survived: under a rollover rule a
        # shared finish pays nothing, and a trial set where every finish was shared carries no
        # information about the money however much it says about survival. The week's verdict
        # is a dollar comparison, so this counts dollars.
        cashed = int(np.count_nonzero((a > 0.0) | (b > 0.0)))
    elif ahead:
        cashed = int(np.count_nonzero(each[best.team] > 0.0))
    return Weekly(
        week=week, recommend=best.team, fallback=free,
        matched=best.team == free,
        given_up=(fb.survives - best.survives) if fb else 0.0,
        pot=pot, resolution=res, candidates=cands,
        given_up_se=gse, trials=trials if ahead else 0, cashed=cashed)


def _places(se: float, *, cap: int) -> int:
    """How many decimals a figure carrying this standard error may honestly be printed to.

    The largest `d` at or under `cap` for which `10 ** -d` is still at or above the error, so
    the last digit shown is one the run can actually see. A figure printed to a precision its
    Monte Carlo error does not support is a claim about the trial count, not about the pool:
    `$21.39` off 400 trials that resolve $2.53 is four digits of which two are noise, and a
    reader comparing two such figures is comparing the seed.

    `cap` is the caller's own ceiling -- dollars are not worth more than two decimals however
    many trials were run -- and zero or a non-finite error means there is nothing to round
    for, which is the closed-form week rather than an unknown.
    """
    if not np.isfinite(se) or se <= 0.0:
        return cap
    return max(0, min(cap, int(np.floor(-np.log10(se)))))


def weekly_report(w: Weekly, *, places: int = 2) -> list[str]:
    """The week as lines rather than prints, so it can be composed and asserted on.

    A week with no auto-pick says that, rather than naming one. `fallback` is `None` there
    and was the string `"none available"`, which every reader of the field had to know to
    compare against -- including `hub.season.journal`, which would have recorded a survival
    cost against a free pick that did not exist.

    **`places` is a ceiling and not a setting**, which is #159's last acceptance criterion.
    Every figure here is rounded to what its own standard error supports, so a run at 400
    trials prints fewer digits than the same run at 4000 and the difference is visible on the
    page rather than buried in a footnote. Each candidate carries its own error, because they
    are not alike: a 3% team's dollar figure is a hundredth the size of the chalk team's and
    is resolved to a hundredth of the precision.

    **A week that could not resolve says so instead of reporting a tie.** With our share often
    on the order of a thousandth, a few hundred trials can return zero for every candidate;
    the difference is then zero because nothing was measured, not because two plans were
    measured alike. Both land on the free pick and only one of them is evidence.
    """
    if w.fallback is None:
        against = " -- auto-pick had no team left to assign, so there is nothing free to " \
                  "beat this week"
    elif w.matched:
        against = " -- which is what auto-pick would have given you for nothing"
    else:
        against = f", over auto-pick's {w.fallback}"
    head = f"\n  week {w.week}: {w.recommend}{against}"
    body = [f"  ${c.expected_dollars:.{_places(c.dollars_se, cap=places)}f}  {c.team:<4} "
            f"win {c.win_prob * 100:.0f}%  "
            f"survives {c.survives * 100:.{_places(c.survives_se * 100, cap=1)}f}%"
            + ("   <- free" if c.is_fallback else "")
            for c in w.candidates]
    # A departure can be free: gaining survival over the free pick is the case worth having,
    # and reading it out as a cost of zero would say the two plans were alike, which they were
    # not. `docs/method.md` rule 4 is about exactly this class of unread sign.
    # Survival is counted over a whole number of trials, so two candidates landing on exactly
    # the same figure is a real outcome rather than a float coincidence -- and reading that out
    # as gaining zero points would claim a direction the trials did not find.
    if w.given_up == 0.0:
        said = f"  survives exactly as well as the free pick {w.fallback}"
    else:
        verb = "costs" if w.given_up > 0 else "gains"
        said = (f"  {verb} {abs(w.given_up) * 100:.{_places(w.given_up_se * 100, cap=2)}f} "
                f"points of survival against the free pick {w.fallback}")
    cost = [] if w.matched or w.fallback is None else [said]
    if w.unresolved:
        # Named apart from the line below, because the two send a reader to the same pick for
        # opposite reasons. This one is not a measurement of zero, it is the absence of one.
        undecided = [f"  but no trial of {w.trials} cashed for either pick, so this week is "
                     "unresolved rather than a measured tie -- these trials found no "
                     "difference because they found nothing at all"
                     + (f". Take {w.fallback}, which is free" if w.fallback else "")]
    elif not w.decisive:
        res = f"{w.resolution:.{_places(w.resolution, cap=places)}f}"
        undecided = [f"  but that is inside the ${res} these trials can resolve at "
                     f"{DECISIVE_SIGMA:.0f} standard errors, so take {w.fallback} -- it is "
                     "free and no worse"]
    else:
        undecided = []
    return [head, *body, *cost, *undecided]


def report(b: Buyback, *, places: int = 2) -> list[str]:
    """The decision as lines rather than prints, so it can be composed and asserted on.

    `hub.models.experiment.paired_report` exists for the same reason: a block that prints
    cannot be capped, composed, or tested.
    """
    if not b.available:
        return [f"\n  {b.reason}"]
    span = (f"week {b.ahead[0]}" if len(b.ahead) == 1 else
            f"weeks {b.ahead[0]}-{b.ahead[-1]}") if b.ahead else "no weeks"
    # An unbounded breakeven is a sentence and not a number: `inf` on the page reads as a
    # formatting fault, and the claim it stands for is that every dollar of fee brings at
    # least a dollar of pot to our side, so no fee flips the verdict.
    flip = (f"  breakeven fee ${b.breakeven:.{places}f}" if np.isfinite(b.breakeven) else
            f"  no breakeven: a {b.share * 100:.1f}% share of a pot "
            f"{_plural(1 + b.rivals, 'buyback')} pay into gains at least a dollar per "
            "dollar of fee, so no fee flips this verdict")
    return [
        f"\n  {b.reason}",
        f"  pot ${b.pot:.{places}f} across {b.field} entries over {span}   "
        f"share {b.share * 100:.1f}%   net ${b.net:+.{places}f}",
        flip,
    ]


def simulate(grid: pl.DataFrame, weeks: Sequence[int], *, entries: int,
             pool: PoolConfig | None = None, ledgers: Sequence[set[str]] | None = None,
             trials: int = DEFAULT_TRIALS,
             rng: np.random.Generator | None = None) -> PoolOutcome:
    """Play the pool forward `trials` times and report what the field did.

    `ledgers` is what each entry has already spent, in entry order, for a mid-season run. Left
    out, every entry starts clean -- which is week 1 and is the case the model knows least
    about. Given, there must be exactly one per entry: the two are matched by position, so a
    short list is not "the rest start clean", it is an entry reading the wrong ledger or an
    index error thrown partway through a trial, and neither is an answer.
    """
    if ledgers is not None and len(ledgers) != entries:
        raise ValueError(
            f"{_plural(len(ledgers), 'starting ledger')} for {entries} entries: `ledgers` is "
            "matched to entries by position, so it has to name every one of them. Pass a "
            "ledger per entry -- an empty set for an entry that has spent nothing.")
    rng = rng or np.random.default_rng(0)
    wks = weeks_from_grid(grid, weeks, pool)
    ended: Counter[int] = Counter()
    co: Counter[int] = Counter()
    alive_tot = dict.fromkeys(weeks, 0)

    for _ in range(trials):
        led = [set(ledgers[i]) if ledgers is not None else set() for i in range(entries)]
        alive = [True] * entries
        died, counts, _ = _play(rng, wks, weeks, led, alive)
        for w, n in zip(weeks, counts, strict=False):
            alive_tot[w] += n
        if died is not None:
            ended[died] += 1
        else:
            co[sum(alive)] += 1

    return PoolOutcome(
        trials=trials,
        ending_week={w: c / trials for w, c in sorted(ended.items())},
        co_survivors={n: c / trials for n, c in sorted(co.items())},
        alive_by_week={w: alive_tot[w] / trials for w in weeks},
    )


class Sensitivity(NamedTuple):
    """One point on the field-concentration axis, and everything that moved with it.

    Both halves on one row on purpose. `chalk_share` is what the knob *sets* and the field
    and entry figures are what it *costs*, and a reader deciding whether the knob matters has
    to see the two together -- a table of lifetimes with no ownership beside them says the
    answer moved without saying what moved it.

    `resolution` is the dollar difference `trials` can actually resolve on this row, so an
    equity range across the axis can be compared against the noise inside it rather than read
    as a finding by width alone.
    """
    concentration: float
    chalk: str                  # the best team in the sweep's first week
    chalk_share: float          # the share of a clean-ledger field that draws it
    field: PoolOutcome          # ending week, co-survivors, entries alive by week
    entry: EntryOutcome         # our own entry, playing our plan against that field
    pot: float
    resolution: float           # the dollar difference these trials can resolve

    @property
    def equity(self) -> float:
        """What our position is worth in dollars at this concentration."""
        return self.entry.share * self.pot

    @property
    def wiped_out(self) -> float:
        """P(the whole field is gone before the last week ends) -- the knob's headline cost."""
        return sum(self.field.ending_week.values())


def sensitivity(grid: pl.DataFrame, weeks: Sequence[int], *, entries: int,
                at: Sequence[float] = DEFAULT_CONCENTRATIONS,
                ledger: Sequence[str] = (), pot: float | None = None,
                pool: PoolConfig | None = None, trials: int = DEFAULT_TRIALS,
                rng: np.random.Generator | None = None) -> list[Sensitivity]:
    """The pool and our position, re-answered at each field concentration in `at`.

    **The deliverable of #152, and the reason the knob defaults to a no-op.** Nobody has
    observed this pool's rivals, no pick-popularity data is fetched anywhere under
    `hub.fetch`, and under Hidden Picks nobody can observe them before a deadline -- so
    `PoolConfig.field_concentration` cannot be fitted, only stated. ADR-0024's disposition for
    a parameter in exactly that position is to measure the alternatives side by side and
    report the sensitivity rather than pick one, and this is that table. A figure quoted from
    a single concentration is a figure quoting an assumption; the range is the honest form.

    Two simulations per point -- the field's, and ours against it -- so the real cost is twice
    `trials` times `len(at)`, which at the defaults is ten `DEFAULT_TRIALS` runs. A caller
    scanning the shape of the axis should lower `trials` before it drops points from `at`:
    every point dropped is a stretch of the axis nobody looked at, and the axis is the answer.

    **The rows are separate runs and not paired trials**, which is the opposite of `weekly`
    and is stated because the difference matters to how the table reads. Each row is reseeded
    to the same value, so every row starts from the same stream and a row at 1.0 reproduces
    `simulate` at the default exactly. But the concentration changes what the *sampler*
    consumes from that stream, so the draws diverge at the first pick and no two rows share a
    season. A difference between two rows therefore carries both rows' noise, and `resolution`
    is what it has to clear -- `docs/method.md` rule 3, in its general form: the rows are not
    repeated measures of one thing that may be pooled into a tighter interval.

    `pot` defaults to the field's entry fees, `entry_fee * field_size`, because that is the
    pot the configured pool starts with and a sweep run for its shape should not need one
    named. A caller pricing a real week passes the pot it actually has.

    **What the first run of it found**, on the synthetic 32-team board in
    `tests/unit/test_pool.py` over weeks 1-14 with 21 entries and 1600 trials, 2026-09-10.
    Quoted as the shape of the response and not as figures about the real board, which this
    was not run against:

      * The ticket's premise holds and is larger than it reads. At 1.0 the field is wiped out
        before week 14 in **99.4%** of trials and only **3.3%** of endings fall in weeks 13-14
        -- so the double-pick machinery is priced into essentially nothing. At 16.0 those are
        70.4% and 43.2%. Concentration is what lets this pool reach its own back half.
      * **Our survival barely moves** across the whole axis (5.1% to 6.6%, inside the noise),
        but our *share* falls hard at the top of it (5.1% to 2.9%). The knob does not decide
        whether we survive; it decides how many rivals survive **with** us, and a split pot is
        what that costs. Reading survival alone would have reported the knob as inert.
      * `co_survivor_rule` is the rule nobody has confirmed, and at 1.0 it is unreachable --
        a co-survivor occurs in 0.6% of trials. At 16.0 it decides 29.5% of them. An
        unconfirmed rule looking harmless can be an artefact of an unmeasured assumption.
      * Equity ran $21.39, $25.99, $24.28, $21.97, $12.37 against a resolution of $2.53. Only
        the fall at 16.0 clears it: **1.0 through 8.0 are one flat region**, and the peak near
        2.0 is not a peak this many trials can see. It is a range straddling a $20 buyback
        fee, which is the sense in which that verdict is currently about the assumption.

    **Re-run with #157 in, 2026-09-11**, same board over weeks 1-14, 21 entries, 1600 trials,
    seed 0, both rules at `split`. The week the field empties was priced at zero above and
    is priced by `co_elimination_rule` now, and it is the larger term at every point:

      * `last_out` -- ours went out in the week the field emptied -- runs 33.5%, 24.4%,
        15.2%, 10.2%, 7.5% across the axis against a `survives` that stays 5-7%. At 1.0
        most of that is ours **alone**: 26.6% of trials our plan outlives the whole field
        and then loses, which is the pot in full. Share therefore runs 35.7%, 26.9%, 17.3%,
        11.6%, 6.6% and equity $149.96, $112.86, $72.63, $48.90, $27.81 against $4.85.
      * **The flat region above is gone.** Every step down the axis clears the resolution,
        because concentration is what decides whether the field outlasts our plan or dies
        underneath it, and that was the term worth nothing. The knob was never inert; it
        was priced into a state the model refused to pay.
      * `co_elimination_rule` decides 5-7% of trials at every concentration -- flat, where
        `co_survivor_rule` runs 0% to 3.3%. Under `rollover` for both, equity is $137.29,
        $98.96, $60.90, $35.70, $14.70: $12-13 lower at every point, and the verdict at 16.0
        moves from above a $20 fee to below it.
    """
    cfg = pool or PoolConfig()
    rng = rng or np.random.default_rng(0)
    seed = int(rng.integers(2 ** 32))
    stake = cfg.entry_fee * cfg.field_size if pot is None else pot
    rows: list[Sensitivity] = []
    for k in at:
        # A whole config per point, not a loose float threaded past one: everything that reads
        # a pool rule on this path -- `week_fixtures`, `solve`, `buyback`'s caller -- takes a
        # `PoolConfig`, and a second way to say what the concentration is would be a second
        # thing to keep in step with `pool_digest`.
        this = replace(cfg, field_concentration=float(k))
        chalk, own = _chalk_share(weeks_from_grid(grid, weeks, this)[0])
        field = simulate(grid, weeks, entries=entries, pool=this, trials=trials,
                         rng=np.random.default_rng(seed))
        entry = entry_outcome(grid, weeks, entries=entries, ledger=ledger, pool=this,
                              trials=trials, rng=np.random.default_rng(seed))
        rows.append(Sensitivity(
            concentration=float(k), chalk=chalk, chalk_share=own, field=field, entry=entry,
            pot=stake, resolution=entry.share_sd / np.sqrt(trials) * stake))
    return rows


def sensitivity_report(rows: Sequence[Sensitivity], *, places: int = 2) -> list[str]:
    """The sweep as lines rather than prints, so it can be composed and asserted on.

    The final line is the point of the whole table: the dollar figure as a **range over the
    knob**, beside the resolution the widest row can actually resolve. A range narrower than
    that is the sweep reporting that the concentration did not matter here, which is a result
    and is worth being able to read off directly.
    """
    if not rows:
        return ["\n  no concentrations swept"]
    weeks = sorted(rows[0].field.alive_by_week)
    last = weeks[-1]
    out = [f"\n  field concentration: {_plural(len(rows), 'point')} on the axis, "
           f"{rows[0].field.trials} trials each. 1.0 is sampling proportional to win "
           "probability -- stated, never fitted",
           f"  {'k':>5}  {'own':>7}  {'wiped':>7}  {'alive':>7}  {'survives':>9}  "
           f"{'last':>7}  {'share':>7}  {'equity':>10}",
           f"  {'':>5}  {rows[0].chalk:>7}  {'by ' + str(last):>7}  {'wk ' + str(last):>7}"
           f"  {'':>9}  {'out':>7}"]
    for r in rows:
        # `last` is the other way a trial pays (#157): ours went out in the week the field
        # emptied. Beside `survives` because the share is built from both and a reader
        # comparing two rows has to see which of the two moved.
        out.append(
            f"  {r.concentration:>5.2f}  {r.chalk_share * 100:>6.1f}%  "
            f"{r.wiped_out * 100:>6.1f}%  {r.field.alive_by_week[last]:>7.2f}  "
            f"{r.entry.survives * 100:>8.1f}%  {r.entry.last_out * 100:>6.1f}%  "
            f"{r.entry.share * 100:>6.1f}%  ${r.equity:>9.{places}f}")
    lo, hi = min(r.equity for r in rows), max(r.equity for r in rows)
    res = max(r.resolution for r in rows)
    out.append(f"  equity ${lo:.{places}f} to ${hi:.{places}f} across the axis, against "
               f"${res:.{places}f} these trials resolve")
    return out
