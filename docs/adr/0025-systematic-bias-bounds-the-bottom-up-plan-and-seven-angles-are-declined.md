# Systematic bias bounds the bottom-up plan, and seven angles are declined

**Status:** accepted 2026-09-10, recording seven declines made **2026-09-07**. From the
research artifact of that date — `plan.do_not_build` and finding G6 — filed as issue #216.

**Decision.** Seven angles are declined, each on a stated reason rather than a hunch, and each
with the condition that would reopen it. Alongside them, the arithmetic that **bounds** the
bottom-up programme is recorded with its numbers rather than as a summary, because it is the
reason that programme has a stop condition at all instead of an open-ended list of
refinements.

This record exists so no future session rebuilds any of them. Nothing here is reopened,
adopted or reversed by writing it down; the reopen conditions below are conditions, not
predictions.

## The arithmetic that bounds it

The bottom-up vision is: project players, aggregate them into a team, and price the game off
the aggregate. The intuition that makes it attractive is that player-level error washes out —
and for one kind of error it does.

**Random, independent player error averages out in relative terms as players are added.
Systematic bias does not: it adds linearly.** That is the whole finding, and the numbers say
how much it costs:

| step | figure |
|---|---|
| bias in projected **Usage** | **3%** |
| therefore, bias in team yards | **3%** |
| team yards | **≈ 350** |
| so, bias in yards | 350 × 0.03 = **10.5** |
| yards per point | **≈ 14** |
| so, bias in points | 10.5 ÷ 14 = **≈ 0.75 a team** |
| both teams, added | **≈ 1.5 on the total** |
| band this repo could exploit at all | **≈ 1 point** |

So a 3% bias in Usage — not an outlandish figure, and not one any run here has bounded below
it — spends about three quarters of the whole available band on one team's spread, and half as
much again as the entire band on the total.

**Usage projections are exactly where correlated bias lives.** Pace, pass-rate regression and
injury-replacement rules are shared across a whole roster, so their errors are common-mode
across the players they touch rather than independent between them. The aggregation step that
is supposed to cancel error is being handed the one kind it cannot cancel.

**The two figures behave differently and the difference is the point.** Common-mode error adds
in the total and cancels in the spread. The 0.75 a team is therefore the per-team magnitude
*before* any cancellation — an upper bound on the spread exposure, reached only where the two
teams' biases are independent — while the 1.5 on the total is the case where they are common,
which is the case Usage bias actually is. Totals are the worst exposure and the spread is
merely bad, which is what declines angle 7 below on its own.

Note what this is not. It is not a **gate** and does not pretend to be one: no arm was scored
against a comparator. It is an arithmetic bound of the kind `docs/method.md` rule 8 asks for —
compute the ceiling before chasing the gap — and the gap here is under the ceiling.

## The seven angles

Each was declined **2026-09-07**.

| # | angle | why it was declined |
|---|---|---|
| 1 | A non-quarterback spread adjustment | Two independent methods converge on roughly **half a point** for the best non-quarterback in football, and about **0.12** for the best pass rusher — against a betting-market tick of half a point. The best case in the league is one tick, and everything below it is inside the price grid |
| 2 | An offensive-line continuity model | Killed by a published study, which also identified the **selection bias** generating the illusion: the continuity that looks predictive is itself downstream of the teams that keep a line together |
| 3 | A bye-week or rest-differential angle | Measured on **5,679 games**: **+0.31** points post-2011 and **not significant**, against **+0.97** priced. The error runs *against* you — the betting market prices about three times the effect the measurement can find, so the mispricing such as it is sits on the side you would be taking |
| 4 | A west-coast-travel angle | Dead, and probably never alive |
| 5 | A slow-simulator / fast-surrogate distillation | Solves a problem this repo does not have. Distillation buys decisions inside a runtime budget; there is **a week between slates** |
| 6 | Aggregating a betting-market-conditioned player model into a game prediction | Not hard — **invalid**. Conditioning a player model on the betting market's own number and then aggregating those players back into a team number launders that number: the comparison stops being biased and becomes meaningless. **No engineering fixes it.** #212 puts a guard test on the constraint |
| 7 | Beating totals via aggregation | The systematic-bias exposure above is worst exactly there, because common-mode error **adds in the total and cancels in the spread**. The one place aggregation is most attractive is the one place the bound bites hardest |

## What would reopen each

A decline is not permanent by default. These are the conditions, and each follows from the
evidence in the row above it rather than from an appetite to revisit.

| # | reopens on |
|---|---|
| 1 | A method that puts a single non-quarterback's spread value clearly **above** one tick, or a betting market quoting finer than half a point. Two methods already agree at about a tick, so a third disagreeing with both is the evidence required |
| 2 | A study that handles the named selection bias and **still** finds the effect. Re-running the naive version is not that |
| 3 | A re-measurement that separates **+0.31** from zero, or one that closes the distance to the **+0.97** priced. Note the direction: if it reopens at all it reopens as a fade of the priced number, not as a play on rest |
| 4 | A measurement. The decline rests on there being none worth the name, which is also the weakest ground of the seven — see below |
| 5 | A change in the calendar, not in the simulator. If a decision ever has to be made inside the simulator's own runtime rather than with a week to spare, the trade this buys becomes real |
| 6 | Nothing an engineer can do. It reopens only if the player model **stops conditioning on the betting market**, at which point it is a different model and this is a different question. #212's guard is what would notice a future feature quietly re-introducing the condition |
| 7 | Angle 6 first, and then a Usage projection whose systematic bias is **bounded below** the band in the table above — bounded, not assumed. Until a bound exists this is declined by arithmetic rather than by taste |

## Two of the seven are recorded thinner than the rest

Named rather than smoothed over, because a record that hides its own weak rows is worse than
one that has none.

**Angle 2 does not name its study.** The record says a published study killed it and identified
the selection bias; it does not carry the citation. Reopening on the terms above therefore
begins by finding the study again, which is work this decline was supposed to save. That is a
real cost and it is stated rather than papered over.

**Angle 4 carries no number at all.** *"Dead, and probably never alive"* is a verdict without a
measurement behind it in this record, and it is the only one of the seven in that position. It
is kept because the cost of the angle is high and the prior against it is strong, and it is
flagged because by this repo's own standards — `docs/method.md` rule 12 — a decision taken
without a measurement is reported as judgment and never as evidence. Read it as judgment.

No figure has been invented for either. Where the research thread recorded no number, none
appears here.

## Why this is one record and not seven

The seven are not independent. Angles 6 and 7 are consequences of the arithmetic above, angle 7
doubly so; angles 1 and 3 are the same shape as each other — a real effect smaller than, or
already inside, what the betting market charges to act on it. Splitting them would put the
bound in one file and the things it bounds in seven others, and the bound is the part a future
session needs first.

## What would reopen this record as a whole

The arithmetic moving. It rests on three numbers — 3% bias in Usage, ≈350 team yards, ≈14 yards
per point — and on the band being about a point. A measured bound on Usage bias materially
below 3%, or a band materially above a point, changes what the table says and reopens angles 6
and 7 with it. The other five stand or fall on their own rows.
