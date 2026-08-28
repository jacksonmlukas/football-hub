# The Weekly projection is shown and never ranked on

**Status:** accepted 2026-08-28.

**Decision.** `hub.models.weekly` stays in the repo, is printed beside weekly consensus, and
**does not set lineups**. Start your highest-ranked player by consensus rank.

This is the pre-registered middle branch of the gate, read off the result rather than chosen
after it. The rule and all three branches were written into
[weekly-projection-plan.md](../weekly-projection-plan.md) and unit-tested before the run.

## The measurement

`hub.season.weekly_gate`, three held-out seasons, 60 rosters, 680 roster-weeks, paired by
roster-week and bootstrapped by roster:

    weekly - consensus = -0.684 points per team-week
    95% CI [-1.519, +0.159]   P(weekly better) 5.8%
    lost in 3 of 3 seasons

The interval contains zero, so this is absence of evidence rather than evidence of a loss —
but the direction is consistent across every season, and the honest reading is that consensus
is probably better, not that the two are equivalent.

## Why it is worth recording

**The projection is measurably more accurate and still loses the decision.** Against the flat
projection this repo ships today it gains **+0.074 MAE at 5.9 se in every held-out season**
([weekly-projection.md](../weekly-projection.md)). Against consensus *rank* at setting a
lineup, it loses. A future reader looking at an accuracy number that good will wonder why
nothing starts on it, and this is the answer.

The mechanism is the one [ADR-0015](0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md)
predicted when it made the decision gate primary: a lineup is a **max over a roster**. Being
wrong about a player you were never going to start costs nothing, and being wrong in a
direction that does not change the order costs nothing. Most projection error never reaches the
decision.

This is the same shape as [ADR-0013](0013-the-snap-trend-is-shown-and-never-ranked-on.md) one
level down — a signal at partial r +0.24 that lost its decision gate at −3.81 points — and the
snap-share trend is, not coincidentally, the largest term in this model.

## What it took to believe the number

The first run of this gate returned **+11.14 points per team-week at P(better) 100%** and was
reported **VOID**: 6.2% of roster-weeks were a join failure against a 2% floor pre-registered
hours earlier. The cause was an off-by-one in the as-of join that attached each consensus scrape
to the week *after* the one it ranked, benching the incumbent's arm. Fixed, the same harness
returns −0.684.

The floor is the only reason a spurious ten-point win was not written up as a result. That is
recorded because it is the argument for pre-registering a void condition at all.

## Consequences

- **Nothing in the draft or lineup path changes.** `hub.season.lineup.optimize` still does not
  set lineups (ADR-0012), and now neither does the Weekly projection.
- **The module stays**, per ADR-0007: a measurement that steered a decision remains in the tree
  with its harness, and this one steered *this* decision.
- **Displaying it is the action**, not a consolation. Beside a consensus rank, a projection that
  disagrees is information for a human deciding a close start/sit — which is the disposition the
  snap trend already has.
- **Four screened signals are real and unused.** The prior TD rate, the snap-share trend,
  defence-vs-position and the injury designation all clear a joint screen at 5/5 seasons. Two of
  them are in the model; none of them sets a lineup.

## What would change it

**Rosters that are not static.** The gate drafts a roster and freezes it for the season, so it
deletes the streaming decision — deciding who to add for one week is a large part of what a
weekly projection is *for*, and it is the only decision where a week-specific number has an
obvious edge over a season-long rank. A version with waiver churn is the one experiment that
could plausibly flip this, and it was named as a known limitation before the run rather than
after it.

Re-running for any other reason means re-running the harness, not re-arguing the result.
