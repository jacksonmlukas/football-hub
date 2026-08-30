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

    weekly - consensus = -0.304 points per team-week
    95% CI [-1.043, +0.415]   P(weekly better) 20.0%
    lost in 2 of 3 seasons

(Corrected 2026-08-28: the first figure published here was -0.684, one draw from a
non-reproducible board -- see improvements.md #18. The verdict is unchanged.)

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

## The experiment this ADR named has been run

**Measured 2026-08-28, and it strengthens the decision rather than reopening it.** With one
waiver add/drop a week — both arms, identical rules, a pool both can score — the Weekly arm
does not close the gap, it loses **−3.790 points per team-week, CI [−5.243, −2.382], 0 of 3
seasons**. Unrestricting the pool, which favours the arm under test, gives −3.661: the same
answer.

The cause is a **winner's curse**, and it is worth stating precisely because it is not "the
projection is bad". The projection is unbiased at every sample size (−0.36 to −0.09 points).
But a waiver pick is the *maximum* of a noisy estimator over hundreds of candidates, and among
players with four games or fewer the top 400 by projection project 23.00 and score 17.12 —
**+5.88**. A rostered player has history; the waiver pool is exactly the thin-sample
population. Consensus rank, being human-curated and coarse, is not the argmax of a per-player
regression fitted on four games.

**The remedy is known and deliberately not applied.** Shrinking each projection toward its
positional mean by sample size would be a model change made after seeing a gate fail, which is
the tripwire lesson [ADR-0009](0009-championship-equity-does-not-pick.md) records. It is
pre-registered as the next experiment in
[weekly-projection-plan.md](../weekly-projection-plan.md), not folded in tonight.

See [weekly-gate.md](../weekly-gate.md) for both runs and the waiver-rule artifact found in
between.

## And so has the shrinkage that was supposed to fix it

**Measured 2026-08-28.** Pre-registered, fitted on training seasons, both gates re-run:

| shrinkage | frozen | churn |
|---|---|---|
| none | −0.304 | −3.790 |
| as pre-registered | −0.235 | **−3.773** |
| aggressive | −0.474 | −12.442 |

The pre-registered fit chose **no shrinkage at all** in every held-out season, because the
projection is already unbiased at every sample size and mean absolute error cannot see a
winner's curse — it lives at the maximum over hundreds of candidates and the mean is dominated
by the bulk. The aggressive variant removes the tail bias by removing the signal: 16% of MAE,
and the churn loss triples.

Uniform shrinkage toward a single positional mean is the wrong instrument, which
[component-projection.md](../component-projection.md) had already found four days earlier at
season grain — *"a WR1 and a WR5 do not regress toward the same place"*. Same failure, weekly
grain, independent route. See [weekly-shrinkage.md](../weekly-shrinkage.md).

## And so has the market-implied variant

**Measured 2026-08-28**, at the user's request, as the fourth attempt at the same rescue.
Shrinking toward what a player's *preseason* rank implies — refitted per fold, never the
weekly ranking — gives the **first positive point estimate in the programme**: frozen
**+0.159** [−0.433, +0.736], churn **−2.450** [−3.637, −1.290].

**It does not change this decision, for two reasons stated before the run.** The interval
contains zero and it won 2 of 3 seasons, so the verdict is SHOW either way. And the
pre-registered tripwire *fired*: the frozen gate was to move less than 0.3 and moved 0.46,
which correctly identifies that `w = n/(n+k)` never reaches 1 and so this blends market
information into every projection rather than regularising thin ones. It is a different model,
not a shrunk one.

A probe rules out the obvious confound: the market prior *alone* scores −1.146, worse than
either the model alone (−0.304) or the blend (+0.159). The gain is complementarity, not
import.

**What it licenses is a new pre-registration, not an amendment to this one.** The market/Usage
blend is a model in its own right and should be gated as one, once, cleanly. See
[weekly-market-shrinkage.md](../weekly-market-shrinkage.md).

## Closed, 2026-08-29

That clean run happened, under [ADR-0017](0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md):
frozen **+0.215** CI [−0.249, +0.659] on four held-out seasons, churn **−1.806**. The
every-season half fails — it lost 2025 — and the frozen gain decays monotonically across the
four seasons, **+0.983 → −0.504**, negative in the one closest to the season being drafted.

(Restated 2026-08-30. It first ran at frozen **+0.711** [+0.313, +1.129] and churn **−2.006**,
against a `board_as_of` that was not reproducible — improvements.md #18. Those figures are
superseded rather than wrong-at-the-time; under the fixed board the same command returns +0.215
every time. The interval now *contains* zero rather than excluding it, which makes the result
weaker rather than differently-shaped, and the verdict does not move. Full restatement in
[weekly-blend-gate.md](../weekly-blend-gate.md).)

**Verdict SHOW on both gates. This decision stands and the weekly programme is closed.**

## What would change it

New evidence, not a new variant: a season measured *forward* from here. The 2026 season will
produce one, and the blend can be scored against consensus as it happens without any of the
data being spent in advance. That is the only remaining honest test, and it costs nothing but
waiting.

**Rosters that are not static.** The gate drafts a roster and freezes it for the season, so it
deletes the streaming decision — deciding who to add for one week is a large part of what a
weekly projection is *for*, and it is the only decision where a week-specific number has an
obvious edge over a season-long rank. A version with waiver churn is the one experiment that
could plausibly flip this, and it was named as a known limitation before the run rather than
after it.

Re-running for any other reason means re-running the harness, not re-arguing the result.
