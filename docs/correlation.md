# Teammate correlation: the L1 gate

**Measured 2026-08-24.** `championship-leverage.md` calls this L1 and gates everything above
it on one question:

> does the correlated joint beat independent marginals on held-out weekly *team-score*
> distributions? Score with log-loss and calibration, not correlation recovery.

**It passes, narrowly and specifically.** Only the quarterback's edges matter. Everything
else can stay independent, which makes L1 far smaller than the doc envisioned.

## What correlates

Within-game correlation of standardised weekly points between teammates, 2022-25:

| pair | n | r | doc's prior |
|---|---|---|---|
| QB–WR1 | 1,860 | **+0.323** | +0.3 to +0.5 ✓ |
| QB–WR2 | 1,693 | +0.293 | |
| QB–TE1 | 1,724 | +0.252 | "second strongest positive" |
| QB–RB1 | 1,834 | +0.060 | |
| WR1–WR2 | 1,727 | +0.025 | ~+0.16 at ceiling |
| RB1–WR1 | 1,846 | −0.024 | ~−0.07 ✓ |
| WR1–TE1 | 1,736 | +0.013 | |
| RB1–RB2 | 1,630 | +0.013 | |

The published priors hold where the doc gave them. Two corrections: the doc ranks QB–TE1 as
the second strongest positive and it is third, behind QB–WR2. And WR1–WR2 is **+0.025**, not
the +0.16 the doc cites — that figure is described there as holding "at ceiling outcomes",
so the two are not necessarily in conflict, but the average-case number is essentially zero
and that is what a lineup is priced on.

At position level, which is all a roster actually knows:

| pair | n | r |
|---|---|---|
| QB–WR | 6,352 | **+0.232** |
| QB–TE | 3,057 | **+0.225** |
| QB–RB | 4,077 | +0.054 |
| everything else | — | within ±0.03 of zero |

## The gate

Correlations fitted on 2022-24, evaluated on held-out 2025. An 80% interval should cover 80%
of outcomes; below that the model is overconfident.

| grouping | model | n | 80% coverage | mean log score |
|---|---|---|---|---|
| QB + his pass catchers | independent | 542 | **72.9%** | −4.122 |
| QB + his pass catchers | correlated | 542 | **80.4%** | −4.094 |
| QB + WR1 only | independent | 418 | 75.4% | −3.923 |
| QB + WR1 only | correlated | 418 | 81.3% | −3.910 |
| no QB (WR1/WR2/RB1/TE1) | independent | 541 | 79.3% | −4.001 |
| no QB (WR1/WR2/RB1/TE1) | correlated | 541 | 79.1% | −4.001 |

**Independence is materially overconfident for a stack** — a nominal 80% interval covers
72.9% — and correlation fixes the calibration almost exactly while improving the log score.
**Without a quarterback, independence is already right** and correlation changes nothing.

That is a much narrower result than "build a copula over six positions conditioned on game
total and spread". The honest version of L1 is three numbers.

## What it changes

`hub/season/lineup.py` prices a lineup's spread with `components.group_sd`, which adds the
covariance terms for teammates. Nothing else changes: players on different NFL teams
contribute no covariance, and a roster with no `nfl_team` column behaves exactly as before.

The consequence is the same state-dependence as everywhere else in this repo. Correlation is
volatility, so **stacking your quarterback with his own receiver helps when you are an
underdog and hurts when you are favoured**, and the optimizer now works that out per matchup
rather than taking a view on stacking in general.

## What this does not cover

**The simulator still draws players independently.** `simulate_weeks` has no NFL-team input,
so `champion_probability` and everything built on it — the draft optimizer, the leverage
harness — still treat a stacked roster as uncorrelated. That understates the variance of a
stacked roster at draft time, in exactly the weeks a stack is for. Wiring team identity
through the simulator is the next step and is not done.

**Opponent correlation is not modelled.** `championship-leverage.md` makes the point that
correlation matters twice: within your roster, and against your opponent's. Shared game
exposure shrinks *margin* variance, which helps whoever is favoured. Nobody prices it here
either.

**QB–opposing-DST (~−0.45 per the doc) is unmeasured**, since this league's DST handling was
never in scope for the component layer.

## Reproduce

The measurement and gate run from `hub.fetch.nflverse` weekly player stats; see
`docs/component-projection.md` for the data layer.
