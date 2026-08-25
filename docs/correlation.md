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

## In the simulator too

`simulate_weeks` takes an `nfl_team` argument and correlates teammates by Cholesky on each
team's own small block. `champion_probability` passes it through and `hub.draft.optimize`
supplies it from the board's existing `team` column, so the draft optimizer now prices a
stacked roster as the more volatile thing it is.

That required one change to how a week is drawn. Weekly points were drawn from a shifted
gamma matched to (mean, spread, skew); a gamma cannot easily be correlated, so the draw is
now Cornish-Fisher on a Gaussian latent — the latent correlates trivially, and the quadratic
term supplies the skew with the variance it adds divided back out. Mean and spread are
unchanged by construction.

One approximation worth naming: correlating through a shared quarterback implies a small
positive correlation between two pass catchers on the same team, around +0.05 where the
measured figure is +0.014. The star topology the data actually shows — the quarterback
correlated with each catcher, catchers not with each other — is not exactly representable
this way, because catchers compete for the same targets and would need a slightly negative
conditional correlation. The overstatement is small and in the conservative direction for a
roster holding two catchers from one team.

## What this does not cover

**Opponent correlation is not modelled.** `championship-leverage.md` makes the point that
correlation matters twice: within your roster, and against your opponent's. Shared game
exposure shrinks *margin* variance, which helps whoever is favoured. Nobody prices it here
either.

> **Measured 2026-08-25** — [opponent-correlation.md](opponent-correlation.md). It is real and
> concentrated on the passing game: two opposing quarterbacks correlate at **+0.148** (4.5 se),
> which is larger than the QB-RB *teammate* edge this document fits. QB-TE +0.066 and QB-WR
> +0.055 both clear four standard errors, and RB-RB goes *negative* at −0.024, which is what
> game script implies. Still not priced, and deliberately: the two consumers of a correlation
> term — `lineup.optimize` and `simulate_weeks` — are both inert (ADR-0012, ADR-0009), so
> wiring it would add a parameter and its plumbing to buy nothing measurable.

**QB–opposing-DST (~−0.45 per the doc) is unmeasured**, since this league's DST handling was
never in scope for the component layer.

## Reproduce

The measurement and gate run from `hub.fetch.nflverse` weekly player stats; see
`docs/component-projection.md` for the data layer.
