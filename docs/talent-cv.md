# Fitting TALENT_CV

**Fitted 2026-08-23** with `hub.draft.calibrate`, and **made per position the same day**.

| position | TALENT_CV | vs pool |
|---|---|---|
| QB | 0.41 | −0.4 se |
| **RB** | **0.49** | **+2.6 se** |
| WR | 0.41 | −0.9 se |
| **TE** | **0.32** | **−3.8 se** |
| *pooled* | *0.41* | — |

Replacing a guessed 0.35 that sat 4.6 standard errors below the pooled fit. Going per
position also turned up a model bug that this constant had been silently absorbing — see
[The model could not produce a bust](#the-model-could-not-produce-a-bust).

## Why it mattered enough to fit

`hub/draft/season.py` described this constant as *"the single most important number in the
model"* and then admitted it had not been fitted. It sets how wrong a preseason projection
typically is about a player's season: at 0 the projection is truth and drafting on it is
clairvoyance; the larger it gets, the more a draft is a lottery.

[six-of-twelve.md](six-of-twelve.md) then made it load-bearing. The season-long variance
sweep that produced the corrected draft-time advice — *buy season-long upside, do not buy
weekly volatility* — is a sweep in exactly this quantity, and that write-up named it as the
one number that would change the conclusions if it were wrong.

## The instrument: this league's own drafts

A **draft pick is market opinion recorded before week 1** and cannot be revised afterwards.
So E[realized points | pick, position] is the market's projection, fitted per position as a
power law in pick number with season intercepts and rescaled so the market is unbiased by
construction. The spread of realized/projected around that curve is the quantity wanted.

The alternative — ESPN's stored projection for a past season — was rejected because it may
have been revised after the season it predicts. That worry turned out to be unfounded but
in an unhelpful direction: as a predictor it is materially *worse*, not better, with RB
alone coming out near 1.0 — which looks like a units problem rather than a signal.
Contamination would have pushed the number down, not up, so leakage is ruled out, and the
draft board is the better instrument regardless.

The league drafted 792 players across 2022-25 with **zero keepers**, so pick number is pure
market opinion with nothing carried over.

## Result

460 drafted skill players, 2023-25, inside pick 168 (14 rounds × 12 teams — the roster the
simulator actually holds):

| | value |
|---|---|
| raw dispersion of realized/projected | 0.439 |
| weekly sampling, removed | 0.146 |
| dispersion net of it | 0.414 (95% CI 0.387–0.440) |
| **nominal — what the model needs** | **0.407**, shipped as 0.41 |
| previous value | 0.35, **−4.6 se** |

By position:

| position | n | raw | se | shrunk | shipped |
|---|---|---|---|---|---|
| QB | 59 | 0.399 | 0.036 | 0.402 | 0.41 |
| RB | 150 | **0.478** | 0.025 | 0.470 | **0.49** |
| WR | 200 | 0.396 | 0.019 | 0.397 | 0.41 |
| TE | 51 | **0.290** | 0.032 | 0.315 | **0.32** |

**Shrunk, not raw.** Four positions with 51 to 200 players each do not support four
independent numbers: the spread between them is part real and part sampling error, and
using the raw estimates treats all of it as real. Each estimate is pulled toward the pool in
proportion to its own noise, by the fraction of the between-position spread that survives
subtracting sampling variance. Only **RB (+2.6 se)** and **TE (−3.8 se)** really differ; QB
and WR sit within one standard error and shrink back onto 0.41, which is the honest answer
rather than a tidier one.

The reading: an early running back is more of a lottery than his projection admits, and a
tight end less of one.

Stable under every sensitivity tried: 0.412 on 2023-25, 0.409 adding 2022, 0.425 across the
full 204-pick draft.

## Two corrections, both of which move the answer

**Weekly sampling is not talent.** A season average over ~15 games has its own spread, worth
0.146 of the 0.439 raw dispersion. Counting it as talent would have given 0.44 rather than
0.41. The correction is applied per player, since eight games carries twice the sampling
variance of sixteen.

**Vanished players are the busts.** 2022 matched only 151 of 204 picks — retired players
drop out of ESPN's player universe — and the loss is uniform across rounds, 32 of the first
48 included. A season missing a quarter of its outcomes reads as more predictable than it
was, so the headline excludes 2022. (Including it barely moves the number, 0.409 vs 0.412,
which is reassuring rather than decisive.)

## Availability is inside the number, deliberately

Realized scoring is measured as **points per team game (total / 17)**, not per game played.
A player who missed ten weeks really did deliver close to nothing, and the simulator's
best-lineup rule benches a low-talent player exactly the way you bench an injured one — so
availability belongs in the talent term rather than being modelled separately.

Measured the other way, as scoring level only among games played, the answer is materially
lower — around 0.30. The gap between the two is the price of missed games. That figure is
also less trustworthy: it needs the weekly-noise subtraction to do much more work, and for QBs it
over-subtracts to the point of returning 0.041, which is not credible and is a symptom of
`weekly_moments`' `sd = 0.55·mu` being a scalar that does not fit every position either.

## The shape was not fitted, and came out right

Only the second moment was fitted. The rest is free validation of the model's assumption
that talent is normal and multiplicative:

| | observed | normal model |
|---|---|---|
| p10 | 0.43 | 0.44 |
| p25 | 0.68 | 0.70 |
| p75 | 1.31 | 1.30 |
| p90 | 1.55 | 1.56 |
| skew | 0.07 | 0.00 |

Near-zero skew is mildly surprising — a multiplicative model should lean right — and the
likely reason is that injuries chop the right tail back down, roughly cancelling it.

## Effect on the conclusions it was load-bearing for

Rerunning [six-of-twelve.md](six-of-twelve.md) at 0.41 **moved every number slightly and
strengthened every conclusion**:

| | at 0.35 (guessed) | per-position (shipped) |
|---|---|---|
| dP(title)/d(win), median roster | +4.1 pp | +4.1 pp |
| P(playoff), average roster | 50.0% | 50.1% |
| P(title \| seed 1) | 37.9% | 41.3% |
| bye seeds vs seeds 3-6 | 3.7x | 4.4x |
| strong roster, weekly spread 0.7 → 1.8 | 23.1% → 18.8% | 21.1% → 18.2% |
| strong roster, season spread 0.5 → 2.0 | 19.1% → 26.6% | 16.7% → 25.4% |

Nothing reversed, across both the refit and the per-position change. Seeding is worth
somewhat more than it looked, and the case for season-long upside is slightly stronger.

## The model could not produce a bust

Going per position surfaced this, and it had been distorting the constant.

Weekly points were drawn as `N(realised talent, 0.55 × *projection*)` and clipped at zero.
For a player projected at 15 whose talent collapsed to nothing, that is `N(0, 8.25)` clipped
— a half-normal averaging **3.3 points a game, 22% of his preseason projection, produced
entirely by the clip**. Every drafted bust in the simulation was quietly a useful bench
player, and there was no value of `TALENT_CV` that could express a real one.

It showed up as a 10% gap between the fitted dispersion and the value the model needed to
reproduce it: the model kept losing spread to the floor, so the inversion kept asking for a
larger constant (0.457) to compensate. That would have been papering over the bug.

`simulate_weeks` now scales weekly spread by **realised** talent rather than the projection.
An average player is untouched — at realised = projected the two formulations are identical
— while a player who loses his job loses his variance with it. With that fixed the gap
nearly vanishes: the model needs 0.407 to reproduce a fitted 0.414.

Keying weekly spread to the projection was always the wrong choice inside the simulation.
The projection is all you know at draft time, which is why `weekly_moments` uses it, but
once talent has been drawn the spread should follow the talent.

## The fit is a few percent low, and it is inverted rather than argued about

Even with the bug fixed, a fit run on the model's own output lands slightly below the
nominal that generated it, because talent and weekly points are both clipped at zero and the
curve is fitted on the same data it is scored against. Rather than size each effect,
`nominal_for` generates at a candidate value, runs the entire fit on it, and searches for the
one that returns the observed number. Whatever the bias is made of, that inverts it, and a
round-trip test pins it.

One subtlety worth stating: the inversion simulates **full seasons for everyone**, because
`simulate_weeks` has no concept of absence — every player is drawn every week and the lineup
benches whoever scores least. Availability therefore has to be carried *by* this constant.
Feeding the real games-played distribution back in would let the simulation reproduce the
observed dispersion using missed games the model does not have, and the constant would come
out too low. The first version did exactly that and returned a correction of zero.

## What is still wrong with it

**The weekly-spread constant is the next unfitted number.** `sd = 0.55·mu` is a guess of
the same kind, it varies by position at least as much — the per-game-played fit returns an
implausible 0.041 for QBs precisely because 0.55 over-subtracts for them — and it is what
the noise correction above leans on.

**One league, three seasons, 460 players.** The CI is honest about sampling error within
that, but this is one room's drafts, and a room that drafts unusually would move the number.
TE rests on 51 players, which is why it is shrunk hard.

**One league, three seasons, 460 players.** The CI is honest about sampling error within
that, but this is one room's drafts, and a room that drafts unusually would move the number.

## Effect on draft valuation

Two rosters with identical projections and identical slot counts, one leaning on RBs and one
on TEs, 20,000 seasons each:

| model | RB-heavy | TE-heavy | gap |
|---|---|---|---|
| scalar 0.41 | 7.7% | 7.3% | +0.5 pp |
| per-position | 7.7% | 6.8% | +0.9 pp |

Under a scalar the two rosters are the same lottery and the gap is pure slot eligibility —
an RB can fill the flex, a third TE cannot. Per position the gap widens by about **0.4 pp**
of title equity, roughly 5% of a baseline 8.3% chance, in favour of the RB-tilted roster.

Real and in the predicted direction, but small, and close to the resolution of this harness.
It nudges RB valuation up; it does not change a strategy.

## Reproduce

```bash
uv run python -m hub.draft.calibrate
```
