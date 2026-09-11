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

> **Nominal restated 2026-09-11 — net of absence, issue #235.** Nothing in the fit below
> moved: the dispersion is 0.408, CI [0.370, 0.453], on the same 460 player-seasons. What
> moved is the inversion that turns it into the constant the model needs. Since #183 the
> season simulator draws missed games for itself, and `calibrate.nominal_for` was still
> inverting through a full season for every player — so the constant carried the absence
> variance a second time. Inverted through each player's real games played (sd 2.9–4.3 of
> ~14 by position):
>
> | | 2026-09-07 | re-run 2026-09-11 |
> |---|---|---|
> | nominal, pooled | 0.424, shipped 0.42 | **0.322, shipped 0.32** |
> | dispersion interval's ends, inverted | — | **[0.267, 0.385]** |
> | QB / RB / WR / TE nominal | 0.419 / 0.482 / 0.419 / 0.332 | **0.198 / 0.380 / 0.314 / 0.181** |
> | shrunk raw by position | 0.408 / 0.451 / 0.390 / 0.315 | unchanged |
>
> The double count, measured through `season.simulate_weeks` at a mean-14 projection with
> two missed games prior, old constants against new with absence drawn either way: season-
> total spread was overstated **46% (QB), 38% (TE), 23% (WR), 18% (RB)**; means unchanged.
> Quarterbacks and tight ends move most because their seasons vary most in length relative
> to their spread. On synthetic rows with a quarter of the players missing six to fourteen
> games, the full-season inversion returned 0.464 for a true 0.30; the real-games inversion
> recovers it (`test_calibrate`). "Availability belongs inside the number" was right while
> the simulator had no absence and is not now; it is drawn, and the number is talent net of
> it. Downstream: the `six-of-twelve.md` variance sweep is a sweep in this quantity at the
> old base and is not re-run here.

> **Interval restated 2026-09-07 — it was over the wrong unit, around a curve it treated as
> known.** Issue #172. Everything in the table below is a figure from the 2026-08-23 fit and
> keeps its original text; what follows supersedes the interval and the two shrunk
> per-position values, and comes from a re-run rather than an argument.
>
> | | published | re-run 2026-09-07 |
> |---|---|---|
> | dispersion net of weekly sampling | 0.408 | **0.408** (unchanged) |
> | 95% CI | [0.380, 0.434] | **[0.370, 0.453]** |
> | width | 0.054 | **0.083**, 54% wider |
> | population | 460 player-seasons | **460 player-seasons over 235 players** |
> | RB, shrunk and debiased | 0.501, **+2.6 se** from the pool | **0.482, +1.7 se** |
> | TE, shrunk and debiased | 0.321, −3.9 se | **0.332, −3.8 se** |
>
> (The [0.387, 0.440] in the table below is older still — it predates the square-root spread
> law, which the two constants are coupled through. [0.380, 0.434] is what the shipped code
> produced up to today, and is the figure this supersedes.)
>
> **Three defects, and they are not equal.** The bootstrap fitted the curve *once, outside*
> the resample and then resampled the residuals that fit produced; its resample unit was the
> player-*season*, so a player drafted three times counted as three independent draws; and it
> indexed the weekly-noise correction by the same draw as the residuals, tying two estimates
> that vary independently. Switching them on one at a time, against the old width of 0.053
> (4,000 replicates, same seed):
>
> | correction | width | vs old |
> |---|---|---|
> | *(none — reproduces the published interval)* | 0.053 | 1.00× |
> | player as the resample unit | 0.051 | 0.97× |
> | noise term from its own draw | 0.054 | 1.02× |
> | **curve refitted inside the draw** | **0.075** | **1.42×** |
> | all three (shipped) | 0.080 | 1.52× |
>
> **The refit is essentially the whole of it.** The curve is a power law in pick number with
> season intercepts — four parameters, estimated from the same 460 rows the residuals are
> measured against — and treating it as known omitted its uncertainty from every interval
> downstream.
>
> **The unit was the wrong unit and it barely matters here, which is worth stating rather
> than hiding.** 140 of the 235 players appear more than once, so nearly half the rows were
> being counted as independent — and yet clustering alone moves the width by 0.97×, inside
> Monte Carlo noise. The reason is measurable: **the correlation between one player's two
> residual ratios is +0.013** over 310 within-player pairs. The market re-prices him every
> August, so his 2024 miss carries almost no information about his 2023 miss, and the repeated
> rows really were close to independent draws. The unit is still wrong to leave as the
> player-season — the correction costs nothing and becomes load-bearing the moment those
> residuals correlate — but on this data it is the refit that moved the number.
>
> **What the verdict does not do is move.** 0.35 was 4.6 se low against the old interval and
> is still far outside the new one; `TALENT_CV = 0.42` still sits inside it. The interval is
> wider, which makes the result weaker rather than differently-shaped.
>
> **One shipped constant changes its status.** No raw per-position estimate moved — QB 0.407,
> RB 0.471, WR 0.387, TE 0.282 are what they were. But `calibrate._shrink` reads the
> per-position standard errors to decide how much of the spread between positions is real, and
> those errors were understated by the same three defects, by 1.05× (TE) to 2.09× (QB). Larger
> errors shrink harder, so **`TALENT_CV_BY_POS` moves RB 0.50 → 0.48 and TE 0.32 → 0.33**.
> More importantly, **RB no longer clears two standard errors from the pool** (+2.6 se →
> +1.7 se). "An early running back is more of a lottery than his projection suggests" is now a
> direction the fit leans, not a difference it establishes; only TE still separates. The
> per-position split stays, because a shrunk estimate is still the best available number, but
> it is no longer two significant differences — it is one.

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

**The weekly-spread constant has since been fitted too** —
[weekly-spread.md](weekly-spread.md). It came back the wrong *shape*, not the wrong value:
spread goes as `k·√mu`, not `0.55·mu`. Since the fit above subtracts weekly sampling, the
two constants are coupled and TALENT_CV was refitted against the new law; pooled moved
0.41 → 0.42 and the per-position values by about 0.01, all inside their intervals.

The QB anomaly flagged here — an implausible 0.041 from the per-game-played variant —
resolved exactly as suspected. Under the √ law quarterbacks land at −0.0 se from the pool;
0.55 had been over-subtracting for the steadiest position.

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
