# Fitting TALENT_CV

**Fitted 2026-08-23** with `hub.draft.calibrate`. **0.411, 95% CI [0.384, 0.437]**,
replacing a guessed 0.35 that sat 4.6 standard errors low.

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
in an unhelpful direction: as a predictor it is *worse*, giving 0.646 rather than 0.411
(RB alone comes out at 1.000, which looks like a units problem rather than a signal).
Contamination would have pushed the number down, not up, so leakage is ruled out — and the
draft board is the better instrument anyway.

The league drafted 792 players across 2022-25 with **zero keepers**, so pick number is pure
market opinion with nothing carried over.

## Result

460 drafted skill players, 2023-25, inside pick 168 (14 rounds × 12 teams — the roster the
simulator actually holds):

| | value |
|---|---|
| raw dispersion of realized/projected | 0.439 |
| weekly sampling, removed | 0.154 |
| **TALENT_CV** | **0.411** (95% CI 0.384–0.437) |
| previous value | 0.35, **−4.6 se** |

By position:

| position | n | TALENT_CV |
|---|---|---|
| QB | 59 | 0.397 |
| RB | 150 | **0.474** |
| WR | 200 | 0.393 |
| TE | 51 | **0.288** |

Stable under every sensitivity tried: 0.412 on 2023-25, 0.409 adding 2022, 0.425 across the
full 204-pick draft.

## Two corrections, both of which move the answer

**Weekly sampling is not talent.** A season average over ~15 games has its own spread, worth
0.154 of the 0.439 raw dispersion. Counting it as talent would have given 0.44 rather than
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

Measured the other way, as scoring level only among games played, the answer is 0.303. The
gap between 0.303 and 0.411 is the price of missed games. That figure is also less
trustworthy: it needs the weekly-noise subtraction to do much more work, and for QBs it
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

| | at 0.35 | at 0.41 |
|---|---|---|
| dP(title)/d(win), median roster | +4.1 pp | +4.0 pp |
| P(playoff), average roster | 50.0% | 50.0% |
| P(title \| seed 1) | 37.9% | 40.5% |
| bye seeds vs seeds 3-6 | 3.7x | 4.2x |
| strong roster, weekly spread 0.7 → 1.8 | 23.1% → 18.8% | 21.6% → 18.5% |
| strong roster, season spread 0.5 → 2.0 | 19.1% → 26.6% | 17.4% → 25.7% |

Nothing reversed. Seeding is worth slightly more than it looked, and the case for
season-long upside is slightly stronger.

## What is still wrong with it

**A single scalar is a compromise.** RB fits at 0.474 and TE at 0.288 — a real difference,
not sampling noise at n = 150 and 51. Running one number across all four positions
understates how much of a lottery an early RB is and overstates it for a TE. Making
`TALENT_CV` per-position is nearly free mechanically (`simulate_weeks` already broadcasts a
per-player array into `rng.normal`), but it would change draft recommendations and needs its
own validation, so it is not done here.

**The weekly-spread constant is the next unfitted number.** `sd = 0.55·mu` is a guess of
the same kind, it varies by position at least as much, and it is what the noise correction
above leans on.

**One league, three seasons, 460 players.** The CI is honest about sampling error within
that, but this is one room's drafts, and a room that drafts unusually would move the number.

## Reproduce

```bash
uv run python -m hub.draft.calibrate
```
