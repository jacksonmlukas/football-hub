# Projecting components, not points

**Built 2026-08-24.** `hub.models.components` builds fantasy points from the counts that
make them up, instead of projecting the total directly.

Points are an aggregate of receptions, carries, yards and touchdowns. Modelling the
aggregate throws away everything the counts know — and what they know turns out to matter
more for the *distribution* than for the mean.

## Screen first: is it worth building?

1,049 player-season pairs, 2021-25, predicting per-game points from the prior season:

| predictor | RMSE | MAE | Spearman |
|---|---|---|---|
| A carry points forward | 3.384 | 2.658 | 0.795 |
| B components, own TD rate | 3.373 | 2.650 | 0.797 |
| **C components, TD regressed** | **3.299** | **2.587** | **0.806** |

B ≈ A is the sanity check: reconstructing points from components and carrying them forward
is the same predictor, so the aggregation is faithful. The gain is entirely in C, and comes
from one thing — **regressing touchdowns while leaving volume alone**.

Paired bootstrap on C − A: **−0.086 RMSE, 95% CI [−0.145, −0.031], 99.7%**. Real, and small.

Sweeping the shrink weight from "keep his own touchdown rate" to "use his position's rate on
his own yards" improves monotonically and flattens at **1.0**. A player's own touchdown rate
carries no information beyond his yardage.

**Be clear about the size.** That is a ~2% RMSE gain on the mean, concentrated in RB (+2.1%)
and WR (+1.9%); TE is +0.3% and QB is −0.9%. If the mean were the only reason to do this, it
would be marginal. It is not the reason.

## The real payoff: the distribution

Weekly scoring, measured within player-season across 12,673 player-weeks:

| position | skew | median / mean |
|---|---|---|
| QB | 0.29 | 0.99 |
| RB | 1.15 | 0.89 |
| WR | 1.07 | 0.90 |
| TE | 0.98 | 0.90 |

A normal says skew 0.00 and median/mean 1.00. **The model was drawing normals**, so it
believed the typical week equalled the projection. It does not — the mean is carried by
touchdown spikes, and the median week is about 10% below it. Every floor-based decision was
being flattered.

Sampling the counts reproduces this instead of assuming it away.

## The distributions are measured, not assumed

Weekly variance-to-mean ratios, 2022-25, which is what picks the family:

| quantity | var/mean | family |
|---|---|---|
| receptions | 1.20 | negative binomial |
| carries | 2.39 | negative binomial |
| pass attempts | 2.45 | negative binomial |
| receiving / rushing / passing TDs | 0.83 – 0.86 | **binomial** |

Volume counts are **over**dispersed because usage itself moves week to week — a team that is
ahead runs, a quarterback who is behind throws. A Poisson around a fixed rate cannot say
that, and using one left simulated spread ~13% light.

Touchdowns are **under**dispersed, being bounded by a small number of scoring chances.
Poisson overstated them, which overstated skew.

Yardage is compound: a week's receiving yards are the sum of one gain per catch. Modelling
it as a Gamma with constant CV makes its variance grow with the *square* of the mean, which
quietly reimposes the proportional spread law this whole layer exists to replace — the first
version of the file did exactly that and produced spread growing as `mu`, not `√mu`. Units
and yards-per-unit are sampled separately instead.

## Validation: simulated against observed, 760 real player-seasons

Each player's own component means fed in, simulated weekly shape compared to his actual:

| position | observed sd | simulated sd | observed skew | simulated skew | mean error |
|---|---|---|---|---|---|
| QB | 7.32 | 6.45 | 0.15 | 0.39 | +0.24 |
| RB | 6.85 | 5.63 | 0.67 | 0.68 | +0.07 |
| WR | 6.83 | 6.40 | 0.66 | 0.83 | +0.00 |
| TE | 5.99 | 5.52 | 0.72 | 0.81 | +0.01 |
| **all** | **6.76** | **6.05** | **0.60** | **0.73** | **+0.05** |

The mean is unbiased to +0.05 points a game, which is the property that matters most — a
sampler that moved the mean would bias every projection.

**On the spread gap.** 6.05 against 6.76 looks like 10%, but 5.3% of the observed figure is
*within-season role drift*: a player whose usage grows across a season shows that growth as
weekly variance. Detrending each player-season linearly gives an observed 6.40, so the real
gap is **2.7%** — and it is arguable that a forward projection should not reproduce role
drift at all, since it is projecting a role rather than an average over a changing one.

**On the skew gap.** 0.73 against 0.60 overstates it, RB aside where it lands almost exactly
(0.68 vs 0.67). The likely remaining cause is that units and efficiency co-move within a game
— a week with ten targets is usually a week the offence is moving the ball — and the compound
model treats them as independent. Not fixed; recorded.

## What changed downstream

**`simulate_weeks` now draws a shifted gamma** matched to (mean, spread, skew) rather than a
normal. A gamma's skew is `2/√shape`, so three moments pin three parameters, and it stays a
single vectorised draw. Mean and spread are unchanged by construction; only the shape moves.

**`hub.draft.leverage` was silently running a superseded model.** It re-implemented
`simulate_weeks`' internals rather than calling it, so it kept proportional spread, normal
draws, and spread keyed to the projection long after the simulator moved on — meaning every
number in [six-of-twelve.md](six-of-twelve.md) was computed against a model the repo had
stopped using. It now draws through the simulator, and a test pins the two together by
behaviour so they cannot drift apart again.

Rerunning [six-of-twelve.md](six-of-twelve.md) through the corrected path leaves every
conclusion standing, with one refinement: extra weekly spread now *slightly helps* a weak
roster (2.5% → 3.1% title equity) where it used to read as flat, because a skewed
distribution gives an underdog real upside. It still clearly hurts a strong roster
(19.8% → 15.9%). The state-dependence that [lineup.py](../src/hub/season/lineup.py) applies
weekly is now visible at draft level too.

## The volume model was screened and came back null

The obvious next step was to stop carrying volume forward and model it: shrink volume and
efficiency toward positional means by their measured persistence, then aggregate. Year-over
-year persistence says exactly how much to shrink each piece, with no free parameters —
the optimal keep-weight is the correlation itself:

| quantity | year-over-year r |
|---|---|
| targets | 0.805 |
| carries | 0.791 |
| receiving yards | 0.805 |
| **points per game** | **0.775** |
| pass attempts | 0.745 |
| catch rate | 0.402 |
| yards per target | 0.369 |
| yards per carry | 0.108 |
| **receiving TD rate** | **−0.004** |
| **rushing TD rate** | **−0.030** |

Two things worth reading twice. **Volume persists better than points do** (targets 0.805
against 0.775), which is the premise of the whole approach and it holds. And **touchdown
rate per yard has literally zero persistence** — −0.004 and −0.030 — which independently
confirms the full-shrink result from the fit above by a completely different route.

So the premise is right. The model built on it is still not worth shipping:

| | RMSE, TD regression only | RMSE, + volume/efficiency shrinkage | gain |
|---|---|---|---|
| QB | 3.764 | 3.397 | +9.7% |
| RB | 3.772 | 3.868 | −2.6% |
| WR | 3.265 | 3.301 | −1.1% |
| TE | 2.454 | 2.476 | −0.9% |
| **all** | **3.303** | **3.311** | **−0.2%** |

Paired bootstrap on the difference: **+0.008 RMSE, 95% CI [−0.057, +0.070], P(better)
40.2%.** A null, and slightly the wrong way.

QB alone comes back at −0.354 [−0.724, 0.000], P(better) 97.5%, and it is tempting. It is
not taken, for two reasons: it is one of four positions tested, so under the null you would
expect one to cross 97.5% about 10% of the time, and its interval touches zero at the top.
Shipping the one slice that worked is the failure mode this repo screens to avoid.

The likely reason shrinking hurts skill positions: shrinking toward a single positional mean
over-shrinks the studs. A WR1 and a WR5 do not regress toward the same place, and the market
already prices that. A volume model that shrank toward an *ADP-implied* prior rather than a
positional average is a different and more promising thing, and is not what was tested here.

**So `project()` keeps carrying volume forward and regressing touchdowns**, which the screen
says is the best of the options tried.

## What is still missing

**Correlation between players.** Every draw here is independent. A quarterback and his WR1
share touchdowns, and that term is exactly what
[championship-leverage.md](championship-leverage.md) calls L1 and what a points-level model
structurally could not express. Component-level sampling is the precondition for it; it is
not built.

**Within-game correlation between volume and efficiency**, per the skew gap above.

## Reproduce

```bash
uv run python -m hub.fetch.nflverse --refresh --season 2025
```
