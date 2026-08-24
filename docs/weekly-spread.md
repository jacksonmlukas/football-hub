# Fitting the weekly spread

**Fitted 2026-08-23.** `sd = 0.55 × mu` is replaced by **`sd = k × √mu`**, with
k = {QB 1.88, RB 2.07, WR 2.13, TE 1.99} and 2.04 pooled.

This is the last of the model's guessed constants, and it is the one that came back with a
different *shape* rather than a different number.

## The assumption was the wrong form, not the wrong value

`weekly_moments` assumed weekly spread is proportional to the mean. Fitted across 1,174
player-seasons of nflverse weekly scoring (2022-25, ≥8 games, >3 ppg), the exponent in
`sd = k·mu^b` is:

| | b | se(b) |
|---|---|---|
| QB | 0.161 | 0.067 |
| RB | 0.469 | 0.024 |
| WR | 0.518 | 0.019 |
| TE | 0.622 | 0.032 |
| **pooled** | **0.498** | **0.012** |

**0.498 ± 0.012.** That is the Poisson value, and about 42 standard errors from the
assumed 1. Predicting a player's weekly sd with `k√mu` cuts RMSE by **31-36%** against the
old constant, at every position.

QB is the exception at 0.161, and it should be read with suspicion rather than believed:
quarterbacks who start a full season occupy a narrow band of about 11-20 ppg, so the
exponent is barely identified there. Pinning b at ½ costs QB almost nothing in fit.

## Why ½ is the right answer and not a curve-fitting accident

Because fantasy points are an aggregate of component stats. Receptions, carries and
touchdowns are counts; a count's variance grows with its mean, so a sum of them has a
spread that grows with the *square root* of its mean. The exponent is not something to fit
and hope about — it is what the structure implies, and the data returning 0.498 is the
structure being confirmed.

Which components do the damage, as a share of a player's weekly variance:

| position | volume | touchdowns | covariance | TD share of *points* |
|---|---|---|---|---|
| QB | 22.1% | **53.8%** | 25.3% | 31.6% |
| RB | 44.8% | 29.4% | 24.5% | 21.0% |
| WR | 55.5% | 20.6% | 21.8% | 16.7% |
| TE | 52.8% | 27.1% | 17.8% | 17.4% |

Touchdowns are consistently over-represented in variance relative to their share of points
— 54% of a quarterback's variance from 32% of his points. That is the lumpiness the old
constant was averaging over.

## What it changes

**Relative volatility is now a property of the projection, not a setting.** Implied weekly
CV:

| ppg | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| WR, fitted | 0.95 | 0.67 | 0.55 | 0.48 |
| old constant | 0.55 | 0.55 | 0.55 | 0.55 |

A 5-point-a-game flier really is nearly twice the lottery, per point, that a 20-point
starter is. The old constant said they were identical, which is what made
[six-of-twelve.md](six-of-twelve.md)'s weekly-variance sweep a sweep in a quantity nobody
could actually buy.

**Most of the position difference was never a position difference.** Raw weekly CV runs
QB 0.47 / RB 0.68 / WR 0.71 / TE 0.73, which looks like a large positional effect. Once the
√ law is in, k is 1.88-2.13 across all four. The spread in CV was mostly positions differing
in *mean points*.

**The floor is gone.** `sd` had a floor of 2.0, so a player projected at nothing still
carried real weekly spread — and the best-lineup rule is a max over the roster, which turned
that into free points off the end of the bench. √0 is 0.

**`simulate_weeks` scales spread by √(realised/projected)**, not by the ratio. A player who
realises a quarter of his projection keeps half his weekly spread.

**`TALENT_CV` was refitted**, because the two are coupled: that fit subtracts weekly
sampling, and the amount to subtract just changed. Pooled moved 0.41 → 0.42 and the
per-position values by about 0.01 — all well inside their intervals, so nothing downstream
turned over. One detail worth recording: quarterbacks previously fitted oddly low and
[talent-cv.md](talent-cv.md) flagged 0.55 as the likely culprit. Under the √ law QB lands at
−0.0 se from the pool. The anomaly was the wrong weekly law, as suspected.

Rerunning [six-of-twelve.md](six-of-twelve.md) under both new laws leaves every conclusion
standing: dP(title)/d(win) +4.2 pp at the median, P(playoff) 50.1% for an average roster,
seed 1 at 41.8% against seed 3 at 12.3%.

## This is the first payoff of projecting components, not points

The direction is right and the next step is bigger than this one. Weekly *spread* now comes
from the component structure. Weekly *mean* still comes from a points projection handed over
whole.

Doing the same thing to the mean — project targets, carries, catch rate, yards per
touch and touchdown rate, then aggregate through the scoring rules — should pay for two
independent reasons:

- **Volume persists and touchdowns do not.** Targets and carries are far more predictable
  year over year than touchdowns, which are close to a rate times noisy volume. Projecting
  the persistent parts and letting the lumpy part be lumpy beats projecting their sum.
- **It gives the whole distribution, not two moments.** The joint sampler that
  [championship-leverage.md](championship-leverage.md) calls L1 needs component-level
  correlation — a quarterback and his WR1 share touchdowns, which is precisely the term a
  points-level model cannot express.

`hub.fetch.nflverse` now carries `player_stats` (weekly, per player, with every component),
and full-PPR points reconstruct from those components to within 0.01 for 99.4% of
player-weeks — so the aggregation is exact and the data is in place.

## What is still wrong with it

**Measured on games played, and on scorers.** The sample is player-seasons with at least 8
games and more than 3 ppg. Fringe players are excluded, and their spread is the least
well-described by any of this.

**Within-season trend counts as spread.** A player whose role grows across a season shows
that growth as weekly variance. Some of k is really usage drift.

**Normal, still.** The model draws weekly points from a normal. Real weekly scoring is
right-skewed — that is what the touchdown share of variance above is telling you — and a
normal with the right spread still puts the mass in the wrong places. Component aggregation
would fix this too, since it would sample the counts.

## Reproduce

```bash
uv run python -m hub.fetch.nflverse --refresh --season 2025
```
