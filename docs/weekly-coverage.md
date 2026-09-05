# Weekly interval coverage: the shape is right, and the floor explains the rest

**Measured 2026-09-05**, over 22,571 player-weeks from 2021-2025. This tests the half of the
weekly forecast that no gate has ever scored.

`hub.models.predict.moments` returns three moments — `mu`, `sd = K[position]·sqrt(mu)`, and a
fitted Cornish-Fisher `skew`. Two of the three are load-bearing: `season/lineup.py` chooses a
lineup by maximising P(beat your opponent) from `(mu, sd)`, and `draft/optimize.py` draws
skewed correlated normals from all three. Every gate then scores **MAE**, which is a proper
scoring rule for the median and cannot see either of them. `component_error.py` states the
reason plainly — "fantasy points are linear, so MAE is the loss that matters" — and that is
true about the *quantity*. It leaves the spread and the skew ungraded:

> A wrong mean is observable and a wrong spread is not. If the model says 14 and he scores 4,
> you notice. If the model says ±7 when it is really ±11, no single week announces it.

So the question is not whether MAE is the right loss. It is whether the thing MAE cannot see
is wrong. It is not.

## Result

Centre is each player-season's own realised mean, which isolates the *shape* of the weekly
distribution from projection error. Weeks are weeks he recorded stats; players with at least
8 such weeks and a mean of at least 2.0 points.

| pos | weeks | 80% cov | 68% cov | < p10 | > p90 | 80% cov, no skew | sd realised/model |
|---|---|---|---|---|---|---|---|
| QB | 2,473 | 79.9% | 69.0% | 10.5% | 9.5% | 80.3% | 0.97 |
| RB | 5,933 | 83.4% | 71.4% | 6.2% | 10.4% | 84.9% | 0.99 |
| WR | 9,565 | 84.5% | 73.1% | 5.6% | 9.9% | 85.7% | 0.97 |
| TE | 4,600 | 85.7% | 74.1% | 5.0% | 9.4% | 86.7% | 0.96 |
| **all** | **22,571** | **83.9%** | **72.4%** | **6.2%** | **9.9%** | **85.1%** | **0.97** |
| *nominal* | | *80.0%* | *68.0%* | *10.0%* | *10.0%* | *80.0%* | *1.00* |

Two things read straight off it. The **scale law holds**: realised sd over model sd is 0.97,
so `K[position]·sqrt(mu)` is about 3% wide and no more. And the **skew earns its place**:
deleting it moves 80% coverage from 83.9% to 85.1%, away from nominal, so the term is pulling
in the right direction.

The residual miss is entirely one-sided. The upper tail is calibrated at 9.9% against a
nominal 10%; the lower tail holds only 6.2%. And it is worst at TE and absent at QB, which is
the clue.

## The part that actually closes the question

`skewed()` clips at zero — deliberately, because "nobody scores negative points often enough
to matter". For a low-`mu` player the 10th percentile lands at or below zero, gets clipped to
zero, and then *nothing can fall below it*. The deficit is structural, not a modelling error.

Splitting on whether the model's own p10 survives the floor:

| model's p10 | weeks | < p10 | 80% cov |
|---|---|---|---|
| clipped at zero | 7,828 | 1.0% | 89.3% |
| strictly positive | 14,743 | **8.9%** | **81.1%** |
| *nominal* | | *10.0%* | *80.0%* |

**Among the 65% of player-weeks whose interval is not pinned at the floor, the shipped
distribution is calibrated to about one percentage point.** That also explains the position
pattern: QB is the position whose means are high enough that few of its intervals are clipped,
and QB is the position that was already calibrated in the full sample.

## What this does to the CRPS question

`docs/improvements.md` and the 2026-09-04 plan both carry CRPS as deferred work, on the
argument that a distributional model graded by MAE is graded on one moment. The argument is
sound and this measurement does not retire it. What it retires is the *expected payoff*: the
moments MAE cannot see are already within a point of nominal, so re-scoring the weekly gate
under CRPS would be unlikely to flip an adopt-or-reject decision that MAE got wrong.

This is the `docs/player-spread.md` move, and it is the second time the same instrument has
closed a question here — compute what the ceiling permits before running candidates against
it. A candidate failing is weak evidence; a ceiling is strong evidence.

## What is deliberately not concluded

- **Not "the shipped interval covers."** The centre here is the player's own realised mean,
  which is lookahead. A production interval also carries projection error, which is larger and
  is measured elsewhere. This says the shape is right, not that the interval is.
- **Nothing about absence.** Weeks are weeks he recorded stats, so byes and injuries are out of
  the sample entirely. `draft/season.py` still models a missed season as a low-mean,
  low-variance player rather than as zeros, and that remains open.
- **Nothing about sharpness.** Coverage says the intervals are the right *width*, not that the
  model tells a volatile player from a steady one. By construction it cannot: `sd` is a
  function of `mu` alone. `docs/player-spread.md` closed that separately, at ±9.3% of true
  variation and 0.085 MAE of total headroom.
- **Only the weekly player distribution.** Survivor win probability by spread bucket, the
  margin model's behaviour at the key numbers 3 and 7, and `TALENT_CV`'s season-level
  dispersion are three further distributions, none of them tested here, and the first two are
  where the audit expects a miss.

## Reproduce

No committed harness. ADR-0007 requires committed code for a measurement that *steers* the
product, and this one steers nothing — it declines to change anything. If a future result
turns it into a decision, it needs to become code first.

Load `player_stats` for 2021-2025 through `hub.fetch.nflverse`, keep regular-season rows at
QB/RB/WR/TE, and group by (player, season). For each group with at least 8 rows and a mean of
at least 2.0: take `mu` as the group mean, `sd = WEEKLY_K[pos]·sqrt(mu)`, and
`skew = WEEKLY_SKEW[pos]`. Build the interval bounds by evaluating
`predict.skewed(mu, sd, skew, z)` at `z = Φ⁻¹(p)` for the p of interest — the transform is
monotone in `z`, so that is the quantile — and count realised weeks inside. For the floor
split, recompute the p10 without the clip and test whether it is positive.
