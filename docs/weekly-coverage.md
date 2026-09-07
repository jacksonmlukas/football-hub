# Weekly interval coverage: the shape is right, and the floor explains the rest

> **This title is superseded.** The measurement below is centred on the player's own realised
> mean, which is a lookahead, and its headline does not survive removing it. The original text
> is kept whole; the restatement of 2026-09-07 sits directly above the result table it
> supersedes.

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

> **Restated 2026-09-07, on a centre that cannot see the week it scores, and now committed
> code.** Everything below this line is the run of 2026-09-05 and is kept verbatim. Its
> numbers reproduce exactly — `hub.models.coverage --measure --centre realised` returns
> 79.9 / 83.4 / 84.5 / 85.7 and a pool of 83.9%, the same floor split, and the same
> `85.1%` with the skew deleted, on 22,586 player-weeks against the 22,571 first measured
> (nflverse has revised fifteen 2021–25 receiver weeks in since). The one column that does
> not reproduce is `sd realised/model`: this document does not say which estimator produced
> 0.97, and the harness's — per player-season sample sd of the residual over its mean model
> sd, averaged over player-seasons — returns **0.99**.
>
> **What does not survive is the centre.** This document names its own lookahead under
> *What is deliberately not concluded* and then draws a conclusion that depends on it. Run
> with the centre built from that player's **strictly earlier** weeks only — no information
> from the week being scored — the same shipped moments over the same five seasons give:
>
> | | weeks | 80% cov | 68% cov | < p10 | > p90 |
> |---|---|---|---|---|---|
> | **all** | 16,061 | 79.8% | 67.7% | 7.8% | 12.4% |
> | clipped at zero | 5,525 | 84.5% | 74.2% | 0.9% | 14.6% |
> | **strictly positive** | **10,536** | **77.4%** | **64.3%** | **11.4%** | **11.2%** |
> | *nominal* | | *80.0%* | *68.0%* | *10.0%* | *10.0%* |
>
> **Three things move, and the headline is one of them.**
>
> 1. *"Among the 65% of player-weeks whose interval is not pinned at the floor, the shipped
>    distribution is calibrated to about one percentage point"* — **superseded.** Those weeks
>    cover at **77.4%**, 2.6 points under nominal, not 81.1%.
> 2. *The residual miss is entirely one-sided* — **superseded as a property of the model.**
>    On the unclipped weeks both tails now over-shoot and they over-shoot by the same amount
>    (11.4% low, 11.2% high). The one-sidedness was the lookahead centre and the zero clip
>    together; the clip's own effect is unchanged and still visible in the clipped row.
> 3. The pooled 79.8% at the top of that table is a **cancellation, not a pass.** Its two
>    tails are 7.8% and 12.4% against a nominal 10% each — the floor holds the lower one down
>    by as much as the upper one runs hot — and they sum to 20.2% against 20%. Read the
>    headline alone and this is a calibrated interval; read the tails and it is an interval
>    in the wrong place. This is why the gate reads the unclipped split and not the pool.
>
> **Not small-sample noise in the centre.** Splitting the unclipped weeks by how many earlier
> weeks the centre was built from: 75.5% at 4–5 prior weeks, 77.6% at 6–8, 77.9% at 9–12,
> **78.0% at 13+**. It improves and then plateaus about two points short. Some of the gap is
> the centre's own estimation error, which a production interval carries and `sd =
> K·sqrt(mu)` has no term for; the part that survives at 13+ prior weeks is not that.
>
> **What still stands.** The scale law and the correction term both survive: pooled realised
> over model spread is 0.99 on the realised centre and 0.98 on the prior one, and deleting the
> skew moves the pool *away* from nominal (79.8% → 81.3%) exactly as it did before. The zero
> clip is still the reason the clipped weeks report 84.5%. So the miss is in the interval's
> *width in the tails*, not in the law that sets its scale.
>
> **What it costs.** The CRPS argument below turned on the moments MAE cannot see being
> "already within a point of nominal". On a centre that is allowed to be wrong they are 2.6
> points under it, so the *ceiling* that section computes is not the ceiling. That paragraph
> is superseded with the headline; whether CRPS is worth running is open again.
>
> The gate now refuses on this: `uv run python -m hub.models.coverage --gate` exits 1, and
> `hub.publish` carries the verdict in `track_record.json`. See
> [the survivor price](#the-survivor-price-measured-2026-09-07), below, for the second
> distribution this document listed as untested.

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

## The survivor price, measured 2026-09-07

The last of the three distributions listed above as untested, and the one the audit expected
a miss in. `season/survivor.py` prices every pick as `normal_cdf(close_spread / MARGIN_SD)`
with `MARGIN_SD = 12.741`, and multiplies those into the survival probability it prints.
Graded by **spread** bucket rather than by probability bucket — a survivor pick is chosen by
spread, so a miss concentrated in one spread range is what a pick rule walks into — over
3,018 completed games from 2015–2025, both sides of each, ties dropped by `margin.home_won`:

| home spread | sides | predicted | actual | gap |
|---|---|---|---|---|
| 0 to 3 | 703 | 0.557 | 0.522 | −0.035 |
| 3 to 6 | 1,182 | 0.616 | 0.621 | +0.005 |
| 6 to 9 | 704 | 0.708 | 0.749 | **+0.041** |
| 9 to 14 | 332 | 0.802 | 0.837 | **+0.035** |
| 14+ | 101 | 0.884 | 0.901 | +0.017 |
| **favourites of 7+** | **859** | **0.771** | **0.803** | **+0.032** at 2.4 se |

**The price is under-confident exactly where survivor picks.** A 7-point-or-better favourite
is priced at 77.1% and wins 80.3%. Two things follow, and only the second is comfortable:
the printed survival probability for a plan of big favourites is **understated**, which is
the safe direction to be wrong in; and the *shape* is wrong, because the same normal is
over-confident on pick'em games (−3.5 points, 1.8 se, not significant on its own) and
under-confident on favourites. A single `MARGIN_SD` cannot fix both ends at once — a smaller
one closes the favourite gap and widens the pick'em one.

Nothing is refit on this. It is a finding with a direction and a magnitude, and the pick rule
is unchanged until something gates it.

## Reproduce

The harness is `hub.models.coverage`, committed under ADR-0007 because `--gate` now refuses
on the result and `hub.publish` carries it. It calls `predict.moments` and `predict.skewed`
rather than restating the laws behind them, and `tests/unit/test_coverage.py` holds it to
that by moving `WEEKLY_K` and requiring the graded table to move with it.

```bash
uv run python -m hub.models.coverage --measure                    # the real one
uv run python -m hub.models.coverage --measure --centre realised  # this document's
uv run python -m hub.models.coverage --survivor --seasons 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025
uv run python -m hub.models.coverage --gate                       # exits 1 today
uv run python -m hub.models.coverage --measure --write            # for the publisher
```

*(Superseded, kept as the record of how this was first run.)* No committed harness. ADR-0007
requires committed code for a measurement that *steers* the product, and this one steers
nothing — it declines to change anything. If a future result turns it into a decision, it
needs to become code first.

Load `player_stats` for 2021-2025 through `hub.fetch.nflverse`, keep regular-season rows at
QB/RB/WR/TE, and group by (player, season). For each group with at least 8 rows and a mean of
at least 2.0: take `mu` as the group mean, `sd = WEEKLY_K[pos]·sqrt(mu)`, and
`skew = WEEKLY_SKEW[pos]`. Build the interval bounds by evaluating
`predict.skewed(mu, sd, skew, z)` at `z = Φ⁻¹(p)` for the p of interest — the transform is
monotone in `z`, so that is the quantile — and count realised weeks inside. For the floor
split, recompute the p10 without the clip and test whether it is positive.
