# The market/Usage blend, gated once

**Run 2026-08-29**, under [ADR-0017](adr/0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md),
which fixed every rule and was committed *before* the run — the commit order is the record.

**2,000 roster-weeks over 160 rosters, four held-out seasons, 79 covered weeks.
Join failure 0.1% against a 2% floor.**

## Restated 2026-09-07 — the effect changes sign and the verdict reverses

The arm under test was scoring on a column holding two incommensurable things: negated ranks
for players it could not project, and fantasy points for those it could. Any player carrying a
projection therefore outranked every player carrying only a rank, whatever either was worth.
`7873de6` fixed that and said so plainly — *"the frozen +0.215 will move, and re-running it is
the next action, not something this commit may claim."* This is that re-run, issue #44.

Same command as the Reproduce section below, unchanged: five seasons supplied, forty drafts,
`--shrink mae-market`. Same shape as the original — 2,000 roster-weeks over 160 rosters on the
79 weeks consensus covers.

| | superseded | re-run 2026-09-07 |
|---|---|---|
| weekly − consensus | **+0.215** | **−1.004** |
| 95% CI | [−0.242, +0.684] | **[−1.391, −0.621]** |
| seasons won | 3 of 4 | **0 of 4** |
| P(weekly better) | — | **0.0%** |
| verdict | SHOW, NEVER RANK ON | **REMOVE** |

Per season: 2022 −0.452, 2023 −1.490, 2024 −0.868, 2025 −1.204. Worse in every held-out
season, and the interval no longer straddles zero.

**What moved, and why it moves this far.** The mixed-scale column was not noise — it was a
systematic advantage to the arm under test, because carrying a projection at all was worth
more than any amount of being ranked well. Removing it does not shrink the effect toward zero;
it takes the effect through zero, because the advantage was the effect.

**What this re-run cannot separate, measured 2026-09-07.** The fix replaced the mixed-scale
fallback with rank-interpolated points, which is a *new estimator* sitting inside the arm under
test. The arm's score column is a mixture: 54.2% model projection, 16.7% rank-interpolated,
29.1% unscoreable. Re-scoring with the interpolation stripped out — an unprojected player simply
unscoreable — gives **+0.537 [+0.133, +0.939]**, so the fallback is worth **−1.546** to the arm.

| how unprojected players are handled | effect |
|---|---|
| mixed scale (superseded) | +0.215 |
| rank-interpolated (this re-run) | −1.004 |
| unscoreable | +0.537 |

> **This table was maintained by hand and is not any more (2026-09-07, #207).** Every figure in
> it was obtained by probing the arm from outside the gate and typing the answer in here. The
> gate prints the shares and all three effects itself now — see *the gate prints the spread*
> below, which carries its block rather than a second copy of these numbers.

**The gate is measuring the fallback, not the weekly model.** The spread across three choices
is 1.5 points, larger than any effect any of them reports. And no row above is a clean
measurement of the arm: the mixed scale handed it a preference unrelated to either estimate;
the interpolation makes it carry that estimator's error; and *unscoreable* hands it a free rule
against starting players it cannot price, while consensus still ranks and starts them — the
"arms see different universes" defect `_one_scale`'s own docstring names.

What the re-run does establish is that **+0.215 was not a property of the weekly model**, and
that the verdict is not robust to a choice nobody has justified. What it does not establish is
the weekly model's own merit, in either direction.

**The disposition is not applied here.** The pre-registered rule says REMOVE, and REMOVE means
deleting the module rather than shipping it as an option. That is a product decision and it is
recorded as pending, not taken — the same treatment the draft gate's REMOVE received.

The rows below are kept as published. They are superseded, not wrong at the time.

## Restated 2026-09-07 — the unit becomes the season, and the percentile interval narrows

Issue #45 moved every gate's resampling unit from the roster to the **season**. This gate
clustered on `("season", "roster")`; it now clusters on `("season",)`, so its interval is over
four units rather than 160. `docs/method.md` rule 13: a dated restatement beside the original,
not an edit over the top of it.

**The argument is the one that already moved this gate once.** A roster's fourteen weeks share
its players, its bye and its draft — which is why resampling rows was wrong. A season's forty
rosters share a board, a player pool, a schedule and one realisation of the year, which is the
same argument one level up. The previous correction stopped a level short.

| | re-run 2026-09-07 (roster-clustered) | **restated (season-clustered)** |
|---|---|---|
| weekly − consensus | −1.004 | **−1.004** |
| clusters | 160 | **4** |
| 95% CI, percentile | [−1.391, −0.621] | **[−1.347, −0.640]** |
| 95% CI, t on 3 df | — | **[−1.611, −0.396]** |
| bootstrap SE | — | **0.191** |
| MDE at 80% power | — | **0.768** |
| seasons won | 0 of 4 | 0 of 4 |
| verdict | REMOVE | **REMOVE** |

**How these were obtained, and the limit on them.** Not from a re-run: this gate builds its
paired frame from the network and persists nothing, which is the same reason `gate-power.md`
could not measure it. But a cluster bootstrap resamples the *cluster means*, and under
`("season",)` those are exactly the four per-season gains already published above — −0.452,
−1.490, −0.868, −1.204 — over four balanced seasons. So the season-clustered statistic is
recoverable from the page itself, with the same seed and bootstrap count the harness uses.
The inputs are rounded to three decimals, so these carry about ±0.001; the authoritative
figures still want a harness run, and a frozen paired frame for this gate remains the right
answer.

**The mean does not move**, and that is the check that the clustering changed the claim rather
than the estimate: the seasons are balanced, so the mean of the season means is the mean of
the rows. −1.0035, which is the −1.004 already published.

**The verdict does not move.** All four season means are negative, so no resample of them can
reach zero; the interval excludes zero in the same direction, 0 of 4 seasons still lose, and
the pre-registered rule still says REMOVE. As before, the disposition is recorded as pending
rather than taken.

### The percentile interval got *narrower*, and that is the flagged case

`[−1.347, −0.640]` is **0.707** wide against the roster-clustered `[−1.391, −0.621]`'s
**0.770** — a ratio of **0.92**. Clustering on a coarser unit is supposed to widen. This run
therefore **requires review** under #45's criterion 5, which exists because the weekly screen
(#169) hit exactly this on 2026-09-07: five of nine intervals narrowed under clustering and it
was noticed only because someone compared.

**It is not a defect here, and the reason is the other half of the same ticket.** A
nonparametric percentile bootstrap over four units resamples four numbers: its draws are means
of multisets drawn from a space of 256, and it cannot express a tail it never drew. It
under-covers, and it under-covers *narrowly* — which looks exactly like precision. The t
interval on the same four units is `[−1.611, −0.396]`, **1.215** wide, which is 1.58x the
roster-clustered interval and the widening the argument predicts.

So the percentile interval narrowed and the honest interval widened, on the same data, in the
same run. That is precisely why criterion 3 prints both at eight clusters or fewer and why
criterion 5 makes a narrowing announce itself instead of passing as a tighter result.

**Against the ceiling.** The MDE is **0.768** against a reported effect of −1.004, so this
gate can resolve the effect it reports — stage 1, on the weak comparator. Stage 2 needs this
gate's **foresight ceiling**, which has never been measured: nothing here hands `summarise` a
ceiling, so the NOT-RUNNABLE branch does not fire and cannot. What this restatement establishes
is that the question is now askable of this gate, not that it has been answered.

## Restated 2026-09-07 — the gate prints the spread, and this page stops keeping it

Issue #207. Two sections up, this page carries a three-row table and a mixture — 54.2% model
projection, 16.7% rank-interpolated, 29.1% unscoreable — that no run ever printed. Both were
obtained by probing the arm from outside and typed in here, so the shipped gate reported one of
the three numbers and the 1.5-point spread across them lived only on this page. It could go
stale against the code without anything noticing, which is the failure mode
[method.md](method.md) rule 13 exists downstream of.

The gate now emits this with every run, off the same paired frame the verdict is read from and
under the same seed and the same `("season",)` cluster #45 fixed:

```text
  the arm's score column over 2000 roster-week cells: 54.2% model projection, 16.7%
  rank-interpolated, 29.1% unscoreable

  the same rows under each treatment of the fallback, same seed and same cluster -- the
  verdict below is the primary's, and #206 is where that was chosen:
    rank-interpolated (primary)  -1.004  95% CI [-1.347, -0.640]
    unscoreable (comparison)     +0.537  95% CI [+0.133, +0.939]
    mixed scale (superseded)     +0.215  95% CI [-0.242, +0.684]
  spread across treatments 1.541 points per team-week -- larger than any effect any of them
  reports (1.004). How much of the verdict is the fallback.
    rank-interpolated: reads off nothing but the week's own paired observations, so neither
      arm gets information the other lacks -- the only one of the three not disqualified,
      which is not the same as correct (#206)
    unscoreable: benches every player the arm cannot price while consensus still ranks and
      starts him -- a free lineup rule the arm did not earn, which is the
      different-universes defect `_one_scale` names
    mixed scale: negated ranks and fantasy points in one column, so carrying a projection at
      all beat being ranked well; removed by #44 and scored here only because it is the
      treatment the published +0.215 was measured under
```

No new flag: it is the published command in *Reproduce* below, unchanged. The cost is that a
run scores three arms against the incumbent rather than one — the primary's frame is the one
the verdict is read off and is not re-scored, so it is two extra scorings and not three.

**Which of those figures is a run and which is a transcription.** The block above is the
formatter over the 2026-09-07 measurements, not a captured run: this gate builds its paired
frame from the network and persists nothing, the same limit the section above records for the
season-clustered restatement. So:

| line | standing |
|---|---|
| the three shares | as measured 2026-09-07, and now what the gate counts off its own column |
| the three effects | as measured 2026-09-07. They do not move under #45's re-clustering — the seasons are balanced, so the mean of the season means is the mean of the rows, which is the check the restatement above already made for −1.004 |
| the primary's interval | the **season-clustered** one restated above |
| the two comparison intervals | the **roster-clustered** ones as first published. Nothing has re-clustered them, and a run prints season-clustered intervals for all three, so these two will move when one is taken |
| the spread, and *larger* | computed by the gate from the three means in front of it, not carried |

**Nothing here picks a treatment.** Rank-interpolation is primary, that was decided in #206 on
2026-09-07 as a pre-registration, and the verdict line below the block is still the primary's
alone. What this changes is that a reader of the output can see how much of that verdict is the
fallback without leaving the terminal. The limitation the primary carries — that the arm is
scored on a second estimator's error — is unchanged and is why #206 stayed open.

## The result

| gate | weekly − consensus | 95% CI | seasons won | verdict |
|---|---|---|---|---|
| **frozen** *(primary)* | **+0.215** | [−0.242, +0.684] | 3/4 | **SHOW, NEVER RANK ON** |
| churn | −1.806 | [−2.710, −0.921] | 2/4 | SHOW, NEVER RANK ON |

> **Interval restated 2026-09-04, and now reproducible.** The CI above was
> [−0.249, +0.659] as first published and [−0.251, +0.663] on a re-run of the identical
> command — the *mean* was bit-stable at +0.215 every time, but the interval drifted. The
> cause: `cluster_bootstrap` took its clusters in `.unique()` order and the bootstrap indexes
> into that order, so a permutation of the same cluster means moved the percentiles while
> leaving their average alone. Same defect as [#18](improvements.md), one layer down. Clusters
> are now sorted before resampling (`hub.models.experiment.summarise`), and the interval is
> **[−0.242, +0.684]** every run. Nothing about the verdict, the seasons won, or the decay
> moves — an interval that contained zero still contains it.

> **Restated 2026-08-30 under a reproducible board.** This ran first at **+0.711
> [+0.313, +1.129]** and −2.006, and those figures are superseded rather than wrong-at-the-time:
> `board_as_of` was **not reproducible** when they were measured
> ([improvements.md #18](improvements.md)), so the roster sample they drew was one of many the
> same command could produce. With the board fixed — it now sorts on `(ecr, player)` and its
> DvP aggregation no longer hands a hash-ordered frame to a mean — the same command returns
> +0.215 every time.
>
> **The verdict does not move**, and neither does anything the write-up concluded from it: the
> every-season half still fails, 2025 is still the season that loses, and the interval now
> *contains* zero rather than excluding it, which makes the result weaker rather than
> differently-shaped. What changed is that it is now a number that can be re-run.

Under the reproducible board the interval **contains** zero, so the headline that first ran
here — *"the interval excludes zero for the first time in the programme"* — does not survive.
It did not adopt then and does not now, because the bar has two halves and the every-season
half fails: 2025 loses either way.

## The thing that matters most is the trend

| season | frozen gain (as first run) | under the reproducible board | |
|---|---|---|---|
| **2022** | +1.894 | **+0.983** | **never scored before — the one out-of-sample season** |
| 2023 | +0.961 | +0.027 | |
| 2024 | +0.487 | +0.355 | |
| **2025** | −0.497 | **−0.504** | the season closest to the one being drafted |

**The edge decays monotonically and is negative in the most recent season.** Whatever this is,
it was worth two points a team-week in 2022 and is worth nothing now. For a 2026 draft that is
the decision-relevant fact, and it points the same way as the verdict.

No mechanism is asserted for the decay — [signal-screens.md](signal-screens.md) point 6, which
this repo got wrong once already. The obvious candidate is that weekly consensus improved; it
is untested.

## Against the pre-registration

| | pre-stated | measured | |
|---|---|---|---|
| frozen | positive but small, **interval contains zero** | +0.711, **interval excludes zero** | direction right, precision wrong |
| churn | between −1.5 and −3 | **−2.006** | right |
| 2022 vs the rest | in line | +1.894, agrees with 2023–24 | right; 2025 is the dissenter |

The interval tightened partly because rosters were doubled to 40 — pre-registered for exactly
that reason, before the numbers.

**No tripwire fired.** Join failure 0.1%; 2022 did not disagree with the other three as a
block; the churn gate did not come out positive.

## What this settles

ADR-0017 fixed the consequence of each outcome before the run:

> **SHOW**: ADR-0016 stands, the blend is printed beside consensus, and **the rescue attempts
> end.** Five variants is where a negotiation with a result becomes a search for one.

So: [ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) stands. Nothing
sets a lineup. **The weekly programme is closed.**

What it produced, in order: four screened signals, a component model that beats the flat
projection by +0.074 MAE at 5.9 se, a lineup gate it loses, a winner's curse diagnosed at the
waiver pool, and a market/Usage blend that beats a free public ranking by +0.711 points a
team-week on average and is going backwards. Fifteen measurements, and consensus has won
fourteen and a half.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run --seasons 2021,2022,2023,2024,2025 \
  --drafts 40 --shrink mae-market
uv run python -m hub.season.weekly_gate --run --seasons 2021,2022,2023,2024,2025 \
  --drafts 40 --churn --shrink mae-market
```
