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
