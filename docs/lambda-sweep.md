# projection_lambda: six holdouts, and the answer

Phase 2.4 of `docs/foundation-plan.md`. Run 2026-08-23: one season pair, then three, then
six. The answer never changed. The reasoning changed twice, and both revisions are recorded
because they are the point.

**Result: `projection_lambda = 0.0`.**

> **Metric corrected and the sweep re-run, 2026-09-04. The result did not move.** The Spearman
> in `tune.score` was computed with `actual.argsort().argsort()`, which gives tied values
> distinct sequential ranks instead of the mean rank Spearman requires. `actual_points` is
> `fill_null(0.0)`, so every player with no recorded production is an exact tie -- on a
> 450-row board with 180 zeros that understated a true rho of +0.650 as +0.563, and returned
> +0.623 for the same data in a different row order. Since `sweep` re-sorts the board by
> `adj_ecr` for each lambda, every lambda was scored with its own arbitrary tie-breaking and
> then compared against the others.
>
> Re-run with average ranks (`tune.average_ranks`): **`lam = 0.00` is still selected**, on
> both metrics. The conclusion below is unchanged, and the reason it was robust is visible in
> the tables -- they report Spearman *deltas* between lambdas, and a bias shared by every
> lambda largely cancels in the difference. The absolute rho was wrong; the ordering it was
> read for was not.

## The question

`hub.draft.projection` nudges the consensus board by last season's expected-vs-actual gap:

```
adj_ecr = ecr * exp(-lam * z)
```

Rankers anchor on realised fantasy points, which bake in touchdown variance that does not
repeat. ffopportunity gives expected points from opportunity alone, so the gap is a signal
consensus should in principle underweight. `lam` was 0.08 by judgment.

## Method

- **Signal:** season N expected-vs-actual gap, standardised within position.
- **Board:** the last FantasyPros redraft snapshot before season N+1 opened — what a
  drafter would have had in hand. Window bounded on both sides so no stale rank carries
  forward.
- **Truth:** season N+1 realised PPR points.
- **Metric:** top-50 points captured, bootstrapped over 300 resamples per season.

**Page taxonomy caveat.** FantasyPros used `redraft-offense` through 2020 and
`redraft-overall` from 2021. The two never coexist in any preseason, so there is no overlap
year in which to verify they are the same board. The 2019→2020 row therefore is not
strictly interchangeable with the rest, and every pooled figure below is given with and
without it.

## Six holdouts

Bootstrap mean delta in top-50 points against the untouched consensus board.

| lam | 19→20 | 20→21 | 21→22 | 22→23 | 23→24 | 24→25 |
|---|---|---|---|---|---|---|
| 0.00 | — | — | — | — | — | — |
| 0.02 | +10 | +22 | −15 | +11 | +71 | −15 |
| 0.04 | −14 | −10 | −37 | −22 | +127 | −36 |
| 0.06 | −9 | −2 | −74 | −48 | +162 | −72 |
| **0.08** | **−62** | **−13** | **−110** | **−87** | **+184** | **−94** |
| 0.12 | −147 | −41 | −107 | −78 | +233 | −169 |
| 0.16 | −202 | −34 | −170 | −38 | +213 | −166 |
| 0.24 | −293 | −96 | −222 | −80 | +122 | −193 |
| 0.32 | −300 | −216 | −378 | −105 | −283 | −253 |

Best lambda per season: **0.00, 0.02, 0.00, 0.02, 0.12, 0.00**.

### Pooled

| lam | mean | sd | t | seasons negative |
|---|---|---|---|---|
| 0.02 | +14.1 | 31.6 | +1.09 | 2/6 |
| 0.04 | +1.1 | 62.5 | +0.04 | 5/6 |
| **0.08** | **−30.4** | **110.2** | **−0.68** | **5/6** |
| 0.16 | −66.6 | 154.9 | −1.05 | 5/6 |
| 0.24 | −127.1 | 145.9 | −2.13 | 5/6 |
| **0.32** | **−255.8** | **91.7** | **−6.83** | **6/6** |

Dropping the legacy-page year leaves the picture unchanged: at 0.08, mean −23.9, t −0.44,
4 of 5 negative.

## What six seasons say

**The direction is consistent; the magnitude is swamped by one year.** At lambda 0.08 five
of six seasons are negative, but 2023→24 is +184 against a spread of roughly −13 to −110,
which drags the pooled t to −0.68. The sign test is the more honest statistic here and it
is not decisive either: 5 of 6 gives p = 0.22.

**2023→24 is an outlier, not a regime.** With three seasons it looked like the signal
worked in some years and not others. With six it is one year in six, and every other season
including the two added last points the same way.

**One finding is unambiguous.** At lambda 0.32 all six seasons are negative, mean −256,
t = −6.83, sign-test p = 0.031. A large nudge is reliably harmful. Spearman also falls
monotonically with lambda in every season without exception.

**Nothing supports a nonzero lambda.** The best value per season is 0.00 or 0.02 in four of
six. No lambda is significantly positive pooled, under any grouping.

## Two corrections, in order

**After one season** I reported the signal as clearly negative and offered an explanation:
that the regression is already priced by preseason, since ffopportunity is free and widely
used.

**After three seasons** that explanation failed. 2023→24 was strongly positive, which a
priced signal would not be. I revised to "the sign is unstable and the spread swamps the
effect" — pooled t was +0.01 across three years.

**After six seasons** the instability reading is also too strong. One positive year in six
is an outlier, not evidence of alternating regimes. The most defensible description is:
*mildly harmful on average, with one anomalous season, and too noisy to call at the 0.05
level either way.* The first reading was closest, and it was reached on the least evidence.

Recording the sequence rather than only the conclusion, because the pattern is the useful
part: each additional pair of seasons moved the interpretation, and the first two readings
were both stated with more confidence than one and three seasons could support.

## What was different about 2023-24

Nothing systematic. It was one torn ACL.

**The metric turns on two or three players a season.** At lambda 0.08 the adjustment moves
only a handful of players across the top-50 boundary, so the whole delta is the points of
those who entered minus those who left:

| pair | swaps | in | out | net |
|---|---|---|---|---|
| 19-20 | 3 | 796 | 704 | +92 |
| 20-21 | 3 | 585 | 781 | -197 |
| 21-22 | 2 | 241 | 331 | -90 |
| 22-23 | 3 | 589 | 483 | +106 |
| 23-24 | 2 | 545 | 309 | **+236** |
| 24-25 | 2 | 255 | 540 | -285 |

**The two swaps in 2023-24:**

| | player | ecr | z | 2024 pts |
|---|---|---|---|---|
| in | Terry McLaurin | 53.0 | +0.32 | 322.5 |
| in | Tee Higgins | 54.0 | -0.26 | 222.1 |
| out | Trey McBride | 51.8 | -0.80 | 243.8 |
| out | **Rashee Rice** | 52.3 | **-2.71** | **64.9** |

Rice had the most negative z on the board, so the adjustment demoted him hardest. He then
scored 64.9 points -- **in three games. He tore his ACL, and his last appearance was week 3.**

The signal did not predict regression. It demoted a player who got injured.

### The counterfactual

Replaying 2023-24 with Rice healthy and everything else untouched:

| Rice 2024 | delta at lam 0.08 |
|---|---|
| 65 pts (what happened, 3 games) | **+183.7** |
| 255 pts (healthy at a modest 15/gm) | +84.9 |
| 368 pts (healthy at his own 2024 rate) | **+26.2** |

About 86% of the only positive season in six is that one injury. At +26 it is
indistinguishable from zero, and the pooled picture stops being ambiguous:

| | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | mean | t |
|---|---|---|---|---|---|---|---|---|
| as it happened | -64 | -13 | -110 | -87 | **+184** | -94 | -30.5 | -0.68 |
| Rice healthy | -64 | -13 | -110 | -87 | **+26** | -94 | -56.8 | **-2.64** |

### The methodological lesson

The bootstrap in `sweep()` resamples all ~430 players in a holdout, which made the precision
look far better than it is. **The decision does not ride on 430 players. It rides on the two
or three sitting closest to the top-50 boundary.** Across six seasons that is roughly fifteen
player-outcomes, and one ACL moves the pooled mean by 26 points and t by nearly two units.

That is the real reason to set lambda to zero, and it is a stronger one than any of the three
readings that preceded it. The signal was never measured at a sample size that could support
a conclusion in either direction. What looked like "mildly harmful with one anomalous year"
is better described as **a coin flip on a handful of boundary players, reported to four
significant figures.**

## Scoring the whole board instead

The top-50 points metric rides on 2-3 boundary swaps. Spearman reads every player, so it
is the properly powered criterion and it settles the question the other metric could not.

**Change in whole-board Spearman against the untouched board, bootstrapped, six seasons:**

| lam | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | mean | t |
|---|---|---|---|---|---|---|---|---|
| 0.02 | -0.0003 | +0.0007 | +0.0005 | +0.0008 | -0.0013 | +0.0003 | +0.0001 | +0.37 |
| 0.04 | -0.0010 | +0.0010 | -0.0012 | +0.0007 | -0.0035 | -0.0002 | -0.0007 | -1.08 |
| **0.06** | -0.0023 | -0.0006 | -0.0033 | -0.0011 | -0.0058 | -0.0015 | **-0.0025** | **-3.18** |
| **0.08** | -0.0038 | -0.0023 | -0.0063 | -0.0035 | -0.0093 | -0.0039 | **-0.0048** | **-4.69** |
| 0.16 | -0.0130 | -0.0123 | -0.0176 | -0.0116 | -0.0250 | -0.0193 | -0.0165 | -7.78 |
| 0.32 | -0.0418 | -0.0469 | -0.0523 | -0.0478 | -0.0747 | -0.0614 | -0.0541 | -11.03 |

**The 2023-24 anomaly is gone.** Under top-50 points it was the one positive season, +184
and the only one selecting a large lambda. Under Spearman it is the **worst** season at
every lambda, and the lambda it selects drops from 0.12 to 0.00:

| season | selected by top-50 points | selected by Spearman |
|---|---|---|
| 2019-20 | 0.00 | 0.00 |
| 2020-21 | 0.02 | 0.04 |
| 2021-22 | 0.00 | 0.02 |
| 2022-23 | 0.02 | 0.02 |
| **2023-24** | **0.12** | **0.00** |
| 2024-25 | 0.00 | 0.02 |

That is the Rashee Rice swap, and nothing else. Remove the top-50 boundary and the season
that looked exceptional becomes ordinary -- worse than ordinary.

### Where full-board Spearman is itself misleading

It is not a clean win, and the reason is worth stating rather than banking the result.

The adjustment is multiplicative, so `exp(-lam * z)` moves a player at pick 400 far further
than one at pick 5 -- by design, and the design is defensible. But full-board Spearman reads
those deep ranks at full weight, and **nobody drafts pick 400.** A large share of the
measured damage is in a region no draft consults.

Restricting to the draftable pool -- the top 192, a 12-team 16-round draft -- the effect
largely vanishes:

| lam | 0.00 | 0.04 | 0.08 | 0.16 | 0.32 |
|---|---|---|---|---|---|
| mean pool Spearman | 0.4529 | 0.4609 | 0.4605 | 0.4607 | 0.4455 |

Flat, with no consistent ordering across seasons. So the honest reading is not "the
adjustment is significantly harmful". It is:

- **No metric selects a lambda above 0.04, in any season.** Best per season is 0.00-0.04
  under Spearman and 0.00-0.02 under points once the Rice swap is discounted.
- **Whole-board Spearman falls sharply**, but a large part of that is the undrafted tail.
- **Draftable-pool Spearman is flat**, which is the most decision-relevant view and it
  shows nothing worth acting on either way.

Three metrics, six seasons, one conclusion: there is no evidence for a nonzero lambda, and
the strongest-looking evidence against one is partly an artefact of measuring ranks nobody
uses.

## Evaluating over simulated drafts

The metric the other two could not be. `hub.draft.evaluate` plays a full sixteen-round
snake per trial: I draft from the lambda-adjusted board, the room drafts consensus with
noise, and the roster is scored on **starters only** -- QB1 RB2 WR3 TE1 FLEX1, because a
surplus player scores zero. Each season is replayed from all twelve seats, paired so every
lambda faces the same opponent draws from the same seat.

There is no boundary for one injured player to sit on, and every pick reads the whole
board.

**Lift in starter points against the consensus board, 144 drafts per cell:**

| lam | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | mean | t | neg |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | -49.0 | +10.3 | -4.7 | -20.2 | +12.3 | +28.9 | -3.7 | -0.33 | 3/6 |
| 0.04 | -108.0 | +4.6 | -0.1 | -24.7 | +8.9 | +43.3 | -12.7 | -0.60 | 3/6 |
| **0.08** | -189.3 | +18.7 | -70.4 | -87.2 | **-20.7** | +77.2 | **-45.3** | **-1.20** | 4/6 |
| 0.16 | -171.1 | -12.9 | -148.6 | -158.5 | -36.3 | +24.3 | -83.8 | -2.41 | 5/6 |
| 0.24 | -124.1 | -128.9 | -251.8 | -227.2 | -77.3 | +29.8 | -129.9 | -3.10 | 5/6 |
| 0.32 | -55.3 | -219.3 | -288.7 | -276.6 | -96.3 | +18.6 | -153.0 | -2.96 | 5/6 |

Best lambda per season: **0.00, 0.06, 0.00, 0.00, 0.00, 0.08** -- four of six select exactly
zero, and the two that do not are the two seasons with positive lift.

**2023-24 is finished as an anomaly.** Under top-50 points it was +184 and selected 0.12.
Played out as actual drafts it is *negative* at every lambda from 0.06 up and selects 0.00.
Drafting from the adjusted board produced fewer starter points even in the season that
looked exceptional.

### The three metrics disagree by season, and that is the finding

| season | top-50 points | whole-board Spearman | simulated drafts |
|---|---|---|---|
| 2023-24 | **+184** (best) | -0.0093 (worst) | -20.7 |
| 2024-25 | -94 (worst) | -0.0039 | **+77.2** (best) |

The same season is the best year under one metric and the worst under another. That is not
three metrics disagreeing about a real effect; it is three views of something too small to
locate. The per-season number is noise whichever way it is measured, and only the pooled
view means anything.

### More simulation cannot help

| draws per season | within-season se | between-season sd | pooled t |
|---|---|---|---|
| 144 | 15.6 | 92.3 | -1.20 |
| 480 | 8.5 | 95.1 | -1.23 |

Tripling the simulation halves the within-season error and moves the pooled t by 0.03. The
binding constraint is **six seasons**, not compute, and 2019 is where the ranking record
begins. This question cannot be answered better than it has been.

## The answer, from three independent evaluations

| evaluation | reads | pooled at 0.08 | seasons selecting 0 |
|---|---|---|---|
| top-50 points | 2-3 boundary players | -30.4, t -0.68 | 3 of 6 |
| whole-board Spearman | all ~430 ranks | -0.0048, t -4.69 | 2 of 6 |
| **simulated drafts** | **full 16-round roster** | **-45.3, t -1.20** | **4 of 6** |

No evaluation selects a lambda above 0.08 in any season. None shows a significant positive
effect at any lambda. The two that are significant are significant in the *negative*
direction and only at large lambda, where all three agree the adjustment is harmful.

**`projection_lambda = 0.0`**, and the draft-based evaluation is the one to trust: it is
the only one that scores the thing the board is actually for.

## The recency-weighted signal

The season-total gap treats week 1 and week 17 as equally informative about next year.
Roles change, so a different hypothesis: weight recent weeks more. `weighted_signal` does
this with an exponential half-life, and at `half_life=None` it reproduces the season-total
signal exactly, so the comparison is like-for-like rather than two different quantities.

**Mean lift in starter points, pooled over six seasons, evaluated over simulated drafts:**

| half-life | lam=0.04 | lam=0.08 | lam=0.16 |
|---|---|---|---|
| **uniform** | **-12.4** (t -0.60) | **-45.2** (t -1.20) | -83.8 (t -2.41) |
| 8 wk | -23.5 (t -1.02) | -47.9 (t -1.25) | -122.8 (t -2.59) |
| 6 wk | -23.2 (t -1.00) | -47.2 (t -1.25) | -129.8 (t -2.69) |
| 4 wk | -25.6 (t -1.28) | -60.2 (t -1.67) | -113.1 (t -2.62) |
| 3 wk | -25.8 (t -1.17) | -62.8 (t -1.90) | -93.2 (t -2.54) |
| 2 wk | -28.7 (t -1.32) | -65.6 (t -1.62) | -90.4 (t -2.25) |

**Recency makes it worse, and nearly monotonically.** No cell in the grid is positive. The
best of twenty-four is uniform weighting at lambda 0.04, and that is still -12.4.

### Why, measured rather than guessed

If the signal were real but stale, recency would help. It hurts, which points the other
way -- so measure the signal's own persistence directly. Correlation of a player's z in one
season with the same player's z in the next:

| half-life | 21-22 | 22-23 | 23-24 | 24-25 | mean |
|---|---|---|---|---|---|
| **uniform** | +0.249 | +0.152 | +0.192 | +0.242 | **+0.209** |
| 8 wk | +0.221 | +0.177 | +0.190 | +0.184 | +0.193 |
| 4 wk | +0.174 | +0.155 | +0.149 | +0.111 | +0.147 |
| 2 wk | +0.127 | +0.107 | +0.070 | +0.057 | +0.090 |

Two things fall out, and together they close the whole investigation.

**Shortening the window makes the signal less stable, monotonically.** That is the
signature of noise, not of staleness: a shorter window means fewer weeks averaged, so more
of what survives is week-to-week variance. Recency weighting does not sharpen this signal,
it thins it.

**Even at its most stable the signal barely persists at all.** r = 0.21 year over year is
about 4% shared variance. Whatever the expected-versus-actual gap measures, roughly
96% of it does not carry into the next season.

That is the answer to the whole thread. Not "lambda is mis-tuned", not "the metric was
wrong", not "the signal decayed" -- **there was never enough signal to tune.** A quantity
that reproduces itself at r = 0.21 cannot move a board built by hundreds of analysts, and
six seasons of holdouts were measuring the absence of an effect rather than the size of
one.

## What would still change it

Nothing available. Not more lambda values, not more seasons, not more simulated drafts, and
not a different weighting -- each has now been shown not to bind, and the last one measured
why.

A genuinely different signal would need a different input, not a different transform of
this one. Expected points from opportunity is one view of a player; something with more
year-over-year persistence -- age curves, contract or depth-chart status, offensive line
continuity -- is a separate hypothesis and would deserve this same treatment rather than
inheriting its conclusion.

**Depth-chart status was tried next and rejected** -- see `docs/depth-chart-signal.md`.
It is null at every horizon tested: next season beyond ECR (r = +0.01), rest-of-season
beyond season-to-date, and among uncontested waiver candidates specifically. An earlier
version of that document claimed the signal was real within a season at 7.4 sigma; that was
lookahead, the climb having been measured inside the outcome window, and the corrected
figure is r = -0.005.

The apparatus is reusable and that is the durable part. `hub.draft.evaluate` scores any
board by the roster it drafts, and it is the harness Phase 5's waiver and trade evaluators
want.

## What changed in the code

- `projection_lambda` = **0.0** in `conf/config.yaml` and `hub.config`.
- `DEFAULT_LAMBDA` = 0.0 in `hub.draft.projection`, so the module is a no-op by default.
- `holdout()` bounds the ECR window on both sides.
- `board_page()` selects the page by season and names the 2020 taxonomy change explicitly,
  rather than silently comparing two different boards.

`hub.draft.projection` is imported by nothing except its own test. Given six seasons of
this, that stays the right state.

## Reproduce

```bash
uv run python -m hub.draft.tune --sweep
uv run python -m hub.draft.tune --sweep --signal-season 2019 --board-season 2020
```
