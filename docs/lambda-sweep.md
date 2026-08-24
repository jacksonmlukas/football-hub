# projection_lambda: six holdouts, and the answer

Phase 2.4 of `docs/foundation-plan.md`. Run 2026-08-23: one season pair, then three, then
six. The answer never changed. The reasoning changed twice, and both revisions are recorded
because they are the point.

**Result: `projection_lambda = 0.0`.**

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

## What would still change it

A different kind of evaluation, not a different statistic. Every metric here scores one
static ordering; a draft is a sequence of decisions against a depleting pool. Evaluating
over many simulated drafts -- which `hub.draft.optimize` already does for championship
equity -- would measure the thing that actually matters, and would not have a boundary for
one injured player to sit on.

Post-draft work. Nothing here changes what to do on Sep 3.

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
