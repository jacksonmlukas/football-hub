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

## What would still change it

Not more lambda values, and not more seasons — 2019 is where `load_ff_rankings("all")`
begins, so six pairs is the whole available record.

What is left is to ask what was different about 2023→24 rather than averaging it away. If
the gain concentrates in one position or one band of the board, the whole-board top-50
metric is hiding a real effect. That is post-draft work and it does not change what to do
on Sep 3.

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
