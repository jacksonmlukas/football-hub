# The weekly injury cost, and the first model to clear its gate

**Measured 2026-08-25**, `hub/models/injury.py`. The repo had no weekly injury input. It has
one now, and unlike championship equity, VOR ordering, `edge` and the lineup optimiser, it
**beat the simple rule it was gated against**.

## This is not `INJURY_BETA`

`hub.draft.durability.INJURY_BETA` prices OUT/DOUBTFUL/IR at −1.631 and applies it to
`proj_blend`, a *season-long per-game* projection. It answers "what does a preseason
designation cost across the whole season" — a draft question. A player ruled out in week 1
misses week 1 and plays the other sixteen, so the season-average cost is small.

This answers the *lineup* question: what does the designation cost **in the week it is issued**.

I nearly reported the two as a 5× discrepancy. They are different quantities and comparing them
would have been nonsense. `INJURY_BETA` is unchanged.

## The estimand changed, and why

`docs/next.md` framed this as `P(plays week N | injury type, practice status, weeks since)`.
Measured directly, "plays" is not cleanly observable: `player_stats` carries a row only for a
player who *recorded a stat*, so about 18% of demonstrably healthy players on the injury report
have no row — a WR3 who dressed and was never targeted is indistinguishable from one who was
inactive. Separating them needs a `gsis_id`-to-`pfr_player_id` crosswalk into `snap_counts`.

It also would not help. For fantasy a player who dressed and scored nothing is identical to one
who did not dress, and every consumer wants points. So the estimand is **points against the
player's own healthy baseline**, which subsumes P(plays) and is directly usable.

## What it costs

Fraction of his own healthy production a designated player keeps. 2022-25, 4,939 designated
player-weeks with at least six healthy weeks to form a baseline:

| status | practice | n | keeps |
|---|---|---|---|
| Doubtful | did not participate | 98 | **0.0%** |
| Out | limited | 63 | **0.0%** |
| Out | did not participate | 790 | **0.0%** |
| Questionable | did not participate | 207 | 40.9% |
| Questionable | limited | 819 | 59.0% |
| (none) | did not participate | 138 | 61.3% |
| Questionable | full | 260 | 71.7% |
| (none) | limited | 308 | 87.9% |
| (none) | full | 2154 | 92.5% |

**Monotone in both dimensions, independently.** A Questionable player who did not practise
keeps 41%; one who practised fully keeps 72%. `INJURY_BETA` prices QUESTIONABLE at zero — right
for a preseason board, where the Questionable group is far healthier, and clearly wrong in
season once the practice report is attached.

Note the ratio is of *totals*, not a mean of ratios: a player whose baseline is near zero
produces a ratio near infinity, and averaging those measures nothing.

## The gate, and the additive table that failed it

Pre-registered: the fitted table must beat **both** simpler rules on held-out mean absolute
error, walking forward one season at a time.

* `baseline` — ignore the designation. The null.
* `out_zero` — bench anyone Out or Doubtful, otherwise ignore it. **The rule every manager
  already follows for free**, and the one worth beating.

Beating only the null would show that injuries matter, which nobody doubts.

**The first attempt lost.** An additive table — `baseline + penalty` — scored 4.978 against
4.229 for `out_zero`. The reason is mechanical and is the finding: an Out player scores exactly
zero, but `baseline − 8.35` predicts 3.65 for a 12-point player. **Additive is the wrong
functional form**, and a multiplicative one expresses `Out → 0` exactly.

That second candidate was declared before being run, and it clears:

| candidate | held-out MAE |
|---|---|
| ignore the report | 5.9164 |
| bench the ruled-out | 4.2291 |
| additive table | 4.9778 |
| **retention (multiplicative)** | **4.0586** |

Better in **all three** held-out seasons. Paired across 3,687 held-out player-weeks it beats
`out_zero` by **0.170 MAE at 3.8 se**.

## What it is not

It wins on 51.5% of individual observations — a small edge applied consistently, not a
transformation. And the practice report is only available in season, so this does nothing for
the draft. Its consumer is the weekly lineup, which is
[currently inert](adr/0012-the-lineup-optimiser-waits-for-real-variance.md) for a different
reason: `sd = k·sqrt(mu)` gives the optimiser no variance to exploit. This table gives it a
better `mu`; the usage layer is what would give it a real `sd`.

## Reproduce

```bash
uv run python -m hub.models.injury --fit
```

31 offline tests in `tests/unit/test_injury.py`, including that an Out player's retention is
exactly zero and that a designated week with no stat row scores zero rather than being dropped
— dropping him would measure the cost of an injury among players who played through it.
