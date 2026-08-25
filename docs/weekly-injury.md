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

## Does *what is wrong with him* add anything? Measured 2026-08-25: not by the gate

The table above prices a designation by `report_status` × `practice_status` and ignores
`report_primary_injury` entirely. A hamstring is not an ankle is not a concussion, so the
obvious extension is a per-type multiplier on what the table already predicts.

**The gate was declared before running, and the incumbent moved.** The arm to beat is
`retention` — the thing that already won — not `out_zero`. To be adopted it had to beat it in
**every** held-out season **and** clear 2 se on the paired difference.

| held-out season | n | `retention` | type-adjusted |
|---|---|---|---|
| 2023 | 1,191 | 4.0336 | **3.9942** |
| 2024 | 1,247 | 4.3477 | **4.2951** |
| 2025 | 1,249 | **3.7943** | 3.8125 |

    KEEP 'retention': what is wrong with him adds nothing measurable to how he practised.
      type-adjusted: mean gain +0.0244 MAE at 2.5 se, wins 2/3 seasons

**This is the closest call in the repo, and it is the mirror image of
[player-spread.md](player-spread.md).** There, `own_k` won every season and missed on
significance at 1.8 se. Here the type adjustment clears significance at 2.5 se and loses a
season — and the season it loses is 2025, the most recent one, which is the one you would
weight most.

Not adopted. Two halves, both required, and this is exactly the case the two-halves rule is
for: a mean gain of 0.024 points on a 4.0 baseline is 0.6%, and one bad season out of three is
most of what there was to see.

### The multipliers, which are the interesting part

Fitted on all four seasons, shrunk toward 1.0 (`k` came out at 25 on every fold):

    Hamstring 0.691   Forearm 0.744   Right Shoulder 0.745
    Foot 1.190        Illness 1.416   Achilles 1.469

**Hamstring is the largest and the mechanism is the folk one**: a hamstring keeps 31% less than
his practice report implies, which is what everyone who has ever been burned by one believes.
Above 1.0 the reading is different and worth stating — these are conditional on being
*designated and still playing*, so an Achilles or an illness that did not rule a player out is
a milder thing than the word suggests.

### The type field was three fields, and fixing it did not change the answer

The numbers above were measured on the raw field, and it is dirty. nflverse passes the club's
own wording through, so `Shoulder`, `Right Shoulder` and `left Shoulder` are three categories
for one injury: **110 distinct values across 2022-25, collapsing to 74** on casefolding and
stripping laterality. One of them is a free-text sentence beginning "Player was ill this
morning". A further 2,601 designated player-weeks carry no type at all.

That is a defect in the feature extraction, not a modelling choice — a right hamstring costs
what a left one costs — so it was fixed and **the same gate re-run, unchanged**:

| held-out season | n | `retention` | type-adjusted, raw | type-adjusted, cleaned |
|---|---|---|---|---|
| 2023 | 1,191 | 4.0336 | 3.9942 | **3.9829** |
| 2024 | 1,247 | 4.3477 | 4.2951 | **4.2875** |
| 2025 | 1,249 | **3.7943** | 3.8125 | 3.8093 |
| | | | +0.0244 at 2.5 se | **+0.0317 at 3.1 se** |

Pooling the evidence helped, by about what you would expect. **The verdict did not move**:
still 2/3 seasons, still losing 2025.

Stating the obvious risk plainly, because this was the second run of one hypothesis: had the
answer flipped, it would have been much weaker evidence than a single pre-registered run, and
would have had to be reported as such. It did not flip.

### The sign flips in the most recent season, which this repo has a rule about

[depth-chart-signal.md](depth-chart-signal.md) records, from a screen that went wrong:

> A significant result whose sign flips between seasons is a bug, not a finding.

That is this shape. The type adjustment helps in 2023 and 2024 and hurts in 2025, and 3.1 se
is computed by pooling all three. The every-season half of the gate is not an arbitrary
second hurdle here — it is the half that catches exactly this, and it is why the gate has two
halves.

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
