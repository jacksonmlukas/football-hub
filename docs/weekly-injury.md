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

> *Restated 2026-10-07 (#361): this table's baseline was a within-season lookahead; the cells
> on the strictly-prior baseline are in the box under "The gate" below, and its prior values
> are the ones printed here.*

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

> **Restated 2026-10-06 — #311: the standard error is over the season, not the row
> ([method.md](method.md) rule 13; the figure above stands as published).** 3.8 se is a t over
> 3,687 correlated player-weeks. On the three held-out seasons as the unit (`paired_gain`'s
> cluster, rebuilt from `walk_forward`'s own fits, the same rows): **+0.1700 MAE (equal-season
> weighted; row-weighted +0.1691), t = +4.19 on 2 df, p = 0.053**, se 0.0406; the seasons are
> +0.2325, +0.0727, +0.2047. **Better in all three seasons stands** on the sign
> (3 of 3 with the declared no-op within-season unit; won 2, tied 1 once the player is the
> repeated-measure unit, 2024's +0.0727 sitting inside its own within-season noise). **What
> changes is the pooled half:** 4.19 sits just under the 4.30 a 95% t interval needs on 2 df,
> so the interval contains zero by a hair. `retention` was *adopted* on `injury.verdict`'s
> argmin, which read no standard error at all — that decision is #360's and is not moved by this
> restatement — but the page's "at 3.8 se" no longer describes the evidence: it is a
> three-season result at p = 0.053, and under the corrected bar
> ([ADR-0019](adr/0019-a-gate-requires-every-season.md), the S1 t interval) it would not clear
> the pooled half. Named, not resolved: the figure moves, the module's decision stays frozen
> behind #360.
>
> > **Restated 2026-10-07 — #361 (S4): the baseline every figure on this page is measured
> > against was a within-season lookahead ([method.md](method.md) rule 2; the figures above stand
> > as published).** `observations` took a designated week's baseline as the player's mean over
> > the healthy weeks of the **whole season**, so a week-9 designation was scored against weeks
> > 10-18: the held-out arm was handed the player's realised in-season level, a column no Sunday
> > has. The baseline is now the **expanding mean of his strictly earlier healthy weeks of that
> > season**, and the six-healthy-weeks floor is on that *prior* count. Re-run
> > `--fit --seasons 2022,2023,2024,2025` (planted-leak control first: a baseline that moves
> > when only later weeks move now fails a test, and the old code fails it).
> >
> > | | before (lookahead baseline; re-run 2026-10-07 at `9fc1a3a`, matches the published figures to rounding) | now (strictly prior) |
> > |---|---|---|
> > | designated player-weeks with a usable baseline | 4,939 | **2,581** (the first six healthy weeks of a season can no longer be scored: early-season designations drop out) |
> > | held-out rows, 2023-25 | 3,687 | **1,963** |
> > | held-out MAE, `retention` | 4.0576 | **4.2444** |
> > | held-out MAE, `out_zero` | 4.2276 | **4.3067** |
> > | retention vs `out_zero`, mean gain | +0.1700 (t = +4.19 on 2 df, p = 0.053, per #311) | **+0.0623** (equal-season weighted; se 0.0588, **t = +1.06 on 2 df, p = 0.40**; t interval [−0.1909, +0.3155]) |
> > | seasons | +0.2325, +0.0727, +0.2047: all three | **+0.0441, −0.0529, +0.1957: won 2, lost 1** (2024 reverses) |
> >
> > **The gain falls by about two thirds and the sign no longer holds in every season.** Roughly
> > **0.011 of the 0.108 lost is the lookahead itself**; the rest is the population. Scoring the
> > *same* 1,963 held-out rows with the old, whole-season baseline gives +0.0729 (+0.0475,
> > −0.0330, +0.2043), against +0.0623 honestly: a lookahead worth about one hundredth of a point
> > here, and the other ~0.097 is the population: the early-season designations the strictly-prior
> > rule can no longer score were rows where `retention` won by more (the training windows
> > shrink too). Both are named because the
> > ticket asked for the first and the second came with it: a designation in weeks 1-6 has no six
> > weeks of its own history, and carrying the previous season over is a different baseline this
> > ticket does not choose.
> >
> > The retention cells move with it (n, keeps; 2022-25, 2,581 rows): Out / did not participate
> > 440, **0.0%**; Questionable / DNP 128, **41.1%**; (none) / DNP 77, 46.6%; Questionable /
> > limited 404, **64.3%**; Questionable / full 127, **79.8%**; (none) / limited 165, 91.3%;
> > (none) / full 1,104, **92.1%**. The Doubtful cell (98) is now below `MIN_CELL` and is folded
> > into the pooled value. Monotone in both dimensions still stands; Out keeps exactly nothing
> > still stands.
> >
> > **What this page does not decide.** `injury.verdict` is still the argmin that adopted
> > `retention`, and *whether the one adopted model stays adopted is #360's decision and the
> > maintainer's*, not this restatement's. Read through the Gate's own half-rules (the sign in
> > every resolved season, and a pooled interval excluding zero), retention vs `out_zero` would
> > fail both on these figures; nothing about the module's product role changes here.
> >
> > **What production would use instead of this column.** On a Sunday there is no in-season
> > healthy mean: `roster.availability` (`src/hub/season/roster.py`) reads ESPN's
> > `projected_total / projected_avg` ratio, a season-level figure that says nothing about
> > which games are missed, and `grep` finds no caller of `retention_table` or
> > `predict_retention` outside `injury.py`, so the adopted model is not in the lineup path at
> > all. The deployable form of this baseline is the strictly-prior expanding mean built here,
> > which is also what a live lineup call could compute from the weeks already played; it is
> > not wired to anything, and wiring it is a separate decision.
> >
> > **The type comparison, same re-run (ledger `injury_type`, 2026-10-07, `resolved: 3,
> > abstained: 0`, recipe `seasons=2022+2023+2024+2025`):** published +0.0320 (t = +1.69 on
> > 2 df, p = 0.23, won 2 lost 1) → **+0.0399, t interval [−0.0531, +0.1328], won 2 lost 1
> > (2023 −0.0041, 2024 +0.0867, 2025 +0.0370)**; MDE +0.1111 against a declared ceiling of
> > +0.3852 (was +0.0974 against +0.2641), so the design still runs. **SHOW — KEEP `retention`,
> > unchanged.**
> >
> > **Ledger (#436):** the first post-#361 `injury_type` entry (2026-10-07T12:51Z) shares
> > `config_digest`/`data_digest` (`c4606f91`/`3028f320`) with the lookahead-baseline entries,
> > because neither digest sees code. The recipe now carries `baseline=strictly-prior`
> > (`injury.BASELINE`); a corrected entry (2026-10-07T13:43Z, recipe
> > `baseline=strictly-prior,seasons=2022+2023+2024+2025`) was appended and the earlier ones
> > kept. The run now prints "3 earlier run(s) of this gate at another recipe ... not compared".
> > A code digest in the key is #435's.

### Pre-registration, 2026-10-07 (#360): retention against `out_zero`, through the Gate

**Written and committed before the gate is run for a number (method.md rule 1).** The decision
that adopted `retention` was `injury.verdict`, an argmin over four candidates with no interval
and no every-season half; this section replaces it with the Gate every other comparison reads
(`experiment.run_gate`, ADR-0019). Decided by the maintainer in session on 2026-10-07 on #360.

* **Arms: two.** `retention` (arm A, the fitted multiplicative table of `retention_table`)
  against `out_zero` (arm B, the incumbent: bench anyone Out or Doubtful, otherwise ignore the
  report). `table` and `baseline` are diagnostics: still measured and printed, never gated.
* **Paired unit.** One row per held-out designated player-week; `diff` is `out_zero`'s absolute
  error minus `retention`'s, positive when retention helps. Walk-forward over
  `expanding_seasons`, fitting on strictly earlier seasons only (unchanged).
* **Baseline.** Stays within-season: the strictly-prior expanding mean of #361
  (`injury.BASELINE = "strictly-prior"`). Carrying the previous season over is not chosen here.
* **Ceiling.** In-sample per-cell retention: per held-out season, the (status, practice)
  retention table fitted on that season's own designated rows (same `MIN_CELL`, thin cells folded
  to that season's pooled ratio) and scored on the same rows. `ceiling_diff` is `out_zero`'s
  error minus this arm's. Flattered by construction, which is what a ceiling is: a bound on the
  functional form, never read off an outcome the rule then judges.
* **Within-season unit.** `gsis_id`, the player (rule 3; #335). This also replaces the declared
  no-op (`season`) on the type comparison's `injury.HARNESS`, discharging the ADR-0019 note.
* **Actions, in the gate's own words.**
  ADOPT: *Retention is the availability model the product wires in place of zeroing a
  designated player.* REMOVE: *Retention is worse than zeroing a designated player; it is
  dropped from the model surface.* SHOW: *Retention is kept as a measurement and not wired; the
  product keeps ESPN's availability.*
* **Power, stated before the run (rule 16).** The #361 re-run's pooled standard error was 0.0588
  on 3 held-out seasons (2 df), so the gate's MDE is about (4.303 + 0.842) x 0.0588 = 0.30 MAE
  points against a realised gain of +0.062. Whether the gate can run depends on the in-sample
  ceiling, which is not known until it is run: if the ceiling is below the MDE the branch is
  **NOT-RUNNABLE**, and that is named here as reachable and, on these figures, plausible. It is
  then an exemption and not a null: it decides neither ADOPT nor REMOVE, `retention` stays
  unwired as it is today (`roster.availability` reads ESPN), and what the page says is "the
  design cannot resolve this", with the MDE. Four seasons of nflverse injury reports exist; a
  fourth held-out season does not until 2026 completes.
* **The gate's abstentions** are read as #381 (C) reads them: the verdict is over resolved
  seasons and prints "N resolved of K, A abstained".
* **What this lane does not do.** A non-ADOPT verdict is restated here and not acted on: nothing
  about what the product wires changes; that action is the maintainer's.

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

> **Restated 2026-10-06 — #343 routes the type comparison through the Gate; the season cluster
> (#311) comes with it ([method.md](method.md) rule 13; the figures above stand as published).**
> Re-run today on the same rows, the held-out MAEs and the cleaned **+0.0317 at 3.1 se,
> 2/3 seasons reproduce**; what is restated is the statistic and the rule.
>
> | | published (cleaned) | now |
> |---|---|---|
> | mean gain | +0.0317 MAE (row-weighted) | **+0.0320** (equal-season weighted; the three seasons are +0.0507, +0.0602, −0.0150) |
> | significance | 3.1 se over 3,687 pooled player-weeks | **t = +1.69 on 2 df, p = 0.23**; t interval [−0.0495, +0.1134] |
> | seasons | 2/3 | won 2, tied 0, lost 1 — *sign only*: the within-season unit is the declared no-op (`season`, one cluster a season), pending #360 |
> | stage 2 | none | **runs: MDE +0.0974 against the declared ceiling +0.2641** (the in-sample per-type multiplier; retention minus that oracle, +0.203 / +0.369 / +0.217 a season) |
>
> **Verdict: SHOW — KEEP `retention`. The decision does not move; the evidence for it is both
> weaker for the type adjustment and cleaner than this page said.** 3.1 se is 1.7 on 2 degrees
> of freedom — the interval contains zero by a wide margin, so the page's reading that the
> adjustment "clears significance and loses a season" overstated the first half as well as
> resting on the second. The design *can* run (the MDE is a third of the ceiling), so this is a
> null the design could have contradicted, which the unrouted rule could not say. The type
> adjustment recovers +0.032 of a +0.264 flattering in-sample bound: 12%.
> **Statistic restatement only:** `injury.verdict` (the argmin that picked `retention`) is #360's
> and untouched, and the within-season unit stays the declared no-op until #360 chooses one.
>
> > **Restated 2026-10-06 (#381, re-run; the verdict is kept).** *Prior value: SHOW — KEEP
> > `retention`, "won 2, tied 0, lost 1".* **Now: SHOW — KEEP `retention`, on 3 resolved of 3, 0
> > abstained** (won 2, lost 1, the interval containing zero). Nothing moves: no season
> > abstains here (the within-season unit is the declared no-op, so each season reads on its sign),
> > so (C) reads the same three seasons (A) did, and the sentence now says so. Re-run
> > `--fit --seasons 2022,2023,2024,2025`, figures identical to the digit. Ledger:
> > `injury_type`, 2026-10-06, `resolved: 3, abstained: 0`.

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
