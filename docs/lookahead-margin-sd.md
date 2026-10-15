# A margin sd fitted by weeks-out, on the snapshot archive (#249)

**Pre-registered 2026-10-14, before any 2026 margin residual was computed.** The commit that
adds this file is the timestamp ([track-record.md](track-record.md) rule 1). What had been read
when it was written, all of it an input and none of it an outcome: the span of `captured_at` and
a count of snapshots per weeks-out bucket (below); the number of 2026 games with a result and the
week they were played in (a count, not a score); and a simulation on *synthetic* spreads, whose
parameters were chosen here, for the power section. No 2026 `result` was subtracted from any
spread. The prior method is [margin-sd.md](margin-sd.md); this changes nothing in it.

## The question

The survivor grid prices a distant week's game from a snapshot spread with `MARGIN_SD`
(12.741), an sd fitted on **closing** spreads (`schedules.spread_line` and `result`, 1999-2025).
A spread posted four weeks out is a less informed number than the close, so the margin around it
is plausibly wider and `Phi(spread / 12.741)` overconfident; survival multiplies two dozen of
them. #249 asks whether an sd fitted by weeks-out prices those rows better.

## What the archive holds (measured 2026-10-14, inputs only)

`captured_at` runs 2026-08-25 to 2026-10-14: **50 days, 7.1 weeks**, 15 distinct capture
instants. The gate in the ticket (six weeks) is met by the span and not by the coverage: captures
fall in **four calendar weeks** (Aug 24, Aug 31, Oct 5, Oct 12) with a **30-day hole**, 2026-09-07
to 2026-10-06, the stretch #383 repaired for the future and could not back-fill. 79 games have a
result (weeks 1-5). Scored (game, bucket) rows by bucket, buckets as fixed below:

| bucket | weeks out | scored (game, bucket) rows |
|---|---|---|
| b0 | [0, 1) | 17 |
| b1 | [1, 2) | 15 |
| b2 | [2, 4) | 44 |
| b3 | [4, 8) | 43 |
| b4 | [8, inf) | 0 |

The edges (next section) were written down before these counts were taken; the one earlier count
was of all snapshots, scored or not, by whole week out. They are one-week steps near the close
and wider steps far out. **b4 is empty**, and is named so rather than merged away.

## The arms

- **Incumbent (arm A).** `P(home wins) = Phi(spread / MARGIN_SD)`, `MARGIN_SD` imported from
  `hub.models.market`. Every row, every bucket.
- **Challenger (arm B).** The same with `sd_b` for the row's bucket `b`, fitted on **strictly
  earlier weeks only** (`experiment.expanding_weeks`), by `hub.models.margin.fit` on
  `result - spread`.
  - **b0 is the closing end.** It is priced at `MARGIN_SD` in both arms and is not scored in the
    gate: the ticket's "nothing changes for a live price", held by construction.
  - **A bucket with fewer than `MIN_FIT_N = 30` earlier rows is not fitted.** It inherits the
    sd of the nearest bucket *nearer the kickoff* that is fitted, ending at `MARGIN_SD`, and the
    report says "inherited" beside it. (The ticket says "the nearest deeper one"; read here as
    toward the close, the direction that cannot widen a price on thin evidence. The other reading
    is a different arm and would need its own pre-registration.)

**Buckets, in weeks between `captured_at` and kickoff: edges 1, 2, 4, 8.** Fixed here; the code
holds them as a constant with no parameter that takes edges, and a test reads this paragraph
and fails if the two differ.

**Unit.** One row per (game, bucket): the **last** snapshot captured in the bucket, before
kickoff (a snapshot at or after kickoff is dropped: the lookahead guard). Tied games are dropped,
`margin.DROP_TIES`'s own rule. A bucket a game has no snapshot in is simply absent for that game.

## The decision

**Metric.** Held-out log loss, `diff = ll_A - ll_B` per row (positive when B is better), unit
"nats per game-bucket". **Held-out unit: the week**, fitted on weeks strictly earlier.

**Cluster: the week, not the season** (the one place this repo does that is
`hub.season.weekly_forward`, for the reason it gives). 2026 is the only season with snapshots, so
a season cluster is one observation and has no t interval at all; a `Harness`, which is
season-clustered by construction, cannot run this design, so this is a direct `run_gate`
(`code_modules` is the closure of `ARM_ROOTS`, so the ledger sees code). The within-season
repeated-measure unit is the game.

**Ceiling arm.** Each bucket's sd fitted *in sample on every scored row of that bucket*: what the
best per-bucket constant could earn on this exact sample, flattered by construction.

**The Gate is `experiment.gate`'s, unchanged:** ADOPT, REMOVE, SHOW, NOT-RUNNABLE as written in
its docstring and ADR-0019 with #381's (C). Nothing is added or tuned.

**What each verdict does** (nothing here wires a price; wiring `hub.models.market` or the survivor
grid is the maintainer's decision on every branch):

| verdict | action |
|---|---|
| ADOPT | Reported to the maintainer with the fitted table. This module does not move `MARGIN_SD`, `hub.models.market`, `survivor` or any page. |
| REMOVE | The bucketed sd is worse than `MARGIN_SD`: `MARGIN_SD` stays on every row. |
| SHOW | Nothing changes; the gate could not establish that it pays. |
| NOT-RUNNABLE | Nothing changes; the question could not be answered with this data. |

## Power, and the branches that are reachable (rule 16)

The MDE is `summarise`'s, printed on the run. What it can be is bounded before the run, from a
simulation on synthetic spreads (`N(0, 6)`, 200,000 draws) and a margin that is the spread plus
`N(0, sd_true)`, scored `ll(12.741) - ll(sd_true)`:

| true sd far from the close | gain per row | sd of the per-row gain | MDE, 8 week clusters, 16 rows a week | MDE, 16 clusters, 30 rows |
|---|---|---|---|---|
| 13.5 (+6%) | 0.0002 | 0.019 | 0.0054 | 0.0026 |
| 14.0 (+10%) | 0.0004 | 0.031 | 0.0087 | 0.0042 |
| 15.0 (+18%) | 0.0012 | 0.052 | 0.0148 | 0.0071 |

**The MDE is larger than the gain it is looking for in every cell**, by a factor of 6 to 27. A
log-loss gate on this question cannot return ADOPT or REMOVE for a plausible effect in 2026
(four held-out week clusters today; at most about sixteen by week 18), and with the ceiling
printed it will return **NOT-RUNNABLE**. That is the branch this gate will take, and **it is
named an exemption**: a NOT-RUNNABLE here decides nothing, neither that the lookahead sd is
equal to the closing one nor that it is wider. This is rule 16, written before the first
number rather than after the first NOT-RUNNABLE.

What the data *can* speak to is the dispersion itself, which is far better determined than a
log-loss difference: `se(sd) = sd / sqrt(2(n - 1))`, so a bucket of n rows resolves a ratio to
`MARGIN_SD` of `exp(2.80 / sqrt(2(n - 1)))` at 80% power (2.80 = 1.96 + 0.84) -- 1.26 at n = 45,
1.17 at n = 100. The run therefore also prints, **as a description and not as a gate**, each
bucket's in-sample sd over every scored row with its n, its 95% interval (`sd +/- 1.96 se`) and
that ratio, whether it was fitted or inherited, and whether the interval excludes `MARGIN_SD`
from above. An interval excluding `MARGIN_SD` is shown to the maintainer; it is not an ADOPT and
moves nothing.

## What is not built here, and why it is named

The ticket's second and third criteria -- the survival product over a September-priced season
printed beside the old figure, and live-price rows byte-identical before and after -- belong to a
consumer in `hub.season.survivor`. **That consumer stays HELD while #379 is dormant** and is not
wired: no module outside this one imports this one, and a test holds that. So the acceptance
criteria of #249 are not met by this work, and its final commit says `Advances`, not `Closes`.

## The rule-18 controls, pre-registered

1. A planted archive in which the sd truly grows with weeks-out: the walk-forward must show a
   positive mean gain, and on a sample large enough to resolve it the Gate must ADOPT.
2. A flat planted archive (the same sd in every bucket, `MARGIN_SD`): no gain, no ADOPT.
3. The edges cannot be tuned after the data is seen: they are a declared constant, no function
   here takes them as an argument, and this document's edges are held against the constant.
4. A snapshot at or after kickoff never reaches a row; an unfitted bucket prices as `MARGIN_SD`
   or the nearer fitted bucket and says so; b0 is `MARGIN_SD` in both arms.
