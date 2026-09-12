# The market/Usage blend is a model, and gets one clean gate

**Status:** accepted 2026-08-29. **Resolved: SHOW — superseded 2026-09-07 (#44).** Run under
these rules the same day; the result is in [weekly-blend-gate.md](../weekly-blend-gate.md) and
summarised at the foot of this file. The proposal was committed before the run, so the commit
order is the record. The rules below are unchanged; the numbers they were read against are not,
and the pre-registration is re-scored against the re-run in
[its own section](#re-scored-2026-09-07-the-pre-registration-against-the-re-run) below.

**Decision.** The market/Usage blend stops being treated as a repair to the Weekly projection
and is gated once, as a model in its own right, under rules fixed here.

## Why it needs its own ADR

Four variants of the same rescue have now been run
([weekly-shrinkage.md](../weekly-shrinkage.md),
[weekly-market-shrinkage.md](../weekly-market-shrinkage.md)), and the fourth produced the first
positive point estimate in the programme — with the pre-registered tripwire firing on it. That
tripwire was correct: `w = n/(n+k)` never reaches 1, so the blend mixes market information into
*every* projection rather than regularising thin ones. **It is not a shrunk Weekly projection.
It is a different model**, and calling it a repair is what let it be tuned four times.

A model gets one gate.

## The model, fixed

    weekly Usage = [ w * in-season Usage + (1 - w) * market prior ] * exp(coef . snap_trend)
    w            = n / (n + k),  n = games played before the week
    market prior = exp(a + b * log(preseason rank)) - 1, per position, per count
    weekly TDs   = weekly yards * the POSITION's touchdown rate
    weekly points = league scoring applied to the counts

**Everything fitted on strictly earlier seasons, every fold**: `coef`, `k`, and the market
curves `(a, b)`. `k` is fitted by the **`mae`** objective. The `tail` objective is dropped — it
was exploratory, it cost 16% of MAE, and it lost.

**The rank is the August board's `ecr`, never the weekly ranking.** Shrinking toward the
incumbent would make the arm partly be the thing it is measured against.

> **Restated 2026-09-11 — the `exp(coef · snap_trend)` factor is 1.** Issue #233 revoked the
> snap-share trend's licence — on the settled `(yds_prior, ecr)` basis it is +0.0142 at
> permutation *p* 0.24 at the published anchor, and clears at anchor 10 alone — and #248
> implements it in `hub.models.weekly`, which this model is built on. The first line of the
> formula above therefore reads `weekly Usage = w · in-season Usage + (1 − w) · market prior`
> and the fitting sentence loses one of its three: **`k` and `(a, b)` are still fitted on
> strictly earlier seasons, every fold; `coef` is not fitted, and there is no longer a
> parameter to pass it through** (`fit_shrink` and `project` take none). The two coefficients
> named in the model, side by side:
>
> | | fate | in the model since #248 |
> |---|---|---|
> | `coef` on the snap-share trend, the Usage multiplier | **revoked** — #233 | the identity, exactly 1.0 for every player-week |
> | the touchdown term, `weekly TDs = weekly yards × the POSITION's touchdown rate` | **qualified, not withdrawn** — #229; not decided by #233 or #248 | untouched; `components.td_rate` unchanged and `tds_hat` byte-identical to the `f = 1` arm |
>
> **What this does and does not do to the record below.** The rules of the run are
> untouched. The gate's inputs still build under the same command, and its verdict —
> **REMOVE**, −1.004, CI [−1.391, −0.621], 0 of 4 seasons, from the re-run under #44 — stands
> as scored: it was scored on the fitted arm, and it is not re-run in #248. The maintainer
> re-runs it once after the change lands, on the machine, to record what the change cost, and
> that figure is restated here when it exists and not before. No fitted constant moved and
> `config_digest` / `fitted_digest` are the same before and after — neither weekly module is
> in `FITTED_MODULES`. The weekly projection is shown and never ranked on
> ([ADR-0016](0016-the-weekly-projection-is-shown-and-never-ranked-on.md)), so nothing that
> picks moves.

## Its null is not `f = 1`

The Weekly projection's null was the identity, because a multiplier of one recovers the
incumbent. **The blend has no such null**: `k = 0` is the Weekly projection (already gated,
SHOW) and `k → ∞` is the market prior alone (measured at −1.146 frozen, the worst arm tried).
The blend joins two things that have both already lost.

So its null is the **incumbent itself**: a lineup set by weekly consensus rank, exactly as in
[ADR-0015](0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md). It has to beat that or
it is nothing.

## The run, fixed

| | |
|---|---|
| Seasons | 2021–2025, **scored 2022–2025** — four held-out seasons |
| Rosters | **40 per season**, doubled from 20; the bootstrap clusters on rosters and 60 was thin |
| Weeks | 1–14, restricted to weeks consensus covers |
| Gates | **frozen is primary**; churn reported beside it |
| Metric | points per team-week, paired by roster-week, clustered by roster |
| Verdict | the three branches already in `weekly_gate.verdict` |
| Void | join failure above 2% |

**One run. Both gates. Whatever it says.**

## What is honest about this and what is not

**2022 has never been scored.** It has been training-only in every gate run so far, so it is
the one genuinely out-of-sample season here.

**2023, 2024 and 2025 have been seen**, at 20 rosters, for this exact estimator: frozen +0.159,
churn −2.450. This run is **not blind and cannot be** — the data is finite and it is spent.
What is fixed in advance is the *rule*: the estimator, the fitting objective, the sample, the
verdict, and the tripwires. That is worth less than a blind run and more than nothing, and
pretending otherwise would be the failure this repo exists to avoid.

**So 2022 carries the evidential weight**, and it is reported first.

## Tripwires, fixed

1. **Join failure above 2% → VOID.** As always.
2. **If 2022 disagrees in sign with the other three**, that is the tell that the other three
   are a fit to seen data, and the run is reported as inconclusive regardless of the pooled
   figure.
3. **If the churn gate comes out positive**, treat the run as suspect: no variant has come
   within two points of that, and a reversal that large means something changed that was not
   the model.

## Pre-stated expectation

Frozen: **positive but small, interval containing zero — SHOW.** Churn: **negative, between
−1.5 and −3.** 2022 in line with the rest.

## Consequences of each outcome

- **ADOPT** (beats consensus every season, interval excludes zero): the blend sets lineups, and
  ADR-0016 is superseded.
- **SHOW**: ADR-0016 stands, the blend is printed beside consensus, and **the rescue attempts
  end.** Five variants is where a negotiation with a result becomes a search for one.
- **REMOVE**: delete it.


---

## Resolved, 2026-08-29: SHOW

    frozen  +0.215  CI [-0.242, +0.684]  3/4 seasons   <- primary, SUPERSEDED 2026-09-07
    frozen  -1.004  CI [-1.391, -0.621]  0/4 seasons   <- re-run, #44, verdict REMOVE
    churn   -1.806  CI [-2.710, -0.921]  2/4 seasons

    (Restated 2026-08-30. It first ran at +0.711 / -2.006 under a board that was not
    reproducible -- improvements.md #18 -- so that roster sample was one of many the same
    command could draw. The verdict is unchanged and the decay across seasons still holds.)

2,000 roster-weeks over 160 rosters, four held-out seasons, join failure 0.1%. The every-season half fails, losing 2025 either way, and under the reproducible board the
interval contains zero. No tripwire fired.

**The decisive detail is not the pooled figure.** The frozen gain decays monotonically —
**+0.983, +0.027, +0.355, −0.504** across 2022 to 2025 — and is negative in the season closest
to the one being drafted. Whatever this is, it was worth two points a team-week in 2022 and is
worth nothing now. No mechanism is asserted for the decay.

**Consequence, as fixed above:** ADR-0016 stands, the blend is printed beside consensus and
never sorted on, and **the rescue attempts end.**

---

## Re-scored 2026-09-07: the pre-registration against the re-run

Issue #51. The rules above were fixed before the run and are not touched here. What follows
reads them against **−1.004, CI [−1.391, −0.621], 0 of 4 seasons** — the interval the re-run
produced under #44 — rather than against the **+0.215** they were first read against. Per
season: 2022 **−0.452**, 2023 **−1.490**, 2024 **−0.868**, 2025 **−1.204**.

| fixed in advance | read against +0.215 | read against −1.004 |
|---|---|---|
| **Pre-stated expectation**, frozen: *positive but small, interval containing zero* | met | **not met** — negative, and the interval excludes zero |
| **Pre-stated expectation**, churn: *negative, between −1.5 and −3* | met at −1.806 | not re-reported by the re-run; the superseded −1.806 is the only figure on the churn gate |
| **Pre-stated expectation**, *2022 in line with the rest* | met | met — and more strictly, see below |
| **Verdict**, on ADR-0019's two halves | SHOW: interval contains zero, 3 of 4 seasons | **REMOVE**: the interval excludes zero **and** the sign holds in every held-out season, both in the same direction |
| **Tripwire 1**, join failure above 2% → VOID | silent, at 0.1% | silent — same run, same join |
| **Tripwire 2**, 2022 disagreeing in sign with the other three → inconclusive | silent, though 2023–24 were positive and 2025 was not | silent, and for the first time all four seasons agree |
| **Tripwire 3**, churn positive → suspect | silent | silent |

**Two things are worth stating plainly, and neither is a decision.**

**The re-run satisfies the pre-registered bar more completely than the run it supersedes.** The
+0.215 result was SHOW because it failed the every-season half; the −1.004 result meets *both*
halves, in the negative direction, and no tripwire fires on it. Read only against the rules
fixed here, this is the cleanest reading the gate has produced.

**And the pre-registration did not anticipate the thing that actually moved.** None of the
tripwires above is about how an unprojected player is scored, and that is the choice the effect
turned out to be dominated by: [weekly-blend-gate.md](../weekly-blend-gate.md) measures
**+0.215 / −1.004 / +0.537** across three defensible handlings, a spread of 1.5 points against a
largest reported effect of 1.0. A verdict that flips across a choice nobody pre-registered is
not a verdict this ADR's rules can adjudicate — *"what it does not establish is the weekly
model's own merit, in either direction."* The MDE is 0.768 and the effect is −1.004, so #45's
NOT-RUNNABLE precondition does not fire and, absent a foresight ceiling for this gate, cannot.

**What is deliberately left open.** The REMOVE branch above says *"delete it"* and says nothing
about what becomes of [ADR-0016](0016-the-weekly-projection-is-shown-and-never-ranked-on.md) —
only the ADOPT branch names it. So the pre-registration is silent on exactly the case that
occurred. That gap, and whether the disposition is applied at all, belong to **#44**, where the
disposition is recorded as pending rather than taken. This section re-scores; it does not
resolve.
