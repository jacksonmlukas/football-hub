# The market/Usage blend is a model, and gets one clean gate

**Status:** accepted 2026-08-29. **Resolved: SHOW.** Run under these rules the same day; the
result is in [weekly-blend-gate.md](../weekly-blend-gate.md) and summarised at the foot of this
file. The proposal was committed before the run, so the commit order is the record.

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

    frozen  +0.711  CI [+0.313, +1.129]  3/4 seasons   <- primary
    churn   -2.006  CI [-2.761, -1.229]  1/4 seasons

2,000 roster-weeks over 160 rosters, four held-out seasons, join failure 0.1%. **The interval
excludes zero and favours the blend, for the first time in the programme** — and the
every-season half fails, losing 2025 by −0.497. No tripwire fired.

**The decisive detail is not the pooled figure.** The frozen gain decays monotonically —
**+1.894, +0.961, +0.487, −0.497** across 2022 to 2025 — and is negative in the season closest
to the one being drafted. Whatever this is, it was worth two points a team-week in 2022 and is
worth nothing now. No mechanism is asserted for the decay.

**Consequence, as fixed above:** ADR-0016 stands, the blend is printed beside consensus and
never sorted on, and **the rescue attempts end.**
