# Per-player weekly spread: real, small, and not estimable

**Measured 2026-08-25**, `hub/models/spread.py`. This tests the claim
[ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md) rests on:

> Two players projected at 12 points a game are not equally volatile in reality — a boom-bust
> deep threat and a target-hogging possession receiver are not the same asset — and the
> square-root law cannot say so because it only knows the mean.

The first half is true. The second half is where it fails: they are not the same asset, but
the difference is about **±9% in sd**, and a season of data recovers so little of it that
`sd = K[position]·sqrt(mu)` is already within 8% of the best any model can do.

## Why this question and not a usage model

Improvement #6 was scoped as "build the usage model, gated against pick-anchored volume".
That gate is the wrong one for the thing ADR-0012 promises. The optimiser does not need a
better *mean* — it needs `sd` to stop being a function of the mean. So the estimand here is a
player-season's realised weekly PPR spread, and the shipped positional constant is the arm to
beat.

## The candidates, ordered by how much they assume

* `positional` — `k = K[position]`. **The shipped model.**
* `own_k` — the player's own prior-season `k`, shrunk toward `K[position]` in logs. Assumes
  only that volatility is a persistent property of a player, and needs no usage data at all.
* `usage` — `K[position]` scaled by a fitted function of prior-season role: target share,
  air-yards share, snap share, touchdown rate, within-season snap drift, and log mean.

`usage` is fitted as a *residual* from `positional`, so a zero coefficient vector reproduces
the shipped model exactly and the arm can only win by earning it.

**The gate, fixed before running:** beat `positional` on held-out MAE in **every** held-out
season **and** clear 2 standard errors on the paired difference. Both arms always receive the
same `mu`, so the comparison cannot be contaminated by projection error.

## Result: nothing is adopted

2019-25, 2,077 qualifying player-seasons (≥8 games, >3 ppg — matching
[weekly-spread.md](weekly-spread.md)'s sample), 1,240 consecutive-season pairs. Snap share
matched through the `pfr_id`→`gsis_id` crosswalk on 99.8%.

| held-out season | n | `positional` | `own_k` | `usage` |
|---|---|---|---|---|
| 2021 | 207 | 1.0914 | 1.0881 | 1.1152 |
| 2022 | 216 | 1.1523 | 1.1442 | 1.1089 |
| 2023 | 206 | 1.0374 | 1.0296 | 1.0220 |
| 2024 | 205 | 1.1573 | 1.1467 | 1.1666 |
| 2025 | 205 | 1.0414 | 1.0385 | 1.0712 |

    KEEP 'positional': no candidate cleared both halves of the gate.
      own_k: mean gain +0.0065 MAE at 1.8 se, wins 5/5 seasons
      usage: mean gain -0.0004 MAE at -0.0 se, wins 2/5 seasons

`own_k` wins every season and misses the significance half at 1.8 se. Chosen shrinkage was
0.05–0.10 — the fit itself wanted to keep almost none of a player's own prior `k`.

## The part that actually closes the question

A candidate failing is weak evidence; the ceiling is strong evidence. The outcome being
predicted is a *realised* sd from ~14 games, which is itself an estimate with sampling error
of about `σ/sqrt(2(n−1))`. Even a model that knew every player's true volatility exactly would
still miss the realised value by that much:

| | MAE |
|---|---|
| irreducible sampling noise in the outcome | **1.0113** |
| the shipped positional model | **1.0965** |

**Total headroom for every future model combined: 0.085.** `own_k` took 0.0065 of it, about
8%. And that floor is computed under a normal approximation, while weekly scoring is
right-skewed — which makes the sampling variance of an sd *larger*, so the real floor is
higher and the real headroom smaller still.

## Is per-player spread real at all?

Yes, and it is worth separating from "we cannot measure it". Splitting each season into odd
and even weeks — so a within-season role trend cannot drive it — the log-`k` residual
correlates with itself at **+0.081** (n=1,625). By Spearman-Brown that puts the reliability of
a full season's `k` at **0.150**: 85% of what looks like a player's distinctive volatility is
noise.

Year over year the same residual correlates at **+0.127** (n=1,240) — *higher* than the
within-season split-half, which is only consistent with a genuinely persistent trait whose
measurement is heavily attenuated. Correcting for reliability, the true spread of per-player
volatility is:

    sd of log-k residual, observed : 0.2305   (±25.9% in sd)
    sd of log-k residual, true     : 0.0892   (±9.3%  in sd)

So for a 12-ppg receiver, the shipped model says sd 7.38; players one standard deviation apart
in true volatility sit at 6.75 and 8.07. **Real, and roughly a quarter the size the observed
scatter suggests.** The trap this avoids is fitting the observed 25.9%, which is mostly noise.

## What this does to ADR-0012

ADR-0012's decision is unchanged — the optimiser still does not set lineups. What changes is
its forecast. It says "when `sd` stops being a function of `mu`, re-run this gate", and treats
the usage layer as the thing that would make that happen. It would not, or not by enough: the
recoverable per-player signal is ±9% in `sd`, and an individual estimate of it is 85% noise.

This is the seventh measured attempt to beat a simple incumbent in this repo, and the sixth to
fail. It is also the cheapest one to have run, and it retires a roadmap item that looked like
the highest-leverage work available.

## What is deliberately not concluded

The `usage` coefficients have sensible signs and sensible magnitudes — target share −0.56
(steady volume, steadier scoring), air-yards share +0.38 (deep threats are boom-bust), snap
share −0.16. The mechanism is real. A restricted model on those two features would very likely
score better than the six-feature one.

**That model is not fitted here, and would not be adopted if it were.** Choosing features
after seeing which ones came out large is how a gate gets retuned into passing, and the
headroom calculation says the prize is at most 0.085 MAE regardless. The question is closed by
the ceiling, not by any one candidate's failure.

## Also recorded: the gate's first version was missing half of itself

`verdict` originally checked only that a candidate won every held-out season — the 2-standard-
error requirement was written in the module docstring and never implemented. On the first run
it printed `ADOPT 'own_k'` on a gain of 0.0065 MAE, 0.6% of the baseline.

A gate that is documented but not implemented is worse than no gate, because it is quoted in
the write-up. Same defect class as the draft tripwire and the lineup gate's first version, and
recorded here for the same reason: it was not obvious while being written.

The `usage` arm was also wrong on that first run — fitted on `log k` but predicted as a
multiplier on `K[position]`, double-counting the positional constant and scoring 6.3 against
1.1. A result six times worse than the null is a bug, not a finding.

## Reproduce

```bash
uv run python -m hub.models.spread --fit --seasons 2019,2020,2021,2022,2023,2024,2025
```

34 offline tests in `tests/unit/test_spread.py`, including that the `usage` arm with no
coefficients is byte-for-byte the shipped model, that shrinkage is geometric rather than
arithmetic, and that a consistently-signed gain too small to distinguish from noise is
rejected — the case the first version of the gate got wrong.
