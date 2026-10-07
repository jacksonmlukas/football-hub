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

> **Restated 2026-10-06 — #343 routes this comparison through the Gate and #311's season cluster
> comes with it ([method.md](method.md) rule 13; the figures above stand as published).** Re-run
> the same way (`--seasons 2019,…,2025`, same 5 held-out seasons) the **held-out MAEs reproduce
> to the digit**; what is restated is the statistic and the rule that reads it.
>
> | | published | before #343, run today | now |
> |---|---|---|---|
> | `own_k` mean gain | +0.0065 MAE | +0.0065 | **+0.0065** (+0.00653; equal-season weighted) |
> | its significance | 1.8 se, over 1,039 pooled rows | 1.8 se | **t = +4.93 on 4 df, p = 0.008**; se 0.0013 (a bootstrap over the 5 season means); t interval [+0.0029, +0.0102] |
> | its seasons | wins 5/5 | wins 0, ties 5, losses 0 (#335 already in force) | **won 0, tied 5, lost 0** — each season's gain (+0.0034, +0.0081, +0.0078, +0.0106, +0.0029) is inside 2 × its own within-season se (0.005–0.010, over ~200 players) |
> | stage 2 | none | none | **runs: MDE +0.0048 against the declared ceiling +0.0852** (this page's own headroom) |
> | `usage` mean gain | −0.0004 at −0.0 se, wins 2/5 | −0.0004, wins 0, ties 5 | **−0.0008**, t = −0.07 on 4 df (p 0.95), t interval [−0.0342, +0.0325]; won 0, tied 5, lost 0; MDE +0.0434 against +0.0852 |
>
> **Verdict, both candidates: SHOW — KEEP `positional`. The decision does not move; two things
> about *why* do.** (1) **The 1.8 se was the wrong statistic and it understated `own_k`.**
> Clustering on the season turned 1.8 into **4.9**, which is the opposite of what the repo's
> usual worry (rule 3) leads one to expect, because the five seasons agree closely: the
> between-season scatter of `own_k`'s gain is small beside what 1,039 pooled rows implied. It
> is the first candidate in this record whose pooled interval excludes zero on a design that
> can run. (2) **It is blocked by the other half, and not by the sign.** Every season's gain
> sits inside its own noise, so none is a win and the tie blocks ADOPT (ADR-0019, #335). The
> verdict sentence's "the sign is not consistent across seasons" is the rule's wording for any
> season that is not a win and here means *tied*, not *lost* — all five are positive. This is
> exactly the pre-registered case (`docs/gate-power.md`, #343) of `own_k` becoming a live
> candidate, and the tie rule is what holds it: #381's question, not this page's. **`own_k`
> took +0.0065 of the +0.0852 headroom, 7.7%, and that is unchanged.**
>
> The `wins 5/5` and `wins 2/5` above were already superseded by #335's tie-aware rule before
> this ticket; they are restated here, with their prior values kept, because #335's own comment
> on #311 lists this page.
>
> > **Restated 2026-10-06 (#381, re-run; the paragraph above is kept).** *Prior value, both
> > candidates: SHOW — KEEP `positional`, "the tie blocks ADOPT".* **Now: SHOW — KEEP
> > `positional` for both, on "0 resolved of 5, 5 abstained".** The verdict does not move; what
> > blocks it does. Under (C) (#381, ADOPTED the same day) a tie no longer blocks anything by
> > itself: ADOPT needs a resolved season and every resolved season a win, and here **no season
> > resolves**: all five Abstain (`own_k`'s each gain inside 2 × its own se; `usage`'s likewise),
> > so the Gate can neither adopt nor remove. Without that guard `own_k`'s pooled interval
> > (t = +4.93, [+0.0029, +0.0102]) would have adopted on "every resolved season won" over zero
> > seasons; it is the case the zero-resolved rule exists for. Re-run
> > `--fit --seasons 2019,…,2025`: gains, SEs and intervals identical to the digit. The verdict
> > sentences now read "0 resolved of 5, 5 abstained (won 0, tied 5, lost 0 of 5 seasons)".
> > Ledger: `player_spread`, both candidates, 2026-10-06, `resolved: 0, abstained: 5`.

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
