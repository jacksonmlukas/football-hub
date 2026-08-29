# Parameter uncertainty: the quantity ADR-0012 did not measure

**Pre-registered 2026-08-29, before either run.** Proposed in
[what-the-field-knows.md](what-the-field-knows.md) after the waiver gate failed on a winner's
curse and two shrinkage variants failed to fix it.

## The distinction, which is the whole idea

[ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md) closed the lineup
optimiser because `sd = k·√mu` makes spread a deterministic increasing function of the mean
(r = 0.985), so ranking by `mu` and ranking by any increasing function of `(mu, sd)` give the
same order. It then withdrew its own re-run clause on the grounds that **per-player weekly
volatility** beyond the positional constant is ±9.3% and not estimable.

**That is a different quantity from how well we know his mean.** Measured 2026-08-29 on the
panel: median within-player weekly sd is **5.06**, and the predictive sd of a *projection* is

| games played before the week | se of the mean | predictive sd | vs 12 games |
|---|---|---|---|
| 1 | 5.06 | 7.15 | **+36%** |
| 2 | 3.58 | 6.20 | +18% |
| 4 | 2.53 | 5.66 | +7% |
| 12 | 1.46 | 5.27 | — |

**Four times larger at the thin end than the volatility ADR-0012 called too small, and
trivially estimable, because it is just `n`.** And it is **not a function of `mu`** — it is a
function of `n` — which is the exact condition ADR-0012 named as what would revive the
optimiser.

## Experiment A — rank waiver adds by a lower confidence bound

**The defect it targets.** [weekly-gate.md](weekly-gate.md) diagnosed the churn loss as a
winner's curse: a waiver pick is the *maximum* of a noisy estimator over hundreds of
candidates, and among players with ≤4 games the top 400 by projection project 23.00 and score
17.12. Two shrinkage variants failed to fix it
([weekly-shrinkage.md](weekly-shrinkage.md)) because both **moved the mean**, and the mean was
already unbiased at every sample size.

**The change.** The waiver decision ranks candidates by `mu − z·se` instead of by `mu`, where
`se = sigma_pos / sqrt(n)` and `sigma_pos` is the positional weekly sd fitted on **training
seasons only**. Nothing else moves: the lineup is still set on `mu`, and the incumbent arm is
untouched.

This does not shrink anything. It penalises *uncertainty*, which is the textbook treatment for
selection bias when you take the maximum of many noisy estimates.

**`z = 1.0`**, one standard error, fixed here. Not fitted — a `z` chosen against the gate's own
metric would be fitting to the test. `z = 0.5` and `z = 2.0` will be reported as an explicitly
exploratory sensitivity and never as the result.

**Pre-stated expectation.** The churn gate improves materially — **better than −1.0**, from
−2.006 — and the frozen gate does not move at all, because a frozen roster makes no waiver
decisions. **If the frozen gate moves by more than 0.05, something is wired wrong**, since the
change is unreachable from that path.

### Result, 2026-08-29: the curse is real and is not what costs the two points

| z | churn |
|---|---|
| 0 (baseline) | −2.006 |
| 0.5 | **−1.866** |
| **1.0** *(pre-registered)* | **−1.926** |
| 2.0 | −2.470 |
| 3.0 | −2.642 |

**The prediction is wrong.** −1.926 against a pre-stated "better than −1.0", and the
exploratory sweep is non-monotone with a shallow optimum near z = 0.5 worth 0.14 points out of
a two-point deficit. **The tripwire is clean**: the frozen gate returned +0.711, unchanged to
three decimals.

So the winner's curse is real — the +5.88 tail bias is still there — and correcting for it
recovers about a fourteenth of the loss. It was diagnosed correctly and it is not the
mechanism.

### What is: we simply pick worse players

Both arms churning under identical rules, ten rosters a season, 2021–2025:

| arm | adds | the added player scored | the dropped player scored | net |
|---|---|---|---|---|
| consensus | 443 | **16.42** | 6.77 | **+9.65** |
| weekly | 477 | **13.74** | 6.93 | +6.81 |

Both churn about as often and both gain from churning. **The weekly arm's adds simply score 2.7
points less.** It is not that we churn too much, or drop the wrong people, or read our own
estimate too optimistically — three things fixed and none of them mattered. We identify worse
waiver adds.

Which is the timeliness thesis pointing the other way for once. A waiver-pool player has almost
no history to process, so a model that reads only box scores is competing against a human
ranking that reads a coach's press conference. **This is the one place in the system where
consensus's information advantage is largest, and it is exactly where the model loses.**

## Experiment B — give the lineup optimiser the variance it was promised

**The change.** Per-player predictive spread becomes

    sd = sqrt( (k * sqrt(mu))^2 + se^2 )

— outcome volatility as shipped, plus parameter uncertainty. Then re-run
`hub.season.lineup_gate`, the gate that returned a structural zero.

**Pre-stated expectation.** It stays a null. The optimiser's advantage needs spread that varies
independently of the mean, and at the thick end where rostered players live this adds ~4%.
**Better than −0.02 and worse than +0.10 points per game**, and the honest outcome is
*"START YOUR PROJECTIONS"* again — this time for a reason that has been measured rather than
assumed.

Running it anyway because ADR-0012's withdrawal rests on a quantity it never measured, and
"we checked and it is still nothing" is worth more than an unexamined clause.

### Result, 2026-08-29: no longer a *structural* zero, still a null

| | optimiser − projections | 95% CI | P(better) |
|---|---|---|---|
| as shipped | **+0.00** | **[−0.00, +0.00]** | 72.7% |
| with parameter uncertainty | **+0.18** | **[−0.28, +0.79]** | 75.5% |

**The verdict is unchanged and the structure is not.** ADR-0012's zero was *structural* — the
interval was [−0.00, +0.00] because `sd` was a deterministic function of `mu` and the optimiser
was choosing among orderings that could not differ. Adding `sigma_pos/√games` opens the interval
to [−0.28, +0.79]: the optimiser now has variance to read, and reading it is worth **+0.18
points per game and not significant**.

My pre-stated range was "better than −0.02 and worse than +0.10", so the magnitude half is
wrong — it moved more than I said — and the verdict half held: *START YOUR PROJECTIONS*.

**This amends ADR-0012's reasoning without touching its decision.** The re-run clause was
withdrawn on the grounds that *"there is no known route to that, and the headroom would not
justify one."* There is a route, it is one line, and the headroom is +0.18 [−0.28, +0.79] —
which is a measured null rather than an assumed one, and that is the difference worth having.

## What would make either adoptable

The pre-registered bar, unchanged: beat the incumbent in **every** held-out season and clear
`MIN_SE`. Neither experiment gets a new bar for being cleverer than the last one.
