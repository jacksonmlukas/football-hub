---
name: model-eval
description: Use when evaluating, comparing, or gating a forecasting model — scoring predictions, checking calibration, running a backtest, or deciding whether a model beats the market baseline. Triggers on "is this model any good", "calibration", "backtest", "log-loss", "CLV", "go/no-go", "does it beat the spread".
---

# Model Evaluation

Extends `sim-eval` to football. Same metric machinery, different acceptance thresholds.

## Non-negotiables

1. **Benchmark is the closing line.** Backtesting against openers measures stale-line capture you
   could not have executed. If closing lines are missing for a game, drop the game.
2. **Temporal splits only.** No random k-fold. Week *t* is never predicted with week *t+1* data.
   A rating fit on the full season and evaluated on week 6 is leakage, not a result.
3. **Log-loss and CLV. Never win rate.** Win rate over 17 weeks has a standard error wide enough
   to make a coin flip look skilled. Report Brier as a secondary.
4. **Every run writes predictions with a timestamp and model hash** to
   `data/processed/preds/{model}/{ts}.parquet`. December must be able to audit August.

## The comparison that matters

Not "is the model accurate." It is: **does the model add anything over the closing line alone?**

```
uv run python -m hub.models.eval --compare market_only,market_plus_residual --split temporal
```

Report the log-loss delta with a bootstrap interval. If the interval crosses zero, you have no
evidence of edge. Say so plainly rather than reaching for a subgroup where it looks better.

## Calibration

Reliability diagram in ten bins, plus empirical coverage of the conformal intervals against
nominal. Coverage drift is the early warning that exchangeability broke — usually from a
mid-season structural change like a starting QB injury.

Recalibrate on a rolling window. Do not refit on the full season, since that reintroduces the
drift you are correcting for.

## Gates route, they do not kill

Priority order for this project is: (1) win the league and pools, (2) understand the methods,
(3) portfolio artifact, (4) demonstrable market edge. Edge ranks last, so **a gate decides whether
a model feeds a production decision — never whether the model gets built.**

| Track | Gate | Deadline | Pass | Fail |
|---|---|---|---|---|
| D: Survivor IP | Feasible 18-week assignment | Week 1 | Drives picks | Blocking bug, fix it |
| C: Conformal | Coverage within 3pts of nominal | Week 2 | Sizes lineup risk | Widen window, retry |
| A: Bayesian ratings | Beats closing line, held-out log-loss | Week 4 | Feeds the blend | Weeks 1-4 use only |
| B: Sequence model | Reproduces league marginals | Week 6 | Continue | Fix tokenizer |
| B: Sequence model | Per-player fantasy beats plain xFP | Week 8 | Feeds projections | Stays a study |

Tracks D and C are load-bearing: a failure there is a bug blocking objective 1. Tracks A and B are
learning-first. "Stays a study" is a legitimate terminal state, not a loss.

**Write up negative results in `docs/notebook.md` with the same care as positive ones.** A
documented finding that a play-level sequence model does not beat an opportunity-share baseline
on 1.2M plays is worth more for objectives 2 and 3 than a model that quietly works.

## Derive before you import

Objective 2 outranks objective 4, so build the mechanism once by hand before reaching for the
library: split conformal from residual quantiles before MAPIE, the survivor DP on a 4-week toy
before pulp, a random-walk posterior on 8 teams before 136. Keep the hand-rolled version as a
test oracle. It catches library misuse that a passing pipeline hides.

## When not to use

Not for operational decisions. This skill judges models. `weekly-slate` runs them.
