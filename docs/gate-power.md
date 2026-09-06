# What four seasons can settle: the rule, written before the numbers

Pre-registration for the measurement that decides issues #37-#52. **Written and committed on
2026-09-06, before the measurement was run.** The commit order is the point: if the bar is set
after the number is known, the bar is not a bar.

This repo already pre-registers predictions -- `docs/track-record.md` rule 1 counts one only if
its commit predates kickoff. A measurement that decides sixteen tickets deserves the same
treatment, and ADR-0007 says a measurement that steers the product is committed code rather
than a figure in a commit message.

## The question

Every gate here asks whether an arm beat the simplest thing that already works, and adopts on
two conditions together (ADR-0019): a pooled interval excluding zero, and the sign holding in
every held-out season.

What no gate currently reports is **what it had the power to detect at all**. The interval is
bootstrapped over paired observations, and each gate names its own independent unit --
`weekly_gate.compare` clusters on `("season", "roster")`, `backtest.compare` and
`lineup_gate.compare` treat the row as the unit. Issue #45 proposes clustering on the *season*,
on the argument that within-season rows are near-identical and the honest replication count is
the number of seasons.

There are four seasons. Four is a small number of independent observations, and an interval
built on four clusters is wide. So the question this measurement answers is not "which arm
wins" but the prior one:

> **Is any of these gates able to detect an effect the size of the one it reports, once the
> season is the unit of replication?**

If the answer is no, then fitting coefficients and applying dispositions is work nobody is
entitled to adopt the results of, and the honest output of the programme is a published null.

## What is measured

For each gate, on its existing paired data, with nothing else changed:

1. The bootstrap is re-run with `cluster=("season",)`.
2. **MDE** -- the minimum detectable effect at 80% power, two-sided 5% -- is computed from the
   same bootstrap that produces the interval, as `(z(0.975) + z(0.80)) * SE`, where `SE` is the
   standard deviation of the bootstrap draws. Not a t approximation, and not a separate
   simulation: the draws that give the interval give the standard error.
3. The gate's reported effect and its interval, under both the current cluster and the season
   cluster, so the widening is visible rather than asserted.

## The bar, set now

A gate's **comparator** is the largest effect it could conceivably show. Issues #42 and #43
build that comparator as a foresight ceiling and are not done, so this measurement runs in two
stages and both rules are fixed here, now.

**Stage 1 -- available today, comparator is the gate's own reported effect.**

> If a gate's season-clustered MDE exceeds the absolute effect that gate currently reports,
> then that gate cannot reliably detect the effect it is reporting. The finding is recorded
> with both numbers.

This is a weaker statement than stage 2 and it is not a licence to adopt anything. A gate that
passes stage 1 has only shown that its MDE is below its own point estimate, which is a low bar
by construction -- the point estimate is what the noise produced.

**Stage 2 -- after #42 and #43, comparator is the measured ceiling.**

> If a gate's season-clustered MDE exceeds its foresight ceiling, that gate is **declared
> unable to run on four seasons**. It is recorded with the numbers, and the tickets that fit or
> apply constants for it close as *not planned* -- not as *failed*.

The distinction matters. *Failed* would say the arm lost. *Not planned* says the design cannot
answer the question with the data that exists, which is a different and more useful thing for a
future reader to know.

## What this measurement cannot do

- It cannot rescue a gate by finding a different clustering. The cluster is a claim about what
  an independent observation is, and #45's claim -- that within-season rows are near-identical
  -- is either true or false about the data, not chosen to suit the power.
- It cannot be re-run with a different seed until it passes. The seed is fixed at the value the
  gates already use.
- It cannot license adoption. Passing stage 1 or stage 2 means the gate is *able to run*, not
  that anything won.

## What happens either way

If the gates are underpowered, the programme's remaining output is this document plus the
numbers, and the fitting tickets close unstarted. That is a shorter and more honest programme
than sixteen tickets of tuning, and it is the result `docs/lambda-sweep.md` already set a
precedent for: a sweep that found nothing to tune, kept in the tree with its harness, because
the null is the finding.

If they are not underpowered, the chain runs as planned and this document is the reason each
verdict may be believed.
