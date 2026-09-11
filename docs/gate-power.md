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

> If a gate's season-clustered MDE exceeds its **declared ceiling arm**, that gate is
> **declared unable to run on four seasons**. It is recorded with the numbers, and the tickets
> that fit or apply constants for it close as *not planned* -- not as *failed*.

> **Amended 2026-09-11 under #138, before any stage 2 run.** As pre-registered on 2026-09-06
> this read *"its foresight ceiling"*. #43 then built the lineup gate's ceiling as a
> **variance oracle** -- mu untouched, sd taken from the realised spread -- deliberately and
> correctly: a foresight lineup is the best XI after the fact and bounds nothing interesting,
> where a perfect *spread* bounds what any spread model could deliver. So the rule as written
> named a quantity that gate does not produce under the arm it runs with. Three options were
> weighed: amend the wording, expose the foresight arm under `--ceiling`, or report both and
> apply the rule to foresight. The second and third both apply the *weaker* comparator --
> foresight is strictly the larger of the two arms, so a smaller ceiling declares underpowered
> more readily, and the variance oracle is the stricter test. The wording was amended instead.
> What was known at the time: no stage 2 number existed for any gate, so this changes which
> comparator each gate is measured against and not which side of it any gate falls. Each gate
> now declares its arm by name where it prints -- the draft gate *perfect foresight, the
> season known in advance*; the weekly gate *perfect foresight*; the lineup gate
> *variance-oracle* -- and `tests/contracts/test_each_gate_declares_its_ceiling_arm.py` holds
> that all three declare, that no two share a name, and that this paragraph does not revert.

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

---

# The measurement, 2026-09-06

Run against the rule above, which was committed first in `a41301b`.

## Draft backtest

The only gate with a persisted paired frame: `data/processed/p0b_paired.parquet`, 80 rows,
one per (season, draft), seasons 2022-2025. The other two gates assemble their pairs from the
network and have nothing on disk, so they are **not measured here** -- see below.

| clustered on | clusters | effect | 95% CI | bootstrap SE | MDE (80%) |
|---|---|---|---|---|---|
| row (current) | 80 | -19.66 | [-23.16, -16.20] | 1.79 | **5.03** |
| season | 4 | -19.66 | [-26.68, -11.45] | 3.67 | **10.29** |

> **The two MDEs in this table are superseded, restated 2026-09-07 under #45.** They were
> computed with a normal quantile, `(z(0.975) + z(0.80)) * SE`, which is what item 2 of the
> rule above specified. At four clusters that is the wrong reference distribution, and the
> corrected figures are below. The SEs, the effects and the intervals are unchanged and were
> never in question — this restates the power arithmetic built on them, not the measurement.

The season-clustered interval was checked against `experiment.summarise(cluster=("season",))`
and reproduces it exactly, so the standard error above comes from the same bootstrap that
produces the published interval rather than from a second one that happens to agree.

**Stage 1: this gate passes.** Clustering on the season roughly doubles the standard error and
widens the interval by about 60% -- the effect #45 predicts -- but the reported effect is
nearly twice the season-clustered MDE, and the interval still excludes zero.

Two things this does not say. The effect is **negative**: the arm under test is behind its
incumbent by about twenty points on this frame, so what the widening changes is the confidence
in a loss. And stage 1 is a weak bar by construction, as the rule says -- passing it means the
MDE sits below the point estimate the noise itself produced. The real bar is stage 2 against
the foresight ceiling, and #42 has not been built.

## Re-measured 2026-09-07, and the effect does not reproduce

Two re-runs of the draft gate over the same four seasons and the same 80 (season, draft)
pairs. Neither reproduces the published figure, and the second isolates how much of the gap
the pick-noise refit accounts for.

| run | effect | 95% CI | estimator | constants | data digest |
|---|---|---|---|---|---|
| published (P0b) | **−19.66** | [−23.16, −16.20] | pre-repair | 1.00 / 0.253 | not recorded |
| re-run A | **−12.48** | [−14.97, −10.03] | repaired | 1.00 / 0.253 | `621cb5dd` |
| re-run B | **−11.59** | [−14.52, −8.64] | repaired | 1.31 / 0.169 | `621cb5dd` |

**The refit is not the explanation.** Between A and B the only change is the pick-noise
constants #150 shipped, and the effect moves **0.89** points. Between the published figure and
A it moves **7.18**, and nothing in the tree accounts for that. The refit is eleven percent of
the total movement.

**The movement is in the code, not the data.** A and B carry the same digest over the same
eight pinned sources, so the input bytes are identical and the difference is entirely what the
harness did with them. The published run recorded no digest at all, which is why the first leg
cannot be closed the same way — and is its own argument for #165, which made a partially
pinned run declare itself rather than print a clean-looking digest over the wrong set.

**The verdict has never been in question.** All three runs are worse in 4 of 4 held-out
seasons, all three intervals exclude zero, and `P(optimizer better)` is 0.0% in every one. The
pre-registered rule says REMOVE on any of them. What is in question is whether any of the
three magnitudes means anything, and until the 7.18 is attributed, none of them should be
quoted as the size of the effect.

**A third run rules out a single cause.** `90a9bbb` — the commit that bound the board to its
own preseason and dropped 814 of 1,372 players from 2025 — was the strongest candidate for the
7.18. Run at that commit, with the same old constants the published figure used:

| point in history | effect | commits since the one above |
|---|---|---|
| published (P0b) | −19.66 | — |
| at `90a9bbb` | **−15.65** | 259 |
| session start | −12.48 | 11 |
| shipped constants | −11.59 | the refit |

So the effect did not step once. It moved **4.01** points over the first 259 commits, **3.17**
over the next 11, and **0.89** with the refit. No commit in the shortlist owns it, and the
recent stretch moved it faster per commit than the long one.

**That is the finding, and it is worse than a wrong number.** The data digest is `621cb5dd` in
every one of these runs — the same eight pinned sources, the same bytes — so nothing about the
inputs changed across 270 commits while the answer moved by eight points. A harness that
returns a different effect at fixed inputs as unrelated code lands is not measuring the arm it
names, and no figure it has produced can be quoted as the size of the effect. The three
published occurrences of −19.66 are not one stale number; they are a number of unknown
provenance.

What this does **not** disturb: every run is worse in 4 of 4 held-out seasons with an interval
excluding zero and `P(optimizer better)` at 0.0%. The REMOVE disposition rests on the sign and
the consistency, neither of which has moved.

Paired rows for all three re-runs are under `data/processed/gate/`, which is gitignored, so a
future re-run has something to diff against on this machine and nothing on a fresh clone.
Issue #190 carries the attribution work.

## Why the other two gates are unmeasured

`weekly_gate.compare` and `lineup_gate.compare` build their paired frames from the network at
run time and persist nothing, and `data/processed/` is not publishable, so neither gate can be
measured offline or re-measured in CI. Measuring them needs either a run with credentials or
a frozen paired frame of their own -- the second is the better answer and is the same shape as
the panel archive #108 froze.

Their clusters also differ from the backtest's: `weekly_gate` already clusters on
`("season", "roster")`, so its current interval is not the row-level one this table compares
against. **This result does not transfer to them.**

## A finding for #45, from trying it

Adding the MDE to `summarise`'s return is not additive. `paired_report` already prints an
`MDE at 80% power` line whenever the key is present -- the reporting plumbing was written
ahead of the producer, as `summarise`'s own docstring says it was -- so supplying the key
changes every gate's printed output, and moves the sweep digest that
`tests/unit/test_experiment.py` pins.

That is the digest doing its job. It also means this is a change to published gate output
rather than a measurement, so it belongs to #45 with its blockers done and a before/after on
the affected verdicts, not to a probe. It was tried, reverted, and is recorded here so #45
starts knowing the plumbing exists and the digest will move.

---

# Restated 2026-09-07: the MDE's reference distribution, under #45

The rule above, item 2, specified `(z(0.975) + z(0.80)) * SE`. **The quantile is wrong at four
clusters and the rule is corrected to `(t(0.975, k-1) + z(0.80)) * SE`**, `k` the cluster
count. The SE still comes from the same bootstrap that produces the interval; that half is
unchanged and is the half the original wording was written to protect.

`docs/method.md` rule 13: this is a dated restatement beside the original, not an edit over
the top of it. The superseded figures keep their text in the table above.

**Why it is a correction and not a preference.** `SE` is estimated from `k` observations, so
the reference distribution for a two-sided interval on it is a t on `k-1` degrees of freedom.
With `k = 4` that is `t(0.975, 3) = 3.1824` against `z(0.975) = 1.9600` — the normal
understates the MDE by **1.44x**. Using the normal makes the published SE and the published
interval consistent with each other rather than correct. Only the 0.975 term is a t; the 0.80
power term stays normal, which is the standard form.

| gate | clusters | t(0.975, k−1) | SE | MDE, superseded (normal) | **MDE, restated (t)** |
|---|---|---|---|---|---|
| draft, row-clustered | 80 | 1.9905 | 1.79 | 5.03 | **5.07** |
| draft, season-clustered | 4 | 3.1824 | 3.67 | 10.29 | **14.77** |

At 80 clusters the correction is **1.1%**; at 4 it is **44%**. That is the whole shape of the
thing — the normal quantile is a good approximation exactly where the cluster count is large
enough not to need it, and the case this repo is actually in is the other one.

*(On the SE rounded to 3.67 the normal arithmetic lands on 10.28 rather than the 10.29 in the
table above; the SE that reproduces 10.29 exactly is 3.6729, which rounds to 3.67 and gives a
t MDE of 14.78. The ratio, which is free of the rounding, is 1.4364 either way. The restated
figures above are quoted on the published 3.67.)*

**Stage 1 still passes for the draft gate, with a much thinner margin.** The comparator is the
gate's own reported effect, |−19.66|, and the restated season-clustered MDE is 14.77:
19.66 / 14.77 = **1.33x**, where the superseded arithmetic read 1.9x. It passes, and stage 1
was already "a low bar by construction". Nothing about the REMOVE disposition moves — it rests
on the sign and the 4-of-4 consistency, neither of which any of this touches — and the effect
itself is under attribution as #190, which is the larger question hanging over the number.

**For the weekly gate the correction is the difference between underpowered and not remotely
close**, and that is now measurable: see the restatement in
[weekly-blend-gate.md](weekly-blend-gate.md).

## Both gates now report this themselves

The finding recorded above — that adding the MDE is not additive — is discharged. `summarise`
computes `se` and `mde` from the interval's own bootstrap; `paired_report` prints the MDE line
whose plumbing was already there; the block sweep digest moved once, from `bf5b1af281ea0f0c`
to `5c1be1a2ba3ba17f`, and `test_every_block_grew_exactly_the_mde_line_and_nothing_else` is
what says the move is only that. All three harnesses now pass `SEASON_CLUSTER`, asserted on
the call by `tests/contracts/test_gates_cluster_on_the_season.py`.

**Stage 2 is now mechanised but is not thereby answered.** `experiment.gate` reports
NOT-RUNNABLE when a gate's MDE exceeds its ceiling, ahead of every branch but VOID
([ADR-0019](adr/0019-a-gate-requires-every-season.md), amended). It fires only when a gate
hands in a measured ceiling, so the two gates that have never measured one still reach the
verdicts they reached before. What stage 2 needs to actually run is what the section above
already says it needs: frozen paired frames for the two network-built gates.

**And for the lineup gate, stage 2's comparator is still open.** This document pre-registers
that stage against a *foresight* ceiling; #43 built a **variance oracle** and argued for it,
on the grounds that both arms of that gate already share `mu`. The two bound different
questions, and which one this gate declares is **#138** — a pre-registration question, not an
implementation detail. The mechanism takes it as a parameter and declares nothing:
`hub.season.lineup_gate.DECLARED_CEILING_ARM` is the single line #138 sets.

# Restated 2026-09-07: every number above was measured on a harness with a foresight leak

Under [method.md rule 13](method.md). **Nothing in the tables above is edited.** They record
what those runs returned, and they still do. What is restated is what they are evidence *of*,
because the harness that produced them has been found to have been scoring one of its two arms
partly on a draft it had already seen.

## The mechanism

`backtest.compare` built one integer per draft, `seed + 1000 * season + k`, and handed the same
integer to two different levels of the same experiment: to the room, as the seed the eleven
opponents were drawn from, and to `optimizer_strategy` as `seed`. Inside, `win_probability`
opened `default_rng(seed + k)` for each of its `n_draft_sims` evaluation futures. At `k = 0`
that is `default_rng(room)` — bit-for-bit the room being played.

`simulate_remaining_draft` consumes its generator exactly once, at the top, before either
`forced` or `state` is read, so the draw depends only on the seed and on `mu_pick`. **One of
arm B's twelve evaluation futures was therefore the room it was about to be scored in, at every
one of its picks.** Arm A plays no futures at all and got nothing of the kind.

Two further consequences of the same arithmetic:

- consecutive drafts' rooms sat one apart, so they shared **11 of their 12** evaluation
  futures — a repeated-measures dependence between rows that no clustering key named;
- the season-simulation seeds `room + 1000 + k` landed exactly on the next season's room
  lattice, **20 times per season pair** — and [ADR-0019](adr/0019-a-gate-requires-every-season.md)
  ties adoption to consistency *across held-out seasons*, which is the comparison that
  contaminates.

Fixed under #195: the room, the rollouts and the season simulation are now three spawn
coordinates of one `SeedSequence` root per `(season, draft)` rather than three offsets on one
integer. Offsets on a shared line stay separate only while nobody changes the counts; spawn
coordinates cannot meet at all. `tests/unit/test_backtest.py` asserts the disjointness on the
generator states, and asserts it in the form that fails on the pre-#195 code.

## The direction, which is the part that matters

**The leak favours arm B.** Arm B is the arm under test; arm A is the incumbent. So every
effect in the tables above — all of them negative, all of them against arm B — is if anything
**understated**. A harness that hands the treatment arm a look at its own grading draft and
still measures it losing by eleven to twenty points has not overstated the loss.

That is why this restatement does **not** disturb the REMOVE disposition. It rests on sign and
on consistency across held-out seasons, and the leak runs the other way from the sign. What it
does disturb is any reading of the tables as *magnitudes*, which the section above had already
withdrawn on separate grounds — the eight points of drift at a fixed data digest (#190).

## What a re-run should now be expected to show

**#194 measures this**, and it should now be able to. Setting expectations before the number
arrives, which is the point of writing them here:

1. **The effect should move, and the sign should not.** Removing a one-directional advantage
   from arm B should push the measured effect *further* from zero, not towards it — a larger
   loss for championship equity, not a smaller one. An effect that moves towards zero, or
   changes sign, would mean the leak was not the dominant term and something else is being
   measured; either would be a finding in its own right and belongs in #190.
2. **The run-to-run spread should narrow, and this is #194's actual question.** Two of the
   three collisions were between rows *within* one sweep. With them gone, the eighty rows are
   closer to the eighty independent observations the frame's shape claims, so the same sweep
   re-run at a different seed should vary less than it did.
3. **The size of the shift is bounded by `1/n_draft_sims`.** One of twelve futures was leaking,
   so a first-order guess is that arm B loses about a twelfth of whatever the leak was worth.
   That is a guess and not a prediction; it is written down so that a much larger movement is
   recognised as needing its own explanation rather than absorbed.
4. **The seasons should stay 4 of 4.** The cross-level collision was the one that could plausibly
   have manufactured agreement between adjacent seasons. If consistency across held-out seasons
   *survives* its removal, ADR-0019's second condition is standing on its own for the first
   time.

None of the four is a licence to quote a magnitude. #190 is open on eight points of unattributed
drift at a fixed data digest, and #195 and #196 together are its leading mechanism rather than
its resolution.

## What was measured on which board is now recorded

Separately, under #196: a run stamped `cfg_digest` and `data_digest`, and the second is a digest
of the upstream *source bytes*, one layer above the Board. Nothing hashed the Board itself,
which is what `compare` is actually handed — so the table above can say `621cb5dd` for three runs
and still not name the frames they were played on. `board_digest` now stamps that, and the
paired output carries the commit that produced it. The rows above predate the stamp and cannot
be given one retroactively; a re-run under #194 will carry all four.
