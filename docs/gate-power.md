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

> **The weekly gate's ceiling, measured 2026-09-21 under #376.** A measurement, not a
> modelling change: it moves no model, no constant, no published effect. `hub.season.weekly_gate
> --run --ceiling` (seasons 2022-2025, drafts=20, seed 0, frozen rosters, restricted, mask_pool):
>
> | | this run |
> |---|---|
> | weekly − consensus | −1.156 |
> | 95% CI, percentile | [−1.761, −0.259] |
> | clusters | **3**, not 4 |
> | season-clustered MDE (80% power) | 1.926 |
> | **ceiling (perfect foresight)** | **+10.799** |
> | verdict | REMOVE |
>
> **Stage 2 passes, and passes by a wide margin: the ceiling (10.799) clears the MDE (1.926)
> roughly 5.6x over,** so `experiment.gate`'s NOT-RUNNABLE branch does not fire -- the weekly
> gate is not the underpowered one, whatever its verdict says about the arm. That is what #376
> asks this ticket to end: "NOT-RUNNABLE until this ticket measures the ceilings" no longer
> describes the weekly half.
>
> **Flagged, not chased down:** this run scored **3** clusters where the gate's default
> `--seasons` names four -- season 2022 produced no row in the per-season breakdown the run
> printed, so it silently dropped rather than erroring. That is why this run's own MDE (1.926)
> reads wider than the 0.768 this page restated 2026-09-07 on four seasons, and it is a
> question about this run's coverage, not about the ceiling: whatever the true 4-season MDE
> is, it sits between 0.768 and 1.926 and the ceiling clears either end by several times over.
> Chasing the missing season is out of this ticket's scope (S6/#376 asks for the ceiling, not
> a coverage audit); worth a look before this table is treated as the last word on the MDE.
>
> `state/gate-width.json` is untouched by this run: `run_gate`'s `record_width=True` wrote a
> `weekly` entry with `clusters: 3` and `width: 1.503` to it, and that write was reverted
> (`git checkout -- state/gate-width.json`) immediately after, so #362's evidence -- the
> `requires_review` entries the live file already carries -- stays exactly what it was before
> tonight. `state/gate-width.2026-09-20.pre-ceiling.json` is the byte-identical pre-run copy.
>
> The draft gate's ceiling (#42, "n_drafts=20") is not run tonight: it contends for CPU with
> #368's leverage study and #376's own acceptance criteria say it may land in a separate
> commit. Still owed.

> **Resolved 2026-09-21 (#378): the flag above was the bug, not a coverage curiosity.**
> `weekly_gate.compare`'s walk-forward split (`experiment.expanding_seasons`) trains each
> scored season on every season before it, so the *earliest* season handed to it is always
> spent as training data and never itself scored — `docs/weekly-blend-gate.md`'s Reproduce
> section already worked around this by naming five seasons so the buffer is a sacrificial
> fifth. The run above did not: it took `weekly_gate`'s own `--seasons` default, which named
> exactly the four held-out seasons with nothing ahead of them, so `expanding_seasons` dropped
> 2022 — the earliest of *those four* — rather than a season nobody meant to score at all. The
> run's own per-season table said so, correctly; nothing printed alongside it said that three
> was short of four, which is the defect and the reason this was filed as #378 rather than
> corrected in place.
>
> Fixed two ways: `weekly_gate`'s `--seasons` default now names the buffer season ahead of the
> four (`2021,2022,2023,2024,2025`, matching the Reproduce section above), and
> `hub.season.weekly_gate_data.SeasonDropped` refuses a run outright if any season goes missing
> from the scored output beyond that one expected buffer drop — a partition absent on disk, a
> VOID condition, a join failure, whichever it is, now stops the run instead of quietly
> shortening its cluster count. Every run also now prints what it asked for beside what it
> scored (`docs/gate-power.md` and `#378`'s acceptance criteria (b)).
>
> Re-run at the corrected default, same recipe otherwise (`uv run python -m
> hub.season.weekly_gate --run --ceiling`, drafts=20, seed 0, frozen rosters, restricted,
> mask_pool):
>
> | | #376's run (wrong) | **corrected** |
> |---|---|---|
> | weekly − consensus | −1.156 | **−1.147** |
> | 95% CI, percentile | [−1.761, −0.259] | **[−1.605, −0.556]** |
> | clusters | 3, not 4 | **4** |
> | season-clustered MDE (80% power) | 1.926 | **1.119** |
> | **ceiling (perfect foresight)** | +10.799 | **+10.873** |
> | ceiling / MDE | 5.6x | **9.7x** |
>
> **Stage 2 still passes, by a wider margin than #376 reported**, not a narrower one — the
> corrected MDE is smaller than the mistaken one, because a fourth genuine season narrows the
> interval a dropped-season run cannot. The weekly gate is not the underpowered one, at the k
> its own docs and the ADR-0019 amendment's rule-16 table both name.
>
> **This run's own verdict is not the published one and is not read as such.** With no
> `--shrink` and `drafts=20` — the ceiling-measurement recipe #376 set, not the published
> gate's `--shrink mae-market --drafts 40` — this run's season table (2022 −1.120, 2023
> −1.448, 2024 −0.259, 2025 −1.761) reads SHOW under the post-S1 tie-aware rule (won 0, tied 1,
> lost 3 of 4) rather than the REMOVE the mistaken 3-cluster run printed. That is the tie rule
> reading 2024's small, noisy gain as indistinguishable from zero once four seasons — rather
> than three with a different season composition — are on the table, and it says nothing about
> `docs/weekly-blend-gate.md`'s published −1.004 REMOVE, which this ticket does not re-run and
> does not move.
>
> `state/gate-width.json` is restored after this run the same way #376's was:
> `git checkout -- state/gate-width.json` once the numbers above were recorded, so #362's
> evidence is untouched by this ticket either.
>
> ADR-0019's rule-16 weekly row is unstruck with this: the row was computed at k=4 all along
> and the number that briefly contradicted it was reading a different, accidental k.

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

## Measured 2026-09-12 — the run-to-run spread at one commit, and it is not the drift

Issue #194. The same commit, `e14ab56` (the tree after #195's seeding tree, #187's PSD
repair and #155's pick-noise refit), run twice with two root seeds, the same four seasons,
twenty drafts a season, 12 × 250 sims, Board digest `6769f585` in both — which is the run's
own condition for two runs being comparable at all. The tree was dirty only by an untracked
symlink to `data/`; the code is that commit.

| | seed 0 | seed 7 | difference |
|---|---|---|---|
| optimizer − market | **−12.19** | **−11.88** | **0.31** |
| 95% percentile CI | [−18.33, −6.05] | [−17.09, −6.67] | |
| 95% t CI, 3 df | [−22.06, −2.32] | [−20.34, −3.42] | |
| MDE at 80% power | 12.48 | 10.70 | |
| 2022 | −19.95 | −18.23 | 1.72 |
| 2023 | −5.38 | −6.00 | 0.62 |
| 2024 | −6.72 | −7.34 | 0.62 |
| 2025 | −16.71 | −15.95 | 0.76 |
| seasons worse | 4 of 4 | 4 of 4 | |
| verdict | REMOVE | REMOVE | |

**The two-seed spread is 0.31 points on the mean and under two on any season.** The
four-point table above moved **8.07** between the published figure and the shipped
constants, and 3.17 over eleven commits at one data digest. A difference between two
independent re-draws of this harness carries a standard error near 0.5 (the four per-season
differences have sd 0.53), so the historical movement is fifteen-plus of those. **The drift
is real and it is in the code**, which is the second of #194's two worlds: #190 is an
attribution problem and not a precision problem, and nothing in it is closed by this.

**Against the four expectations registered above, and the fifth from #187.**

1. *The effect should move away from zero, and the sign should not.* Against re-run B
   (−11.59, the last run before #195), both seeds sit further from zero by 0.3 to 0.6, and
   the sign and all four seasons hold. A move that small is inside this measurement's own
   noise, so the direction is consistent with the expectation rather than a confirmation
   of it.
2. *The run-to-run spread should narrow.* No two-seed run exists at a pre-#195 commit, so
   there is nothing to narrow *from*; what is established is the spread's size now.
3. *Bounded by 1/n_draft_sims.* The move against re-run B is of the order the guess allowed
   and nothing larger needs explaining.
4. *Seasons should stay 4 of 4.* They do, at both seeds — ADR-0019's second condition is
   standing on its own for the first time.
5. *Zero blocks repaired by silence.* Twenty-one team blocks were repaired to PSD and each
   was printed with its eigenvalue move (largest single pairing +0.057, CLE); none was
   silent.

**What has changed since this commit, and what it means for the number.** `e14ab56` is not
the tree. Since it: #235 refit `TALENT_CV` net of absence (0.42 → 0.32; the #197 pin showed
arm B's roster move under it), #246 found four drafted players scored as zero in 2022
because the stats source spelled their nickname (Gabriel Davis, Ken Walker III, Kenneth
Gainwell, Joshua Palmer — 2022 is the season with the largest effect in every run), #226
added byes and #87 widened imputed players' spread. Each moves the arm or the scoring, so
the shipped-constants run for #245 and #246 is the next figure and not this one. Paired rows
for both seeds are under `data/processed/gate/p194_e14ab56_seed{0,7}.parquet`.

## What was measured on which board is now recorded

Separately, under #196: a run stamped `cfg_digest` and `data_digest`, and the second is a digest
of the upstream *source bytes*, one layer above the Board. Nothing hashed the Board itself,
which is what `compare` is actually handed — so the table above can say `621cb5dd` for three runs
and still not name the frames they were played on. `board_digest` now stamps that, and the
paired output carries the commit that produced it. The rows above predate the stamp and cannot
be given one retroactively; a re-run under #194 will carry all four.

---

# Pre-registered 2026-09-13: the quarterback adjustment's own gate (#291)

**Written and committed before the harness was built or run**, in the form above: the bar
first, the power arithmetic on the corrected reference distribution, then the run. The
adjustment (`hub.models.quarterback`, #218/#268) reaches published predictions and survivor
picks on a validation of *538's* adjustment against *538's* base Elo. #270 marks every row it
touches and names this gate as the pull trigger. What follows is the gate.

> **Amended 2026-09-17 (#300).** #270's pull trigger, above, is superseded: this gate is
> demoted to a diagnostic and decides neither ADOPT nor REMOVE for the module. See *Amended
> 2026-09-17 (#300)* at the end of **The bar**, below, for what replaced it and why, and
> [qb-adjustment.md](qb-adjustment.md) for what it means to a reader of a row.

## The question

Where the betting market has no live price, the adjustment moves a frozen quote by the
source's `qb_adj / 25`. It matters only when the frozen quote predates a **starter change** —
on every other game the adjustment is near zero and the frozen quote already embeds the
starter. So the gate is asked on exactly those games:

> Over games whose frozen price predates a starter change, is the quarterback-adjusted frozen
> line a better forecast of the result than the unadjusted one?

## The event set — shared with #221, built once

A **starter-change event** is a team whose starting quarterback on game *g* differs from its
starter on its previous game *g − 1* **of the same season**. The starter is observed, never
reported: the source's own starter column on the row (`qb1`/`qb2` in `hub.fetch.nfeloqb`,
the observed starter on a played row and the named one on the coming week's), or the passer
of the first pass attempt in play-by-play where the cache carries that column. The injury
report is not consulted — #221 records why (one row per player-week, timestamped Friday,
fourteen quarterbacks out across all of 2024). A change across the offseason is not an event:
the betting market priced it all summer. Postseason games are excluded, as in the rule that
licensed the adjustment. The **event game** is *g*; a game where both teams changed is one
event game with two changes on it. Event counts are reported per season beside every number.

## The two arms, and the comparator

* **Incumbent:** the **frozen price** — the last snapshot of the event game captured before
  the Eastern game day of the changed team's previous game, which is before any change could
  have been known. This is the price the adjustment would actually have replaced, and not the
  close; a gate against the close would score the adjustment against the betting market's own
  repricing and could only lose.
* **Arm under test:** the same frozen price moved by `hub.models.quarterback.apply` — the
  shipped estimator, called as `ratings` calls it, on a row labelled as having no live price —
  with the state the live path would read on the morning of the game: both teams' starter and
  `qb_adj` from the event game's own row, which the source publishes before kickoff.

  > **Amended 2026-09-13, on review, before any row existed to score.** As first written the
  > arm read the event game's *own* row — the arriving starter known with certainty. That is
  > not the shipped path. `hub.models.ratings._rated_by_week` rates a week that has kicked
  > off from `nfeloqb.state(rows, as_of=<its first kickoff>)`, rows strictly before that
  > game day (#272), and says in its own docstring that this is the seam #291 reads. That
  > state's latest row for every team is its previous game, so on an event game the shipped
  > estimator prices the *departing* starter — the source's file carries a new starter on
  > the row of his first game and on no earlier row, and a replay cannot know him before it.
  > An arm reading the same-week row replays a better estimator than the one being gated and
  > biases the difference toward ADOPT. **The arm is the shipped seam:** `nfeloqb.state(rows,
  > as_of=<the week's first game day>)` handed to `quarterback.apply`, exactly as `ratings`
  > does. The same-week-row arm is kept as an **oracle, a diagnostic and not the gate**,
  > reported beside the shipped arm under its own column (`oracle_diff`) because it answers
  > a different question #270's disposition needs — whether the mechanism could help *if*
  > the state were timely — and it is never read by the rule
  > (`tests/unit/test_starter_change.py::test_the_rule_reads_the_shipped_arm_and_never_the_oracle`).
  > The wording above is left as written, per [method.md](method.md) rule 13.
* Both arms convert a spread to a home win probability by `MarketBaseline`'s own conversion
  (`normal_cdf(spread / MARGIN_SD)`), so the two arms differ in the spread and nothing else.

**The unit** is one event game, scored by log-loss on the home result; ties score 0.5. The
paired difference is `log-loss(unadjusted) − log-loss(adjusted)`, positive when the
adjustment helped. **The cluster is the season** (`experiment.SEASON_CLUSTER`), and an
*event-season* is a season in which the archive holds at least one scored event game.

## The bar

The house rule, `experiment.gate`, and nothing beside it:

* **ADOPT** — the interval excludes zero on the positive side *and* the sign is positive in
  every event-season. The adjustment has earned its place; the `-qb` suffix stays on the row
  for the track record and the sentence *ungated in this repo* leaves
  [qb-adjustment.md](qb-adjustment.md).
* **Anything else that is a verdict** — REMOVE, or SHOW — fails the house rule, and that is
  #270's pull trigger: the module leaves `ratings` and `survivor` that day and is reachable
  only from the harness. The burden is on the arm: it is in the published path today on no
  verdict of this repo's, and a null is not a pass.
* **NOT-RUNNABLE, or VOID** — no verdict. The mark stands, and the run records what would
  make it runnable.

Two preconditions ahead of the verdict, both pre-registered here:

1. **Fewer than three event-seasons is NOT-RUNNABLE.** ADR-0019: "no gate in the repo runs
   at fewer than three, and one that did should say so rather than quietly inheriting a bar
   built for four". At two seasons the every-season half is a coin flip.
2. **Stage 2 as written above:** an MDE above the declared ceiling arm is NOT-RUNNABLE. The
   ceiling arm is **the betting market's own repricing** — the last snapshot before the game
   day, scored against the same frozen line. What the betting market recovered by the close
   is the most an adjustment built to anticipate it could recover.

### Amended 2026-09-17 (#300) — demoted to a diagnostic, and NOT-RUNNABLE is named an exemption

**The bar above is superseded as the module's ADOPT condition.** It is kept as written, per
[method.md](method.md) rule 13; what follows is what replaced it and why. The maintainer's
`ADOPTED:` comment on #300 (2026-09-17) names the line-move coefficient — sign and magnitude
against 0.132, season-clustered once two seasons exist — as the ADOPT condition, and names
this gate's NOT-RUNNABLE branch an exemption rather than a bar, under
[method.md rule 16](method.md): a gate that cannot return a verdict is not a bar, it is an
exemption, and it must be named as one in the pre-registration that created it.

**ADOPT no longer comes from this gate.** The ADOPT condition for the module is #221's
line-move coefficient, below (*Amended 2026-09-17 (#300) — the ADOPT condition*, under
*Pre-registered 2026-09-13: the line-move study's power (#221)*). This gate's four branches
above still run on every archive and still print — ADOPT, REMOVE, SHOW and NOT-RUNNABLE keep
their meanings as answers to the log-loss question stated here — but no branch of this gate
licenses shipping the module or pulls it from `ratings` and `survivor`; #270's pull trigger,
named in the branches above and in this section's own intro, is superseded with them. The
gate is read beside the coefficient as a **diagnostic**, its power requirement stated beside
it exactly because a diagnostic that decided ADOPT with no power requirement attached — able
to fire only from its own NOT-RUNNABLE branch — is the permanent licence this amendment
closes: **29 event-seasons at 80% power**, against the pilot's target of 0.0180 log-loss and
between-season sd of 0.0332 (*Measured 2026-09-13*, below), a bar this gate will not clear in
this project's lifetime at one event-season a year.

**Precondition 1 is an exemption, not a bar.** "Fewer than three event-seasons is
NOT-RUNNABLE" excuses this diagnostic from firing before it has enough seasons to say
anything about the log-loss question; it was never, and is not now, a condition on the
module's ADOPT or REMOVE status. A NOT-RUNNABLE verdict here — which is what this archive
prints today, and will keep printing for decades at the season-a-year rate the pilot implies —
changes nothing about what #221's coefficient is free to decide. This is [method.md rule
16](method.md) applied: a branch this gate cannot leave is named an exemption here rather than
left to be discovered from thirty seasons of the same print.

## The MDE, on the corrected reference distribution

`(t(0.975, k − 1) + z(0.80)) × SE`, `k` the number of event-seasons, `SE` the standard error of
the season-clustered bootstrap that produces the interval (`experiment.summarise`), exactly as
restated on 2026-09-07 above. `SE ≈ s / √k` where `s` is the between-season standard deviation
of the per-season mean difference, so the multiplier on `s` is:

| event-seasons k | t(0.975, k−1) | multiplier on s | MDE / s |
|---|---|---|---|
| 2 | 12.706 | 13.548 / √2 | **9.58** |
| 3 | 4.303 | 5.144 / √3 | **2.97** |
| 4 | 3.182 | 4.024 / √4 | **2.01** |
| 5 | 2.776 | 3.618 / √5 | **1.62** |
| 6 | 2.571 | 3.412 / √6 | **1.39** |
| 8 | 2.365 | 3.206 / √8 | **1.13** |
| 10 | 2.262 | 3.104 / √10 | **0.98** |

**The number of event-seasons needed** is the smallest `k` at which that MDE is at or below
the **target** — the effect there is to find if this estimator reproduces the source's own
gain, which `qb_adj / 25` is built to do. Target and `s` are read from a **pilot** the pinned
file supplies without a line of this repo's: the source publishes a base and a
quarterback-adjusted probability on every row (`elo_prob1`, `qbelo_prob1`), so their paired
log-loss difference over the same event games, 2022–2025, gives the per-season mean (the
target is its absolute value across seasons) and the between-season `s`. The pilot is a power
input and not a verdict: it is the source's arm on nflfastR outcomes, not this estimator on a
frozen line. Both numbers are computed by the harness and recorded in the run below, with
their `n` and season counts.

## What the archive is expected to hold

The lookahead archive's polls run 2026-08-25 to 2026-09-06 and the 2026 season's first
kickoff is 2026-09-10, so **no in-season starter change can yet have a scored event game**;
the expected run is zero rows, zero event-seasons, NOT-RUNNABLE under precondition 1, with
the event-seasons needed stated from the pilot. If the poller keeps the archive through the
season, 2026 becomes the first event-season in January; the third arrives no earlier than
January 2029, and the number the pilot says is needed may be larger still. That is recorded as
**not-runnable** (ADR-0014's category: an absence of evidence), never as a null.

---

# Pre-registered 2026-09-13: the line-move study's power (#221)

Restating the MDE line #221's disposition of 2026-09-12 wrote with placeholder gap spreads,
before the study runs and with the same event construction as the gate above.

**The estimand** is points of home-spread movement per unit of ex-ante quality gap. For each
event game: the move is the last snapshot before the game day minus the frozen price (the
gate's comparator, the last snapshot before the changed team's previous game day), less the
mean move over every other archived game of the same week between the same two poll days —
the week fixed effect, written as the subtraction it is. The regressor is the **net gap**,
home minus away, where a side's gap is the arriving starter's `qb_value_pre` on the event row
minus the departing starter's on the previous row — both ex ante, both the pinned file's. A
game where both quarterbacks changed is one row with both gaps on it. The benchmark is
**0.132 points per value unit**, 538's 3.3 Elo per unit over 25 Elo per point: a coefficient
that matches it says the betting market does what the source does and there is no edge by
construction.

**The noise floor is #214's** (`hub.fetch.odds --noise-floor`): sd 0.840 points per poll
interval, 0.40 per root-day, over 34 intervals of 12 games. The study's window runs from the
previous game day to the event's game day, about a week, so the floor per window is
`0.40 × √days`; a move inside it is not a move, and the coefficient's standard error is taken
from that floor rather than from a residual the event rows themselves could not resolve.

**The MDE**, before the run: `(t(0.975, n − 1) + z(0.80)) × floor_window / (sd(gap) × √n)`,
`n` the event games (the cluster is the game, as #214's is) — the 2.8 in the disposition's
line, with the t in place of the normal. `sd(gap)` is read off the pinned file's 2022–2025
events by the harness and the line is restated with it below. The disposition's own
arithmetic at gap spreads of 10 / 20 / 30 stands as written there; the pinned file decides
which spread is real.

> **Restated 2026-09-17 (#303): the denominator is `√(n − 1)`, OLS's own.** The formula
> above is kept as written; `study_fit`, `study_mde` and `study_events_needed` all divide by
> `sd(gap) × √(n − 1)` since #303, one formula in three places. Immaterial at n = 53 (about
> 1%; the 0.0063 below becomes 0.0064) and 41% at n = 2, which is the regime the first
> in-season rows arrive in. The *Measured 2026-09-13* table below is a record under the old
> denominator and is re-taken, not edited, when real event rows exist.

**Censoring and timing**, reported with the run: an event whose change predates the first
snapshot cannot be seen and is counted, not dropped silently — every offseason change is of
this kind against an archive that opens 2026-08-25. The change-point is the first poll day
after the frozen one at which the event game's move clears the floor per root-day, reported
in days from the previous game day; the depth-chart date that would place it against the
report date is not cached here and is *not established* until it is.

**Expected on this archive:** the same zero as the gate — no in-season event precedes the
archive's last poll — so the run records the event construction's counts, the pilot gap
spread, the restated MDE, and the censored count, and the coefficient is *not established*.

## Amended 2026-09-17 (#300) — the ADOPT condition

Nothing above is edited; per [method.md](method.md) rule 13 a superseded decision keeps its
original text, and this pre-registration named an estimand and its power but never said what a
reader was to do with the fitted coefficient once one existed — the gap #300 closes, converting
the maintainer's `ADOPTED:` comment of 2026-09-17 into the rule this document was missing.

**ADOPT** — the fitted coefficient's sign matches the benchmark's (positive: the arriving
starter's side gains) and its magnitude sits within reach of **0.132**, the benchmark stated
above, on an interval excluding zero on the benchmark's side. **Season-clustered once two
seasons exist**: with one event-season in the archive the coefficient is scored at the
game-level standard error above (`study_fit`'s floor-based SE, `n` the event games, exactly as
restated below); the day a second event-season is scored, the cluster becomes the season,
matching every other gate in this repo (ADR-0019) rather than staying at the game level by
default. Anything else — a null interval, a wrong sign, or a magnitude that never approaches
0.132 as seasons accumulate — leaves the module exactly where #299 put it: reachable only from
`hub.models.starter_change`, the harness. #291's gate, demoted above, is read beside the
coefficient for the log-loss question and decides neither outcome.

**What this does not do.** It does not add to the interval's construction beyond what *The
MDE* above already fixes (the t-reference, the floor-based standard error, the cluster moving
to the season at two seasons), and it does not touch `study_fit`, `study_mde` or any other
estimator in `hub.models.starter_change` — #221 and #303 own the computation this section
names a rule for.

### Restated 2026-09-17 (#329) — the decidable rule

The paragraph above is kept as written, per [method.md](method.md) rule 13. "Magnitude sits
within reach of 0.132" was never decidable — a fitted coefficient of 0.09 on an interval of
[0.02, 0.16]: within reach or not, and two readers answer differently — and that is method
rule 1's incident, a rule documented but not implemented. #329 replaces it with the rule the
maintainer's `ADOPTED:` comment names, implemented and unit-tested (every branch, including
the losing ones) as `hub.models.starter_change.verdict()`.

**The rule is non-inferiority against a lower bound, DELTA = 0.0075 points per value unit** —
one half-point tick (the betting market's own resolution) on a one-sd starter change, gap sd 66.3
value units over 231 in-season events 2022–2025 (the per-season table above: 80.4 / 62.2 /
69.2 / 50.3 on n=62/61/51/57, summing to 231). DELTA is a constant, pre-registered here, never
re-derived from the events under test — the containment form first drafted for #329 ("interval
contains 0.132") is withdrawn: at one season the interval's half-width is narrow enough that
containing 0.132 would mean a ±3% window around a constant borrowed from another construction,
which a correct, useful coefficient would miss almost always and which rewards a *wider*
interval.

| branch | condition | consequence |
|---|---|---|
| **ADOPT** | the interval's lower bound exceeds DELTA | the route back opens: a ticket restores `quarterback.apply` to the published path with the `-qb` mark, the separate partition and the track-record split kept. |
| **SHOW** | the interval excludes zero positively but its lower bound is at or below DELTA | a real effect too small to price (rule 8's gap below the ceiling). Stays harness-only; the run names what a second event-season's season-clustered MDE would resolve. |
| **REMOVE** | the interval excludes zero on the negative side | a refutation: the route back closes, the module becomes an **Exhibit** under `hub.exhibits` per ADR-0007, the refuting measurement re-runnable, the module gone from `hub.models`. |
| **NOT-RUNNABLE** | the MDE exceeds DELTA | no branch is read; the run names how many event games DELTA needs (`study_events_needed`). |

**Restatement trigger, pre-registered:** the run prints the test events' own gap sd beside
DELTA; per-season values to date are 80.4 / 62.2 / 69.2 / 50.3, and a value outside **50–85**
flags the derivation for restatement — a print (`gap_sd_restatement_flag`), never a branch.
Inside the band nobody decides anything.

**0.132 is reported, never gated on:** printed beside the fitted coefficient as replication
(the interval contains it), below, or above (`benchmark_reading`) — called from `main`, never
from `verdict`, and it appears in none of the four branches above.

**The NOT-RUNNABLE precondition is the same one *The MDE* above states:** an event count whose
MDE exceeds DELTA reads no branch, the same stage-2 shape `experiment.gate` already uses
(precondition 2 under *The bar*, above). Against DELTA, one event-season resolves it at the
pilot's own numbers (MDE ≈ 0.0063, `docs/gate-power.md`'s *Pre-registered 2026-09-13* section);
`study_events_needed` states the count for whatever spread the run at hand has, off the same
`sd_gap` and `floor_window` `study_fit` produces, so the number the sentence names and the
number that triggered NOT-RUNNABLE cannot be two numbers.

---

# Measured 2026-09-13: the quarterback gate is not-runnable, and the archive needs about 29 event-seasons (#291)

Run against the pre-registration above, committed first in `8e4e502`, by
`uv run python -m hub.models.starter_change --events --gate --ceiling` against the pinned
nfeloqb file (`2c95e5fc`, 16,088 rows) and the 2026 lookahead archive (272 games, 8 polls,
2026-08-25 to 2026-09-06). Nothing was fetched.

## The event construction, on the pinned file

Events off the source's starter column, a change between two regular-season games of one
season; a team's first row in the window has nothing to differ from, so 2022's offseason
count is not computable from a window that opens there.

| season | changes | event games | gap sd (value units) | offseason changes, not events |
|---|---|---|---|---|
| 2022 | 62 | 57 | 80.4 (n=62) | — |
| 2023 | 61 | 53 | 62.2 (n=61) | 15 |
| 2024 | 51 | 48 | 69.2 (n=51) | 19 |
| 2025 | 57 | 54 | 50.3 (n=57) | 19 |
| 2026 | **0** | **0** | — | 13 |

About fifty-five a season, which is the count #221's disposition assumed. **2026 holds no
event yet**: the file's 2026 rows are week 1, two of them played, and every one of the 13
changes it shows is a team whose week-1 starter differs from its last 2025 start — priced
all summer, and censored against an archive that opens 2026-08-25 in any case.

## The gate

| | |
|---|---|
| scored event games in the archive | **0** |
| event-seasons | **0** (pre-registered minimum 3) |
| verdict | **NOT-RUNNABLE** — precondition 1 |
| the mark | stands on every `-qb` row (#270); this is not a null |

The archive's last poll (2026-09-06) precedes the season's first kickoff (2026-09-10), so no
frozen price yet predates an in-season change. The harness is built and held on synthetic
fixtures (`tests/unit/test_starter_change.py`): the frozen price is the last snapshot before
the changed team's previous game day, the arm is `quarterback.apply` on a row labelled
`stale`, the pair is log-loss on the home result, the ceiling arm is the last snapshot
before the game day, and the precondition fires ahead of the house rule.

**Re-run 2026-09-13 after the review amendment above, and nothing numerical moves.** The arm
is now the shipped seam (`nfeloqb.state` as of the week's first game day) and the oracle is
reported beside it as a diagnostic; the gate still has zero rows, so neither arm has a
figure. The event counts do not depend on the arm and are unchanged. The pilot is scored on
the source's own two probability columns and never touched either arm; it is unchanged to
the fourth decimal (target 0.0180, s 0.0332, 29 event-seasons). Held by two mutants: the
arm swapped back to the same-week row, and the as-of moved onto the game day, each fails
`test_the_arm_is_the_shipped_seam_and_prices_the_departing_starter`.

**What the seam implies for the gate's answer, said plainly.** On the shipped replay, an
event game's adjustment is the departing starter's — near zero for an established starter
— so the shipped arm on an event game is close to the frozen line, and the gate is asking
whether a *stale* adjustment beats no adjustment. The oracle is the arm the module's
docstring describes. The gap between the two, once rows exist, is the cost of the source's
timing, and [qb-adjustment.md](qb-adjustment.md) records what that timing is.

## The pilot, and the number of event-seasons needed

The source's own two columns (`elo_prob1`, `qbelo_prob1`) on the same event games, log-loss
base minus quarterback-adjusted, positive when the adjustment helped:

| season | event games | mean | sd |
|---|---|---|---|
| 2022 | 57 | **+0.0248** | 0.224 |
| 2023 | 53 | **−0.0466** | 0.207 |
| 2024 | 48 | **−0.0417** | 0.216 |
| 2025 | 54 | **−0.0085** | 0.216 |

Over 212 event games and 4 seasons: mean of the season means **−0.0180**, between-season sd
**0.0332**. The pre-registered target is the absolute value, 0.0180, and the smallest `k`
whose MDE `(t(0.975, k−1) + 0.8416) × 0.0332 / √k` is at or below it is **k = 29
event-seasons**. At one season a year that is the 2050s; at the three-season floor the MDE
is 0.099, five times the target.

**Two things the pilot says beside the power figure, neither a verdict on this gate.** The
sign is negative in three seasons of four: on event games, on nflfastR outcomes, the
source's quarterback-adjusted probability scored *worse* than its base in 2023, 2024 and
2025, and the pooled figure is the wrong way for the arm. That is the source's arm and the
source's base, not this estimator on a frozen line — the gate's question — and the per-game
sd of 0.21 against per-season means of a few hundredths says a single season's sign is
mostly noise. It is recorded because the licence in [qb-adjustment.md](qb-adjustment.md)
rests on the *same* two columns of a different file (538's, 2013–2022, all games, Brier),
and on the event games this repo's adjustment exists for, the nfeloqb file does not
reproduce that gain. The second: the target is small against the noise because the
adjustment's whole effect is a few hundredths of log-loss per event game, so a gate on it
was always going to be a decades-scale question. That is the not-runnable finding, and it
is what #270's mark is for.

**Not established.** The gate's own effect, interval and ceiling — no row exists to compute
them from. The run stamps `commit` as dirty because the harness was run before its own
commit; the numbers above depend on the pinned file and the archive alone, both unchanged.

---

# Measured 2026-09-13: the line-move study's MDE is restated on the real gap spread, and its coefficient is not established (#221)

Run against the pre-registration above by `uv run python -m hub.models.starter_change
--events --study`, on the same pinned file and archive as the gate. The event construction
is the gate's (its table above); what follows is the study's own.

## The gap spread, and the MDE restated

The disposition of 2026-09-12 stated the MDE at gap spreads of 10 / 20 / 30 value units and
said the pinned file would decide. It has: the ex-ante gap between the departing and the
arriving starter has a standard deviation of **66.3 value units over 231 in-season events,
2022–2025** (per season 80.4 / 62.2 / 69.2 / 50.3, n = 62 / 61 / 51 / 57). A backup is not
a slightly worse starter; on the source's scale he is a hundred units worse, which is why
the source's adjustments run to −125 Elo.

The noise floor, recomputed off this archive with frozen lookaheads excluded, is **0.404
points per root-day over 12 live games** — #214's 0.40 on the same twelve, reproduced. Over a
seven-day window that is 1.06 points per event.

| | MDE, points per value unit |
|---|---|
| one season, n = 53 event games, 7-day window, t(0.975, 52) | **0.0063** |
| two seasons, n = 106 | 0.0044 |

Against the 0.132 benchmark, **one season resolves the benchmark twenty times over, and
resolves 0.132 from 0.10 at five MDEs.** The disposition's "two-season question" was a
question about the gap spread, and the gap spread answers it: what a second season buys is
not power but a second cluster for the sign. The 2.8 in the disposition's line is 2.85 on the
t at fifty-two clusters; the difference is in the third decimal.

## The run

| | |
|---|---|
| in-season events in 2026 | **0** |
| offseason changes in 2026 | 13, every one before the first snapshot (2026-08-25): **censored by construction** |
| event games with a frozen and a pre-game price | **0** |
| coefficient | **not established** |
| change-point against the report date | **not established** — depth charts are not cached, and the play-by-play cache carries no passer column, so the first-pass-attempt reader (`starters_from_pbp`) is held on a fixture and has not met live data |

The harness runs end to end on a synthetic archive (`tests/unit/test_starter_change.py`):
the week's mean move between the same two poll days is subtracted, the slope on the net gap
recovers a manufactured 0.132, the standard error is the floor's, and the change-point is the
first poll clearing the floor per root-day. On the real archive there is nothing for it to
read until the poller has carried the season past its first in-season change, which the
first table above says will be about the fourth week.

**What the study cannot do on this archive, and says so.** An event whose change was known
before the frozen snapshot — a benching announced midweek before the previous game — is
priced in the frozen quote already, and the study would read a move of zero against a gap of
a hundred: a censoring in the other direction that only the depth-chart timestamp can
separate. It is the same limit as the timing criterion, and it closes when the depth charts
are cached.


---

# Pre-registered 2026-09-13: the weekly interval's shape law, decided by CRPS (#292)

**Written and committed before the comparison was run.** As with the rule at the top of this
document, the commit order is the point: this section lands in one commit, the harness and
its result in the next, and a reader can check that the bar was set before the number was
known. Parent decisions: [#289](weekly-coverage.md), which restated the interval's claim and
named this as the refit it declined to make by hand, and #274, which promoted CRPS to a
deciding metric for exactly this decision and nothing else.

## The question

The weekly interval `hub.models.predict.moments` serves is pushed through a fitted
Cornish-Fisher skew (`WEEKLY_SKEW`, per position) before it reaches `draft/optimize.py`,
`season/roster.py` and the lineup gate. `hub.models.coverage` measures both interval forms on
the same weeks -- 77.4% with the skew and 79.5% without, both under an 80% label -- and
nothing decides between them. Coverage cannot: it reads two quantiles and is blind to the
shape between them, and "closer to 80%" is a property of a label the interval has already
been restated away from. A proper scoring rule of the whole distribution is what a shape
question needs, and CRPS is the one this repo already computes beside MAE.

> **Does removing the skew law improve the served weekly distribution, scored by CRPS on the
> same player-weeks, under the house rule with the season as the unit of replication?**

## What is measured

For the paired frame, with nothing else changed from the coverage harness:

1. **The rows.** `hub.models.coverage`'s prior-centre player-weeks -- 2021-2025, the four
   drafted positions, `MIN_WEEKS = 8`, `MIN_MU = 2.0`, `MIN_PRIOR = 4`, centre from strictly
   earlier weeks -- restricted to the **unclipped** subset the coverage gate reads (10,536 at
   the last run). These are the weeks the two coverage figures above were quoted on, and the
   subset is fixed here so the rows cannot be chosen after the sign is seen. The pooled 16,061
   is printed beside it as a **diagnostic** and decides nothing.
2. **Arm B, the incumbent: the deployed distribution.** `predict.skewed(mu, sd, skew, z)` on
   `predict.moments`' own `mu`, `sd` and `skew`, evaluated at `scoring_rules.quantile_levels()`
   (400 levels) and scored by `scoring_rules.crps_from_quantiles` -- the same path
   `hub.models.weekly.shipped_quantiles` takes, zero clip included because the clip is served.
3. **Arm A, under test: the same function with the skew law removed.** `predict.skewed(mu,
   sd, 0.0, z)`, which the function floors at `MIN_SKEW = 0.05` -- so the arm is exactly what
   would be served if `WEEKLY_SKEW` were zeroed, floor and all, and not a plain normal written
   out beside the deployed one. Same `mu`, same `sd`, same clip; only the skew term moves.
4. **The paired difference** per player-week is `diff = CRPS_B - CRPS_A`, positive when the
   skew-free arm scores better, in points of CRPS per player-week.
5. `experiment.run_gate(cluster=SEASON_CLUSTER)`: five clusters, the percentile bootstrap at
   `BOOTSTRAP = 4000` draws, `seed = 0`, the interval and the standard error from the same
   draws, and **MDE `(t(0.975, 4) + z(0.80)) * SE`** from that standard error -- the restated
   form above, not the superseded normal quantile.
6. **The ceiling arm, declared: *the best per-position skew, chosen on these rows*.**
   `predict.skewed` with the skew for each position chosen on a grid from 0 to 2 by 0.05 to
   minimise mean CRPS over these same rows. In-sample by construction, so it is a bound: the
   largest gain any per-position skew law -- the deployed one, none, or any other -- could show
   over the deployed one on this frame. Declared as `hub.models.coverage.CEILING_ARM` and
   printed on the ceiling line, distinct by name from the three gates' arms.

## The bar, set now

In `experiment.gate`'s order, and nothing is added to it:

- **NOT-RUNNABLE** if the season-clustered MDE exceeds the ceiling (stage 2 above, ADR-0019
  as amended). Recorded as *not runnable* under
  [ADR-0014](adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md): no verdict, the
  skew stays, and this document says the design could not answer the question on five
  seasons. Stage 1 -- MDE against the reported effect -- is printed beside it for the record
  and licenses nothing.
- **ADOPT the skew-free interval** only if the interval excludes zero on the positive side
  *and* the skew-free arm has the lower mean CRPS in **every one of the five** held-out
  seasons. What follows is the maintainer's switch and not this lane's: the deployed function
  changes, and #289's restated claim and band are re-registered against the skew-free
  function before its gate is read again.
- **REMOVE** -- the deployed skew wins in every season and the interval is entirely negative:
  the skew stays, with evidence that it earns its place.
- **SHOW** otherwise: the skew stays, and this document says the CRPS comparison could not
  remove it. Absence of evidence, not equivalence.

**Expectation, written before the number.** Two shapes on identical `mu` and `sd` differ in
CRPS by a small fraction of the score itself; the effect is expected to be of the order of a
hundredth of a point per player-week either way, against a mean CRPS of several points. That
is a guess and not a prediction; it is written so that a much larger movement is recognised
as needing its own explanation rather than absorbed.

## What this measurement cannot do

- It cannot pick a different clustering, a different seed, or a different subset once the
  sign is seen. All three are fixed above.
- It cannot license any change to `sd`. `WEEKLY_K` is identical in both arms; the only term
  on trial is the skew, and a CRPS gain here is the skew term's alone.
- It cannot say anything about the weeks a player did not play, which are outside the coverage
  harness's sample by construction, or about the clipped weeks the gate excludes.
- It cannot promote CRPS anywhere else. #274's promotion is for this decision; every other
  CRPS line in the repo stays a diagnostic.

## What happens either way

If the gate cannot run, the interval keeps its skew and its restated 77% claim, and the
programme's honest output is this section plus the numbers. If it runs and the skew-free arm
does not win every season with an interval clear of zero, the skew stays and the claim stays.
If it runs and the skew-free arm wins, the deployed function is the maintainer's to change,
and the claim is re-measured on the function that replaces it before anything is published
against it.

## Measured 2026-09-13: not runnable on five seasons

Run against the rule above, which was committed first in `12405d0`, by
`uv run python -m hub.models.coverage --shape` on the harness the next commit carries.
Data digest `b1a14540` over one pinned source (`player_stats`, 2021–2025); seed 0; 4,000
bootstrap draws; 10,536 unclipped player-weeks, exactly the rows the coverage gate read on
the same day.

| | value |
|---|---|
| skew-free − deployed skew, mean paired CRPS | **−0.0364** points per player-week |
| 95% percentile CI, season-clustered (5 clusters) | [−0.0423, −0.0296] |
| 95% t CI, 4 df | [−0.0453, −0.0274] |
| P(skew-free better) | 0.0% |
| per season | 2021 −0.0436 · 2022 −0.0428 · 2023 −0.0324 · 2024 −0.0243 · 2025 −0.0389 |
| **MDE at 80% power** | **+0.0117** |
| **ceiling** — *the best per-position skew, chosen on these rows* | **+0.0081** |
| pooled over every row, clipped weeks included (diagnostic) | −0.0313 over 16,061 |

**Verdict: NOT-RUNNABLE.** The season-clustered MDE (+0.0117) exceeds the ceiling (+0.0081):
the largest CRPS gain any per-position skew law could show over the deployed one on these
rows is smaller than the smallest effect five seasons can resolve at 80% power. Under the
rule as pre-registered — and under
[ADR-0019](adr/0019-a-gate-requires-every-season.md) as amended — no verdict below that line
is reported. Recorded as *not runnable* under
[ADR-0014](adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md): the arm did not
lose, the question cannot be answered with the data that exists. **The skew stays**, and
[weekly-coverage.md](weekly-coverage.md) says the CRPS comparison could not remove it.

**What the numbers say, without a verdict attached.** They are printed by the run and
recorded here because the run printed them, not because the rule reads them. The skew-free
arm scores *worse* by 0.036 points of CRPS per player-week, in all five seasons, with both
intervals clear of zero; had the gate been runnable it would have reached REMOVE — the skew
earns its place. The reason it is not runnable is worth stating plainly: the ceiling is the
gain available from re-choosing the skew, and on these rows that is 0.008 — the deployed
per-position skews (0.15 / 0.67 / 0.66 / 0.72) sit below the in-sample optima (0.35 / 1.05
/ 1.15 / 0.95), but the CRPS surface is flat enough near them that the whole of that
headroom is under the MDE. A design that cannot distinguish the deployed law from the best
one also cannot be trusted to have distinguished it from none, and the order of the branches
is what refuses to let the second claim through on the strength of the first.

**Against the expectation written above.** The effect was guessed at "of the order of a
hundredth of a point"; it is three and a half hundredths, in the direction of the skew. That
is inside the range the guess allowed and needs no separate explanation. The per-position
optima being *above* the deployed values is a finding the pre-registration did not
anticipate and does not act on: it is in-sample, it is the ceiling arm and not a fit, and a
refit of `WEEKLY_SKEW` would be its own pre-registered ticket under ADR-0024 — with the same
five clusters and, on this evidence, the same MDE problem.

**The deployed function does not change.** #289's claim (77 ± 2) stands against the function
it was measured on. The `interval_shape` entry in `state/gate-width.json` is this run's
season-clustered width, 0.0127, for the next run to compare against.

## Hold-out constants: the draft gate can replay each season on constants fitted without it (#294)

**Built 2026-09-13; not run.** The draft backtest's last `LIMITATIONS` entry (#279) names the
defect: the simulator's constants — `TALENT_CV`, `TALENT_CV_BY_POS`, `IMPUTE_CV`, `WEEKLY_K`,
`WEEKLY_SKEW`, `TEAMMATE_RHO`, `PICK_NOISE_INTERCEPT`, `PICK_NOISE_SLOPE` — were fitted on the
seasons the gate replays, so arm B carries in-sample constants on every held-out season and the
headline is a lower bound on how badly it loses out of sample. The hold-out *run* is the
maintainer's (#290); this section is the machinery and what it could and could not fit.

**The mechanism.** Every fitting script under `scripts/` takes `--exclude-season N` and, with
`--record`, writes what it fitted into `conf/holdout/N.json` under the constant's digest key
with the command that produced it (`hub.holdout`). `hub.draft.backtest --holdout` and
`hub.season.weekly_gate --holdout` load every replayed season's set *before* a draft is played
— a season with no set refuses the run rather than playing shipped under the hold-out's name —
print one run line per season saying which constants were refitted and which stayed shipped
and why, then rebind the module attributes to the set for that season's play and restore the
shipped values after. Production reads nothing from `conf/holdout/`; the shipped constants and
every digest are unchanged, and the stamp on a hold-out run carries the shipped
`fitted_digest` with the run line beside it saying what each season read.

**What the four committed sets refit, and what they could not.** Eleven keys per set — the
eight above and the three companions read with them (`IMPUTE_CV_BY_POS`, `WEEKLY_K_POOLED`,
`WEEKLY_SKEW_POOLED`). Each of 2022–2025:

| constant | refitted without the season? | why not |
|---|---|---|
| `WEEKLY_K`, `WEEKLY_K_POOLED` | **yes** — `scripts/fit_weekly_spread.py` reproduces the shipped values exactly with nothing excluded (1,174 player-seasons; 1.880 / 2.068 / 2.125 / 1.994, pooled 2.042; exponents 0.161 / 0.469 / 0.518 / 0.622, pooled 0.498) | |
| `TEAMMATE_RHO` | **yes** — `scripts/fit_teammate_rho.py`, the `correlate` method on the Panel's cached slice | reproduces the shipped edges to within 0.02 (+0.222 / +0.205 / +0.052 against +0.232 / +0.225 / +0.054), inside two of its own standard errors; the shipped run's exact sample is not in the tree |
| `WEEKLY_SKEW`, `WEEKLY_SKEW_POOLED` | no | the shipped estimator (a 760-player-season validation, [component-projection.md](component-projection.md)) is not in the tree; the script's own reads 0.68 pooled against 0.60 and a set carrying it would confound the estimator with the season |
| `IMPUTE_CV`, `IMPUTE_CV_BY_POS` | no | since #298 (2026-09-16) the shipped value is the rookie measurement on all five seasons ([impute-cv.md](impute-cv.md)); a four-season refit has one cluster fewer than the decision was taken on, and the reason carries the hold-out's own number |
| `TALENT_CV`, `TALENT_CV_BY_POS` | no | `scripts/fit_talent_cv.py` needs an ESPN session (`calibrate.draft_outcomes`); none on 2026-09-13 |
| `PICK_NOISE_INTERCEPT`, `PICK_NOISE_SLOPE` | no | `scripts/fit_pick_noise.py` needs an ESPN session (`availability.historical_picks`); none on 2026-09-13 |

So a hold-out replay today moves the weekly law and the teammate correlation and nothing else,
and says so on every season's line. The held-out `WEEKLY_K` values sit within 0.05 of the
shipped at every position and season (QB 1.83–1.92, RB 2.04–2.10, WR 2.12–2.13, TE 1.98–2.01),
and `TEAMMATE_RHO`'s QB–WR edge within 0.02 (0.212–0.232) — the two constants the archive can
refit are the two that barely move when a season leaves, which is itself the first hold-out
finding. Whether the four that could not be refitted move more is exactly what the sets do not
yet say; the two ESPN-bound scripts record their reason into the set and re-run with a session
(`--exclude-season N --record` overwrites the key), and the two estimator-bound ones wait on
the decisions named.

**What a run should print.** Per season, `season 2024: constants from conf/holdout/2024.json
(fitted without 2024); refitted: predict.WEEKLY_K, predict.WEEKLY_K_POOLED,
predict.TEAMMATE_RHO; shipped (not refitted): …` followed by the reasons; then the gate's own
lines. Two runs are comparable only at an identical `board_digest`, as before, and a hold-out
run is not the gate's width record — it is a sensitivity of the gate to its own constants,
read the way the noise-scale sweep (#49) is.

# Pre-registered 2026-09-19 — PROPOSED: the implied-total gate (#305)

**Status: PROPOSED.** Drafted during the operate-mode freeze (#326), which blocks a modelling
change *landing* and does not block writing the rule before the number — #329 is the
precedent, pre-registered on 2026-09-13 before the module that would run it existed. This
becomes the rule on the maintainer's `ADOPTED:` comment on #305; until then it decides
nothing and #305's build does not start against it. Parent decisions: ADR-0016 (the weekly
projection is shown and never ranked on — a gate here changes what is *published*, not what
picks), ADR-0017 (a market–usage blend is a model, not a shrinkage), and #212 (the guard,
below), whose disposition is the one genuine choice in this document.

## The question

`hub.models.weekly.project` is season-to-date Usage × own efficiency × a positional
constant: no opponent, no total, no spread, no venue, no rest. The implied team total is on
the panel already (`implied_total = total_line/2 + own_spread/2`, on every one of the
24,677 player-weeks), and it is the strongest signal the repo's own screen has found —
+0.055 partial correlation across all five seasons. #305 lets it into the projection as a
multiplicative scaler on volume and on the touchdown rate.

> **Does a lineup set off the projection that scales volume and touchdown rate by the implied
> team total beat one set off weekly consensus rank, in realised points per roster-week, under
> `hub.season.weekly_gate`'s pre-registered rules — and, as the diagnostic beneath it, does that
> projection beat the incumbent projection on absolute error per player-week?**

**Restated 2026-09-20 — the deciding statistic is the lineup, not the MAE.** The first draft
gated on MAE per player-week. MAE is not the objective of any product this repo ships: the
lineup is a max over starters, the draft a season simulation, survivor a game outcome. A
gain concentrated on high-μ starters is worth real lineup wins; the same gain spread over the
bench is worth nothing, because a max never reads it — and the pilots below show both #305's
and #308's gains sit on the starters. And the weekly projection is **shown and never ranked
on** (ADR-0016): its own lineup gate read −0.304 points per roster-week, 2 of 3 seasons lost.
So a change to it changes no lineup until *that* gate flips, and that gate is where the
decision lives. The MAE contrast stays, as the diagnostic that says the mean moved and where.

## What is measured

1. **The rows.** The screen panel `hub.models.panel.build_panel(SEASONS, SCREEN_SPEC)` —
   2021–2025, the four drafted positions, `MIN_GAMES_BEFORE = 3` — walked forward by
   `experiment.expanding_seasons` exactly as `weekly.walk_forward` does today, so the held-out
   seasons are **2022, 2023, 2024, 2025** with 4,928 / 5,034 / 4,343 / 5,147 player-weeks.
   These are the rows the published rebuild figure was taken on; fixed here so they cannot
   be chosen after the sign is seen.
2. **Arm B, the incumbent: `project` as shipped.** `f = 1`, the `weekly` arm of
   `walk_forward` — the identity multiplier on season-to-date Usage, `components.td_rate` on
   the projected yards, byte for byte what the site publishes.
3. **Arms A₁ and A₂, under test — one axis, both reported.** The same `project` with a
   multiplicative scaler `s = exp(β · log(implied_total / league mean implied total))` on
   the volume counts (`targets`, `carries`, `attempts`) and on the touchdown rate, with β
   fitted **walk-forward on strictly earlier seasons** as (A₁) a Poisson GLM with the log
   ratio as offset, (A₂) a Gamma GLM on the same regressor. Nothing else moves: efficiency
   stays the player's own, `sd = k√μ` stays, the zero clip stays. The two arms are ADR-0024's
   table, not two decisions; the ticket's own reopen clause (sign disagreement between them)
   stands.
4. **The deciding gate: `hub.season.weekly_gate`, its rules unchanged.** Arm A's projection
   sets a lineup by *start your highest*; the incumbent sets one by weekly consensus rank
   (`weekly-op` ECR), which is what ranks today. Realised points per roster-week, weeks 1–14,
   paired by roster-week, the cluster bootstrap by roster, both arms scoring only the
   roster-weeks both can price, inactive weeks scoring zero — every rule as
   `docs/weekly-projection-plan.md` pre-registered them and #206/#207 amended them, with its
   own foresight ceiling (#138: the declared ceiling arm) and stage 2 ahead of every branch.
   A₁ and A₂ each run it; the incumbent projection (`f = 1`) is run beside them so the
   table shows whether the scaler moved the lineup, not only whether it beat consensus.
   **The diagnostic beneath it** is the paired MAE difference `diff = |err_B| − |err_A|` per
   player-week through `_contrast`, positive when the scaled arm is closer, **reported by μ
   tier** (< 6, 6–10, 10–15, 15+) so a reader sees where the mean moved. MAE, not CRPS:
   #274's promotion of CRPS was for #292's shape decision and nothing else.
5. **The statistic — and the dependency it surfaces.** `experiment.paired_gain` through
   `weekly._contrast`, with **`cluster = SEASON_CLUSTER`**. Today `_contrast` calls
   `paired_gain` unclustered over ~20,000 player-weeks, which is what #311 fixes and why
   **#311 is first out of the freeze and #305 is `blocked_by` it**: this gate is not runnable
   as pre-registered until the cluster argument exists. The percentile bootstrap at
   `BOOTSTRAP = 4000`, `seed = 0`; interval and standard error from the same draws;
   **MDE `(t(0.975, 3) + z(0.80)) × SE`** on the t reference, four clusters.
6. **The ceiling arm, declared: *the best one-parameter scaler on the implied total, fitted
   in-sample on these rows.*** `μ · exp(b · log ratio)` with `b` chosen per season to minimise
   MAE on that season — in-sample by construction, so it bounds what any walk-forward fit of
   the same form can earn on this input. Declared by name, distinct from the arms. **Why not
   the realised team total** (the first draft's ceiling, withdrawn 2026-09-19 after measuring
   it): a scaler reading the team's *actual* points recovers **+0.228** MAE per player-week
   (per season +0.232 / +0.252 / +0.207 / +0.221, SE 0.0095) against the implied total's
   +0.012 — but the realised total is the outcome of the same game the player scored in, not
   a better estimate of it, so that 0.228 is overwhelmingly in-game variance no pre-kickoff
   number can reach. As a ceiling it passes stage 2 by construction (0.23 against any MDE) and
   answers nothing rule 8 asks. It is kept here as the measurement of how large the game
   environment is as a lever — and that it is not a *forecasting* lever: the market's total is
   the best public forecast of it, and the forecastable share is about the 0.012. Under the
   corrected ceiling, stage 2 asks the right question: is the run's MDE below what this input
   can give at all.

**The MDE, before the run — measured, not guessed (pilot of 2026-09-19, no model changed).**
Two noise scales exist on these rows and they differ by an order of magnitude. The
incumbent's own contrast (rebuild vs flat) has per-season gains **+0.0944 / +0.0528 /
+0.0000 / +0.1020**, clustered SE 0.0234, proxy MDE ≈ 0.094 — but two arms that differ only
by a scaler are far less noisy across seasons than two arms that differ in structure, so that
proxy is an over-estimate and it is recorded only because it was the first number written.
The pilot that bounds the effect itself: regress the incumbent's residual `y − μ` on
`log(implied_total / mean)` per held-out season, and fit a single multiplicative scaler
`μ · exp(b · log ratio)` in-sample (an upper bound for any one-parameter scaler) and
cross-fit on strictly earlier seasons (what a walk-forward would actually earn):

| season | corr(residual, log ratio) | b in-sample | MAE gain, in-sample bound | b cross-fit | MAE gain, cross-fit |
|---|---|---|---|---|---|
| 2022 | +0.025 | +0.19 | +0.0064 | — | — |
| 2023 | +0.028 | +0.22 | +0.0065 | +0.19 | +0.0065 |
| 2024 | +0.031 | +0.35 | +0.0132 | +0.20 | +0.0108 |
| 2025 | +0.051 | +0.37 | +0.0208 | +0.25 | +0.0187 |

In-sample bound: mean **+0.0117**, clustered SE **0.0034**, k = 4, **MDE ≈ 0.014**, t ≈ 3.4.
Cross-fit: mean **+0.0120**, SE 0.0036, k = 3, MDE ≈ 0.018. So, written before the run: the
effect is **real in sign** (positive residual correlation in all four seasons, rising), of
size **about +0.01 MAE per player-week** — a fifth of the rebuild's effect — and it sits
**at the four-season MDE**, not clearly above or below it. That is neither "likely
NOT-RUNNABLE" nor "should pass": on four clusters stage 2 is roughly a coin flip and a fifth
season resolves it, and the ceiling arm (the realised total) is what says how much of the
recoverable error a *knowable* total can reach. The rising 2024–2025 figures are noted and
not read as a trend: two points are not a trend (rule 4's shape, in reverse). **Where the gain lands (2026-09-20), which is what makes the lineup the right gate.** The
same pilot, in-sample scaler applied, broken by the incumbent's μ:

| μ tier | share of rows | #305 gain | #308 gain (RB/WR/TE) |
|---|---|---|---|
| < 6 (bench) | 48% | **−0.0015** | +0.0007 |
| 6–10 (flex) | 19% | +0.0130 | +0.0213 |
| 10–15 (starter) | 19% | +0.0255 | +0.0133 |
| 15+ (star) | 14% | +0.0286 | **+0.1109** |

Neither gain is on the bench; #305's is flat-to-negative there and rises with μ, and
three-quarters of #308's whole effect sits on the 14% of rows at μ ≥ 15 — exactly the rows
a lineup max reads. A per-player-week MAE hides this, and a gate on it could pass on a
statistic no product optimises. Cross-fit and ceiling are consistent on the same seasons
(2023–2025: in-sample +0.0135, cross-fit +0.0120); the earlier "+0.012 against +0.0117"
compared a three-season cross-fit to a four-season ceiling.

**Calibration, written down because it is uncomfortable.** #305's effect (+0.012) is its own
ceiling (+0.0117): whatever is built on the implied total earns about a hundredth of a point
per player-week, with no better specification behind it. #308's +0.020 is a *floor from a
crude instrument* — a residual correlation — and a properly specified count model on
touchdowns per scoring opportunity should do at least as well. So the comparison is a floor
that may rise against a ceiling that will not. Together the best-known and best-measured
features are worth a few tenths of a point per lineup per week; over a season that is a
couple of points of cumulative MAE against head-to-head margins in the tens. That is not an
argument against Phase 2; it is the argument for #306 moving up — the share layer is
structural, not a feature, and a structural change is not bounded by a residual correlation.

**What the −0.304 says about Phase 2 as a whole (2026-09-20; arithmetic restated the same
day).** The load-bearing claim needs no arithmetic: **consensus rank is a crowd of analysts
who already see the implied total, the depth chart, the beat reports and who is getting
goal-line work. `implied_total` and red-zone share are information the benchmark already
prices. Adding them closes distance; it cannot create edge, because it is catching up to
what the projection is scored against.** An edge has to come from what a rank does not
produce at all. A rank is a point estimate — no distribution, no correlation structure, no
coherence constraint — which is exactly #306 (additivity, injury substitution, the negative
teammate correlation), #314 (opponent and season-level correlation, which an optimiser reads
and a rank cannot supply) and #309 (a calibrated interval, so the optimiser has a
distribution rather than a mean). Those change what the projection can *say*, not how much
it knows; they are the three nobody has bounded; and #306's bound cannot come from a
residual correlation, which is why it is a grilling and not a pilot.

*The magnitude check, with its caveat, beneath that.* The lineup gate's standing figure is
−0.304 points per roster-week, 2 of 3 seasons lost. The two features are the same order of
magnitude as that gap under any plausible translation from MAE per player-week to roster
points — nine starters at a one-for-one rate gives ≈ +0.29; weighting by where the μ-tier
table puts the gain gives several times that — **and the translation itself is unmeasured.**
Lineup value does not live at μ ≥ 15: the gate scores the realised points of the *chosen*
lineup, so accuracy on a player whose start/sit status does not change contributes exactly
zero, and the gain comes only from changed selections — the flex spot, the WR3-vs-WR4 call —
which are mid-μ by construction because the marginal starter is the replacement-level one.
The top and the bottom of the μ table are both worth nothing to the gate; the value is in
the band that decides, and the μ-tier table does not cut there. So the first draft's
"parity, not past it" was a number doing work its derivation did not license — the same
shape as the containment rule and Δ = 0.05 earlier in this document. What the evidence
supports is: same order of magnitude, direction unknown until the selection-flip rate is
measured, and **the only thing that settles sufficiency is `weekly_gate` on the built
arms.** The structural argument above does not depend on it and stands.

**The decision-boundary cut (2026-09-20), which is the first number here that bounds lineup
value rather than proxies it.** The gate's own universe — `weekly_gate_data.assemble_universe`,
the drafted cohorts, weeks 1–14, both arms restricted to `priced_by_both` — built twice with
the same seed: once as shipped and once with `project`'s μ scaled by
`exp(0.25 · log(implied_total / mean))` (0.25 a middle value of the pilot's fitted b; a
diagnostic, not the arm). Lineups under the gate's own `starting_lineup` in both. The
**margin band** is, per roster-week and position group, the lowest-μ starter and the
highest-μ bench player under the baseline — about six of eleven fieldable rows.

| season | roster-weeks | selection flips | gain on the band | elsewhere on the roster |
|---|---|---|---|---|
| 2022 | 229 | 14.0% | +0.0253 | −0.0395 |
| 2023 | 229 | 17.5% | +0.0319 | −0.0554 |
| 2024 | 192 | 9.4% | +0.0190 | +0.0336 |
| 2025 | 239 | 11.7% | +0.0184 | −0.0364 |
| all | | **13.1%** | **+0.0236** (SE 0.0032, 4 of 4) | −0.0244 |

Three things. The scaler changes a lineup in about **one roster-week in eight**, so the
selections do move and the translation is no longer unmeasured in kind. On the rows that
decide, the mean improves in every season. And off the band, on drafted rosters, the scaler
is *negative* in three seasons of four: the panel-wide +0.012 was a gain where decisions are
made averaged against a loss where they are not, which is one more reason the per-player-week
MAE is the wrong deciding statistic. What this does **not** measure is the value of a flip —
the realised points of the changed lineup against the unchanged one — because that is
`weekly_gate`'s own statistic on an arm that does not yet exist, and putting that number on
the table before the arm is built is the thing this document exists to prevent. #308's
band cut is the same computation with its modifier injected and is owed when its
pre-registration is written.

**Which population, and what the net is (2026-09-20, same build).** The +0.012 is on the
screen panel — every consensus-ranked player — and does not decompose into the two figures
above, which are on drafted rosters; "the average was masking a split" does not follow across
populations. On the drafted rosters themselves, with the counts:

| b | band n / off-band n (per season) | band gain | off-band | **net on drafted rosters** | positive seasons |
|---|---|---|---|---|---|
| 0.25 (the pilot's panel fit) | ≈ 1,400 / 1,170 | +0.0236 | −0.0244 | **+0.0017** (SE 0.0080) | 1 of 4 |
| 0.12 (half) | same rows | +0.0150 | −0.0065 | **+0.0052** (SE 0.0037) | 4 of 4 |

So on the population that actually forms lineups, the scaler as fitted is **MAE-neutral and
negative in three seasons of four; only its ranking improves.** That reads very differently
from "a split under an average", and it is what the pre-registration below has to be written
against. The mechanism is sizing: the band gain and the off-band loss scale together with b,
because the b that reorders the margin most also moves the locked starters most — the
do-no-harm case in its cleanest form.

**The 0.12 row is a sensitivity of the mechanism and not a candidate, and b is not free.**
"Halve b and the net turns positive in every season" is a threshold read off four seasons of
outcomes, and if the do-no-harm clause could be satisfied by moving b until it is, the clause
would not be a bar — the adoption rule and the parameter it is evaluated at would be chosen
together, which is the hazard this document has already caught three times. So, fixed here:
**b is fitted by the stated procedure — the GLM of arm A₁/A₂, walk-forward on strictly earlier
seasons, on the screen panel — and the gate, the band diagnostic and the do-no-harm clause are
all evaluated at that b and at no other.** b does not move after any of the three is seen. A
run that would only pass at a smaller b reports that as SHOW with the sensitivity printed
beside it, and a differently sized scaler is a new pre-registration, not a re-run.
The same
incumbent figures put the published rebuild at **t = 2.66 on 3 df, p = 0.076** once
clustered — it passes `|t| ≥ 2` and fails `p < 0.05`, which is #312's corrected verdict —
and that is #311's and #312's restatement to make, not this document's.

**The incumbent's own standing, named so the two figures are not stacked.** This gate is
paired against the projection that *ships*, which is correct whatever that projection's
provenance. But the two numbers will sit near each other in the record — the implied total
at about +0.012 over the incumbent, the incumbent at +0.0623 over flat at p = 0.076 once
clustered — and a reader can add them into a significance neither leg has. The comparison
here is against what is published, not against a validated baseline; #311/#312 restate the
incumbent's own verdict separately, and nothing in this document depends on it.

## The bar, set now

In `experiment.gate`'s order, and nothing is added to it:

- **NOT-RUNNABLE** if the season-clustered MDE exceeds the ceiling arm's gain (stage 2,
  ADR-0019 as amended). No verdict; the incumbent stays; this document says four seasons
  could not resolve it, and prints beside that how many seasons would. Stage 1 is printed
  and licenses nothing.
- **ADOPT the scaled projection** only if `weekly_gate`'s interval on points per roster-week
  excludes zero on the positive side against consensus **and** the scaled arm wins **every**
  held-out season of that gate — which reopens ADR-0016 by its own gate, and is the only
  route by which the weekly projection comes to rank — **and the do-no-harm condition
  holds.** The gate reads a ranking; every other consumer reads the *number*: the draft
  board's VOR and season simulation, the published `p10/p90` through `shipped_quantiles`,
  and `props.py` should it ever reach `weekly.project`. An accuracy loss on a locked starter
  or the deep bench changes no realised lineup point and is invisible to the gate, and the
  pilot shows the scaler produces exactly that loss (−0.024 off-band at the panel's b). So:
  **the paired MAE difference on the off-band rows of the gate's own universe — locked
  starters and bench, the rows the band excludes — must not have a season-clustered interval
  that excludes zero on the negative side.** A scaler that helps the ranking and hurts the
  number does not adopt whole; its pre-registered resolution is a **scoped adoption** —
  the scaled μ for the lineup rule, the unscaled μ for the published figure and every
  consumer of the number — which the code can express because `_contrast` and
  `shipped_quantiles` are separable, and which this document names now rather than after a
  built arm has a verdict in hand. The MAE diagnostic on the band must also be positive with
  the scaled arm closer in **every one of the four** held-out seasons — a tie is not a win, where a tie is a season whose gain does not clear
  its own player-clustered noise (the rule #335 drafts; 2024's +0.00005 above is the case). Which of A₁/A₂ ships is the one with the larger clustered gain; if they disagree in
  sign, the ticket reopens as a decision (its own clause). What follows is a change to what
  the site publishes under ADR-0016, and #309/#310's interval is re-measured on the new
  mean before its claim is read again.
- **REMOVE** — the incumbent wins every season and the interval is entirely negative: the
  implied total does not enter, with evidence.
- **SHOW** otherwise: the incumbent stays and this document says the comparison could not
  move it. Absence of evidence, not equivalence.

**Expectation, written before the number.** The screen's +0.055 partial correlation on the
total does not convert to MAE; the honest guess is a gain of a few hundredths of a point per
player-week on a mean MAE of about 4.5, concentrated on the weeks where the implied total
sits furthest from the league mean. A gain of the rebuild's own size (+0.06) or more would be
larger than expected and needs its own explanation before it is believed (rule 9).

## The #212 guard — PROPOSED disposition, the one choice here

`tests/contracts/test_the_weekly_model_is_market_free.py` asserts that `models/weekly.py`
and `models/components.py` reference no betting-market column. #305 fails it by
construction. The guard's own stated purpose is narrower than its assertion — *"so a future
feature addition cannot silently invalidate any team-level aggregate built on top of it"* —
and the property that made #222/#223 need it is narrower still and is the one to
re-register on: **what an object is scored against.**

- **Permitted:** the betting market as an *input* to a player-level projection whose gate
  is against a non-market benchmark — the incumbent projection, consensus. #305 is this.
- **Forbidden:** any object *compared to* the market, or any gate whose *benchmark is* the
  market, reading the market. You cannot test whether you beat the close with a number that
  read the close; that is the laundering #212 named, and it is tighter than the column-level
  assertion while permitting #305.

**Consumers that inherit the condition — named now, not discovered.** The one place this
repo scores itself against a market is `hub.models.props`: `edge`, `clv_prob`, the hit rate
all compare our number to a book's, and books price props substantially off the game
environment, so a market-informed statline agreeing with a prop book is partly agreement
with itself, in a direction that flatters every props figure. **Checked 2026-09-19:**
`props.py` prices from `predict.components`, and neither it nor `predict.py` reads
`weekly.project`, so nothing on the props path is market-conditioned today. The
re-registered guard keeps it that way by construction: a test that walks the callers of
`weekly.project` and refuses (a) any summation to team level and (b) any path into
`props.py`'s statline or into any column scored against the market (`edge`, `clv_prob`,
`clv_points`). If a later ticket wants the weekly projection under props, it owes props a
**market-free arm** for its CLV measurement or an explicit restatement that its edge is no
longer a clean market test — decided then, in that ticket's pre-registration, and never by
an import.

Under ADR-0025 no team aggregate exists (#222, #223 declined), so the re-registered guard
holds today. The alternative — keep the column-level assertion, and #305 never lands — is a
real option and it is the maintainer's, not this document's.

## What this measurement cannot do

- It cannot pick the clustering, the seed, the rows, or the metric after the sign is seen.
- It cannot license a change to `sd`, the skew, the clip, or efficiency; only the scaler on
  volume and touchdown rate is on trial.
- It cannot say the implied total is *causal*; it is a market number and the projection is
  a fantasy projection, which #212's re-reading permits and the team aggregate does not.
- It cannot run before #311 lands, and it cannot be read weekly: like #310's, its verdict is
  taken once, at the run pre-registered here.

## What happens either way

If the gate cannot run, the incumbent is published unchanged and this section plus the
numbers is the output — with the seasons-needed figure beside it, which is the number Phase
2's re-plan reads. If it runs and the scaled arm does not win every season with an interval
clear of zero, the incumbent stays. If it runs and the scaled arm wins, the published
projection changes under ADR-0016, #212's guard stands in its re-registered form, and
#309/#310 re-measure on the new mean before any coverage claim is read against it.

# Pilot 2026-09-19: red-zone opportunity share as a touchdown-rate modifier (#308)

Run inside the freeze beside #305's pilot, in the same shape, so Phase 2's order rests on a
measurement rather than on the plan's guess. No model changed. Recorded here under ADR-0007
because it steers the milestone.

**What was measured.** From play-by-play (`yardline_100 ≤ 20`, run and pass plays, regular
season), each player's share of his team's red-zone opportunities (rushes + targets) and of
its opportunities overall, per week; the **prior** form is the season-to-date mean over
strictly earlier weeks, which is the forecastable one. The feature is
`log((rz_share + 0.01) / (opp_share + 0.01))` — red-zone share *relative to* opportunity
share, the "holding yards fixed" proxy — applied as a multiplicative modifier on the
incumbent's touchdown component only (`μ + 6 · tds_hat · (e^{b·x} − 1)`), `b` fitted per
season in-sample as a bound. RB, WR and TE; 14,879 player-weeks with a prior.

| season | gain, prior share (bound) | gain, same-week share (ceiling) |
|---|---|---|
| 2022 | +0.0171 | +0.1173 |
| 2023 | +0.0237 | +0.1098 |
| 2024 | +0.0102 | +0.1202 |
| 2025 | +0.0275 | +0.1163 |

Prior form: mean **+0.0196** MAE per player-week, clustered SE 0.0038, **MDE 0.0153 at four
clusters** — the bound clears the MDE. Same-week form: +0.116, which like the realised total
above is outcome, not forecast.

**What it says about Phase 2's order.** #308's forecastable bound (+0.020, clearing its MDE)
is above #305's (+0.012, at its MDE). The plan of 2026-09-16 put #308 at step 3 behind the
implied total and the share layer on the reasoning that the total is the strongest screened
signal; converted into the metric that decides, the red-zone modifier is worth more and is
resolvable on four seasons where the total is marginal. The current model gives two backs
with identical yards identical touchdown expectations, and red-zone share is the single
largest thing it cannot see. Audit V re-plans Phase 2; this is the evidence it reads.
Both bounds are one-parameter in-sample fits, and a fitted Beta-Binomial shrink (#308's
own form) is the thing the build measures.

**Its pre-registration, when written, gates where #305's now does:** `hub.season.weekly_gate`
against consensus rank as the deciding statistic, the MAE contrast by μ tier as the
diagnostic. Three-quarters of this effect is on the 14% of rows at μ ≥ 15 (the table in
#305's section), which is the strongest reason of the three to gate on the lineup.

# Pre-registered 2026-09-21 — PROPOSED: the share layer and additivity's gate (#306)

**Status: PROPOSED.** Drafted in the grilling of 2026-09-20/21, inside the freeze (#326), in
#305's form. It becomes the rule on the maintainer's `ADOPTED:` comment on #306. Parent
decisions: ADR-0016 (the weekly projection is shown, never ranked on), ADR-0006 (a fitted
constant lives with its provenance), #314 (the teammate estimand is conditional on team
output), and the three pilots in #305's section above, which this document argues from.

## The object: one structure, two consumers

**Additivity** — a team's projected volume of each count type is distributed over its **active
set** as **shares** that sum to one, and a player's projected count is team volume × his
share — is the structure. **Substitution** (a starter out, the mass moves) and the **teammate
correlation** the simplex induces are consumers of it: neither exists without shares, and each
has its own gate. So the decision is the *form* of the share model, and the consumers follow.

**The form: a Dirichlet-multinomial over each count type's active set**, with a fitted
concentration — a real constant under ADR-0006, provenance and hold-out. Not a mechanical
rescale: that buys additivity and nothing else, and both reasons to build this live in the
parameter it lacks. **Per count type**: the carry set and the target set are different
populations with different dynamics (below), and one simplex would be two processes wearing
one name.

**The estimator's disattenuation, pre-registered as part of the estimator.** A DM's
concentration is a dispersion, and fitted on observed share variance it absorbs the
multinomial sampling noise of a share computed from ~5 events — the concentration comes out
too low for the reason the TD rate's year-over-year correlation once came out at zero (C5).
The sampling variance of a share at a known count is analytic, `p(1 − p)/n`, and is
subtracted from the observed between-week variance before the concentration is fitted.
Second instance of this correction in Phase 2 (#308 is the first); `method.md` notes it.

**Boundary-running is a pre-registered result, not a fit failure.** Target share's
disattenuated lag-1 is **[0.95, 1.00]** — the data cannot distinguish *very sticky* from
*literally fixed*. That is informative, and it has a mechanical consequence: a DM's
concentration is unbounded above as the share becomes deterministic; the likelihood flattens
toward infinity and the fit runs to a bound or needs a cap or prior to terminate. A fitted
constant sitting at its own boundary has a provenance that reads "the data wanted more than
the parameterisation allows", which is not what ADR-0006 means by a fitted value. So, fixed
now rather than at fit time: **if the target-side concentration runs to its bound, the finding
is that targets are effectively fixed-share — the DM collapses to a multinomial with a
deterministic share vector — while carries, at 0.88 [0.85, 0.89], clearly do not.** The two
count types may want different *models*, not only different parameters, which is the cleaner
form of the per-count-type argument. The degenerate case is a measurement, not a bug to
work around, and a target-side concentration at its bound is evidence about targets, not
evidence the estimator failed — which protects the null reading of the gate the same way the
active-set caveat does. Neither a cap recorded as a choice nor a weakly-informative prior is
used; both would convert that measurement into a number chosen to make the fit terminate.

**The active set, and how a player leaves it — the structure's weakest link, stated.** The set
is latent: the players with a stat row for that team in strictly earlier weeks, recency-
weighted, per count type. A player leaves it when he is **OUT or IR on the week's injury
report, or has no stat row in the last k = 3 weeks (`MIN_GAMES_BEFORE`)**. **Status wins**:
a player OUT this week with three recent rows is *out* — the opposite precedence is the hole
that silently keeps an absent player on the simplex and dilutes every teammate in exactly the
weeks additivity exists for. The gating source is the one #221 characterised as weak (one
Friday-stamped row per player-week; fourteen quarterbacks OUT across all of 2024), so **a
null on this gate is first a claim about the active set and only then about the share
model**, and the report says which. A cached depth chart is the input substitution wants and
is its own ticket; #306 does not wait on it.

**Which volume**: team attempts and carries from the panel's own history — the incumbent's
estimator lifted one level — with no dependency on #305. The implied total enters, if it
ever does, as a term on team volume *after* shares.

## The hypothesis, as three measurements say it (2026-09-20/21, no model changed)

*Lag-1 autocorrelation within season, consecutive weeks, unfiltered:*

| quantity | observed r | reliability | disattenuated r |
|---|---|---|---|
| team pass attempts | +0.18 | — | — |
| team rush attempts | +0.13 | — | — |
| player targets | +0.51 | | |
| player **target share** | +0.55 [0.52, 0.57] | 0.56 [0.53, 0.59] | **0.97 [0.95, 1.00]** |
| player carries | +0.67 | | |
| player **carry share** | +0.77 [0.75, 0.80] | 0.88 [0.87, 0.90] | **0.88 [0.85, 0.89]** |

(773 / 469 players, 500 player-cluster bootstraps; the correction assumes binomial dispersion,
so under overdispersion the true r is lower.) *And whether volume is predictable before
kickoff:* team pass attempts on a within-season walk, n = 2,238 team-weeks — R² on the
season-to-date mean **0.058**; adding spread and total **0.075**; residual sd 8.0 of 35
attempts either way; the total's coefficient +0.24 attempts per point, the spread's −0.03.

> **Share is predicted by persistence — near-perfectly for targets once sampling noise is
> removed, strongly for carries. Volume is not predictable before kickoff by anything this
> repo has — not persistence, not the market — and it is common to every player on the team.**
> So the decomposition's value is not a better mean per player: the incumbent's season-to-date
> count already averages share × volume. Its value is that the volume error becomes
> **common-mode across teammates** — the positive environment shock #314 says the marginal
> +0.014 hides, the exposure of a lineup holding two players from one team, the thing
> substitution redistributes over. **The structure pays at the joint distribution, and its own
> per-player gate is the least of the three.**

Two earlier forms of this hypothesis were withdrawn on measurement: "team volume is the
stable base" (it is the least stable quantity in the table) and "volume is predicted by the
market" (R² +0.017). Both are kept here as the record of what the numbers refused.

**The primary prediction, falsifiable, written before any number:** **roster-total coverage
improves with no change in per-player MAE.** The scale it should move on is the L1 gate's
own: 72.9% → 80.4% for a lineup holding a quarterback and his pass-catchers when
`TEAMMATE_RHO` was added (`docs/correlation.md`). A near-null on additivity's MAE gate is the
*predicted* outcome, not an excuse available afterwards.

**The sign, per count type — and the double-count risk, pre-registered.** With target share
near-constant, the negative "one up, others down" channel is nearly closed for pass-catchers,
and what the target simplex induces is the **positive** volume term. For the backfield,
0.88 leaves a real share-variation channel open and the induced correlation is a mix.
`TEAMMATE_RHO` is already positive (QB–WR +0.232) and may be partly measuring the same
volume channel. **If the simplex adds positive correlation on top of it, roster-total coverage
can pass 80% from below — too wide, not too narrow — and the two-sided coverage gate would
fail in the unexpected direction.** Written now so that outcome is read as double-counting
to be resolved (the simplex's term replacing, not adding to, the part of `TEAMMATE_RHO` that
is volume), not as the structure failing. #314's section carries the same sentence.

## The gate

1. **Coherence is a VOID condition, not a diagnostic.** Per team-week and count type, projected
   shares over the active set sum to one within floating tolerance and the player projections
   sum to projected team volume *by construction*. Anything else is an implementation failure
   and the run VOIDs, in the shape of the join-failure VOID. Team-volume accuracy (projected
   vs realised attempts) is printed as a diagnostic and decides nothing.
2. **The deciding statistic is `hub.season.weekly_gate`**, exactly as #305's: the share-layer
   projection sets a lineup by *start your highest* against one set by consensus rank, on the
   drafted universe, rules unchanged, with its foresight ceiling and stage 2 first. A coherence
   constraint plausibly moves the decision band most (a WR3's share *is* the margin); that is
   the hypothesis this gate tests rather than assumes.
3. **Do-no-harm, as #305's:** the off-band paired MAE interval must not exclude zero on the
   negative side, and the concentration is fitted by the stated procedure on strictly earlier
   seasons and is not free after any of these is seen.
4. **Diagnostics:** the MAE contrast by μ tier; **share error** — the projected share against
   the realised share, per count type, against the incumbent's implied share
   (`count_prior / team_prior`) — which is where the hypothesis says the gain lives if there
   is one; and roster-total coverage on the L1 harness, printed here and *decided* in #314.
5. **The bar, in `experiment.gate`'s order:** NOT-RUNNABLE if the clustered MDE exceeds the
   ceiling; ADOPT only on a positive interval and a win in every held-out season (ties not
   wins, per #335) *and* do-no-harm; REMOVE on the mirror; SHOW otherwise — and **a SHOW with
   the primary prediction met** (coverage moved, MAE did not) is the structure working, and
   the record says so.

## What this measurement cannot do

- It cannot gate substitution: the redistribution needs an expected active set a depth chart
  supplies, and that loader does not exist. Substitution is its own ticket behind it.
- It cannot decide the teammate correlation: that is #314's coverage gate, reading the
  simplex's term.
- It cannot fix the volume leg: R² 0.075 is what the repo has before kickoff, and the
  game-script term (a spread effect on volume) is its own ticket.

## What happens either way

If the gate cannot run, the incumbent is published unchanged and this section plus the tables
is the output. If it runs and the primary prediction holds — coverage moves, the mean does
not — the simplex stays as the object the consumers read and its per-player verdict is
reported as the null it predicted. If the lineup gate adopts, ADR-0016 reopens by its own
gate and the do-no-harm branch decides scope. If coverage overshoots, the double-count is
the finding and `TEAMMATE_RHO` is re-fitted conditional on the simplex before anything ships.

# Pre-registered 2026-09-21 — PROPOSED: the restatement `paired_gain`'s cluster argument
obliges (#311)

**Status: PROPOSED.** Drafted during the operate-mode freeze (#326), which blocks a *landing*
and does not block writing the rule before the number (#329's precedent, restated at #305's
and #306's own headers). Becomes the rule on the maintainer's `ADOPTED:` comment on #311.
**This is a restatement, not a gate** — #311 has no incumbent to beat and no candidate axis
ADR-0024 could route to a decision; it corrects an input to statistics three modules already
compute, and this section pre-registers which published numbers move, what each is re-read
against, and the order, per `docs/method.md` rule 13. Parent decisions: ADR-0019 and its
2026-09-21 amendments (`SEASON_CLUSTER`, the tie-aware every-season half, S1's t-interval),
rule 3 (repeated measures), rule 13 (a contradicted number is not restated until it moves).

## The defect, restated

`paired_gain`'s `se`/`t` are computed over raw rows — `d.std(ddof=1) / sqrt(len(d))` — with no
cluster argument, while `run_gate`'s own `summarise` has clustered on `SEASON_CLUSTER` since
before this document existed. The three call sites that bypass `run_gate` (`hub.models.weekly`,
`hub.models.spread`, `hub.models.injury`) therefore pool the significance half over ~30,000
player-weeks (weekly) or ~1,000-2,000 player-seasons (spread, injury), and every `t` printed
from any of them is a *t* on that many degrees of freedom minus one rather than on `k - 1`
seasons. #335 already gave `paired_gain` `season=`/`within=` for the every-season half; `cluster`
is the missing third argument, for the pooled half, and — per the pattern `summarise`'s own
docstring states — it takes no default, for the same reason `cluster` and `within` do not
elsewhere: guessing the unit is the mistake, not a convenience a caller can skip.

**What moves inside `paired_gain` when `cluster` lands, named so the restatement is not a
surprise.** Two things change together, not one:

1. **`se`/`t` are computed over cluster-mean units**, the same construction `summarise` and
   `_cluster_se` already use elsewhere in this module — group the rows by `cluster`, average
   `diff` to one number per cluster, then take `se`/`t` from that vector rather than from the
   raw rows. This is an implementation-consistency choice, not a modelling one: the
   alternative — a bootstrap SE over the cluster-mean units, matching `_bootstrap_se`'s
   percentile resampling exactly as `summarise` does for `run_gate`-driven gates, rather than
   the closed-form `sd(cluster means) / sqrt(k)` the 2026-09-19 pilot used by hand — is the
   one thing here that could go two ways without being ADR-0024's "different objects." This
   document recommends the bootstrap form, for one reason only: it is what every other
   clustered SE in `experiment.py` already is, and a fourth mechanism for one function's one
   argument is exactly the drift ADR-0019 unified away. The hand arithmetic below uses the
   closed form because it is what a human can check in a sentence; the two agree closely at
   these `k`, and are not guaranteed to at `k` below `SMALL_CLUSTERS`.
2. **`mean` becomes the mean of the per-cluster gains**, not the row-weighted mean of the raw
   `diff` array. On an unbalanced panel these differ — a season with more player-weeks
   currently pulls the pooled mean toward its own gain — and every "mean gain" figure quoted
   from these three modules is, after this lands, an equal-season-weighted number for the
   first time. This is the same correction `weekly_screen.summarise` already made for the
   screen (#169) and #311 applies it to the three modules that never made it.

## Worked restatement, by hand, from numbers already published (method.md rule 13's own style)

Not a re-run — the same hand arithmetic #311's and #312's own 2026-09-19 comments used on the
weekly gate's rebuild-vs-flat contrast, applied here to `docs/player-spread.md`'s published
per-season MAE table, because that table already has the five numbers a season-clustered `se`
needs and nothing about reading them differently requires code to move first.

**`own_k` vs `positional`**, per-season gain (`mae_positional − mae_own_k`):

| season | positional | own_k | gain |
|---|---|---|---|
| 2021 | 1.0914 | 1.0881 | +0.0033 |
| 2022 | 1.1523 | 1.1442 | +0.0081 |
| 2023 | 1.0374 | 1.0296 | +0.0078 |
| 2024 | 1.1573 | 1.1467 | +0.0106 |
| 2025 | 1.0414 | 1.0385 | +0.0029 |

Mean **+0.0065** (matches the published pooled mean to four places — the panel is close enough
to balanced across these seasons that the two weightings agree here, which will not be true of
every gate this ticket touches). Clustered analytic SE `sd(gains, ddof=1)/sqrt(5)` **≈ 0.00149**,
**t ≈ 4.4** on 4 df — against the published, unclustered **"1.8 se."** Clustering *raises* this
one's significance rather than lowering it, which is exactly `docs/method.md` rule 3's own
caveat that clustering is not a synonym for a wider interval: the between-season scatter of
`own_k`'s gain is small next to what the row-pooled SE implied. **This is pre-registration
arithmetic, not a run** — the real number comes from `paired_gain(cluster=...)` once #311
lands, and it is flagged here because it changes which side of `MIN_SE` `own_k` sits on, which
is exactly the kind of thing rule 13 says must not be left standing once known. It does not by
itself license adoption: `own_k` already wins every season (5/5) and #343's ceiling for this
gate — player-spread.md's own **0.085 headroom** — is what stage 2 checks the restated MDE
against, not this document.

**`usage` vs `positional`**, same construction: gains −0.0238, +0.0434, +0.0154, −0.0093,
−0.0298; mean **≈ −0.0008** (published: −0.0004), matching direction and magnitude. Mixed sign
in both weightings (2 of 5 positive either way), so `usage` fails the every-season half under
clustering exactly as it does today — clustering does not change its disposition, only
`own_k`'s significance does that.

## The restatement order

1. Land `cluster` on `paired_gain`, with the contract test `#311`'s acceptance criteria name
   extended to the three call sites the way `test_gates_cluster_on_the_season.py` already
   covers `run_gate`'s.
2. Re-run `hub.models.weekly`'s rebuild-vs-flat contrast and restate
   `docs/weekly-projection.md` and `docs/weekly-projection-plan.md` first — the incumbent
   figure `#305` and `#306`'s pre-registrations already cite as *"the incumbent's own
   standing"* (t = 2.66 on 3 df, p = 0.076, per #311's and #312's 2026-09-19 comments), and
   every downstream pre-registration in this document that quotes it inherits the restated
   number rather than the superseded one.
3. Re-run `hub.models.spread.verdict` and restate `docs/player-spread.md`, carrying the
   `own_k` finding above into whatever `#343` does with it.
4. Re-run `hub.models.injury.type_verdict` and restate `docs/weekly-injury.md`,
   `docs/method.md`'s measurement table (row 13) and `docs/where-to-look-next.md` — flagged
   explicitly as a *statistic* restatement only: the module's *decision* stays frozen behind
   `#360`/S3, and the restated sentence should say so rather than silently reading as settled.
5. Restate the two `docs/improvements.md` citations of the frozen weekly gate's season count
   (a historical regression-check page, lowest priority of the group since nothing downstream
   cites it as current).
6. Restate `docs/weekly-shrinkage.md`'s `2/4 seasons` figure.
7. Name `#302`'s quarterback log-loss diagnostic in the restatement, per #311's own acceptance
   criterion — it licenses no decision under #300's demotion, so restating it is a courtesy to
   a reader who finds the old number, not a step anything else waits on.

This list is inherited from #335's landing comment on #311 (2026-09-21), itself sourced by
grep rather than an audit of every doc under `docs/` — #311's own work should treat it as a
floor, not a ceiling.

## What this measurement cannot do

- It cannot re-derive a number this document has not already published; the `own_k` table
  above is arithmetic on `docs/player-spread.md`'s own rows, not a new measurement.
- It cannot decide `own_k`'s fate — that is `#343`'s gate, run with a declared ceiling, once
  both land.
- It cannot touch the weekly, lineup, draft, coverage, margin or quarterback-diagnostic gates:
  all six already read `run_gate` → `summarise(cluster=SEASON_CLUSTER)`, independent of
  `paired_gain`'s own `cluster` argument, so #311 does not move any of their published figures.
- It cannot touch a screen: `weekly_screen` is a different test (`CONTEXT.md`) and #312 is its
  own restatement.

## What happens either way

If `cluster` lands as the bootstrap-clustered form recommended above, the restatement order
runs as listed and every figure carries a note that its se/t/mean changed basis. If the
maintainer instead prefers the closed-form analytic SE over cluster means — the one open
choice this section did not make — the restated *numbers* differ only at a decimal a bootstrap
at these `k` would not move by much, and the order above is unchanged either way.

**Closing.** Candidates: (a) bootstrap-clustered SE over cluster-mean units, matching every
other clustered SE `experiment.py` computes (recommended, for consistency alone); (b) the
closed-form `sd(cluster means)/sqrt(k)` the pilot used by hand. Recommendation: (a). What
would reopen this: a gate whose `k` sits at or below `SMALL_CLUSTERS` (8), where the two
mechanisms can disagree enough to matter and `summarise` itself already prints the percentile
bootstrap beside the t interval for exactly that reason.

# Pre-registered 2026-09-21 — PROPOSED: the screen's verdict reads the p it computes (#312)

**Status: PROPOSED.** Drafted under the #326 freeze — writing the rule, not running it.
Becomes the rule on the maintainer's `ADOPTED:` comment on #312. **A restatement, not a
gate**: `weekly_screen.verdict` has no incumbent and no candidate axis; it corrects the
bar a pre-registered rule already uses, and this section says which published screen results
move, what each is re-read against, and the order. Parent decisions: `docs/method.md` rule 14
(the false-discovery threshold is a diagnostic, the pre-registered rule decides), #37 (family
size printed beside the rule), #274 (CRPS's one narrow promotion, the same "a threshold is
not a decision until it is pre-registered as one" logic this ticket extends), and #169 (the
screen's own season-clustering fix, the direct predecessor of this defect in the same module).

## The defect, restated

`weekly_screen.verdict` (`src/hub/models/weekly_screen.py:438`) compares `abs(t) < min_se`
(2.0, flat) rather than the two-sided `p` at the run's own degrees of freedom against `ALPHA`
(0.05) — and `two_sided_p`/`t_quantile` already exist in `experiment.py`, eleven lines away
from every place this bar is read, built for exactly this. At `k = 4` seasons (`df = 3`) a
two-sided `p < 0.05` needs `|t| > 3.182`; at `k = 5` (`df = 4`) it needs `|t| > 2.776`. The
flat bar of 2.0 is neither — it is the `p = 0.05` bar's normal-quantile approximation
(`z(0.975) = 1.96`, rounded), the same reference-distribution confusion `minimum_detectable_effect`
was corrected for on 2026-09-07 (`docs/gate-power.md`, "Restated 2026-09-07: the MDE's
reference distribution"), landed in the gate's own MDE and never carried to the screen's own
`min_se` comparison four lines away in the sibling module.

**The headline case, from #312's own ticket body.** `td_rate_prior`, this screen's one
pre-stated null, is quoted "throughout the tree" at **−2.49 se**, `p = 0.089` unadjusted — and
at the family's false-discovery threshold it already fails at rank one. Under the *current*
rule (`abs(t) < min_se`), `2.49 ≥ 2.0` means it does **not** clear as the pre-stated null and
is read as a broken one (`NULL_BROKEN` if the sign is consistent across seasons, `CLEARS` with
a "noisy" caveat otherwise — `weekly_screen.verdict` lines 451-457). Under the corrected rule,
`0.089 > 0.05` (`ALPHA`) means it **does** clear as consistent with the pre-stated null. This
is the one figure named directly in the ticket body and the clearest case where the bar
change flips a reported status, not only a margin.

**`docs/method.md` rule 3's own restated figure for the same feature is a second, earlier
version of this number** — "the prior TD rate against passing attempts, published as a null at
1.9 se, is a broken pre-stated null at 2.7 se across 5/5 seasons" — computed under an earlier
run (#169's fix landing) at a different season count than the −2.49 se the ticket body quotes
today. The two are not in tension so much as evidence of how often this number has already
moved: #312's restatement is not the first time `td_rate_prior`'s status changed, and its own
write-up should say so rather than treat −2.49 se as freshly discovered.

## What is measured

1. **The corrected verdict.** `weekly_screen.verdict` compares `two_sided_p(t, df=seasons-1)`
   against `ALPHA` in place of `abs(t) < min_se`, or equivalently reads the `t_quantile(1 -
   ALPHA/2, df)` at the run's own `df` rather than a constant — the ticket's acceptance
   criterion permits either phrasing and this document does not choose between them, since
   they are the same comparison stated two ways and not a modelling alternative.
2. **The joint size of the two-part rule, by the existing permutation harness.** The screen's
   `every_season_null` (`weekly_screen.py:710`, already the source of
   `docs/weekly-screen.md`'s "null: P(≥1 season wrong sign)" table) simulates the every-season
   half under the null. It is extended — not replaced — to also gate the simulated draws on
   the corrected `p < ALPHA` condition and report the **joint** rejection rate beside the
   marginal ones already printed, the same shape `test_the_size_check_flags_a_planted_degenerate_rule`
   uses for rule 17's gate-side check. This answers the question the ticket's own body raises
   and does not answer — "the two halves are correlated and their joint size has never been
   computed" — without inventing a second mechanism: the harness already exists and the
   ticket names it as the thing that "could produce it directly."
3. **The family size, counted across anchors and bases.** `with_family`'s `m` is scoped per
   call today (rule 14's own table: 8 for `--run`, 9 with `--routes`, 13 with `--scheme`); the
   ticket's third acceptance criterion asks it to count everything actually run in one report,
   which is a reporting change to `with_family`'s caller rather than to the false-discovery
   arithmetic itself — `false_discovery`'s own `m` semantics (every test the caller ran, NaN
   included) are unchanged.

## The restatement order

1. Land the corrected `verdict`, with the exact wording (`p` vs `min_se`, or the equivalent
   `t_quantile` phrasing) at the maintainer's choice.
2. Re-run the screen at its published anchor and restate `td_rate_prior`'s status first — it
   is the one figure named in the ticket body and quoted "throughout the tree."
3. Extend `every_season_null` for the joint size and print it beside the existing marginal
   figures in `docs/weekly-screen.md`.
4. Restate `docs/method.md` rule 3's TD-rate paragraph and rule 14's four-survivor table,
   both of which quote `t`/`p` figures a corrected bar reads differently even where the
   verdict does not flip (rank order and margin both move).
5. Correct every module citing `td_rate_prior` as an established finding — the ticket's own
   fourth acceptance criterion — which a grep for the feature name across `src/` and `docs/`
   should enumerate at implementation time; not attempted here without running one.

## What this measurement cannot do

- It cannot decide whether the family-size fix (item 3 above) changes any other feature's
  status; only `td_rate_prior`'s flip is asserted here, from the number already in the ticket
  body.
- It cannot touch a gate: `experiment.gate`'s own interval half already reads a t interval
  (S1/#357) rather than a flat bar, so this defect's twin on the gate side is already fixed;
  #312 closes the screen's copy of the same class of error.
- It cannot re-run every anchor and basis combination; the restatement lands at the published
  anchor first (rule 14's own convention) and other anchors follow if their status is cited
  anywhere as current.

## What happens either way

If the corrected bar changes no status beyond `td_rate_prior`, the restatement is short and
the joint-size figure is the more interesting output — the first honest answer to how
correlated the two-part screen's halves are. If it changes others, each is restated in place
with the superseded figure kept beside it per rule 13, and the false-discovery table is
re-printed at the same run so a reader sees the diagnostic and the corrected rule together.

# Pre-registered 2026-09-21 — PROPOSED: four arithmetic items that move published numbers
(#315)

**Status: PROPOSED.** Drafted under the #326 freeze. **A restatement, four times over, not a
gate**: none of the four items compares a candidate to an incumbent under ADR-0019 — each
corrects a closed-form step inside an existing estimator, and this section names the location,
the correction, and the published figure each moves. Parent decisions: rule 9 (a result too
large to believe is a bug), rule 13 (restatement protocol), ADR-0006 (a fitted constant lives
with its provenance — relevant to item 4, below, which borrows a constant that already has
one). Located by reading the code the audit finding (`prediction_evaluation` rank 8) names,
not by re-deriving each closed form from scratch in this document.

## Item 1 — the absence factor scales the draw by `1/f`, not `1/√f`

**Location.** `hub.draft.season._absence_factor` (`src/hub/draft/season.py:110-174`). The
mean-preserving scale at the function's last line, `played / play_frac[None, None, :]`, divides
every drawn weekly value — for a played week — by the play fraction `f`. That is the correct
transform for the **mean** (it is the "definitional conversion" the function's own docstring
argues for, and that argument is not in question): `mu` arriving here is already a per-scheduled-game
average net of expected absence, so dividing a played week's draw by `f` recovers the
*healthy* per-game rate. **The same division also scales that week's simulated *spread* by
`1/f`**, when the closed form for a mean-preserving rescale of a played-week's *noise* term —
holding the season-total variance to what an unbiased absence model implies — is `1/√f`,
one square root short of what the code does. The audit's own estimate: **about +10% too wide**
in weekly sd for a player expected to miss three of seventeen (`f ≈ 14/17 ≈ 0.824`,
`1/f ≈ 1.214` against `1/√f ≈ 1.102`).
**What moves.** Every season-simulation figure downstream of `_absence_factor` for a player
with nonzero `missed` — win probabilities, VOR, the draft board's ranking sensitivity to
injury history — inherits an overstated variance for exactly the players durability already
flags. `docs/durability.md` cites the function and is the doc to restate first.

## Item 2 — the consensus target is biased low by an uncorrected log transform

**Location.** `hub.models.weekly.fit_consensus_prior`/`consensus_target`
(`src/hub/models/weekly.py:127-173`). The OLS is fit on `log(1 + count)` (`np.log1p`) against
`log(rank)`, and the prediction is `expm1(a + b·log(rank))` — a back-transform with **no
smearing correction**. `E[exp(X)] > exp(E[X])` under any residual spread, so a naive
`expm1` of a log-scale linear prediction is a biased-low estimate of the level-scale mean,
by a factor Duan's smearing estimator (`mean(exp(residuals))` in place of `exp(0)` in the
back-transform) corrects for. `hub.models.spread` already carries this exact argument in its
own docstring and *declines* the correction there, for a stated reason: its loss is MAE, and
Duan's correction targets a mean-loss estimator, not a median-loss one. `hub.models.weekly`
uses the target as a **shrinkage mean** (`Shrink.consensus_prior`, feeding a mean, not a
median), so the same argument that excuses `spread` obligates `weekly`. **What moves.** Every
figure the consensus-prior shrinkage feeds — the imputed/thin-sample projections it pulls
toward — moves up by the smearing factor, concentrated on the thin-sample players the prior
exists for. No doc currently quotes `consensus_target`'s own bias by number; the restatement's
first job is to measure the smearing factor once fit, which is new information rather than a
correction of a published figure, and the *downstream* projections it moves are what rule 13
obligates.

## Item 3 — the imputation is a greedy projection, not the unbiased monotone fit

**Location.** `hub.draft.board._impute_xfp` (`src/hub/draft/board.py:564-606`), specifically
`sm = np.minimum.accumulate(sm)` at line 603. A running minimum forces monotone decline by
clamping every point down to the smallest value seen so far — which sits **at or below** the
smoothed curve everywhere a violation existed, never above — rather than the least-squares
monotone fit, which moves the *violating neighbours* toward each other (pooling their
average) and can move either up or down. **`docs/impute-cv.md:87` already names the symptom**
— "the curve also sits low for rookies" — without identifying the mechanical cause; this item
is that cause. PAVA (pool-adjacent-violators) is the closed-form unbiased least-squares
monotone regression and replaces the rolling-median-then-clamp construction; the acceptance
criterion is stated as a direction ("the median residual moves toward zero"), not a target
value, because the fitted curve moving is the finding.
**What moves.** Every imputed `xfp_per_game` for a rookie or thin-sample player — `#87`'s own
`talent_cv_for` widening logic reads the imputation's *measured error*, so a less-biased
imputation changes that widened spread too, not only the point estimate. `docs/impute-cv.md`,
`docs/talent-cv.md` and `docs/decisions.md` cite the imputation and are the restatement's
targets, in roughly that order (impute-cv.md names the mechanism, the other two cite its
output).

## Item 4 — defence-versus-position has no opponent-strength adjustment

**Location.** `hub.models.panel`'s `dvp` column (`src/hub/models/panel.py:~1180-1194`),
`dvp = allowed_prior / lg` — a raw sum of points allowed over a league mean, with no term for
the strength of the offences that produced `allowed_prior`. The identical problem, for the
identical reason, was already solved once in this repo: `hub.draft.playoff_sos._dvp_from_stats`
and `_ridge_defence_effects` (`src/hub/draft/playoff_sos.py:67-149`) fit a ridge-penalised
two-way (offence, defence) effects model for exactly the strength-of-schedule question #180
built for the playoff slate, with the penalty itself a stated choice
(`RIDGE_PENALTIES`/`DraftConfig.sos_ridge`, `not_an_input` per ADR-0006's own provenance rule).
**What "shares the ridge" means, precisely, and the one place this is not purely mechanical.**
`_ridge_defence_effects` is fit per-position on a season's worth of games with `TEAM_GAMES`-scale
counts; the weekly panel's `dvp` is read week-by-week, strictly before the outcome (rule 2), so
the ridge would need to be **refit walk-forward on strictly earlier games within the season**
rather than reused from the playoff module's own season-level fit — a different estimation
window on the same functional form, not a different form. That is an implementation detail
the ticket's acceptance criterion ("opponent-adjusted, sharing the playoff-schedule ridge")
does not resolve by itself, and this document does not resolve it either: it is the one place
in this ticket closest to a modelling choice (which games are "strictly earlier" for a
week-*w* `dvp` — the current season only, or prior seasons pooled in), and per ADR-0024's own
test it stays a candidates-named question rather than a silent pick, below.
**What moves.** `dvp` is a screened, surviving feature (rule 14's table: `t = +3.76`, adjusted
`p = 0.053`, one of the four that "survives" at the published anchor) and feeds
`hub.models.weekly`'s components directly — an opponent-adjusted `dvp` changes the screen's own
`t` (a new number, not a restatement of the old one, since the feature itself changes) and
every downstream weekly figure that reads it. `docs/weekly-screen.md`, `docs/method.md`
(rule 14's table) and `docs/weekly-projection-plan.md` are the restatement's targets.

## The restatement order

1. Item 1 (absence factor) first — narrowest blast radius (the draft season simulator alone),
   no dependency on the other three, and the closed form is the least contestable of the four.
2. Item 3 (imputation) second — same reasoning, contained to `board.py`'s imputed rows, and
   `docs/impute-cv.md`'s own text already anticipates the direction the fix moves.
3. Item 2 (consensus target) third — touches `hub.models.weekly`'s shrinkage machinery, which
   #311's restatement also touches (a different statistic, the same module); landing this
   before #311's re-run means the smearing-corrected target is what #311's restated weekly
   figures are computed against, not a target that moves a second time under them.
4. Item 4 (defence-vs-position) last — the one item with an open estimation-window choice
   (below) and the one whose restated feature changes what the weekly screen measures rather
   than only how accurately it is reported, so it should not be rushed ahead of the other three
   to avoid a second screen re-run.

## What this measurement cannot do

- It cannot fit item 4's ridge penalty or its estimation window here; both are chosen at
  implementation time, and the window choice is named as an open question rather than decided.
- It cannot quantify items 2 and 4's exact movement without running the corrected code — no
  smearing factor or opponent-adjustment coefficient is fabricated in this section.
- It cannot say whether item 4 changes `dvp`'s screen status (survives vs. does not); the
  screen's own re-run, not this document, answers that.

## What happens either way

Items 1 and 3 restate cleanly regardless of any other decision in this document. Item 2's
restated figures depend on nothing else landing first (spread's own decision not to correct is
unaffected — it is a stated exception, not an oversight). Item 4 either resolves the estimation
window as a stated choice at implementation time or is split into its own decision ticket if
the maintainer reads the window as ADR-0024's "different objects" rather than a detail — the
candidates are named below for that reading.

**Closing, item 4 only — the other three have no candidates to choose between.** Candidates:
(a) refit the ridge walk-forward, within-season, on strictly earlier weeks of the *same*
season (consistent with rule 2, and with a thin early-season sample); (b) refit it once per
season on the *prior* season's full game log, the way `playoff_sos` itself is computed (a
stabler fit, available from week 1, but not rule 2's "strictly earlier" window in the sense
`expanding_seasons` means it — a game log a season old rather than a season boundary).
Recommendation: none — the choice changes what `dvp` *is* (in-season current-form signal vs.
last-season baseline), not only how well it is estimated, which is exactly ADR-0024's line
between a sensitivity and a decision. What would reopen this: either candidate measured
against the other and reported as a sensitivity, per ADR-0024's own default, if the maintainer
reads it as one axis rather than two objects.

# Pre-registered 2026-09-21 — PROPOSED: spread and injury route through the one Gate rule
(#343)

**Status: PROPOSED.** Drafted under the #326 freeze; #343 is itself `blocked_by` #326 in the
tracker (the ticket body says so) and this section is what lets its build start against a
written rule the day the freeze lifts, per this document's own standing instruction. Becomes
the rule on the maintainer's `ADOPTED:` comment on #343. Parent decisions: ADR-0019 (one Gate
rule, `experiment.gate`, read by every gate in the repo — the claim #343 makes true) and its
2026-09-21 amendments (S1's t-interval, #335's tie-aware every-season half, S6's ceiling
precondition — all three land on spread and injury for the first time here), rule 8 (compute
the ceiling before chasing the gap — `docs/player-spread.md` already did, for one of the two),
rule 3 (the repeated-measure unit, per gate). **Precedent, named because #343's own ticket
body names it:** `hub.models.margin`'s `_house_rule` already wraps `summarise`+`gate` directly
(not literally `run_gate`, but the same rule) and had its ceiling wired 2026-09-21 (`03d119f`)
— the shape of what this section proposes for spread and injury is the shape margin already
has, not a new pattern.

## The question

**Does `own_k` or `usage` beat `positional` as the estimator of a player's weekly spread, and
does the injury type-adjustment beat retention, each under the fixed Gate — ADOPT/REMOVE only
on a t interval excluding zero *and* every held-out season won (ties not wins, per #335), and
NOT-RUNNABLE, in both directions, if no ceiling is declared (S6)?** Today both harnesses answer
a narrower question by hand (`g.wins == seasons and g.t >= MIN_SE`, a flat bar with no stage 2
and no ceiling), which is exactly the drift ADR-0019 was written to end in the other three
gates and never reached in these two.

## What is measured

Two independent gates, not one — `own_k`/`usage` vs `positional` and `type` vs `retention`
share no rows, no incumbent and no ceiling, and #343's ticket body's "each comparison ... is
one `run_gate` call" is read literally: **three `run_gate` calls** (own_k, usage, and the
injury type), each against its own incumbent.

1. **The rows, unchanged from today.** `hub.models.spread.walk_forward`'s per-player-season
   errors (2019–2025, `MIN_GAMES=8`, `MIN_PPG=3.0`, matching `docs/weekly-spread.md`'s sample)
   and `hub.models.injury.walk_forward_type`'s per-player-week errors (2023–2025, the seasons
   `MIN_CELL` and the crosswalk currently support). Nothing about the rows changes; only the
   rule reading them does.
2. **The arms, unchanged.** `own_k` and `usage` against `positional` (spread); `type` against
   `retention` (injury) — `docs/weekly-spread.md`'s and `docs/weekly-injury.md`'s own
   incumbents, exactly as pre-registered when each was first run.
3. **Pairing (rule 7).** Already paired — both arms score the same player-season or
   player-week, the construction `paired_gain` already assumes. Unchanged.
4. **Clusters.** `cluster=SEASON_CLUSTER` for the pooled half (the season), same as every
   other gate — this is what #311 has to land first for the pooled half to mean what
   `run_gate` means elsewhere; #343's own acceptance criterion says so ("before #311, so the
   cluster argument reaches all eight harnesses at once" is #343's own framing of the
   dependency, read the other way: #343 is what makes #311's fix reach these two).
   **Within-season unit, per #335's own table:** spread's is already `player_id` (wired at
   `spread.py:373`, ahead of `run_gate` adopting it); injury type's stays the **declared
   no-op** `season` (`injury.py:315`), pending #360 — #343 does not change it, and this
   section does not pre-empt #360's own ticket.
5. **The ceiling arm, declared for each — mandatory under S6, and this is the one thing
   neither harness has ever measured.**
   - **Spread: `docs/player-spread.md`'s own headroom, already computed.** The outcome being
     predicted is a *realised* per-player-season sd from ~14 games, itself an estimate with
     sampling error; the ceiling is *a model that knows every player's true volatility
     exactly*, bounded only by that sampling noise. Published: irreducible sampling noise
     **1.0113** MAE against the shipped positional model's **1.0965** — ceiling gain
     **+0.0852**. This is rule 8's own worked example in this repo, already written up, and
     #343 is what lets `experiment.gate`'s stage 2 read it as a `Ceiling` rather than leave it
     as prose. No re-measurement is proposed; the ceiling arm is what `player-spread.md`
     already is, wired to `summarise(ceiling=...)`.
   - **Injury type: declared, not yet measured — *the best per-injury-type multiplier chosen
     in-sample on these rows*.** Same construction as #305's and #306's own ceiling arms and
     the coverage gate's ("the best per-position skew, chosen on these rows",
     `docs/gate-power.md` line 967): fit the type multiplier's `k` and the per-type
     coefficients on **each held-out season itself**, not on strictly earlier ones, bounding
     what any walk-forward fit of the same functional form could earn. **Why not a ceiling of
     "perfect knowledge of the true injury cost"**: that ceiling would read the outcome (the
     player's actual game score net of what the report explained), the same disqualification
     rule 8's own worked case gives the realised-team-total ceiling #305 withdrew — outcome,
     not forecast, and answers nothing stage 2 asks. The in-sample multiplier is a forecast
     bound on the same functional form under test, which is what a ceiling in this document
     has meant every other time it has been declared, so this is not presented as a choice
     between candidates: it is the one construction consistent with every other declared
     ceiling in this document.
6. **The statistic.** `experiment.paired_gain` through each harness's own contrast, feeding
   `run_gate`'s `summarise`/`per_season`/`gate` — the same three functions margin's
   `_house_rule` already calls, `run_gate` being the version of that wrapper with the report,
   the width review and the stamp attached.

## The bar, set now

In `experiment.gate`'s order, for each of the three comparisons independently:

- **NOT-RUNNABLE** if no ceiling is present as a value (S6, before anything else is read — the
  branch both harnesses are in *today*, unnamed) or if the season-clustered MDE exceeds the
  declared ceiling (stage 2).
- **ADOPT** the candidate (`own_k`, `usage`, or `type`) only if the t interval on the paired
  MAE gain excludes zero on the positive side **and** the candidate wins every held-out season
  (ties not wins, #335). For spread this replaces `positional` as the shipped estimator of
  `k`; for injury this replaces `retention` with the type-adjusted table.
- **REMOVE** — the incumbent stays and the candidate is named as established worse, not merely
  unproven: the interval excludes zero on the negative side and the candidate loses every
  season.
- **SHOW** otherwise — the incumbent stays, absence of evidence. In each module's own
  vocabulary this and REMOVE both render as "KEEP `<incumbent>`"; the distinction the report
  carries is REMOVE naming the candidate as excluded with evidence, exactly as `gate`'s own
  `Actions` triple already separates them for every other gate in the repo.

## Power before the run (rule 16) — CELLS this ticket would add, not a run

This document does not execute `scripts/rule16_combined_power.py` here — doing so would
produce a number, which the freeze this section is written under does not permit. What
follows is the arithmetic `run_gate`'s own restatement needs to state the CELLS entries a
real run would use, computed by hand from already-published per-season figures the same way
#311's `own_k` worked example above was — not a new measurement.

**Spread (`own_k`), from `docs/player-spread.md`'s per-season table** (worked in full in
#311's section above): `k=5`, between-season `s ≈ 0.00333` (sd of the five per-season gains,
ddof=1), **`m ≈ 205`** — `n` per season in the published table, since spread's within-season
unit (`player_id`) is already one row per cluster at the walk-forward's own grain, unlike
draft's `m=20` and weekly's `m=40`, which are stated pre-registration stand-ins for an unknown
within-season sd. `m ≈ 205` is comfortably above `TIE_MIN_CLUSTERS` (12), so the tie test
reads the bootstrap SE rather than falling back to the sign — the case #357's own two cells
(draft, weekly) do not exercise, since both sit at stated `m` values chosen for illustration
rather than measured ones. **δ = +0.0065** (the observed `own_k` gain), matching how #305's
own MDE pilot used its measured effect as the delta rather than an arbitrary round number.
**Expectation, stated before any run:** with `m` this far above the floor, the tie-aware every-
season half is doing real work rather than degenerating to the sign test the way the draft
cell's `m=20` mostly does — this cell is closer to `#357`'s weekly-blend cell (`m=40`,
combined power 0.1925 against unanimity-alone's 0.378) than to its draft cell (`m=20`,
combined power 0.0324 against 0.136), and a real run is expected to show combined power
*below* unanimity-alone power but not collapsed to a named exemption the way the draft gate's
ADOPT branch is — stated as an expectation to be checked, not assumed.

**Injury type, from `docs/weekly-injury.md`'s per-season table**: `k=3` (2023–2025), gains
+0.0394 / +0.0526 / −0.0182, mean +0.0246 (published: +0.0244), between-season `s ≈ 0.0377`.
**`m = 1` by construction — the declared no-op** (within-season unit is the season column
itself, per #335's table and unchanged by #343). Below `TIE_MIN_CLUSTERS` unconditionally, so
`_disposition` falls back to the sign alone regardless of what a bootstrap would say — **the
tie mechanism contributes zero power loss here**, unlike the two published cells in #357's
table, because there is no bootstrap for it to attenuate. The combined rule for injury type is
therefore just S1's t-interval **and** unanimity-of-sign — closer to the pre-#335 rule's own
shape than either #357 cell, and named here so a future run is not surprised that clustering
`k=3` gives few degrees of freedom (`t_quantile(0.975, df=2) ≈ 4.303`) regardless of the tie
question. **δ = +0.0244** (the observed gain). **At `k=3` a t interval needs an unusually large
t to exclude zero at all** — `docs/weekly-injury.md`'s own published figure, 2.5 se, already
fails S1's t-interval at this df (two-sided `p` at `t=2.5`, `df=2` is well above 0.05), which
is consistent with the module's own frozen verdict (KEEP retention) and is named as **a
rule-16 exemption candidate for injury type's ADOPT branch**: at `k=3` the every-season half
alone requires winning all three seasons, which the type adjustment does not do today (2 of 3),
so ADOPT is unreachable at the currently observed effect independent of any power question —
this is #300's and #357's "named, not silently reachable" logic, applied here because the
published data, not a hypothetical, already shows the branch closed at this sample.

## Exclusions (rule 11)

- **`hub.models.injury.verdict`** (the retention/table/baseline/out_zero argmin comparison) is
  explicitly **not** in scope — it is S3/#360's ticket, a different defect (no gate at all,
  not a hand-rolled one), frozen behind the same #326 freeze but tracked separately.
- **`hub.models.margin`** is excluded because it already reads the shared rule via
  `_house_rule` (`summarise`+`gate`), with its ceiling wired 2026-09-21 (`03d119f`) — #343's
  own ticket body names this as precedent, not as remaining work.
- **The weekly, lineup, draft and coverage gates** are excluded — all four already call
  `run_gate` or `_house_rule` directly and are unaffected by this section.
- **Injury type's within-season unit** stays the declared no-op pending #360; #343 converts
  the *rule* (hand-rolled → `run_gate`) without pre-empting #360's own conversion of the
  *unit*.

## The published numbers the result would move (rule 13)

`docs/player-spread.md` (`own_k`'s and `usage`'s verdict sentences, once a ceiling and stage 2
are wired in — the KEEP verdict itself is not expected to move, since `own_k`'s gain
(+0.0065) sits well under its own ceiling (+0.0852), but the *reported* MDE and the presence
of a stage-2 line are new); `docs/weekly-injury.md`'s type-adjusted verdict (same expectation:
KEEP retention stands, now with a declared ceiling and a stage-2 line it has never printed);
ADR-0019's own claim, per #343's fourth acceptance criterion, gains a dated note that "read by
every gate in the repo" held for six of eight gates until this ticket landed.

## Constants: chosen / fitted

No new constant is fitted by this section. Spread's ceiling (**+0.0852**) is arithmetic
already published in `docs/player-spread.md` (itself `fitted`, per that page's own
provenance). Injury type's ceiling (the in-sample per-type multiplier) is **to be fitted**
at implementation time — declared here by construction, not by value, the same way #305's and
#306's ceiling arms were declared before their numbers existed.

## What this measurement cannot do

- It cannot pick the ceiling, the cluster, or the within-season unit after a sign is seen —
  both are stated above, before any of the three gates run.
- It cannot resolve #360's own scope (injury's within-season unit, or the argmin defect in
  `injury.verdict`).
- It cannot promise a verdict change: the arithmetic above expects both modules' recorded
  KEEP/KEEP to stand, with stage 2 and a ceiling reported for the first time rather than a
  different winner.

## What happens either way

If both gates run and reach the verdicts expected above, the two docs gain a stage-2 line and
a ceiling and no headline changes. If `own_k`'s restated, season-clustered significance (worked
in #311's section: t ≈ 4.4 against the published unclustered 1.8 se) holds under the real
clustered run and its MDE clears the +0.0852 ceiling, `own_k` becomes a live ADOPT candidate
for the first time in this repo's record — which is exactly the kind of number rule 13 says
must not be left standing once the arithmetic above is known, and is named here rather than
left implicit. If injury type's `k=3` keeps it NOT-RUNNABLE-adjacent or SHOW under the named
rule-16 exemption above, `docs/weekly-injury.md` restates that as the reason, not as a fresh
failure.

# Pre-registered 2026-09-21 — PROPOSED: every dollar figure against the rivals' real ledgers
(#317)

**Status: PROPOSED.** Drafted under the #326 freeze. This section uses the template's headings
throughout so it reads beside the others, but names plainly where a heading does not apply:
**#317 corrects an input to a Monte Carlo simulator, not a modelling alternative under
ADR-0019 or ADR-0024** — there is no incumbent model to beat, no candidate axis, and no
ADOPT/REMOVE verdict. What moves is every dollar figure the survivor module prints, restated
once against a corrected field, per rule 13. Parent decisions: rule 7 (pairing — the fix is
read entirely through the paired difference between today's clean-ledger field and a real one,
scored on the identical trials), rule 13 (restatement protocol), and #334 (the pool-rules
ticket this one shares its one human prerequisite with).

## The defect, restated

`Field.entry`'s trial loop (`src/hub/season/pool.py:1237`) builds every trial's rival ledgers
as `[set(ledger)] + [set() for _ in range(entries - 1)]` — our own entry's ledger, and every
rival clean, every trial, regardless of the week. In a week-12 run our entry carries eleven
spent teams over the remaining weeks while all twenty rivals can never be eliminated by the
no-repeat rule and may re-use a team they spent in week 2. Every public question this module
answers through `Field.entry` (`weekly`, `buyback`, `sensitivity`) inherits this; `leverage`
already reads the field honestly through `Field.outcome`'s own `ledgers` parameter, which
exists and is correct — the defect is that `Field.entry`'s trial loop does not use it, not
that the module cannot express real ledgers at all. The data to fill it is already fetched and
unused: `hub.fetch.pool.ledgers` (`src/hub/fetch/pool.py:387`) returns every entry's ledger, in
index order, ours first, in exactly the shape `Field.entry`'s `ledger` parameter takes for one
entry — and it has zero callers in `src/`.

## The estimand, arms and pairing (rules 6, 7) — restated for this module's shape

**The estimand** is the same one every figure in this module already computes — expected
survival share, expected dollar equity, the buyback's net — evaluated against the **observed**
field state rather than an assumed clean one. **There is no second arm to gate against**: this
is not "does a corrected simulator beat the shipped one," it is "the shipped one was computing
the wrong quantity, and the corrected one computes the right one." Read as a paired comparison
anyway, for the one thing rule 7 buys here: **the same trials, the same seed, the same board,
scored once with `ledgers=[clean]*entries` and once with the pool's real ledgers**, so the
restated figures are a paired difference against today's published ones rather than two
independently-noisy runs that happen to disagree.

## Clusters, ceiling, the gate, power (rules 3, 8, 16) — not applicable, stated why

- **Clustering**: N/A. There is no walk-forward across held-out seasons here — one board, one
  week, `trials` Monte Carlo draws of the *same* season's remaining weeks. The repeated-measure
  unit this module already respects is the trial itself (`share_sd`, `given_up_se` are already
  computed across trials, paired, per `pool.py`'s own docstrings), and #317 does not change it.
- **Ceiling arm**: N/A, for the same reason #318's section below gives in full — there is no
  competing arm to bound; the correction is unconditional once #334's prerequisite is met.
- **The Gate, ADOPT/REMOVE/SHOW/NOT-RUNNABLE**: N/A. Nothing here is adopted or removed; the
  restated figures simply replace the superseded ones, per rule 13, the moment the fix lands
  and the real ledger is available.
- **Rule 16 power**: N/A as a gate-power question. The relevant power question is a different
  one and already answered by the ticket's own measurement: at 500 trials on the synthetic
  32-team pilot board, the corrected share (0.0311) and the clean-ledger share (0.0417) are
  well outside each other's Monte Carlo noise at that trial count — the ticket's own acceptance
  criteria do not ask for a resolution check the way #318's do, because the direction (real
  ledgers thin the field less than clean ones assume, so our own survival share falls) is not
  in question; only the magnitude, on the real pool's field, is.

## What is measured — the pilot already run, restated as what a real run would replace

**Measured (already published in the ticket body, restated here for the numbers this section
carries forward), on the synthetic 32-team board `tests/unit/test_pool.py::_board` builds**,
weeks 11–18, our ledger ten deep, 21 entries, 500 trials: share **0.0417** with clean rivals
against **0.0311** with ten-deep rivals — **$17.53** against **$13.06** on a $420 pot, a **25%**
move. This is a pilot on a synthetic board, not the pool this repo actually plays — the real
correction, on the real field, is what #317 measures once it can run.

## Exclusions (rule 11)

- **`Field.outcome`** is excluded — its `ledgers` parameter already threads real state
  correctly; only `Field.entry`'s trial loop (and, transitively, `weekly`, `buyback`,
  `sensitivity`, every caller reached through it) is in scope.
- **`leverage`** is excluded from the restatement's first pass. `docs/pool-leverage.md`
  already reads real ledgers where it can and is a study result about the instrument, not a
  slate artifact (its own page says so); it restates on its own schedule if #317's fix changes
  its inputs, not as part of this ticket.
- **PoolConfig's three unconfirmed rules** (`co_survivor_rule`, `co_elimination_rule`,
  `buyback_cap`) are excluded — that is #333/#334's scope, not #317's, even though both share
  the one human prerequisite below.

## The one human prerequisite, named rather than assumed away

**#317 is `blocked_by` a browser save-as, not by code.** `POOL_URL` and `POOL_SESSION` reach
no environment an agent runs in, and `data/processed/pool_state.json` has never been written
in this checkout (per #317's own 2026-09-18 comment, recorded by AI from the maintainer's
re-reading of the frozen audit). The door is `hub.fetch.pool --payload FILE`, and the payload
is **the same saved page #334 reads for the pool's three rules** — one maintainer action
unblocks both tickets. This section does not pre-register a work-around; it names the
dependency so whoever picks this up after the freeze starts from the estimate the 2026-09-18
comment gives, not the original audit's.

## The published numbers the result would move (rule 13)

Every dollar figure `hub.season.pool` prints through `weekly`, `buyback` and `sensitivity` on
the live pool — none are cited by number in this document today (the pilot's $17.53/$13.06 pair
is the only quoted figures, and both are synthetic-board, not live-pool). Whatever
`docs/track-record.md` or a slate write-up has quoted from a live run since is the restatement
target at implementation time, each once, with the prior value in a restatement box per rule
13's own convention and the ticket's fourth acceptance criterion.

## Constants: chosen / fitted

None. #317 changes an input (which ledgers a trial samples against), not a fitted or chosen
constant in `hub.season.pool`.

## What this measurement cannot do

- It cannot run before #334's page-save lands — not a code dependency, a data one.
- It cannot say how large the real move is: the 25% figure is the synthetic pilot's, offered
  as the mechanism's demonstrated size, not a prediction for the actual pool, whose ledger
  depth, entry count and week differ from the pilot board's.
- It cannot touch `PoolConfig`'s three unconfirmed rules; that is #333/#334's write.

## What happens either way

Once the payload lands, `Field.entry`'s trial loop threads `fetch.pool.ledgers(state)` in
place of the hardcoded clean rivals, every figure the module publishes is restated once
against the real field, and the restatement box names the prior (wrong) figure beside each.
Nothing here is provisional in ADR-0014's sense — this is not a case where no gate can run; it
is a correctness fix whose only blocker is a file that does not yet exist.

# Pre-registered 2026-09-21 — PROPOSED: the buyback carries an interval and a three-state
verdict (#318)

**Status: PROPOSED.** Drafted under the #326 freeze. Like #317, this is not an ADR-0019 gate —
there is no incumbent model and nothing is adopted or removed — but unlike #317 it **does**
introduce a decision rule where none currently exists (buy / stay / unresolved), so this
section maps the template's headings onto that rule where they apply and marks the two that
do not (ceiling, rule-16's script) with the reason, rather than skipping them silently. Parent
decisions: `DECISIVE_SIGMA` (`hub.season.pool`, `chosen(2.0)`, "the evidential bar the repo's
gates are stated at" — the module's own declared constant, reused rather than invented here),
rule 9 (a result too large to believe is a bug — the audit's own framing: "the run line then
prints *the verdict holds at every point* across the concentration axis, attributing to the
assumption a stability that Monte Carlo noise alone destroys").

## The defect, restated

`buyback()` (`src/hub/season/pool.py:1321-1456`) computes `equity = out.share * grown` and
`net = equity - cfg.buyback_fee` from a single `Field.entry` call and recommends on
`net > 0` alone — no standard error, though `EntryOutcome.share_sd` is sitting in `out` unread.
Every other reader of this module refuses to call a difference inside its own resolution:
`weekly`'s recommendation is gated on `DECISIVE_SIGMA` standard errors of the *paired*
trial-by-trial difference (`pool.py:1743`), and `sensitivity`'s own rows carry
`se = entry.share_sd / sqrt(trials) * stake` and an `unpaired_bar` (`pool.py:1901`, the
identical construction this ticket asks `buyback` to share). `buyback` alone has none. The
ticket's own eight-seed pilot at the CLI's default trial count: equity from **$17.90** to
**$30.16**, net from **−$2.10** to **+$10.16**, **four BUY and four stay** — on the *same*
board, ledger and field, differing only in seed — against the module's own bar on that equity,
about **±$9.74**. The sign of the recommendation is not stable at the resolution the module's
own trial count provides, and today's `buyback` reports one seed's sign as a fact.

## The estimand, and what "arm" means here

**The estimand is `net`** — the dollar value of buying back, `equity − fee`, at a stated field
concentration and a stated rival-buyback assumption (both already named as such in `buyback`'s
own docstring: "one point on the field-concentration axis," "an argument, not a model"). There
is **one simulated arm**, not two: staying out is not simulated, it is `net = 0` by
definition, so this is a one-sample question — is `net` resolvably different from zero at this
trial count — rather than a paired two-arm comparison the way `weekly`'s recommendation is.
That is the one place this ticket's shape genuinely differs from `weekly`'s, despite the
ticket asking for a standard error "built the way the weekly figure's is": the *construction*
(`share_sd / sqrt(trials)`, scaled to dollars) is shared; the *comparison* (one-sample against
zero, not paired against a second simulated arm) is not, because there is nothing to pair
against.

## What is measured

1. **`se_net = grown * out.share_sd / sqrt(trials)`** — the same scaling `sensitivity` already
   applies to `entry.share_sd` (`pool.py:1901`), read off the same `Field.entry` call
   `buyback` already makes, at the pot the buyback has already grown to (`grown`, not the
   pre-buyback `pot`, since that is the dollar figure `equity` and `net` are stated in).
2. **The three-state verdict**, in place of `net > 0`:
   - **BUY** if `net - DECISIVE_SIGMA * se_net > 0` — net is positive and resolvably so.
   - **STAY** if `net + DECISIVE_SIGMA * se_net < 0` — net is negative and resolvably so.
   - **UNRESOLVED** otherwise — zero sits inside the `DECISIVE_SIGMA`-wide band around `net`,
     and the trial count cannot tell buy from stay apart at this concentration and this
     assumption about rival re-entries.
3. **The eight-seed check, already named by the ticket's own acceptance criteria as the power
   question this ticket asks — not `scripts/rule16_combined_power.py`'s machinery, which does
   not apply (below).** Re-run the same board, ledger, field and concentration at eight seeds,
   the CLI's default trial count: either all eight verdicts agree, or every one of them reads
   UNRESOLVED. A run where some seeds say BUY, some say STAY, and none say UNRESOLVED is the
   defect this ticket exists to close, restated as a passing/failing check rather than a
   one-time pilot finding.
4. **The run line's "holds at every point,"** printed across the field-concentration
   sensitivity axis, is conditioned on the verdict being resolved at every point on that axis —
   the ticket's fourth acceptance criterion, and rule 9's own instruction applied directly: a
   claim of stability that Monte Carlo noise alone could produce is the "result too large to
   believe" this rule exists to catch, restated here as a precondition on the sentence rather
   than a fact the sentence asserts by default.

## Ceiling, the Gate, rule 16 — not applicable, stated why

- **Ceiling arm: N/A.** A ceiling in this document's sense bounds what a *forecasting* input
  could earn against sampling noise in an *outcome* (rule 8's own construction, declared for
  every gate above). `buyback`'s question is not "does this beat an incumbent" but "can this
  trial count tell a computed dollar figure from zero" — there is no forecast to bound and no
  perfect-information arm that means anything here; the analogous quantity is the trial
  count's own resolution, which item 1 above computes directly rather than bounding.
- **ADOPT/REMOVE/SHOW/NOT-RUNNABLE: N/A as spelled.** The three-state rule above (BUY/STAY/
  UNRESOLVED) is this ticket's own version of the same shape — a decision with a positive
  branch, a negative branch and an explicit "cannot tell" branch that must be named rather
  than silently defaulted, which is exactly what NOT-RUNNABLE is for every other gate in this
  document. UNRESOLVED **is** this ticket's NOT-RUNNABLE, stated in the module's own
  vocabulary rather than borrowed wholesale from a walk-forward gate that has seasons to
  cluster on and this one does not.
- **`scripts/rule16_combined_power.py`: N/A.** That script's machinery (`SEASON_CLUSTER`,
  `per_season`, the tie-aware every-season half) is built for a walk-forward gate with
  multiple held-out seasons; `buyback` has one board, one week, one decision, and no season
  axis to cluster on. The power question rule 16 asks — can this design resolve a real effect
  before it runs — is answered directly by item 3 above: the eight-seed agreement check *is*
  this ticket's rule-16 equivalent, pre-registered in the ticket's own acceptance criteria
  before this section was written, which is what led this document to name it rather than
  invent a second mechanism.

## Exclusions (rule 11)

- **`weekly`'s own resolution machinery** is excluded — already correct, already the pattern
  item 1 above borrows, not itself part of this ticket.
- **`sensitivity`** is excluded from the code change — its own `se`/`unpaired_bar` pair is
  already the two-arm version of what `buyback` is getting the one-arm version of — but its
  **reporting** is in scope for item 4: the run line the ticket names is printed alongside
  `sensitivity`'s own rows.
- **`PoolConfig`'s field-concentration axis and rival-buyback assumption** are excluded from
  this ticket's decision — both stay stated assumptions the caller supplies, exactly as
  `buyback`'s own docstring already frames them; #318 does not turn either into a fitted or
  measured quantity.

## The published numbers the result would move (rule 13)

None are cited by number in this document today; the eight-seed pilot ($17.90–$30.16 equity,
−$2.10 to +$10.16 net, four/four) is the ticket's own and is not a `docs/` publication. The
restatement obligation is forward-looking: whatever a live buyback decision has printed since
is restated once, in three-state form, at implementation time — #318's own comment names it as
exempt from the #326 freeze the moment "a live buyback decision arrives before Audit V," which
this section does not pre-empt.

## Constants: chosen / fitted

`DECISIVE_SIGMA = chosen(2.0)` is reused, not refit — the module's own existing declared
constant, "the evidential bar the repo's gates are stated at." No new constant is introduced.

## What this measurement cannot do

- It cannot make the sign resolvable at the CLI's current default trial count if the true net
  sits inside the `±$9.74` band the pilot already measured; UNRESOLVED is a legitimate outcome
  of this fix, not a failure of it, and raising the trial count to resolve it is a separate,
  later decision (the same shape `LEVERAGE_STUDY_TRIALS` was for the leverage term).
- It cannot decide the field-concentration or rival-buyback assumptions; both remain the
  caller's stated inputs.

## What happens either way

If the eight-seed check agrees, `buyback` prints BUY or STAY with its interval and the run
line's "holds at every point" claim is earned rather than assumed. If seeds disagree or the
band swallows zero, every affected point reports UNRESOLVED, the run line is restated to name
which points on the axis resolve and which do not, and a live buyback decision made under an
UNRESOLVED verdict is recorded as exactly that rather than silently rounded to a sign.
