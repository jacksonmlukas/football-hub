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
> > **Restated 2026-10-06 (#381, rule 13; the verdict above is kept, not edited).** *Prior value:
> > SHOW (won 0, tied 1, lost 3 of 4).* **Now: REMOVE on 3 resolved of 4, 1 abstained.** The same
> > recipe re-run through `gate` under (C) (`uv run python -m hub.season.weekly_gate --run
> > --ceiling`; the per-season table is the same to the digit: 2022 −1.120, 2023 −1.448, 2024
> > −0.259 an Abstention, 2025 −1.761; t interval [−2.032, −0.262], excluding zero). Under (A) the
> > one Abstention vetoed REMOVE; under (C) every resolved season is a loss, the pooled interval
> > excludes zero, and the verdict sentence says three of the four resolved. Nothing about the
> > design moved: MDE +1.119 against a ceiling of +10.873 (9.7x), as above. The number that
> > changed is the reading of 2024, whose |−0.259| sits below every season's 2·SE (the narrowest
> > is 0.82): an Abstention about the effect, not the season (ADR-0019, 2026-10-06). This is the
> > ceiling-recipe run, not the published gate's `--shrink mae-market --drafts 40`, so
> > `docs/weekly-blend-gate.md`'s own REMOVE is unchanged. Ledger entry: `state/gate-width.json`,
> > `weekly`, 2026-10-06, `resolved: 3, abstained: 1`.
>
> `state/gate-width.json` is restored after this run the same way #376's was:
> `git checkout -- state/gate-width.json` once the numbers above were recorded, so #362's
> evidence is untouched by this ticket either.
>
> ADR-0019's rule-16 weekly row is unstruck with this: the row was computed at k=4 all along
> and the number that briefly contradicted it was reading a different, accidental k.

> **The draft gate's ceiling, measured 2026-09-21 -- #376's second, owed half.** A measurement,
> not a modelling change: it moves no model, no constant, no published effect.
>
>     uv run python -m hub.draft.backtest --seasons 2022,2023,2024,2025 --drafts 20 --seed 0 --ceiling
>
> the module's own documented recipe (its docstring's example command, plus `--ceiling`) at the
> default `n_drafts=20`. `--shrink` does not exist on this gate -- that flag belongs to
> `weekly_gate`, not `backtest`, so there is no published-recipe choice to make here the way
> there was for the weekly half:
>
> | | this run |
> |---|---|
> | optimizer − market | −12.40 |
> | 95% CI, percentile | [−17.60, −7.20] |
> | 95% CI, t | [−21.11, −3.69] |
> | clusters | 4 |
> | season-clustered MDE (80% power) | 11.01 |
> | **ceiling (perfect foresight)** | **+21.90** |
> | **ceiling / MDE** | **≈1.99x** |
> | verdict, this run | SHOW (tie-aware rule: won 0, tied 1, lost 3 of 4 seasons) |
>
> **Stage 2 passes, but by a much thinner margin than the weekly half.** The ceiling (21.90)
> clears the MDE (11.01) by about 1.99x -- against the weekly gate's 9.7x above. `experiment.gate`'s
> NOT-RUNNABLE branch does not fire for the draft gate either: both season gates ADR-0019's
> #363 (S6) amendment named NOT-RUNNABLE pending this ticket now carry a measured ceiling, and
> neither MDE exceeds it. That is what #376 asks this ticket to end for the draft half, the same
> way the note above ended it for the weekly half.
>
> > **Restated 2026-10-06 (#381, rule 13; the table above is kept, not edited).** *Prior value:
> > SHOW (won 0, tied 1, lost 3 of 4 seasons).* **Now: REMOVE on 3 resolved of 4, 1 abstained.**
> > Under (C) the one Abstention (2024) no longer vetoes REMOVE: the three resolved seasons are
> > losses and the pooled interval excludes zero. The same recipe re-run today
> > (`hub.draft.backtest --seasons 2022,2023,2024,2025 --drafts 20 --seed 0 --ceiling`, about 45
> > minutes) **is not a bit-for-bit re-reading of the run above, and the difference is flagged
> > rather than argued away**: 2022 −15.35, 2023 −8.41 and 2024 −5.99 are identical, 2025 is
> > **−14.34 where this run recorded −19.86**, so the mean is **−11.02 [−14.84, −7.20]** (percentile;
> > t [−17.22, −4.82]) against −12.40 [−17.60, −7.20] (t [−21.11, −3.69]), and the season-clustered
> > MDE is **7.84 against 11.01** (ceiling +21.90 either way, so ceiling / MDE is 2.8x against
> > 1.99x). Something between 2026-09-21 and today moved 2025's replay (the run's own join-failure
> > line reads optimizer 0.6% of 1,120 drafted names); this restatement does not attribute it and
> > reads the verdict off today's run. On #376's own recorded numbers (the table above) the (C)
> > reading is REMOVE as well (three losses, one Abstention, t interval excluding zero). The run's per-season
> > SEs, the first the draft gate has recorded: 3.31 / 3.11 / **3.75** / 2.03 (m = 20 each);
> > 2024's is the widest, which ADR-0019's 2026-10-06 amendment reads for the 2024 sub-question.
> > Ledger entry: `state/gate-width.json`, `draft`, 2026-10-06, `resolved: 3, abstained: 1`.
>
> > **The draft re-run, attributed (2026-10-06, #429, rule 13; the box above is kept as written).**
> > *Prior value: "does not attribute it".* **Attributed to inputs the run did not record, not to
> > code, seeds or ordering; the run that stands is today's** (−11.02, t [−17.22, −4.82], MDE
> > 7.84, REMOVE on 3 resolved of 4, 1 abstained), **and the verdict word does not change.**
> > Evidence, in the order it was gathered:
> >
> > 1. **Code is exonerated by a bisect.** #376's own commit (`8418380`) checked out in a temporary
> >    worktree and run on today's `data/` with the same recipe (`--seasons 2022,2023,2024,2025
> >    --drafts 20 --seed 0 --ceiling`) returns **−11.02, MDE 7.84, ceiling +21.90, t [−17.22,
> >    −4.82]**, the same to the cent as the tree at the tip of #381. So nothing between 2026-09-21
> >    and today (#384, #311's sort, #343, #392/#393, #398's nflverse seam, the Ledger) moved this
> >    number, and neither did the hold-out set (neither recipe passes `--holdout`; the 11.10 of the
> >    2026-09-16 hold-out run is a different recipe, #376's own diagnosis).
> > 2. **Seeds and worker order are exonerated.** The gate is a single process at `--seed 0`, and
> >    two different trees on one data directory agree to the cent; a third replay (the tip, with the
> >    new input record below) agrees again. Three replays, one answer.
> > 3. **The difference is in the inputs, and only 2025's.** 2022, 2023 and 2024 are identical to
> >    the cent between #376's record and today (−15.35 / −8.41 / −5.99) and so is the foresight
> >    ceiling (+21.90), which reads the realised seasons; only 2025's *optimizer − market* moved
> >    (−19.86 → −14.34). That fits an input to the 2025 board or season simulation, not to
> >    realised outcomes.
> > 4. **The inputs cannot be recovered, and the reason is the finding.** #376 ran in an agent
> >    worktree with its own copy of `data/`, recorded no digest (its ledger write was reverted),
> >    and the draft gate's `data_digest` is `unpinned` in every run. The 36 reads of today's run,
> >    now recorded one by one (below), include **12 `player_stats` cache entries with no pin
> >    sidecar** (written 2026-08-23 to 08-29, unchanged since; they are the same files on the
> >    primary checkout before and after 2026-09-21): bytes this run cannot name, and `player_stats`
> >    is the source nflverse revises in place (`nflverse._write_by_week`). One unpinned read turns
> >    the whole `data_digest` into `unpinned`, so the digest could not discriminate between #376's
> >    data and today's. **Which of those twelve differed in #376's worktree is not knowable now.**
> >    This is attribution by elimination plus a named class of unrecorded input, not a named
> >    changed file, and it is stated as that.
> >
> > **What changed so the next replay can be checked.** `WidthEntry` carries `inputs`: every source
> > the run read as `{source, as_of, digest}`, `unpinned` where a read has no pin
> > (`state/gate-width.json`, `state/README.md`; tests in `test_gate_run.py` and `test_ledger.py`).
> > Today's draft entry records them: 36 reads, 12 of them the unpinned `player_stats` entries,
> > the rest pinned (`ff_rankings` as-of 2021-2025, `ff_opportunity`, `schedules`, four pinned
> > `player_stats`). A later replay that moves will show, source by source, which pin moved or
> > that an unpinned one is still in the set. No rule, constant or verdict changes.
> > What stands where it was published: the REMOVE restated in ADR-0009, `README.md` row 1,
> > `docs/method.md` row 6 and `docs/audits/README.md` rests on today's run, which is the
> > reproducible one; #376's −12.40 / 11.01 stays as the record of what that run printed.
>
> **This run's own verdict is not the published one and is not read as such** -- the same
> caveat the weekly half carried. `n_drafts=20`, no `--shrink` (there is none to give): this
> run's per-season gains (2022 −15.35, 2023 −8.41, 2024 −5.99, 2025 −19.86) read **SHOW** under
> the post-S1 tie-aware rule (won 0, tied 1, lost 3 of 4) because 2024's small loss reads as a
> tie rather than a fourth loss, not the **REMOVE**
> [ADR-0009](adr/0009-championship-equity-does-not-pick.md) recorded (championship equity is
> already out of the draft-night output, in `hub.exhibits`). This run's own point effect
> (−12.40) also does not resolve the attribution question this document's own 2026-09-07
> re-measurement section opened (−19.66 published, −12.48 / −11.59 re-run, #190 still open on
> which commit owns the movement) -- it is one more data point in that range, not a fourth
> measurement offered to settle it. Neither the ADR-0009 disposition nor #190 moves.
>
> **Corrected the same day.** "Neither moves" was half right. The *disposition* — championship
> equity does not pick — does not move; it is a decision, and ADR-0014 keeps decisions and
> their evidence apart. The *verdict word* does: a measured SHOW against a published REMOVE is
> rule 13's exact case, and #167's escape (inputs gone, no re-run possible) does not apply to a
> run that just happened. `README.md`'s first row, `docs/method.md`'s row 6 and ADR-0009 carry
> the dated restatement — REMOVE under the superseded rule, SHOW under the tie-aware one at won
> 0, tied 1, lost 3 of 4 — and say which of the two stands on what. This is the first published
> removal to weaken under the fixed gate; the record of removals was always going to get
> weaker, not different, and the flagship row going first is what the most seasons and the
> most attention predict.
>
> **Runnable is not the same as able to adopt.** Passing stage 2 means this design has the
> power to resolve a real effect of about this size at 80% power; it says nothing about which
> branch it could resolve one *into*. ADR-0019's rule-16 section separately measured the draft
> gate's ADOPT branch at 0.0324 combined power against δ=2.0 -- against 0.136 for the sign test
> alone -- and named it a rule-16 exemption: reachable in principle, not at any effect size this
> project would plausibly observe. Both facts hold together and neither substitutes for the
> other: the stage-2 guard is now satisfied (MDE below ceiling, ~1.99x), and the ADOPT branch it
> guards remains practically unreachable by power regardless. See that section's own dated
> addition for the combined statement.
>
> **Restated 2026-10-06 (#418, rule 13):** the 0.0324 above is reproduced to the digit by
> `scripts/rule16_combined_power.py` today and stands as the combined rule's power under that
> script's stand-in within-season spread. At the draft gate's own recorded precision the same
> harness gives **0.0075** (null 0.0014). The exemption stands, more strongly; see *What contradicts a published figure* (#388's results section, attributed under #418).
>
> Flagged in passing, not chased down here (out of #376's scope): this run also printed
> `REQUIRES REVIEW` -- its season-clustered interval (width 10.40) is narrower than the
> previously recorded one (11.10, ratio 0.94), the shape #169 found unnoticed in five of nine
> cases. Worth a look before this table is treated as settled.
>
> **Chased down 2026-09-21 (diagnosis lane, this flag): explained by two things, neither a
> bug.** The 11.10 entry (`lo -18.761, hi -7.665`, carried unchanged into
> `state/gate-width.2026-09-20.pre-ceiling.json`) is the record `7e6625f` (2026-09-16)
> committed for [ADR-0009](adr/0009-championship-equity-does-not-pick.md)'s #290 hold-out
> run at `9855979` -- `--holdout --workers 4`, seed 0, 20 drafts x 4 seasons -- which
> replays every season under `conf/holdout/<season>.json`, refitting `WEEKLY_K` and
> `TEAMMATE_RHO` leave-one-season-out. Tonight's #376 run carried `--ceiling` and no
> `--holdout`, so `compare()` played every season on the shipped constants instead --
> **the recipes do not match.** `--ceiling` itself never touches the interval:
> `default_gate_mode` calls `ceiling()` only to hand `run_gate` a bound carried beside the
> summary (`experiment.summarise`'s `carried = {} if ceiling is None else {"ceiling":
> ceiling}`) -- the bootstrap that produces `mean`/`lo`/`hi`/`width` runs only over
> `compare()`'s `paired` frame, and that call reads `holdout=a.holdout`, `False` on
> tonight's command line. This is also the direction ADR-0009's own 2026-09-16 restatement
> already named: in-sample constants (no hold-out, tonight's run) flatter the losing arm and
> pull the mean toward zero, and the season variance a refit otherwise absorbs stays in the
> pooled estimate, narrowing the interval.
>
> **A second, independent thing also moved, and it is not gated by `--holdout` at all.**
> Diffing every `fitted`/`chosen` constant `hub.declare` covers on the draft path
> (`TALENT_CV`, `WEEKLY_K`, `WEEKLY_SKEW`, `MARGIN_SD`, `TEAMMATE_RHO`,
> `PICK_NOISE_INTERCEPT`, `PICK_NOISE_SLOPE`) between `9855979` and `main` turns up nothing
> -- except `IMPUTE_CV`/`IMPUTE_CV_BY_POS` (`src/hub/models/predict.py`), which commit
> `02561b7` moved **32 minutes after** the hold-out run, at 2026-09-16 18:08: pooled
> 0.260 -> 0.315 (QB 0.217 -> 0.315, RB 0.334 -> 0.359, WR 0.223 -> 0.288, TE 0.220 -> 0.315;
> #298, the rookie-population remeasurement ADR-0009's own #290 restatement flags as still
> open at the time of that run). `hub.holdout`'s `HELD_OUT` set never overrides this
> constant either way -- the sets record it as "not refitted", so `compare()` reads whatever
> `IMPUTE_CV` is on the checked-out tree regardless of `--holdout` -- so this move applies to
> tonight's run and would apply to a `--holdout` re-run alike; it is not explained by the
> flag above. More rookie-imputation noise on every skill position but RB is the kind of
> input that can move a mean and a width either way; it is named here as a mover, not
> quantified further (that would be a re-run, out of scope). No other declared draft-path
> constant, the #197 frozen-Board test pin, or `board.py`/`backtest.py`'s own `fitted`/
> `chosen` lines moved between the two commits. One more source of drift neither run controls
> for and this diagnosis does not chase, because it predates both entries and CLAUDE.md's
> rule 1 forbids reading it to check: the live Board each run builds from `data/` (ADP,
> consensus, ffopportunity) is rebuilt at run time and moves with nflverse backfills between
> any two dates regardless of code -- ADR-0009 says so explicitly for its own re-runs.
>
> **So: recipe mismatch (`--holdout` on vs. off) plus one constant that moved for an
> unrelated, already-tracked reason (#298), not a defect in the gate or the width ledger.**
> No bug filed, no constant changed by this diagnosis, no re-run performed.
>
> **Restated 2026-10-08 (#320, option B; the note above kept).** Two things in it are now
> different. (1) "`hub.holdout`'s `HELD_OUT` set never overrides this constant either way" is
> no longer true: the four sets bind `IMPUTE_CV` and `IMPUTE_CV_BY_POS`, so a `--holdout`
> replay no longer reads the checked-out tree's rookie measurement. (2) The ledger's 2026-09-16
> hold-out run (`--holdout --workers 4`, seed 0, 20 drafts x 4 seasons) is **not** re-run by
> this change, but its recipe was re-measured on today's code and board `fa974b0d`, same flags,
> **with the old sets** (both imputation keys declined, so shipped 0.315 etc.) and **with the
> new**; only the sets differ: optimizer - market **-11.90 -> -10.19** points per team game;
> 95% t CI **[-18.82, -4.97] -> [-16.71, -3.67]**, percentile [-16.24, -7.55] -> [-14.21,
> -6.17]; interval width **8.69 -> 8.04**; MDE 8.76 -> 8.25; per season (2022-25) -16.25 /
> -8.42 / -6.68 / -16.23 -> -14.77 / -4.88 / -7.46 / -13.64. The sign and the verdict class do
> not move (the optimizer loses in the pooled estimate and in every season by point estimate;
> neither run carried `--ceiling`, so both print NOT RUNNABLE under ADR-0019's #363). The
> binding takes the replay about 1.7 points per team game toward zero and narrows the
> interval by 0.65, consistent with 2022 and 2023 replaying at 0.187 and 0.248 rather than
> 0.315 (less imputation noise in those seasons' sims); a single seed does not isolate the
> mechanism. Both runs are single seed-0 draws, 1.7 is about three of #194's 0.5-point
> run-to-run standard errors, and each run's own write to `state/gate-width.json` was
> reverted. The old-sets run executed on the tree before `main`'s #405 merge and the new-sets
> run's workers had already imported that same code when the merge landed, so the ledger
> prints "other arm source" between them; the board digest `fa974b0d` is identical.
> Why the 2026-09-21 diagnosis missed the mechanism: it read `championship_equity` as an
> exhibit, but `win_probability` is arm B's scorer and hands each row's `xfp_imputed` flag to
> `talent_cv_for`; the `backtest.LIMITATIONS` sentence saying otherwise is corrected.
>
> `state/gate-width.json` is restored after this run the same way both weekly runs were:
> `git checkout -- state/gate-width.json` once the numbers above were recorded, so #362's
> evidence -- the `requires_review` entries the live file already carries -- stays exactly what
> it was before tonight. The run's own write, reverted, was `draft`, clusters 4, width
> **10.403535714285717**, verdict SHOW, timestamp 2026-09-21T18:24:02+00:00.
> `state/gate-width.2026-09-20.pre-ceiling.json` is the byte-identical pre-run copy, unchanged,
> predating both halves of this ticket.
>
> Both halves of #376 are now recorded -- weekly above, draft here.

> **And the guard is fixed, not just explained (2026-09-21).** The ledger had recorded both
> digests since #362 and `review_width` matched on the gate's name alone -- the data needed to
> make the comparison meaningful was in the row and the check did not read it, the same shape
> as `two_sided_p` eleven lines from the verdict that ignored it. `review_width` now compares
> only against a previous entry at an identical `config_digest` **and** `data_digest`; an
> earlier run at another digest, or a pre-#362 entry with none, is counted and named, never
> compared. Positive control (rule 18): two runs at different digests must not compare
> (`tests/unit/test_experiment.py::test_two_runs_at_different_digests_are_not_compared`), and
> the same digest still must. Registered in the rule-18 contract as an explicit guard.


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

> **Amended 2026-10-08 (#320, option B adopted; prior text above kept).** The `IMPUTE_CV` /
> `IMPUTE_CV_BY_POS` row of the table is no longer true: the four sets now **refit both**, on
> the seasons strictly before the replayed one (rule 2), and `--holdout` reads them. Values
> (pooled / RB / WR; QB and TE at pooled in every set, as fewer than 20 rookies cannot split):
> 2022 0.187 (fit on 2021 alone, one cluster), 2023 0.248 (2021-22), 2024 0.287 / 0.287 /
> 0.273 (2021-23), 2025 0.317 / 0.379 / 0.279 (2021-24), against the shipped 0.315 /
> 0.359 / 0.288. The reason the sets carried for declining ("fewer clusters than the decision
> was taken on") applied to every leave-one-season-out fit by construction and is dropped.
> The row's other claim, that the constant "is not a simulator constant", was also wrong --
> see the restatement beside the 2026-09-21 note. So a hold-out replay now moves the weekly
> law, the teammate correlation **and the rookie imputation error**. Full table and method:
> [impute-cv.md](impute-cv.md).

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

> **Amended 2026-09-21, at adoption: the caller-walk test gets its own positive control
> before it is trusted.** A guard that walks a call graph is rule 18's shape — an assertion
> that can pass because it cannot see. An AST walk is blind to dynamic dispatch, `getattr`,
> and config-driven paths, and the test says so in its docstring. Before the guard is relied
> on, a path from a market input into a market-scored object is planted and the test is seen
> to catch it; the test is registered in `tests/contracts/test_every_check_has_a_positive_control.py`
> beside the other guards, so a renamed or narrowed extractor fails there.

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

## Landed 2026-10-06 (#311): option (a), and what the restatement found

**Adopted 2026-09-22 as option (a); landed 2026-10-06. This box is dated and the section above is
not edited.** `paired_gain` takes `cluster` with no default; `mean` is the mean of the cluster
means, `se` the percentile bootstrap's over them, `t = mean / se`. The restatement order above
ran as listed, with these results:

| step | what was restated | result |
|---|---|---|
| 2 | weekly rebuild vs flat, `docs/weekly-projection.md` (and the arms it published) | `--fit` panel: +0.0646, **t = +5.00 on 3 df (p 0.015)**, won 4 of 4 (published +5.3 se). The 2026-09-19 comment's t = 2.66 (p 0.076, closed form) is on the **unfiltered** Panel, where the bootstrap t is +3.10 (p 0.053) and 2024 ties; on the `--fit` panel the page publishes, the figure clears. The panel is the open question, recorded in the restatement box and not decided there. |
| 3 | spread, `docs/player-spread.md` | done under #343, which routes it through `run_gate`: `own_k` **t = +4.93 on 4 df** (this section's hand arithmetic, closed form, said ≈ 4.4 against the published 1.8 se: the direction and the side of 2 are as pre-registered; the bootstrap reads 12% higher at k = 5), runnable (MDE +0.0048 < +0.0852), and **tied in all five seasons** |
| 4 | injury type and retention, `docs/weekly-injury.md`, method.md, where-to-look-next | type adjustment: t = +1.69 on 2 df (p 0.23), SHOW (#343). **Retention vs out_zero, the adopted result: t = +4.19 on 2 df, p = 0.053** — the interval contains zero by a hair; the decision is #360's *(Restated 2026-10-07, #361: that baseline was a within-season lookahead. Strictly-prior, retention vs out_zero is **+0.0623, se 0.0588, t = +1.06 on 2 df, p = 0.40**, interval [−0.1909, +0.3155], won 2 of 3 with 2024 at −0.0529; usable designated weeks 4,939 → 2,581, about 0.011 of the drop the lookahead itself. Prior value kept above; "the adopted result" is no longer t = +4.19, and the decision is still #360's and the maintainer's.)* |
| 5, 6 | `docs/improvements.md` ×2 (not `paired_gain` statistics: the frozen weekly gate), `docs/weekly-shrinkage.md` | improvements: not moved, read not re-run. Shrinkage: rebuilt, −0.0061 at t = −0.92 on 3 df |
| 7 | #302's quarterback log-loss diagnostic | re-run: NOT-RUNNABLE, 0 of 5 seasons archived; reads `run_gate`, cannot move |

**Two things the pre-registration did not anticipate.** (1) At k = 4 the bootstrap t overstates
the closed-form t by about `sqrt(k/(k-1))` = 15% (a percentile bootstrap of four units has the
variance of the mean of four draws, not the unbiased one): +5.00 against +4.29 on the same four
gains. Option (a) was adopted knowing the two "can disagree enough to matter" below
`SMALL_CLUSTERS`; this is the size of it at k = 4, and it is why the restatement boxes print the
p for the t on k − 1 df rather than the t alone. (2) `_cluster_se` read its units unsorted (the
issue-#45 order defect `summarise` and `per_season` already fixed); it is sorted now, which moves
a season's tie-disposition SE at the noise floor and nothing else.

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

> **Adopted and landed 2026-10-07 (#312), as drafted; the section above is kept as written.**
> Three places where the run read differently from the draft, each stated rather than smoothed:
>
> * **`td_rate_prior` was not a flip.** The draft read it as `NULL_BROKEN` under the flat bar
>   from the ticket's own figures. On the settled basis the sign is 4/5 seasons, so the
>   every-season half had already classed it `clears` (*noisy, not a signal*); the p bar keeps
>   the status and changes the note to *null as pre-stated (p 0.071)*. The statuses that do
>   move are `inj_sev` in the joint screen (clears → killed, −2.70 se, p 0.054, every anchor),
>   `snap_trend` at anchors 4 and 6 on the two superseded bases, and two Usage cells. The
>   ticket's p 0.089 is a `t` of 2.49 on three degrees of freedom; five seasons give 0.068.
> * **The joint size is about 0.02** (0.017–0.0225 over four features, 2000 permutation draws),
>   from the same draws as `every_season_null`'s other figures, and the two halves are
>   positively dependent (their product is 0.0016). Under the old bar it was 0.025–0.034.
> * **The family counts distinct tests.** 120 printed rows across anchors and bases are 48
>   tests, since an anchor-invariant feature is one test printed five times. Threshold 0.0361,
>   18 below it. `weekly-screen.md`, "Restated 2026-10-07", carries every figure.

> **Dated 2026-10-07: the family definition is a choice made after the draft (#312); the draft
> text above is kept.** The draft's third measurement says the family should count "everything
> actually run in one report". The implementation counts something narrower in two ways, and
> neither was pre-registered:
>
> * **Alone tests only.** The joint-screen rows (survivors re-screened with the others as
>   controls) are not in the run family. They re-test features already in it, so adding them is
>   a different family question (`with_family`'s own: the alone and joint screens ask different
>   questions of different rows), and the per-anchor joint lines still print their own count.
> * **Distinct tests, not printed rows.** A feature whose rows do not depend on the anchor
>   prints the same p at all five, so 120 printed rows are 48 distinct tests. Counting the copies
>   puts five identical p-values into the ranking.
>
> **Direction, stated plainly: the definition lowers m from 120 to 48, which is the liberal
> direction for m.** It is not the liberal choice overall, because the duplicates also occupy
> ranks and BH's step-up then rejects against a ranked list with copies in it. Measured on the
> same run, counting all 120 gives threshold **0.0477** and 75 rows below it (the 18 distinct
> rejections at 48, repeated, plus one more); the 48-test family gives **0.0361** and 18.
> **Counting 120 changes exactly one BH decision**: `snap_trend` on the settled basis at week
> ≥ 10 (p 0.0477, adjusted 0.076 at 48) is rejected at 120 and not at 48. The snap-share trend
> is unlicensed (#233) and the diagnostic decides nothing (#274), so no verdict or licence moves
> either way. The choice was made because it counts hypotheses, not printing.

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

> **Adopted 2026-09-21: (a), and the reason is corrected on the way.** The parenthetical
> above — that (b) is "not rule 2's window" — is wrong and is withdrawn. **Both windows satisfy
> rule 2:** a prior season's full game log lies entirely outside the outcome window, which is
> point-in-time correct, not a leakage risk. The trade is thin-early against stale — a football
> question, not a methodology one. (a) is chosen as **current form over a season-old baseline,
> accepting four thin weeks at the start of a season**. (b) is not disqualified: it stays
> available as a *separate feature* with its own screen, and a reader who finds it excluded on
> leakage grounds has been misled by the sentence this note corrects.

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

**Restated 2026-10-06 (#418, rule 13):** the two figures cited there (0.1925 weekly, 0.0324
draft) are reproduced to the digit by the script today and are power under its stand-in
within-season spread (cluster SD = the between-season `s`). At the gates' own recorded precision
(#388's estimated process) the same rule gives 0.0008 weekly and 0.0075 draft at k=4, so the
comparison cells' reference powers are those, not 0.1925 and 0.0324. This cell (m ≈ 205) was not
re-run under #418; what the note changes is the reference its expectation was set against. see *What contradicts a published figure* (#388's results section, attributed under #418).

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

> **Amended 2026-09-21, at adoption: the *method* is pre-registered, not only the fact.** "To
> be fitted at implementation" left how open, which is a degree of freedom chosen after the
> arm is known. The ceiling arm for injury type is the `hub.models.margin.ceiling` analogue:
> for each held-out season, a per-type retention multiplier fitted **in sample on that
> season's own designated rows** and scored on the same rows — the oracle that knows each
> type's realised multiplier — with the ceiling gain per season being the shipped arm's MAE
> minus the oracle's on those rows, handed to `run_gate` as `Ceiling(<arm name>, diffs)` with
> the arm named as a module constant the way `margin.CEILING_ARM` is. Flattered by
> construction, which is what a ceiling is; no other construction is used.

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

## Landed 2026-10-06 (#343): what the three runs returned against what this section expected

**Adopted 2026-09-22; code landed 2026-10-06; this box is dated and the section above is not
edited.** Three gate runs, all on the day's data, ledger entries in `state/gate-width.json`
(`player_spread` ×2, `injury_type`, `component_calibration`).

| comparison | expected (above) | returned |
|---|---|---|
| spread `own_k` | KEEP; stage-2 line new; "if its restated significance holds and its MDE clears +0.0852, a live ADOPT candidate" | **SHOW.** t = +4.93 on 4 df, MDE +0.0048 < +0.0852: *live*, as flagged — and **tied in all five seasons** (won 0, tied 5, lost 0), which blocks ADOPT. The tie half, not the interval, holds it. |
| spread `usage` | KEEP | **SHOW**; t = −0.07, MDE +0.0434 < +0.0852 |
| injury type | KEEP; "rule-16 exemption for the ADOPT branch at k=3" | **SHOW**; t = +1.69 on 2 df, MDE +0.0974 < the declared in-sample ceiling +0.2641 — runnable, so the null is one the design could have contradicted |
| component calibration (not in this section; added by the 2026-10-02 review) | — | **NOT-RUNNABLE**: MDE +0.170 against a +0.019 ceiling; was *NULL — not taken* |

**Two declarations the section did not make, named.** (1) The component calibration's
ceiling arm was not pre-registered here; it was declared in the #343 commit as the
`injury`/`coverage` construction — each component's calibration fitted in sample on the
held-out season — before any run, and the verdict does not depend on it: no calibration could
earn the +0.170 the design would need. A least-squares line is not MAE-optimal, so the
in-sample arm is a flattering-by-construction bound and not a tight one. (2) The calibration's
within-season unit is `player_id`.

**Rule 16, not run.** The CELLS arithmetic above was not executed as a script; the three
real runs are the measurement. They do not change what the section said about injury type's
ADOPT branch at k=3 (the observed record already closes it: 2 of 3 seasons won).

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

# Pre-registered 2026-09-21 — PROPOSED: the state-space team rating (#365)

**Status: PROPOSED.** Drafted from `docs/audits/2026-09-20-method-audit.json` finding **S8**,
in #305's form: this becomes the rule on the maintainer's `ADOPTED:` comment on #365 and
decides nothing until then, per the precedent #329 set — pre-registered before the module that
would run it exists. Parent decisions: [ADR-0019](adr/0019-a-gate-requires-every-season.md) as
amended by #357 (S1, the t-interval half) and #363 (S6, the ceiling precondition, both
directions), [ADR-0006](adr/0006-fitted-constants-live-with-their-provenance.md) (a fitted
constant lives with its provenance), and
[ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) — *"a measurement
that steers the product must be committed code"* — which `hub.exhibits.championship_equity`'s
own docstring cites for the reason a removed model's harness stays in the tree, re-runnable,
rather than leaving with the model: an exhibit's whole point is that its measurement can be
checked again, and the same requirement is why an *unwired* fit belongs in `hub.exhibits` from
its first commit rather than arriving there only on REMOVE. `hub.declare` (#253:
`fitted`/`chosen`/`not_an_input`, and `covered()`, which walks exactly the declarations below)
is the other load-bearing piece.

**#365 was `blocked_by` #326 on 2026-09-21** for a stated reason — quoting the maintainer's own
comment on #326: *"a Glickman-Stern rating declares fitted constants, `config_digest` is walked
off those declarations, so landing it moves the model version stamped on every published
prediction — for a model nothing reads."* **That reason is withdrawn as of this
pre-registration, not the edge itself** (the edge is the maintainer's write, not this
document's): the design below declares every constant `not_an_input` while the module is
unwired, precisely so the digest does not move, which is what the paragraph immediately below
states in full. What the freeze still holds is the *gate*, and for a different reason — see
"The gate," below.

## The maintainer's framing (2026-09-21), stated here because this document is what a future reader checks it against

The freeze (#326) is about the **serving path**. A rating fitted on 1999–2025 schedules changes
nothing that serves Sunday *as long as nothing imports it*: it lives as an **exhibit** in
`hub.exhibits`, exactly where `championship_equity` and `leverage` already live under
ADR-0007, with its constants declared `not_an_input` so the config digest stamped on every
published prediction does not move, and `ratings.forecaster()` keeps returning
`MarketBaseline()` until the gate below adopts. The real reason to wait on the *gate* — not the
fit — is that the instrument the gate reads changed on 2026-09-21 (S1, the t-interval half; S6,
the ceiling precondition) and no record has been re-run under it yet: this gate runs in October,
under the fixed rule, after Audit V (#325).

**Order: pre-register (this document) → fit as an unwired exhibit (after the maintainer's
`ADOPTED:` line on this section) → gate in October.** Pre-registering first is what makes the
middle step safe. `docs/method.md` rule 1's incident — a rating fitted and *seen* to beat the
close before its gate is written — is exactly what an unwired exhibit with `not_an_input`
constants cannot produce: nothing reads it, so nothing it does can be seen by a published
number before the gate says so.

## The model

A **Glickman–Stern** state-space rating (Glickman & Stern 1998, the standard NFL state-space
form this repo has cited nowhere until now): a Normal observation model on the game margin,
team strengths a Gaussian random walk **across weeks**, a **between-season shrink** pulling
every team's strength toward the league mean at the season boundary, and home advantage a
single scalar parameter. Fitted on `nflverse.load("schedules")` back to 1999 — the same source
and the same `spread_line`/`result` columns `hub.models.margin` already fits against
(`docs/margin-sd.md`), so no new fetch and no new source enters the repo.

    θ[i, w] = θ[i, w-1] + ε[i, w],  ε ~ N(0, WEEK_STEP_SD²)             (within a season)
    θ[i, season+1, week 1] = (1 - SEASON_SHRINK) · θ[i, last] + SEASON_SHRINK · 0 + η
    margin_hat[g] = θ[home, w] - θ[away, w] + HOME_ADV
    result[g] ~ N(margin_hat[g], RATING_MARGIN_SD²)

**Every constant named, and its spelling while the exhibit is unwired:**

| constant | what it is | fitted or chosen | spelling today |
|---|---|---|---|
| `WEEK_STEP_SD` | the random walk's per-week innovation sd — how fast a team's strength can move inside a season | **fitted** (maximum likelihood / Kalman variance estimation on the walk-forward training fold) | `not_an_input`, "an exhibit's own constant, unwired from every prediction and excluded until the gate below adopts" |
| `SEASON_SHRINK` | the fraction of a team's rating that regresses to the league mean between seasons | **fitted** | `not_an_input`, same reason |
| `HOME_ADV` | the single home-advantage points parameter | **fitted** | `not_an_input`, same reason |
| `RATING_MARGIN_SD` | the residual sd converting a strength gap into a win probability, this model's own analogue of `market.MARGIN_SD` — a different number for a different model, not a reuse of the market's | **fitted** | `not_an_input`, same reason |
| `INITIAL_STRENGTH_VAR` | the prior variance on a team's strength at its first-ever observed game (1999 for an original-32/expansion team, its first season for a later expansion team) | **chosen** (a weakly-informative prior width, not a measurement — no data exists to measure a team's variance before its first game) | `not_an_input`, same reason |

**Every one of the five is `covered()` by `hub.declare` the moment it is spelled `fitted` or
`chosen` on landing — that is the whole mechanism #253 built, and it is exactly what must
*not* happen before the gate below has a maintainer's `ADOPTED:` line.** `not_an_input` is not
a weaker claim about these numbers — two of them plainly are measurements — it is the
mechanism's *only* lever for keeping a real measurement out of a digest that must not move yet,
and `hub.declare.declared_in`'s own eight-word minimum on the reason is what stops that from
being silence. `tests/contracts/test_the_exhibit_is_not_a_dependency.py`, extended to this
module alongside `championship_equity` and `leverage`, is what holds nothing in `hub.models` or
`hub.season` importing it while it is spelled this way.

## The estimand

Walk-forward log-loss on the home result, one season held out at a time, the rating fitted on
strictly earlier seasons only (`docs/method.md` rule 2 — `hub.models.experiment.expanding_seasons`
is the one place in `src/` allowed to write the `<` this needs, and this gate calls it rather
than writing its own).

    per game: -[y · log(p_home) + (1-y) · log(1 - p_home)],  y = 1 if home won

`p_home` for the rating is `normal_cdf(margin_hat[g] / RATING_MARGIN_SD)`, the same conversion
form `MarketBaseline` uses on the close — the two arms differ in what produces the margin, not
in how a margin becomes a probability (rule 6, below).

## The arms

* **Arm A, the incumbent: the close.** `normal_cdf(close_spread / MARGIN_SD)` exactly as
  `hub.season.survivor.grid_from_schedule` builds it (`hub/season/survivor.py:578-584`, which
  imports `MARGIN_SD` and `normal_cdf` from `hub.models.market` directly rather than through
  `ratings.forecaster()` — a fact this document returns to under "What published number moves,"
  below, because it means adopting this gate does not, by itself, move that computation).
* **Arm B, under test: the rating.** `normal_cdf(margin_hat[g] / RATING_MARGIN_SD)` from the
  state-space model above, walk-forward.
* **Rule 6 — both arms must have the same information.** The rating sees results and closing
  spreads strictly before the held-out week; the close is, definitionally, the market's own
  number as of the same moment. Neither arm reads anything the other does not have access to as
  of kickoff of the game being scored — in particular the rating does **not** read
  `close_spread` as a feature (that would make Arm B partly Arm A, the laundering rule 6's own
  incident describes), and the close does not read any rating output. Held by
  `tests/unit/test_ratings_gate.py::test_the_rating_arm_never_reads_the_close_as_a_feature`
  (to be written with the fit, named here so the test that must exist is on the record before
  the code is).

## Pairing

**By game** (rule 7) — common random numbers do no work here because neither arm is stochastic
at inference time (both are deterministic functions of their inputs once fitted), so pairing is
just: the same game, scored by both arms, differenced. `diff = logloss(A) - logloss(B)`,
positive when the rating helps.

## Clusters

**Between-season: `SEASON_CLUSTER`** (`("season",)`), the cluster every gate in this repo
resamples on since ADR-0019.

**Within-season: the game.** Stated explicitly because #335's table (`docs/adr/0019-…md`,
"Per-gate within-season units") does not yet carry this gate, and because the answer here is
the opposite of the quarterback gate's: `starter_change`'s within-season unit is a **no-op**
— by choice, `("season",)`, one cluster per season (corrected 2026-09-21, #335 item 2: the
first draft said "one row per event-season", but that frame is one row per event *game*,
48–57 a season, and the tie test was live there). **This gate's frame is not that shape.** A season holds one row per game (~267
for a 32-team, 17-game NFL season, before exclusions), so `within=("game_id",)` is a real
degree of freedom distinct from the season cluster itself, and #335's tie test
(`gain >= 2 * SE` over the within-season clusters, `TIE_MIN_CLUSTERS = 12`) reads real
within-season variation rather than falling back to the bare sign the way `starter_change` and
`injury.type_verdict` do. `tests/contracts/test_gates_tie_test_names_its_within_season_unit.py`
gains this gate's `within` argument alongside the five it already holds.

## The ceiling arm, declared

S6 (#363) makes this mandatory before either half of the bar is read: *"A gate that measured no
ceiling returns NOT-RUNNABLE, full stop… the same way a `void` condition does."* The precedent
this document follows is the quarterback gate's own (`docs/gate-power.md`, "Two preconditions
ahead of the verdict"): *"The ceiling arm is **the betting market's own repricing** — the last
snapshot before the game day, scored against the same frozen line. What the betting market
recovered by the close is the most an adjustment built to anticipate it could recover."*

**For this gate, Arm A already *is* the close** — unlike the quarterback gate, whose incumbent
is a stale pre-change snapshot with the close still ahead of it, there is no later market number
this repo's own current archive holds beyond the close for a game already played. So the
ceiling has to be read season by season, and the two eras of this repo's data are not the same
shape:

* **2026 onward — a real ceiling exists.** `data/processed/lines` (`hub.fetch.odds`, live
  polling) holds more than one snapshot per game, so a genuine "last pre-kickoff poll" distinct
  from whatever `close_spread` resolves to at an earlier `at` is on disk, exactly the shape
  `hub.schedule.priced_games`'s own `live`/`stale`/`schedule` source ladder already
  distinguishes. The ceiling for a 2026-onward season is the gain of scoring against that last
  poll instead of Arm A's close, on this same harness — declared as `RATING_CEILING_ARM =
  "the market's last pre-kickoff poll"`, distinct by name from the quarterback gate's own
  ceiling and the other three gates' declared arms per
  `tests/contracts/test_each_gate_declares_its_ceiling_arm.py`, extended to this one.
* **1999–2025 — no ceiling exists, and that is the finding, not a gap to paper over.**
  `nflverse`'s `spread_line` is, by its own documentation and by `hub.models.margin`'s own
  usage of it, already the historical closing line — there is no later, more-repriced number
  this repo holds underneath it for a game played in 2011. **Per S6, a season with no measured
  ceiling reports NOT-RUNNABLE in both directions, before the pooled interval or the
  every-season half is read at all — the same way a `void` condition does, one rung below it.**
  That is not a caveat on 26 of the walk-forward's 27 possible seasons (2000–2025, one season
  short of the archive's own 1999 start, which the random walk needs as a burn-in year with no
  prior season to walk forward from — the same reasoning `margin.py`'s own
  `FITTED_SHAPE_SEASONS = 27` walk-forward already uses). **It is what the run for those seasons
  actually reports**, and it is reported that way rather than silently scored against a ceiling
  of zero or omitted from the table.

**What this means for the gate as a whole.** A single `experiment.gate()` call reads one
pooled, season-clustered summary — it does not read season-by-season NOT-RUNNABLE. So the
practical shape is: **the walk-forward gate as pre-registered here is NOT-RUNNABLE against
1999–2025 in full**, because the pooled run's ceiling has nothing to average over 26 of its 27
seasons; **a second, smaller run restricted to 2026-onward seasons is the only shape that can
ever produce a ceiling-qualified verdict**, and it inherits ADR-0019's own three-season floor
(below three seasons the every-season half is close to a coin flip) — meaning the earliest this
gate can report anything but NOT-RUNNABLE is **no sooner than the 2028 season**, three years of
this repo's own live poll archive existing at all. This is named here, before the first run,
rather than discovered from watching NOT-RUNNABLE print for two years with no explanation on
the page (rule 16's own incident, restated).

## The gate

`experiment.gate`, under S1's t-interval and S6's ceiling precondition, exactly as every other
gate in the repo reads it — nothing added, nothing loosened:

* **NOT-RUNNABLE** — no ceiling measured (all of 1999–2025 today; see above), or a ceiling
  measured but the season-clustered MDE exceeds it. No verdict; the exhibit stays unwired and
  unfitted-as-shipped; this section records why and what would change it.
* **ADOPT** — the t-interval excludes zero on the positive side **and** the sign holds (win, not
  a tie, per #335) in every held-out season with a ceiling. **What it publishes:** the route
  back into the served path opens — `ratings.forecaster()` is switched from `MarketBaseline()`
  to the rating, behind the flag the ticket's own acceptance criteria name, and the constants
  above are re-spelled `fitted`/`chosen` in the same commit that flips the flag, which is the
  commit that moves `config_digest` and every published prediction's provenance. Track A, named
  in `ratings.py`'s own docstring as the seam this module exists for, arrives here. **What it
  does not publish by itself:** the six `win_prob` consumers the ticket names — see "What
  published number moves," below.
* **REMOVE** — the interval excludes zero on the negative side and the every-season loss is
  unanimous. **What it publishes:** exactly what the ticket's acceptance criteria call
  publishable on this branch — *"you have a calibrated second opinion and a residual to
  study."* The module moves fully into `hub.exhibits`, alongside `championship_equity` and
  `leverage`, re-runnable under ADR-0007, its constants staying `not_an_input` for the reason
  every other exhibit's do (`tests/contracts/test_the_exhibit_is_not_a_dependency.py` covers
  it the same way). Nothing about a REMOVE here is a failure this repo hides — thirteen of
  fifteen prior measurements ended this way (`docs/method.md`, "The record"), and a calibrated
  loser is exactly the "residual to study" the ticket asks a REMOVE to leave behind.
* **SHOW** — otherwise. `ratings.forecaster()` stays `MarketBaseline()`; the flag stays off;
  this section says the comparison could not move it. Absence of evidence, not equivalence.

## Power before the run (rule 16)

No pilot exists for this estimand — unlike the quarterback gate (#291), which had the source's
own two probability columns to read a pilot mean and sd from, nothing in this repo has ever
scored a state-space rating's log-loss against the NFL close. So both the between-season sd and
a plausible effect size are **stated, conservative stand-ins**, by the same convention
`scripts/rule16_combined_power.py` already uses for its own within-season sd:

* **`s` (between-season sd of the log-loss gain), stood in at 0.0332** — the only
  log-loss-against-a-frozen-line pilot sd this repo has ever measured, the quarterback gate's
  own (`docs/gate-power.md`, "Measured 2026-09-13": target 0.0180, between-season sd 0.0332).
  Borrowed rather than invented, on the same order of magnitude a log-loss gate on this repo's
  own spreads is going to produce, and flagged as borrowed rather than presented as measured.
* **`δ`, tested at 0.0090 and 0.0180** — half and the whole of that same pilot's target.
* **`k`, tested at 10 and 26** — a trailing-decade window (the same convention
  `hub.models.margin`'s own `MARGIN_SD` fit uses, "a trailing window beat all-history on
  held-out log-loss," `docs/margin-sd.md`) and the archive's largest fully-realized walk-forward
  count today. `margin.py`'s own `FITTED_SHAPE_SEASONS = 27` names "2000-2026," but 2026 is
  this pre-registration's own in-progress season and is not yet a *completed* held-out season
  to score a log-loss against — so 26 (2000-2025) is what "the archive's own ceiling" means
  here, and 27 becomes reachable only once 2026 closes. **Both k values are moot against the
  ceiling finding above**, since neither window has a measured ceiling under today's archive,
  but computed anyway because the question rule 16 asks — *is ADOPT reachable at any plausible
  δ* — is a property of the combined rule and not of whether a ceiling happens to exist yet.

**Two separate reasons ADOPT is blocked, kept separate rather than conflated into one number.**
The ceiling section above already establishes that **no season in the archive has a measured
ceiling**, which per S6 makes every real run of this gate NOT-RUNNABLE before the pooled
interval or the every-season half is even read — that is a fact about *this archive today*, not
about the combined rule's own statistical power. Rule 16 asks a different, narrower question —
*if* a ceiling existed and did not bind, could the interval-and-tie combined rule ever say
ADOPT at a plausible effect size? — so the run below supplies `experiment.summarise(...,
ceiling=999.0)`, a stated, clearly non-binding placeholder whose only job is to keep S6's
precondition from firing so the combined rule underneath it is what gets measured. Omitting
`ceiling` entirely (as `scripts/rule16_combined_power.py` itself does) would, under the `gate`
this tree ships today, return NOT-RUNNABLE on every trial by that precondition alone — a
correct fact about a gate that never measured a ceiling, but not an answer to rule 16's
question, and conflating the two would have been the very shape rule 17 warns against: a check
whose outcome cannot vary with the thing it is supposed to be testing.

**Run against `hub.models.experiment.summarise`/`per_season`/`gate` directly** (the shipped
harness, not a reimplementation — the same discipline `scripts/rule16_combined_power.py`
follows), at `m = 50` within-season clusters (a tractable stand-in; the real per-season game
count, ~267 for the NFL, only shrinks the within-season SE further and makes ties *rarer*, so
`m = 50` is not an inflated power estimate — see the run's own docstring for the argument in
full), 3,000-trial simulation, `bootstrap = 150`:

| k | δ | combined-rule null size | combined-rule power |
|---|---|---|---|
| 10 | 0.0090 | 0.0000 | **0.0007** |
| 26 | 0.0090 | 0.0000 | 0.0000 |
| 10 | 0.0180 | 0.0000 | **0.0060** |
| 26 | 0.0180 | 0.0000 | 0.0000 |
| 29 | 0.0180 | 0.0000 | 0.0000 |

**Every cell reads ADOPT power under 1%, at k up to the archive's own ceiling and δ up to twice
the only pilot this repo has to scale by.** The reason is structural rather than a matter of
trial count: ADOPT needs a **win in every one of k independently-drawn seasons**, and at
`s = 0.0332` against `δ = 0.0090`–`0.0180`, the between-season noise is 1.8×–3.7× the mean
effect — so a meaningful share of individual seasons draw a losing or tied sign by chance
alone, and the probability that all k do not falls off fast in k. **That last part is the
finding worth naming on its own: power at k=26/29 is not merely small, it is *smaller* than at
k=10 for the same δ** (0.0000 against 0.0007 and 0.0060) — more held-out seasons tighten the
pooled interval but each one is an independent chance to lose the unanimity requirement, so
under this combined rule more data does not monotonically buy more power the way it would
under the interval alone. This is the same shape #357/#335 found for the draft gate at δ=2.0
(combined power 0.0324 against unanimity-alone's 0.136, both lower than the interval alone
would give) — here it is more extreme, because the borrowed `s` is large relative to the
borrowed `δ`.

**Restated 2026-10-06 (#418, rule 13):** the 0.0324 against 0.136 cited here is reproduced to
the digit by the script today and is the draft gate's power under the script's stand-in
within-season spread. At the draft gate's own recorded precision the same harness gives **0.0075**
(null 0.0014); the shape this paragraph names (combined power below the interval alone, more
seasons not monotonically buying power) is not what #388's table contradicts, only the level. see *What contradicts a published figure* (#388's results section, attributed under #418).

**The rule-16 exemption, named, for two independent reasons.** First, **no season has a
measured ceiling today**, so under S6 the gate is NOT-RUNNABLE regardless of what the interval
or every-season half would say — this alone is already enough to make ADOPT unreachable at
this archive's current state, and it is not a power question at all. Second, **even granting a
hypothetical non-binding ceiling, the combined rule's own power is under 1% at every plausible
δ and k tested** — so removing the ceiling obstacle would not, by itself, make ADOPT
practically reachable either. **Unless the rating's true edge over the close is large relative
to its own season-to-season variance — a stronger and more falsifiable claim than "beats the
close by a few hundredths of a nat" — ADOPT is this gate's named exemption, not its
expectation, on both counts.** This is written down now, before the first row is fitted, so a
future reader sees the same years of a printed NOT-RUNNABLE this document itself criticises
rule 16's incident for, and knows why on both axes. REMOVE and SHOW are not exempted by either
finding — an interval that excludes zero on the negative side, or a null, needs no unanimity in
the same brittle way and both remain fully reachable outcomes on 2026-onward data once three
ceiling-qualified seasons exist.

> **Amended 2026-09-21, at adoption: what ends the first reason, by ticket.** The ceiling arm
> is the market's own repricing from the last pre-kickoff snapshot, and that archive holds
> only from 2026-08-25 — and only where a capture persists, which CI captures currently do
> not (#383: every `poll_odds` and `refresh` snapshot on the Actions runner is discarded with
> it). So **stage 2 is blocked until the pre-kickoff snapshot archive holds three seasons** —
> ADR-0019's minimum k for any gate — **tracked by #383**; on captures that persist from
> 2026, that is the 2028 season at the earliest. This gate does not become decidable in
> October; it becomes decidable when the archive has depth. The first NOT-RUNNABLE print is
> the plan, not a surprise, and it names #383. Meanwhile the fit records the walk-forward
> log-loss gap per season *without a verdict word*, so #375's posterior has rows the day it
> lands.
>
> **Three seasons is permission to run, not a reading to act on.** At k = 3 the interval half
> is a t on two degrees of freedom — multiplier 4.30, against 3.18 at k = 4 and 2.78 at k = 5 —
> so the first runnable gate's interval swallows almost anything, and the sign half's null size
> is 2⁻³ = 12.5%, one run in eight. The conjunction still protects; what it cannot do at k = 3
> is say anything a reader would act on. The honest sequence: stage 2 *unblocks* at three
> seasons; the verdict word becomes *informative* at many more; and #375's posterior is what
> makes a three-season fit worth looking at, because a posterior over an effect says something
> at k = 3 where a verdict word does not — the same argument that made #375 load-bearing for
> this section, applied one level down.

## Exclusions (rule 11)

* **Unpriced games** — `close_spread` null (`hub.schedule.priced_games`'s own `unpriced`
  source). Neither arm scores a game the incumbent could not price; scoring the rating alone on
  those rows would hand Arm B information Arm A never had a chance on (rule 6 in reverse).
* **Postseason games** — excluded, as `hub.models.margin`'s own gate already does over the same
  1999–2025 archive ("1999-2025 has a handful; dropping them is cleaner than inventing a
  convention," `margin.py:94`). A playoff game's stakes and roster usage are not what a
  regular-season walk-forward log-loss is built to price, and the postseason bracket is what
  `hub.exhibits.championship_equity` already exists to reason about separately.
* **Ties** — excluded, not scored at 0.5 the way the quarterback gate's rarer event games are.
  An NFL tie is a genuinely different outcome from either team winning, `y` has no natural value
  in the log-loss formula above without inventing one, and the population is small enough
  (roughly one every few seasons) that excluding them changes no season's sample size
  materially.

## What published number moves if it adopts (rule 13)

The ticket names six: *"The survivor IP, `pool.Field`, `weekly`, `leverage`, `buyback` and
`sensitivity` all read it [win_prob]."* Read literally against the code as it stands today,
**none of the six moves on this gate's ADOPT alone**, and that is worth stating precisely
rather than assuming the obvious wiring:

`ratings.forecaster()` is read by `hub.models.ratings.fit()` — the predictions the site
publishes (`site/data/preds_*.json`), which is what actually moves on ADOPT: every `win_prob`,
margin and provenance stamp `fit()` writes. **`hub.season.survivor.grid_from_schedule` does not
call `ratings.forecaster()` at all** — it imports `MARGIN_SD` and `normal_cdf` from
`hub.models.market` directly and recomputes the conversion by hand
(`survivor.py:578-584`, and its own docstring: *"The spread-to-probability conversion is
`MarketBaseline`'s, so a survivor pick and a [game prediction] agree"* — an intentional
agreement enforced by duplication, not by a shared call). The six consumers the ticket lists —
survivor's own IP, `pool.Field`, `weekly`, `leverage`, `buyback`, `sensitivity` — all sit
downstream of `grid_from_schedule`, not of `ratings.forecaster()`. **So an ADOPT on this gate
moves the published game predictions and moves none of the six**, until a second, separate,
later ticket rewires `grid_from_schedule` onto the rating — which is itself a serving-path
change and belongs behind its own pre-registration under this same discipline, not something
this gate's ADOPT branch licenses by implication. Naming this now is what stops the six from
being quietly assumed to have moved the day this section reads ADOPT.

## Closing

**PROPOSED.** Awaiting the maintainer's `ADOPTED:` comment on this section before the fit
begins, per the order stated at the top: pre-register → fit as an unwired exhibit → gate in
October, after Audit V (#325).

---

# Pre-registered 2026-09-21 — PROPOSED: the state-space team rating on CFB (#366)

**Status: PROPOSED.** Drafted from `docs/audits/2026-09-20-method-audit.json` finding **S8a**,
`blocked_by` #365 (S8) on the tracker and by this pre-registration's own order: the CFB
extension reuses #365's model and gate machinery and this section states only where it
differs. Parent decisions: the same four as #365's section, plus `docs/cfbd-quota.md` (the
quota budget below is read against it) and `hub.schedule.LeagueUnavailable` /
`hub.schedule.SLATES` (`src/hub/schedule.py:150-192`), which is the mechanism #366's serving
change touches.

**#366 was `blocked_by` #326 for the same stated reason as #365 and it is withdrawn for the
same reason: this design declares every fitted constant `not_an_input` while unwired.** What
remains frozen is the *gate*, for the *same* instrument-change reason as #365 — S1 landed
2026-09-21 and no CFB record has been scored under it — and, separately, for a *serving* change
named below that is frozen on its own terms regardless of the gate.

## The question

> Over FBS games — including and especially G5 and non-conference games a sharp market does not
> price closely, where an independent rating plausibly has edge a second opinion on a good price
> does not — does the CFB extension of #365's state-space rating beat the close on walk-forward
> log-loss, under the same fixed rule, where a close exists; and, where none does, what does the
> rating cover that today serves nothing at all?

## The model — #365's, with two differences

**Same state-space form** (Normal margin, a per-week Gaussian random walk, a between-season
shrink, home advantage a scalar), fitted on CFBD's own game results rather than nflverse's, over
whatever seasons CFBD's `/games` and `/lines` endpoints return.

**Difference 1 — priors from `/ratings/sp` and `/player/returning`.** 136 FBS teams a season
against a random walk with no burn-in comparable to the NFL's 26 spare seasons (CFBD's
useful coverage does not reach back to 1999) means a flat, uninformative starting prior is a
real cost here in a way it is not for the NFL. `/ratings/sp` (SP+, a full-team composite
already published preseason) and `/player/returning` (returning production, a decomposable
proxy for how much of last season's team survives) are both already in the CFBD quota budget
and read by no model (the ticket's own finding). **Both enter as a `chosen` linear combination
weighting the preseason prior mean, not a `fitted` one**: `PRIOR_SP_WEIGHT` and
`PRIOR_RETURNING_WEIGHT`, `chosen` rather than `fitted` because setting their *relative* weight
from the same walk-forward log-loss this gate exists to run would be circular — the prior would
be tuned on the estimand — and a `chosen` weight is exactly ADR-0006's category for that case.
`WEEK_STEP_SD`, `SEASON_SHRINK`, `HOME_ADV` and `RATING_MARGIN_SD` stay `fitted`, on CFB's own
data, as separate constants from #365's NFL fit — a shared name across two different
populations is exactly the stem-collision `hub.declare.declarations()` refuses, so this module
declares its own (`CFB_WEEK_STEP_SD`, etc.), not a re-use of #365's. **Every constant here,
`fitted` or `chosen`, is spelled `not_an_input` while the exhibit is unwired, for the identical
reason #365's are.**

**Difference 2 — the CFBD quota, stated against the budget.** `docs/cfbd-quota.md`'s own
weekly in-season budget is ~12–15 calls (~60/month) already spent by the slate, with roughly
940/month of headroom; the ticket's own estimate for this fit is **~56–90 calls a month**, and
**the one-time historical backfill is the larger one-off cost** — `docs/cfbd-quota.md`'s own
"Historical backfill" section prices 2015–2025 across games, lines, box scores and SP+ at
**roughly 250–350 calls**, to be spent in the first week of a billing month exactly as that
section already instructs, and never inside a loop over teams or games (`CLAUDE.md` hard rule
3; `/ratings/sp?year` and `/player/returning` are both already season-bulk, one call per year,
so nothing about this fit is the per-team pattern the quota architecture exists to forbid). In
the billing month that carries the backfill, the two together run **roughly 306–440 calls**
(250–350 backfill plus 56–90 that month's fit) against the 1,000/month free tier — inside it
with margin even alongside the slate's own ~60/month and the Big Ten archive's ~30, but not so
much margin that the backfill belongs anywhere but its own scheduled week, exactly as
`docs/cfbd-quota.md`'s own "Historical backfill" section instructs.

## The population

**136 FBS teams**, not the NFL's 32 — the ticket's own argument for why this is worth building
at all: *"G5 and non-conference games are not priced by a sharp market… the one place an
independent rating plausibly has edge, rather than a second opinion on a price that is already
good."* A Power-conference game against a Power-conference opponent, priced tightly, is closer
in kind to #365's NFL question — a second opinion on a good price. A G5 team's non-conference
game against a team CFBD prices thinly or not at all is a different question this gate has to
answer separately, which is exactly why the arms below split on whether a price exists.

## The arms

* **Where a price exists: Arm A the close, Arm B the rating** — identical in form to #365,
  `normal_cdf(close_spread / MARGIN_SD)` against `normal_cdf(margin_hat / CFB_RATING_MARGIN_SD)`,
  scored by the same walk-forward log-loss, paired by game, clustered the same way (between:
  season; within: game — the same "not a no-op" reasoning as #365, and a CFB season holds
  roughly 800 FBS-vs-FBS games before exclusions, 136 teams × ~12 games / 2).
* **Where no price exists: reported as unpriced coverage, not scored against anything.** This is
  the population the ticket's own argument is about, and rule 6 forbids inventing a comparator
  for it — there is no incumbent to beat on a game nothing prices. What this gate reports for
  that population is **coverage**: the count and share of FBS games each season with no
  `close_spread`, split by whether both teams are FBS (in scope) or one is FCS (excluded,
  below), and the rating's own predicted win probability on those games, published as a
  descriptive number and never gated — the same status the market-free guard on
  `hub.models.weekly` (#212's re-registration, above) already gives a number that has nothing to
  be scored against: shown, not ranked on, until a benchmark exists.

## Pairing, clusters, ceiling — #365's, on the priced population only

Pairing by game, clusters `SEASON_CLUSTER` between and the game within, exactly as #365. **The
ceiling arm is the same declared quantity — the market's own repricing, the last pre-kickoff
poll `data/processed/lines` holds, distinct from whatever `close_spread` resolved to
earlier** — and it inherits #365's own finding about where that arm is thin: CFBD's `/lines`
endpoint and this repo's own live-polled archive both start with the 2026 season (the Big Ten
availability archive under `docs/cfbd-quota.md` is the same infrastructure this ceiling would
read), so **every season before 2026 has no measured ceiling and is NOT-RUNNABLE under S6 by
the identical reasoning #365 states for 1999–2025** — here total, since CFB has no analogue of
the NFL's `spread_line`-as-historical-close column already in the tree. The gate as
pre-registered here cannot produce anything but NOT-RUNNABLE until the 2026 season's own poll
archive has accumulated the three seasons ADR-0019's own floor requires — **no sooner than
2028**, the same date #365 names for the NFL side, for the same reason.

## Power at the seasons CFBD returns

The same borrowed `s = 0.0332` and `δ = 0.0090`/`0.0180` as #365 (no CFB-specific pilot exists
either, and inventing a different stand-in for a population this repo has never scored would be
choosing a number to make the arithmetic prettier, not measuring anything), at `k` bounded by
what a ceiling-qualified archive could ever hold: **the earliest possible k is 1 (2026), and it
does not clear ADR-0019's three-season floor until 2028 — the same non-runnable regime as
#365's finding**, so the power table is the same one: ADOPT is a named exemption on this gate
too, for the identical structural reason (unanimity across k independently-drawn seasons
against a between-season noise the size of the only pilot sd this repo has). REMOVE and SHOW
stay reachable on the priced population once three ceiling-qualified seasons exist; the unpriced
coverage report is not gated at all and needs no power calculation, because rule 16 is about a
gate that returns a verdict and the coverage number returns none.

## `LeagueUnavailable` degrading to the rating — a serving change, named and kept separate

`hub.schedule.priced_games(league="cfb")` raises `LeagueUnavailable` today because `SLATES`
holds only `{"nfl": _nfl_slate}` (`schedule.py:192`) — there is no CFB loader at all, so the
refusal is not about pricing, it is about there being no schedule to price. The ticket's
acceptance criterion — *"`LeagueUnavailable` degrades to the rating instead of refusing"* —
presupposes a CFB `Slate` loader exists (reading `hub.fetch.cfbd`'s `/games`, which the ticket
notes is already fetched and priced by nothing) and, once one does, changes what a caller
receives on a league CFBD cannot price at all: today, a raise naming the league (`schedule.py`'s
own `LeagueUnavailable` docstring: *"a caller's only way to tell 'no college schedule' from 'no
games this week'"*); after, the rating's own number where the close has none.

**This is graceful degradation in `CLAUDE.md`'s own sense** ("every module must produce a
usable answer with zero attention… serve last-good state rather than erroring"), and it is also,
precisely because of that, **a change to what the serving path does on a refusal it currently
recognises as a refusal** — which is a *landing*, not a fit, under the freeze's own axis
(`docs/gate-power.md`'s quote of #326: *"blocks a modelling change landing"*). **So it is named
here and frozen on its own terms, separate from the gate above**: adding a CFB `Slate` loader
and wiring a degrade path is its own commit, behind its own flag, and does not land ahead of
#365's gate resolving — a CFB rating degrading `LeagueUnavailable` before the NFL rating has
even been fitted would serve an unvalidated number to the one caller (`hub.season.survivor`, if
it is ever asked for `league="cfb"`; nothing in `src/` asks today) with no gate behind it at
all, which is a stronger version of rule 1's incident than either #365 or #366's own gate risks
being. **What this pre-registration authorizes today is naming the change and its precondition
— not building it.**

## Exclusions

* **FCS opponents.** An FBS-vs-FCS game is not this population — CFBD's own SP+ and returning
  production are computed for FBS programs, an FCS opponent has no comparable rating input, and
  the result is dominated by a talent gap this model was never built to price. Excluded from
  both the priced and the unpriced-coverage counts; noted, not silently dropped, in the coverage
  report's own denominator.
* **Bowl games?** — an open question, named as one rather than decided by default. The
  quarterback gate and #365 both exclude the postseason on the argument that stakes and roster
  usage change (`margin.py:94`'s "a handful; dropping them is cleaner than inventing a
  convention," reapplied). CFB's bowl season is a larger share of its slate than the NFL's
  playoffs are of its regular season, opt-outs and transfer-portal departures move rosters
  between the regular season and a bowl in a way the NFL's do not, and a walk-forward rating
  fitted through the regular season is answering a different question about a bowl roster than
  about the team that earned the bid. **This document does not decide it** — a bowl-inclusion
  choice changes what the estimand *is*, which is ADR-0024's line between a sensitivity an agent
  reports and a decision that stays with the maintainer, and it is filed here as the latter,
  for the `ADOPTED:` comment to resolve alongside the rest of this section.

## Closing

**PROPOSED.** Awaiting the maintainer's `ADOPTED:` comment, in the same order #365 states:
pre-register → fit as an unwired exhibit, `blocked_by` #365's own `ADOPTED:` line → gate in
October alongside #365, after Audit V. The `LeagueUnavailable` degrade path is a separate,
later, named landing and is not authorized by this comment alone.

# Pre-registered 2026-09-21 — PROPOSED: the game-level betting bar (#364)

**Status: PROPOSED.** Drafted under the operate-mode freeze (#326), which blocks a modelling
change *landing* and not writing the rule before the number — #329 and #305 are the
precedent, and this section follows #305's form (below), per the maintainer's own instruction
on #364. It becomes the rule on a maintainer `ADOPTED:` comment on *this section*; until then
it decides nothing and no game-level model is gated against it. Parent decisions: the S7
finding (`docs/audits/2026-09-20-method-audit.json`) that raised the question — the house
rule, built for constant refits where a false negative costs nothing, applied unmodified to a
betting edge where a false negative costs the whole enterprise; ADR-0024 (a modelling decision
becomes agent work only when its alternatives sit on one axis — S7's three options changed what
the bar *is*, not a parameter on it, so #364 stayed a decision and was converted under that
ADR's step 4 once the maintainer chose one); ADR-0019 as amended by #357 (S1, the t-interval
half) and #335 (the tie-aware every-season half) and by #363 (S6, the ceiling precondition),
all three read into this bar rather than restated; and `docs/method.md` rules 6, 7, 11 and 16,
each applied below rather than only cited. Decided together with #375 (the phase-2 posterior),
whose utility this bar's linear EV *is* — #375's own pre-registration is a separate ticket and
is not written here.

**The maintainer's `ADOPTED:` comment on #364 (2026-09-21), restated rather than repeated:**
option 3 — a decision-theoretic bar, linear EV at a flat unit stake, the form pre-registered
now and the dollar figure named later. Indifferent to ruin: a bar for whether a model is
*better*, never a staking rule. Utility and prior are `chosen`, not fitted, declared in
`hub.declare`. Decided together with #375. No model is built by this ticket.

## The estimand

> Over held-out seasons, does a game-level model's picks — staked flat at one unit against the
> closing line — clear the standard −110 break-even (52.38%) in expected value, and is that
> gain distinguishable from the noise of ~285 games a season?

A **pick** is a side (favorite/underdog against the closing spread, or over/under against the
closing total) the model names for a game, at a flat one-unit stake, priced at the close —
`hub.fetch.odds`' last snapshot before kickoff, the `Snapshot`/`Live price` vocabulary
`CONTEXT.md` already defines and this section reads rather than re-derives. A graded pick's
realised profit is **+100/110 unit** if the side covers, **−1 unit** if it does not, and is
excluded from `n` on a push. **A season's EV is the mean realised profit per graded pick over
that season's games** — the repeated measure is the game, per `docs/method.md` rule 3, never
a season total standing in for one observation. Break-even at −110 is **p = 52.38%**, exactly
where EV = 0; "does EV clear zero" and "does the hit rate clear break-even" are the same
question in two units, and this section reads EV throughout because that is the unit the
ceiling arm (below) is stated in.

## The arms

**Rule 6 (`docs/method.md`): both arms must have the same information.** The model's pick is
made from information available strictly before kickoff — the cut `hub.schedule.priced_games`
already enforces for every prediction in this repo — and is priced at a close the pick
predates; a "pick" read off the closing line itself, or off a line captured after the pick was
notionally made, is scored against a number it already saw, which rule 6 forbids on either
arm. There is no second, more complicated incumbent the way `weekly_gate` compares two
projections: **the comparator is the flat one-unit stake itself — "do not bet" — whose EV is
identically zero by construction.** So the gate's `gain` is the model's own realised profit
per graded pick, and the pooled test is against zero rather than a paired two-model
difference — `docs/method.md` rule 5 (gate against the simplest thing that works) taken to its
floor: nothing is simpler than not staking, and its EV needs no measurement.

**Rule 7 (`docs/method.md`): pair the comparison.** The unit is one graded pick on one game;
gains are never aggregated to a season before the season-level statistic is computed, so a
season's `n` and its own within-season spread are read off the games it actually graded.

## Clusters

**Between:** `SEASON_CLUSTER` (`hub.models.experiment.SEASON_CLUSTER`, `("season",)`), the
unit every gate in this repo resamples on and the one rule 3 requires for a repeated measure
within a season.

**Within-season: the game — not a no-op.** ADR-0019's per-gate table marks three within-season
units **no-op** (quarterback's event, margin's season, injury type's season) because each of
those paired frames is already one row per season, so every season's own cluster count sits
below `TIE_MIN_CLUSTERS = 12` by construction and #335's tie test falls back to the sign
alone. A game-level betting season carries on the order of 285 graded picks — the floor is
cleared roughly 24× over — so here the within-season unit does real work: a season's gain is a
**win** only if it clears `2 × se` over its own ~285 games, a **tie** otherwise, never a raw
sign read the way a one-row-per-season gate must settle for.

**Checked for degeneracy against its own inputs (rule 17), before this is pre-registered.**
The interval half here is the same `t_interval` #357 fixed — a distributional claim on the
season-clustered mean, not a resample of the seasons' own signs — so the defect rule 17 found
(the interval half silently restating the sign half under the old percentile bootstrap) does
not recur; `gate()` is read unaltered, not re-derived. What is specific to this gate is
whether the *tie* test collapses back into a sign test at this scale, and it does not: at
m ≈ 285, `_disposition` reads each season's real `2 × se` band rather than falling back to the
sign, so the every-season half is doing work the interval half does not already do. The rule-16
simulation below checks the conjunction's power rather than arguing it, the same way #357's own
planted-rule test does for the interval half.

## The gate, as it now stands

`hub.models.experiment.gate`, unaltered — this ticket adds no branch and moves no threshold:

- **NOT-RUNNABLE** — first, per #363 (S6) — if no ceiling was measured, in either direction,
  or (second) if the season-clustered MDE exceeds a measured ceiling (stage 2, ADR-0019).
- **ADOPT** — the t interval on per-game EV excludes zero on the positive side **and** the
  model wins every held-out season (ties are not wins, #335).
- **REMOVE** — the mirror: the interval excludes zero on the negative side and the model loses
  every held-out season (ties are not losses).
- **SHOW** — otherwise: absence of evidence, not evidence of equivalence.

**What REMOVE means for a model that does not exist yet.** Every other REMOVE in this repo
retires a shipped module to `hub.exhibits` (`championship_equity`, `leverage`, under #363's
own amendment). No game-level model is built or published by this ticket, so there is nothing
here for REMOVE to retire — it means **the bar was cleared in the losing direction: a
candidate that reaches this gate loses money against the close, with evidence, and does not
ship.** ADOPT, symmetrically, is not "keep a thing already live" but "this is the first
game-level model to earn a place." The branches are unchanged; what is new today, and for as
long as no such model exists, is that the gate has no rows to read and prints exactly what an
empty frame prints — `SHOW`, "nothing measured — no paired observation" — `docs/method.md`
rule 12's category, an absence of evidence, never a verdict on any model.

**Indifferent to ruin, restated as `gate` actually reads it.** Nothing in `experiment.gate`'s
branches sizes a bet, compounds a bankroll, or bounds variance — the flat one-unit stake
(declared below) is chosen precisely so EV is linear and a season's mean is a season's mean,
with no path-dependence for the gate to reward or punish by accident. A model that clears
ADOPT here is not thereby shown safe to stake at any particular size; sizing is #375's
question and everything downstream of "how much," never this gate's.

## The ceiling arm, declared

#363 (S6) makes a declared ceiling mandatory before any branch below NOT-RUNNABLE is read, in
both directions. The precedent for a market-facing gate is already set in this document, for
#291's quarterback gate: **the ceiling arm is the betting market's own repricing — the last
snapshot before the game, scored the same way the candidate is, against the price the
candidate's own pick predates.** In that section's own words: "What the betting market
recovered by the close is the most an adjustment built to anticipate it could recover." #364's
ceiling is the same construction, in EV units at the same flat unit stake and vig: what a
bettor earns from nothing but the market's own subsequent movement — no forecast of their own,
only the market's later, better-informed price — is the most a *forecasting* model, built to
anticipate that movement ahead of time, could earn. This is also why the ceiling is not the
game's own final result: #305's section above tried and withdrew exactly that shape of
ceiling — a scaler reading the realised team total "recovers +0.228... but the realised total
is the outcome of the same game the player scored in, not a better estimate of it," so it
answers nothing rule 8 asks and fails rule 6 by construction (it is information no pre-kickoff
model has). The market's own repricing is a forecasting ceiling in the same units the
candidate is scored in; the game's own result is not a ceiling at all.

**Declared, not yet measured.** No ceiling exists until a #375-shaped model states its own
action time and an archive of closing-line snapshots exists to compute the repricing against
it, so today's gate is NOT-RUNNABLE on the ceiling branch alone — independent of, and prior
to, the power question below.

## Power before the run (rule 16)

Computed with `scripts/rule16_combined_power.py`'s own harness — `numpy`/`polars` and the
shipped `experiment.summarise`/`per_season`/`gate`, nothing reimplemented — extended by a cell
built from this ticket's own pre-registered figures, since no gate has run and there is no
measured table to read a cell from: **k = 10** seasons, **m ≈ 285** games/season, at −110.

**A wrinkle, checked rather than assumed.** `scripts/rule16_combined_power.py`'s existing
cells (draft, weekly) call `summarise()` with no `ceiling=`. Run unmodified against `main`
today, both report **0.0000/0.0000**, not the 0.0113/0.0324 and 0.0186/0.1925 ADR-0019's own
condition-2 table publishes — because `gate()` now carries #363 (S6) and returns NOT-RUNNABLE
whenever no ceiling is supplied, before the interval or every-season half is read at all. That
table was computed, by the script's own docstring, before the #335 amendment — and, it turns
out, before S6 as well — so it no longer reproduces against the `gate()` it calls today. That
is `docs/method.md` rule 13's situation and this ticket does not own the fix; flagged
separately rather than silently worked around here. What this cell needs is the **combined
rule's own** null size and power — the #357 interval half and #335 tie-aware half, the
conjunction this amendment is actually about — independent of the (separately declared, above)
S6 ceiling precondition, so `summarise()` below is called with a large placeholder
`ceiling=1e6` solely to hold that branch open and let the interval+tie-aware conjunction be
what the numbers measure.

**Restated 2026-10-06 (#418, rule 13): the "wrinkle" above no longer holds, and its cause was
not the table.** The script was changed fifty minutes after this paragraph was written
(`51909c2`, 2026-09-21: the harness hands `gate` a ceiling that cannot bind, and says why in its
own comment), and #386 later moved it to the frame-in `gate`. Re-run on current main it returns
**draft 0.0113 / 0.0324 and weekly 0.0186 / 0.1925, to the digit**, not 0.0000/0.0000. The
0.0000/0.0000 is real for a call with `ceiling=None` (300 of 300 trials NOT-RUNNABLE, measured
under #418), so the diagnosis of *that call* stands; the conclusion that the ADR table
"no longer reproduces" does not. The #363 path owns none of the gap to #388's figures; the
generating process owns almost all of it. see *What contradicts a published figure* (#388's results section, attributed under #418).

**Converting the ticket's own hit-rate SE to EV units.** EV(p) = p·(100/110) − (1 − p) =
1.90909·p − 1, so d(EV)/dp = 1.90909 at any p — the constant conversion factor at this vig.
At the break-even null p = 0.5238, a single game's win/loss standard deviation is
`sqrt(0.5238 × 0.4762) ≈ 0.4995` in probability units, **s ≈ 0.9535** in EV units — matching
the ticket's own **per-season SE ≈ 2.96pp** (`sqrt(0.25/285) × 100`, the same figure near
p ≈ 0.5) converted the same way: `0.0296 × 1.90909 ≈ 0.0565` EV units of season-level sampling
noise. Unlike the draft/weekly cells' own `s` — a between-season figure reused as a stated,
conservative stand-in for an unmeasured within-season spread — this cell's `s` is not a
stand-in: under the null, a game's outcome really is close to Bernoulli(0.5238) and its
per-game standard deviation is what the formula above gives directly, so using the same `s` at
both the between- and within-season scale is the correct model here, not merely the script's
usual simplification. At a **realistic edge — a 53% model** — the detectable delta is
`EV(0.53) − EV(0.5238) ≈ 0.01184` EV units per game, about 1.18 points of ROI, sitting just
under the ticket's own stated MDE (≈2.9pp ≈ 0.0554 EV units).

**Result, 500 trials, bootstrap 200, seeds 0 (size) / 1 (power) — `ceiling=1e6` throughout so
only the interval+tie-aware conjunction is read.** Reduced from the script's usual 10,000
trials because of sustained system load (load average in the 40s–50s) during this run; the
same qualitative result — zero adoptions at both the null and the edge — held at 100 and at
500 trials, the two counts actually run after the ceiling correction above, so the reduction
costs precision on the exact rate, not the direction of the finding:

| rule | k | s | m | δ (53% model) | null size | power at δ |
|---|---|---|---|---|---|---|
| combined (t-interval + tie-aware every-season, #357+#335) | 10 | 0.9535 | 285 | 0.01184 | **0/500 (<0.2%)** | **0/500 (<0.2%)** |
| unanimity-alone (interval + raw sign, pre-#335) | 10 | 0.9535 | 285 | 0.01184 | **1/500 (0.2%)** | **0/500 (<0.2%)** |

**Reading the table.** Both null sizes sit comfortably under `ALPHA` (0.05) — consistent with
#335's own finding that a tie-aware rule can only be *stricter* than the interval alone, never
more permissive, so the combined rule is not the source of any excess false-adopt risk here.
The power figures are the finding: at k = 10 seasons the combined rule adopts on **at most a
few tenths of a percent** of runs where a true 53%-edge model exists, and the same is true of
unanimity-alone. That is markedly lower than the ticket's own every-season-alone figure —
`P(positive in all 10) ≈ 0.12` — because unanimity-alone here still carries the interval's own
exclusion requirement, and the interval's MDE (≈2.9pp ≈ 0.0554 EV units) is **more than four
times** the 53%-model's own edge (0.62pp ≈ 0.01184 EV units): **the interval half, not the
every-season half, is what starves power at this k.** Ten seasons resolves an edge roughly
4–5× larger than a 53% model's before the interval alone will even exclude zero; #335's tie
test then narrows the reachable band further on top of that.

**ADOPT is unreachable at a realistic edge — named as a rule-16 exemption, not a bar
(2026-09-21).** A 53% game-level model — a strong edge, by any description this repo would
give a betting model — clears this gate's ADOPT branch on well under 1% of runs at k = 10
seasons of ~285 games. Per rule 16 and #300's precedent (the draft gate's own ADOPT branch,
named an exemption in this document for the identical reason: reachable in principle, not at
any effect size this project's data can plausibly produce), **ADOPT under this bar is named an
exemption rather than a bar for as long as k stays near ten:** the interval half alone needs
roughly a 4–5× larger edge than a 53% model carries before it will exclude zero at this k, and
closing that gap by season count rather than edge size needs on the order of `4.5² ≈ 20×` the
clusters — decades at one season a year, the same shape #291's quarterback diagnostic already
lives with. This does not weaken the bar: a model that cannot clear the interval's own MDE has
not been shown to lose money either, and **SHOW remains the honest, expected reading for a
plausible edge at a realistic k** — not a defect in the gate, and not evidence the model is
bad. What the exemption changes is only how that SHOW is read when it arrives: as ten seasons
being, on their own, too few to resolve a realistic edge, not as this bar failing to notice a
good model.

## Exclusions (rule 11)

- No live/in-game betting, no parlays or correlated same-game combinations, no alternate lines
  or reduced juice — one closing line per game, one flat unit, one side.
- No staking rule is in scope. Kelly, fractional bankroll, confidence-weighted sizing are all
  #375's question, not this bar's — the same `ADOPTED:` comment that created both tickets
  together keeps them apart.
- No claim that a model's edge is causal, or that the closing line is inefficient in a way this
  gate can itself demonstrate a mechanism for — `CONTEXT.md`'s own definition of the betting
  market ("treated as already-efficient... the repo backtests against it to audit itself,
  never to beat it") is read literally: this bar tests whether a model beats the close
  *empirically*, once, on held-out seasons, and a pass is evidence, not an explanation.
- Not read before a game-level model and a closing-line archive both exist — no `--ceiling` or
  interim number is published from this section before then.
- The dollar value of the flat unit stays out of scope, per the `ADOPTED:` comment: "the
  dollar figure is named later, the form is pre-registered now."

## The `hub.declare` entries

The unit stake form and the utility are **chosen, not fitted** — the maintainer's `ADOPTED:`
comment on #364 says so directly, and `hub.declare`'s own vocabulary (`src/hub/declare.py`)
reserves `chosen` for exactly this: "a stated choice a prediction reads... the shape of a
random draw, a pin, a threshold chosen from data," which a flat stake and a linear utility are.

**No line is added to `src/` by this ticket.** `declare.declarations()` walks module-level
assignments in real source files — a declaration exists where a constant is *used*, not where
it is discussed — and #305's PROPOSED section, the template this one follows, declared no
constant either, for the same reason: nothing in `src/` yet reads a stake size or a utility
function to declare. What is pre-registered now is the *form* the eventual declarations must
take, written here so a later implementer does not have to re-derive it from the `ADOPTED:`
comment alone:

- the stake is **flat**, one unit, unconditional on the model's own confidence — the form is
  chosen now; the dollar value is deferred by the `ADOPTED:` comment itself;
- the utility is **linear in profit**, `u(profit) = profit` — the identity, which is what
  makes a season's mean EV additive across its games and indifferent to bet order or bankroll;
- the prior over a model's true edge that #375 reads is **chosen, not fitted**, and is #375's
  own declaration — named here only because the `ADOPTED:` comment ties the two tickets
  together, not because this section owns it.

When #375's model and this bar's own call site exist, `STAKE_UNIT = chosen(...)` (or its
eventual name) and the utility function are where `hub.declare` actually reads them; nothing
about that is pre-empted here.

## PROPOSED — what is already decided, and what would reopen it

**Restated, not repeated:** the maintainer's `ADOPTED:` comment on #364 (2026-09-21) chose
option 3 — a decision-theoretic bar, linear EV at a flat unit stake, the form pre-registered
now and the dollar figure named later. Indifferent to ruin: a bar for whether a model is
*better*, never a staking rule, and every branch above is written so that reading holds.
Utility and prior are `chosen`, not fitted, declared in `hub.declare` at call sites that do
not exist yet. Decided together with #375, whose utility this bar's linear EV supplies.

**What this section still needs before it decides anything.** The option was adopted; the
mechanics were not — the estimand, the arms, the declared ceiling and the power table above
are new in this section and have not been reviewed. This becomes the rule on its own
maintainer `ADOPTED:` comment, the same two-step #305 went through: the choice first, the
detailed pre-registration drafted and adopted second.

**What would reopen the decision already made.** A stake form other than flat-unit — Kelly,
fractional bankroll, anything that sizes a bet on the model's own confidence — which the
`ADOPTED:` comment names as "a different object," not a parameter on this one. A change to
that form is a new ADR-0024 decision, not an amendment to this section.

# Pre-registered 2026-09-21 — PROPOSED: the posterior gate (#375)

**Status: PROPOSED.** Drafted under the operate-mode freeze (#326), which blocks a modelling
change *landing* and not writing the rule before the number — #329, #305 and #364 (above) are
the precedent. It becomes the rule on a maintainer `ADOPTED:` comment on this section; until
then it decides nothing and no gate reads it. Parent decisions: `docs/audits/2026-09-20-method-
audit.json`'s Q9 finding, which named the model in the form reproduced below; ADR-0019 as
amended by #357 (S1, the t-interval half), #335 (the tie-aware every-season half) and #363
(S6, the ceiling precondition) — this section changes what the gate *reports*, not the
mechanics those three already fixed; ADR-0024 (a modelling decision becomes agent work only
when its alternatives sit on one axis — the prior scale, the switching cost's form and the
fitting engine are three separate axes with no single measurement that closes all three, so
this stays a `PROPOSED` decision needing a maintainer `ADOPTED:` line, exactly as #364 did
before it); ADR-0006 (`fitted`/`chosen`, `hub.declare`); and `docs/method.md` rules 1, 3, 11,
13, 16, 17 and 18, each applied below rather than only cited. The utility this ticket reads is
**#364's own, by heading** ("The gate, as it now stands" → "Indifferent to ruin" in that
section, above) — linear EV at a flat unit stake — and is not restated or re-decided here.
Blocked by #357, which closed 2026-09-21: the edge is discharged and nothing below is waiting
on it.

## What this replaces, and what it keeps

**It replaces the verdict word — the `ADOPT`/`REMOVE`/`SHOW`/`NOT-RUNNABLE` string
`experiment.gate` prints — with a posterior over the effect and the action that posterior's
expected utility recommends.** S1 and #335 made the existing two-part rule strict again; they
did not change what a gate *says*, which is still one of four words with no notion of how
confident the word is or what it would cost to act on it differently. At k = 3–5 seasons —
every gate in this repo — "is the sign distinguishable from zero in every held-out season" and
"given the evidence, what do I believe about the effect, and which action maximises expected
utility under that belief" are different questions, and only the second uses the graded
information a t-statistic and a season count both throw away the moment they cross a
threshold.

**It keeps everything else.** Every discipline this document already enforces — the decision
rule fixed and tested before the numbers (rule 1), the season as the unit of repeated measures
and never pooled (rule 3), both arms reading the same information (rule 6), the comparison
paired (rule 7), a ceiling computed before any gap is chased (rule 8), exclusions named (rule
11), a measurement that contradicts a published number not finished until the number moves
(rule 13), power and calibration computed before the run (rule 16), a decision rule checked for
degeneracy against its own inputs before it is pre-registered (rule 17), and a positive control
for every check (rule 18) — applies to the posterior gate exactly as it applies to the sign-
and-interval one. **Only what gets pre-registered changes: a prior, a fitting procedure and a
utility threshold, in place of a significance bar and a season count.** This is not a
loosening — #375's own issue body says so, and the "shrinks a 2-se effect by half" prior below
is the concrete argument for why not.

## The model, and its constants

```
delta_s ~ Normal(delta, tau)      # s = 1..k held-out seasons, one per gate's own paired frame
delta   ~ Normal(0, prior_sd)     # the population-level effect; skeptical, pre-registered
tau     ~ HalfNormal(tau_scale)   # season-to-season heterogeneity; pre-registered
```

`delta_s` is read off exactly the vector `per_season` already returns for that gate —
`seasons["gain"]`, the per-season paired mean `paired.group_by(["season"]).agg(mean("diff"))`
computes today — with its own within-season SE (`per_season`'s `se`/`m` columns, #335's own
addition) supplying the observation-level noise around each `delta_s`. Nothing about what is
*measured* changes; see "The ceiling arm," below.

Three constants, each `chosen` — not `fitted`, because none of the three is a measurement with
a confidence interval, and ADR-0006 is explicit that a stated choice a prediction reads is
`chosen` whatever its provenance:

- **`prior_sd`.** Proposed **per gate**, not as one repo-wide number, because "skeptical" only
  means something in a gate's own units. Set to that gate's own **season-clustered pooled SE**
  — `se = s / sqrt(k)`, the same `s` #357's (S1) table and ADR-0019's rule-16 section publish —
  so that a pooled point estimate sitting exactly at `experiment.MIN_SE`'s bar (2 SE, "the
  repo's usual bar," rule 1) shrinks under a simple normal–normal update to **1 SE**: a
  concrete, checkable statement of what "skeptical" costs, in the gate's own units, rather than
  an off-the-shelf number like `prior_sd = 1`. At #357's own published figures: **draft**
  `s = 7.34` pts/team-game, `k = 4` → `se = 3.67`, so `prior_sd = 3.67` pts/team-game; **weekly
  blend** `s = 0.382` pts/team-week, `k = 4` → `se = 0.191`, so `prior_sd = 0.191`
  pts/team-week. Every other gate declares its own the same way, off its own `s` and `k`, the
  day this lands in code.
- **`tau_scale`.** Set to that gate's own **observed between-season scatter**, `s` itself —
  **draft** `tau_scale = 7.34`, **weekly blend** `tau_scale = 0.382` — so the prior on how much
  the true effect could vary season to season starts at exactly what the current small sample
  already shows, rather than an unrelated default that lets `tau` float to an arbitrary
  multiple of the effect it is meant to explain. `HalfNormal`'s mode sits at zero, so `tau` is
  free to collapse toward full pooling (`tau → 0`) if the seasons agree — the quantity rule 4's
  sign-flip diagnostic was reaching for without a parameter to read.
- **The switching cost.** Proposed as **`chosen(0.0)`**, in the utility's own EV units, as the
  pre-registered default until #364 names its dollar figure: the strictest transition test
  available without it — any posterior edge whose expected utility is positive switches. This
  is a placeholder pinned to #364's own unresolved number, not a claim about the true
  operational cost of shipping a new model over an incumbent (#305's scoped-adoption clause is
  evidence that cost is real and nonzero); it is named as one of the closing candidates below
  rather than settled here.

All three are declared `chosen` in `hub.declare` at the call site that reads them — see "The
`hub.declare` entries," below — so the digest moves the day any of the three moves, per rule
11's own argument: an exclusion or a choice is a decision on the record, never a number that
quietly drifts off a list.

**Checked against rule 17, briefly, because it was named for reading first.** Rule 17's defect
was a *conjunction* whose second term was implied by its first. The posterior gate is not a
conjunction of two independently-branching halves the way the pre-#357 sign-and-interval rule
was — there is one statistic, the posterior expected utility, and one threshold, the switching
cost — so that specific failure shape has no second term to collapse into the first. What
still needs checking, because nothing about a single statistic is automatically well-behaved,
is whether the *decision rule built on it* has the size it claims, which is exactly what the
rule-16 section below is for.

## The fitting

**NumPyro, not Stan — and it needs no new dependency.** `pyproject.toml` already declares
`numpyro`, `jax` and `arviz` under `[project.optional-dependencies]` as the `bayes` extra, with
its own comment recording why: "nothing in `src/` or `tests/` imports torch, numpyro, jax or
arviz, so an environment without them still runs every gate." **This section corrects the
issue body's own framing rather than restating it**, per rule 13's spirit — the model is not a
*new* dependency group; it is the first thing under `src/` that would actually import an
*existing, deliberately optional* extra. It stays an extra and not a `[dependency-groups]`
group for the same reason `bayes` was written that way in the first place: a group is
installed on every `uv sync` and every `uv run` (the toolchain's own hard-won lesson, `CLAUDE.md`'s
Worktrees section), and this repo's other thirty-odd gates must keep running, unmodified, in an
environment that has never installed JAX. NumPyro over Stan for the same reason the extra was
written the way it was: it is pure Python plus JAX, so `uv sync --extra bayes` is the whole
setup step — no `cmdstanpy` install, no compiled model binary to cache per ephemeral worktree,
which is exactly the kind of hidden setup step `CLAUDE.md`'s Worktrees section (`"there is no
setup step"`) exists to keep out of this repo. It also composes with the existing harness
style (`scripts/rule16_combined_power.py`, `experiment.summarise`/`per_season`): numpy arrays
in, numpy arrays out, nothing reimplemented twice.

## The transition

**The new outputs are printed alongside the existing verdict word, never instead of it, for as
long as this section says PROPOSED and for some time after it is adopted.** The first landing
adds a posterior mean, a credible interval, `P(delta > 0)` and the expected-utility action to
every gate's `SHOW`/`ADOPT`/`REMOVE`/`NOT-RUNNABLE` sentence — printed beside that word, not
substituted for it — so a reader sees both readings on the same line while the new rule is
unproven in production. **The old word is retired by a dated decision, not by silent
disuse**, once two things have happened: the record is re-run under #358 (the detectable-
effect restatement already `ADOPTED:` option 1 and closed) and #311 (the missing `cluster`
argument on `paired_gain`'s pooled half, still open and blocking #305), so the posterior gate's
first production reading is taken on inputs that are themselves no longer under an open
rule-13 incident. Until that dated decision, `docs/track-record.md` and every ADR keep
publishing the verdict word exactly as they do today — see "What published number moves,"
below.

## The decision rule

**NOT-RUNNABLE (S6, #363) stays ahead of everything below it, unchanged.** A gate that
measured no ceiling, in either direction, returns `NOT-RUNNABLE` before the posterior is even
read — the same ordering S6 gave the sign-and-interval rule, because a gate with no ceiling
has nothing to compare a posterior's expected utility against either. The posterior gate adds
no new way to become runnable; it only changes what happens once a gate already is.

Below that precondition, the four words become:

- **ADOPT** — the posterior's expected utility of switching to the candidate, under #364's
  linear-EV form, exceeds the switching cost: `E[utility(delta) | data] > switching_cost`.
- **REMOVE** — the mirror in the losing direction: `E[utility(delta) | data] < -switching_cost`.
  For a module that already ships, this retires it to `hub.exhibits` exactly as it does today
  (#363's own restatement of what REMOVE means for a model that does not yet exist applies
  unchanged to #364's own gate).
- **SHOW** — the expected utility sits inside the switching-cost band: `-switching_cost ≤
  E[utility(delta) | data] ≤ switching_cost`. Neither adopting nor removing is utility-
  maximising given the cost of moving off the current default, which is a *different* reading
  from "absence of evidence" — it says the decision-maker is rationally indifferent given
  frictions, not merely that the interval failed to exclude zero — and the pre-registration
  says so explicitly rather than leaving `SHOW` to carry two meanings under one word.

`P(delta > 0)` is reported beside the action on every branch as a sign-only diagnostic — it
decides nothing, the same way the false-discovery threshold decides nothing beside a screen's
verdict (`docs/method.md` rule 14) — because a reader comparing the posterior gate's action to
the sign-and-interval word during the transition needs the number that tracks the old rule's
own intuition most closely, without it being read as a second decision rule.

## Rule 16 before the run: calibration and size

Two checks, pre-registered now, computed before any gate is asked to fit real data — the same
"before the first `NOT-RUNNABLE` print" discipline rule 16 states in general and #357's and
#364's own rule-16 sections both apply to their instruments:

**1. Simulation-based calibration on the fitted model.** Draw `(delta, tau)` from the priors
above; draw `delta_s` for `s = 1..k`; draw each season's observed mean around `delta_s` with
that season's own within-season SE (`s / sqrt(m)` at the gate's own `m`, the within-season
cluster count #335's table already names per gate); fit the model on the simulated data; record
the rank of the true `(delta, tau)` among the posterior draws. Repeat at least 1,000 times and
confirm the rank histogram is uniform (a standard SBC diagnostic, not a bespoke one). Run at
**both** ends of this repo's own scale rather than one generic cell: the draft/weekly cells
(`k = 4`, `s = 7.34` / `0.382`, from #357's table) where heterogeneity is large relative to `k`,
and #364's own game-level cell (`k = 10`, `m ≈ 285`) where within-season noise is small and `k`
is larger — a model calibrated at one end and silently miscalibrated at the other is exactly
the gap a single test cell would hide.

**2. A null-size check of the decision rule, in S1's own shape (#357).** Simulate `k` season
means under the null, at each gate's own `k`/`s`/`m`, twice — once at `tau = 0` (no true
heterogeneity at all) and once at `tau = tau_scale` (the null with as much heterogeneity as the
prior itself allows) — fit the model, apply the full decision rule (posterior, expected
utility, switching-cost threshold) and tabulate the realised `ADOPT` rate across 10,000 trials
per cell, the same trial count `scripts/rule16_combined_power.py` uses. The pre-registered
target is `experiment.ALPHA` (0.05), the bar every other size check in this repo reads —
stated here as a ceiling rather than an assumption, because a Bayesian decision rule's
frequentist size is not automatically controlled by the prior being proper; this check is what
establishes it, not what confirms something already guaranteed. This is S1's own lesson
generalised past the conjunction it was found in: rule 17 asked whether a *conjunction*
degenerates; a single expected-utility statistic cannot degenerate that way, but it can still
have the wrong size, and only a simulation says which.

**The positive control rule 18 requires.** Neither check above is trusted until it has been
shown to fail on purpose. Plant `delta_true` at an effect this repo has never measured and
never will by accident — `20 × tau_scale` (≈ 73 pts/team-game for the draft cell, ≈ 7.6
pts/team-week for weekly blend, both roughly the "a hundred standard errors" shape
`test_rule16_combined_power.py`'s own positive control already uses) — and confirm the decision
rule reaches `ADOPT` on essentially every trial (≥ 99%, matching that test's own bar). A
planted effect this large that fails to `ADOPT` means the harness is broken, not that the rule
is strict — exactly the reading `test_rule16_combined_power.py`'s own docstring gives its
enormous-δ assertion. The eventual test module owes rule 18's registry
(`tests/contracts/test_every_check_has_a_positive_control.py`) two names the day this code
exists: `test_the_posterior_rule_s_null_size_is_controlled` and
`test_a_planted_enormous_effect_always_adopts`, on the naming pattern
`test_experiment.py`'s `#357 (S1)` section already set.

## Prior sensitivity

**Required output, not an optional robustness note.** Every run reports the decision — the
posterior mean, the credible interval, `P(delta > 0)` and the expected-utility action — under
**at least two** skeptical priors, so a reader sees whether the conclusion is the evidence's or
the prior's: (a) `prior_sd = se` as proposed above (shrinks a 2-SE effect to 1 SE), and (b) a
stricter `prior_sd = se / 2` (shrinks the same 2-SE effect to 0.5 SE — half again as skeptical).
Both are printed side by side on the same run, at the same gate's own `s` and `k`; a decision
that flips between (a) and (b) is not published as settled during the transition (see
"Transition," above), and is itself the finding rule 13 would then require restating.

## The ceiling arm — unchanged per gate

**The posterior model reads exactly the same paired frame every gate reads today.** Nothing
about what is *measured* changes — `paired.group_by(["season"]).agg(mean("diff"))`, the same
vector `per_season` returns as `seasons["gain"]`, is the input to `delta_s` above precisely as
it is the input to `summarise`'s sign-and-interval statistics now. Each gate's own declared
ceiling arm stays whatever ADR-0019 and S6 already name it — the foresight ceiling for the
draft and weekly gates, the market's own repricing for #364's game-level bar (declared,
"The ceiling arm, declared," above) — and NOT-RUNNABLE still fires on a missing or
insufficient ceiling ahead of the posterior exactly as it fires ahead of the sign-and-interval
rule today. This ticket changes how the k season numbers are turned into a decision; it does
not change what produces them, what bounds them, or which arm bounds them.

## Exclusions (rule 11)

- No change to what counts as a season, an arm, a cluster, or any exclusion a gate's own
  pre-registration already names — #364's exclusions on parlays, alternate lines and staking
  rules; #305's on choosing the clustering, seed, rows or metric after the sign is seen — all
  carry over unmodified. This ticket adds no new estimand to any gate; it changes how an
  existing estimand's evidence is read.
- **The switching cost is not a staking rule**, restated for this ticket's own action mapping
  exactly as #364's own exclusions state it for the utility it supplies: it sizes a decision
  between publishing and not publishing a model, never a bet.
- **No claim of calibration is implied by using a posterior.** A Bayesian model is not exempt
  from being wrong about its own uncertainty; the SBC and null-size checks above are the
  precondition for trusting any number this section produces, not a formality after the fact —
  a model that fails either is excluded from deciding anything until it is fixed, named here so
  a future run cannot skip the check because the model "is Bayesian."
- Not read before a candidate gate's own ceiling is measured (S6, unchanged) and not read
  before the SBC and null-size checks above have themselves been run and passed — a `NOT-
  RUNNABLE`-shaped exclusion for the method itself, distinct from any one gate's data.

## The `hub.declare` entries

**No line is added to `src/` by this ticket**, for the same reason #364's own section states:
`declare.declarations()` walks module-level assignments in real source, and a declaration
exists where a constant is *used*, not where it is discussed. What is pre-registered now is
the form the eventual declarations take, so an implementer does not have to re-derive it from
this section or from the audit finding alone:

- `PRIOR_SD` (or its eventual per-gate name) is **`chosen`**, set per gate to `s / sqrt(k)` at
  that gate's own published season-clustered figures;
- `TAU_SCALE` is **`chosen`**, set per gate to that gate's own `s`;
- `SWITCHING_COST` is **`chosen`**, proposed at `0.0` pending #364's own dollar figure, in the
  EV units #364's utility declares.

All three read as `chosen`, never `fitted`: none carries a confidence interval or a write-up
of a measurement, per ADR-0006's own line between the two — a `prior_sd` derived from a
published `s` is still a decision about how skeptical to be, not a measurement of skepticism.
When the call site exists, moving any of the three moves `fitted_digest`, exactly as moving
`TALENT_CV` or `FLEX_SHARES` does today.

## What published number moves (rule 13)

**None, until the transition ends.** During the transition (above), the posterior's outputs
are additive — printed beside the existing verdict word — and nothing in `README.md`, any ADR,
or `docs/track-record.md` is edited by this section landing. **When the transition ends** —
the record re-run under #358 and #311, the old word retired by its own dated decision — **every
verdict this repo currently publishes as a sign-and-interval word moves to an expected-utility
action**, and `docs/track-record.md`'s own weekly-published verdicts move with it. Naming that
now, before the transition starts, is what lets the eventual retirement ticket know what it
owes rule 13 rather than discovering the scope of the restatement after the fact — the same
mistake rule 13's own incidents were about.

## Sequencing

**Before phase-2 steps 4 and 5** (the score model, the distributional projection) — #375's own
issue body states this, and it is restated here because this document is what a future reader
checks sequencing against: a model gated by the rule S1 already fixed gets deleted, which
`hub.exhibits` already holds two instances of, and gating the next two phase-2 steps by a rule
this section is about to replace would be building on ground already known to move.
**Blocked by #357 — closed 2026-09-21, so the edge is discharged.** Nothing in this section is
waiting on it; the block existed only because the posterior gate reads the t-interval S1
installed, and that landed before this section was written. **Re-running the record under the
new rule is #358's and #311's work joined, not this ticket's** — this section pre-registers the
rule; it does not run it.

## PROPOSED — what is already decided, and what would reopen it

**Nothing is adopted yet.** Unlike #364, no maintainer `ADOPTED:` comment has chosen among the
model's own open axes, because ADR-0024's own test — can the alternatives be placed on one axis
and reported in one table? — fails here for the same reason it failed #364's three options:
the prior scale, the switching cost's form and the fitting engine are three separate objects,
not three points on one axis, so a maintainer decision is owed on each rather than a
sensitivity table across all three at once.

**Candidate 1 — the prior scale (`prior_sd`, and with it `tau_scale`).**
(a) *Recommended.* `prior_sd = s / sqrt(k)`, `tau_scale = s`, both off each gate's own
`s`/`k` — grounded in this repo's own published noise, no free parameter to tune, and the
"shrinks a 2-SE effect by half" property is checkable by anyone reading the number. **Named
at adoption (2026-09-21) for what it is: a unit-information prior, data-informed through
`s`.** It takes its scale from the same gate's published between-season noise, so it is not
independent of the data the posterior fits, and nobody should later read it as if it were;
what it does not do is take its *location* or its scale from the measured effects the prior
will judge — that is (c), rejected below.
(b) A single fixed value across every gate (e.g. `prior_sd = 1`, in whatever unit) — rejected:
arbitrary across gates whose `s` differ by an order of magnitude (7.34 vs 0.382), and "skeptical"
would stop meaning anything a reader could check.
(c) A prior fit from the cross-gate spread of measured effects — rejected under rule 1: choosing
a prior's scale from the very data distribution the prior will judge is rule 1's incident in a
new instrument, peeking at the answer before writing the rule.

**Candidate 2 — the switching cost's form.**
(a) *Recommended, as the pre-registered default until #364 resolves.* `chosen(0.0)` — the
strictest possible transition test, and the one that needs no further decision to start
computing sensible numbers the day #375's model exists.
(b) Pegged directly to #364's own eventual dollar figure, converted into these EV units —
correct once that figure exists, and the natural successor to (a).
(c) A data-free operational estimate of what re-deploying a model actually costs (engineering
time, the scoped-adoption overhead #305's own pre-registration names) — plausible, but nobody
has priced it, and pricing it is its own measurement this ticket does not do.

**Candidate 3 — Stan vs NumPyro.**
*Recommended:* NumPyro, for the reasons in "The fitting" above — it is already the repo's own
declared (if unused) `bayes` extra, needs no compiled binary in an ephemeral worktree, and
composes with the existing numpy-based harness style. Stan is not rejected on any technical
ground; it is the road not taken because this repo already chose the other one and never used
it.

**What would reopen this.** Evidence that the SBC or the null-size check (rule 16, above) fails
at the constants proposed here — a mis-ranked SBC histogram, or a null `ADOPT` rate that clears
`ALPHA` — would force revisiting `prior_sd`/`tau_scale` before anything built on this section is
trusted, whatever the maintainer's `ADOPTED:` comment says. Short of that, a maintainer
`ADOPTED:` comment naming a different choice among the three candidates above reopens exactly
the axis it names and none of the others, the same granularity #364's own closing section
used.

---

# Pre-registered 2026-10-05: gate horizon — null and power over cluster count, and what within-season rows buy (#388)

**Written and committed before any number in this section's tables exists.** The commit that
adds this text adds the harness (`scripts/gate_horizon.py`, tests in
`tests/unit/test_gate_horizon.py`) and nothing it computed: the table, the controls' results and
the answer to #381 are filled in by a later commit under *Results*, below, and the design above
that heading is not edited after it. ADR-0024: the alternatives sit on one axis (cluster count),
so the table is the deliverable; ADR-0007: a measurement that steers the product is committed
code. **This changes no rule and no constant that reaches a published number**, spends no reading
and consumes no ledger entry; freeze #326 is not touched.

## The question, and the axis

The combined rule's `k` is **seasons**. Every gate clusters on `SEASON_CLUSTER`; `summarise`
bootstraps over per-season means and `per_season` groups the same way. Scored weeks in 2026 add
no cluster. They shrink that one season's within-season SE, which decides whether the season
resolves or **ties**, so weeks act on the verdict only through the tie mechanism. ADR-0019's
rule-16 table read null 0.019 / power 0.19 (weekly, δ=0.3) and 0.011 / 0.032 (draft, δ=2.0) at
k=4; nothing apportions that gap between **k=4** and **the tie mechanism**, and #381 has to decide
the second without knowing how much of the gap it owns.

## The simulated generating process

Per trial, for `k` seasons and each season `s`:

```
true season effect   μ_s            ~ N(δ, τ²)             between-season spread
cluster mean         x_{s,c}        ~ N(μ_s, σ_row² / r)   c = 1..m within-season clusters
```

`m` clusters a season (rosters for the weekly path, rooms for the draft path) and `r` rows
averaged into each cluster (scored weeks for the weekly path, 1 for the draft path). A trial's
paired frame is one row per cluster carrying that cluster's mean, and goes through the shipped
`experiment.gate(paired, cluster=SEASON_CLUSTER, within=("unit",), ...)` — `summarise`,
`per_season`, `_disposition` and `_verdict` as they stand on `main`, `BOOTSTRAP = 4000` as
shipped (not the precedent script's 200: the tie test reads a bootstrap SE, and 200 draws leave
it 5% wrong). Collapsing a balanced cluster to its mean is not an approximation: it is the first
thing `summarise` and `per_season` do to the rows, and a unit test holds the row-level and the
collapsed frame to the same verdict and the same SEs.

**Ceiling.** Held non-binding (`rule16_combined_power.py`'s precedent): this measures the
interval half and the tie-aware every-season half, not stage 2. A binding ceiling would make every
trial NOT-RUNNABLE and the table would read 0 whatever the inputs. Stage 2 passes at 9.7× for the
weekly gate and 1.99× for the draft gate at k=4 (ADR-0019, #376/#378) and only improves with k, so
holding it open does not flatter either gate's *runnability*, which this table does not speak to.

**Estimated from history, not assumed.** The precedent script set the within-season spread equal
to the between-season `s`; that is the stand-in this ticket exists to replace. Each constant
below is computed by `gate_horizon.estimates()` (`--estimates` prints it), from numbers already
published:

*Weekly* (`state/gate-width.json`, #382's re-run of the #378 recipe, the only run that recorded
per-season SEs; m = 20 rosters a season; scored weeks 13 / 13 / 11 / 13, read off the gains'
denominators). Per-season gains −1.1198, −1.4479, −0.2587, −1.7614; SEs 0.4088, 0.4474, 0.5428,
0.5457.

- A bootstrap SE over `m` cluster means is `s·√((m−1)/m)/√m`, so the cluster SD is
  `s = se·m/√(m−1)`: 1.876, 2.053, 2.490, 2.504.
- Cluster variance is `σ_row²/r`, so `σ_row² = s²·r`, pooled by the mean over the four seasons:
  **σ_row² = 62.56 (σ_row = 7.91 points per roster-week)**.
- Is there a roster component beyond week noise? From the frozen frame
  `data/processed/gate/weekly_post248.parquet` (40 rosters × 11–13 weeks, read through code, a
  summary only): the between-roster variance component, disattenuated by the within-roster week
  variance over the weeks, is −1.98, 0.45, 0.61, −0.11 across the four seasons (mean −0.26) — zero
  within its own noise — and the lag-1 autocorrelation of a roster's weekly diffs is −0.08, −0.04,
  −0.09, −0.03. So the model is **row-iid with no roster component**, which is the claim that makes
  weeks the thing that buys precision: a season's SE is `σ_row/√(m·r)`.
- Between-season variance, **disattenuated** (the shape method.md's *Noted twice* records, now a
  third instance: a dispersion fitted on observed variance absorbs the sampling noise of the thing
  it is fitted on): `τ² = var(gains, ddof=1) − mean(s²/m) = 0.4193 − 0.2525 = 0.1668`,
  **τ = 0.408**.

*Draft* (the gains are #376's hold-out run, ADR-0019: −15.35, −8.41, −5.99, −19.86; the ledger's
draft entries carry no SE, so the within-room spread is read from the one frozen draft frame with
per-room rows, `p245_shipped_seed0.parquet`, 20 rooms × 4 seasons: room SDs 13.671, 14.062,
13.005, 9.558). **σ_row² = mean(SD²) = 161.28 (σ_row = 12.70 per team-game)**;
`τ² = 40.47 − 161.28/20 = 32.41`, **τ = 5.69**. The p245 frame is a different run from #376's; its
SEs (2.1–3.1) bracket the one bound #376 itself gives (2024's tie at |−5.99| means its SE exceeds
3.0). That mismatch is named here and carried into the sensitivity table below, not argued away.

**Not in the model, stated:** season-to-season differences in σ_row (the four weekly SEs differ by
a factor of 1.33, which is what 20 clusters' sampling variation of an SD predicts, so one common
σ_row is used), correlation between the arm's gain and a season's noise, and any non-normality.

## The grid

| axis | values | why |
|---|---|---|
| k (seasons) | 4, 5, 6, 7, 8 | the ticket |
| weekly within-season rows | 220 (20 rosters × 11 weeks), 260 (× 13), 280 (× 14) | 11 and 13 are the scored weeks the four held-out seasons actually had (2024 had 11); 14 is `REG_SEASON_WEEKS`, a full season |
| weekly δ | 0.3, 0.5 (and 0 for the null) | the ticket; points per roster-week |
| draft within-season rows | 20, 40, 80 rooms | 20 is the observed `--drafts` default and the only value any run has had; the draft has no partial season, so its "rows" are rooms, a parameter of the harness, and the larger two say what more rooms would buy |
| draft δ | 2.0 (and 0 for the null) | ADR-0019's own row |
| the fifth season | k=5, seasons 1–4 at 13 weeks, the fifth at **4** weeks (2026 as of this writing) or at **14** | what a complete 2026 buys, against what it has now |
| τ sensitivity | 0× and 2× the estimate, at the observed rows, k = 4, 6, 8, both paths | τ rests on four seasons and is the least certain input; the table must not rest on it unexamined |

## What each column and share means (defined now, read later)

- **tie rate** — the mean over trials of (seasons whose disposition is `tie`) / k, at the first
  nonzero δ of the row. A "P(≥1 tie)" column is also kept: ADR-0019's mechanism is that one tie
  vetoes both directions.
- **null ADOPT** — the fraction of trials at δ=0 on which `gate` returns ADOPT.
- **power δ** — the same at the stated δ. **Modal verdict** — the most frequent of
  ADOPT / REMOVE / SHOW / NOT-RUNNABLE at the first nonzero δ, with its frequency.
- **Three rules on the same frames.** On every trial the one `GateRun` is read three ways: the
  **shipped** rule (the verdict `gate` returns); the **pre-#335** rule (the same summary with the
  seasons frame stripped of `se`/`m`, which `_season_records` documents as reading the sign alone
  — equal, as a unit test holds, to raising `TIE_MIN_CLUSTERS` past any cluster count); and the
  **interval alone** (`t_lo > 0`, the half #357 made a distributional claim). Per trial the
  three are nested: shipped ⊆ pre-#335 ⊆ interval-alone on ADOPT.
- **The gap and its owners, pre-registered for #381.** Gap `G = 0.80 − P(shipped)` at the cell
  (0.80 is `experiment.POWER`). The **tie mechanism** owns `P(pre-#335) − P(shipped)` at the low
  end — what #335 added over the sign test — and `P(interval alone) − P(shipped)` at the high
  end, which is the most any tie rule can recover, up to dropping the every-season half altogether
  (#381's candidate E). **k owns** `0.80 − P(interval alone)`, the power a gate at this k does not
  have even with no every-season half and no tie to veto it. Shares are those over `G`; the tie
  mechanism's is reported as the interval between its low and high end.

## Trials, standard errors, seeds

A rate's SE is `√(p(1−p)/N) ≤ 0.5/√N`. **The table claims a difference between two rates only when
it is at least 0.025; everything under that is called not distinguished.** The SE target is a
tenth of that, 0.0025, so **N = 40,000 trials per cell** (worst case 0.0025 at p = 0.5, 0.0018 at
p = 0.1, 0.0006 at the null sizes near 0.02). The sensitivity table claims only differences of
0.05 or more and runs **N = 20,000** (worst case 0.0035 < 0.005). The leverage study's 4,000-trial
re-run is the precedent for not trusting a first pass, so a **2,000-trial pass is run first** over
the whole grid and its rates must sit within 3.5 of their own SEs of the 40,000-trial pass; if any
does not, the SE does not scale as claimed and the full pass is not the evidence.

Seeds: `numpy.random.SeedSequence([388, cell_index])` per cell, spawned one child per 2,000-trial
chunk; each trial's bootstrap seed is drawn from that same stream. A cell's counts therefore
depend on nothing but its index in the grid, not on the worker count (a unit test holds it).
Parallelism is capped at 6 worker processes, one polars thread each.

## The three controls (rule 18), each of which must pass or the table is not published

1. **A planted δ far above range reaches power ≈ 1.** Weekly δ = 20 (≈ 40 season SEs) at k = 8 and
   k = 4, draft δ = 60 at k = 8, at 40,000 trials: shipped ADOPT rate ≥ 0.99 and zero
   NOT-RUNNABLE. A rate that cannot get there is a harness that is not driving the rule it claims to.
2. **δ = 0 reproduces the null column, and its sign half matches 2^-k.** At every k = 4..8 on both
   paths at the observed rows: the fraction of trials in which every season's gain is positive
   equals `2^-k` within 4 SEs (the same for every gain negative), and the shipped null ADOPT rate
   is at most `ALPHA` = 0.05 (and, nested, at most the pre-#335 and interval-alone null rates).
   A null that does not is a harness bug, not a finding.
3. **A run against the pre-#335 tie handling differs from the shipped one in the direction #335
   predicted** — "a tie-aware rule can only be *stricter* than the interval alone, never more
   permissive". At the weekly cell k=4, 260 rows, δ=0.3 and 0.5, and the draft cell k=4, 20 rooms,
   δ=2.0, at 40,000 trials: (i) **zero** frames on which the shipped rule ADOPTs (or REMOVEs) and
   the pre-#335 rule does not; (ii) the pre-#335 ADOPT rate exceeds the shipped one by at least 4
   SEs of the paired difference (`√d/N`, `d` the discordant frames); (iii) the pre-#335 reading has no
   ties by construction (a sign test has none) while the shipped tie rate is positive. A control that cannot tell the
   two rules apart is measuring neither.

## What this cannot do

It does not say whether any arm beats any incumbent: δ is planted, never read. It estimates a
between-season *variance* and a tie *rate* from four seasons, and four seasons estimate a variance
badly — the sensitivity table is that admission in numbers. It does not name a horizon for Audit V
(#326's dispatch is about artifacts existing, not decidability), does not decide #381, and
**adopts nothing**: where the results below contradict a figure ADR-0019 published, rule 13 says
the figure moves, and that is a separate edit by whoever owns the ADR, flagged and not made here.

## What would reopen this

A control that fails; a 2,000-trial pass whose rates do not sit within their SEs of the full pass;
or a proposal to change the rule being measured (then it is a decision, ADR-0024, not a table).

## Results

Run 2026-10-06 against the design above, which `34cbb48` committed first and which was not edited.
Harness `scripts/gate_horizon.py` at that commit, 6 workers, `BOOTSTRAP = 4000`, seeds
`SeedSequence([388, cell_index])`. Raw counts are not committed (`$TMPDIR/388-*`); re-running the
commands in the harness's docstring reproduces them to the bit.

### Estimates (printed by `--estimates`, matching the arithmetic above)

weekly τ = 0.4083, σ_row² = 62.564 (σ_row 7.910); draft τ = 5.6912, σ_row² = 161.280 (σ_row 12.700).

### The three controls: all pass, at 40,000 trials a cell

1. **Planted δ.** Weekly δ=20 at k=8 and k=4, draft δ=60 at k=8: shipped ADOPT rate **1.00000** in
   all three, NOT-RUNNABLE **0**.
2. **δ = 0.** At every k=4..8 on both paths, the all-seasons-positive and all-negative fractions sit
   within 4 SEs of 2^-k (worst: weekly k=4 all-negative 0.0593 against 0.0625, 2.6 SEs; 20 of
   20 pairs inside 4). The shipped null
   ADOPT rate is at most 0.0013 (draft k=4), under ALPHA = 0.05, and nested under the pre-#335
   (0.0338 at draft k=4, 0.0341 weekly k=4) and interval-alone (0.0344, 0.0347) null rates.
3. **Pre-#335 against shipped.** Weekly k=4, 260 rows, δ=0.3 / 0.5 and draft k=4, 20 rooms, δ=2.0:
   zero frames on which shipped ADOPTs or REMOVEs and the pre-#335 rule does not (0 / 0 / 0
   violations); pre-#335 ADOPT exceeds shipped by 0.1345 / 0.2505 / 0.0822 against 4 paired SEs of
   0.0073 / 0.0100 / 0.0057 (5,381 / 10,020 / 3,287 discordant frames); shipped ties per trial
   3.23 / 2.94 / 2.33 of 4 seasons, pre-#335 none by construction.

### Trials and standard errors

**40,000 trials per cell in the table (81 cells), 20,000 in the sensitivity table (24 cells), 40,000
in the controls (16 cells).** Worst SE of any rate in the table is **0.0025** (p = 0.5); the table
claims differences only at 0.025 or more, so the SE is a tenth of the smallest claimed difference
as pre-registered. **Everything below 0.025 is not distinguished**, which includes every shipped
power in the weekly table and every cell-to-cell change in a tie rate between row counts.
The 2,000-trial pass over the same 81 cells (4 min) agrees with the 40,000-trial pass on all 567
compared rates, worst deviation 2.9 SEs, none beyond 3.5. One reading choice: the SE used to
compare them is the 40,000-trial rate's, because at a rate of 0 in 2,000 trials a plug-in SE is 0
and any nonzero full-pass rate (0.0005) then reads as an infinite deviation; with the plug-in
SE 16 of 567 comparisons exceed 3.5, all of them such rate-near-0 or rate-near-1 cells
(ADOPT 0 against 0.0005, SHOW or any-tie 1 against 0.9995) plus one draft k=8 sign-rule rate
(0.0155 against 0.0261, 3.7 SEs). Run time: table 5,532 s (92 min), controls 702 s, sensitivity
613 s, on 6 workers; about 2 hours in all.

### The table

**Weekly path** (δ in points per roster-week); rows are rosters × scored weeks.

| k (seasons) | within-season rows | tie rate | null ADOPT | power δ=0.3 | power δ=0.5 | modal verdict (δ=0.3) |
|---|---|---|---|---|---|---|
| 4 | 220 (20×11) | 0.824 | 0.0000 | 0.0005 | 0.0024 | SHOW (1.00) |
| 4 | 260 (20×13) | 0.807 | 0.0000 | 0.0007 | 0.0043 | SHOW (1.00) |
| 4 | 280 (20×14) | 0.799 | 0.0001 | 0.0010 | 0.0046 | SHOW (1.00) |
| 5 | 220 (20×11) | 0.825 | 0.0000 | 0.0001 | 0.0006 | SHOW (1.00) |
| 5 | 260 (20×13) | 0.809 | 0.0000 | 0.0002 | 0.0014 | SHOW (1.00) |
| 5 | 280 (20×14) | 0.799 | 0.0000 | 0.0003 | 0.0011 | SHOW (1.00) |
| 6 | 220 (20×11) | 0.826 | 0.0000 | 0.0000 | 0.0001 | SHOW (1.00) |
| 6 | 260 (20×13) | 0.806 | 0.0000 | 0.0000 | 0.0004 | SHOW (1.00) |
| 6 | 280 (20×14) | 0.798 | 0.0000 | 0.0000 | 0.0003 | SHOW (1.00) |
| 7 | 220 (20×11) | 0.826 | 0.0000 | 0.0000 | 0.0001 | SHOW (1.00) |
| 7 | 260 (20×13) | 0.807 | 0.0000 | 0.0000 | 0.0001 | SHOW (1.00) |
| 7 | 280 (20×14) | 0.798 | 0.0000 | 0.0000 | 0.0001 | SHOW (1.00) |
| 8 | 220 (20×11) | 0.825 | 0.0000 | 0.0000 | 0.0000 | SHOW (1.00) |
| 8 | 260 (20×13) | 0.809 | 0.0000 | 0.0000 | 0.0000 | SHOW (1.00) |
| 8 | 280 (20×14) | 0.799 | 0.0000 | 0.0000 | 0.0001 | SHOW (1.00) |

**Draft path** (δ = 2.0 points per team-game); rows are rooms.

| k (seasons) | within-season rows | tie rate | null ADOPT | power δ=2.0 | modal verdict (δ=2.0) |
|---|---|---|---|---|---|
| 4 | 20 | 0.581 | 0.0014 | 0.0076 | SHOW (0.99) |
| 4 | 40 | 0.462 | 0.0040 | 0.0180 | SHOW (0.98) |
| 4 | 80 | 0.348 | 0.0094 | 0.0379 | SHOW (0.96) |
| 5 | 20 | 0.582 | 0.0003 | 0.0020 | SHOW (1.00) |
| 5 | 40 | 0.463 | 0.0012 | 0.0068 | SHOW (0.99) |
| 5 | 80 | 0.348 | 0.0031 | 0.0175 | SHOW (0.98) |
| 6 | 20 | 0.583 | 0.0001 | 0.0008 | SHOW (1.00) |
| 6 | 40 | 0.463 | 0.0003 | 0.0027 | SHOW (1.00) |
| 6 | 80 | 0.349 | 0.0007 | 0.0074 | SHOW (0.99) |
| 7 | 20 | 0.583 | 0.0000 | 0.0001 | SHOW (1.00) |
| 7 | 40 | 0.462 | 0.0001 | 0.0011 | SHOW (1.00) |
| 7 | 80 | 0.349 | 0.0003 | 0.0034 | SHOW (1.00) |
| 8 | 20 | 0.583 | 0.0000 | 0.0002 | SHOW (1.00) |
| 8 | 40 | 0.463 | 0.0000 | 0.0003 | SHOW (1.00) |
| 8 | 80 | 0.349 | 0.0001 | 0.0011 | SHOW (1.00) |

**The fifth season, through 4 weeks or all 14** (k = 5; seasons 1-4 at 13 weeks).

| 2026 weeks scored | tie rate | P(≥1 tie) δ=0.3 | null ADOPT | power δ=0.3 | power δ=0.5 | modal verdict (δ=0.3) |
|---|---|---|---|---|---|---|
| 4 | 0.824 | 1.000 | 0.0000 | 0.0001 | 0.0004 | SHOW (1.00) |
| 14 | 0.807 | 1.000 | 0.0000 | 0.0001 | 0.0012 | SHOW (1.00) |

**Where the gap to 80% power sits.** Weekly δ=0.3, then δ=0.5, then draft δ=2.0.

| k | rows | P(≥1 tie) | P(interval alone) | P(pre-#335) | P(shipped) | gap to 0.80 | tie mechanism owns (pre-#335 → alone) | k owns |
|---|---|---|---|---|---|---|---|---|
| *weekly δ=0.3* | | | | | | | | |
| 4 | 220 | 0.999 | 0.1241 | 0.1226 | 0.0005 | 0.800 | 15% – 15% | 85% |
| 4 | 260 | 0.999 | 0.1379 | 0.1356 | 0.0007 | 0.799 | 17% – 17% | 83% |
| 4 | 280 | 0.998 | 0.1409 | 0.1389 | 0.0010 | 0.799 | 17% – 18% | 82% |
| 5 | 220 | 1.000 | 0.1537 | 0.1143 | 0.0001 | 0.800 | 14% – 19% | 81% |
| 5 | 260 | 1.000 | 0.1636 | 0.1237 | 0.0002 | 0.800 | 15% – 20% | 80% |
| 5 | 280 | 0.999 | 0.1659 | 0.1266 | 0.0003 | 0.800 | 16% – 21% | 79% |
| 6 | 220 | 1.000 | 0.1763 | 0.0870 | 0.0000 | 0.800 | 11% – 22% | 78% |
| 6 | 260 | 1.000 | 0.1916 | 0.0943 | 0.0000 | 0.800 | 12% – 24% | 76% |
| 6 | 280 | 1.000 | 0.1961 | 0.0959 | 0.0000 | 0.800 | 12% – 25% | 75% |
| 7 | 220 | 1.000 | 0.2011 | 0.0606 | 0.0000 | 0.800 | 8% – 25% | 75% |
| 7 | 260 | 1.000 | 0.2175 | 0.0655 | 0.0000 | 0.800 | 8% – 27% | 73% |
| 7 | 280 | 1.000 | 0.2243 | 0.0681 | 0.0000 | 0.800 | 9% – 28% | 72% |
| 8 | 220 | 1.000 | 0.2251 | 0.0392 | 0.0000 | 0.800 | 5% – 28% | 72% |
| 8 | 260 | 1.000 | 0.2412 | 0.0450 | 0.0000 | 0.800 | 6% – 30% | 70% |
| 8 | 280 | 1.000 | 0.2544 | 0.0477 | 0.0000 | 0.800 | 6% – 32% | 68% |
| *weekly δ=0.5* | | | | | | | | |
| 4 | 220 | 0.997 | 0.2435 | 0.2404 | 0.0024 | 0.798 | 30% – 30% | 70% |
| 4 | 260 | 0.995 | 0.2601 | 0.2571 | 0.0043 | 0.796 | 32% – 32% | 68% |
| 4 | 280 | 0.994 | 0.2673 | 0.2639 | 0.0046 | 0.795 | 33% – 33% | 67% |
| 5 | 220 | 0.999 | 0.3104 | 0.2401 | 0.0006 | 0.799 | 30% – 39% | 61% |
| 5 | 260 | 0.998 | 0.3352 | 0.2635 | 0.0014 | 0.799 | 33% – 42% | 58% |
| 5 | 280 | 0.999 | 0.3468 | 0.2708 | 0.0011 | 0.799 | 34% – 43% | 57% |
| 6 | 220 | 1.000 | 0.3745 | 0.2052 | 0.0001 | 0.800 | 26% – 47% | 53% |
| 6 | 260 | 0.999 | 0.4000 | 0.2216 | 0.0004 | 0.800 | 28% – 50% | 50% |
| 6 | 280 | 1.000 | 0.4185 | 0.2343 | 0.0003 | 0.800 | 29% – 52% | 48% |
| 7 | 220 | 1.000 | 0.4388 | 0.1671 | 0.0001 | 0.800 | 21% – 55% | 45% |
| 7 | 260 | 1.000 | 0.4724 | 0.1814 | 0.0001 | 0.800 | 23% – 59% | 41% |
| 7 | 280 | 1.000 | 0.4860 | 0.1858 | 0.0001 | 0.800 | 23% – 61% | 39% |
| 8 | 220 | 1.000 | 0.4916 | 0.1249 | 0.0000 | 0.800 | 16% – 61% | 39% |
| 8 | 260 | 1.000 | 0.5306 | 0.1433 | 0.0000 | 0.800 | 18% – 66% | 34% |
| 8 | 280 | 1.000 | 0.5497 | 0.1511 | 0.0001 | 0.800 | 19% – 69% | 31% |
| *draft δ=2.0* | | | | | | | | |
| 4 | 20 | 0.970 | 0.0922 | 0.0907 | 0.0076 | 0.792 | 10% – 11% | 89% |
| 4 | 40 | 0.919 | 0.0968 | 0.0956 | 0.0180 | 0.782 | 10% – 10% | 90% |
| 4 | 80 | 0.818 | 0.0993 | 0.0979 | 0.0379 | 0.762 | 8% – 8% | 92% |
| 5 | 20 | 0.988 | 0.1056 | 0.0772 | 0.0020 | 0.798 | 9% – 13% | 87% |
| 5 | 40 | 0.956 | 0.1078 | 0.0801 | 0.0068 | 0.793 | 9% – 13% | 87% |
| 5 | 80 | 0.884 | 0.1134 | 0.0837 | 0.0175 | 0.782 | 8% – 12% | 88% |
| 6 | 20 | 0.995 | 0.1145 | 0.0562 | 0.0008 | 0.799 | 7% – 14% | 86% |
| 6 | 40 | 0.976 | 0.1249 | 0.0583 | 0.0027 | 0.797 | 7% – 15% | 85% |
| 6 | 80 | 0.925 | 0.1294 | 0.0606 | 0.0074 | 0.793 | 7% – 15% | 85% |
| 7 | 20 | 0.998 | 0.1282 | 0.0343 | 0.0001 | 0.800 | 4% – 16% | 84% |
| 7 | 40 | 0.987 | 0.1377 | 0.0383 | 0.0011 | 0.799 | 5% – 17% | 83% |
| 7 | 80 | 0.951 | 0.1406 | 0.0390 | 0.0034 | 0.797 | 4% – 17% | 83% |
| 8 | 20 | 0.999 | 0.1415 | 0.0241 | 0.0002 | 0.800 | 3% – 18% | 82% |
| 8 | 40 | 0.993 | 0.1505 | 0.0236 | 0.0003 | 0.800 | 3% – 19% | 81% |
| 8 | 80 | 0.968 | 0.1547 | 0.0261 | 0.0011 | 0.799 | 3% – 19% | 81% |

**Sensitivity to the between-season SD** (observed rows; τ as a multiple of its estimate).

| path | k | τ × | null ADOPT | power | P(≥1 tie) |
|---|---|---|---|---|---|
| weekly δ=0.3 | 4 | 0 | 0.0000 | 0.0002 | 1.000 |
| weekly δ=0.3 | 4 | 2 | 0.0006 | 0.0042 | 0.986 |
| weekly δ=0.3 | 6 | 0 | 0.0000 | 0.0000 | 1.000 |
| weekly δ=0.3 | 6 | 2 | 0.0001 | 0.0003 | 0.998 |
| weekly δ=0.3 | 8 | 0 | 0.0000 | 0.0000 | 1.000 |
| weekly δ=0.3 | 8 | 2 | 0.0000 | 0.0001 | 1.000 |
| draft δ=2.0 | 4 | 0 | 0.0000 | 0.0002 | 1.000 |
| draft δ=2.0 | 4 | 2 | 0.0092 | 0.0216 | 0.823 |
| draft δ=2.0 | 6 | 0 | 0.0000 | 0.0001 | 1.000 |
| draft δ=2.0 | 6 | 2 | 0.0010 | 0.0040 | 0.925 |
| draft δ=2.0 | 8 | 0 | 0.0000 | 0.0000 | 1.000 |
| draft δ=2.0 | 8 | 2 | 0.0002 | 0.0006 | 0.971 |


### What it answers for #381

**The tie mechanism owns a minority of the k=4 gap, and a full 2026 season barely changes that
share: not distinguished at weekly δ=0.3, about five points higher at δ=0.5.** Gap `G = 0.80 − P(shipped)` is 0.80 in every weekly cell, because the shipped rule's power
is under 0.005 everywhere on the weekly grid; the gap is in effect "all of it".

- *Weekly, k=4, 260 rows.* Interval alone reaches **0.138** at δ=0.3 and **0.260** at δ=0.5; the
  pre-#335 sign rule 0.136 and 0.257; shipped 0.0007 and 0.0043. So the tie mechanism owns
  **17%** of the gap at δ=0.3 (0.136 of 0.799) and **32%** at δ=0.5; **k owns the other 83% and
  68%**, power the gate does not have even with no every-season half and no tie to veto it.
- *Draft, k=4, 20 rooms, δ=2.0.* Interval alone 0.092, pre-#335 0.091, shipped 0.0076: the tie
  mechanism owns **10–11%**, k owns **89%**.
- *The mechanism is not subtle where it acts.* A season ties at about 0.8 of weekly seasons and
  0.58 of draft seasons, so P(≥1 tie) is 0.97 to 1.00 at every cell: one tie vetoes both directions,
  and the shipped rule reads SHOW in 96–100% of trials at the planted δ. Its power is
  not "low", it is nil. The interval half, with no tie rule at all, is itself 0.14 / 0.26 / 0.09
  at k=4, so even a perfect tie rule recovers at most that much.
- *Adding seasons moves the mechanism's share up, not down.* At weekly δ=0.5 the high end of its
  share goes from 32% (k=4) to 66% (k=8) because the interval alone climbs to 0.53 while the
  shipped rule stays at zero; at δ=0.3 the high end goes 17% to 30%. k owns the larger
  part at every k at δ=0.3 (68-85%) and, at δ=0.5, down to k=6 (53% at 220 rows, 50% at 260); from
  k=7 the mechanism's high end is the larger (55-61% against 39-45% at 220-260 rows).
  More seasons do not rescue the shipped rule: its power stays under 0.005 at every k and both
  δ on the weekly grid, and the table cannot order those rates (every difference between them is
  under the 0.025 resolution). *Restated 2026-10-06 (#418): this bullet first read "its power
  falls with k (0.0043 at k=4 to 0.0000 at k=8 for δ=0.5)", a difference of 0.0043, a tenth of
  the resolution, so the direction is not a finding. What the table does show is that the
  interval alone climbs with k (0.26 at k=4 to 0.53 at k=8 at δ=0.5) while the shipped rule does
  not move.*
- *A full 2026 season.* Seasons 1-4 at 13 weeks and a fifth at 4 weeks versus 14 weeks: tie rate
  0.824 against 0.807 (δ=0.3) and shipped power 0.0001 against 0.0001 (0.0004 against 0.0012 at
  δ=0.5), P(≥1 tie) 1.000 against 1.000. The tie mechanism's share of the gap is **14–18%** at 4
  weeks and **16–21%** at 14 at δ=0.3 (the rates behind it move 0.015 and 0.021, under 0.025:
  **not distinguished**), and **28–36%** against **33–42%** at δ=0.5, where the interval-alone
  rate moves 0.288 to 0.337 (0.049, distinguished). So a full season shifts the shipped rule
  by nothing the table can see, and shifts what a perfect tie rule could recover by a few points
  at the larger δ only. The reason is arithmetic, not a finding about the
  gate: a season's SE is σ_row/√(m·r) = 7.91/√(20·r), 0.49 at 13 weeks and 0.47 at 14, against a
  true season effect of 0.3 with a between-season spread of 0.41; going from 11 to 14 weeks shrinks
  the SE by 11%.
- *What that means for #381, as numbers and nothing adopted.* If the mechanism owned most of
  the gap, its candidates would matter more than the horizon. At k=4 it owns 10–17% at the
  gate's own δ rows (30% at weekly δ=0.5), and the remaining 83–90% is a gap no tie rule can
  close at this k, because the interval alone is under 0.15. **So this season, no tie rule
  recovers the reading, and that is the finding** (the ticket's wording for this branch). The
  one nuance is that the share the mechanism owns grows with k and with δ, so a candidate that
  matters now is not the same as one that would matter at k=8.
- *Restated 2026-10-06 (#418): the baseline, and what did not move.* This section's shares are
  read off #388's estimated process, which #418 found to be the process that reproduces the
  recorded season SEs and the observed tie rate (see the next section); that is the baseline #381
  should read, and the figures above stand on it. The ticket's own framing, "the published
  0.19 / 0.032", is a different baseline: ADR-0019's figures are the rule's power under a
  stand-in process whose weekly within-season SE is about a ninth of the recorded one. Run
  through the same harness, that stand-in gives the tie mechanism **10–11%** (weekly δ=0.3) and
  **6%** (draft δ=2.0) of the gap to 0.80 at k=4 against #388's 17% and 11%, and k **89%** and
  **94%** against 83% and 89%. The answer for #381 (a minority for the mechanism, most of the gap
  to k at k=4) does not depend on which of the two it reads; the *level* of the shipped rule's
  power does, and #381 should quote 0.0008 (weekly) and 0.0075 (draft), not 0.19 and 0.032.
- *Sensitivity to τ, the least certain input.* τ at 0× and 2× the estimate, observed rows, k=4, 6,
  8: shipped power stays under 0.0092 null / 0.022 power on every cell and ties stay at P(≥1 tie)
  0.82–1.00. No τ in that range changes the reading; the table does not rest on τ.

### What contradicts a published figure (rule 13: flagged, not edited; attributed 2026-10-06 under #418)

**Restated 2026-10-06 (#418): attributed.** The paragraph below, as first written, is kept
unedited under this note. It was wrong in two ways that #418's reconciliation corrects: it left
the gap "not attributed" (now apportioned, next), and it compared the draft path's
*interval-alone* power, 0.092, against ADR-0019's 0.032, which is the *combined* rule's power;
the ADR published no interval-alone column, so there was no draft disagreement there. The
weekly path's "0.14 and 0.136 against 0.19" compared the same two different things.

**The reconciliation (#418; no rule, constant or verdict changes).** Four findings, in the order
they were checked.

1. **`scripts/rule16_combined_power.py` re-run on main (`71b2adc`+): it reproduces the ADR to
   the digit**: draft null **0.0113** / power **0.0324**, weekly null **0.0186** / power
   **0.1925**, 10,000 trials, bootstrap 200, seeds 0 / 1. It does not return 0.0000/0.0000.
   The "wrinkle" paragraph in the rule-16 section above (the one measuring #365's cell) was
   true of the script as it stood at `d2e4cbf` and stale 50 minutes later: `51909c2` (the
   same day, 2026-09-21) made the harness hand `gate` a ceiling that cannot bind, and #386
   later moved it to the frame-in `gate` without moving a digit. Dated notes at that paragraph
   and at the ADR record it.
2. **The #363 NOT-RUNNABLE path owns none of the gap.** Both ends of it were measured, 300
   trials of rule-16's own frame at weekly k=4: `ceiling=None` returns NOT-RUNNABLE **300 of
   300** (the 0.0000/0.0000 the "wrinkle" described), a ceiling that cannot bind returns ADOPT
   47 of 300 (0.157, SE 0.021; the 0.1925 of the full run). #388's harness uses the
   non-binding ceiling too, so on this axis the two harnesses are the same rule. Before S6, a
   missing ceiling and a non-binding one read identically, which is why the ADR's
   pre-S6 figures are reproduced by the post-S6 script with the ceiling held open.
3. **The gap is the generating process, almost all of it.** #388's harness, `gate_horizon.py
   --rule16` (the same `run_chunk`, `gate` and `Cell` the table uses, 40,000 trials a cell),
   run from rule-16's process and moved to #388's one factor at a time, k=4 throughout:

| weekly, δ=0.3 | null ADOPT | power | SE | interval alone | ties / season |
|---|---|---|---|---|---|
| 1. rule-16's process (cluster SD = s = 0.382, m=40, bootstrap 200) | 0.0162 | **0.1937** | 0.0020 | 0.257 | 0.180 |
| 2. + the shipped bootstrap, 4000 | 0.0198 | 0.1895 | 0.0020 | 0.254 | 0.180 |
| 3. + m=20 clusters, as #388 has it | 0.0120 | 0.1522 | 0.0018 | 0.254 | 0.247 |
| 4. + the estimated τ (0.408, not 0.382) | 0.0138 | 0.1420 | 0.0017 | 0.228 | 0.238 |
| 5. + the within-season spread estimated from published SEs (σ_row 7.91, 13 weeks; **= #388's cell**) | 0.0001 | **0.0008** | 0.0001 | 0.135 | 0.807 |
| 6. #388's cell at bootstrap 200 | 0.0000 | 0.0009 | 0.0002 | 0.139 | 0.806 |

| draft, δ=2.0 | null ADOPT | power | SE | interval alone | ties / season |
|---|---|---|---|---|---|
| 1. rule-16's process (cluster SD = s = 7.34, m=20, bootstrap 200) | 0.0107 | **0.0350** | 0.0009 | 0.080 | 0.312 |
| 2. + the shipped bootstrap, 4000 | 0.0121 | 0.0348 | 0.0009 | 0.079 | 0.313 |
| 4. + the estimated τ (5.69, not 7.34); m is 20 already | 0.0077 | 0.0306 | 0.0009 | 0.097 | 0.384 |
| 5. + the within-season spread estimated (σ_row 12.70; **= #388's cell**) | 0.0014 | **0.0075** | 0.0004 | 0.093 | 0.581 |
| 6. #388's cell at bootstrap 200 | 0.0014 | 0.0072 | 0.0004 | 0.090 | 0.583 |

   Attributed, in this order of steps (the order matters when factors interact; this is one
   order, not the only one): weekly power falls 0.1929 in all (0.1937 to 0.0008), of which the
   **within-season spread is 0.1412 (73%)**, m=40 to 20 is 0.0373 (19%), the estimated τ
   0.0102 (5%), and the bootstrap 0.0042 (2%, 1.5 SEs of a difference and under the table's 0.025
   resolution: not distinguished). Draft power falls 0.0275, of which the **within-season
   spread is 0.0231 (84%)**, τ 0.0042 (15%), the bootstrap 0.0002 (not distinguished). So
   (a) owns nothing, (b) the generating process owns all of it and within-season spread is
   most of that, and (c) the rest is m (weekly only), τ, and a bootstrap difference too small
   to see. Interactions between the factors were not measured: another order would split the 0.1412
   differently, and only the size of that one step, against 0.04 and 0.01 for the others, is
   what the attribution rests on.
4. **Which process is the right one: #388's.** Rule-16's stand-in set the within-season cluster
   SD equal to the between-season `s`, which at m=40 is a season SE of 0.382/√40 = **0.060**;
   the weekly seasons' own recorded SEs are 0.409 / 0.447 / 0.543 / 0.546 (`state/gate-width.json`),
   seven to nine times wider. The script called its stand-in "conservative"; for weekly it is
   the opposite, since a season it can resolve at 0.06 is a season #388's tie test cannot
   resolve at 0.5. Checked against the record, not only against the SEs: **observed ties are one
   season of four** at both gates (weekly 2024, gain −0.259 against SE 0.543; draft 2024,
   −5.99 inside its own noise; ADR-0019 and `state/gate-width.json`), a per-season rate of 0.25.
   At each process's own prediction **at the mean gain the gate actually observed** (weekly
   −1.15, draft −12.40; step 1 and step 5 above): rule-16's process predicts **0.003** ties per
   weekly season (P(≥1 tie in 4) ≈ 1%: the observed 2024 tie is close to impossible under it),
   #388's **0.378** (P(≤1 tie in 4) ≈ 0.51: consistent); the draft path's two are 0.091 and
   0.136, both consistent with 1 of 4 (the draft per-season SE is not published, which is why
   step 5 there leans on a different run's frame, named in the design). The 0.8 tie rate that
   only #388's harness produces is not a harness artefact, and it is not in tension with the
   observed 0.25 either: 0.8 is the tie rate **at a true effect of δ=0.3**, which sits inside
   the tie band (about 2 × SE ≈ 1.0 per season), while the observed seasons' gains sit far
   outside it. It is what the published SEs imply, and rule-16's 0.18 at the same δ is what its
   stand-in implies.

**What stands, what does not.** ADR-0019's figures stand as exactly what they say they are:
the combined rule's null and power *under the stand-in generating process stated beside them*,
reproduced to the digit today. They do not stand as the rule's power at the gate's own
measured precision: for the weekly gate that is **0.0008 (null 0.0001)**, not 0.19, and for the
draft gate **0.0075 (null 0.0014)**, not 0.032. The null sizes were never in question (all under
`ALPHA`). The draft gate's rule-16 *exemption* is unchanged, and the weekly gate's ADOPT branch
is further out of reach than the table said, not nearer. **The baseline #381 should read is
#388's** (the estimated process), because it reproduces the recorded SEs and the observed tie
rate and the stand-in does not. The apportionment of #381 is not sensitive to the choice: the
tie mechanism owns 10–11% (weekly δ=0.3) and 6% (draft) of the gap to 0.80 under rule-16's
process, against 17% and 11% under #388's, and **k owns 83–94% under both**. Positive control
(rule 18): the harness at rule-16's process (step 1) reproduces the script's own published
numbers, weekly **0.0162 / 0.1937** against **0.0186 / 0.1925** (10,000 trials, SE 0.0013 /
0.0039 for the published ones; null 1.6 SEs, power 0.3) and draft **0.0107 / 0.0350** against
**0.0113 / 0.0324** (0.5 and 1.3 SEs); and at #388's process (step 5) it reproduces #388's own
table, **0.0008** against 0.0007 and **0.0075** against 0.0076, by separately seeded runs.
Run 2026-10-06, `gate_horizon.py --rule16 --trials 40000 --workers 6`, 15 minutes, worst SE of
any reported rate 0.0020; raw counts not committed (`$TMPDIR/418-run`).

**Original paragraph, as first written (2026-10-06, #388), kept as the record of what was
believed before #418:**

ADR-0019's rule-16 table read null 0.019 / power 0.19 (weekly, δ=0.3) and 0.011 / 0.032 (draft,
δ=2.0) at k=4. The shipped-rule figures here are **null 0.0000 / power 0.0007** (weekly, 260 rows)
and **0.0014 / 0.0076** (draft, 20 rooms). The interval-alone and pre-#335 columns are close to
the ADR's order for the weekly path (0.14 and 0.136 against 0.19) and for the draft path (0.092
and 0.091 against 0.032 for the draft power, which this table puts roughly three times higher).
The two sets of figures were produced with different generating processes (the precedent set the
within-season spread equal to the between-season one; this table's is estimated from
published SEs, a season's SE of about 0.5 against a τ of 0.41 on the weekly path), and `rule16_combined_power.py` was not re-run here, so the discrepancy is **not
attributed**. Which of the two processes the ADR should have used is a decision for whoever owns
it.

### Where the design was hard to honour

- **The 2,000-trial agreement test** as written ("within 3.5 of their own SEs") is ill-defined
  at a rate of 0 or 1, where an observed SE is zero; resolved above by using the full pass's rate
  for both SEs, reported with the unadjusted count so the reader can choose.
- **The shipped weekly power is below the table's own resolution everywhere**
  (≤ 0.0046), so the shipped columns cannot be compared across cells under the pre-registered
  0.025 rule; the apportionment, which compares the *interval-alone* and pre-#335 columns
  against shipped, carries the content. That follows from the design and was not worked
  around.
- **The generating process is row-iid with no roster component and one common σ_row**, as
  pre-registered; both are estimated from one weekly run and one draft frame whose SEs bracket but
  do not equal #376's.

---

# Pre-registered 2026-10-06: (C), the verdict over resolved seasons, before it lands (#381)

**Written and committed before any number in this section's tables exists.** Rule 16: a rule's
null size and power are computed before it is adopted into `gate`. The maintainer ADOPTED (C) on
#381 on 2026-10-06, with this computation first and a stop condition (below). This commit adds
the harness extension (`scripts/gate_horizon.py --rule-c`, tests in
`tests/unit/test_gate_horizon.py`) and **does not touch `experiment.gate`**; the results go under
*Results (C)*, and the design above that heading is not edited after it.

## The rule being measured

A season has one **Disposition**: a win, a loss, or an **Abstention** (CONTEXT.md; the code's
field is still `ties`). A season with a win or a loss is **resolved**.

- **ADOPT** — at least one resolved season, *every resolved season a win*, and the t interval
  over all k seasons' pooled gains excludes zero from above.
- **REMOVE** — the mirror: at least one resolved season, every resolved season a loss, the t
  interval excludes zero from below.
- With **no resolved season** the Gate can neither ADOPT nor REMOVE: SHOW, with "0 resolved of
  k, k abstained". It is not NOT-RUNNABLE.
- **No floor beyond one resolved season.** The interval half still pools all k seasons,
  Abstentions included.
- The verdict sentence and the ledger record carry the abstention count.

**What is measured.** `verdict_abstain` (A, as shipped) and `verdict_resolved` (C) in the harness
are the two rules as functions of the one summary and seasons frame that the shipped
`experiment.gate` returns, so each trial reads both on identical frames (a paired comparison)
and nothing about `summarise`, `per_season`, `_disposition` or the t interval is reimplemented.
The only reimplemented part is the four-line conjunction, and a unit test holds the harness's
"ship" column (what `gate` itself returned) equal to the landed rule's function on every trial.
The ceiling is held non-binding, as in #388; VOID and NOT-RUNNABLE do not arise.

## Baseline, grid, trials

**Baseline: #388's estimated process**, which #418 found reproduces the recorded season SEs and
the observed tie rate (weekly τ = 0.4083, σ_row² = 62.564; draft τ = 5.6912, σ_row² = 161.280).
Cells: k = 4..8 at 260 weekly rows (20×13) at δ = 0, 0.3, 0.5, and at 20 draft rooms at δ = 0,
2.0; and at k = 4 the other row counts (weekly 220 and 280; draft 40 and 80 rooms), plus the
three planted-δ controls. **40,000 trials a cell, worst SE 0.0025** (the target; the null cells
are far below it). Seeds `SeedSequence([388, cell_index])` as #388's harness derives them, so the
(A) column is the same draws as a #388-style run of the same grid, not the same draws as #388's
81-cell table (the cell indices differ).

## The pre-registered stop condition (the maintainer's, 2026-10-06)

**If (C)'s null ADOPT at k = 4 exceeds 0.05 on either path, (C) does not land and returns to the
maintainer; `experiment.gate` is not changed.** The expectation, stated before the run: (C)
ADOPTs only where the interval half alone does (it is that half with a weaker condition on the
seasons than (A)'s), whose null at k = 4 #388 measured at 0.034 (weekly) and 0.034 (draft), so
the null is expected under 0.035 on both. A null above 0.05 would be a harness fault before it
was a finding.

## Controls (rule 18), each of which must pass or the table is not published

1. **The (A) column reproduces #388's shipped column.** At weekly k = 4, 260 rows, δ = 0.3 and
   0.5: (A) within 3 SEs of **0.0007** and **0.0043**; draft k = 4, 20 rooms, δ = 2.0: within 3 SEs
   of **0.0076**; null (A) within 3 SEs of **0.0000** (weekly) and **0.0014** (draft), where a SE
   is the larger of the two runs' (a rate near 0 has a plug-in SE near 0, so the comparison is
   also stated in counts). A harness whose (A) is not #388's (A) is not driving the same rule.
2. **Inclusions, per frame:** zero frames on which (A) adopts or removes and (C) does not; zero
   on which (C) adopts or removes and the interval half alone does not; zero on which `gate`'s own
   verdict is neither rule's.
3. **A planted δ far above range** (weekly δ = 20 at k = 4 and 8, draft δ = 60 at k = 8): (C)
   ADOPT rate ≥ 0.99, so (C) can adopt.
4. **The paired difference is real where it must be:** at weekly k = 4, δ = 0.5, (C) ADOPT
   exceeds (A) by more than 4 SEs of the paired difference (`√d/N`, `d` the discordant frames);
   a control that cannot tell (A) from (C) measures neither.

## What this cannot do

It does not say whether any arm beats any incumbent (δ is planted), it estimates a between-season
variance from four seasons (#388's sensitivity table is that admission, and (C) is bounded above
by the interval-alone column, which that table already covers), and it **adopts nothing**:
landing (C) is the maintainer's decision, already made, conditional on the stop condition above.

## Results (C)

Run 2026-10-06 against the design above, which `3201859` committed first and which was not
edited. `gate_horizon.py --rule-c --trials 40000 --workers 6`, 38 cells × 40,000 trials, 25
minutes, seeds `SeedSequence([388, cell_index])`; raw counts not committed (`$TMPDIR/381-*`),
re-running the command reproduces them to the bit. **Worst SE of any rate in the table: 0.0025**
(the (C) power at k=7–8, δ=0.5); the null cells' are 0.0006–0.0009.

### The stop condition did not fire

**(C)'s null ADOPT at k = 4 is 0.0140 (562 of 40,000, SE 0.0006) on the weekly path and 0.0291
(1,163 of 40,000, SE 0.0008) on the draft path, both under the 0.05 bar.** The largest null (C)
ADOPT anywhere in the grid is **0.0351** (draft, k = 4, 80 rooms), and across k = 4..8 on both
paths it is 0.0140–0.0232 (weekly) and 0.0285–0.0321 (draft). The mirrored null REMOVE at k = 4
is 0.0143 / 0.0285, the same as ADOPT within a SE, as symmetry predicts. The expectation written
before the run (under 0.035) held on both paths. (C) is not free: it raises the null from (A)'s
0.0001 / 0.0014 to 0.0140 / 0.0291 at k = 4, a factor of over 100 (weekly) and 20 (draft) on a
base that was effectively zero, and it stays under `ALPHA`.

### The table

**Weekly path** (δ in points per roster-week); (A) abstain-and-veto and (C) resolved seasons, both
read off the same frames. null is ADOPT at δ=0.

| k | rows | null (A) | null (C) | SE | power δ=0.3 (A) | (C) | SE | power δ=0.5 (A) | (C) | SE | interval alone δ=0.3 | P(0 resolved) δ=0.3 | resolved seasons of k, δ=0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 220 (20×11) | 0.0001 | 0.0129 | 0.0006 | 0.0006 | 0.0757 | 0.0013 | 0.0022 | 0.1747 | 0.0019 | 0.1286 | 0.4603 | 0.70 |
| 4 | 260 (20×13) | 0.0001 | 0.0140 | 0.0006 | 0.0008 | 0.0829 | 0.0014 | 0.0039 | 0.2003 | 0.0020 | 0.1313 | 0.4225 | 0.77 |
| 4 | 280 (20×14) | 0.0001 | 0.0145 | 0.0006 | 0.0006 | 0.0920 | 0.0014 | 0.0050 | 0.2111 | 0.0020 | 0.1420 | 0.4045 | 0.81 |
| 5 | 260 (20×13) | 0.0000 | 0.0167 | 0.0006 | 0.0002 | 0.1167 | 0.0016 | 0.0010 | 0.2822 | 0.0023 | 0.1622 | 0.3449 | 0.96 |
| 6 | 260 (20×13) | 0.0000 | 0.0194 | 0.0007 | 0.0000 | 0.1510 | 0.0018 | 0.0004 | 0.3619 | 0.0024 | 0.1908 | 0.2809 | 1.15 |
| 7 | 260 (20×13) | 0.0000 | 0.0219 | 0.0007 | 0.0000 | 0.1795 | 0.0019 | 0.0000 | 0.4311 | 0.0025 | 0.2149 | 0.2250 | 1.34 |
| 8 | 260 (20×13) | 0.0000 | 0.0232 | 0.0008 | 0.0000 | 0.2105 | 0.0020 | 0.0000 | 0.4926 | 0.0025 | 0.2420 | 0.1776 | 1.54 |

**Draft path** (δ = 2.0 points per team-game); rows are rooms.

| k | rooms | null (A) | null (C) | SE | power δ=2.0 (A) | (C) | SE | interval alone | P(0 resolved) | resolved seasons of k |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 20 | 0.0014 | 0.0291 | 0.0008 | 0.0077 | 0.0828 | 0.0014 | 0.0919 | 0.1147 | 1.67 |
| 4 | 40 | 0.0047 | 0.0317 | 0.0009 | 0.0196 | 0.0958 | 0.0015 | 0.0982 | 0.0451 | 2.15 |
| 4 | 80 | 0.0100 | 0.0351 | 0.0009 | 0.0382 | 0.0974 | 0.0015 | 0.0983 | 0.0144 | 2.61 |
| 5 | 20 | 0.0003 | 0.0316 | 0.0009 | 0.0019 | 0.0983 | 0.0015 | 0.1027 | 0.0664 | 2.09 |
| 6 | 20 | 0.0001 | 0.0321 | 0.0009 | 0.0008 | 0.1143 | 0.0016 | 0.1180 | 0.0391 | 2.51 |
| 7 | 20 | 0.0000 | 0.0307 | 0.0009 | 0.0003 | 0.1232 | 0.0016 | 0.1303 | 0.0222 | 2.93 |
| 8 | 20 | 0.0000 | 0.0285 | 0.0008 | 0.0001 | 0.1258 | 0.0017 | 0.1390 | 0.0132 | 3.34 |

### What it says

- **(C)'s power at k = 4** is **0.083** (weekly δ=0.3), **0.200** (weekly δ=0.5) and **0.083**
  (draft δ=2.0), against (A)'s 0.0008 / 0.0039 / 0.0077 on the same frames. The paired differences
  are 0.082, 0.196 and 0.075 (SEs of the paired difference 0.0014, 0.0022, 0.0014: 57, 89 and 54
  SEs). (C) recovers **63%** (weekly δ=0.3) and **90%** (draft) of what the interval half alone
  reaches, 0.131 and 0.092. So at the weekly gate's noise, an Abstention still costs most of the
  residual: not through the every-season half any more, which now asks only that the resolved
  seasons agree, but because 42% of weekly frames have no resolved season at all (SHOW, "0
  resolved of 4, 4 abstained") and the interval half pools them anyway.
- **(C) does not make either gate powerful.** 0.08 at k = 4 is nowhere near `POWER` = 0.80;
  at weekly δ=0.5 it reaches 0.49 only at k = 8. It makes the Gate able to adopt (0.083 against
  0.0008, a factor of about 100 on the weekly path) and no more. Where a Gate stays unable to
  reach a verdict, the printed "0 resolved of k" says so rather than a bare SHOW.
- **More rows help the draft path's null and power together and the weekly path's barely:** the
  draft null climbs 0.029 → 0.032 → 0.035 across 20 / 40 / 80 rooms while power climbs 0.083 → 0.096
  → 0.097 and the interval-alone ceiling is 0.098, so (C) is at the interval's limit by 80 rooms.
- **Sensitivity.** #388's τ sensitivity table bounds this one from above: (C) cannot exceed the
  interval alone on either null or power (a per-frame inclusion, control 2 below), and #388's
  interval-alone null was at most 0.035 at 0× and 2× τ. So no τ in that range takes (C)'s null
  past 0.05; this was not re-simulated.

### The four controls: all pass

1. **(A) reproduces #388's shipped column.** Weekly k = 4, 260 rows: null 3 of 40,000 against
   #388's 0.0000 (at most 2); δ=0.3 **0.0008** (32) against **0.0007** (28), difference 4 counts
   against a SE of 7.7 (0.5 SE); δ=0.5 **0.0039** (155) against **0.0043** (172), 17 counts against
   18 (0.9 SE). Draft k = 4, 20 rooms: null **0.0014** (55) against 0.0014; δ=2.0 **0.0077** (307)
   against **0.0076** (304). All inside 1 SE.
2. **Inclusions, per frame, over all 38 cells:** frames on which (A) adopts or removes and (C) does
   not: **0**. Frames on which (C) adopts or removes and the interval half alone does not: **0**.
   Frames on which `gate`'s own verdict is neither rule's: **0**; and the shipped verdict equalled
   (A) on every one (the `ship` and `abstain` counters agree cell by cell, e.g. 3 / 32 / 155 weekly).
3. **Planted δ.** Weekly δ = 20 at k = 4 and 8, draft δ = 60 at k = 8: (C) ADOPT **1.00000** (40,000
   of 40,000) in all three, NOT-RUNNABLE 0.
4. **The paired difference is real.** At weekly k = 4, δ = 0.5 (C) exceeds (A) by 0.1964 (7,858
   discordant frames, SE of the difference 0.0022), 89 SEs, against the pre-registered 4.

### Where the design was hard to honour

- The (A) column is the same rule as #388's but not the same draws (the cell indices differ), so
  control 1 compares two independent runs and is stated in counts and SEs; it is not a bitwise
  reproduction.
- The harness reads (C) from `gate`'s own summary and seasons rather than from a `gate` that
  implements it, because `gate` was not to change before this section. The inclusion control and
  the `ship` column are what hold the harness's conjunction to the code that lands.
