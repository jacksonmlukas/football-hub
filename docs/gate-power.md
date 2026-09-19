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

> **Does a projection that scales volume and touchdown rate by the implied team total beat
> the incumbent projection on absolute error per player-week, under the house rule with the
> season as the unit of replication — and by more than four seasons can resolve?**

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
4. **The paired difference** per player-week is `diff = |err_B| − |err_A|`, in fantasy points
   of MAE, positive when the scaled arm is closer. MAE, not CRPS: #274's promotion of CRPS
   was for #292's shape decision and nothing else, and this is a question about the mean.
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
not read as a trend: two points are not a trend (rule 4's shape, in reverse). The same
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
- **ADOPT the scaled projection** only if the interval on `diff` excludes zero on the
  positive side **and** the scaled arm has the lower MAE in **every one of the four**
  held-out seasons — a tie is not a win, where a tie is a season whose gain does not clear
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
