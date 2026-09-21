# A Gate requires every season, not only the interval

**Status:** accepted 2026-09-04. **Amended 2026-09-07** (issue #45) with a precondition; the
adoption bar itself is unchanged.

**Decision.** A **Gate** adopts only when the pooled interval excludes zero **and** the sign
holds in every held-out season. It removes only when both hold in the other direction.
Everything else shows and is never ranked on. One implementation, `hub.models.experiment.gate`,
read by every gate in the repo.

**Amendment, 2026-09-07 — a Gate must first be able to run.** Before any of the three branches
above is read, a Gate whose **minimum detectable effect exceeds its measured ceiling** records
that it *cannot run* and reports no verdict. Both halves of the adoption bar are untouched:
this adds a precondition ahead of them, it does not weaken or restate either one. The
amendment is set out in full below.

---

## The question

`CONTEXT.md` defines a Gate exactly — *does this beat the simplest thing that already works?* —
and three modules answered it with three copies of the same three branches. They had diverged.
`hub.season.weekly_gate` required the sign to hold in every held-out season before adopting.
`hub.season.lineup_gate` and `hub.draft.backtest` adopted on the pooled interval alone.

Nobody decided that. It is what happens when one rule has three homes.

## Why the stricter form, and why it is not a preference

The looser form is not the considered alternative. It is the version nobody went back to.

`hub.models.spread` had already been bitten by it and says so in its own docstring:

> A candidate is adopted only if it beats `positional` in **every** held-out season *and* the
> paired difference clears `MIN_SE` standard errors. The first half stops a fit being adopted
> on one lucky year; the second stops one being adopted on a gain too small to distinguish
> from noise, **which is exactly what the first version of this function — which checked only
> the seasons — would have done**.

That copy was found weak and strengthened. The others were never revisited, because nothing
connected them. Unifying the rule is what makes a correction to it reach every gate instead of
one.

An interval excluding zero says the pooled effect is unlikely to be noise. It says nothing
about whether one season carried it, and this repo's record contains a case where repeated
measures turned noise into an apparent 4-sigma result (`docs/signal-screens.md`, protocol item
3). The seasons are the cheap defence against that and every gate already has the data.

## What it does not move, checked rather than hoped

Tightening two of the three gates could have flipped a published decision. It does not, and
`tests/unit/test_experiment.py` asserts it from the recorded statistics rather than leaving it
to this paragraph:

| Recorded | Interval | Seasons | Verdict, before and after |
|---|---|---|---|
| [ADR-0009](0009-championship-equity-does-not-pick.md) | [−23.16, −16.20] | lost all four | REMOVE |
| [ADR-0012](0012-the-lineup-optimiser-waits-for-real-variance.md) | [−0.00, +0.00] | — | SHOW |
| Frozen weekly gate | [−0.242, +0.684] | won 3 of 4 | SHOW |

A unification that changed one of these would have been a very different decision, and would
have needed reopening the ADR it changed rather than this one.

> **Re-scored 2026-09-07 (issue #51): the third row is superseded, and the conclusion drawn
> from it survives anyway.** The frozen weekly gate re-ran under #44 at **−1.004**,
> CI **[−1.391, −0.621]**, **0 of 4** seasons — so the row above records an interval and a
> season count that no longer describe that gate.
>
> Re-scored on the re-run, the row reads REMOVE rather than SHOW. **But the claim this table
> makes is about the unification, not about the gate's value**, and it holds either way: an
> interval excluding zero in the negative direction, with the sign holding in every held-out
> season, is REMOVE under the strict bar *and* REMOVE under the looser interval-only bar the
> other two gates used. Before and after still agree, which is what this section asserts.
>
> The row is left as published rather than edited, per `docs/method.md` rule 13, and
> `tests/unit/test_experiment.py` continues to assert from the recorded statistics.
> **Owned by #44.**

## What is deliberately not unified

Ten functions in this repo are called `verdict` and only three are this rule. The others are
different tests and flattening them would erase distinctions that are load-bearing:

- a **Screen** asks *is this real?* against a pre-stated sign (`weekly_screen`), and
  `CONTEXT.md` is explicit that a signal can pass one and fail the other;
- a walk-forward error comparison with ties to the incumbent (`margin`), an every-round MAE
  test (`component_error`), an every-season-plus-`MIN_SE` fit selection (`spread`).

They share a disposition — the rule is fixed before the numbers — and not a rule.

## What would reopen this

A gate whose held-out seasons are too few for consistency to mean anything. At two seasons the
every-season requirement is close to a coin flip and the interval is doing all the work. No
gate in the repo runs at fewer than three, and one that did should say so rather than quietly
inheriting a bar built for four.

---

# Amendment, 2026-09-07: a Gate must first be able to run

Issue #45, under the rule pre-registered in [gate-power.md](../gate-power.md) on 2026-09-06 —
before any of these numbers were known.

**Amended rather than replaced, and that is the point.** Nothing above is withdrawn. The two
halves of the adoption bar are exactly the two halves accepted on 2026-09-04, the table of
what the unification did not move still holds, and every gate that can run reaches the verdict
this ADR gave it. What is added is a question that must be answered *before* those halves are
read.

## The precondition

> A Gate whose **minimum detectable effect exceeds its measured ceiling** reports
> **NOT-RUNNABLE** and no verdict. The MDE is the smallest effect the run had 80% power to
> detect, two-sided at 5%; the ceiling is the largest effect there was to find — what a
> perfect arm gains over that gate's own incumbent, on that gate's own harness, in that gate's
> own units.

It sits **ahead of every branch but VOID**. VOID stays above it because a void gate's inputs
are broken, which makes its MDE and its ceiling untrustworthy too — there is nothing to
compare. Everything else sits below it.

## Why ahead of SHOW, and not after it

Because SHOW is the branch that would otherwise be wrong, and it is the likely one.

A gate that cannot resolve its own ceiling cannot tell a real effect from a perfect one. Every
branch below the precondition would be reading noise with a decimal point — but the two
excluding branches are self-limiting, since an underpowered gate rarely produces an interval
that excludes zero. The middle branch is not. It prints a null, a null from an underpowered
gate looks exactly like a null from a well-powered one, and this repo's record is largely a
record of nulls. Ordering the precondition after SHOW would let precisely the verdict that
must not be published be published first.

`docs/method.md` already separates the three ways a question closes: rule 8 closes one by
computing a ceiling, rule 12 acts where no gate *can* run and says so, rule 13 restates a
number that moved. This is the disposition for a gate that ran and should not have been asked.

## Not planned, not failed

A gate that cannot run closes its dependent tickets as **not planned**. *Failed* would say the
arm lost. *Not planned* says the design cannot answer the question with the data that exists,
which is a different and more useful thing for a future reader — and the arm may well be fine.

## What it does not move, checked rather than hoped

The precondition fires only when **both** numbers are present as values. A gate that measured
no ceiling has not shown it cannot run; it has shown nothing, and `experiment.Field.NO_SLOT`
is that third state rather than a licence to guess. Every published verdict in the table above
was reached without a ceiling in its summary and is reached identically now —
`tests/unit/test_experiment.py` holds the eighty-verdict sweep at its recorded digest, and
`test_a_runnable_gate_reaches_the_verdict_adr_0019_gave_it` runs it again for each state
either field can be in.

The comparison is signed rather than absolute, so a **measured ceiling of zero** — a perfect
arm gaining nothing at all over the incumbent — makes any positive MDE not-runnable. That is
the correct reading of the strongest finding a ceiling can carry, not an edge case.

## What this amendment does not decide

**Which arm a gate's ceiling is measured with.** For the draft gate and the weekly gate the
arm is settled and is full foresight. For the **lineup gate it is not**: gate-power.md's
stage 2 pre-registers a foresight ceiling and issue #43 deliberately built a *variance oracle*
instead, on the argument that both arms of that gate already share `mu`. The two bound
different questions and disagree about whether that gate is runnable.

That is a pre-registration question — whether a pre-registration may be re-read after the arm
it names was built — and it is open as **#138**. The mechanism here takes the arm as a
parameter and never inspects which arm produced the number it was handed;
`hub.season.lineup_gate.DECLARED_CEILING_ARM` is the one line #138 changes.

---

# Amendment, 2026-09-21: the every-season half says what a tie is (#335)

Filed 2026-09-19 in the pilot for #305's pre-registration; **ADOPTED 2026-09-20** by the
maintainer: **(A) with (i)**, below. Landed in S1's lane (#357) per the maintainer's own
disposition on #357, exempt from the #326 freeze as the finding that holds it open.

**Amended rather than replaced.** Nothing above is withdrawn. The pooled-interval half is
exactly what it was before this amendment — see the 2026-09-21 note at the end of this
section for the one thing that changed about it, under a different issue — and the
every-season half is still "the sign must hold in every held-out season." What changes is
what *counts* as holding: a season is no longer read on its raw sign alone.

## The defect

`experiment.paired_gain`'s `wins` was `(arm_mae < base_mae).sum()` — a strict inequality on
the point estimate, with no notion of resolution. Found 2026-09-19 while pre-registering
#305's MDE: the published rebuild-vs-flat contrast reads `wins 4/4`, and its 2024 gain is
**+0.00005** MAE per player-week — the two arms differ on 4,089 of 4,343 rows and net to
zero. That season is counted as a win. "4/4" was doing the work "3 wins and a tie" would not,
and nothing in this ADR or the code said which it was.

## The rule

**(A) A tie is not a win, and ADOPT requires a win in every season; a tie is not a loss for
REMOVE, which needs a loss in every season.** A tie is absence of evidence in that season,
and both directions need evidence in *every* season — symmetric, conservative both ways. This
is option (A) from the pilot; option (B), skipping ties and reading only the resolved
seasons, was rejected as a way for fewer seasons to license a verdict, and option (C), a tie
counting as whatever its sign says, is the defect made explicit.

**(i) A season is a win if `gain >= 2 * SE` over its within-season clusters, a loss if
`gain <= -2 * SE`, a tie between.** The within-season unit is rule 3's (`docs/method.md`) —
never the row — and is named per gate, below. **Below `TIE_MIN_CLUSTERS = 12`** (chosen,
`hub.models.experiment.TIE_MIN_CLUSTERS`; the relative error of a sample SD is approximately
`1 / sqrt(2 * (m - 1))` under normality, and 12 puts that near 20% — never placed on any
one gate's own default, so an off-by-one in an unrelated flag cannot flip the rule's shape
for every gate at once) **the tie test falls back to the sign alone**, and says so: `m` is
printed beside the threshold, per season, in every verdict sentence.

The quarterback gate (`hub.models.starter_change`) is a no-op, named as one: its paired frame
is already one row per event-season, so its within-season unit is the event, and the
within-season SE this computes is over rows — exactly what it would do anyway.

## Per-gate within-season units

| gate | within-season unit | why |
|---|---|---|
| draft (`hub.draft.backtest`) | `draft` | the room a season's rosters were drafted in |
| weekly (`hub.season.weekly_gate`) | `roster` | one row per (season, roster, week) |
| lineup (`hub.season.lineup_gate`) | `roster` | one row per (season, roster) |
| coverage (`hub.models.coverage`) | `player_id` | rule 3's own unit |
| quarterback (`hub.models.starter_change`) | the event (`game_id`) — **a no-op** | already one row per event-season |
| the margin/shape house rule (`hub.models.margin`) | `season` — **a declared no-op** | `paired` is already one row per season; no finer unit exists |
| injury type (`hub.models.injury`) | `season` — **a declared no-op, and deliberately so** | `#360` is frozen; see below |

`injury.type_verdict` is frozen by #326/#360 (S3) — this amendment's own lane is not, so the
statistics function it calls changed signature under it, but the module's *decision* does
not move. Its `within` is set to its own `season` column, which makes every season's group
size exactly 1 by construction — always below `TIE_MIN_CLUSTERS` — so `_disposition` falls
back to the sign unconditionally, reading exactly `arm_mae < base_mae` in every season, the
same condition this line read before #335 (`gain_s`, the mean of `err_retention - err_type`
over a season, is `mae_retention - mae_type` by construction). Nothing about the frozen
verdict moves; only the printed line grows `ties`/`losses`, both always 0 there.

## Where it lives in code

`experiment.per_season` and `experiment.paired_gain` both take `within` with **no default**,
for the reason `summarise`'s `cluster` has none: guessing the repeated-measure unit is the
mistake, not a convenience a caller can skip. `per_season` gains `se` and `m` columns;
`experiment.gate`'s every-season half reads them through `_disposition` rather than
`seasons["gain"]`'s sign, falling back to the sign alone when a `seasons` frame carries
neither column (every hand-built summary in this repo's own test suite, so the pre-#335
branch-logic tests are unaffected) or when a season's own `m` is below the floor.
`paired_gain` returns `wins`, `ties` and `losses`, computed the same way, for the three
verdicts that bypass `gate` (`hub.models.weekly`, `hub.models.injury`, `hub.models.spread`).
Every verdict sentence prints all three counts.
`tests/contracts/test_gates_tie_test_names_its_within_season_unit.py` holds the five
`run_gate` call sites' `within` argument against the table above.

## Pre-registered condition 1: the size test proven against a planted degenerate rule

Landed with #357 (S1), the ticket that also fixed the interval half: `docs/method.md` rule
15's shape — a size test that only ever passes is not evidence the check works. Two
permanent tests, `tests/unit/test_experiment.py::test_the_size_check_flags_a_planted_degenerate_rule`
and `::test_the_fixed_rule_s_null_size_is_not_degenerate`, simulate the pre-#357 rule
(planted, expected degenerate at `~2**-k`) and the fixed rule (expected well under it) under
the same null, through the same `_null_adopt_rate` harness.

## Pre-registered condition 2 (rule 16): the combined rule's null size and power

Computed **before** this amendment was written, per `scripts/rule16_combined_power.py` —
numpy and the shipped `experiment.summarise`/`per_season`/`gate`, nothing reimplemented.
10,000 trials per cell, bootstrap 200, at the repo's own published season-clustered figures
from #357's (S1) table, with within-season cluster counts `m=20` (draft) and `m=40` (weekly)
as this condition's own pre-registration — neither gate's real within-season SD is published,
so both simulations use the gate's own between-season `s` as a stated, conservative stand-in
for it (see the script's docstring):

| gate | k | s | m | δ | combined null size | combined power at δ | unanimity-alone power at δ (#357's table) |
|---|---|---|---|---|---|---|---|
| draft | 4 | 7.34 | 20 | 2.0 | **0.0113** | **0.0324** | 0.136 |
| weekly blend (**confirmed 2026-09-21 under #378**) | 4 | 0.382 | 40 | 0.3 | **0.0186** | **0.1925** | 0.378 |

**Both null sizes sit comfortably under `ALPHA` (0.05)** — the tie requirement can only make
the conjunction rarer than the interval-alone test, never more permissive, and the simulation
confirms it rather than assuming it.

**Combined power is below unanimity-alone power at both cells, and the rule lands anyway.**
This is the pre-registered disposition, stated before the numbers were known: if combined
power came in lower, the cost is the finding, not a reason to re-choose `TIE_MIN_CLUSTERS` or
the `2 * SE` threshold after seeing this table. A tie-aware rule is stricter by construction —
a season that would have counted as a win under the sign alone can now cost a season's worth
of evidence — and at these SEs and cluster counts, that strictness measurably lowers power at
a realistic delta. The alternative is the defect this amendment closes: a rule with less
power is a rule that is actually testing what it claims to.

**The mechanism behind the number (2026-09-21, after the first two runs).** The power figures
above say how often; this says why, because without it the next reader of 0.032 reaches for
the threshold, and the threshold is not the lever. **One tie vetoes every verdict, in both
directions.** ADOPT needs a win in every season and a tie is not a win; REMOVE needs a loss in
every season and a tie is not a loss. So a single unresolvable season forces SHOW however large
or consistent the effect is elsewhere. With k held-out seasons and a per-season tie
probability p, the chance that at least one season abstains is 1 − (1 − p)^k: at k = 4 and
p = 0.25 that is 0.68 — SHOW roughly two runs in three, before the effect size enters at all.
The first two gates run under this rule did exactly that: the draft gate (#376: 2022 −15.35,
2023 −8.41, **2024 −5.99 tie**, 2025 −19.86) and the weekly gate (#378: 2022 −1.120, 2023
−1.448, **2024 −0.259 tie**, 2025 −1.761) both read SHOW at won 0, tied 1, lost 3 of 4, by the
same mechanism. That is (A) working as adopted, and the pre-registered disposition holds.

**The open question this leaves, stated now and not decided now.** What a gate should do with
an unresolvable season has three coherent answers, and the adoption chose one: **(A)
abstain-and-veto** (this rule: a tie blocks both halves); **(B) evaluate over resolved seasons
only** (rejected above: fewer seasons license); and a third nobody listed — **(C) report the
verdict over resolved seasons *with the abstention count beside it***, which keeps (B)'s power
without (B)'s licensing problem, because a reader sees "ADOPT on 3 resolved of 4, 1 abstained"
and not "ADOPT". Re-deciding after two SHOWs is what rule 1 forbids, so this is recorded as the
question the first three runs under the rule will inform, and the decision waits for them
(#381). One sub-question those runs should answer on the way: 2024 tied in both harnesses.
At one in four each that is one in sixteen jointly — probably coincidence — but if 2024 has
systematically wider *within-season* spread in both, the tie rule is abstaining on noisy
seasons rather than on small effects, which is a different property from "a tie means the
effect is small" and would need saying. Neither run recorded its per-season SE beside the
gain; #382 makes every run record it, and re-runs the weekly harness once to read 2024's.

**The draft gate's ADOPT branch is a named exemption, not a bar (2026-09-21).** At δ=2.0 the
combined rule's power for the draft gate is **0.0324** against unanimity-alone's 0.136 — an
ADOPT that is reachable in principle but not at any effect size this project would plausibly
observe. Per rule 16 and #300, a branch a gate cannot practically reach is not a strict bar on
that gate; it is an exemption, and this pre-registration names it as one rather than leaving a
reader to infer it from the branch never firing.

**Runnable and able to adopt are two different questions, and #376 answers only the first
(2026-09-21).** The row above is about *power to adopt a given effect size*, computed without a
ceiling. `docs/gate-power.md`'s dated restatement, same day, is about *whether the design can
resolve a real effect at all*: the draft gate's own `--ceiling` run (`hub.draft.backtest
--seasons 2022,2023,2024,2025 --drafts 20 --ceiling`, `n_drafts=20`, the module's documented
default) measured a season-clustered MDE of **11.01** against a foresight ceiling of **+21.90**
— the stage-2 guard (this ADR's 2026-09-07 amendment; widened by #363/S6 above) passes, at about
**1.99x**, a much thinner margin than the weekly gate's 9.7x. `experiment.gate`'s NOT-RUNNABLE
branch therefore does not fire for the draft gate either, closing what the S6 amendment below
names as owed to #376. **Both facts are true at once, and neither implies the other:** the
draft gate is *runnable* — its MDE sits below its measured ceiling, so a real effect of a
plausible size would show up rather than being lost to noise — and its ADOPT branch is
separately *unreachable by power* at the row's own δ=2.0, a property of the combined rule's
size at k=4 that a ceiling does not change. A gate can clear the stage-2 guard and still never
practically adopt; this is that gate. Full numbers and the run's other caveats:
`docs/gate-power.md`'s dated 2026-09-21 restatement, beside this same section's #376 note for
the weekly half.

**The weekly row above is withheld pending #378 (2026-09-21).** #376's weekly `--ceiling` run
produced **3** clusters, with season 2022 silently absent, while this row was computed at
**k = 4** — one of the two is wrong, and k is the axis that sets this row's null size, MDE, and
every power figure in it, so the row cannot be read as established until #378 resolves which k
is correct. It is struck rather than removed so the original computation stays visible, and
will be recomputed at the corrected k when #378 lands.

**Resolved 2026-09-21 (#378): k = 4 was always correct, and the strike is lifted rather than
the row recomputed.** `weekly_gate.compare`'s walk-forward split (`experiment.expanding_seasons`)
always consumes the earliest season it is handed as training data for the one after it —
`docs/weekly-blend-gate.md`'s Reproduce section names five seasons for exactly this reason, the
first a sacrificial buffer so the four held-out seasons (2022-2025) all score. #376's ceiling
run used `weekly_gate`'s own `--seasons` default instead, which named only the four held-out
seasons with no buffer ahead of them — so `expanding_seasons` dropped 2022, the earliest of
*those four*, silently, and the run scored three. Nothing was wrong with `k = 4` or with this
row's numbers; the run that appeared to contradict them was reading a different, accidental k.
Fixed by widening `weekly_gate`'s `--seasons` default to include the buffer season and by
`weekly_gate_data.SeasonDropped`, which now refuses a run whose scored seasons fall short of
what it was asked for by more than that one expected buffer season. Re-run at the corrected
default (`uv run python -m hub.season.weekly_gate --run --ceiling`, drafts=20, seed 0, frozen
rosters, restricted, mask_pool — the same recipe #376 used): **4** clusters, ceiling **+10.873**
against an MDE of **1.119** (9.7x over, wider than #376's mistaken 5.6x). Full before/after is
`docs/gate-power.md`'s dated restatement under this same date. The table row above is therefore
unstruck rather than recomputed — the k it was always computed at is the k the gate reads.

## What it does not move, checked rather than hoped

`tests/unit/test_experiment.py::test_the_eighty_gate_verdicts_are_unmoved` is a regression
pin over a synthetic grid, not over a published headline figure — recomputed and the move
documented at the test (three verdicts flip, all `(0.0, 0.0, 0.0)`-gains rows moving from
REMOVE to SHOW, since an exact-zero season is now a tie rather than silently "not a win"). No
published verdict in this repo is known to rest on a tie; #311's restatement work reads every
published `wins k/k` under the new rule and says which, if any, do.

---

# Note, 2026-09-21 (#357): the interval half no longer reduces to the sign half

Filed as its own ticket (S1, from the 2026-09-20 method audit) and landed first in this same
lane, immediately ahead of #335 above — the pooled-interval half both amendments describe is
the *same* half, and #335's "combined rule" (its own pre-registered condition 2) means the two
landed together.

`gate`'s ADOPT/REMOVE conjunction used to read the *percentile* bootstrap's `lo`/`hi`. Under
`SEASON_CLUSTER`, a nonparametric percentile bootstrap over `k` clusters can only resample the
`k` numbers it was handed, so when every season's gain is positive, every resample is a convex
combination of positive numbers and `lo > 0` follows from `won == total` **by construction** —
the interval half was the sign half, read twice, and the whole rule was a one-sided sign test
of size `2**-k`. `gate` now reads a **t interval** (`t_lo`/`t_hi`, `experiment.t_interval`,
computed from `mean`/`se`/`clusters`) instead — a distributional claim rather than a resampling
one, not implied by the seasons' signs the same way. `summarise` also now computes `power`
(`experiment.achieved_power`, the MDE relation inverted: power against the effect actually
observed), which `paired_report` prints beside the MDE so a SHOW says what the design could
detect. Full detail, the proof this is no longer degenerate, and the planted-rule test that
would have caught the original defect are in `hub.models.experiment.t_interval`'s and
`gate`'s own docstrings and `tests/unit/test_experiment.py`'s `#357 (S1)` section. No published
verdict moves: the eighty-verdict sweep's boolean decisions are unchanged (only the rendered
sentence's wording moved, checked against the pre-#357 `gate` on the same grid), and the three
recorded-verdict regression cases (ADR-0009, ADR-0012, the frozen weekly gate) reproduce.

---

# Amendment, 2026-09-21: NOT-RUNNABLE requires a ceiling, in both directions (#363, S6)

From `docs/audits/2026-09-20-method-audit.json`, finding **S6** (severity: decision).
**ADOPTED, option 1**, by the maintainer.

**Amended rather than replaced.** The precondition this ADR's first amendment (2026-09-07)
added is not withdrawn — a gate whose MDE exceeds its ceiling is still NOT-RUNNABLE — this
widens *when the precondition is even asked*.

## The asymmetry S6 found

The precondition used to fire only when *both* the MDE and a measured ceiling were present as
values. A gate that never measured one had not shown it cannot run — true, and quoted from
this ADR's own prior text — but what that protected turned out to be one-sided: SHOW (a null)
and ADOPT are both excluding branches, and an underpowered design rarely produces an interval
that excludes zero, so they are self-limiting even unprotected. REMOVE is not different in
kind, but per `docs/gate-power.md`'s own account **two of the three season gates have never
measured a ceiling**, so for them REMOVE was reachable with no protection at all — and REMOVE
is not a label: a removed module moves to `hub.exhibits`. `championship_equity` and `leverage`
are there now, removed on evidence S1 (#357) separately showed was a 14%-power sign test.

This is the mirror of the amendment above (R6/#300): that one found a branch the gate could
never *leave* granted a standing licence; this one finds a branch the gate could never *reach*
withholding a protection from only one direction.

## The rule

**A gate that measured no ceiling returns NOT-RUNNABLE, full stop — before the pooled interval
or the every-season half is read at all — the same way a `void` condition does, one rung
below it.** `reading(summary, "ceiling")` at anything other than `Field.VALUE` — no slot,
because the caller never measured one, or no data, because it tried and got nothing — trips
it. The prior precondition (MDE exceeding a *measured* ceiling) still applies below this one,
for the case where a ceiling exists but is too small; it is now unreachable except when a
ceiling is already present, which is the only state that makes reading it meaningful.

**Guarded behind `has_data` — the one thing this does not touch.** A gate with no rows at all
still reports "nothing measured" rather than a sentence about a ceiling there was no chance to
measure; `clusters` gates this branch exactly as it already gated the old one.

## What this makes NOT-RUNNABLE today, and what ends it

As of this amendment: the **weekly** gate's ceiling was measured 2026-09-21 under #376 (stage
2 passes, ~5.6x — see this document's own #376 note, above) and is unaffected. The **draft**
gate's ceiling is not yet measured and is NOT-RUNNABLE, in both directions, until #376's
second half closes it — cited by number, per rule 16, so this does not become a second
unnamed permanent NOT-RUNNABLE state. The **lineup** gate carries its own open question
(#138, which arm its ceiling should be) independent of this amendment. The **coverage** gate
(`interval_shape`) always hands in a ceiling at its own call site and is unaffected. The
**quarterback** diagnostic (`starter_change`) is unaffected in the same way when run with
`--ceiling`, and continues to read its own `EVENT_SEASONS_MINIMUM` exemption first regardless.

**Resolved 2026-09-21: the draft half closed under #376.** Measured the same day as the weekly
half, same recipe shape (`hub.draft.backtest --seasons 2022,2023,2024,2025 --drafts 20
--ceiling`, `n_drafts=20` the module's own default): MDE **11.01** against a ceiling of
**+21.90**, stage 2 passing at about **1.99x** — thinner than the weekly gate's 9.7x but still
clear. Both season gates this amendment named are therefore NOT-RUNNABLE no longer, and #376's
own acceptance criteria are both discharged. Full numbers, the run's own SHOW verdict and why it
is not the published one, and the pre-registered rule-16 ADOPT-power finding read together with
this stage-2 result, are in this document's own rule-16 section above and
`docs/gate-power.md`'s dated 2026-09-21 restatement beside the weekly note.

## What it does not move, checked rather than hoped

No published verdict is known to have been read from a gate run with no ceiling and a REMOVE
or ADOPT that a measured ceiling would have overturned — the point of this amendment is that
nothing before it could have told the difference. `tests/unit/test_experiment.py`'s
`test_no_ceiling_measured_is_not_runnable_in_both_directions`,
`test_a_ceiling_measured_as_no_data_is_also_not_runnable` and
`test_an_empty_frame_with_no_ceiling_still_says_nothing_measured` pin the new branch
structure; `test_the_restated_weekly_gate_figures_reproduce` holds the one place a published
figure now needs the ceiling handed in explicitly to reproduce at all (#376's +10.799), and
still reproduces.

**Not planned, not failed, in both directions now.** The distinction the first amendment drew
— a gate that cannot run has not been shown to have lost the arm — was already the correct
reading for ADOPT and SHOW. It is now the reading for REMOVE as well.
