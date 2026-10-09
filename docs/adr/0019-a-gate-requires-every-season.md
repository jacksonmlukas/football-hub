# A Gate requires every season, not only the interval

**Status:** accepted 2026-09-04. **Amended 2026-09-07** (issue #45) with a precondition; the
adoption bar itself is unchanged. **Amended 2026-10-06** (#381): "the sign holds in every
held-out season" below now reads over the *resolved* seasons, with the abstention count printed
beside the verdict; the text of this paragraph and of the earlier amendments is kept as written,
and the last amendment in this file says what changed and why.

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

> **Amended 2026-09-21 — ADOPTED on #335 (item 1) the same day, after a code review found
> it.** The delta is one point, stated exactly: at a bootstrap SE of
> exactly zero — every paired diff in the season identical — the adopted test `gain >= 2 * se`
> read a gain of **exactly zero** as a win (`0 >= 0`), so three real wins plus one season in
> which the arm changed nothing adopted 4/4. A zero SE now takes the sign-alone branch, where
> exactly zero is a tie; a strictly positive gain at zero SE was a win before and is a win
> still. Strictly stricter on ADOPT, unchanged on REMOVE, and the rule-16 table above is
> unaffected (a zero SE has probability zero under its simulation). It is nonetheless an edit
> to a rule adopted the same morning, after two gates ran under it, so it was not adopted by
> the review's note: the code carried it because the fix is right and harmless, the record
> said PROPOSED until the maintainer's line was written, and it was. Control:
> `test_a_no_op_season_is_a_tie_and_cannot_carry_an_adopt`.

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
| quarterback (`hub.models.starter_change`) | `season` — **a real no-op, corrected 2026-09-21** | one cluster per season, always below `TIE_MIN_CLUSTERS`; the first version said `game_id` was a no-op because the frame is "one row per event-season" — the clustering was, the tie test was not (48–57 event games a season, four times the floor). #335 item 2; the row-level question is #381's |
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

**Restated 2026-10-06 (#418, rule 13; the table above is not edited).** Re-run today,
`scripts/rule16_combined_power.py` reproduces every figure in it to the digit (0.0113 / 0.0324,
0.0186 / 0.1925), so they stand **as the combined rule's null and power under the stand-in
generating process stated beside them**: within-season cluster SD set equal to the between-season
`s`, which at these `m` is a season SE of 0.060 (weekly) and 1.64 (draft). They are not the
rule's power at the gates' own measured precision. The weekly gate's recorded season SEs are
0.409 / 0.447 / 0.543 / 0.546 (seven to nine times wider than the stand-in's 0.060), and run at
the same k=4 through the same `gate`, with the within-season spread estimated from them (#388's
harness, `gate_horizon.py --rule16`, 40,000 trials), the combined rule gives **weekly null 0.0001 /
power 0.0008** (δ=0.3) and **draft null 0.0014 / power 0.0075** (δ=2.0). The stand-in's own
description of itself, "conservative", was wrong for the weekly gate: a season its tie test
resolves at SE 0.06 is one the real gate ties at SE 0.5, and the observed gates tied 1 season of
4 where the stand-in predicts 0.003 per weekly season at the gains observed. Of the 0.193 weekly
power lost between the two processes, 73% is that within-season spread, 19% is m (40 to 20),
5% the estimated τ, 2% the shipped bootstrap of 4000 against 200 (not distinguished); the #363
NOT-RUNNABLE path owns none of it (the script hands `gate` a ceiling that cannot bind, and the
same harness without one returns NOT-RUNNABLE 300 times in 300). The mechanism above, one tie
vetoes both directions, holds more strongly, not less; the draft ADOPT branch's rule-16
exemption below stands at 0.0075 rather than 0.0324. No rule, constant or verdict changes.
Attribution, table and control: `docs/gate-power.md`, *What contradicts a published figure*,
#418.

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
(#381). *(Decided 2026-10-06: (C), with the details settled the same day; see the amendment at
the end of this file. This paragraph is kept as it was written.)* One sub-question those runs should answer on the way: 2024 tied in both harnesses.
At one in four each that is one in sixteen jointly — probably coincidence — but if 2024 has
systematically wider *within-season* spread in both, the tie rule is abstaining on noisy
seasons rather than on small effects, which is a different property from "a tie means the
effect is small" and would need saying. Neither run recorded its per-season SE beside the
gain; #382 makes every run record it, and re-ran the weekly harness once to read 2024's: SEs 0.409 / 0.447 / **0.543** / 0.546 for 2022–2025, so 2024's is wider than two seasons and equal to the third, and its gain (−0.259) sits below every season's threshold (the narrowest, 0.82, by a factor of three). The cleaner discriminator is the pair: **2024 at 0.543 and 2025 at 0.546 are the same precision and resolved differently — tie versus loss — so the rule discriminated on the effect**, not the season. One season in one gate: a hint written down, not a finding; the draft's SE waits for its next run, and #381's release condition is three entries for this reason.

**The draft gate's ADOPT branch is a named exemption, not a bar (2026-09-21).** At δ=2.0 the
combined rule's power for the draft gate is **0.0324** against unanimity-alone's 0.136 — an
ADOPT that is reachable in principle but not at any effect size this project would plausibly
observe. Per rule 16 and #300, a branch a gate cannot practically reach is not a strict bar on
that gate; it is an exemption, and this pre-registration names it as one rather than leaving a
reader to infer it from the branch never firing. *Restated 2026-10-06 (#418, rule 13): the 0.0324
here is the stand-in-process figure; at the draft gate's own recorded precision the same rule
gives 0.0075, so the exemption stands on a smaller number (the restatement under the table
above).*

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

---

# Note, 2026-09-22 (#386): the rule lives in `gate(paired, ...)`, one seam

`experiment.gate` moved from a summary-dict interface (`gate(summary, seasons, actions)`) to
`gate(paired, *, cluster, within, ceiling, actions) -> GateRun` — a paired frame in, a verdict
out, and pure: no Ledger row, no width stamp, no render, so calling it twice on the same frame
at the same seed returns the identical `GateRun`. The old signature is `_verdict` now, the
internal half `gate` materialises `summary`/`seasons` for (via `summarise`/`per_season`) and
calls; `run_gate` stays the composition (`gate` + render + `stamped_for_publication` +
`review_width`), unchanged at every one of its six call sites (`draft/backtest.py` x2,
`season/weekly_gate.py`, `season/lineup_gate.py`, `models/coverage.py`,
`models/starter_change.py`). `margin._house_rule` is deleted in favour of the same `gate`.

**Implemented in `gate(paired, ...)` from `0b582cd`. The rule is unchanged; where it lives
moved.** The equivalence control is `tests/unit/test_gate_seam_equivalence.py`, captured
against `main` before the seam moved and its golden committed alongside it
(`tests/unit/fixtures/gate_seam_equivalence_control.json`, commit `4259a53`) — green on
unchanged code, red on the ticket's own literal plant (`_disposition`'s `>=` → `>`,
`_verdict`'s own every-season half), and, after the move, byte-identical on both sides of the
seam. Rule 18 was also planted on the interval/ceiling half — the values `gate` computes
(`summary["mde"]`, `summary["ceiling"]`) and hands down to `_verdict` — with a one-token
`>` → `>=` at the stage-2 comparison, caught by `test_an_mde_exactly_at_the_ceiling_still_runs`
through the real frame-in pipeline with no floating-point luck required (the ceiling in that
fixture is `gate`'s own computed MDE, handed back to it verbatim, so the two sides of the
comparison are bit-identical by construction rather than by chance). Both plants were applied
by hand, confirmed red, and reverted; `git diff` on `experiment.py` was empty after each.

**The flat-2.0 bar `spread.verdict` and `injury.type_verdict` read is a second, named rule.**
`g.t >= MIN_SE`, no degrees-of-freedom correction, is a different test from the t interval
`gate()` reads (`t_quantile` at `df = k − 1`) — out of scope here, owned by #343 (frozen behind
#326), and this line exists so a reader who concludes "all four call sites now read the same
bar" is corrected rather than left to assume it.

---

# Amendment, 2026-10-02 (#397): the one-implementation rule covers Gates, not Screens

**Amended rather than replaced.** Nothing above is withdrawn. The decision paragraph's last
sentence — *one implementation, `hub.models.experiment.gate`, read by every gate in the repo* —
is true of the six modules that declare a `Harness` and was read, by omission, as covering every
function in the repo that returns a verdict. It does not. `hub.models.weekly_screen.verdict`
reads a different rule on purpose and is not a call site of `gate`.

**The scope, stated.** "One implementation, read by every gate" covers **Gates**. It does not
cover **Screens**. `weekly_screen.verdict` is a Screen, and a Screen is not held to the
every-season adoption bar or to the NOT-RUNNABLE precondition above.

**Why, and it is not a loophole.** `CONTEXT.md` separates the two tests and says they are not
interchangeable: a Screen asks "is this real?" and a Gate asks "is this better than what it
replaces?". The rule this ADR centralises is an *adoption* rule — it decides whether a candidate
replaces the thing already working, which is the Gate's question and only the Gate's. A Screen
adopts nothing. `weekly_screen.verdict` takes a sign fixed before the run, clears a feature when
that sign holds in every season and the pooled statistic clears `MIN_SE`, and also has a
pre-stated-null branch (`sign="0"`) that a Gate has no analogue of. Routing it through `gate`
would make a screening question answer an adoption one: the Screen has no arm to beat and no
ceiling to run against, so the precondition has nothing to read. Merging the two would be the
confusion `CONTEXT.md` warns about, written into code.

**The other verdicts that bypass `gate` are not covered by this amendment.** They are Gates (or
are meant to read one) and are handled by their own tickets, not excused here:
`spread.verdict` and `injury.type_verdict` / `injury.verdict` route through the shared Gate
under #343 (frozen behind #326); `component_error.verdict`, a Gate by this ADR's own list,
joins #343's batch. Until those land, "read by every gate" is a target for them and a fact for
the six `Harness` modules.

**What this does not move.** Both halves of the adoption bar and the precondition stand, and
`weekly_screen.verdict` is untouched: this amendment changes what the ADR claims, not what any
code does.

---

# Note, 2026-10-06 (#343): "read by every gate" now holds for nine modules, and held for six before

**Dated, not edited.** Nothing above is withdrawn and the #397 amendment's text is left as
written, including its last paragraph, which says what was *then* true. What changed is the
code: the claim in this ADR's decision paragraph — *one implementation, `hub.models.experiment.gate`,
read by every gate in the repo* — **held for six of the eight gate-shaped modules #343's ticket counted, and was a target for the
other two** (`spread.verdict`, `injury.type_verdict`). `component_error.verdict`, which the
2026-10-02 architecture review added to the batch, was a ninth that did not read it either. After
#343 it holds for all nine.

**What landed.** Each candidate-vs-baseline comparison in those three modules is one
`Harness.run` (`Harness.decide` for the pure half a test reads): the spread candidates `own_k`
and `usage` each against `positional`; the injury type adjustment against `retention`; the
component calibration against the raw projection. The three hand-built rules are deleted —
`g.wins == seasons and g.t >= MIN_SE` in both spread and injury, and "MAE improves in every
held-out season" over three aggregates in `component_error`. All three now reach stage 2, the
S6 ceiling precondition, the tie-aware every-season half, the S1 t interval, the width review
and the stamps, for the first time. The declared ceilings are in each module's `CEILING_ARM`
and in `docs/gate-power.md`'s #343 pre-registration (spread: the published +0.0852 headroom;
injury type: the in-sample per-type multiplier of that pre-registration's adoption comment;
component calibration: the in-sample calibration, declared in the #343 commit, which the
pre-registration did not cover).

**What still does not read it, named so the claim is not over-read.** `hub.models.injury.verdict`
— the retention / table / baseline / out_zero argmin — has no gate in it at all, hand-built or
otherwise, and is #360's (S3). It is not a Gate comparison of a candidate against an incumbent
and this note does not say it is covered; `weekly_screen.verdict` is a Screen (#397) and is out
of scope by that amendment. The `Harness` count is ten over nine modules
(`tests/contracts/test_each_gate_declares_its_ceiling_arm.py`).

**What this does not move.** Both halves of the adoption bar, the precondition, the tie rule and
the S1 interval are unchanged; this routes three more call sites to them. Which verdicts moved
on real data, and what each doc that published them now says, is in the #343 commit and in the
restatement boxes of `docs/player-spread.md`, `docs/weekly-injury.md` and
`docs/component-projection.md`.

---

# Amendment, 2026-10-06: a Gate's verdict reads over its resolved seasons, and says how many abstained (#381)

Filed 2026-09-21 as the question the 2026-09-21 amendment left open (*The mechanism behind the
number*, above); **ADOPTED 2026-10-06 by the maintainer: option (C)**, with the details settled
in the same session. (A), the rule of the 2026-09-21 amendment, is not kept; (B), (D) and (E) are
not taken.

**Amended rather than replaced.** Nothing above is withdrawn: the 2026-09-21 amendment's wording
about (A), its "the mechanism behind the number" and its open question are kept as they were
written, because they say what was true and why the question was asked. What changes is the
sentence "the sign must hold in every held-out season", which now reads over the **resolved**
seasons.

## The rule

A season has one **Disposition**: a win, a loss, or an **Abstention** (CONTEXT.md, 2026-10-06).
A season with a win or a loss is *resolved*. An Abstention is a fact about the season's evidence,
not about the effect, and is never counted as a win.

- **ADOPT** needs at least one resolved season, every resolved season a win, and the pooled t
  interval above zero.
- **REMOVE** is symmetric: at least one resolved season, every resolved season a loss, the
  interval below zero.
- **Zero resolved seasons: SHOW**, "0 resolved of k, k abstained". It is **not** NOT-RUNNABLE,
  which stays the verdict for a design that cannot reach its own effect (read ahead of this
  half, unchanged). Without the guard, "every resolved season won" is vacuously true at zero and
  the interval half alone would adopt.
- **No floor beyond one resolved season.** A floor such as ceil(k/2) would be a second threshold
  chosen after seeing abstention rates.
- **The interval half is unchanged and still pools all k seasons**, abstaining ones included.
  An Abstention votes for nobody but is still evidence about the pooled effect, so a
  resolved-unanimous frame whose pooled interval crosses zero is SHOW.
- **Every verdict sentence carries the count**, never a bare verdict: "ADOPT ... (3 resolved of
  4, 1 abstained; won 3, tied 1, lost 0 of 4 seasons)". The tally keeps the code's word *tied*
  (the field is still `ties`); the prose says Abstention. `GateRun.resolved` / `.abstained` and
  the ledger entry (`state/gate-width.json`, `resolved`, `abstained`) carry the two numbers.
- **#375** (the posterior replacing the verdict) supersedes (C) when it lands, not (A).

## Why this is not rule 1's incident

Rule 1 forbids choosing a rule after the numbers it judges. The release condition for this ticket
was five tie-aware ledger entries from four gates, and the 2026-09-22 comment on #381 asked which
quantity those entries supply. What the ledger and #388 / #418 supplied is an **abstention rate**:
how often a season's gain fails to clear its own noise, and at what within-season row count. That
is a variance estimate, not an effect reading. No verdict's direction informed the choice. The
fact that decided it is a power figure computed on spent seasons: under (A) the shipped combined
rule's power is **about 0.0008 (weekly, δ=0.3) and 0.0075 (draft, δ=2.0)** at k=4 (#388, baseline
#418) and stays under 0.005 weekly up to k=8, so (A) means a Gate cannot adopt. The pre-registered
stop condition for (C) was fixed before (C)'s numbers (below), and the reading rule for the 2024
sub-question was stated before its number (the 2026-09-21 comment on #381).

## The measured null and power (rule 16, before landing)

`scripts/gate_horizon.py --rule-c`, #388's harness reading (A) and (C) off the same `gate` run on
the same frames, on #388's estimated process (#418's chosen baseline), 40,000 trials a cell,
worst SE 0.0025. Pre-registered in `docs/gate-power.md` (*(C) before it lands*) and committed
before any number existed; the full table, the four controls and the rows at k = 5..8 are there.

| k=4 | null ADOPT (A) | **null ADOPT (C)** | power (A) | **power (C)** | interval half alone |
|---|---|---|---|---|---|
| weekly, δ=0.3 (260 rows) | 0.0001 | **0.0140** | 0.0008 | **0.0829** | 0.1313 |
| weekly, δ=0.5 | 0.0001 | 0.0140 | 0.0039 | **0.2003** | |
| draft, δ=2.0 (20 rooms) | 0.0014 | **0.0291** | 0.0077 | **0.0828** | 0.0919 |

**The stop condition did not fire.** The maintainer's pre-registered bar was a null ADOPT above
0.05 at k=4 on either path; (C)'s is 0.014 and 0.029, and the largest null anywhere in the grid is
0.035 (draft, 80 rooms). (C) raises the null by a factor of over a hundred on the weekly path from
a base that was effectively zero, and it stays under `ALPHA`. Power at k=4 rises about a hundredfold
weekly and tenfold draft and is **not** a repaired gate: 0.08 is nowhere near `POWER` = 0.80, and
42% of weekly frames at δ=0.3 still have no resolved season at all. (C) makes a Gate *able* to adopt and
says so on the page when it could not decide. The (A) column reproduces #388's within 1 SE; the
inclusions (A) ⊆ (C) ⊆ interval-alone hold on every one of 1.5 million frames.

## What moves

- **Verdict sentences**, all of them, by the count phrase. The seam-equivalence golden
  (`tests/unit/fixtures/gate_seam_equivalence_control.json`) moved in twelve sentences and two
  statuses (`battery_tie` and `battery_threshold`, SHOW to ADOPT: three wins and an Abstention).
  `margin`'s frozen-read digest moved by exactly that phrase and returns to its old value with
  it removed.
- **Verdicts that change when the ledgered gates are re-run**, restated with their prior values in
  the doc that published each (`docs/gate-power.md`, `docs/improvements.md`, the box in each
  gate's own page): the weekly gate's #378 run, **SHOW (won 0, tied 1, lost 3 of 4) to REMOVE on
  3 resolved of 4, 1 abstained** (t interval [−2.032, −0.262]); the draft gate's #376 run, **SHOW (won 0, tied 1, lost 3 of 4) to
REMOVE on 3 resolved of 4, 1 abstained** -- on today's re-run, which is not a bit-for-bit re-reading
(2025's gain is −14.34 where #376 recorded −19.86, so the mean is −11.02 against −12.40 and the
MDE 7.84 against 11.01; 2022-2024 and the ceiling +21.90 are identical; t interval [−17.22,
−4.82]), and which #376's own recorded numbers (t interval [−21.11, −3.69], three losses and one
Abstention) give the same reading on. *(Attributed 2026-10-06, #429: the gap to #376 is an input
that run held and did not record -- #376's own commit re-run on today's data gives today's numbers
to the cent -- so today's run, replayed three times, is the one that stands; `docs/gate-power.md`.)*
`player_spread` own_k
  and usage (SHOW; **0 resolved of 5, 5 abstained**: the reason moves from "a tie blocks ADOPT" to
  "no season resolved"), `injury_type` (SHOW, 3 resolved of 3, 0 abstained) and
  `component_calibration` (NOT-RUNNABLE) do not change status.
- **The ledger**: each run writes `resolved` and `abstained` beside the verdict
  (`state/README.md`).
- **The `wins k/k` restatements of #311** are re-read under it: a Gate whose record said "won 3,
  tied 1" now says what that was, three resolved seasons of four.

## The 2024 sub-question, answered

*Is the tie rule abstaining on noisy seasons rather than on small effects?* The question was asked
because 2024 abstained in both the draft and the weekly harness. **Weekly** (#381's 2026-09-21
comment, from #382's run; m = 20 each): within-season SEs 0.409 / 0.447 / **0.543** / 0.546 for
2022-2025 and gains −1.120 / −1.448 / **−0.259** / −1.761. 2024's SE equals 2025's (0.543 against
0.546), so it is not distinctly wide, and its |−0.259| is below **every** season's 2·SE (the
narrowest is 0.82) by a factor of three: 2024 abstains at any of the four SEs, while 2025 at
the same precision resolves as a loss because its gain is seven times larger. **The weekly
Abstention is about the effect, not the season.** Re-run today (#381, the same recipe), the
per-season table reproduces to the digit. **Draft** (the first draft run to record per-season SEs, today's re-run
of #376's recipe, m = 20 rooms each): SEs 3.31 / 3.11 / **3.75** / 2.03 and gains −15.35 / −8.41 /
**−5.99** / −14.34 for 2022-2025. Here 2024's SE *is* the widest (13% over 2022, 21% over 2023, 85%
over 2025), so the weekly reading does not simply carry over, and the draft answer is mixed: its
|−5.99| is also the smallest gain of the four, and it would still abstain at 2022's and 2023's SEs
(thresholds 2·SE of 6.62 and 6.22 against 5.99, a ratio of 1.81 and 1.93 to the SE where a win
or loss needs 2.0), and resolve as a loss only at 2025's 2.03. **The draft Abstention is about
both the effect and the season** -- a small gain and the widest SE in the run -- which is the
property the 2026-09-21 amendment said would need saying out loud if it appeared. It appears
here, at one season in one gate. The second column the 2026-09-21 comments asked to be watched,
`max(se) / min(se)` within a run, reads **1.34 weekly** (0.546 / 0.409) and **1.85 draft**: the
threshold is close to a fixed-width dead zone weekly and clearly not draft. One season in one
gate is a hint, not a finding,
and the abstention rate that (C) is measured on (0.8 weekly, 0.58 draft per season at δ in the
gate's range) is exactly the quantity this sub-question concerns: it is a property of how wide
a season's own noise is against the effect, which (C) now prints beside every verdict.

## What it does not do

It does not repair power (see the table); it is not #375, which replaces the verdict with a
posterior and supersedes (C); and it does not touch NOT-RUNNABLE, VOID, the precondition, the
t interval of #357 or the `2·SE` / `TIE_MIN_CLUSTERS` choices of #335, which the 2026-09-21
amendment said were not the lever and still are not.


# Amendment, 2026-10-07: the injury comparisons' within-season unit is the player (#360)

**Amended rather than replaced.** Nothing above is withdrawn; the per-gate table in the
2026-09-21 amendment is kept as written, including its `injury type` row (`season`, a declared
no-op, "deliberately so" while `#360` was frozen), and this section is what supersedes that row.

**What it discharges.** The 2026-09-21 amendment landed the injury type comparison with its
within-season unit set to `season`, one cluster a season, so every season read on its sign alone,
and said that was #360's to choose. #360 chose it on 2026-10-07: the unit is **`gsis_id`, the
player** (method.md rule 3: a player is designated in many weeks of one season), the unit #335
adopted for this gate. `injury.HARNESS` (type-adjusted against retention) now declares
`within=("gsis_id",)`.

**What it adds.** `injury.verdict` was not a Gate: an argmin over four candidates with no
interval and no every-season half, the lowest bar in the repo, and the module's one ADOPT rested
on it. It is now `experiment.run_gate` over two arms, `retention` against `out_zero` (`table` and
`baseline` are diagnostics, reported and not gated), declared as `injury.RETENTION_HARNESS`,
`within=("gsis_id",)`, with the in-sample per-cell retention as its declared ceiling arm and the
three actions pre-registered in `docs/weekly-injury.md` before the run. The rule itself
(`experiment.gate`) is unchanged.

| gate | within-season unit | why |
|---|---|---|
| injury type (`hub.models.injury.HARNESS`) | `gsis_id` | the player; supersedes the declared no-op |
| injury retention (`hub.models.injury.RETENTION_HARNESS`) | `gsis_id` | the player |

> **Superseded figures, 2026-10-09.** The −11.02 / MDE 7.84 quoted in the 2026-10-06 amendment
> are restated in [gate-power.md](../gate-power.md): first for #315 item 1 (−12.58, MDE 8.64),
> then for #467, whose 2025 season had priced absence at zero (−13.49, MDE 9.90, 2025 −18.00).
> The verdict is REMOVE, 4 resolved of 4, unchanged. The text above is kept as written.
