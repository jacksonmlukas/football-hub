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
