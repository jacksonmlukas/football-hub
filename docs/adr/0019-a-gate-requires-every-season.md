# A Gate requires every season, not only the interval

**Status:** accepted 2026-09-04.

**Decision.** A **Gate** adopts only when the pooled interval excludes zero **and** the sign
holds in every held-out season. It removes only when both hold in the other direction.
Everything else shows and is never ranked on. One implementation, `hub.models.experiment.gate`,
read by every gate in the repo.

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
