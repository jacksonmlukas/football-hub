# A modelling decision becomes agent work when its alternatives can be measured side by side

**Status:** accepted 2026-09-09. Decided under the grilling session that unblocked #155, #178,
#179, #180, #183, #184, #186 and #206.

**Decision.** When a ticket is blocked on a modelling choice, the default disposition is to
**measure the alternatives side by side and report the sensitivity**, not to pick one. Picking
one is reserved for the cases where the alternatives change what the number *means* rather than
what it *is*. A ticket in the first class is `ready-for-agent` and its deliverable is the
sensitivity; a ticket in the second class is `ready-for-human` and stays there until decided.

## Why it was proposed

Nine open issues carried the `ready-for-agent` label while their own bodies said **"Not ready
for an agent."** None had a single comment. Whatever converted the labels never touched the
bodies, so the tracker's most load-bearing field disagreed with the text underneath it on
thirteen per cent of the open set — and the disagreement was invisible to both a human skimming
labels and an agent reading the body.

That is this repo's recurring defect in its triage layer rather than its code: **a check whose
validity depends on context it cannot see.** The label cannot see the body.

## Why the default is measurement rather than choice

**Because it has already been run once, and it changed the answer.** #206 was split at exactly
this seam: the decision-free half — *report the gate's result under all three fallback
treatments* — became #207 and was implemented. Running it is what established that the spread
across treatments was **1.5 points, larger than any effect any single treatment reported**. No
amount of deciding which fallback was correct would have surfaced that; only measuring all
three did.

The pattern generalises to any ticket whose alternatives differ in a parameter rather than in
kind. #178 is the clearest instance: the anchors 4, 6, 8, 10 and 12 were *already measured* for
`docs/snap-trend-signal.md`, so re-running the screens at each costs almost nothing and answers
whether the choice ever mattered. A stable feature set across the range retires the question
with evidence; an unstable one is a finding worth more than the constant.

**And it is already sanctioned.** `docs/method.md` rule 12 permits acting provisionally where
no gate can run, and [ADR-0014](0014-a-provisional-rule-may-act-where-no-gate-can-run.md)
requires a provisional rule to record what would settle it. A sensitivity across the
alternatives *is* the record of what would settle it, produced rather than promised.

## Where it does not apply

Three of the nine stayed decisions, and the line between them is not difficulty:

- **#184** — measuring the flex shares does not change their value, it changes their **kind**.
  A measured constant leaves `HubConfig` under [ADR-0006](0006-fitted-constants-live-with-their-provenance.md)
  and stops being Hydra-overridable. A sensitivity across candidate values would report numbers
  while dodging the only question that matters.
- **#183** — whether absence is a games-played draw, a zero-inflated week model or a mixture
  component changes what every downstream constant *is fitted against*. The alternatives are not
  points on an axis.
- **#186** — the alternatives presume a correction whose held-out performance is currently
  worse than applying none. Measuring refinements of a negative is chasing a gap below the
  ceiling, which rule 8 forbids.

The test is therefore: **can the alternatives be placed on one axis and reported in one table?**
If yes, the ticket is agent work and the table is the deliverable. If the alternatives are
different objects, it is a decision and stays one.

## What this obliges

A ticket moved to `ready-for-agent` under this ADR **must have its body updated**, not only its
label. The nine that prompted this were mislabelled precisely because the body was left saying
the opposite; a label that disagrees with its own body is worse than an untriaged one, because
it reads as settled.

## What would reopen this

A sensitivity that a reader cannot act on — a table of alternatives that leaves the choice
exactly as open as it was, at the cost of a measurement run. That would be evidence the seam
was in the wrong place, and the ticket should have been a decision.
