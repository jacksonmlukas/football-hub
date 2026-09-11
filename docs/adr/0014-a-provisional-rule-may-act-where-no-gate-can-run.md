# A provisional rule may act where no gate can run

**Status:** accepted 2026-08-27. **Amended 2026-09-07** (issue #51) — criterion 2 is closed
against a reclassification; the five requirements are otherwise unchanged. **Re-scored
2026-09-07**: requirement 4 is currently discharged by no code, see below. **Amended
2026-09-11** (issue #209): the survivor threshold's logged quantity is named, see below.

**Decision.** A decision rule may be adopted without passing a gate **only** when all five hold:

1. the underlying signal **passed a screen** — it is real beyond ECR;
2. **no gate can run at available n** — the obstacle is sample size, not an adverse result;
3. the rule is **written down before it is used**, not described afterwards;
4. **every application is logged** — the rule's recommendation, what was actually done, and
   what happened;
5. a **horizon** is stated at which it will be judged.

Such a rule is called a **provisional rule**. It is never reported as validated, and the word
"provisional" travels with it into any write-up.

## Why this exists

Objective 1 is to win the league. The standard everything else here is held to is a
pre-registered gate on held-out data. On 2026-08-25 those two came into direct conflict for
the first time.

The snap-share trend screened positive — partial **r = +0.236** beyond season-to-date scoring
*and* rest-of-season ECR, twelve of twelve season-anchor cells, placebo-clean, and stronger in
the waiver-relevant tail. Consensus prices essentially none of it.
[ADR-0013](0013-the-snap-trend-is-shown-and-never-ranked-on.md) then measured the *decision*
built on it and could not resolve it: a single waiver claim has a standard deviation near 20
points over three weeks, so a one-point edge needs roughly 1,800 paired decisions — about
**44 seasons**. Four exist.

So the position without this ADR is: hold a signal your opponents demonstrably do not use, in
the decision with the weakest opponent in the league, and decline to use it — forever, because
the horizon never arrives.

**A standard of "never act on what cannot be validated" is, in a domain where almost nothing
validates at one season of n, a standard that guarantees you never act on anything.** Including
the one screen in six that came back positive.

## The loophole this closes

"Act anyway and log it" is now a move that exists, and the obvious risk is that it becomes
available to anything that has ever failed a test. It is not, and the line is a category rather
than a degree:

* **Championship equity failed a gate that ran.** −19.66 points per team game, n=80, losing in
  all four seasons. [ADR-0009](0009-championship-equity-does-not-pick.md) already says
  reopening means re-running the harness, not re-arguing. It is **permanently excluded** from
  this mechanism.
* **The snap trend never had a gate that could run.** The harness is not capable of resolving
  it at any n this project will ever see.

A gate that fired against you is *evidence*. A gate that cannot run is an *absence of evidence*.
Only the second is eligible.

### Amendment, 2026-09-07: a reclassification is not a second door into criterion 2

Issue #51, following the precondition
[ADR-0019](0019-a-gate-requires-every-season.md) gained on 2026-09-07 under #45.

That precondition made **NOT-RUNNABLE a computed branch**: a Gate whose minimum detectable
effect exceeds its measured ceiling reports that it cannot run, and reports no verdict. When
this ADR was written, "no gate can run" was a standing property of a question — the snap trend's
harness cannot resolve its decision at any *n* this project will see. It is now also something a
gate that *already ran* can be reclassified as, by measuring a ceiling that did not exist
before.

Read literally, that is a second door into criterion 2, and it opens onto exactly the thing the
loophole section above closes. Championship equity is **permanently excluded** because a gate
fired against it at −19.66. If that gate's ceiling were later measured and its MDE found to
exceed it, the gate would report NOT-RUNNABLE — and criterion 2, read as a status check, would
read as satisfied. A mechanical reclassification would have erased a recorded adverse result and
made a failed decision eligible for provisional adoption, without anyone deciding anything.

**So, added ahead of criterion 2 and taking nothing away from it:**

> **A Gate reclassified NOT-RUNNABLE is not, by that fact, a gate that never ran.** A
> NOT-RUNNABLE reclassification withdraws a *verdict*; it does not withdraw the *observations*
> that produced it. Where an adverse result is on the record, criterion 2 is not satisfied by
> the reclassification alone. It is satisfied only by a re-measurement, on a harness able to
> resolve the question, that does not come back adverse.

**Why this direction and not the other.** A reclassification is real information and it is
information *against* the gate, not for the arm: it says the run should not have been asked, and
the honest consequence is that its verdict stops being quotable. Nothing about that makes the
arm look better. Treating the same fact as a licence to act on the arm would take a result this
repo paid to measure and convert it, by arithmetic, into permission — which is the shape of
every failure the five requirements exist to prevent. The five requirements are unchanged; this
adds a precondition ahead of the second, in the same way #45 added one ahead of ADR-0019's two
halves.

**What it does not move.** The eligibility table below is reached identically. The snap trend
qualifies for the reason it always did — its harness cannot resolve the decision at any
reachable *n*, which is a property of the design and not a reclassification of a run. No entry
in that table changes state under this amendment, and the count is still one.

## How many things qualify today: one

This is the part worth checking a year from now, because it is the evidence that the mechanism
is an exception rather than a policy.

| candidate | eligible? | why |
|---|---|---|
| Snap-share trend | **yes** | screened +0.236; decision needs ~44 seasons |
| Championship equity | no | failed a gate that ran, at −19.66 |
| VOR ordering | no | failed, at −5.06 |
| `edge` | no | never screened — it cannot be, without historical ADP |
| Opponent correlation | no | measured, but has no decision attached to act on |
| Per-player weekly spread | no | screened and found ~absent: ±9.3%, 85% noise |

**One.** If that column ever fills up, the mechanism has stopped being an exception and this
ADR should be revisited rather than stretched.

## The horizons and the logs, stated

Requirements 4 and 5 above are obligations on *this document*, not aspirations. Until 2026-08-27
it did not meet them, which made the horizon clause the one part of its own rule the ADR
violated. Here they are, per rule.

> **Re-scored 2026-09-07 (issue #51): requirement 4 is discharged by no code.** What follows
> specifies the log completely, and `hub.season.journal.record` implements it — with real
> invariant checks written against `pool.Weekly`'s field names. A call-site census across
> `src/` finds **no caller**; `pool.py` mentions it in prose at two places and nothing invokes
> it. So the one requirement whose whole purpose is to let a provisional rule stop being
> provisional is, today, met by a specification and a module rather than by a log.
>
> This is not a re-decision — the requirement is right and the module matches it. It is the
> observation that the requirement is unmet in the tree, and that its invariant checks are
> exercised only by hand-built fixtures, so a units or sign-convention mismatch with a real
> `Weekly` would surface the first time it is wired up rather than now. **Owned by #204.**
>
> It bears directly on the horizon below. A horizon of *150 logged claims* is counted from a
> log; a log that no code writes reaches 150 never, which converts the weaker of the two
> provisional rules into a permanent one by omission rather than by argument.

### The waiver tiebreaker

**Horizon: 150 logged claims, whenever that arrives.** A count, not a calendar — power is what
determines when you may conclude, and the number of Januaries is not. At roughly 14 claims a
season that is about a decade, and it is deliberately not reachable early: a January judgement
on fourteen claims would conclude something from nothing, which is the failure this whole
document exists to prevent.

**And 150 is not 1,800.** It is the point at which *looking* becomes informative, not the point
at which the question resolves. If the accumulated difference at 150 is not clearly signed, the
honest report is still "unresolved" — reaching a horizon is permission to look, not permission
to decide.

**What is logged, per claim** — and only this:

| field | why |
|---|---|
| date, week | orders the record |
| the rule's recommendation | the top of the ECR-chosen set by snap delta |
| what was actually claimed | the decision |
| why, if those differ | **unrecoverable by any replay** |

**What is deliberately not logged.** Points scored afterwards by anyone claimed or passed over:
nflverse carries those forever, and typing them at 11pm on a Tuesday buys nothing. The candidate
pool: reconstructible from the board plus rosters, so it belongs in the replay rather than the
log.

The principle is the one the draft-night deviation protocol reached independently a day later:
**log only what no replay can recover.** A log heavier than that stops being kept by week five —
and a log that stops being kept is worse than no log, because it still looks like evidence.

### The survivor contrarian threshold

**Horizon: none that is reachable, and saying so is the point.** One pool a season, eighteen
decisions a year, each resolving to a binary outcome dominated by a single game. There is no
count at which this becomes measurable on any timescale this project will see.

It is adopted anyway, because the alternative — maximising P(survive every week) in a 20-100
entry pool — is *known* to be answering the wrong question, and a rule that is probably right
beats one that is definitely mis-specified. But it is the weaker of the two provisional rules by
some distance, and if this mechanism is ever questioned it should be the first thing
reconsidered.

> **Amended 2026-09-11 (issue #209): which column is the threshold quantity.** The rule is
> stated in win probability on the *week* — "under ~8pp" — and the journal that discharged its
> logging duty recorded `survival_given_up`, a **season-survival** difference. On the pinned grid
> the two disagree in sign: the chalk is 2.0pp better on the week and 27.8pp worse over the
> season. That is the survivor thesis working — a worse pick this week for a better season — and
> it is exactly why logging one under the other's name was the error, and why the cost the rule
> fires on was not recoverable from its own log.
>
> A survivor pick is a week decision, so the threshold stays on the week. **The threshold
> quantity is `journal.week_cost`, the fallback's win probability minus the chosen team's**, and
> that column is what discharges requirement 4 for this rule; `journal.fallback_price` and
> `journal.market_price` are the two prices it is the difference of, and a row that carries one
> without the other is refused. `survival_given_up` stays beside it under its own name as the
> thesis of our plan working — not the threshold quantity, and not dropped, because dropping it
> loses the argument for the plan. The threshold itself has not moved, so nothing is re-derived.
> `tests/unit/test_journal.py` holds this line against the schema: if this ADR is re-pointed at
> another column, or the column's arithmetic changes, that test fails.

## Consequences

* The waiver tiebreaker — ECR chooses the candidate set, snap delta chooses within it — becomes
  a provisional rule, logged per claim.
* The survivor contrarian threshold (take a differentiation week when the win-probability cost
  is under ~8pp) becomes a provisional rule on the same terms. Its logged cost is
  `journal.week_cost`, on the week, since 2026-09-11 (#209).
* Logging is not bookkeeping. It is the only mechanism by which a provisional rule can ever stop
  being provisional, and it is the same move the ADP archive made on 2026-08-25: data not kept
  on the day is not recoverable later.
* Any public write-up reports these as judgment. An unlabelled provisional rule would destroy
  the thing this repo is actually for, which is that its claims are gated and its record is
  honest about which ones are not.
