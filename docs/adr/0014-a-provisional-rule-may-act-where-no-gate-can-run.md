# A provisional rule may act where no gate can run

**Status:** accepted 2026-08-27.

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

## Consequences

* The waiver tiebreaker — ECR chooses the candidate set, snap delta chooses within it — becomes
  a provisional rule, logged per claim.
* The survivor contrarian threshold (take a differentiation week when the win-probability cost
  is under ~8pp) becomes a provisional rule on the same terms.
* Logging is not bookkeeping. It is the only mechanism by which a provisional rule can ever stop
  being provisional, and it is the same move the ADP archive made on 2026-08-25: data not kept
  on the day is not recoverable later.
* Any public write-up reports these as judgment. An unlabelled provisional rule would destroy
  the thing this repo is actually for, which is that its claims are gated and its record is
  honest about which ones are not.
