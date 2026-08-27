# The method

The interesting problem in fantasy football is not predicting football. It is **knowing whether
your predictions are any good** — in a domain with ~285 NFL games a season, an efficient market
sitting there as the benchmark, no ability to run an A/B test, and a season of n=1 to judge
yourself on.

Almost every rule below exists because a specific mistake was made here first. That is the
point of the page: the method is not recited from a textbook, it is what is left after the
errors were caught. Each rule names the incident that produced it.

---

## Two questions, and they are not interchangeable

The single most expensive confusion in this repo was treating these as one question.

**A signal** claims some quantity predicts outcomes *beyond what consensus already knows*. It is
tested by a **screen**: partial correlation against expert consensus rank, on held-out data.
Asks *"is this real?"*

**A model** produces a projection or a decision. It is tested by a **gate**: does it beat the
simplest thing that already works? Asks *"is this better than what it replaces?"*

A thing can pass one and fail the other, and the repo has an example of exactly that. The
snap-share trend screened at partial **r = +0.236** beyond both season-to-date scoring *and*
rest-of-season consensus — real, twelve of twelve season-anchor cells, placebo-clean. Then the
decision built on it was gated against "take the best available player by consensus rank" and
lost at **−3.81 points**, because a partial correlation says a quantity *adds to* the board and
never that it should *be* the board.

Confusing the two is how a well-built, well-tested model ships while being confidently worse
than a one-line rule.

---

## The rules, and the mistake behind each

### 1. Fix the decision rule before the numbers — in code, and test it

Every gate's branches are written as a `verdict()` function and unit-tested, *including the
branch where the elaborate thing loses*, before the measurement runs.

**The incident.** A gate's `verdict()` documented a two-part rule — beat the incumbent in every
held-out season **and** clear 2 standard errors — and implemented only the first half. Its first
run printed `ADOPT` on a gain of 0.0065 MAE, 0.6% of baseline. A rule that is documented but not
implemented is worse than no rule, because it gets quoted in the write-up.

**And the harder version.** A tripwire fired against the thing its author was building. It was
argued past — narrowed, with a reason that sounded good — and the very next measurement produced
exactly the preference it had flagged, at −19.66 points per team-game. Both the amendment and
its vindication are in the record ([ADR-0009](adr/0009-championship-equity-does-not-pick.md)).
The rule is not "have a tripwire". It is *do not touch it after you have seen what it caught.*

### 2. Measure the predictor strictly before the outcome window

**The incident.** Depth-chart climb appeared real at **7.4 sigma** — because the climb was
measured *inside* the window it was predicting. Honestly measured, it is null at every horizon:
partial r = +0.008 beyond consensus. Obvious, embarrassing to restate, and it happened anyway in
an analysis that bypassed the store layer built to prevent it.

### 3. Repeated measures are not independent observations

**The incident.** Eight weeks of the same player is not eight data points. Pooling them turned
noise into an apparent 4-sigma result. Screens now report one row per player per anchor, and
never pool anchors.

### 4. A significant result whose sign flips between seasons is a bug, not a finding

The cheapest diagnostic available, and it was the tell in both errors above.

It is also why gates here have **two** halves rather than one. An injury-type adjustment cleared
significance at 3.1 se and still failed, because it won 2 of 3 held-out seasons and the one it
lost was the most recent. The every-season half is not a second hurdle for its own sake — it is
the half that catches sign flips.

### 5. Gate against the simplest thing that works, not against the null

**The incident.** A weekly injury table was gated against two rules: *ignore the report* and
*bench anyone ruled Out*. Beating the first shows injuries matter, which nobody doubts. The
second — the rule every manager already follows for free — is the one worth beating. The table
beat it by **0.170 MAE at 3.8 se** and was adopted. An earlier, additive version of the same
table lost to it, and the reason is the finding: an Out player scores exactly zero, which a
multiplicative form expresses and an additive one cannot.

### 6. Both arms must have the same information

**The incident.** A lineup-optimiser gate returned **+31.15 points per game** for the optimiser.
The optimiser arm was choosing lineups from *realised* scores while the baseline used
projections. The number was the value of perfect foresight. Rebuilt honestly, the answer is
**+0.00** ([ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md)).

**A gate whose treatment arm has information its control arm lacks cannot fail.**

### 7. Pair the comparison

Score both arms on the same games, the same drafts, the same random draws — common random
numbers. The standard error of a *difference* is far tighter than the difference of two standard
errors, and it removes the question of which arm got the easier sample.

### 8. Compute the ceiling before chasing the gap

**The incident.** Per-player weekly spread looked like a promising target: the observed scatter
in volatility across players is ±25.9%. But the *outcome* being predicted is a realised standard
deviation from ~14 games, which carries its own sampling error of **1.0113 MAE** against the
shipped model's **1.0965**.

Total headroom for every future model combined: **0.085**. Correcting for reliability, the true
per-player effect is ±9.3% and a single season recovers 15% of it. The question was closed by
the ceiling, not by any one candidate's failure — which is a much stronger form of "no".

### 9. A result too large to believe is a bug until shown otherwise

**The incident.** A weekly injury coefficient nearly shipped as a 5× discrepancy against an
existing constant. They were different quantities — one a *season-long* per-game cost, the other
a *within-week* one. A player ruled out in week 1 misses week 1 and plays the other sixteen.
Comparing them would have been nonsense.

### 10. Coverage measures lines executed, not whether the seam exists

**The incident.** Three bugs found in one day, all in code at or above 80% coverage: a CLI that
raised `ConnectionError` instead of serving the board on disk; one that died on
`BinderException` for a column that had never existed; one that raised `FileNotFoundError` for
a file nothing writes. All three lived where a module meets the real world, and every test had
handed the function its own frame.

**A fourth, worse one.** Two tests written to catch this passed only because *the developer's
machine had data* — they would have failed in CI, which checks out fresh. The repo's own
headline command, `make draft`, had never once worked on a fresh clone.

### 11. Name the exclusions

Provenance is a hash of the resolved config *and* every fitted constant, so refitting a
coefficient moves the model version. When five constants turned out to describe code no
prediction could reach, they were not silently dropped — they are listed one at a time in a
`NOT_IN_DIGEST` map with the reason. An exclusion should be a decision on the record, not a
module quietly falling off a list.

### 12. Where no gate *can* run, act provisionally — and say so

Some decisions cannot be validated at any n this project will see. A single waiver claim has a
standard deviation near 20 points over three weeks, so resolving a one-point edge needs ~1,800
paired claims — about **44 seasons**.

A standard of "never act on what cannot be validated" is, in that setting, a standard that
guarantees you never act on anything. So a **provisional rule** may be adopted when the signal
passed a screen, no gate *can* run, the rule is written down before use, every application is
logged, and a horizon is stated. It is never reported as validated
([ADR-0014](adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md)).

The line that keeps this from being a loophole: **a gate that fired against you is evidence; a
gate that cannot run is an absence of evidence.** Only the second is eligible. Championship
equity failed a gate that ran and is permanently excluded. Exactly one thing qualifies today,
and the ADR counts them — if that count ever grows, the mechanism has stopped being an
exception.

---

## The record

Fourteen things have been measured properly. **Two came back positive** — and one of those two
produced a decision that then failed its own gate.

| # | attempt | result |
|---|---|---|
| 1–2 | Expected-vs-actual points; recency-weighted | null — r = 0.21 self-persistence |
| 3–4 | Depth-chart climb, two horizons | null — +0.008 beyond consensus |
| 5 | Age | null |
| 6 | Championship equity as the objective | **−19.66** pts/team-game |
| 7 | VOR ordering | **−5.06** pts/team-game |
| 8 | `edge`, the repo's original signal | unvalidatable — needs ADP nobody retains |
| 9 | Volume model beating the market's mean | null |
| 10 | Lineup optimiser | **+0.00** — a structural zero |
| 11 | Per-player weekly spread | null — 0.085 MAE of headroom exists at all |
| 12 | **Weekly injury retention** | **adopted** — +0.170 MAE at 3.8 se |
| 13 | Injury type on top of it | null by the gate — 3.1 se but 2/3 seasons |
| 14 | **Snap-share trend** | **screen positive** — +0.236 beyond consensus |

What separates #12 and #14 from the other twelve is not sophistication — #12 is a nine-cell
lookup table of ratios. It is *what information they use*. The twelve failures all tried to
out-think a market using information that market had had all summer. #12 used Wednesday's
practice report to set Sunday's lineup; #14 used snap counts published on a Monday.

**Edge came from timeliness, not from better processing of shared information.**

---

## What it costs, and what it buys

It killed most of the work. Twelve of fourteen measurements ended in a removal, a demotion or a
null, and the components that were hardest to build — a nested draft-and-season simulation
optimising championship probability, a component-level projection layer, a lineup optimiser —
are the ones that lost.

What it buys is that the two things left standing are worth trusting, and that the repo can
state precisely what it knows, what it does not, and which of its own numbers are judgment
rather than evidence. Demonstrable market edge is an explicit non-goal here: the system audits
itself against markets and does not try to beat them.

The most useful artifact in this repo is the record of what was measured and then removed.

---

## Primary sources

| | |
|---|---|
| The two tests, defined | [`CONTEXT.md`](../CONTEXT.md) |
| Six screens, five null | [signal-screens.md](signal-screens.md) |
| The one positive screen | [snap-trend-signal.md](snap-trend-signal.md) |
| The one adopted model | [weekly-injury.md](weekly-injury.md) |
| Why the ceiling closed a question | [player-spread.md](player-spread.md) |
| Fourteen decisions, with their trade-offs | [`docs/adr/`](adr/) |
| Objectives, and how objective 1 is judged | [decisions.md](decisions.md) |
