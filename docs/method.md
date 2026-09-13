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
exactly the preference it had flagged, at −19.66 points per team-game as first measured (the
magnitude has since been withdrawn as quotable; the verdict has not — [ADR-0009](adr/0009-championship-equity-does-not-pick.md),
restated 2026-09-11). Both the amendment and
its vindication are in the record ([ADR-0009](adr/0009-championship-equity-does-not-pick.md)).
The rule is not "have a tripwire". It is *do not touch it after you have seen what it caught.*

**Where the rule lives in code.** The *statistic* is shared and the *rule* is not.
`hub.models.experiment.paired_gain` returns the four numbers a two-half gate reads — mean, se,
t, and seasons won — and decides nothing; each `verdict()` still spells out its own incumbent,
its own thresholds and its own sentences, because a pre-registered rule is specific to what it
decides and a shared one would drift toward being decorative. The significance bar is
`experiment.MIN_SE`, one name, after being declared twice as 2.0 (`spread.MIN_SE` and
`injury.TYPE_MIN_SE`), each commented "the repo's usual bar".

### 2. Measure the predictor strictly before the outcome window

**The incident.** Depth-chart climb appeared real at **7.4 sigma** — because the climb was
measured *inside* the window it was predicting. Honestly measured, it is null at every horizon:
partial r = +0.008 beyond consensus. Obvious, embarrassing to restate, and it happened anyway in
an analysis that bypassed the store layer built to prevent it.

**The second incident, and it is the quieter kind.** `wind` reached the weekly screen as a
week-*w* **pre-kickoff** feature with a pre-registered negative sign. It is read off the
schedule as *recorded conditions* — the observation at the game — so the predictor was measured
inside the outcome window, exactly as above, and this time nothing looked wrong: it came back
**null**, and a null is the result nobody audits. It was found by a re-audit reading the
classification rather than the number (#170, finding B12).

**And the null-filling is the half worth remembering.** A dome or an absent reading was coded
`0.0`, so "no measurement" and "no wind" were the same number on a feature whose sign was
pre-registered. On the frozen archive, **175 of 416 game rows carry no reading and not one game
has a measured wind of zero** — so every zero in that column was a non-measurement, and the
screen could not have separated the two because the measured population contained no zeros to
separate them from. A default that is also a legal value is not a default, it is a fabrication
with a plausible face; `hub.models.panel.injury_columns` had already made the same call the
other way, refusing to fill a missing injury report with `Healthy`.

Wind is now `RECORDED` on the Panel — observed during week *w*, describable after the fact,
refused as a predictor of it — the published figure is superseded and unestablished under rule
13, and the screened family moved from nine features to eight
([weekly-screen.md](weekly-screen.md)).

**Where the rule lives in code.** `hub.models.experiment.expanding_seasons` — one generator
yielding `(season, past, now)` where `past` is strictly earlier, and the only place in `src/`
allowed to write a `<` against the season column. It was written out four times until
2026-08-27 (`margin.walk_forward`, `injury.walk_forward`, `injury.walk_forward_type`,
`spread.walk_forward`), already differing three ways, and only one copy was pinned by a test.
Four hand-written copies of a leakage invariant is four places a `<` can become a `<=` — and
it would be silent, because a leaking model does not crash, it looks good. The AST guard
`test_the_split_is_written_once` is what stops the four copies coming back.

### 3. Repeated measures are not independent observations

**The incident.** Eight weeks of the same player is not eight data points. Pooling them turned
noise into an apparent 4-sigma result. Screens now report one row per player per anchor, and
never pool anchors.

**The second incident, in the module that enforces the first.** The weekly screen refuses to
pool player-weeks — its docstring says pooling would inflate every `t` by about √14 — and then
took its standard error across the 55 season-week **cells** while its `verdict()` requires the
sign to hold across the five **per-season means**. The precision came from dozens of cells and
the decision from five seasons, so the `t` on the page and the rule reading it were about
different quantities. Corrected over the seasons
([weekly-screen.md](weekly-screen.md), issue #169), and one status moved: the prior TD rate
against passing attempts, published as a null at 1.9 se, is a **broken pre-stated null** at
2.7 se across 5/5 seasons.

**The rule cuts both ways, which is the part worth keeping.** The correction *narrowed* five of
the nine intervals rather than widening them, because for these features the week-to-week
scatter inside a season is large and averages out while the between-season scatter is small.
Clustering is not a synonym for a wider interval; it is a claim about which observations are
exchangeable, and the number moves whichever way the data says. Choosing the unit by the
direction it moves the answer is the failure this whole page exists to prevent.

**A third form, 2026-09-11 (#161).** The survivor pool's leverage term was measured at five
concentrations for five candidates: twenty-five comparisons, two clearing two standard errors,
twenty-four positive in sign. Neither count is twenty-five observations. The rows are one seed,
so both arms meet the same game results at every concentration and the five candidates at one
concentration share the free pick's arm. `hub.season.pool.resolvable_on_the_axis` holds the
count of hits to the same bar as a single row -- and says it treats the rows as independent,
which overstates them -- and the sign count is reported as a direction, never pooled into a
finding. Two hits in twenty-five is what the null gives; the term is unresolved, not zero.

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
prediction could reach, they were not silently dropped — each says `not_an_input` where it
is written, with the reason beside it (`hub.declare`, #253). An exclusion should be a
decision on the record, not a module quietly falling off a list — and since #253 there is no
list to fall off: a constant declares `fitted`, `chosen` or `not_an_input` at its own line,
the digest is walked off those declarations, and a number nobody declared is refused by name.

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

### 13. A measurement that contradicts a published number is not finished until it moves

> **Discharged 2026-09-12.** All three incidents this rule was written on are closed, each by
> the published number moving: the season-clustered interval reached `README.md` and ADR-0009
> under #52; the +0.215 blend gate was re-run to **−1.004** under #44 and the record restated
> under #282; the pick-noise constants were refit under #150 and again under #155. The rule
> stands as a rule; what it no longer has is an open incident.

**A measurement that contradicts a published number is not finished until the published number
moves.** Measuring is the cheap half. The published figure is what anyone reads, so a refutation
that stops at the working note leaves a number standing that its own author already knows is
wrong, and the reader of the artifact has no way to see the correction.

**The incidents.** Three when this rule was written, all of them landing on 2026-09-06. A
fourth, of a shape this rule did not cover, is restated after them.

*A season-clustered interval, measured and reverted.* ADR-0009 publishes championship equity at
−19.66 pts/team-game, 95% CI **[−23.16, −16.20]**, bootstrapped over 80 rows. Re-run on the same
frame with the season as the unit of replication — four independent replications, not eighty —
it is **[−26.68, −11.45]**, at an MDE of 10.29 ([gate-power.md](gate-power.md), `2496a73`). The
verdict survives; the interval does not. The producer that would have printed it was *"tried,
reverted, and is recorded here"* because supplying the key moved a pinned digest, so
`experiment.summarise` still says nothing computes an MDE, and the narrow interval is still the
one in `README.md`, ADR-0009 and ADR-0019. Issue #45 is open. *(Discharged: #45 closed
2026-09-07 — `summarise` clusters on the season and prints the MDE, restated at 14.77 under the
t-quantile in [gate-power.md](gate-power.md); the season-clustered interval reached `README.md`
and ADR-0009 on 2026-09-11 under #52, ADR-0019 by its own amendment.)*

*A gate figure invalidated by its own fix, never re-run.* The weekly gate's treatment arm scored
negated consensus rank and fantasy points in one column, so every player it could project
outranked every player it could not, whatever either was worth. Repairing that changes the arm
under test, which the commit says itself: *"the frozen +0.215 will move — and re-running it is
the next action, not something this commit may claim"* (`7873de6`). Nothing re-ran it. **+0.215** *(re-run 2026-09-07 (#44): **−1.004**, verdict REMOVE — this incident is now discharged; see `docs/weekly-blend-gate.md`)*
is still the headline of [weekly-blend-gate.md](weekly-blend-gate.md), ADR-0016 and ADR-0017.
Issue #44 is open on its fourth criterion alone — *the gate's result is re-run and the movement
recorded* — with the code half verified done. *(Discharged: #44 closed 2026-09-07 with the
re-run; all three documents carry −1.004 beside the kept +0.215.)*

*An estimator repaired, its estimate left pinned.* `fit_pick_noise` carried two defects, both
inflating the fitted slope: it fitted against a consensus rank over a 300-plus-player board and
was applied to an expected pick number, and it refitted through `max(sigma_hat − a, 0)`, which
zeroes every residual under the pinned intercept and so fits the slope to the upper envelope of
the data. Both were fixed (`02488c0`). `PICK_NOISE_SLOPE` is still **0.253**, the value the
uncorrected fit produced (`6ffd302`) — held in place by a test asserting it, under a comment
describing a fit the code no longer performs, and the board's cost of waiting is still computed
from it. Issue #150 is open.

> **Restated 2026-09-07 — the third incident is closed.** `fit_pick_noise(league, [2022, 2023,
> 2024, 2025])` was re-run in the tree that holds the repair, and the shipped constants are now
> what it returned. **`PICK_NOISE_SLOPE` 0.253 → 0.169** and **`PICK_NOISE_INTERCEPT` 1.00 →
> 1.31**, with the slope's draft-clustered 95% CI **[0.159, 0.179]** — four drafts, not 672
> picks — published beside the point estimate for the first time. The population is named as
> well: **672 picks inside a 204-pick draftable pool** over four drafts, not the "734 picks
> across this league's 2022-25 drafts" the superseded comment claimed over a whole board.
>
> **The cause is the one this entry describes**, plus one it did not: 1.00 was exactly
> `MIN_SIGMA`, the floor `_constrained` pins the intercept at when the unconstrained line wants
> to go negative — so it was the constraint speaking, not the data, and neither shipped number
> had been identified by the corrected fit. The repaired intercept clears the floor.
>
> **The direction reverses.** The superseded pin said the fitted law was *wider* than the
> 2.0 + 0.18 prior deep on the board, making that prior over-confident about who survives.
> Repaired, it is narrower than the prior at every pick in the pool — at pick 100, sigma 18.2
> against the prior's 20.0 and the superseded fit's 26.3 — so availability falls and
> `cost_of_waiting` rises, pushing the board toward scarcity rather than away from it. A board
> built either side of the change, naming the players whose availability moved, is in
> [pick-noise.md](pick-noise.md). Issue #150 is closed; #44 and #45 are still open.

> **Restated 2026-09-10 — a fourth instance, and it is the case this rule did not cover.**
> Issue #167 corrected what `rank_tiers` divides a tier gap by: two `lift_se` values combined
> in quadrature — the formula for a difference between *unrelated* estimates — became
> `lead_gap_se`, the standard error of the paired difference across the same simulated
> seasons. On correlated lifts the two differ by a factor of two; on uncorrelated ones they
> agree to five significant figures. Three published claims had been measured under the
> superseded expression: the Nacua/McCaffrey tie and its 5-of-5 seed count in
> [decisions.md](decisions.md), the co-leader counts in [next.md](next.md), and the
> `rank_tiers` docstring's own illustration. All three are restated beside their originals
> under #189.
>
> **None of the three could be re-run, and that is the new part.** The boards they were
> measured on were built 2026-08-24 off live ESPN ADP, which nobody retains
> ([ADR-0010](adr/0010-edge-is-displayed-but-never-ranked-on.md)) and which this repo began
> archiving on 2026-08-25; `lead_gap_se` is the spread of a per-future difference and nothing
> persists the futures; and the constants the simulator draws opponents' noise from have moved
> since (#150), so a re-run would produce a new claim rather than a restatement.
>
> **What the rule says in that case.** It says the replacement must come from a re-run and
> never from an argument about which way the old figure would have gone — so where no re-run
> is possible, the number moves to **superseded and unestablished**, and that is the whole of
> what "moved" can mean. Not to a corrected value, and emphatically not to the old value with
> a note that it is probably fine. The docstring's illustration was not re-measured either; it
> was **replaced** by one that can be, and is — it now cites the committed fixture the suite
> re-derives on every run, so it can never again be a number nobody is able to check.
>
> **The lesson is upstream of the restatement.** A measurement whose inputs are not pinned can
> be contradicted and never corrected, which is ADR-0007 one level down, and the same gap
> [next.md](next.md) already records against `--diagnose`.

**What "moved" means.** Not an edit over the top of the old figure: a superseded number keeps
its original text here, the same convention [architecture.md](architecture.md) states for a
superseded decision. It means a **dated restatement beside the original** — the superseded
figure, the cause, and whether the verdict moves, said explicitly — in the form
[weekly-blend-gate.md](weekly-blend-gate.md) already carries twice. And it means the new figure
comes from a **re-run**, not from an argument about which way the old one would have gone.

**What it costs.** Rule 8 closes a question by computing the ceiling; rule 12 acts where no gate
*can* run, and says so. This is the third case, and the most expensive of the three, because
here the gate ran and answered. The whole cost of measuring was paid and then discarded at the
last step, and what is left on the page is a number this repo's own record contradicts. A
project whose most useful artifact is its record of what was measured and removed cannot keep
the removals in its commit messages.

### 14. Count the family, and print the threshold beside the rule — not instead of it

A screen over eight features is eight tests. Eight verdicts at a bar of 2 standard errors
are eight chances for noise to clear it, and a reader of the verdict column is owed the count
and what the count does to the bar.

**The incident.** The weekly screen's family size was quoted in a docstring — *nine features*
— and moved to eight when wind was withdrawn (#170) without any number on the page moving
with it, because nothing on the page depended on the count. A multiple-comparison family that
lives only in prose is one that understates itself the first time a feature is added or
removed, and nobody notices because no figure changes.

**What the run prints now (#37).** Every screen run reports, for each family it screens, how
many tests it ran and the Benjamini–Hochberg threshold controlling the false discovery rate at
**q = 0.10** — `experiment.FDR_Q`, a stated choice and not a fitted constant — with the
two-sided *p* of the season-clustered *t* and its adjusted value beside every row. Each
screen counts **its own family**, never the union across screens: the alone screen's family
is every feature in the pool, the joint screen's is that anchor's survivors, the Usage
screen's is every (feature, count) pair. A feature the screen could not measure still counts;
it ran.

**Why it does not decide.** The verdict is the pre-registered rule and only that: a stated
sign holding in every season and the season-clustered *t* clearing `MIN_SE`. A feature that
clears the rule and sits above the threshold is reported as **clearing**, with the adjusted
result beside it — because a rule written down before the run is the whole reason the screen
is trusted, and swapping it for a threshold after seeing the vector would be rule 1's incident
in a new coat. What the threshold does is make the multiplicity visible on the same line. The
arithmetic makes the relationship plain: at five seasons the bar of 2 se is a two-sided *p* of
0.116 on four degrees of freedom, and a BH threshold can never exceed *q*, so a feature at
exactly the bar is one the rule clears and the threshold never does, at any family size. The
rule's every-season half is what carries the burden the bar alone does not, and that is
already rule 4.

**The tally.** The counts below are read off the code by
`tests/contracts/test_method_tally_matches_the_screens.py`, so a feature entering or leaving a
family moves this table or fails the build.

| screen | tests in the family | where the count comes from |
|---|---|---|
| weekly screen, alone (`--run`) | **8** | `len(weekly_screen.FEATURES)` — `snap_trend` counted, unlicensed (#248) |
| weekly screen, alone, `--routes` | **9** | the eight plus `ROUTE_TREND` |
| weekly screen, alone, `--scheme` | **13** | the eight plus `len(SCHEME_TRENDS)` |
| weekly screen, joint | the survivors at that anchor | printed by the run; not fixed here |
| weekly screen, Usage | survivors × **5** | `len(panel.USAGE)` counts per feature |
| preseason screens, by hand | **6** hypotheses, no threshold printed | [signal-screens.md](signal-screens.md); no code ran them |

The preseason row is the honest gap: those six were run by hand before the protocol existed,
have no module, and so print no threshold. Their family size is the count the index page
states and this table repeats; a reader wanting the adjusted result for the age screen has the
five per-season *r* values on that page and can compute it. No table on
[weekly-screen.md](weekly-screen.md) has been re-run under this rule yet — the printed
threshold arrives with the next `--run`, and until then the page says what it will print rather
than what it printed.

### 15. A fixture that sets the condition under which the estimator is trivially correct is not a test of the estimator

**Plant the condition the estimator must handle, not the one it cannot get wrong.** A test's
fixture is a claim about the input the code exists for. When every fixture sets the one input
on which the formula collapses to the right answer by algebra, the suite is green, coverage is
full, and the estimator has never once been exercised.

**The incident (#268, 2026-09-12).** The quarterback adjustment shipped as an *arrival-time*
baseline subtracted from a *current* value, decayed by tenure. The difference carries the
starter's own value drift since he arrived — nothing to do with the gap the module exists to
price. Measured on the 32 live teams: mean absolute error 0.4 spread points, worst 3.6, four
sign flips; Baltimore, an established starter with no quarterback change, at **+4.018** where
the source says **+0.404**. The error is exactly zero when tenure is zero, because then the
arrival row *is* the latest row and the formula reduces to the right one — and **every fixture
in the unit tests set tenure to zero.** Three named tests, each mutation-proved, each proving
the estimator on the one input it could not get wrong.

**The instinct the repo already had, pointed at the wrong modules.** `hub.draft.tune`'s
harness is validated before it is trusted: its tests plant a signal that genuinely predicts
and check the sweep finds it, then plant pure noise and check it returns zero. That is the
shape — the fixture carries the thing the estimator has to *do*, and a version of the estimator
that does nothing fails it. A quarterback fixture with a starter whose value drifted since he
arrived, and tenure above zero, would have failed the shipped formula on its first run. The
mutation discipline (#197's pin, the guard excisions) catches a test that proves nothing about
the code; this rule is about a test that proves the code on nothing.

**What to ask of a fixture:** what input does the estimator exist to handle, and does this
fixture contain it? If the answer is a special case where the code is right by construction,
the test is coverage, not evidence.

---

## The record

Fifteen things have been measured properly. **Two came back positive** — and one of those two
produced a decision that then failed its own gate.

**This table is where the count lives.** [where-to-look-next.md](where-to-look-next.md) was
written at fourteen and [what-the-field-knows.md](what-the-field-knows.md) at fifteen, each
dated; neither restates the count as current, and row 15 is the one those two pages pre-declared
the fifteenth (#52, 2026-09-11). Measurements taken since 2026-08-29 that this table does not yet
row — touchdown luck zeroed, durability shipped, the lambda sweep re-run, pick noise refitted —
are the separate work the count needs next; each has its own page and none of them is a gate.

This fifteen counts **measurements**, not decision records. There are twenty-five of those,
in [`docs/adr/`](adr/). The two numbers were equal for eleven days in August and this file
carried both, which is how the Primary sources table below came to describe the ADRs with the
count belonging to the table beneath this line — corrected 2026-09-07 under #51.

| # | attempt | result |
|---|---|---|
| 1–2 | Expected-vs-actual points; recency-weighted | null — r = 0.21 self-persistence |
| 3–4 | Depth-chart climb, two horizons | null — +0.008 beyond consensus |
| 5 | Age | null |
| 6 | Championship equity as the objective | **REMOVE** — lost 4 of 4 seasons; −19.66 pts/team-game as first measured, magnitude not quotable ([ADR-0009](adr/0009-championship-equity-does-not-pick.md)) |
| 7 | VOR ordering | **−5.06** pts/team-game |
| 8 | `edge`, the repo's original signal | unvalidatable — needs ADP nobody retains |
| 9 | Volume model beating the market's mean | null |
| 10 | Lineup optimiser | **+0.00** — a structural zero |
| 11 | Per-player weekly spread | null — 0.085 MAE of headroom exists at all |
| 12 | **Weekly injury retention** | **adopted** — +0.170 MAE at 3.8 se |
| 13 | Injury type on top of it | null by the gate — 3.1 se but 2/3 seasons |
| 14 | **Snap-share trend** | **screen positive** — +0.236 beyond consensus |
| 15 | Weekly projection vs weekly consensus rank, at setting a lineup | **shown, never ranked on** — −0.304 pts/team-week, 2 of 3 seasons lost; the market/Usage blend it closed on re-scored to −1.004, REMOVE, pending #206 ([weekly-blend-gate.md](weekly-blend-gate.md)) |

What separates #12 and #14 from the other thirteen is not sophistication — #12 is a nine-cell
lookup table of ratios. It is *what information they use*. The first twelve failures all tried
to out-think a market using information that market had had all summer. #12 used Wednesday's
practice report to set Sunday's lineup; #14 used snap counts published on a Monday. #15 is the
one that asked about the week and still lost, which is [what-the-field-knows.md](what-the-field-knows.md)'s
finding: week-level information exists, is measurable, and consensus already has enough of it.

**Edge came from timeliness, not from better processing of shared information.**

---

## What it costs, and what it buys

It killed most of the work. Thirteen of fifteen measurements ended in a removal, a demotion or a
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
| Twenty-five decisions, with their trade-offs | [`docs/adr/`](adr/), indexed in [architecture.md](architecture.md) |
| Objectives, and how objective 1 is judged | [decisions.md](decisions.md) |
