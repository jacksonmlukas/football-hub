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
>
> **Re-opened 2026-09-12 by #260 and discharged 2026-09-16 by #290:** the shipped-constants
> figure in ADR-0009 (−11.07) was measured on the draw before #260 narrowed it, and the
> re-run existed only as the maintainer's next action. The run was made under hold-out
> constants: **−13.21 [−18.76, −7.67]**, REMOVE 4/4, and ADR-0009 carries the box.

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

> **Decided 2026-09-13, issue #274: the false-discovery threshold is a diagnostic, and CRPS
> decides exactly one thing.** Audit III found both computed and neither deciding, and asked
> which each is. The answer, for each, and where it is printed:
>
> **The false-discovery threshold is a diagnostic.** It was pre-registered before the numbers
> (2026-09-11, #37), it makes the multiplicity visible on the same line as the verdict, and it
> decides nothing -- the verdict is the rule above and only that. The report line now says so
> in its own words: `(a diagnostic: the pre-registered rule decides; the threshold does not)`,
> on every family `hub.models.weekly_screen` prints.
>
> **Re-scored at q = 0.10 for the record.** The table's four positives sit in one family --
> the alone screen at the published anchor, week 8, on the settled basis -- and none had been
> read against the threshold. `uv run python -m hub.models.weekly_screen --run
> --trend-min-week 8`, 2026-09-13, cfg `b1f69382`, fitted `04c2d997`, data `a4050443` over
> five pinned sources, 14,483 player-weeks and 849 players (the tables on
> [weekly-screen.md](weekly-screen.md) were 14,370 and 847; the archive has grown since, and
> every `t` below reproduces the published one within a tenth):
>
> | feature | t | p | BH-adjusted p | at q = 0.10 |
> |---|---|---|---|---|
> | implied team total | +12.43 | 0.000 | **0.002** | survives |
> | own spread | +5.37 | 0.006 | **0.023** | survives |
> | defence vs position | +3.76 | 0.020 | **0.053** | survives |
> | injury severity | −3.10 | 0.036 | **0.072** | survives |
>
> Family of eight, threshold 0.0361, four below it; the other four rows (snap-share trend
> 0.205, prior TD rate 0.113, rest 0.348, target-share trend 0.818) are above it and were
> already killed by the rule. The joint family at the same anchor -- four tests, threshold
> 0.0543 -- reads 0.054 / 0.054 / 0.072 for the three survivors and 0.484 for own spread.
> **Nothing falls, so nothing is restated.** What the re-scoring adds is the thing the rule
> could not say: the two weakest positives clear the rule at 3.8 and 3.1 se and clear the
> family at 0.053 and 0.072 -- inside q, and not by much.
>
> **CRPS is a diagnostic beside MAE, except for one decision.** `hub.models.weekly` prints
> CRPS of the published `(mu, sd, skew)` beside MAE (#177) and nothing reads it; the line now
> says `A diagnostic: nothing here decides on it`. It is promoted to a **gate input for
> exactly one decision**: the weekly interval's shape law -- keep the Cornish-Fisher skew or
> drop it -- under #292, with the rule and the minimum detectable effect written in
> [gate-power.md](gate-power.md) before the first run. The promotion is that narrow on
> purpose: a score that decides one pre-registered question is a gate input there and a
> diagnostic everywhere else, and a blanket promotion would be rule 1's incident -- the
> scoring rule chosen after the verdicts it would re-score were known.

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

> **Discharged 2026-09-16, in `quarterback.py`, and not generalised (audit IV).** The incident
> above is fixed — `hub.models.quarterback` no longer collapses to the right answer by algebra
> on the fixture's one input — but the rule was applied to the module that failed it and no
> further. Mutation testing the same afternoon, run against an unmodified copy of the tree,
> found the identical failure shape in the modules next to it: **seven of fourteen mutants
> survive in `starter_change.py`** and **three of ten survive in `pool.py`**, and every
> surviving mutant is a fixture that never constructs the case the mutation needs — the
> Eastern-conversion mutant with every captured game on the same calendar date in both zones,
> the NOT-RUNNABLE guard mutant with `_paired` held constant so the bootstrap never has width,
> `share_sd` scaled without a fixture at more than one trial count catching it. Rule 15 stands
> as a rule; it has one module where it is evidence and two more where it is still a claim.

### 16. A gate that cannot return a verdict is not a bar, it is an exemption — and it must be named as one in the pre-registration that created it

**A NOT-RUNNABLE branch that can never leave NOT-RUNNABLE is not a bar with an edge case; it is
a standing licence, and a pre-registration that does not say so is granting the licence by
omission.** Writing a gate down before the numbers only does its job if every branch the gate
can actually reach is named — including the branch that, at this project's pace, is every
branch it will ever reach.

**The incident (#291, #300).** The quarterback adjustment's log-loss gate was pre-registered
on 2026-09-13 with four branches — ADOPT, REMOVE, SHOW, NOT-RUNNABLE — and its own power
calculation, run the same day, put a verdict in the 2050s: **29 event-seasons at 80% power**
against the pilot's target, at a pace of one event-season a year. The pre-registration pulled
the component out of `ratings` and `survivor` on every branch the gate could reach except the
one branch it was actually going to take. A NOT-RUNNABLE print every season for three decades
is not an absence of evidence the way rule 12 means it; it is the only outcome the gate was
ever going to produce, and nothing in the pre-registration said so — the module stayed
published on a branch nobody had named as the reason it could.

**The fix.** #300 did not lower the bar or invent a shortcut around the power calculation. It
moved the ADOPT condition to the estimand one season of this poller can already resolve — the
line-move coefficient from #221's study, sign and magnitude against 0.132, season-clustered
once two seasons exist, both already built and already pre-registered — and demoted the
log-loss gate to a diagnostic beside it. The NOT-RUNNABLE branch is unchanged; what changed is
that the amendment names it: the precondition *"was never, and is not now, a condition on the
module's ADOPT or REMOVE status"* — a branch this gate cannot leave decides neither
(`docs/gate-power.md`, *Amended 2026-09-17 (#300) — demoted to a diagnostic, and NOT-RUNNABLE
is named an exemption*).

**Where the rule lives in code.** `hub.models.experiment.gate` itself did not move — its
NOT-RUNNABLE branch was already correct per rule 12, printing no verdict where none is earned.
What moved is the pre-registration text around it: the amendment in `docs/gate-power.md` now
states, next to the branch, which decisions it is and is not allowed to make, instead of
leaving a reader to infer the exemption from watching the branch never change for thirty
seasons.

**The sibling rule, for the document rather than the branch.** A pre-registration may be
edited *before its estimand has data* — that is rule 1 working, and it is why #300, #303 and
#329 amended `docs/gate-power.md` freely on 2026-09-17: the study had zero rows. The identical
edit after the study has rows is rule 1 laundered, whatever its size. The test is never how
small the change is; it is whether a number the change could favour exists yet.

**What to ask before a gate ships.** Run the gate's own power calculation before the
pre-registration is written down, not after the first NOT-RUNNABLE print. If the answer puts a
verdict past a horizon this project will see, the pre-registration must say which branch is
actually reachable and name the rest exemptions — not leave a module's published status resting
on a branch that was never going anywhere else.

### 17. A decision rule must be checked for degeneracy against its own inputs before it is pre-registered

**A conjunction whose second term is implied by its first is one term, and writing it down as
two is how a rule comes to be trusted for a strictness it does not have.**

**The incident (M-S1, #357).** ADR-0019's Gate reads two halves as independent: the pooled
interval excludes zero, and the sign holds in every held-out season. Under `SEASON_CLUSTER`,
the cluster every gate in this repo resamples on, the bootstrap draws exactly the `k` season
means the every-season half also reads — and a nonparametric percentile bootstrap over `k`
clusters can only resample the `k` numbers it was handed, so when every one is positive, every
resample is a convex combination of positive numbers and `lo > 0` follows from `won == total`
**by construction**. The interval half was the sign half, read twice: a one-sided sign test of
size `2**-k` — 12.5% at k=3, 6.25% at k=4, 3.1% at k=5 — wearing two names and quoted as a
stricter rule than either alone.

**What made it invisible.** Every safeguard this repo had was pointed at whether the rule was
*followed* — argued in the ADR, spelled out in `gate()`'s own docstring, unit-tested branch by
branch, held by a contract test asserting every gate clusters on the season. None of them asked
whether the rule, followed exactly as written, decided anything a single term did not already
decide on its own. A rule that is followed to the letter and still degenerate passes every test
built to catch drift from the letter, because the letter is exactly what it is drifting inside.

**The check, and why it is cheap.** Simulate the rule under the null — draw `k` season means
with a true effect of zero, at a realistic `k` and cluster count — and confirm the realised
ADOPT rate is the size the rule claims, not the sign test's `2**-k` it may have silently
become. `docs/method.md` rule 15's shape applies here too: a size check that only ever passes
is not evidence the check works, so `_null_adopt_rate`
(`tests/unit/test_experiment.py::test_the_size_check_flags_a_planted_degenerate_rule`) is
proven first against a *planted* degenerate rule — the pre-#357 rule itself, held permanently
rather than as a one-off mutation — before it is trusted against the rule that replaced it
(`::test_the_fixed_rule_s_null_size_is_not_degenerate`).

**Where the rule lives in code.** `experiment.gate`'s interval half now reads a t interval
(`t_interval`, off `mean`/`se`/`clusters`) rather than the percentile bootstrap's `lo`/`hi` — a
distributional claim rather than a resampling one, not implied by the seasons' signs the same
way (`t_interval`'s own docstring has the argument in full). The fix closes the incident; this
rule is the diagnostic that should have run *before* the two-part rule was ever pre-registered,
not only after a finding located the defect by hand.

**What to ask before any conjunction is pre-registered.** Does the second term ever fail when
the first term already holds? If every case that satisfies the first term also satisfies the
second, the conjunction is one term with an extra clause, and simulating the rule under the
null — not arguing about it — is what tells the two apart.

---

### Noted twice, not yet a rule: a dispersion fitted on observed variance absorbs the sampling noise of the thing it is fitted on

Recorded 2026-09-21 so a third instance is recognised as a pattern rather than a coincidence,
and promoted to a rule then. **Instance one (C5, audit IV):** the touchdown rate's
year-over-year correlation came out at zero because the estimate's own sampling variance
swamped the signal — the fix (#308) subtracts the binomial variance at the known count before
the rate is shrunk. **Instance two (#306's pre-registration):** a Dirichlet-multinomial
concentration fitted on observed share variance comes out too low, because a share built on
~5 events carries multinomial sampling noise the concentration would read as volatility; the
disattenuation (`p(1 − p)/n` subtracted from the observed between-week variance) is part of
the estimator, pre-registered. The shape both share: **any constant that is a dispersion, a
correlation or a reliability, fitted on quantities measured from few events, is attenuated
by the events' own sampling variance, and the correction is analytic when the count is
known.** Two is a coincidence; three is rule 16's successor.

### Noted four times in 24 hours, not yet a rule: an assertion whose outcome cannot vary with the thing it asserts about

Recorded 2026-09-21. Rule 17 above covers the first instance and only the first: it is about
a *decision rule*, checked *before pre-registration*. The second and third were neither, so
rule 17 as written would not have caught them. They are filed here so the generalisation, if
it comes, is written from three dated cases and not reconstructed later.

**Instance one (M-S1, #357, a decision rule).** `lo > 0 and won == total`, where the first
term is implied by the second under the cluster every gate uses. The conjunction's outcome
could not vary with the interval — it was the sign half read twice. Rule 17.

**Instance two (#363's wiring, `03d119f`, a test).** `margin.verdict()`'s stage-2 guard fires
on `mde > ceiling`. The tests covered a present non-binding ceiling (ADOPT) and no ceiling at
all (NOT-RUNNABLE), and the firing condition itself lived in a docstring — *"a constant +0.05
gain over three seasons does not exceed it"* — with no fixture that reached it. The first
fixture written to reach it did not: two identical held-out seasons give an MDE of exactly
zero, which no ceiling can be smaller than, so the "binding ceiling" case could not bind. The
outcome could not vary with the guard.

**Instance three (#326's release condition, comment of 2026-09-21, an operational check).**
The first draft of the freeze's ending test was `gh issue list --state open --label blocking`
— and there is no `blocking` label in this tracker; the tag lives in the audit JSON. The query
returns empty whether or not a hold exists. An empty result would have read as clearance. The
outcome could not vary with the hold.

**Instance four (`scripts/rule16_combined_power.py`, found 2026-09-21 by two lanes
independently, a harness).** The script that produced the rule-16 table in ADR-0019's
amendment called `summarise` with no `ceiling`. It was run before #363 landed behind it in the
same lane; after #363, a gate with no ceiling is NOT-RUNNABLE in both directions, so the
harness returned an ADOPT rate of 0 at every δ. The published table was right and could no
longer be re-derived from the tree. Fixed by handing the simulation a ceiling that cannot bind
and saying why, and by a test that plants an effect the harness cannot miss and asserts it
adopts — the outcome must vary with δ before any rate it reports means anything.

**The shape all four share.** Different levels — a rule, a test, a query, a harness — and one
defect:
**the thing checked is not connected to the thing the check is about, so the check passes
regardless.** Rule 15 is the test-shaped special case (a fixture that sets the condition under
which the estimator is trivially correct); rule 17 is the rule-shaped one. What is not yet
written is the general instruction — *before trusting any assertion, ask what observation
would make it come out the other way, and confirm that observation is reachable* — and the
check that would have caught all four is the same one each time: plant the failure and watch
the assertion notice. Four at four levels in one day is more than a coincidence; it is
recorded rather than promoted because the general form has not yet been tested against a
case it was written for.

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
| 1–2 | Expected-vs-actual points; recency-weighted | null — r = 0.21 self-persistence *(Restated 2026-09-21, #358: no effect larger than partial r 0.092 detectable at 80% power for the uniform-weighting screen, k=4 seasons — the measured pooled value was r=+0.21; the recency-weighted variant's bound could not be computed, no per-season breakdown was published to derive s from; [detectable-effects.md](detectable-effects.md) rows 1–2.)* |
| 3–4 | Depth-chart climb, two horizons | null — +0.008 beyond consensus *(Restated 2026-09-21, #358: no effect larger than partial r 0.122 (next season; measured +0.008) or 0.209 (rest-of-season, published se; no single pooled r published, per-season values ran −0.040 to +0.100) detectable at 80% power, k=4 seasons each; [detectable-effects.md](detectable-effects.md) rows 3–4.)* |
| 5 | Age | null *(Restated 2026-09-21, #358: no effect larger than partial r 0.103 detectable at 80% power, k=5 seasons; the measured pooled value was r=+0.0076; [detectable-effects.md](detectable-effects.md) row 5.)* |
| 6 | Championship equity as the objective | **REMOVE** — lost 4 of 4 seasons; −19.66 pts/team-game as first measured, magnitude not quotable ([ADR-0009](adr/0009-championship-equity-does-not-pick.md)) *(Restated 2026-09-21, #358: the verdict stands; this design could detect effects down to 11.17 pts/team-game at 80% power, k=4 seasons; [detectable-effects.md](detectable-effects.md) row 6.)* |
| 7 | VOR ordering | **−5.06** pts/team-game *(Restated 2026-09-21, #358: the verdict stands; this design's detectable effect could not be computed — only a pooled figure across 3 seasons × 40 drafts is published, no per-season breakdown to derive a between-season s from; [detectable-effects.md](detectable-effects.md) row 7.)* |
| 8 | `edge`, the repo's original signal | unvalidatable — needs ADP nobody retains *(Restated 2026-09-21, #358: confirmed — no detectable-effect bound exists for this row; no historical ADP source exists to build one and backtest boards carry no adp/edge column at all; [detectable-effects.md](detectable-effects.md) row 8. Not estimated.)* |
| 9 | Volume model beating the market's mean | null *(Restated 2026-09-21, #358: no detectable-effect bound could be computed — fit on 2022-23 and evaluated pooled on 2024-25 only, one evaluation window rather than multiple held-out seasons to cluster on; [detectable-effects.md](detectable-effects.md) row 9.)* |
| 10 | Lineup optimiser | **+0.00** — a structural zero *(Restated 2026-09-21, #358: the verdict stands; this design's detectable-effect bound could not be computed — the gate builds its paired frame from the network at run time and persists nothing, so a season-clustered s cannot be measured offline; [detectable-effects.md](detectable-effects.md) row 10.)* |
| 11 | Per-player weekly spread | null — 0.085 MAE of headroom exists at all *(Restated 2026-09-21, #358: no effect larger than 0.0054 MAE detectable at 80% power, k=5 seasons; the measured per-season gains were +0.0033, +0.0081, +0.0078, +0.0106, +0.0029; [detectable-effects.md](detectable-effects.md) row 11.)* |
| 12 | **Weekly injury retention** | **adopted** — +0.170 MAE at 3.8 se |
| 13 | Injury type on top of it | null by the gate — 3.1 se but 2/3 seasons *(Restated 2026-09-21, #358: no effect larger than 0.122 MAE detectable at 80% power, k=3 seasons; the measured per-season gains were +0.0507, +0.0602, −0.0150; [detectable-effects.md](detectable-effects.md) row 13.)* |
| 14 | **Snap-share trend** | **screen positive** — +0.236 beyond consensus |
| 15 | Weekly projection vs weekly consensus rank, at setting a lineup | **shown, never ranked on** — −0.304 pts/team-week, 2 of 3 seasons lost; the market/Usage blend it closed on re-scored to −1.004, REMOVE, pending #206 ([weekly-blend-gate.md](weekly-blend-gate.md)) *(Restated 2026-09-21, #358: the verdict stands; this design could detect effects down to 0.768 pts/team-week at 80% power, k=4 seasons; [detectable-effects.md](detectable-effects.md) row 15.)* |

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
| What each of the fifteen measurements could detect, at 80% power | [detectable-effects.md](detectable-effects.md) |
