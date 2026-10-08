# The Weekly projection against consensus, on 2026, forward

**Pre-registered 2026-10-07 (#432), before any 2026 outcome was read by this work.** The commit
that adds this file is the timestamp ([track-record.md](track-record.md) rule 1). Nothing under
`data/` was opened for a 2026 *result* to write it: the figures below that touch 2026 are a
schedule's week count and the date of the newest consensus scrape in the local cache, both
inputs; every variance in it is from the spent seasons 2022-25.

This is the test [ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md)'s
*What would change it* names: *"a season measured forward from here ... the blend can be scored
against consensus as it happens without any of the data being spent in advance."* The weekly gate
reads REMOVE under #381's (C) (3 resolved of 4, 1 abstained; #430 makes the module an Exhibit),
and [#431](https://github.com/jacksonmlukas/football-hub/issues/431) ADOPTED (a): Phase 2's
weekly tickets (#305, #306, #308, #314, #316; #337 and #338 follow #306) are held behind this
measurement.

## Fact-finding: was weekly consensus captured before kickoff?

**By this repo, no, and never.** Nothing in the tree, in `state/` or in the workflows writes a
weekly consensus page when it is published.

What `weekly_gate_data.assemble_universe` reads for the incumbent is
`hub.models.panel.weekly_consensus`: the `weekly-op` page of the FantasyPros archive that
DynastyProcess republishes through nflverse (`load_rankings("all")`, 1.83M rows, every scrape of
every page, each carrying a `scrape_date`). Each scrape is assigned to the first week whose games
are not all played yet (`assign_weeks`, at most 8 days of lead). So for 2021-25 the incumbent is a
**dated third-party capture, not a reconstruction**: a ranking someone could have read on that
day, with the one documented concession that a Friday scrape follows a Thursday game, which
hands the incumbent a game of hindsight and so biases *against* the arm under test.

For 2026 that source exists upstream and is **not in this repo's evidence**:

- The cached archive (`ff_rankings/all`, pinned 2026-09-07, 1,830,022 rows) holds **zero
  `weekly-op` scrapes dated in 2026**; its newest scrape of any page is **2026-09-04**, five days
  before the season. (A count of scrape dates, not a ranking value.)
- The Makefile's `--archive` is skipped under CI and a runner's `data/raw` does not outlive the
  job, so no scheduled run keeps the archive: it is durable only where the maintainer runs it locally, which
  is the failure #383 found for the odds lines.
- Whether upstream has published 2026 `weekly-op` scrapes since 2026-09-07 is **unverified**: that
  needs a live fetch, and a fetch after the fact is not evidence of what could have been read
  before kickoff either.

So the consensus arm's 2026 inputs were not captured before kickoff by anything this repo
controls, for any week through the date of this document. The forward test can start only from
the first week a first-party capture exists, and **the capture is the first deliverable**
(`hub.fetch.consensus`, below).

**The arm under test is not captured live and cannot be** by the existing machinery:
`hub.models.weekly.project` scores player-weeks that exist in `player_stats`, so a projection for
a week not yet played has no row to be run on. It is reproduced after the fact, by code pinned
here, from inputs dated before each week: usage and stats strictly before week *w*, as in every
spent season. The one input that is not strictly pre-kickoff is the game's line, read from
nflverse's schedule at run time, which is the closing line and not Wednesday's. That is the same
information asymmetry the four spent seasons carry, and it favours the arm under test, so it
makes a REMOVE or SHOW harder to dismiss and an ADOPT easier to over-read; a reader of an ADOPT is
told so on the verdict line (`weekly_forward.CAVEATS`).

**#430 (the Exhibit move) does not change how the projection is run for scoring.** The harness
takes the arm as `assemble_universe`'s `arm` parameter and reaches the model only through
`weekly_gate_data`, so a move changes an import path and nothing else. The arm is pinned to
`hub/models/weekly.py` at blob `f6de17b2ca26a6dbaa2e606f567f8227456bc015` (HEAD
`9fc1a3a`). The verdict line records the blob it ran; **a change to what the projection
computes after the first admitted week is a new arm and a new pre-registration**, and a pure
move is not.

> **Amendment, 2026-10-07 (#430): the arm is re-pinned. Prior text above kept.** The move
> happened: `hub/models/weekly.py` is now `hub/exhibits/weekly_projection.py`, and
> `shipped_quantiles` is now `parametric_quantiles` (the word *shipped* names the page's
> conformalised interval and nothing else, CONTEXT.md). **This is a pure move and rename.** What
> the projection computes is unchanged, and the proof is
> `tests/unit/test_weekly_projection_move.py::test_the_moved_projection_computes_exactly_what_the_pinned_arm_computed`:
> it freezes `project` (bare, under each fitted `Shrink`, and under a fixed one), `fit_shrink`
> under both objectives, `positional_sd`, `standard_error` and the parametric quantiles on a
> fixed fixture, from output computed **before any file moved**, and the moved module
> reproduces every figure. The same digest was computed on the module at the originally pinned
> blob `f6de17b2ca26a6dbaa2e606f567f8227456bc015` and found identical.
>
> **The pin above was already stale when this ticket began.** The tree's
> `hub/models/weekly.py` was not at that blob when #430 began: `380486a` (#309, merged after
> `9fc1a3a`) had edited the prose `_what_the_coverage_measurement_says` prints about the
> published interval, and nothing else in the file (`git diff 9fc1a3a HEAD` over it is that one
> function). Run then, the forward measurement would have refused. The digest above is
> identical at both blobs, so the arm was never changed; the pin was stale by a diagnostic's
> wording.
>
> **The arm is now pinned to blob `96114791e9ba5492d92e996c1ad5f302fca5169a`**, `hub/exhibits/weekly_projection.py`
> (`weekly_forward.PINNED_ARM_BLOB`; `arm_blob()` finds it through `weekly_gate_data.project`
> and a test holds it to the file). **No 2026 outcome had been read:** the design reads once,
> after week 14, and the horizon has not passed. The pin is the old constant's replacement and
> not a second arm, so no new pre-registration is owed. The outstanding obligation is the old
> one: any later edit to this file, even to a comment, changes the blob and needs this same
> amendment before the reading.

## The design

### The arms

| arm | what it does | information |
|---|---|---|
| **weekly** | fills each slot with the rostered player of highest `hub.models.weekly.project` mu that week | stats before week *w*; the line; the schedule |
| **consensus** | fills each slot with the rostered player of highest `weekly-op` consensus rank that week: *start your highest-ranked player by consensus rank* | the **first-party capture** of that week's page, taken before kickoff |

The search is *start your highest* in both (ADR-0012), over the same frozen roster, restricted to
the roster-weeks **both** arms can price (`priced_by_both`, #206), scored on the same realised
points. It is `weekly_gate.compare` unchanged: `restrict=True`, `mask_pool=True`, `churn=False`.
**Waiver churn is not part of this measurement.** ADR-0016 names it as the one variant that could
flip the frozen result; it was run and lost by 3.5 points a team-week (`docs/weekly-gate.md`), so
it is not a second route to a reversal here, and a forward churn arm would be a new
pre-registration.

**The roster set** is the gate's own cohort for the season: `board_as_of(2026)`'s preseason
board, 20 simulated drafts from one slot, seed 0, frozen all season (`drafts=20, seed=0`, as
#378/#382). 20 rosters, the same recipe as the spent seasons, so the 2026 reading and the four
before it differ in the season and nothing else.

### The unit and the cluster

2026 is **one season**, so the repo's cluster (`SEASON_CLUSTER`, #45) has *k = 1* and no degrees
of freedom: it cannot be the unit here, and saying so is the first thing this design owes. The
within-season units are declared, and the choice is made on the spent seasons' variance
structure, not on what flatters the power:

- **The cluster of the interval half is the week.** The 20 rosters in a week share a slate, a
  player pool and one realisation of the week; the roster-week diffs are *nearly* but not exactly
  independent across rosters. On the frozen 2022-25 frame (40 rosters, 11-13 weeks a season) the
  mean correlation of two rosters' diffs across weeks is **0.023, 0.026, 0.004, 0.034**, and the
  variance across weeks of the cohort-mean diff is **3.54** against the **1.54** that independent
  rosters would give: a **week component of 2.00 (sd 1.42)**, common to the cohort. The roster
  component is zero within its noise (#388, `WEEKLY_ROSTER_COMPONENT = 0`). So a roster cluster
  would understate the SE, and the week, which absorbs the common part, is the honest unit.
  The lag-1 autocorrelation of the weekly cohort-mean diff is −0.23 / −0.40 / +0.34 / −0.11 (2022-25) over
  11-13 weeks: no sign that weeks are serially dependent beyond noise, and no cluster above the
  week is declared for that reason.
- **The roster is the within-season unit of the season's Disposition** (`WITHIN = ("roster",)`,
  the unit `weekly_gate` already declares), so the *season* resolves (a win or a loss) or abstains
  by `experiment._disposition` exactly as the four spent seasons did.

Both are the shipped `experiment.gate`, called as
`gate(paired, cluster=("week",), within=("roster",), ...)` on a frame whose one season is 2026.
That call is **not** a `Harness` and the module does not declare one: a `Harness` is by
construction a season-clustered gate (`test_gates_cluster_on_the_season`), and this is the one
place the repo clusters elsewhere, for the reason above. The interval half's t reference has
*weeks − 1* degrees of freedom.

### The metric

The weekly gate's own: **the paired difference weekly − consensus, in points per team-week**
(`weekly_gate.UNIT`), positive when the projection wins, one row per (roster, week), weeks 1-14.
The ceiling is the gate's own, perfect foresight on the same restricted rows, measured on the
same admitted weeks (stage 2 of [gate-power.md](gate-power.md) reads it; the spent ceiling was
**+10.80**, against an MDE here of about 2.2, so stage 2 passes).

### The decision rule and its actions

#381's (C), read off `experiment.gate` and not restated: a verdict over *resolved* seasons with
the abstention count beside it. With one season that is two outcomes for the Disposition
(resolved: a win or a loss; abstained) and the pooled t interval over the week clusters.

| verdict | when | action |
|---|---|---|
| **ADOPT** | the 2026 season is resolved a win **and** the week-clustered t interval excludes zero above | ADR-0016 and the weekly gate's REMOVE are re-opened **for the maintainer's decision** on this evidence; the weekly tickets (#305, #306, #308, #314, #316) are released to be taken up. It does not by itself set a lineup, and it speaks for one season. |
| **REMOVE** | resolved a loss **and** the interval excludes zero below | the spent seasons' REMOVE is confirmed on a season it could not have been fitted to; the weekly tickets stay blocked and are closed not planned at the maintainer's disposition. |
| **SHOW** | anything else, including the 2026 season abstaining | **nothing moves.** The REMOVE stands as the record's status, and the weekly tickets stay blocked. Absence of evidence, not evidence of equivalence. |
| **NOT-RUNNABLE** | fewer than **6** admitted weeks at the horizon, or the MDE exceeds the ceiling | no verdict; the design could not reach its question. Named as the exemption it is, below. |
| **NOT-YET** | the horizon has not been reached | no outcome is loaded and no number is printed. |
| **VOID** | the gate's own join-failure floor (2%) | as `weekly_gate`. |

The 2026 reading is on **2026 alone**. The four spent seasons are not pooled into it: they are
the prior that made the gate read REMOVE, they are already spent, and pooling them would let
four seasons outvote the one the test exists to add. A reader who wants the five together is
asking a different question and must pre-register it.

`MIN_WEEKS = 6` is **chosen, not fitted** (`hub.declare.chosen`): below six week clusters the
t reference (5 df, 2.571) and the table below put the design where P(ADOPT) at +2.0 points a
team-week is under 0.47; a verdict from fewer is not one this gate is pre-registered to issue.
It was set before any 2026 outcome was read and is not moved by the outcome.

### The horizon: which weeks count, and when the verdict is read

**A week counts if and only if a first-party capture of that week's `weekly-op` consensus exists
whose `captured_at` precedes the week's first game day** (00:00 America/New_York of the first
`gameday` in the schedule, strictly: a capture on the game day itself is refused, because
Thursday's game is in it) **and** whose newest scrape is within 8 days of that kickoff
(`CONSENSUS_MAX_LEAD_DAYS`), so last week's page is never filed under this week. The weeks the
harness admits are those, within weeks 1-14 (`GATE_WEEKS`); 15-17 are not read.

**The verdict is read once, at the horizon: when every regular-season game of week 14 has been
played and is in nflverse.** Before that the harness prints NOT-YET and loads no outcome at all
(it does not read a partial season "for monitoring": a reading taken to see how it is going is
the forking path rule 1 exists to close). There is no interim look and no early stop.

**Weeks 1-5: they do not count, and this was decided before looking.** They have been played;
nothing first-party was captured; and the reasons are three, in order of weight:

1. **Rule 1.** The design is committed after those weeks were played. The issue says it
   plainly: every week played before the design is committed is a week it may not count. Their
   outcomes were public when this was written, and a design written after its data exists is not
   a pre-registration whatever the author's intent.
2. **Provenance.** The only consensus for them would be the third-party archive's `scrape_date`,
   a label this repo cannot audit and the one thing rule 18's control (a capture written after
   kickoff is refused) cannot be applied to, because it was not written by us at all. Admitting
   them would make the lookahead guarantee of the other weeks conditional on trusting the
   weeks it was needed least.
3. **What it would have cost to count them is stated, not hidden.** Weeks 1-14 would have been
   14 clusters; admitting only from the first capture, at most **10** (weeks 5-14, if a capture
   lands before week 5's first game day) or **9**. The table gives both. It is the price of the
   rule and it is paid in power, not in the verdict.

Week 5 in particular: if its first game day (by nflverse's schedule) is after the capture's
`captured_at`, it counts; if it has started, it does not, and the harness decides this from the
file and the schedule, not from this paragraph.

### The power at that horizon (rule 16)

[`scripts/weekly_forward_power.py`](../scripts/weekly_forward_power.py): the shipped
`experiment.gate` (so (C)) on a simulated 2026, 20 rosters x *W* weeks, on **#388's weekly
process** (`gate_horizon.estimates()`, #418's finding: row variance 62.56, sd 7.91) plus **one**
fitted constant, the week component 2.00 above, estimated from the spent seasons. 10,000 trials
a cell, 4,000 bootstrap draws (the shipped number), seeds from `SeedSequence([432, cell])`. δ is
points per team-week.

**The null is sound.** At δ = 0 and 10 weeks, P(ADOPT) = **0.0247** and P(REMOVE) = **0.0236**
(SE 0.0016), under the 0.05 stop condition #381's (C) was held to and at the two-sided 5% the
t interval is built at. Planted effects: at δ = +100 the harness ADOPTs 2000 of 2000, at −100 it
REMOVEs 2000 of 2000.

| admitted weeks | δ = −1.5 | **−0.825** (spent mean) | −0.3 | 0 | +0.3 | +0.5 | +1.0 | +1.5 | +2.0 |
|---|---|---|---|---|---|---|---|---|---|
| 6 P(ADOPT) | 0.0003 | 0.0019 | 0.0113 | 0.0203 | 0.0428 | 0.0694 | 0.1517 | 0.3024 | 0.4658 |
| 6 P(REMOVE) | 0.2911 | 0.1229 | 0.0415 | 0.0252 | 0.0109 | 0.0061 | 0.0016 | 0.0002 | 0.0000 |
| 8 P(ADOPT) | 0.0000 | 0.0013 | 0.0104 | 0.0253 | 0.0532 | 0.0775 | 0.2154 | 0.4036 | 0.6172 |
| 8 P(REMOVE) | 0.4037 | 0.1574 | 0.0543 | 0.0284 | 0.0105 | 0.0051 | 0.0012 | 0.0002 | 0.0000 |
| 9 P(ADOPT) | 0.0002 | 0.0017 | 0.0101 | 0.0234 | 0.0591 | 0.0918 | 0.2337 | 0.4591 | 0.6877 |
| 9 P(REMOVE) | 0.4462 | 0.1742 | 0.0528 | 0.0240 | 0.0084 | 0.0044 | 0.0008 | 0.0001 | 0.0000 |
| **10 P(ADOPT)** | 0.0000 | 0.0014 | 0.0094 | **0.0247** | **0.0586** | **0.1006** | **0.2633** | 0.4958 | 0.7374 |
| **10 P(REMOVE)** | 0.4945 | **0.1940** | 0.0577 | **0.0236** | 0.0102 | 0.0039 | 0.0010 | 0.0001 | 0.0000 |
| 14 P(ADOPT) | 0.0000 | 0.0004 | 0.0087 | 0.0242 | 0.0703 | 0.1202 | 0.3518 | 0.6509 | 0.8833 |
| 14 P(REMOVE) | 0.6615 | 0.2612 | 0.0679 | 0.0254 | 0.0081 | 0.0036 | 0.0002 | 0.0000 | 0.0000 |

(The 14-week row is not on the table's horizon: it is what counting weeks 1-4 would have
bought, and is there so the cost of the weeks-1-5 decision is read off numbers.) The week
component is the one fitted constant, and its sensitivity at 10 weeks (week variance 0 / **2.00** /
4.00): P(ADOPT) at δ = +0.5 is 0.1042 / **0.1006** / 0.0865; at +1.5 it is 0.6610 / **0.4958** /
0.3905; P(REMOVE) at −0.825 is 0.2307 / **0.1940** / 0.1587. No value of it brings the design
near 80% at any effect the spent seasons suggest.

**What this says, which is rule 16's question.** At the weeks it will have, this gate reaches
80% power for ADOPT only at an effect of about **+2.2 points a team-week**, which is a fifth of
the perfect-foresight ceiling and **nearly three times the size of the spent seasons' mean loss
in the other direction**. At the effects anyone has measured for this projection (−0.825 pooled;
+0.3 to +0.5 is what #388 uses as the effect worth finding) it returns **SHOW with probability
0.80-0.93**. Its REMOVE has 19% power at the spent mean. **SHOW is therefore the branch this gate
will reach, and it is named here as an exemption and not a result:** a SHOW from this
measurement does not say the projection and consensus are equivalent, does not release the
weekly tickets, and does not strengthen or weaken the REMOVE. What the measurement *can* do is
return ADOPT if 2026 is a very large departure from the spent seasons (about 26% at +1.0, 50% at
+1.5), or REMOVE if it is a large continuation (49% at −1.5). It reports the 2026 interval
whatever the verdict, and that interval is the quantity a later posterior (#375) can use.

**A decision this leaves with the maintainer, flagged here and not made.** #431 held five weekly
tickets behind *this measurement*. As designed, the measurement can release them only on ADOPT,
which at every effect short of +1.5 it will not return. If "hold behind #432" was meant to resolve
in a release or a closure by the end of 2026, this design does not deliver that, by a margin the
table shows, and no honest design on one season of weeks does (14 weeks, all counted, gets
P(ADOPT) = 0.35 at +1.0). Saying so now, with the table, is the alternative to a NOT-RUNNABLE
branch nobody named.

## The capture (what the harness reads)

`hub.fetch.consensus` writes `state/consensus/<season>/wk<NN>/cap-<UTC timestamp>.json` for the
**next** week to kick off: the `weekly-op` rows of the newest scrape in the archive that falls
inside the week's window (`player`, `pos`, `team`, `ecr`, `scrape_date`), the `captured_at`, the
week's first game day, the deadline, and the archive's own digest. Append-only and written
atomically through `hub.atomic`, as `hub.store.write_snapshot` does for odds (#383), and
committed by a workflow, not by hand. **Rule 18's control:** a capture whose `captured_at` is at
or after the deadline is **refused at the write** (nothing lands), and one that reaches the tree
any other way is **excluded and named at the read**. A page whose newest scrape is stale (more
than 8 days before kickoff, or already past it) is refused the same way. The harness also reads
git: a capture file whose first commit is after the deadline is not admitted, whatever the file
says about itself (`hub.prereg`'s own argument: the commit is the timestamp).

*Licence, not decided here.* `state/consensus/` is third-party-derived (FantasyPros rankings, as
redistributed by DynastyProcess), kept in a private repository, as `state/odds/` is (#428). The
basis was **not** checked for this source: the odds archive's was, against The Odds API's terms.
If this repository is ever made public, re-examine `state/consensus/` first, and check
FantasyPros' and DynastyProcess' terms before then.

## What this does not claim

- It speaks for 2026, one season of 20 simulated frozen rosters, and for weeks 5/6-14. The spent
  seasons' REMOVE is not re-argued by it in either direction.
- The arm under test reads closing lines at run time (above); the incumbent a pre-kickoff page.
- Nothing here estimates anything for players neither arm can price (#206's restriction).
- The 2026 cohort is simulated from a preseason board, not the maintainer's own roster.

## What would change this document

Before the first admitted week's outcome is read: nothing, except an error in the arithmetic,
dated and with the prior text kept. After: nothing at all (rule 16's sibling rule: the identical
edit after the study has rows is rule 1 laundered).

## Amended 2026-10-07, when the capture and the harness were built (prior text kept)

Written before any capture exists and before any 2026 outcome has been read, which is the
condition under which this document may be edited (rule 16's sibling rule). Two things the build
made more exact than the text above, neither of which moves a number in the power table:

- **A week's page is stricter than "within 8 days of kickoff".** A scrape is week *w*'s page only
  if it is dated **after the previous week's last game day** and before *w*'s first game day, and
  no more than 8 days before it. Wednesday's page for week 5 is eight days ahead of week 6 and is
  week 5's: the panel's `assign_weeks` assigns a scrape to the first week with a game still to
  play, and a window that read "8 days" alone would have filed it under week 6 on a run taken
  just after week 5's deadline. `consensus.scrape_floors` is the window; the harness applies the
  same one when it reads. A tighter window can only refuse more captures, never admit one this
  text would have refused.
- **"When week 14 is played" is a date, not a look at results.** The horizon is reached on the
  second day after week 14's last game day in nflverse's schedule (`weekly_forward.LAG_DAYS`
  gives the data the day it lands). The harness decides this from the schedule's dates alone,
  which is what lets a run before the horizon say NOT-YET without opening a single outcome.

What was built, by name: the capture is `hub.fetch.consensus` (`--capture`), run by
`.github/workflows/consensus.yml` four times a week (Tuesday 21:00 UTC and Wednesday 11:00, 17:00
and 23:00 UTC) and written to `state/consensus/`; the harness is `hub.season.weekly_forward`;
the power is `scripts/weekly_forward_power.py`, whose tests hold its simulated rule equal to the
harness's (`tests/unit/test_weekly_forward.py`, `test_the_simulated_rule_is_the_shipped_rule`).
**It has not been run on 2026 data and must not be before the horizon**: the way to read it is
`uv run python -m hub.season.weekly_forward` on or after the second day following week 14's last
game day; before that it prints NOT-YET and loads nothing.

**The first capture is the maintainer's to take by hand if the workflow has not run before the
next week's first game day**: `uv run python -m hub.fetch.consensus --capture`, then commit
`state/consensus/`. Each week without a capture is a week this measurement does not have.

## Amended 2026-10-07 after the review of the build (#437; prior text kept)

Still before any capture exists and before any 2026 outcome has been read. Each of these makes
the harness stricter than the text above, or says more exactly what the text meant:

- **Whose clock admits a capture.** *"Whatever the file says about itself"* is now enforced, not
  only stated: a capture whose first-commit time the repository history cannot report (a shallow
  clone, an untracked file, no history at all) is **not admitted** and is named in the reading
  with the instruction to read with full history. The file's own `captured_at` never stands in.
  No workflow reads the verdict; it is run by hand
  (`uv run python -m hub.season.weekly_forward`), so what matters is that the clone it is run in
  has full history (unshallow it first). A scheduled reader would have to check out with
  `fetch-depth: 0`.
- **The horizon is dates and then data.** The second day after week 14's last game day is
  necessary; it is not sufficient. The harness also asks whether week 14's rows are in nflverse
  (a presence check) and says NOT-YET, loading no outcome, if they are not.
- **The arm pin is enforced.** A run whose arm is not blob
  `f6de17b2ca26a6dbaa2e606f567f8227456bc015` (`weekly_forward.PINNED_ARM_BLOB`) is REFUSED, with
  no outcome loaded: a different projection is a new arm and needs a new pre-registration. A
  pure move of the file that edits it (#430's import changes) trips the same refusal, and is
  answered by amending this document, dated, with the new blob, before any admitted outcome is
  read, not by editing the constant alone.
- **The roster-level Disposition is weakly informative here.** The rule reads the 2026 season's
  Disposition from a bootstrap over the 20 rosters, whose SE excludes the week component that all
  rosters in a week share (about 0.55 on rosters against about 0.71 on weeks at 10 weeks, from
  the power table's own process). So the week-clustered interval is the **binding half** of (C)
  here, and a "win" or "loss" read within rosters adds little the interval does not already say.
  The power table is of the shipped rule as it stands and is unchanged by this note.
- **Week numbering.** As of 2026-10-07 weeks 1-4 have been played. Week 5's first game is
  Thursday 2026-10-08 on the usual calendar (not read from the cached schedule, which carries no
  game dates), so week 5 is the first week a capture can still admit, and it counts only if the
  capture precedes that day. The heading *Weeks 1-5* above is the issue's; the rule for week 5 is
  the capture's timestamp, not its number.

## Amended 2026-10-07: the arm pin covers the arm's import closure, not one file (#456; prior text kept)

Still before any capture exists and before any 2026 outcome has been read, so this is a design
amendment made before any number does. The earlier amendments above pin the arm by the git blob
of `hub/exhibits/weekly_projection.py` (first `f6de17b2...`, then `96114791...` after #430) and
say a run on any other blob is REFUSED. That text stands as the record of what was pinned until
today; what it pins is superseded below.

**The defect.** The blob covered one file. The forward arm also runs `hub.models.panel` (the
panel build and `weekly_consensus`), `hub.season.weekly_gate_data` and the rest of what those
import. A change there, #315's opponent-adjusted DvP being the example, changes what the arm
computes, and the reading would have gone ahead on the old pin. A pin that misses the code that
produces the number it guards is the shape rule 1 names: documented, not implemented.

**The pin now.** `weekly_forward.PINNED_ARM_MODULES` is the SHA-256 (first 12 hex characters) of
the source of every module in the first-party import closure of the arm, walked from
`weekly_forward.ARM_ROOTS` (`hub.season.weekly_gate_data`, `hub.exhibits.weekly_projection`) by
`hub.ledger.import_closure`: the AST walk #439 wrote for the ledger's code declaration, now one
function that the contract `test_every_gate_declares_the_code_it_runs` and this pin both use,
module-level and function-local imports alike. It stops at the same exemptions
(`hub.ledger.CLOSURE_EXEMPT`, each with its reason: the shared gate rule, the fetch loaders whose
bytes `data_digest` pins, config hashed by `config_digest`, path and I/O plumbing, schema
assertions, markers). Those are neither required nor descended into, so an edit to one does not
refuse the reading, and the reason each is exempt is that a change there cannot change what the
arm computes. `read_forward` takes the closure as it stands and REFUSES, loading no outcome,
when any module differs from the pin, is new in the imports or is gone, naming each. The
digest over the whole table, `weekly_forward.PINNED_ARM_DIGEST`, is printed on every reading.

**Nothing in the arm changed.** The closure at the head of this amendment is the tree's state at
`main` `d8e20f4`; this amendment edits no module in it, and the table below is that state. (The
pin was first computed at `d6f8882`, where its digest was `07aaa20c636fb902`; merging `main`
forward to `d8e20f4` moved exactly one module, `hub.models.coverage`, from `8ea06bda176b` to
`e036e1981cef`, by the 2026 coverage row and the slate's measure-before-publish order
(`b13a320`, `e6a5bc1`). The new pin refused its first tree-versus-pin comparison on it, which is
the defect this amendment exists to close working as designed. The arm reaches that module
through one function-local import, in `weekly_projection`'s report-prose helper that reads
`published_summary` for the interval diagnostic; it does not enter `project`, the shrinkage fit
or any quantity the forward measurement scores, and the identity test below passes on this tree.)
The earlier pin
was the blob `96114791e9ba5492d92e996c1ad5f302fca5169a` of one of these 29 modules, and that
module (`hub.exhibits.weekly_projection`) is the one the behavioural identity test
`tests/unit/test_weekly_projection_move.py` (#430) holds to the figures computed on the pre-move
code; it still passes on this tree, so the closure the new pin covers reproduces them. The new
pin is a wider net over the same arm, not a second arm, and no new pre-registration is owed.

**The closure pinned, digest `7bc93c290a5cc331`** (29 modules; module, then
its source digest):

| module and digest |
|---|
| `hub.draft.adp_history` `286f2d9c667d` |
| `hub.draft.availability` `34b707bf7e75` |
| `hub.draft.board` `6e61e56ecac0` |
| `hub.draft.cohort` `5d6e03fc2cd5` |
| `hub.draft.durability` `59fbe0402d35` |
| `hub.draft.optimize` `b5c2b10a2ec1` |
| `hub.draft.picks` `c695b87cc5e8` |
| `hub.draft.playoff_sos` `42bc422f88eb` |
| `hub.draft.prior_signal` `4b7421479470` |
| `hub.draft.regression` `3ba22b3a649e` |
| `hub.draft.report` `df1f8e20569e` |
| `hub.draft.season` `601db0b4f9f1` |
| `hub.draft.state` `ad958507df6a` |
| `hub.exhibits.weekly_projection` `d5b25ec6130b` |
| `hub.holdout` `a8d095db300c` |
| `hub.league` `1d50b1eaf7ef` |
| `hub.models.base` `8c41cd5f26d1` |
| `hub.models.components` `c402b0c4856f` |
| `hub.models.conformal` `f98558ab2051` |
| `hub.models.coverage` `e036e1981cef` |
| `hub.models.margin` `9da0f4839e89` |
| `hub.models.market` `098baf487556` |
| `hub.models.panel` `2f9f2aad1946` |
| `hub.models.predict` `ddf960a1dfe6` |
| `hub.models.scoring_rules` `3c201f3a0d6f` |
| `hub.models.volume` `31e5321b1e6d` |
| `hub.names` `93e503040186` |
| `hub.season.weekly_gate` `211f84e4a564` |
| `hub.season.weekly_gate_data` `1823ccae01c3` |

**The obligation.** A later edit to any module in this table, a comment included, trips
`test_the_pinned_arm_is_the_closure_the_arm_runs` and, at the reading, a REFUSED verdict. An edit
that did not change what the arm computes is answered by showing so (the identity test), re-pinning
`PINNED_ARM_MODULES` and amending this document, dated, with the new table, before any admitted
outcome is read. An edit that did change it is a new arm and needs a new pre-registration.
`test_the_pin_is_recorded_in_the_document` holds the constant and this table equal.


## Amended 2026-10-07: the pin is re-pinned after #442's behaviour-neutral edits (prior text kept)

Still before any capture exists and before any 2026 outcome has been read. The #456 table above
is kept as the record of what was pinned until today; the pin below supersedes it.

**What changed in the closure, and why nothing the arm computes did.** Three of the 29 modules
were edited by #442 (ledger follow-ups) and none of them in anything the forward measurement
reaches. (1) `hub.models.margin` gained `NFL_WEEKS = 18` (moved there from
`hub.season.survivor`, so that `survival_beside`'s default no longer needs a function-local import
of the survivor pool's module, which the walk would otherwise have had to pin) and lost that
import; its `Harness` declarations now say `arm_roots=("hub.models.margin",)` where they listed
nine modules. (2) `hub.models.coverage` and (3) `hub.season.weekly_gate` changed the same
declaration only (`arm_roots=(...)` in place of a hand-kept `arm_modules` tuple). The set of
modules in the closure is unchanged: 29, the same names. The identity test
`tests/unit/test_weekly_projection_move.py` (#430) passes on this tree, so the figures computed
on the pre-move code are still reproduced; `survival_beside` and the harness declarations are not
called by `project`, the shrinkage fit or anything the forward measurement scores.

| module | was | now |
|---|---|---|
| `hub.models.coverage` | `e036e1981cef` | `fcfc597b98e8` |
| `hub.models.margin` | `9da0f4839e89` | `8d58c201cc4b` |
| `hub.season.weekly_gate` | `211f84e4a564` | `73cc434a55db` |

**The closure pinned, digest `fc8042b7baaf9f88`** (29 modules; was `7bc93c290a5cc331`):

| module and digest |
|---|
| `hub.draft.adp_history` `286f2d9c667d` |
| `hub.draft.availability` `34b707bf7e75` |
| `hub.draft.board` `6e61e56ecac0` |
| `hub.draft.cohort` `5d6e03fc2cd5` |
| `hub.draft.durability` `59fbe0402d35` |
| `hub.draft.optimize` `b5c2b10a2ec1` |
| `hub.draft.picks` `c695b87cc5e8` |
| `hub.draft.playoff_sos` `42bc422f88eb` |
| `hub.draft.prior_signal` `4b7421479470` |
| `hub.draft.regression` `3ba22b3a649e` |
| `hub.draft.report` `df1f8e20569e` |
| `hub.draft.season` `601db0b4f9f1` |
| `hub.draft.state` `ad958507df6a` |
| `hub.exhibits.weekly_projection` `d5b25ec6130b` |
| `hub.holdout` `a8d095db300c` |
| `hub.league` `1d50b1eaf7ef` |
| `hub.models.base` `8c41cd5f26d1` |
| `hub.models.components` `c402b0c4856f` |
| `hub.models.conformal` `f98558ab2051` |
| `hub.models.coverage` `fcfc597b98e8` |
| `hub.models.margin` `8d58c201cc4b` |
| `hub.models.market` `098baf487556` |
| `hub.models.panel` `2f9f2aad1946` |
| `hub.models.predict` `ddf960a1dfe6` |
| `hub.models.scoring_rules` `3c201f3a0d6f` |
| `hub.models.volume` `31e5321b1e6d` |
| `hub.names` `93e503040186` |
| `hub.season.weekly_gate` `73cc434a55db` |
| `hub.season.weekly_gate_data` `1823ccae01c3` |

## Amended 2026-10-07: the arm is re-pinned for two docstrings (#313; prior text kept)

Still before any capture exists and before any 2026 outcome has been read. #313 names three
leakage surfaces in the docstrings of `hub.models.components` and `hub.models.panel` (and
corrects the direction of one existing sentence in `panel.assign_weeks`), which are two modules
in the closure above. The pin is a hash of source bytes, so a docstring edit moves it, and the
amendment obligation above applies: show the edit did not change what the arm computes, re-pin,
date it.

**The evidence.** Parsing each of the two modules at `main` (`3729e6b`) and in this change, and
removing every module, class and function docstring, the two syntax trees are equal
(`ast.dump`): the change is docstrings and nothing else, so no statement the arm executes
differs. The behavioural identity test `tests/unit/test_weekly_projection_move.py` (#430) is
run on this tree beside it (result in the commit). The arm's inputs and outputs are not reached:
no constant, no join, no column and no default moved.

**What did not change:** `hub.models.predict`, `WEEKLY_K` and every fitted constant, so no
interval and no scored quantity on the forward arm moves. The refit of the spread law that #313
also asks for is **not** in this amendment; it is not landed (no adopted design exists for it),
and if it lands it is a separate amendment with its own evidence, because `predict.WEEKLY_K` is
read by the arm.

**The pin now, digest `c891aaf58f135479`.** Exactly two rows moved:

| module and digest now | was |
|---|---|
| `hub.models.components` `b23c5d7d0c49` | `c402b0c4856f` |
| `hub.models.panel` `99d19fba99e0` | `2f9f2aad1946` |

The other 27 rows of the table above stand. (Previous digest `7bc93c290a5cc331`.)

## Amended 2026-10-08: the two re-pins above, combined (#442 with #313; prior text kept)

Both re-pins above were taken from the #456 table (digest `7bc93c290a5cc331`) and landed
together, so neither section's digest is the pin now. They touch disjoint modules: #442 moved
`hub.models.coverage`, `hub.models.margin` and `hub.season.weekly_gate`; #313 moved
`hub.models.components` and `hub.models.panel`. Each table row above that is not one of those
five stands. Nothing else changed in the merge, and the closure is still the same 29 modules.

**The pin now, digest `cf9b4b9eea056369`** (was `fc8042b7baaf9f88` on #442 alone and
`c891aaf58f135479` on #313 alone): the #442 table with `hub.models.components` `b23c5d7d0c49`
and `hub.models.panel` `99d19fba99e0` taken from #313's.


## Amended 2026-10-08: the arm is re-pinned for declaration spellings (#321; prior text kept)

Still before any capture exists and before any 2026 outcome has been read. #321 widened
`hub.declare`'s walk and gave four numbers in the closure a declaration they lacked, which is
an edit to four of the 29 modules. The pin is a hash of source bytes, so it moves, and the
amendment obligation above applies: show the edit did not change what the arm computes, re-pin,
date it.

**What changed, and why nothing the arm computes did.** Each edit wraps an existing value in one
of `hub.declare`'s markers, which return their first argument unchanged at run time:
`weekly_projection.MIN_UNITS` `not_an_input(8.0, ...)` -> `chosen(8.0)`;
`optimize._TD_LUCK_NOTE` `0.5` -> `not_an_input(0.5, ...)`;
`conformal.DEFAULT_ALPHA` `ModelConfig().conformal_alpha` -> `not_an_input(<same>, ...)`;
`coverage._SIGMA_SCALE` `math.sqrt(2.0)` -> `not_an_input(<same>, ...)`; plus the
`hub.declare` import each needed and, in `weekly_projection`, a comment. `hub.declare` is
itself exempt from the closure (`CLOSURE_EXEMPT`), so the set is unchanged: 29 modules, the
same names.

**The evidence.** Parsing each of the four modules at the parent commit and in this change,
removing every module, class and function docstring, replacing each call of `chosen`,
`not_an_input` or `fitted` by its first argument and dropping the `from hub.declare import`
line, the two syntax trees are equal (`ast.dump`) for all four: no statement the arm executes
differs. The behavioural identity test `tests/unit/test_weekly_projection_move.py` (#430) passes
on this tree, so the figures computed on the pre-move code are still reproduced.

**What did move:** `config_digest` `c4606f91` -> `96fee74f` and `fitted_digest` `5024dd03` ->
`772d3bba`, because `MIN_UNITS` entered the covered declarations (a coverage correction, same
value; recorded in `tests/unit/test_config.py`). That is the model version's label and not a
number the arm computes.

| module | was | now |
|---|---|---|
| `hub.draft.optimize` | `b5c2b10a2ec1` | `314a78a8b5ef` |
| `hub.exhibits.weekly_projection` | `d5b25ec6130b` | `372f7180071e` |
| `hub.models.conformal` | `f98558ab2051` | `79827e566bba` |
| `hub.models.coverage` | `fcfc597b98e8` | `3c2d88cc6908` |

**The closure pinned, digest `da1dfa2e9571eea7`** (29 modules; was `cf9b4b9eea056369`):

| module and digest |
|---|
| `hub.draft.adp_history` `286f2d9c667d` |
| `hub.draft.availability` `34b707bf7e75` |
| `hub.draft.board` `6e61e56ecac0` |
| `hub.draft.cohort` `5d6e03fc2cd5` |
| `hub.draft.durability` `59fbe0402d35` |
| `hub.draft.optimize` `314a78a8b5ef` |
| `hub.draft.picks` `c695b87cc5e8` |
| `hub.draft.playoff_sos` `42bc422f88eb` |
| `hub.draft.prior_signal` `4b7421479470` |
| `hub.draft.regression` `3ba22b3a649e` |
| `hub.draft.report` `df1f8e20569e` |
| `hub.draft.season` `601db0b4f9f1` |
| `hub.draft.state` `ad958507df6a` |
| `hub.exhibits.weekly_projection` `372f7180071e` |
| `hub.holdout` `a8d095db300c` |
| `hub.league` `1d50b1eaf7ef` |
| `hub.models.base` `8c41cd5f26d1` |
| `hub.models.components` `b23c5d7d0c49` |
| `hub.models.conformal` `79827e566bba` |
| `hub.models.coverage` `3c2d88cc6908` |
| `hub.models.margin` `8d58c201cc4b` |
| `hub.models.market` `098baf487556` |
| `hub.models.panel` `99d19fba99e0` |
| `hub.models.predict` `ddf960a1dfe6` |
| `hub.models.scoring_rules` `3c201f3a0d6f` |
| `hub.models.volume` `31e5321b1e6d` |
| `hub.names` `93e503040186` |
| `hub.season.weekly_gate` `73cc434a55db` |
| `hub.season.weekly_gate_data` `1823ccae01c3` |


## Amended 2026-10-08: the arm is re-pinned for the hold-out rebinding every reader (#320; prior text kept)

Still before any capture exists and before any 2026 outcome has been read. #320 makes a held-out
constant held out everywhere it is read, and edits two of the 29 modules: `hub.draft.season`
no longer re-exports the six held-out names (`TALENT_CV`, `TALENT_CV_BY_POS`, `WEEKLY_K`,
`WEEKLY_K_POOLED`, `WEEKLY_SKEW`, `WEEKLY_SKEW_POOLED`), which were copies a `holdout.applied`
rebind could not reach; and `hub.holdout` refuses a set that leaves a held-out key unaccounted
for, and holds a lock across the record's read-modify-write. The pin is a hash of source bytes,
so it moves, and the amendment obligation above applies: show the edit did not change what the
arm computes, re-pin, date it.

**Why nothing the arm computes changed.** `hub.draft.season`: parsing the file at the parent
commit and in this change, removing docstrings and the six names from the one `from
hub.models.predict import` line, the two syntax trees are equal (`ast.dump`); nothing in the
closure imports those names from it (the only importers were `hub.draft.calibrate` and
`hub.exhibits.leverage`, neither in the closure). `hub.holdout`: the arm reaches it only through
`--holdout` (`weekly_gate_data`'s `applied(yr)` and `weekly_gate`'s `run_lines`); the four
committed sets in `conf/holdout/` already account for all eleven keys, each with a value or a
reason, and `load` returns an identical `ConstantSet` and `describe` line for each of 2022-25
before and after. The behavioural identity test `tests/unit/test_weekly_projection_move.py`
(#430) passes on this tree.

**What did not move:** `config_digest` `96fee74f` and `fitted_digest` `772d3bba`, the same on
`main` `1bbb8c7` and on this tree: no constant's value or declaration changed.

| module | was | now |
|---|---|---|
| `hub.draft.season` | `601db0b4f9f1` | `55989706066a` |
| `hub.holdout` | `a8d095db300c` | `e74f918fab67` |

**The closure pinned, digest `2932bb4a2c589e97`** (29 modules; was `da1dfa2e9571eea7`):

| module and digest |
|---|
| `hub.draft.adp_history` `286f2d9c667d` |
| `hub.draft.availability` `34b707bf7e75` |
| `hub.draft.board` `6e61e56ecac0` |
| `hub.draft.cohort` `5d6e03fc2cd5` |
| `hub.draft.durability` `59fbe0402d35` |
| `hub.draft.optimize` `314a78a8b5ef` |
| `hub.draft.picks` `c695b87cc5e8` |
| `hub.draft.playoff_sos` `42bc422f88eb` |
| `hub.draft.prior_signal` `4b7421479470` |
| `hub.draft.regression` `3ba22b3a649e` |
| `hub.draft.report` `df1f8e20569e` |
| `hub.draft.season` `55989706066a` |
| `hub.draft.state` `ad958507df6a` |
| `hub.exhibits.weekly_projection` `372f7180071e` |
| `hub.holdout` `e74f918fab67` |
| `hub.league` `1d50b1eaf7ef` |
| `hub.models.base` `8c41cd5f26d1` |
| `hub.models.components` `b23c5d7d0c49` |
| `hub.models.conformal` `79827e566bba` |
| `hub.models.coverage` `3c2d88cc6908` |
| `hub.models.margin` `8d58c201cc4b` |
| `hub.models.market` `098baf487556` |
| `hub.models.panel` `99d19fba99e0` |
| `hub.models.predict` `ddf960a1dfe6` |
| `hub.models.scoring_rules` `3c201f3a0d6f` |
| `hub.models.volume` `31e5321b1e6d` |
| `hub.names` `93e503040186` |
| `hub.season.weekly_gate` `73cc434a55db` |
| `hub.season.weekly_gate_data` `1823ccae01c3` |

## Amended 2026-10-08: the arm is re-pinned for the board's capture stamp (#405; prior text kept)

Still before any capture exists and before any 2026 outcome has been read. #405 makes the age
the board reports (`last_good`, `board_age_hours`, the poller's "board built Nh ago", the
adherence replay's copy note) the age of its *capture* -- a stamp `_persist` writes beside the
parquet, `fetch.cached`'s record -- instead of the file's mtime. That edits one of the 29
modules, `hub.draft.board`; the pin is a hash of source bytes, so it moves, and the amendment
obligation above applies: show the edit did not change what the arm computes, re-pin, date it.

**Why nothing the arm computes changed.** The arm reaches `hub.draft.board` through
`board_as_of` (`weekly_gate_data`). Parsing the file at the parent commit and in this change and
comparing every top-level statement by `ast.dump`, exactly four functions differ -- `_persist`,
`board_age_hours`, `last_good` and `build_or_last_good` -- and none is called by `board_as_of`,
`build` or anything else the forward measurement reaches (`_persist` writes the files `main`
prints, `last_good` is the offline fallback and the poller's reader, `board_age_hours` formats an
age). `build`, `board_as_of`, `recommend` and every other function and constant are
syntax-tree-identical; the only additions are two helpers (`board_stamp_path`,
`board_captured_at`) and the constant `BOARD_STAMP_SUFFIX`, plus `cached` on the existing
`from hub.fetch import` line. `hub.fetch.*` is exempt from the closure (`CLOSURE_EXEMPT`), so the
import adds no module: the closure is still 29, and the walk against the new pin reports no
difference. The behavioural identity test `tests/unit/test_weekly_projection_move.py` (#430)
passes on this tree.

| module | was | now |
|---|---|---|
| `hub.draft.board` | `6e61e56ecac0` | `65ae0104fb6c` |

**The closure pinned, digest `6f3a04e853ad4428`** (29 modules; was `2932bb4a2c589e97`):

| module and digest |
|---|
| `hub.draft.adp_history` `286f2d9c667d` |
| `hub.draft.availability` `34b707bf7e75` |
| `hub.draft.board` `65ae0104fb6c` |
| `hub.draft.cohort` `5d6e03fc2cd5` |
| `hub.draft.durability` `59fbe0402d35` |
| `hub.draft.optimize` `314a78a8b5ef` |
| `hub.draft.picks` `c695b87cc5e8` |
| `hub.draft.playoff_sos` `42bc422f88eb` |
| `hub.draft.prior_signal` `4b7421479470` |
| `hub.draft.regression` `3ba22b3a649e` |
| `hub.draft.report` `df1f8e20569e` |
| `hub.draft.season` `55989706066a` |
| `hub.draft.state` `ad958507df6a` |
| `hub.exhibits.weekly_projection` `372f7180071e` |
| `hub.holdout` `e74f918fab67` |
| `hub.league` `1d50b1eaf7ef` |
| `hub.models.base` `8c41cd5f26d1` |
| `hub.models.components` `b23c5d7d0c49` |
| `hub.models.conformal` `79827e566bba` |
| `hub.models.coverage` `3c2d88cc6908` |
| `hub.models.margin` `8d58c201cc4b` |
| `hub.models.market` `098baf487556` |
| `hub.models.panel` `99d19fba99e0` |
| `hub.models.predict` `ddf960a1dfe6` |
| `hub.models.scoring_rules` `3c201f3a0d6f` |
| `hub.models.volume` `31e5321b1e6d` |
| `hub.names` `93e503040186` |
| `hub.season.weekly_gate` `73cc434a55db` |
| `hub.season.weekly_gate_data` `1823ccae01c3` |


## Amended 2026-10-08: the arm is re-pinned for `survival_beside`'s default (#323; prior text kept)

Still before any capture exists and before any 2026 outcome has been read. #323 makes
`hub.models.margin.survival_beside` raise to the pool's pick count (24: a second pick in each
of weeks 13-18) and not to the 18 weeks of the season. That edits one of the 29 modules: it
adds `from hub.config import PoolConfig` (`hub.config` is in `CLOSURE_EXEMPT`, so the closure's
membership is unchanged: the same 29 names), a function `pool_picks`, the default inside
`survival_beside`, and one string in `survival_line`. The pin is a hash of source bytes, so it
moves, and the amendment obligation above applies: show the edit did not change what the arm
computes, re-pin, date it.

**Why nothing the arm computes changed.** `survival_beside`, `survival_line` and `pool_picks`
are called from `hub.models.margin`'s `--shape` report and from tests, and from nowhere else in
`src/`. The one thing the closure takes from `margin` is `home_win_prob` and `home_won`
(`hub.models.coverage`, a function-local import), and neither moved. `project`, the shrinkage
fit and everything the forward measurement scores are untouched. The behavioural identity test
`tests/unit/test_weekly_projection_move.py` (#430) passes on this tree.

**What did not move:** `config_digest` `96fee74f` and `fitted_digest` `772d3bba`, the same on
`main` and on this tree: no constant's value or declaration changed (`PoolConfig`'s
`double_pick_weeks` is read, not edited).

| module | was | now |
|---|---|---|
| `hub.models.margin` | `8d58c201cc4b` | `8fd45168e983` |

**The closure pinned, digest `060820bfc56a32db`** (29 modules; was `6f3a04e853ad4428`):

| module and digest |
|---|
| `hub.draft.adp_history` `286f2d9c667d` |
| `hub.draft.availability` `34b707bf7e75` |
| `hub.draft.board` `65ae0104fb6c` |
| `hub.draft.cohort` `5d6e03fc2cd5` |
| `hub.draft.durability` `59fbe0402d35` |
| `hub.draft.optimize` `314a78a8b5ef` |
| `hub.draft.picks` `c695b87cc5e8` |
| `hub.draft.playoff_sos` `42bc422f88eb` |
| `hub.draft.prior_signal` `4b7421479470` |
| `hub.draft.regression` `3ba22b3a649e` |
| `hub.draft.report` `df1f8e20569e` |
| `hub.draft.season` `55989706066a` |
| `hub.draft.state` `ad958507df6a` |
| `hub.exhibits.weekly_projection` `372f7180071e` |
| `hub.holdout` `e74f918fab67` |
| `hub.league` `1d50b1eaf7ef` |
| `hub.models.base` `8c41cd5f26d1` |
| `hub.models.components` `b23c5d7d0c49` |
| `hub.models.conformal` `79827e566bba` |
| `hub.models.coverage` `3c2d88cc6908` |
| `hub.models.margin` `8fd45168e983` |
| `hub.models.market` `098baf487556` |
| `hub.models.panel` `99d19fba99e0` |
| `hub.models.predict` `ddf960a1dfe6` |
| `hub.models.scoring_rules` `3c201f3a0d6f` |
| `hub.models.volume` `31e5321b1e6d` |
| `hub.names` `93e503040186` |
| `hub.season.weekly_gate` `73cc434a55db` |
| `hub.season.weekly_gate_data` `1823ccae01c3` |
