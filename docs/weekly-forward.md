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
