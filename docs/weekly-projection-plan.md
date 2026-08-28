# Weekly component projection — the plan

**Written 2026-08-27, before any measurement.** Pre-registration, per
[method.md](method.md) rule #1: the decision rules, the incumbents and both gates are fixed
here, in writing, before a number exists to be tempted by.

---

## What is actually new here

Not "components instead of points". That is already the repo's position and it is built:
[`hub.models.components`](../src/hub/models/components.py) reconstructs points from
receptions, carries, yards and touchdowns, the distribution families behind each count were
*measured* rather than assumed, and the whole thing is written up in
[component-projection.md](component-projection.md).

What is new is **the week**. Every projection this repo makes today is one season-long
per-game mean applied flat to all seventeen weeks. Ja'Marr Chase is 19.6 points in Week 1 and
19.6 points in Week 12, against any opponent, in any game script, whatever his snap count did
last Sunday. There is no week-level term anywhere in the system.

So the honest name for this work is **a week-specific component model**, and the thing it has
to earn is the *week*, not the decomposition. Saying that precisely matters, because two
neighbouring things have already been measured and lost, and this must not quietly re-run
either of them.

## What the record already settles

Do not re-measure these. They are in the tree with their harnesses.

| already measured | result | so |
|---|---|---|
| Component-derived weekly **spread** vs `sd = k·√mu` | lost, 1.365 vs 1.140, P(better) 0.0% | the square-root law stays. This model supplies **means**, not a new spread law ([ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) keeps the losing code) |
| Volume/efficiency shrinkage toward a **positional** mean | null, −0.2% RMSE, P(better) 40.2% | shrinking a WR1 and a WR5 to the same place over-shrinks the studs |
| The same, toward a **pick-implied** prior | 99.9% against that baseline, but still loses to simply reading the pick | ships as a decomposition, not a projection ([volume-model.md](volume-model.md)) |
| Touchdown rate per yard, year over year | r = **−0.004** receiving, **−0.030** rushing | a player's own TD rate carries no information beyond his yardage. **Season-over-season only** — the week-over-week version is unmeasured and is a screen below, not an assumption |
| Efficiency persistence | ypc 0.108, ypt 0.369, catch rate 0.402 | do not try to predict weekly efficiency. Predict opportunity |
| Volume persistence | targets 0.805, carries 0.791 > points 0.775 | volume is the part that carries, which is the premise of all of this and it holds |

And two positives, which are the only two this repo has ever found in fourteen measurements:

* **Snap-share trend**, next three weeks in season: partial r **+0.236** beyond both PPG and
  ECR, 12/12 season-anchor cells, placebo-clean ([snap-trend-signal.md](snap-trend-signal.md)).
  Its *decision* then failed its gate and it is shown, never ranked on
  ([ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md)).
* **Injury retention** by (status, practice): **+0.170** MAE at 3.8 se
  ([weekly-injury.md](weekly-injury.md)).

Both are *timeliness* signals — something published on a Monday that consensus has not yet
absorbed. Every null asked consensus about information it had had all summer. That pattern is
the sharpest prior available for what follows, and it is stated here before the screen rather
than after it.

---

## The shape

One row per **(player, season, week)**. The model predicts a vector of counts, and league
scoring turns the vector into points — never the other way round.

```
                    week-level information
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   team volume          player share           efficiency
   (plays, pass/run)    (targets, carries)     (yds/target, TD rate)
        │                     │                     │
   implied team total    snap-share trend       ~nothing: ypc r=0.108,
   spread/game script    injury designation      TD/yd r≈0
   pace, rest, roof      depth-chart change
        └─────────────────────┴─────────────────────┘
                              │
                    component means per week
                              │
              measured families (component-projection.md):
              volume → negative binomial (var/mean 1.20–2.45)
              TDs    → binomial (var/mean 0.83–0.86, underdispersed)
              yards  → compound, units × yards-per-unit sampled apart
                              │
                     league scoring weights
              (espn.scoring_settings(), read from the league)
                              │
                        weekly points
```

The structural claim, which is testable and which the persistence table above already
supports: **week-level information moves team volume and player share -- together, **Usage** --
and barely touches efficiency.** A model that tries to predict this week's yards per carry is predicting noise. `Usage` and
`Weekly projection` were added to [CONTEXT.md](../CONTEXT.md) rather than reusing `xFP`, which
already owns the word *opportunity*.

### The data is already here

No new fetch is needed to screen or to gate. This was the surprise while scoping.

| need | source | status |
|---|---|---|
| Weekly component **outcomes and expectations** | `ff_opportunity` — `rec_attempt`, `receptions`, `rush_attempt`, `*_yards_gained`, `*_touchdown`, and an `_exp` counterpart for each | **in the store**, 2021-25 |
| Closing **spread and total** per game | `schedules` — `spread_line`, `total_line` | in the fetch layer, free, historical |
| Rest, roof, surface, temp, wind, divisional | `schedules` | same call |
| Snaps / participation | `participation`, `ftn_charting` | fetched, unscreened by design ([improvements.md #4](improvements.md)) |
| Injury designation | `nfl.load_injuries` | already used by `hub.models.injury` |
| Defence vs position | `playoff_sos._dvp_from_stats` | built; currently season-aggregate over playoff weeks |
| Interceptions, fumbles, 2PC | `player_stats` weekly | needs the `cols` list widened past `fantasy_points_ppr` |

**No kicker, no defence.** Confirmed 2026-08-27: the league starts QB / RB×2 / WR×3 / TE /
FLEX and nothing else, which is what `RosterConfig` already says and what ADR-0008 records
`DRAFTED_POSITIONS` excluding on purpose. Nothing to model and nothing to hold fixed.

Implied team total is `total_line / 2 ± spread_line / 2`, computable for every historical game
at zero API cost. The odds API (`hub.fetch.odds`, `MARKET = "spreads"`, `CREDIT_FLOOR = 50`)
is only needed for *live* freshness in-season, and adding a `totals` market would double the
credits per pull — decide that at Phase 4, not now.

---

## Phase 0 — pre-registration

**Deliverable: this file, extended with the exact rules, before Phase 1 runs.**

### The two incumbents, and why there are two

**Floor incumbent — the flat projection.** Today's `proj_blend` applied unchanged to every
week. Beating it proves only that *week-level information exists*, which is close to
guaranteed: the incumbent has literally no week-to-week variation, so any signal that a
Thursday-night game exists will beat it. **A win here is necessary and means almost nothing.**

**Real incumbent — weekly consensus.** The free thing a person would actually read on a
Saturday, and the thing every previous null lost to.

**Corrected 2026-08-27, during the grilling.** The plan first said fetching it was the first
concrete task. It is not a task at all: `nflreadpy.load_ff_rankings("all")` already ships
`weekly-qb/rb/wr/te` **and** a cross-position `weekly-op` page — 15–20 scrape dates a season
across 2020–2025, ~467 ranked players per date, ECR 1–609. Six seasons of held-out weekly
consensus, free, already in the fetch layer.

**But `r2p_pts` is null on every historical weekly row.** The incumbent is an **ECR ranking**
with `sd`/`best`/`worst` — *not* a points projection. That is the single most shaping fact in
this plan, and it decides the gate order below: you cannot take a paired MAE against a thing
that has no points, and fitting a rank→points curve would put free parameters on the
*incumbent's* side of the comparison, doing part of the work you are trying to measure.

### The metric, and which gate is primary

**Gate B is primary.** The question is *"does a lineup set off this projection beat a lineup
set off weekly consensus rank"*, paired, in points per team-week. The incumbent is
parameter-free — fill each slot with your highest-ranked eligible rostered player by
`weekly-op` ECR — which is the same shape as the incumbent that beat the snap trend at −3.81
points ([ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md)), one level down.

This is recorded as
[ADR-0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md), because "just fit a
rank→points curve and run a proper accuracy gate" is exactly what a future review will suggest.

**Gate A is demoted to a reported diagnostic**: paired absolute error on player-week points
against the *flat* projection only, per-observation, via
[`experiment.paired_gain`](../src/hub/models/experiment.py). It cannot run against consensus,
and against the flat projection it is nearly free. It is reported so a Gate B result can be
attributed rather than asserted, and it gates nothing.

Secondary and reported, never gated on: per-component MAE, so a win can be traced to Usage or
to touchdowns.

### The gate, both halves

The repo's usual bar, unchanged: the arm must beat the incumbent **in every held-out season**
*and* clear **`experiment.MIN_SE` = 2.0 standard errors** on the paired difference. Written as
a `verdict()` with its losing branch unit-tested before the measurement runs, per method.md
rule #1 — the rule that a documented-but-unimplemented gate printed ADOPT on a 0.6% gain.

**The three branches, fixed here.** Asymmetric on purpose: the Weekly projection is the
complicated thing and the burden sits on it.

| condition | verdict | what happens |
|---|---|---|
| beats consensus in **every** held-out season **and** t ≥ `MIN_SE` | **ADOPT** | the Weekly projection sets lineups |
| consensus wins every season at t ≥ `MIN_SE` | **REMOVE** | worse than a free ranking; delete the module |
| anything else | **SHOW, NEVER RANK ON** | printed beside consensus, never sorted on |

The middle branch is the expected one and it has an *action* rather than being a
disappointment to explain away — the same disposition the snap trend got in
[ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md). It is written now
precisely because "show it beside consensus" is a satisfying thing to decide *after* seeing a
near-miss.

**QB is reported separately and never pooled into the headline, and a QB-only win does not
adopt.** Pre-registered here because the trap is already on the record:
[component-projection.md](component-projection.md) has the volume model coming back null
overall while QB alone hit −0.354 RMSE at P(better) 97.5%, correctly not taken, because one of
four positions crossing 97.5% happens about 10% of the time under the null. QB is also where
the market-derived features are most likely to be real — pass volume genuinely tracks game
script — which makes it the most tempting and the most dangerous.

### The lineup rule is fixed, and this does not reopen ADR-0012

Gate B varies the **projection** and holds the **search** fixed at *start your highest
projections*. That is the accepted rule
([ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md)), and it must not be
swapped for `lineup.optimize`.

The reason matters. The existing `lineup_gate` returned a **structural zero** — n=80,
+0.00 [−0.00, +0.00] — because `sd = k·√mu` makes spread a deterministic increasing function
of the mean (r = 0.985), so ranking by `mu` and ranking by any increasing function of
`(mu, sd)` give the same order. The optimiser was handed no variance to read. ADR-0012 then
*withdrew* its own "re-run when sd stops being a function of mu": per-player weekly volatility
beyond the positional constant is real but ±9.3%, a season recovers 15% of it, and `k·√mu`
already sits within 0.085 MAE of the irreducible floor.

So the axis here is different and is the one still open: that gate varied the **search** over
identical projections; this one varies the **projection** under an identical search. A
structural zero there says nothing about this.

### The leakage rule needs a new tool

[method.md](method.md) rule #2 lives in `experiment.expanding_seasons`, which splits by
*season*. A weekly model needs the within-season version: predictors from weeks strictly
before `w`, outcome at `w`, never the same week. **Build `expanding_weeks` beside it**, under
the same AST guard, before any feature is computed. This is the exact failure mode that made
depth-chart climb read as 7.4 sigma, and a weekly model has seventeen times the surface for it.

### Inactive weeks

A week a player did not play is **excluded from Gate A and scored as zero in Gate B**. The two
gates ask different questions: "is the projection better" is contaminated by availability,
while "is the lineup better" is partly *about* it — starting an inactive player is the most
expensive weekly mistake there is, and an honest lineup score has to eat it. This is already
how `lineup_gate.weekly_grid` behaves ("a player with no row scored nothing that week"), so
Gate B needs no change. Bye weeks are excluded from both: both arms know about them.

It also puts the injury model — one of only two things this repo has ever measured positive —
somewhere it can actually pay.

### Repeated measures

A player's seventeen weeks are not seventeen observations — protocol item 3, which turned
noise into an apparent 4-sigma result once already. **Cluster by player** in every bootstrap,
and report the season-anchor cell structure the snap-trend screen used (sign must hold in
every cell, not on the pool).

---

## Phase 1 — the screen

Per the protocol: **minutes, not sessions, and no `src/` until something clears.** Ad hoc is
explicitly allowed here — ADR-0007's trigger is *citation*, not exploration.

Partial correlation of each candidate against player-week points, controlling for the flat
projection **and** for weekly consensus once it is fetched. One screen per feature, each with
its sign pre-stated:

| # | feature | pre-stated sign | asks consensus about |
|---|---|---|---|
| 1 | Implied team total | + | information it has had since the line opened |
| 2 | Spread / game script (pass-run split) | + for pass volume when trailing | same |
| 3 | Opponent DvP by position | + | same |
| 4 | Snap-share trend, last 3 weeks | + | **something published on a Monday** |
| 5 | Injury designation (status × practice) | − | **something published on a Friday** |
| 6 | Target-share trend | + | Monday |
| 7 | Rest days, roof, wind | wind − for passing | pre-priced |
| 8 | Week-over-week TD-rate persistence | pre-stated **null** | the season-level result says zero; the weekly version is untested and this is the honest way to test it |

Rows 4, 5 and 6 are the ones with a prior. Rows 1-3 and 7 are shared information and the
expectation is that they beat the *floor* incumbent handsomely and the *real* incumbent not at
all. **Recording that expectation now is the point of writing it down now.**

Kill criteria, pre-registered: a feature whose sign flips between seasons is a bug, not a
signal (protocol item 4). A feature that survives only in the pooled sample and not in the
per-season cells is a repeated-measures artifact.

### If nothing clears

**Stop, and write the null into [signal-screens.md](signal-screens.md) as the fifteenth
measurement.** Pre-registered here so there is nothing to negotiate with once a number exists.

Phase 2 is still non-empty in that case, and exactly two things are named:

* **The injury designation enters without a new screen.** It was measured at player-week
  grain — +0.170 MAE at 3.8 se, [weekly-injury.md](weekly-injury.md) — which is already this
  horizon.
* **The snap trend does not.** Its established result is a *three-week* horizon; using it for a
  one-week lineup asserts a mechanism the screen never tested, which is protocol item 6 and the
  thing this repo got wrong once already. It must be re-screened at a one-week horizon first,
  and at anchors ≥ 8 where it exists at all.

So even the total-null case leaves an honest minimum experiment: a weekly model built from the
one thing already measured at the right grain.

**The screen stays pointed at points, not at the lineup**, even though the gate is a decision.
A screen asks *is this real* and a gate asks *is this better*; pointing the screen at the
decision collapses the two, which is the confusion
[ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md) exists to record. A feature
that is real and never changes a lineup is a true finding about football and a null about
fantasy, and both halves are worth being able to state separately.

---

## Phase 2 — the model, only for features that cleared

Committed code from this point, per ADR-0007: pure core, testable offline, results written
through `hub.store` and stamped with `config_digest`.

`hub/models/weekly.py`:

**The form is a multiplier on Usage, and `f ≡ 1` is the incumbent.**

    weekly Usage = season Usage x f(week features),    f centred on 1.0

Four reasons, the first being the one that matters: **the null is the identity.** `f ≡ 1`
recovers today's flat projection exactly, so the model cannot be much worse than what you
already have. It keeps the season-long xFP and Talent work rather than discarding it. It
isolates the one thing this plan says has to be earned — the *week* — instead of confounding it
with a rebuilt season model. And a failed gate leaves one interpretable object, a multiplier
printable per player per week, rather than a parallel model to debug.

The multiplier acts on the **counts**, never on points. Points are rebuilt from adjusted Usage
through league scoring, which is the whole premise end to end.

* `usage_means(players, week_ctx)` — per-player, per-week expected counts, built as
  season Usage × `f`, with only the surviving features moving any factor of `f`.
* Reuse the **measured** families from `components.py` for dispersion. They were measured
  against 12,673 player-weeks and there is no reason to re-derive them.
* Efficiency held at the player's shrunk season rate, per the persistence table — with the
  single exception of any efficiency feature that cleared Phase 1, which the table says will
  not happen.
* Aggregate with `espn.scoring_settings()` rather than a hardcoded PPR map, since the league's
  own weights are readable and `components.SCORING` is otherwise an assumption.

**Graceful degradation is a hard rule.** No odds on a Sunday morning must produce the flat
projection with a printed note, never an error and never a blank. Same pattern as
`board.build_or_last_good`.

---

## Phase 3 — the gate, and the diagnostic beside it

**Gate B, the decision — this is the gate.** Set each week's lineup off the Weekly projection;
set it off `weekly-op` consensus ECR; score both against realised points, paired. Metric is
**points per team-week**. The lineup rule is fixed at *start your highest projections* in both
arms, both arms are restricted to the same roster, and `OPP_MU`/`OPP_SD` are held identical so
they cannot favour either — the pattern `lineup_gate` already uses.

**The sample, fixed here.**

| | |
|---|---|
| Rosters | the synthetic rosters `lineup_gate` already drafts from `board_as_of(season)` — 80 over four seasons |
| Pairing unit | **roster-week**, ~1,120 of them, with a **cluster bootstrap by roster** |
| Weeks | **1–14**, the fantasy regular season. 15–17 reported separately and never pooled; week 18 dropped |
| Voided if | more than **2%** of roster-weeks fail to join weekly consensus |
| Held-out seasons | the ff_opportunity × weekly-consensus overlap, first season training-only |

Roster-week rather than roster-season because the decision is made weekly and season-averaging
throws away the variation the week is supposed to explain — but 1,120 roster-weeks over 80
rosters is protocol item 3 at fourteen times the scale, so the bootstrap clusters by roster and
the raw n is never quoted as if it were independent.

Weeks 15–17 are reported apart for the same reason QB is: it is where the season is decided, it
is where a positive result is most wanted, and it is a three-week slice — exactly the size at
which you find one if you go looking at it as one of several.

**The known limitation, stated rather than buried.** These rosters are **static all season**. A
large part of what a Weekly projection is worth is deciding who to stream into the flex, and a
frozen roster deletes that decision entirely. So this design is *conservative*: a null here is
weaker evidence than a null usually is in this repo, and the write-up must say so rather than
reporting a clean negative.

**Weeks 1–7 run with the snap trend switched off, and that is not a sample choice.** The
trend is established only from anchor 8 — anchors 4 and 6 are null and flip sign between
seasons, pooled −0.057 and +0.097, because at week 4 the comparison is weeks 3–4 against weeks
1–2 and season-to-date PPG is a weak control on three games. So it is dark before week 8 and
the gate eats the dilution. **Weeks 8–14 are pre-registered as a named secondary, reported
alongside and never substituted for the headline** — restricting the headline to the window
where your best feature works is the same move as adopting QB alone, and this repo has declined
that once already. The model is not featureless early: implied team total, spread, DvP and the
injury designation all run from Week 1.

**The join floor, and why it is tight.** Four key spaces — ff_opportunity and player_stats on
`gsis_id`, weekly consensus on `fantasypros_id` plus name, snaps on `pfr_player_id` with a
99.8% crosswalk whose missing 0.2% is uncharacterised. Under the symmetry rule below, **a
player missing from consensus is ranked last**, so a join failure does not add noise — it
*forces a bench* on the incumbent's arm and biases the result toward us. The failure is
directional, which is why the floor is 2% rather than something comfortable, and why the
unmatched players are reported **by name** rather than as a count. The board's own join
silently dropped 20 of the top 168 by ADP until somebody went looking.

**Information symmetry, which is the rule a gate dies on.** Both arms see everything published
before kickoff and nothing after. A rostered player absent from the weekly consensus page — on
bye, or ruled out — is **ranked last**, because that absence is the incumbent saying "do not
start him" and it is real information the incumbent legitimately has. Our arm therefore gets
the same pre-kickoff injury feed, through `hub.models.injury`. Handing either arm information
the other lacks is the exact defect that made the first `lineup_gate` unable to fail
([ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md)); pointed the other way
it would make this one unable to pass.

`hub/season/lineup_gate.py` supplies the harness: `weekly_grid`, `projection_lineup_points`,
`compare` and a pure `verdict`. What changes is the axis — that gate varied the *search* over
identical projections and returned a structural zero; this one varies the *projection* under
an identical search.

**Gate A, the diagnostic.** Walk-forward by season via `expanding_seasons`, per-player-week
paired errors against the **flat** projection only. It cannot run against consensus, which has
no points. It gates nothing and exists so a Gate B result can be attributed — a lineup win
with no projection win is a ranking artifact and should be treated as one.

**Gate B can fail while Gate A passes, and if it does, the Weekly projection is shown and
never started on** — the same disposition the snap trend got. A projection has to be wrong in
the right *direction* to change a lineup, and a lineup is a max over a roster: most of the
error never reaches the decision.

---

## Phase 4 — operations

Only if Gate B passes.

* A weekly CLI in the shape of the draft-night tools: one command, an answer under a cap, and
  a last-good fallback.
* Live freshness needs a decision on the odds API `totals` market and its credit cost.
* `hub.models.conformal` is built and connected to nothing ([improvements.md #2](improvements.md)).
  Weekly player predictions are the first thing in this repo with enough volume to calibrate
  intervals against — 300+ player-weeks a week. That is a consumer, and it is the reason to
  sequence conformal *after* this rather than before.

---

## Timing

Nothing here is on the draft path, and none of it should start before **2026-09-03**. Phases 0
and 1 are historical and could run today; they will not, because the week before a draft is
not when you start a new model. Phase 1 in Week 1, Phase 2-3 across Weeks 2-4, Phase 4 only if
both gates pass.

Nothing needs doing early. The weekly-consensus fetch turned out not to exist as a task —
`load_ff_rankings("all")` already has six seasons of it — which removes the one item that
would otherwise have had to start before the draft.

## What I expect to happen

Stated before measuring, so it cannot be adjusted afterwards:

1. The model beats the **flat** incumbent by a wide margin. This is nearly free and proves
   little.
2. It does **not** beat weekly consensus on market-derived features alone. Fourteen
   measurements say consensus prices shared information — and consensus reads the same lines
   we do, on the same Saturday.
3. If anything survives against consensus it is the snap-share trend and the injury
   designation — the two things already measured positive, both timeliness.
4. Week-over-week TD rate comes back null, consistent with the season-level −0.004.
5. Gate B fails even where Gate A passes, because a lineup is a max over a roster and most
   projection error never reaches the decision. This is now the *primary* gate, so this
   prediction is the one that decides whether anything ships.

6. The most likely single verdict is **SHOW, NEVER RANK ON** — the middle branch — which is
   why it was written with an action attached.

If (1) and (2) both land, the honest write-up is *"week-level information exists and consensus
already has it"* — which is a null, and the fifteenth measurement, and belongs in
[signal-screens.md](signal-screens.md) beside the others.

---

## What is settled, and by whom

Every decision above was fixed on 2026-08-27 in a grilling session, before any measurement.
Nothing in this file is downstream of a number. The ones most likely to be re-litigated later,
recorded with their reasons so they need not be:

| decision | why |
|---|---|
| Gate B primary, Gate A a diagnostic | historical weekly consensus ships ECR and no `r2p_pts`; a rank→points curve would put free parameters on the incumbent's side — [ADR-0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md) |
| search fixed at *start your highest projections* | ADR-0012 is accepted and withdrawn its own re-run clause; this varies the projection, not the search |
| `sd = k·√mu` untouched | the component-spread measurement is closed at P(better) 0.0% |
| multiplier form, `f ≡ 1` is the null | the model cannot be much worse than the projection it adjusts |
| weeks 1–14 headline, 8–14 secondary | the best feature is dark before week 8 and that is a limitation to measure, not to hide |
| 2% join floor | a missing player is ranked last, so join failure biases toward us |
| roster-week pairing, clustered by roster | the decision is weekly; the observations are not independent |
